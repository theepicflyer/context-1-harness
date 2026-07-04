"""The observe -> infer -> act retrieval-agent loop."""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime

from anthropic import Anthropic, APIError
from dotenv import load_dotenv

from . import context, mcp_client
from .budget import Budget
from .prompts import SYSTEM_PROMPT

load_dotenv()

PRUNE_TOOL = {
    "name": "prune_chunks",
    "description": (
        "Remove the specified chunks from your context to free token budget. "
        "Their contents will be replaced with a pruned marker."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chunk_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["chunk_ids"],
    },
}

COMPLETE_TOOL = {
    "name": "complete",
    "description": (
        "Finish the search. Provide the final ranked list of chunk IDs relevant to the "
        "query, most relevant first."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ranked_chunk_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["ranked_chunk_ids"],
    },
}

# tool name -> its own default max_result_tokens, used as the harness-imposed cap
_MAX_RESULT_TOKENS_DEFAULTS = {"search_corpus": 2000, "read_document": 4000}

_REMINDER = (
    "Reminder: do not answer the question. Continue searching or call complete with "
    "your final ranked chunk IDs."
)


def allowed_at_hard_stop(name: str) -> bool:
    """Whether a tool call is still permitted once the budget has hit the hard limit."""
    return name in {"prune_chunks", "complete"}


@dataclass
class RunResult:
    ranked_chunk_ids: list[str]
    completed: bool
    turns: int
    budget: Budget
    seen: dict[str, dict]
    pruned_count: int


def _log(path: str, event: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def _create_sync(client: Anthropic, model: str, tools: list[dict], messages: list[dict]):
    return client.messages.create(
        model=model, max_tokens=2048, system=SYSTEM_PROMPT, tools=tools, messages=messages
    )


async def _create(client: Anthropic, model: str, tools: list[dict], messages: list[dict]):
    try:
        return await asyncio.to_thread(_create_sync, client, model, tools, messages)
    except APIError:
        return await asyncio.to_thread(_create_sync, client, model, tools, messages)


def _capped_max_result_tokens(name: str, budget: Budget) -> int:
    remaining = max(256, int(budget.limit * budget.hard_frac) - budget.used - 1024)
    return min(remaining, _MAX_RESULT_TOKENS_DEFAULTS[name])


async def run(
    query: str,
    *,
    budget: Budget,
    model: str,
    engine_cmd: str,
    data_dir: str,
    max_turns: int = 30,
    runs_dir: str = "runs",
    on_event=print,
) -> RunResult:
    os.makedirs(runs_dir, exist_ok=True)
    log_path = os.path.join(runs_dir, datetime.now().strftime("%Y%m%d-%H%M%S") + ".jsonl")

    client = Anthropic(
        base_url="https://api.deepinfra.com/anthropic",
        api_key=os.environ.get("DEEPINFRA_API_KEY"),
    )

    seen: dict[str, dict] = {}
    pruned_count = 0
    ranked_chunk_ids: list[str] = []
    completed = False
    text_only_streak = 0
    turn = 0
    messages: list[dict] = [{"role": "user", "content": query}]

    async with mcp_client.engine_session(engine_cmd, data_dir) as session:
        tools = await mcp_client.anthropic_tools(session) + [PRUNE_TOOL, COMPLETE_TOOL]

        while turn < max_turns:
            turn += 1
            response = await _create(client, model, tools, messages)
            budget.update(response.usage.input_tokens, response.usage.output_tokens)
            _log(
                log_path,
                {
                    "event": "response",
                    "turn": turn,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "blocks": [b.model_dump() for b in response.content],
                },
            )

            for block in response.content:
                if block.type == "text" and block.text:
                    on_event(f"[turn {turn}] {block.text}")

            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                text_only_streak += 1
                on_event(f"[turn {turn}] (no tool call)")
                if text_only_streak >= 2:
                    break
                messages.append({"role": "user", "content": _REMINDER})
                continue
            text_only_streak = 0

            result_blocks = []
            done = False
            for tu in tool_uses:
                name = tu.name
                args = dict(tu.input)
                on_event(f"[turn {turn}] tool_use {name}({args})")

                if budget.level == "hard" and not allowed_at_hard_stop(name):
                    result_text = (
                        f"Context budget exceeded ({round(budget.pct * 100)}%): "
                        "only prune_chunks and complete are allowed."
                    )
                    result_blocks.append(
                        {"type": "tool_result", "tool_use_id": tu.id, "content": result_text, "is_error": True}
                    )
                    _log(
                        log_path,
                        {"event": "tool_call", "turn": turn, "name": name, "args": args,
                         "result": result_text, "is_error": True},
                    )
                    continue

                if name == "prune_chunks":
                    n, freed, unknown = context.prune(messages, args.get("chunk_ids", []))
                    pruned_count += n
                    result_text = f"Pruned {n} chunks, freed ~{freed} tokens."
                    if unknown:
                        result_text += f" Unknown ids: {', '.join(unknown)}."
                    result_blocks.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_text})
                    _log(
                        log_path,
                        {"event": "prune", "turn": turn, "chunk_ids": args.get("chunk_ids", []),
                         "n_pruned": n, "freed": freed, "unknown": unknown},
                    )
                elif name == "complete":
                    ranked_chunk_ids = list(args.get("ranked_chunk_ids", []))
                    completed = True
                    done = True
                    result_text = "Search complete."
                    result_blocks.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_text})
                    _log(log_path, {"event": "complete", "turn": turn, "ranked_chunk_ids": ranked_chunk_ids})
                else:
                    if name in ("search_corpus", "grep_corpus"):
                        args["exclude_chunk_ids"] = sorted(seen.keys())
                    if name in _MAX_RESULT_TOKENS_DEFAULTS:
                        args["max_result_tokens"] = _capped_max_result_tokens(name, budget)
                    result_text = await mcp_client.call(session, name, args)
                    seen.update(context.register_chunks(result_text))
                    result_blocks.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_text})
                    _log(
                        log_path,
                        {"event": "tool_call", "turn": turn, "name": name, "args": args, "result": result_text},
                    )

            result_blocks[-1]["content"] = result_blocks[-1]["content"] + "\n" + budget.status_line()
            messages.append({"role": "user", "content": result_blocks})

            if done:
                break

    return RunResult(
        ranked_chunk_ids=ranked_chunk_ids,
        completed=completed,
        turns=turn,
        budget=budget,
        seen=seen,
        pruned_count=pruned_count,
    )
