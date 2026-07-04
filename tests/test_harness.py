"""Offline unit tests for context1_harness (no network, no MCP subprocess)."""

import asyncio

from context1_harness.agent import allowed_at_hard_stop
from context1_harness.budget import Budget
from context1_harness.context import prune, register_chunks
from context1_harness.mcp_client import anthropic_tools


# --- budget -----------------------------------------------------------------


def test_budget_status_line_ok_with_commas():
    b = Budget(limit=32768)
    b.update(10000, 8234)
    assert b.used == 18234
    assert b.level == "ok"
    assert b.status_line() == "[context: 18,234/32,768 tokens — 56%]"


def test_budget_status_line_warn():
    b = Budget(limit=100, used=60)
    assert b.level == "warn"
    assert b.status_line() == (
        "[context: 60/100 tokens — 60%] WARNING: context is above 60% — "
        "prune irrelevant chunks with prune_chunks or finish with complete."
    )


def test_budget_status_line_hard():
    b = Budget(limit=100, used=95)
    assert b.level == "hard"
    assert b.status_line() == (
        "[context: 95/100 tokens — 95%] WARNING: context is above 95% — "
        "only prune_chunks and complete are allowed."
    )


def test_budget_level_boundaries():
    assert Budget(limit=100, used=59).level == "ok"
    assert Budget(limit=100, used=60).level == "warn"
    assert Budget(limit=100, used=94).level == "warn"
    assert Budget(limit=100, used=95).level == "hard"


def test_budget_update_is_not_cumulative():
    b = Budget(limit=1000)
    b.update(100, 50)
    assert b.used == 150
    b.update(10, 5)
    assert b.used == 15


# --- context.register_chunks -------------------------------------------------


def test_register_chunks_with_score():
    text = (
        '<chunk id="pg-ds#004" doc="pg-ds" title="Do Things that Don\'t Scale" score="0.831">\n'
        "some chunk text\n"
        "</chunk>"
    )
    reg = register_chunks(text)
    assert set(reg) == {"pg-ds#004"}
    assert reg["pg-ds#004"] == {"doc": "pg-ds", "title": "Do Things that Don't Scale", "score": "0.831"}


def test_register_chunks_without_score():
    text = (
        '<chunk id="pg-ds#005" doc="pg-ds" title="Another Essay">\n'
        "more text\n"
        "</chunk>\n\n"
        '<chunk id="pg-ds#006" doc="pg-ds" title="Third">\n'
        "third text\n"
        "</chunk>"
    )
    reg = register_chunks(text)
    assert set(reg) == {"pg-ds#005", "pg-ds#006"}
    assert "score" not in reg["pg-ds#005"]
    assert reg["pg-ds#006"]["title"] == "Third"


def test_register_chunks_empty_result():
    assert register_chunks("No results.") == {}


# --- context.prune ------------------------------------------------------------


def _sample_messages():
    tool_result_text = (
        '<chunk id="pg-ds#004" doc="pg-ds" title="Do Things that Don\'t Scale" score="0.831">\n'
        "some fairly long chunk text that should have a nonzero token estimate\n"
        "</chunk>\n\n"
        '<chunk id="pg-ds#005" doc="pg-ds" title="Another Essay" score="0.700">\n'
        "another chunk of text here\n"
        "</chunk>"
    )
    return [
        {"role": "user", "content": "find stuff"},
        {"role": "assistant", "content": [{"type": "text", "text": "searching"}]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": tool_result_text},
            ],
        },
    ]


def test_prune_removes_block_and_reports_unknown():
    messages = _sample_messages()
    n_pruned, freed, unknown = prune(messages, ["pg-ds#004", "pg-ds#999"])

    assert n_pruned == 1
    assert freed > 0
    assert unknown == ["pg-ds#999"]

    tool_result = messages[2]["content"][0]["content"]
    assert "[pruned: pg-ds#004]" in tool_result
    assert 'id="pg-ds#004"' not in tool_result
    # the untouched chunk is still present
    assert 'id="pg-ds#005"' in tool_result


def test_prune_twice_reports_already_pruned_as_unknown():
    messages = _sample_messages()
    prune(messages, ["pg-ds#004"])

    n_pruned, freed, unknown = prune(messages, ["pg-ds#004"])
    assert n_pruned == 0
    assert freed == 0
    assert unknown == ["pg-ds#004"]


def test_prune_never_seen_id_is_unknown():
    messages = _sample_messages()
    n_pruned, freed, unknown = prune(messages, ["nope#001"])
    assert n_pruned == 0
    assert freed == 0
    assert unknown == ["nope#001"]


# --- mcp_client.anthropic_tools schema scrubbing ------------------------------


class _FakeTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class _FakeListToolsResult:
    def __init__(self, tools):
        self.tools = tools


class _FakeSession:
    def __init__(self, tools):
        self._tools = tools

    async def list_tools(self):
        return _FakeListToolsResult(self._tools)


def test_anthropic_tools_strips_exclude_chunk_ids():
    fake_tool = _FakeTool(
        name="search_corpus",
        description="Hybrid retrieval over the corpus.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "exclude_chunk_ids": {"type": "array", "items": {"type": "string"}},
                "max_result_tokens": {"type": "integer"},
            },
            "required": ["query", "exclude_chunk_ids"],
        },
    )
    session = _FakeSession([fake_tool])

    tools = asyncio.run(anthropic_tools(session))

    assert len(tools) == 1
    tool = tools[0]
    assert tool["name"] == "search_corpus"
    assert "exclude_chunk_ids" not in tool["input_schema"]["properties"]
    assert "exclude_chunk_ids" not in tool["input_schema"]["required"]
    assert "query" in tool["input_schema"]["properties"]


# --- agent.allowed_at_hard_stop ------------------------------------------------


def test_allowed_at_hard_stop():
    assert allowed_at_hard_stop("prune_chunks") is True
    assert allowed_at_hard_stop("complete") is True
    assert allowed_at_hard_stop("search_corpus") is False
    assert allowed_at_hard_stop("grep_corpus") is False
    assert allowed_at_hard_stop("read_document") is False
