"""Console entry point: `context1 "<query>" [flags]`."""

import argparse
import asyncio
import sys

from .agent import RunResult, run
from .budget import Budget


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="context1", description="Context-1 retrieval agent")
    p.add_argument("query")
    p.add_argument("--budget", type=int, default=32768)
    p.add_argument("--warn", type=float, default=0.6)
    p.add_argument("--hard", type=float, default=0.95)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    p.add_argument("--engine-cmd", default="context1-engine")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--runs-dir", default="runs")
    return p.parse_args(argv)


def _print_final(result: RunResult) -> None:
    print("\n=== Result ===")
    if result.ranked_chunk_ids:
        print("Ranked chunk IDs:")
        for i, cid in enumerate(result.ranked_chunk_ids, 1):
            entry = result.seen.get(cid)
            if entry:
                print(f"  {i}. {cid}  ({entry.get('title')})")
            else:
                print(f"  {i}. {cid}  [never seen during this run]")
    else:
        print("No chunk IDs returned.")

    print("\n=== Stats ===")
    print(f"turns: {result.turns}")
    print(f"final tokens used: {result.budget.used}/{result.budget.limit}")
    print(f"chunks seen: {len(result.seen)}")
    print(f"chunks pruned: {result.pruned_count}")
    print(f"completed: {result.completed}")


def main() -> None:
    args = _parse_args()
    budget = Budget(limit=args.budget, warn_frac=args.warn, hard_frac=args.hard)

    result = asyncio.run(
        run(
            args.query,
            budget=budget,
            model=args.model,
            engine_cmd=args.engine_cmd,
            data_dir=args.data_dir,
            max_turns=args.max_turns,
            runs_dir=args.runs_dir,
            on_event=print,
        )
    )

    _print_final(result)
    sys.exit(0 if result.completed else 1)


if __name__ == "__main__":
    main()
