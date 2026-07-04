# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A miniature, readable rebuild of the agent harness behind Chroma's Context-1 (https://www.trychroma.com/research/context-1), plus a swappable MCP embeddings engine to run it against. Chroma published the model and technical report but not the harness; this repo reconstructs the harness from the report. **Readability is the point**: minimum code, no agent frameworks, no speculative abstraction, plain functions over classes, files short enough to read top-to-bottom. Push back on changes that add layers.

## Commands

Requires `uv` and `DEEPINFRA_API_KEY` in repo-root `.env`.

```bash
uv sync                          # install workspace (both packages share one venv)
uv run pytest                    # all tests — offline, no network, fast
uv run pytest tests/test_harness.py -q          # one file
uv run pytest tests/test_engine.py -k rrf       # one test by keyword

uv run context1-ingest           # build demo corpus (downloads PG essays, embeds via DeepInfra)
uv run context1-ingest --limit 3 --data-dir /tmp/d   # quick/cheap variant
uv run context1 "some query"     # live agent run (costs API credits)
uv run context1 "q" --budget 6000 --max-turns 12      # exercise warn/hard-stop tiers
```

Tests must stay offline: mock `deepinfra.embed`/`rerank`; never call DeepInfra or spawn the MCP server in tests.

## Architecture

Two uv-workspace packages that communicate **only** over MCP stdio at runtime — they deliberately share no code (each has its own copy of `est_tokens` and the chunk regex/formatting; keep it that way).

**engine/** (`context1_engine`) — corpus side:
- `server.py` exposes 3 MCP tools: `search_corpus`, `grep_corpus`, `read_document`. The `Store` is built in `main()`, not at import (importing must stay side-effect-free).
- `store.py`: Chroma persistent collection `corpus` (dense) + in-memory BM25 over all chunks, fused via `rrf_fuse`, then DeepInfra-reranked; results packed whole-chunk into a token budget.
- `ingest.py`: downloads a hardcoded, slug-verified list of Paul Graham essays, chunks ~1000 chars, embeds, upserts into Chroma at `--data-dir` (default `./data`).

**harness/** (`context1_harness`) — agent side:
- `agent.py`: the observe→infer→act loop. anthropic SDK pointed at DeepInfra's Anthropic-compatible endpoint (`https://api.deepinfra.com/anthropic`), default model `deepseek-ai/DeepSeek-V4-Flash`. Adds two harness-local tools (`prune_chunks`, `complete`) to the MCP tools.
- `context.py`: pruning = literal regex rewrite of `<chunk …>…</chunk>` blocks in prior tool_result strings to `[pruned: id]`. This is what makes pruning genuinely shrink the next API call.
- `budget.py`: three tiers driven by API-reported usage (used = last call's input+output tokens, not cumulative): status line every turn, warn ≥60%, hard ≥95% (at hard, `agent.py` rejects all tools except prune/complete without calling the engine).
- `mcp_client.py`: spawns `--engine-cmd` (default `context1-engine`) over stdio; strips `exclude_chunk_ids` from tool schemas — the model never sees it, the harness injects all seen chunk IDs itself (dedup).
- Full unpruned trajectories go to `runs/*.jsonl`.

## Load-bearing invariants

- **Chunk wire format** is the contract between the packages and the prune mechanism: `<chunk id="pg-ds#004" doc="pg-ds" title="…" score="0.831">text</chunk>`, blocks separated by blank lines, `score` omitted for grep/read results, attribute order fixed. Chunk IDs are `{doc_id}#{index:03d}`, doc_id `pg-{slug}`. If you change this format, change it in engine `store.format_chunk`, harness `context.CHUNK_RE`, and both test files together.
- Token estimation everywhere is `len(s) // 4` — do not introduce a tokenizer.
- The engine is swappable: any MCP stdio server exposing the same three tool signatures works via `--engine-cmd`. Don't leak harness concepts into the engine or vice versa.
- README's "Differences from the real Context-1 harness" section documents intentional deviations from the report — update it if behavior changes.
