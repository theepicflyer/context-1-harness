# context-1-harness

A miniature, readable rebuild of the agent harness behind [Chroma's Context-1](https://www.trychroma.com/research/context-1) — the 20B agentic search model that iteratively searches a corpus and **edits its own context** to stay within a bounded token budget. Chroma published the model and the technical report, but not the harness. This repo rebuilds the harness from the report's description, plus a simple embeddings engine (exposed over MCP, so it's swappable) to run it against.

It is deliberately small: a plain Python agent loop with no agent frameworks, meant to be read top-to-bottom.

## How it works

```
┌─────────────────────────────┐        MCP (stdio)       ┌──────────────────────────┐
│  harness (context1)         │ ───────────────────────► │  engine (context1-engine)│
│                             │  search_corpus           │                          │
│  observe → infer → act loop │  grep_corpus             │  Chroma (dense, cosine)  │
│  token budget tiers         │  read_document           │  BM25 (rank-bm25)        │
│  chunk dedup (exclusion)    │                          │  RRF fusion → reranker   │
│  prune_chunks / complete    │        Anthropic-        │                          │
│         │                   │        compatible API    │  DeepInfra:              │
│         ▼                   │ ───────────────────────► │   Qwen3-Embedding-0.6B   │
│  runs/*.jsonl (full         │  DeepSeek-V4-Flash       │   Qwen3-Reranker-0.6B    │
│  unpruned trajectory)       │  via DeepInfra           │                          │
└─────────────────────────────┘                          └──────────────────────────┘
```

The agent is a retrieval agent, not a question answerer: given a query, it decomposes it, runs parallel searches, evaluates chunks, prunes the irrelevant ones out of its own context, and finishes by calling `complete` with a ranked list of relevant chunk IDs.

### Tools

Following the technical report — three corpus tools served over MCP by the engine, two context tools implemented inside the harness:

| Tool | Where | What it does |
|---|---|---|
| `search_corpus(query)` | engine | Hybrid dense + BM25 search fused with reciprocal rank fusion, reranked, top results returned within a per-call token budget |
| `grep_corpus(pattern)` | engine | Regex search over chunks, up to 5 matches |
| `read_document(doc_id)` | engine | A document's chunks in order, truncated to the remaining token budget |
| `prune_chunks(chunk_ids)` | harness | **Actually removes** those chunk blocks from prior tool results in the message history, replacing each with `[pruned: <id>]` |
| `complete(ranked_chunk_ids)` | harness | Ends the run with the agent's final ranked list of relevant chunk IDs |

### Context management (the interesting part)

The harness reproduces the report's three-tier budget scheme (defaults: 32,768 tokens, warn 60%, hard stop 95% — all configurable):

- **Continuous visibility** — every turn, a status line like `[context: 18,234/32,768 tokens — 56%]` is appended to the last tool result.
- **Soft threshold** — past the warn level, the status line tells the agent to prune or conclude.
- **Hard stop** — past the hard level, every tool call except `prune_chunks` and `complete` is rejected with an error.

Pruning genuinely shrinks the next API call: the chunk text is gone from the messages sent to the model. The full unpruned trajectory is preserved in `runs/*.jsonl` (the report keeps it for reward computation; here it's for inspection).

**Deduplication:** the harness tracks every chunk ID it has seen and silently passes them as `exclude_chunk_ids` on every search — the model never sees or manages that parameter, so repeated searches always surface new material.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and a [DeepInfra](https://deepinfra.com) API key.

```bash
git clone https://github.com/theepicflyer/context-1-harness && cd context-1-harness
echo 'DEEPINFRA_API_KEY=...' > .env
uv sync

# Build the demo corpus (~25 Paul Graham essays, downloaded and embedded)
uv run context1-ingest

# Run the agent
uv run context1 "What does Paul Graham say about why startups should do things that don't scale?"
```

Sample output:

```
turn 1  search_corpus("startups do things that don't scale") ─ search_corpus("manual recruitment early users")
        [context: 7,912/32,768 tokens — 24%]
turn 2  prune_chunks([pg-hs#012, pg-growth#003]) ─ read_document("pg-ds")
        [context: 11,204/32,768 tokens — 34%]
turn 3  complete([pg-ds#000, pg-ds#004, pg-ds#001, ...])

Ranked chunks:
  1. pg-ds#000    Do Things that Don't Scale
  2. pg-ds#004    Do Things that Don't Scale
  ...
completed in 3 turns · 11,204 tokens peak · 14 chunks seen · 2 pruned
```

## Configuration

```
uv run context1 "<query>" \
  --budget 32768          # total token budget
  --warn 0.6              # soft-warning fraction
  --hard 0.95             # hard-stop fraction
  --max-turns 30
  --model deepseek-ai/DeepSeek-V4-Flash
  --engine-cmd context1-engine   # any MCP stdio server with the same tools
  --data-dir ./data
```

Because the engine is just an MCP stdio server, you can point `--engine-cmd` at any other implementation of `search_corpus` / `grep_corpus` / `read_document`.

## Layout

```
engine/   context1-engine — MCP server: Chroma + BM25 + RRF + DeepInfra rerank; context1-ingest builds the corpus
harness/  context1 — the agent loop: budget tiers, dedup, pruning, complete
tests/    offline unit tests (uv run pytest)
```

## Models used

- Agent: `deepseek-ai/DeepSeek-V4-Flash` via DeepInfra's Anthropic-compatible Messages API
- Embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Reranker: `Qwen/Qwen3-Reranker-0.6B`

## Differences from the real Context-1 harness

- The report's harness drives Chroma's own fine-tuned 20B model; this one drives a general instruction-tuned model with a system prompt modeled on the report's, so trajectories are less disciplined.
- Token counts come from the API's reported usage (exact), and result-budget packing uses a rough 4-chars-per-token estimate.
- `read_document` returns chunks in document order (the report reranks them against the query).
- `complete` is an addition: the report's model ends by emitting text; an explicit tool makes the ranked list machine-readable.

## License

MIT
