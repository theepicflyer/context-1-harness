"""Chunk registry and context pruning over the anthropic-format message list."""

import re

CHUNK_RE = re.compile(r'<chunk\s+([^>]*)>(.*?)</chunk>', re.DOTALL)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def est_tokens(s: str) -> int:
    return len(s) // 4


def _attrs(attr_str: str) -> dict[str, str]:
    return dict(ATTR_RE.findall(attr_str))


def register_chunks(text: str) -> dict[str, dict]:
    """Extract id/doc/title (and score, if present) for every chunk block in a tool-result string."""
    out: dict[str, dict] = {}
    for m in CHUNK_RE.finditer(text):
        attrs = _attrs(m.group(1))
        cid = attrs.get("id")
        if not cid:
            continue
        entry = {"doc": attrs.get("doc"), "title": attrs.get("title")}
        if "score" in attrs:
            entry["score"] = attrs["score"]
        out[cid] = entry
    return out


def prune(messages: list[dict], chunk_ids: list[str]) -> tuple[int, int, list[str]]:
    """Replace requested chunk blocks in prior tool_result strings with pruned markers.

    Returns (n_pruned, freed_tokens_estimate, unknown_ids). Ids that were never seen
    or are already pruned are reported in unknown_ids.
    """
    to_prune = set(chunk_ids)
    pruned_now: set[str] = set()
    freed = 0

    def repl(m: re.Match) -> str:
        nonlocal freed
        cid = _attrs(m.group(1)).get("id")
        if cid in to_prune:
            pruned_now.add(cid)
            freed += est_tokens(m.group(0))
            return f"[pruned: {cid}]"
        return m.group(0)

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                text = block.get("content")
                if isinstance(text, str):
                    block["content"] = CHUNK_RE.sub(repl, text)

    unknown_ids = sorted(to_prune - pruned_now)
    return len(pruned_now), freed, unknown_ids
