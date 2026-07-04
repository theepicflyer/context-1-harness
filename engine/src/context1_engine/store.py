"""The corpus store: chroma (dense) + BM25 (sparse) hybrid retrieval."""

import os
import re

import chromadb
from rank_bm25 import BM25Okapi

from context1_engine import deepinfra

DENSE_TOP_K = 50
SPARSE_TOP_K = 50
RRF_K = 60
GREP_MAX_RESULTS = 5


def est_tokens(s: str) -> int:
    return len(s) // 4


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def format_chunk(chunk: dict, score: float | None = None) -> str:
    meta = chunk["metadata"]
    attrs = f'id="{chunk["id"]}" doc="{meta["doc_id"]}" title="{meta["title"]}"'
    if score is not None:
        attrs += f' score="{score:.3f}"'
    return f"<chunk {attrs}>\n{chunk['text']}\n</chunk>"


def rrf_fuse(dense_ids: list[str], sparse_ids: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion over two ranked id lists, returning fused order."""
    scores: dict[str, float] = {}
    for ids in (dense_ids, sparse_ids):
        for rank, chunk_id in enumerate(ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def _pack(chunks: list[dict], scores: dict | None, max_result_tokens: int) -> str:
    """Pack whole formatted chunks until the token budget is exhausted."""
    if not chunks:
        return "No results."
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        score = scores.get(chunk["id"]) if scores is not None else None
        block = format_chunk(chunk, score)
        tokens = est_tokens(block)
        if parts and total + tokens > max_result_tokens:
            break
        parts.append(block)
        total += tokens
    return "\n\n".join(parts)


class Store:
    def __init__(self, data_dir: str | None = None):
        data_dir = data_dir or os.environ.get("CONTEXT1_DATA_DIR", "./data")
        client = chromadb.PersistentClient(path=data_dir)
        self.collection = client.get_or_create_collection("corpus")

        result = self.collection.get(include=["documents", "metadatas"])
        chunks = [
            {"id": cid, "text": text, "metadata": meta}
            for cid, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]
        chunks.sort(key=lambda c: (c["metadata"]["doc_id"], c["metadata"]["index"]))
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None

    def search(
        self, query: str, exclude_chunk_ids: list[str] | None = None, max_result_tokens: int = 2000
    ) -> str:
        exclude = set(exclude_chunk_ids or [])
        if not self.chunks:
            return "No results."

        [query_embedding] = deepinfra.embed([query])
        dense_result = self.collection.query(
            query_embeddings=[query_embedding], n_results=min(DENSE_TOP_K, len(self.chunks))
        )
        dense_ids = dense_result["ids"][0]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        sparse_order = sorted(
            range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True
        )[:SPARSE_TOP_K]
        sparse_ids = [self.chunks[i]["id"] for i in sparse_order]

        fused = [cid for cid in rrf_fuse(dense_ids, sparse_ids) if cid not in exclude]
        if not fused:
            return "No results."
        candidates = [self.by_id[cid] for cid in fused[:50]]

        rerank_scores = deepinfra.rerank(query, [c["text"] for c in candidates])
        scores = {c["id"]: s for c, s in zip(candidates, rerank_scores)}
        ranked = sorted(candidates, key=lambda c: scores[c["id"]], reverse=True)

        return _pack(ranked, scores, max_result_tokens)

    def grep(self, pattern: str, exclude_chunk_ids: list[str] | None = None) -> str:
        exclude = set(exclude_chunk_ids or [])
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid pattern: {e}"

        matches = []
        for chunk in self.chunks:
            if chunk["id"] in exclude:
                continue
            if regex.search(chunk["text"]):
                matches.append(chunk)
                if len(matches) == GREP_MAX_RESULTS:
                    break

        if not matches:
            return "No results."
        return "\n\n".join(format_chunk(c) for c in matches)

    def read_document(self, doc_id: str, max_result_tokens: int = 4000) -> str:
        doc_chunks = [c for c in self.chunks if c["metadata"]["doc_id"] == doc_id]
        if not doc_chunks:
            return f"Unknown document: {doc_id}"

        parts: list[str] = []
        total = 0
        included = 0
        for chunk in doc_chunks:
            block = format_chunk(chunk)
            tokens = est_tokens(block)
            if parts and total + tokens > max_result_tokens:
                break
            parts.append(block)
            total += tokens
            included += 1

        text = "\n\n".join(parts)
        if included < len(doc_chunks):
            text += f"\n\n[truncated: showing {included} of {len(doc_chunks)} chunks]"
        return text
