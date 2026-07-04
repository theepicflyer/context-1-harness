"""Offline unit tests for context1_engine (no network calls)."""

import shutil
import tempfile

import pytest

from context1_engine import deepinfra, ingest, store
from context1_engine.store import Store, format_chunk, rrf_fuse


# --- rrf_fuse -----------------------------------------------------------


def test_rrf_fuse_orders_by_combined_rank():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = rrf_fuse(dense, sparse)
    # "a" and "b" both appear near the top of both lists, so they should
    # outrank "c" and "d" which only appear in one list each.
    assert fused[0] in ("a", "b")
    assert fused[1] in ("a", "b")
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_fuse_only_in_one_list():
    fused = rrf_fuse(["x", "y"], [])
    assert fused == ["x", "y"]


# --- format_chunk ---------------------------------------------------------


def test_format_chunk_matches_contract_with_score():
    chunk = {"id": "pg-ds#004", "text": "hello world", "metadata": {"doc_id": "pg-ds", "title": "Do Things that Don't Scale", "index": 4}}
    out = format_chunk(chunk, 0.8314)
    assert out == (
        '<chunk id="pg-ds#004" doc="pg-ds" title="Do Things that Don\'t Scale" score="0.831">\n'
        "hello world\n"
        "</chunk>"
    )


def test_format_chunk_no_score_omits_attribute():
    chunk = {"id": "pg-ds#000", "text": "hi", "metadata": {"doc_id": "pg-ds", "title": "T", "index": 0}}
    out = format_chunk(chunk)
    assert "score=" not in out
    assert out.startswith('<chunk id="pg-ds#000" doc="pg-ds" title="T">')


# --- Store fixtures ---------------------------------------------------------


@pytest.fixture
def tmp_store(monkeypatch):
    data_dir = tempfile.mkdtemp()
    try:
        chunks = [
            ("pg-a#000", "Cats are great pets and love to nap in sunlight.", "pg-a", "About Cats", 0),
            ("pg-a#001", "Dogs enjoy fetch and long walks outside.", "pg-a", "About Cats", 1),
            ("pg-b#000", "The startup grew quickly by doing things that scale poorly at first.", "pg-b", "About Startups", 0),
            ("pg-b#001", "Founders should talk to users directly and often.", "pg-b", "About Startups", 1),
            ("pg-b#002", "Determination matters more than initial talent for founders.", "pg-b", "About Startups", 2),
        ]

        import chromadb

        client = chromadb.PersistentClient(path=data_dir)
        collection = client.get_or_create_collection("corpus")
        fake_embeddings = [[float(i)] * 4 for i in range(len(chunks))]
        collection.upsert(
            ids=[c[0] for c in chunks],
            documents=[c[1] for c in chunks],
            metadatas=[{"doc_id": c[2], "title": c[3], "index": c[4]} for c in chunks],
            embeddings=fake_embeddings,
        )

        def fake_embed(texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def fake_rerank(query, documents):
            # score longer documents higher, deterministically
            return [float(len(d)) for d in documents]

        monkeypatch.setattr(deepinfra, "embed", fake_embed)
        monkeypatch.setattr(deepinfra, "rerank", fake_rerank)

        s = Store(data_dir=data_dir)
        yield s
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


# --- Store.search ---------------------------------------------------------


def test_search_returns_at_least_one_chunk_even_under_tiny_budget(tmp_store):
    result = tmp_store.search("founders", max_result_tokens=1)
    assert result != "No results."
    assert result.count("<chunk") == 1


def test_search_excludes_chunk_ids(tmp_store):
    all_ids = [c["id"] for c in tmp_store.chunks]
    result = tmp_store.search("founders", exclude_chunk_ids=all_ids, max_result_tokens=2000)
    assert result == "No results."


def test_search_no_results_on_empty_corpus(monkeypatch):
    data_dir = tempfile.mkdtemp()
    try:
        monkeypatch.setattr(deepinfra, "embed", lambda texts: [[0.0] for _ in texts])
        s = Store(data_dir=data_dir)
        assert s.search("anything") == "No results."
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


# --- Store.grep -------------------------------------------------------------


def test_grep_invalid_regex(tmp_store):
    result = tmp_store.grep("(unclosed", [])
    assert result.startswith("Invalid pattern:")


def test_grep_caps_at_five_results(tmp_store):
    result = tmp_store.grep("e", [])  # matches most chunks
    assert result.count("<chunk") <= 5
    assert "score=" not in result


def test_grep_no_results(tmp_store):
    assert tmp_store.grep("zzzznomatch", []) == "No results."


def test_grep_respects_exclude(tmp_store):
    result = tmp_store.grep("Founders", [])
    assert "pg-b#001" in result
    result2 = tmp_store.grep("Founders", ["pg-b#001"])
    assert "pg-b#001" not in result2


# --- Store.read_document ----------------------------------------------------


def test_read_document_unknown():
    pass  # covered via tmp_store fixture below


def test_read_document_full(tmp_store):
    result = tmp_store.read_document("pg-b", max_result_tokens=4000)
    assert result.count("<chunk") == 3
    assert "truncated" not in result


def test_read_document_truncated(tmp_store):
    result = tmp_store.read_document("pg-b", max_result_tokens=1)
    assert result.count("<chunk") == 1
    assert "[truncated: showing 1 of 3 chunks]" in result


def test_read_document_unknown_doc(tmp_store):
    assert tmp_store.read_document("pg-nonexistent") == "Unknown document: pg-nonexistent"


# --- ingest.chunk_text -------------------------------------------------------


def test_chunk_text_packs_paragraphs():
    paragraphs = ["Para one. " * 20, "Para two. " * 20, "Para three. " * 20]
    text = "\n\n".join(paragraphs)
    chunks = ingest.chunk_text(text, size=250)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)  # generous bound, sentence splits can slightly exceed size
    assert "Para one." in chunks[0]


def test_chunk_text_splits_oversized_paragraph_on_sentences():
    sentence = "This is a sentence that repeats. "
    huge_paragraph = sentence * 50
    chunks = ingest.chunk_text(huge_paragraph, size=200)
    assert len(chunks) > 1
    # no sentence should have been cut in half
    for c in chunks:
        assert c.strip().endswith(".")


def test_chunk_text_empty():
    assert ingest.chunk_text("") == []
