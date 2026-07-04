"""Thin sync client for the DeepInfra embeddings and reranker endpoints."""

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

EMBED_URL = "https://api.deepinfra.com/v1/openai/embeddings"
RERANK_URL = "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-0.6B"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 64
TIMEOUT = 60


def _api_key() -> str:
    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY is not set. Add it to the repo-root .env or the process env."
        )
    return key


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batching requests in groups of 64."""
    headers = {"Authorization": f"Bearer {_api_key()}"}
    embeddings: list[list[float]] = []
    with httpx.Client(timeout=TIMEOUT) as client:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = client.post(
                EMBED_URL,
                headers=headers,
                json={"model": EMBED_MODEL, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            embeddings.extend(item["embedding"] for item in data)
    return embeddings


def rerank(query: str, documents: list[str]) -> list[float]:
    """Score each document's relevance to the query."""
    headers = {"Authorization": f"bearer {_api_key()}"}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            RERANK_URL,
            headers=headers,
            json={"queries": [query] * len(documents), "documents": documents},
        )
        resp.raise_for_status()
        return resp.json()["scores"]
