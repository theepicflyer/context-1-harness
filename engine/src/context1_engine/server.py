"""Context-1 embeddings engine: an MCP stdio server over the corpus store."""

from mcp.server.fastmcp import FastMCP

from context1_engine.store import Store

mcp = FastMCP("context1-engine")
store: Store  # built in main(), so importing this module has no side effects


@mcp.tool()
def search_corpus(
    query: str, exclude_chunk_ids: list[str] = [], max_result_tokens: int = 2000
) -> str:
    """Hybrid BM25 + dense vector search with reciprocal rank fusion over the corpus;
    candidates are reranked and the top results returned within a token budget."""
    return store.search(query, exclude_chunk_ids, max_result_tokens)


@mcp.tool()
def grep_corpus(pattern: str, exclude_chunk_ids: list[str] = []) -> str:
    """Case-insensitive regex search over raw chunk text, for finding exact
    strings or patterns that semantic search might miss."""
    return store.grep(pattern, exclude_chunk_ids)


@mcp.tool()
def read_document(doc_id: str, max_result_tokens: int = 4000) -> str:
    """Read a document's chunks in order, for when a search result warrants
    reading the surrounding context in full."""
    return store.read_document(doc_id, max_result_tokens)


def main():
    global store
    store = Store()
    mcp.run()


if __name__ == "__main__":
    main()
