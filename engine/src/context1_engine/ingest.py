"""CLI: fetch a curated set of Paul Graham essays, chunk them, embed, and upsert into chroma."""

import argparse
import re
from html.parser import HTMLParser

import chromadb
import httpx

from context1_engine import deepinfra

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

CHUNK_SIZE = 1000

# (slug, title) pairs -- slugs verified against paulgraham.com/articles.html
ESSAYS = [
    ("ds", "Do Things that Don't Scale"),
    ("greatwork", "How to Do Great Work"),
    ("startupideas", "How to Get Startup Ideas"),
    ("wealth", "How to Make Wealth"),
    ("avg", "Beating the Averages"),
    ("essay", "The Age of the Essay"),
    ("love", "How to Do What You Love"),
    ("growth", "Startup = Growth"),
    ("founders", "What We Look for in Founders"),
    ("mean", "Mean People Fail"),
    ("determination", "The Anatomy of Determination"),
    ("relres", "Relentlessly Resourceful"),
    ("nerds", "Why Nerds are Unpopular"),
    ("gh", "Great Hackers"),
    ("procrastination", "Good and Bad Procrastination"),
    ("cities", "Cities and Ambition"),
    ("makersschedule", "Maker's Schedule, Manager's Schedule"),
    ("ramenprofitable", "Ramen Profitable"),
    ("fundraising", "A Fundraising Survival Guide"),
    ("13sentences", "Startups in 13 Sentences"),
    ("hp", "Hackers and Painters"),
    ("say", "What You Can't Say"),
    ("taste", "Taste for Makers"),
    ("start", "How to Start a Startup"),
    ("submarine", "The Submarine"),
]


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor: skips script/style, collapses whitespace."""

    def __init__(self):
        super().__init__()
        self._skip = False
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "br", "div", "table", "tr"):
            self.chunks.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.chunks.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.chunks)
    # collapse whitespace within lines, but keep paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    paragraphs = []
    current: list[str] = []
    for line in lines:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(p for p in paragraphs if len(p) > 1)


def _split_sentences(paragraph: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", paragraph)


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Greedily pack paragraphs into ~size-character chunks."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # split oversized paragraphs on sentence boundaries first
        pieces = [para] if len(para) <= size else _split_sentences(para)
        for piece in pieces:
            if not piece.strip():
                continue
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) > size and current:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def fetch_essay(slug: str) -> str | None:
    url = f"https://paulgraham.com/{slug}.html"
    try:
        resp = httpx.get(
            url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        print(f"  WARNING: failed to fetch {slug}: {e}")
        return None
    return html_to_text(resp.text)


def main():
    parser = argparse.ArgumentParser(description="Ingest Paul Graham essays into the corpus")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    essays = ESSAYS[: args.limit] if args.limit else ESSAYS

    client = chromadb.PersistentClient(path=args.data_dir)
    collection = client.get_or_create_collection("corpus")

    total_chunks = 0
    total_docs = 0
    for slug, title in essays:
        print(f"Fetching {title!r} ({slug})...")
        text = fetch_essay(slug)
        if text is None:
            continue

        doc_id = f"pg-{slug}"
        pieces = chunk_text(text)
        if not pieces:
            print(f"  WARNING: no text extracted for {slug}, skipping")
            continue

        ids = [f"{doc_id}#{i:03d}" for i in range(len(pieces))]
        metadatas = [{"doc_id": doc_id, "title": title, "index": i} for i in range(len(pieces))]
        embeddings = deepinfra.embed(pieces)

        collection.upsert(ids=ids, documents=pieces, metadatas=metadatas, embeddings=embeddings)
        print(f"  {len(pieces)} chunks")
        total_chunks += len(pieces)
        total_docs += 1

    print(f"\nDone: {total_docs} docs, {total_chunks} chunks ingested into {args.data_dir}")


if __name__ == "__main__":
    main()
