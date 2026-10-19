import re
from dataclasses import dataclass
from functools import cache

from rank_bm25 import BM25Okapi

from assistant.config import DATA_DIR

PAGE_RE = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    brochure: str
    page: int
    text: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted((DATA_DIR / "brochures").glob("*.md")):
        parts = PAGE_RE.split(path.read_text(encoding="utf-8"))
        for page, text in zip(parts[1::2], parts[2::2]):
            if text.strip():
                chunks.append(Chunk(path.stem, int(page), text.strip()))
    return chunks


@cache
def index() -> tuple[list[Chunk], BM25Okapi]:
    chunks = load_chunks()
    return chunks, BM25Okapi([tokenize(f"{c.brochure} {c.text}") for c in chunks])


def search_brochures(query: str, k: int = 5) -> list[dict]:
    """Keyword (BM25) search over product brochure pages. The only source of product facts.

    Brochure wording may differ from the customer's, so call it 2-3 times with rephrased queries
    (e.g. 'feed grade changes' -> 'varying process conditions', 'automatic optimization').

    Args:
        query: Search keywords.
        k: Number of pages to return.
    """
    chunks, bm25 = index()
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"brochure": chunks[i].brochure, "page": chunks[i].page, "score": round(float(scores[i]), 2), "text": chunks[i].text}
        for i in top
        if scores[i] > 0
    ]


def list_products() -> list[dict]:
    """List all products covered by the brochures (one line each). Use to see the catalog before searching or recommending."""
    chunks, _ = index()
    return [
        {"brochure": c.brochure, "title": " ".join(c.text.split())[:120]}
        for c in chunks
        if c.page == 1
    ]
