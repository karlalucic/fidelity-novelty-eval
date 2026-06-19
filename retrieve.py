"""BM25 retrieval over archive chunks. Pure python, no API."""

from __future__ import annotations

from rank_bm25 import BM25Okapi


def build_index(chunks: list[dict], cfg: dict) -> object:
    """Build a BM25 index over a list of {id, text} chunks. cfg is unused but kept for interface consistency."""
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def retrieve(index, chunks: list[dict], query: str, top_k: int) -> list[dict]:
    """Return the top_k chunks for a query, in descending relevance order."""
    tokenized_query = query.lower().split()
    scores = index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]
