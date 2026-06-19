"""Novelty proxies: dual divergence (embedding distance from nearest retrieved
chunk and from the C1 generic-LLM centroid) plus distinct-n. Embeddings are
injectable so tests run without a network."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def embed(
    texts: list[str],
    cfg: dict,
    _model=None,
) -> np.ndarray:
    """all-MiniLM-L6-v2 embeddings; _model (callable: texts -> np.ndarray) is
    injectable for tests. Returns shape (len(texts), embedding_dim)."""
    if _model is not None:
        result = _model(texts)
        return np.array(result)

    from sentence_transformers import SentenceTransformer  # type: ignore

    model_name = cfg["models"]["embed"]
    device = cfg["models"].get("device", "cpu")
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, convert_to_numpy=True)


def c1_centroid(c1_embs: np.ndarray) -> np.ndarray:
    """Mean generic-LLM-output embedding (the novelty anchor). c1_embs has shape
    (n, dim); returns the centroid of shape (dim,)."""
    if c1_embs.shape[0] == 0:
        raise ValueError("c1_embs is empty; cannot compute centroid")
    return c1_embs.mean(axis=0)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def dual_divergence(
    output_emb: np.ndarray,
    retrieved_embs: np.ndarray,
    c1_centroid_emb: np.ndarray,
) -> dict:
    """Return ``{dist_nearest_chunk, dist_c1_centroid, dual}``, where dual is the
    harmonic mean of the two distances and rewards outputs far from both the
    nearest chunk and the C1 centroid."""
    if retrieved_embs.shape[0] == 0:
        dist_nearest = 1.0  # no chunks -> maximally divergent from retrieved
    else:
        dists = [_cosine_dist(output_emb, re) for re in retrieved_embs]
        dist_nearest = float(min(dists))

    dist_c1 = _cosine_dist(output_emb, c1_centroid_emb)

    if dist_nearest + dist_c1 == 0:
        dual = 0.0
    else:
        dual = 2 * dist_nearest * dist_c1 / (dist_nearest + dist_c1)

    return {
        "dist_nearest_chunk": round(dist_nearest, 4),
        "dist_c1_centroid": round(dist_c1, 4),
        "dual": round(dual, 4),
    }


def distinct_n(text: str, orders: list[int]) -> dict:
    """Distinct-n lexical diversity per order.

    Returns ``{distinct_1: float, distinct_2: float, ...}`` where each value
    is the ratio of unique n-grams to total n-grams (Li et al. 2016).
    """
    tokens = text.lower().split()
    result: dict[str, float] = {}
    for n in orders:
        if len(tokens) < n:
            result[f"distinct_{n}"] = 0.0
            continue
        all_grams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
        total = len(all_grams)
        unique = len(set(all_grams))
        result[f"distinct_{n}"] = round(unique / total, 4) if total > 0 else 0.0
    return result


def score_novelty(
    outputs: dict,
    retrieved: dict,
    c1_embs: np.ndarray,
    cfg: dict,
    _embeds: Optional[dict] = None,
) -> pd.DataFrame:
    """Per-output dual divergence and distinct-n. ``retrieved`` maps each
    output_id to its retrieved chunks (C3) or is empty; ``_embeds`` can supply
    pre-computed embeddings. Returns a DataFrame with columns output_id,
    dist_nearest_chunk, dist_c1_centroid, dual, distinct_1, distinct_2."""
    orders = cfg["novelty"]["distinct_n_orders"]
    centroid = c1_centroid(c1_embs)

    if _embeds is not None:
        output_embs = {oid: _embeds[oid] for oid in outputs if oid in _embeds}
        retrieved_embs: dict[str, np.ndarray] = {
            oid: _embeds.get(f"__retrieved_{oid}__", np.zeros((0, centroid.shape[0])))
            for oid in outputs
        }
    else:
        all_texts = list(outputs.values())
        all_ids = list(outputs.keys())
        emb_arr = embed(all_texts, cfg)
        output_embs = {oid: emb_arr[i] for i, oid in enumerate(all_ids)}

        retrieved_embs = {}
        for oid in outputs:
            chunks = retrieved.get(oid, [])
            if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
                chunk_texts = [c["text"] for c in chunks]
            else:
                chunk_texts = [c for c in chunks if isinstance(c, str)]
            if chunk_texts:
                retrieved_embs[oid] = embed(chunk_texts, cfg)
            else:
                retrieved_embs[oid] = np.zeros((0, emb_arr.shape[1]))

    rows: list[dict] = []
    for output_id, text in outputs.items():
        emb = output_embs.get(output_id)
        if emb is None:
            continue

        ret_embs = retrieved_embs.get(output_id, np.zeros((0, centroid.shape[0])))
        dd = dual_divergence(emb, ret_embs, centroid)
        dn = distinct_n(text, orders)

        rows.append({"output_id": output_id, **dd, **dn})

    return pd.DataFrame(rows)
