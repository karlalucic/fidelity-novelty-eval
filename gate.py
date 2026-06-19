"""Memorisation check for generated outputs.

For each output, measure verbatim overlap against its retrieved chunks and the
archive, using n-gram Jaccard and a sliding-window exact-overlap fraction (only
exact token matches count, per Ippolito et al.). Outputs with high overlap are
flagged and reported as a copy-rate rather than folded into the fidelity score.
"""

from __future__ import annotations

import pandas as pd


def ngram_jaccard(a: str, b: str, n: int) -> float:
    """Jaccard similarity of the n-gram sets of two texts (0..1)."""
    ta = a.lower().split()
    tb = b.lower().split()
    if len(ta) < n or len(tb) < n:
        return 0.0
    ga = {tuple(ta[i:i + n]) for i in range(len(ta) - n + 1)}
    gb = {tuple(tb[i:i + n]) for i in range(len(tb) - n + 1)}
    union = ga | gb
    if not union:
        return 0.0
    return len(ga & gb) / len(union)


def window_overlap(output: str, source: str, window_words: int) -> float:
    """Fraction of ``output`` sliding windows that appear verbatim in ``source`` (0..1).

    Exact-match only. Returns 0.0 if either text is shorter than ``window_words``.
    """
    out_tokens = output.lower().split()
    src_tokens = source.lower().split()
    if len(out_tokens) < window_words or len(src_tokens) < window_words:
        return 0.0
    src_windows = {
        tuple(src_tokens[i:i + window_words])
        for i in range(len(src_tokens) - window_words + 1)
    }
    total = len(out_tokens) - window_words + 1
    found = sum(
        1 for i in range(total)
        if tuple(out_tokens[i:i + window_words]) in src_windows
    )
    return found / total if total > 0 else 0.0


def gate_output(
    output: str,
    retrieved_chunks: list[str],
    archive_chunks: list[str],
    cfg: dict,
) -> dict:
    """Score one output for verbatim reuse.

    Returns ``{copy_rate, copy_flag, window_overlap, max_source}`` where:
      - copy_rate   = max n-gram Jaccard over (retrieved ∪ archive) and all configured n.
      - window_overlap = max sliding-window exact-overlap fraction over the same sources.
      - copy_flag   = (copy_rate > jaccard_threshold) OR (window_overlap > window_overlap_threshold).
    """
    g = cfg["gate"]
    ngram_orders = g["ngram_orders"]
    jaccard_threshold = g["jaccard_threshold"]
    window_words = g.get("window_words", 30)
    window_threshold = g.get("window_overlap_threshold", 0.5)

    sources = list(retrieved_chunks) + list(archive_chunks)

    max_jaccard = 0.0
    max_source = ""
    for chunk in sources:
        for n in ngram_orders:
            j = ngram_jaccard(output, chunk, n)
            if j > max_jaccard:
                max_jaccard = j
                max_source = chunk[:80]

    max_window = 0.0
    for chunk in sources:
        max_window = max(max_window, window_overlap(output, chunk, window_words))

    copy_flag = bool(max_jaccard > jaccard_threshold or max_window > window_threshold)
    return {
        "copy_rate": round(float(max_jaccard), 4),
        "copy_flag": copy_flag,
        "window_overlap": round(float(max_window), 4),
        "max_source": max_source,
    }


def gate_batch(
    outputs: dict,
    retrieved: dict,
    archive_chunks: list[str],
    cfg: dict,
) -> pd.DataFrame:
    """Gate every output; one row per output (output_id, copy_rate, copy_flag, window_overlap, max_source)."""
    rows = []
    for output_id, text in outputs.items():
        res = gate_output(text, retrieved.get(output_id, []), archive_chunks, cfg)
        rows.append({"output_id": output_id, **res})
    return pd.DataFrame(rows)
