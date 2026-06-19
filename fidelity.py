"""Fidelity proxies: calibrated-LUAR cosine and Burrows's Delta.

LUAR cosine is scored against a guest floor and a C1 centroid, with a sanity
anchor. Burrows's Delta runs over top-N function words, gated by a held-out
preflight that skips the proxy if it can't separate real Senra from the floor.
Embeddings are injectable for testing without loading the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _load_luar(cfg: dict):
    """Load LUAR-MUD model and tokenizer."""
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    model_name = cfg["models"]["luar"]
    device = cfg["models"].get("device", "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def luar_embed(
    texts: list[str],
    cfg: dict,
    _model=None,
) -> np.ndarray:
    """LUAR embeddings, chunked to luar_max_tokens and mean-pooled.

    Returns an array of shape (len(texts), embedding_dim). Pass _model to
    inject a (model, tokenizer) tuple and skip loading the real model.
    """
    import torch  # type: ignore

    max_tokens = cfg["models"]["luar_max_tokens"]
    device = cfg["models"].get("device", "cpu")

    if _model is None:
        model, tokenizer = _load_luar(cfg)
    else:
        model, tokenizer = _model

    all_embs: list[np.ndarray] = []

    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = tokens["input_ids"][0]
        total_tokens = len(input_ids)

        if total_tokens <= max_tokens:
            chunks_ids = [input_ids]
        else:
            # sliding chunks with 50-token overlap
            overlap = 50
            step = max_tokens - overlap
            chunks_ids = [
                input_ids[i: i + max_tokens]
                for i in range(0, total_tokens, step)
                if i < total_tokens
            ]

        chunk_embs: list[np.ndarray] = []
        for chunk in chunks_ids:
            # LUAR-MUD expects (batch, episodes, seq_len); each chunk is one episode
            ids = chunk.unsqueeze(0).unsqueeze(0).to(device)
            am = torch.ones(1, 1, len(chunk), dtype=torch.long).to(device)
            with torch.no_grad():
                output = model(input_ids=ids, attention_mask=am)

            if torch.is_tensor(output):
                emb = output[0].cpu().numpy()
            elif hasattr(output, "last_hidden_state"):
                emb = output.last_hidden_state[0].cpu().numpy()
                if emb.ndim > 1:
                    emb = emb.mean(axis=0)
            elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                emb = output.pooler_output[0].cpu().numpy()
            else:
                emb = np.asarray(output[0].cpu()).reshape(-1)
            chunk_embs.append(emb)

        # mean-pool episode embeddings into one author vector
        text_emb = np.mean(chunk_embs, axis=0)
        all_embs.append(text_emb)

    return np.array(all_embs)


def luar_sanity_anchor(cfg: dict, _model=None) -> dict:
    """Run LUAR on the anchor texts; return {pass, margin, self_cosine}.

    Checks the model is functional: same-author cosine should exceed
    luar_sanity_min_self_cosine. Returns pass=None if the anchor files are
    missing.
    """
    runs_dir = Path(cfg["paths"]["runs_dir"])
    anchor_files = cfg["fidelity"]["luar_sanity_anchor_texts"]
    texts: list[str] = []
    missing: list[str] = []
    for fname in anchor_files:
        fpath = runs_dir / fname
        if fpath.exists():
            texts.append(fpath.read_text(encoding="utf-8"))
        else:
            missing.append(fname)

    if len(missing) == len(anchor_files):
        return {"pass": None, "reason": "anchor files not found; skipping LUAR sanity anchor"}

    if len(texts) < 2:
        return {"pass": None, "reason": f"need >=2 anchor files; {missing} missing"}

    embs = luar_embed(texts, cfg, _model=_model)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # cosine between the two same-author anchor texts vs the threshold
    self_cosine = cosine(embs[0], embs[1])
    min_threshold = cfg["fidelity"]["luar_sanity_min_self_cosine"]
    passed = self_cosine > min_threshold
    return {
        "pass": passed,
        "margin": round(self_cosine - min_threshold, 4),
        "self_cosine": round(self_cosine, 4),
    }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def calibrated_luar(
    output_emb: np.ndarray,
    target_emb: np.ndarray,
    floor_embs: np.ndarray,
    c1_centroid: np.ndarray,
) -> float:
    """Floor-scaled gap between target similarity and C1 (generic-LLM) similarity.

    Two z-scores against the floor distribution are subtracted:
        z     = (sim_to_target - mean(sim_to_floor)) / (std(sim_to_floor) + 1e-8)
        z_c1  = (sim_to_c1     - mean(sim_to_floor)) / (std(sim_to_floor) + 1e-8)
        score = z - z_c1 = (sim_to_target - sim_to_c1) / (std(sim_to_floor) + 1e-8)
    The floor mean cancels, leaving the target-vs-C1 cosine gap rescaled by the
    floor's spread. Higher means more target-like than the generic-LLM baseline.
    """
    sim_to_target = _cosine_sim(output_emb, target_emb)
    sim_to_c1 = _cosine_sim(output_emb, c1_centroid)

    if floor_embs.shape[0] == 0:
        # no floor: fall back to the raw target - c1 delta
        return round(sim_to_target - sim_to_c1, 4)

    floor_sims = np.array([_cosine_sim(output_emb, fe) for fe in floor_embs])
    floor_mean = float(np.mean(floor_sims))
    floor_std = float(np.std(floor_sims)) + 1e-8

    z = (sim_to_target - floor_mean) / floor_std
    z_c1 = (sim_to_c1 - floor_mean) / floor_std

    return round(float(z - z_c1), 4)


def _get_function_words(
    all_texts: list[str],
    n: int,
) -> list[str]:
    """Top-n most frequent closed-class function words across all_texts."""
    FUNCTION_WORDS = {
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
        "for", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "it", "its", "this",
        "that", "these", "those", "he", "she", "they", "we", "you", "i", "me",
        "him", "her", "them", "us", "my", "your", "his", "their", "our", "its",
        "not", "no", "so", "if", "as", "up", "out", "about", "into", "than",
        "then", "when", "where", "who", "which", "what", "how", "all", "just",
        "more", "also", "very", "only", "even", "back", "there", "here",
        "over", "after", "before", "through", "between", "both", "each",
        "other", "such", "same", "own", "off",
    }
    from collections import Counter
    counts: Counter = Counter()
    for text in all_texts:
        for tok in text.lower().split():
            tok = tok.strip(".,!?;:\"'()[]{}—–-")
            if tok in FUNCTION_WORDS:
                counts[tok] += 1
    return [w for w, _ in counts.most_common(n)]


def _text_freq_vector(text: str, vocab: list[str]) -> np.ndarray:
    """Relative frequency vector of vocab words in text."""
    from collections import Counter
    tokens = [t.strip(".,!?;:\"'()[]{}—–-").lower() for t in text.split()]
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return np.zeros(len(vocab))
    return np.array([counts.get(w, 0) / total for w in vocab])


def burrows_delta(
    text: str,
    target_corpus: list[str],
    floor_corpus: list[str],
    cfg: dict,
) -> float:
    """Burrows's Delta on the top-N function words; lower is closer to target.

        Delta(text, corpus) = mean_over_vocab( |z_text_w - mean(z_corpus_w)| )
    with z scores standardised within the combined corpus.
    """
    n_fw = cfg["fidelity"]["burrows"]["n_function_words"]
    all_texts = target_corpus + floor_corpus + [text]
    vocab = _get_function_words(all_texts, n_fw)
    if not vocab:
        return float("nan")

    # frequency matrix: rows = documents, cols = vocab
    all_docs = target_corpus + floor_corpus + [text]
    matrix = np.array([_text_freq_vector(doc, vocab) for doc in all_docs])

    # z-score each feature across all documents
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0) + 1e-8
    z_matrix = (matrix - means) / stds

    n_target = len(target_corpus)
    z_text = z_matrix[-1]
    z_target = z_matrix[:n_target]
    z_floor = z_matrix[n_target: n_target + len(floor_corpus)]

    target_mean_z = z_target.mean(axis=0)
    delta_to_target = float(np.mean(np.abs(z_text - target_mean_z)))

    return round(delta_to_target, 4)


def burrows_preflight(cfg: dict) -> dict:
    """Rank held-out real-Senra vs real-floor passages; return {pass, accuracy}.

    Below preflight_min_accuracy the proxy is flagged and skipped.
    """
    from corpus import held_out_passages  # type: ignore

    min_acc = cfg["fidelity"]["burrows"]["preflight_min_accuracy"]
    passage_words = cfg["fidelity"]["burrows"]["preflight_passage_words"]
    n_passages = 10

    try:
        senra_passages, floor_passages = held_out_passages(cfg, n_passages, passage_words)
    except Exception as e:
        return {"pass": False, "accuracy": 0.0, "reason": f"held_out_passages failed: {e}"}

    if not senra_passages or not floor_passages:
        return {"pass": False, "accuracy": 0.0, "reason": "empty passage sets for pre-flight"}

    target_corpus = senra_passages[: max(1, len(senra_passages) - 2)]
    floor_corpus_train = floor_passages[: max(1, len(floor_passages) - 2)]
    test_senra = senra_passages[max(1, len(senra_passages) - 2):]
    test_floor = floor_passages[max(1, len(floor_passages) - 2):]

    correct = 0
    total = len(test_senra) + len(test_floor)
    if total == 0:
        return {"pass": False, "accuracy": 0.0, "reason": "no test passages"}

    for text in test_senra:
        delta_t = burrows_delta(text, target_corpus, floor_corpus_train, cfg)
        delta_f = burrows_delta(text, floor_corpus_train, target_corpus, cfg)
        if not np.isnan(delta_t) and not np.isnan(delta_f) and delta_t < delta_f:
            correct += 1

    for text in test_floor:
        delta_t = burrows_delta(text, target_corpus, floor_corpus_train, cfg)
        delta_f = burrows_delta(text, floor_corpus_train, target_corpus, cfg)
        if not np.isnan(delta_t) and not np.isnan(delta_f) and delta_f < delta_t:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    passed = accuracy >= min_acc
    return {
        "pass": passed,
        "accuracy": round(accuracy, 4),
        "reason": "ok" if passed else f"accuracy {accuracy:.3f} < {min_acc} -> SKIPPED",
    }


def score_fidelity(
    outputs: dict,
    cfg: dict,
    _luar=None,
    _embeds: Optional[dict] = None,
) -> pd.DataFrame:
    """Per-output calibrated_luar and burrows_delta (when preflight passes).

    outputs maps {output_id: text}. Pass _luar to inject (model, tokenizer),
    or _embeds to inject precomputed {output_id: np.ndarray} and skip loading.
    Returns columns: output_id, calibrated_luar, burrows_delta, burrows_skipped.
    """
    from corpus import load_pool, build_guest_floor  # type: ignore

    target_texts = load_pool(cfg, "target")
    floor_guest = build_guest_floor(cfg)
    floor_solo = load_pool(cfg, "floor_solo")
    floor_texts = floor_guest + floor_solo

    bf = burrows_preflight(cfg)
    burrows_ok = bf["pass"]

    rows: list[dict] = []

    if _embeds is not None:
        output_embs = {oid: _embeds[oid] for oid in outputs if oid in _embeds}
        target_emb = _embeds.get("__target_centroid__", np.zeros(4))
        floor_embs_arr = _embeds.get("__floor_embs__", np.zeros((1, 4)))
        c1_centroid = _embeds.get("__c1_centroid__", np.zeros(4))
    else:
        all_output_texts = list(outputs.values())
        all_output_ids = list(outputs.keys())
        emb_array = luar_embed(all_output_texts, cfg, _model=_luar)
        output_embs = {oid: emb_array[i] for i, oid in enumerate(all_output_ids)}

        if target_texts:
            t_embs = luar_embed(target_texts, cfg, _model=_luar)
            target_emb = t_embs.mean(axis=0)
        else:
            target_emb = np.zeros(emb_array.shape[1])

        if floor_texts:
            f_embs = luar_embed(floor_texts, cfg, _model=_luar)
            floor_embs_arr = f_embs
        else:
            floor_embs_arr = np.zeros((0, emb_array.shape[1]))

        # C1 centroid over outputs whose id contains "C1"
        c1_outputs = [v for k, v in outputs.items() if "C1" in k]
        if c1_outputs:
            c1_embs = luar_embed(c1_outputs, cfg, _model=_luar)
            c1_centroid = c1_embs.mean(axis=0)
        else:
            c1_centroid = np.zeros(emb_array.shape[1])

    for output_id, text in outputs.items():
        emb = output_embs.get(output_id)
        if emb is None:
            continue

        cal_luar = calibrated_luar(emb, target_emb, floor_embs_arr, c1_centroid)

        if burrows_ok:
            b_delta = burrows_delta(text, target_texts, floor_texts, cfg)
        else:
            b_delta = float("nan")

        rows.append(
            {
                "output_id": output_id,
                "calibrated_luar": cal_luar,
                "burrows_delta": b_delta,
                "burrows_skipped": not burrows_ok,
            }
        )

    return pd.DataFrame(rows)
