"""
Score Part 2 candidates on CPU: calibrated LUAR fidelity, Burrows delta, distinct-n
novelty, nearest-archive distance, a LUAR-independent char-ngram TF-IDF certifier, and
the C3 memorisation gate. Reference embeddings are cached under runs/.

Run:  python partii_score.py            # runs/candidates.csv -> runs/candidate_scores.csv
      python partii_score.py --repair   # also write the dual-divergence old-vs-new table
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import fidelity as fid
import novelty as nov


def load_cfg(p="config.yaml") -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def _c1_texts(cfg) -> list[str]:
    runs = Path(cfg["paths"]["runs_dir"])
    df = pd.read_csv(runs / "outputs.csv")
    return df[df.condition == "C1"]["text"].astype(str).tolist()


def build_luar_refs(cfg) -> dict:
    runs = Path(cfg["paths"]["runs_dir"]); cache = runs / "ref_luar.npz"
    if cache.exists():
        z = np.load(cache)
        return dict(target=z["target"], floor=z["floor"], c1=z["c1"])
    from corpus import load_pool, build_guest_floor
    print("[score] embedding LUAR references (one-time; cached)...")
    luar = fid._load_luar(cfg)
    target = fid.luar_embed(load_pool(cfg, "target"), cfg, _model=luar).mean(axis=0)
    floor_texts = build_guest_floor(cfg) + load_pool(cfg, "floor_solo")
    floor = fid.luar_embed(floor_texts, cfg, _model=luar)
    c1 = fid.luar_embed(_c1_texts(cfg), cfg, _model=luar).mean(axis=0)
    np.savez(cache, target=target, floor=floor, c1=c1)
    return dict(target=target, floor=floor, c1=c1)


def build_minilm_refs(cfg, archive_chunks) -> dict:
    runs = Path(cfg["paths"]["runs_dir"]); cache = runs / "ref_minilm.npz"
    if cache.exists():
        z = np.load(cache)
        return dict(archive=z["archive"], c1=z["c1"])
    print("[score] embedding MiniLM references (archive + C1; one-time; cached)...")
    archive = nov.embed([c["text"] for c in archive_chunks], cfg)
    c1 = nov.embed(_c1_texts(cfg), cfg).mean(axis=0)
    np.savez(cache, archive=archive, c1=c1)
    return dict(archive=archive, c1=c1)


def _char_ngrams(text: str, lo=3, hi=5) -> list[str]:
    t = " ".join(text.lower().split())
    grams = []
    for n in range(lo, hi + 1):
        grams += [t[i:i + n] for i in range(len(t) - n + 1)]
    return grams


def _passages(texts: list[str], words=300) -> list[str]:
    out = []
    for t in texts:
        toks = t.split()
        for i in range(0, len(toks), words):
            seg = toks[i:i + words]
            if len(seg) >= 50:
                out.append(" ".join(seg))
    return out


class HeldOutCertifier:
    """Char-ngram TF-IDF; fidelity = cosine(text, target_centroid) - cosine(text, floor_centroid)."""

    def __init__(self, target_texts, floor_texts, max_features=8000):
        tp = _passages(target_texts); fp = _passages(floor_texts)
        docs = tp + fp
        df = Counter()
        doc_grams = []
        for d in docs:
            g = set(_char_ngrams(d))
            doc_grams.append(g)
            df.update(g)
        vocab = [w for w, _ in df.most_common(max_features)]
        self.vocab = {w: i for i, w in enumerate(vocab)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1.0 for w in vocab])
        self.target_centroid = self._centroid(tp)
        self.floor_centroid = self._centroid(fp)

    def _vec(self, text: str) -> np.ndarray:
        c = Counter(g for g in _char_ngrams(text) if g in self.vocab)
        v = np.zeros(len(self.vocab))
        if not c:
            return v
        tot = sum(c.values())
        for g, k in c.items():
            v[self.vocab[g]] = (k / tot) * self.idf[self.vocab[g]]
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    def _centroid(self, docs) -> np.ndarray:
        vs = [self._vec(d) for d in docs]
        m = np.mean(vs, axis=0)
        nrm = np.linalg.norm(m)
        return m / nrm if nrm > 0 else m

    def score(self, text: str) -> float:
        v = self._vec(text)
        return float(v @ self.target_centroid - v @ self.floor_centroid)


def _nearest_over_archive(emb: np.ndarray, archive_embs: np.ndarray) -> float:
    a = emb / (np.linalg.norm(emb) + 1e-12)
    A = archive_embs / (np.linalg.norm(archive_embs, axis=1, keepdims=True) + 1e-12)
    return float(1.0 - np.max(A @ a))  # distance to nearest archive chunk


def score_candidates(cfg, cands: pd.DataFrame) -> pd.DataFrame:
    from corpus import load_pool, build_guest_floor, chunk_archive
    target_texts = load_pool(cfg, "target")
    floor_texts = build_guest_floor(cfg) + load_pool(cfg, "floor_solo")
    archive_chunks = chunk_archive(target_texts, cfg)

    luar_ref = build_luar_refs(cfg)
    mini_ref = build_minilm_refs(cfg, archive_chunks)
    certifier = HeldOutCertifier(target_texts, floor_texts)
    bf = fid.burrows_preflight(cfg); burrows_ok = bf["pass"]
    print(f"[score] burrows preflight: {bf}")

    texts = cands["text"].astype(str).tolist()
    print(f"[score] LUAR-embedding {len(texts)} candidates...")
    luar_embs = fid.luar_embed(texts, cfg)
    print(f"[score] MiniLM-embedding {len(texts)} candidates...")
    mini_embs = nov.embed(texts, cfg)
    orders = cfg["novelty"]["distinct_n_orders"]

    rows = []
    for i, r in cands.reset_index(drop=True).iterrows():
        text = str(r["text"])
        cal = fid.calibrated_luar(luar_embs[i], luar_ref["target"], luar_ref["floor"], luar_ref["c1"])
        bd = fid.burrows_delta(text, target_texts, floor_texts, cfg) if burrows_ok else float("nan")
        dn = nov.distinct_n(text, orders)
        dual_rep = _nearest_over_archive(mini_embs[i], mini_ref["archive"])
        dist_c1 = nov._cosine_dist(mini_embs[i], mini_ref["c1"])
        rows.append(dict(
            candidate_id=r["candidate_id"], condition=r["condition"], ablation=r["ablation"],
            prompt_id=r["prompt_id"], n_idx=int(r["n_idx"]),
            calibrated_luar=cal, neg_burrows=(-bd if not np.isnan(bd) else float("nan")),
            burrows_delta=bd, distinct_1=dn["distinct_1"], distinct_2=dn["distinct_2"],
            dual_repaired=round(dual_rep, 4), dist_c1_centroid=round(dist_c1, 4),
            heldout_fidelity=round(certifier.score(text), 6),
        ))
    df = pd.DataFrame(rows)

    # memorisation gate for C3 candidates, against the archive
    import gate as gate_mod
    c3 = df[df.condition == "C3"]
    if len(c3):
        c3_texts = {r.candidate_id: str(cands[cands.candidate_id == r.candidate_id]["text"].iloc[0])
                    for r in c3.itertuples()}
        archive_texts = [c["text"] for c in archive_chunks]
        gate_df = gate_mod.gate_batch(c3_texts, {k: [] for k in c3_texts}, archive_texts, cfg)
        gate_df = gate_df.rename(columns={"output_id": "candidate_id"})
        df = df.merge(gate_df[["candidate_id", "copy_rate", "copy_flag"]], on="candidate_id", how="left")
    else:
        df["copy_rate"] = np.nan; df["copy_flag"] = False
    return df


def repair_dual_on_existing(cfg) -> pd.DataFrame:
    """Recompute nearest-over-ALL-archive for the 80 pilot outputs; merge with the old per-retrieval value."""
    from corpus import load_pool, chunk_archive
    runs = Path(cfg["paths"]["runs_dir"])
    outs = pd.read_csv(runs / "outputs.csv")
    old = pd.read_csv(runs / "novelty_scores.csv")[["output_id", "dist_nearest_chunk", "dual"]]
    archive_chunks = chunk_archive(load_pool(cfg, "target"), cfg)
    mini_ref = build_minilm_refs(cfg, archive_chunks)
    embs = nov.embed(outs["text"].astype(str).tolist(), cfg)
    new = [ _nearest_over_archive(embs[i], mini_ref["archive"]) for i in range(len(outs)) ]
    rep = outs[["output_id", "condition"]].copy()
    rep["dist_nearest_old"] = old.set_index("output_id").reindex(outs.output_id)["dist_nearest_chunk"].values
    rep["dist_nearest_new"] = np.round(new, 4)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", action="store_true", help="also write dual-divergence old-vs-new table.")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_cfg(args.config); runs = Path(cfg["paths"]["runs_dir"])

    if args.repair:
        rep = repair_dual_on_existing(cfg)
        rep.to_csv(runs / "dual_div_repair.csv", index=False)
        summ = rep.groupby("condition")[["dist_nearest_old", "dist_nearest_new"]].mean().round(4)
        print("=== DUAL-DIVERGENCE REPAIR (mean nearest-archive distance by condition) ===")
        print(summ.to_string())

    cand_path = runs / "candidates.csv"
    if cand_path.exists():
        cands = pd.read_csv(cand_path)
        print(f"[score] scoring {len(cands)} candidates from {cand_path}")
        scored = score_candidates(cfg, cands)
        scored.to_csv(runs / "candidate_scores.csv", index=False)
        print(f"[score] wrote candidate_scores.csv ({len(scored)} rows)")
        print(scored.groupby(["condition", "ablation"])[
            ["calibrated_luar", "distinct_2", "heldout_fidelity"]].mean().round(3).to_string())
    else:
        print(f"[score] no {cand_path} yet; run partii_generate.py first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
