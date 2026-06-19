"""
Two extra review-driven numbers over the existing CSVs.

E recomputes the MTMM convergent-minus-discriminant gap with the judge-fidelity x
judge-novelty heterotrait pair dropped, with a prompt cluster-bootstrap CI. F re-embeds
the 16 C1 generations with LUAR and checks how stable the C1 anchor centroid is.

Run:  .venv/bin/python revision_analyses2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from construct_validity import load_cfg, load_merged, FIDELITY, NOVELTY, _corr_matrix


def gap_from_matrix(cmat, drop_monomethod: bool) -> float:
    conv_pairs = ([(FIDELITY[i], FIDELITY[j]) for i in range(3) for j in range(i + 1, 3)]
                  + [(NOVELTY[i], NOVELTY[j]) for i in range(3) for j in range(i + 1, 3)])
    disc_pairs = [(f, n) for f in FIDELITY for n in NOVELTY]
    if drop_monomethod:
        disc_pairs = [(f, n) for (f, n) in disc_pairs
                      if not (f == "llm_judge_fidelity" and n == "llm_judge_novelty")]
    conv = np.nanmean([abs(cmat.loc[a, b]) for a, b in conv_pairs])
    disc = np.nanmean([abs(cmat.loc[a, b]) for a, b in disc_pairs])
    return float(conv), float(disc), float(conv - disc)


def mtmm_gap_excluding_monomethod(cfg) -> None:
    df = load_merged(cfg)
    c3 = df[df.condition == "C3"].copy()
    iters = int(cfg["analysis"]["bootstrap_iters"]); ci = float(cfg["analysis"]["ci"])
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    for drop in [False, True]:
        cmat = _corr_matrix(c3, "spearman")
        conv, disc, gap = gap_from_matrix(cmat, drop)
        prompts = c3["prompt_id"].unique()
        boot = []
        by = {p: c3[c3.prompt_id == p] for p in prompts}
        for _ in range(iters):
            pick = rng.choice(prompts, len(prompts), replace=True)
            rows = pd.concat([by[p] for p in pick], ignore_index=True)
            try:
                _, _, g = gap_from_matrix(_corr_matrix(rows, "spearman"), drop)
                if np.isfinite(g):
                    boot.append(g)
            except Exception:
                pass
        boot = np.array(boot)
        lo = float(np.percentile(boot, (1 - ci) / 2 * 100)); hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
        p_gap = float(np.mean(boot <= 0))
        tag = "EXCLUDING judge x judge monomethod pair" if drop else "ALL 9 discriminant pairs (paper)"
        print(f"[{tag}]")
        print(f"  convergent|r|={conv:.4f}  discriminant|r|={disc:.4f}  gap={gap:.4f}  "
              f"95% CI [{lo:.3f}, {hi:.3f}]  p(gap>0)={p_gap:.3f}")


def luar_anchor_stability(cfg) -> None:
    runs = Path(cfg["paths"]["runs_dir"])
    outs = pd.read_csv(runs / "outputs.csv")
    c1 = outs[outs.condition == "C1"].sort_values("prompt_id")
    texts = c1["text"].astype(str).tolist()
    print(f"[anchor] {len(texts)} C1 generations")
    try:
        import fidelity as fid
        embs = fid.luar_embed(texts, cfg)  # (16, d)
    except Exception as e:
        print(f"[anchor] LUAR unavailable, skipping ({e})")
        return
    embs = np.asarray(embs, float)

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n else v

    full = unit(embs.mean(axis=0))
    cos_loo = []
    for i in range(len(embs)):
        loo = unit(np.delete(embs, i, axis=0).mean(axis=0))
        cos_loo.append(float(np.dot(loo, full)))
    cos_loo = np.array(cos_loo)
    # split-half stability over random halves
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    halves = []
    idx = np.arange(len(embs))
    for _ in range(2000):
        rng.shuffle(idx)
        a = unit(embs[idx[:8]].mean(axis=0)); b = unit(embs[idx[8:]].mean(axis=0))
        halves.append(float(np.dot(a, b)))
    halves = np.array(halves)
    print(f"[anchor] leave-one-out centroid cosine to full centroid: "
          f"min={cos_loo.min():.4f} mean={cos_loo.mean():.4f}")
    print(f"[anchor] split-half (8 vs 8) centroid cosine: "
          f"mean={halves.mean():.4f} 2.5%={np.percentile(halves,2.5):.4f}")


def main() -> int:
    cfg = load_cfg()
    print("=" * 70); print("E. MTMM gap excluding the monomethod (judge x judge) pair"); print("=" * 70)
    mtmm_gap_excluding_monomethod(cfg)
    print("\n" + "=" * 70); print("F. LUAR C1-anchor stability"); print("=" * 70)
    luar_anchor_stability(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
