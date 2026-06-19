"""
Known-groups check on the Part-II held-out certifier.

Scores the 80 known-groups outputs with the certifier and tests whether it recovers the
designed C1<C2<C3 gradient (ablation=none) and the style-card ablation (full>none), using
the same analysis machinery as construct_validity.py.

Run: .venv/bin/python revision_analyses3.py  # writes runs/certifier_preflight.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from construct_validity import (
    load_cfg, jonckheere, cliffs_delta, cluster_bootstrap_diff, _paired_by_prompt, _simple_ci,
    COND_ORDER,
)
from partii_score import HeldOutCertifier


def certifier_known_groups(cfg) -> None:
    runs = Path(cfg["paths"]["runs_dir"])
    outs = pd.read_csv(runs / "outputs.csv")

    from corpus import load_pool, build_guest_floor
    target_texts = load_pool(cfg, "target")
    floor_texts = build_guest_floor(cfg) + load_pool(cfg, "floor_solo")
    print(f"[preflight] building certifier from {len(target_texts)} target / {len(floor_texts)} floor docs")
    cert = HeldOutCertifier(target_texts, floor_texts)

    outs["heldout_fidelity"] = [cert.score(str(t)) for t in outs["text"].tolist()]
    outs.to_csv(runs / "certifier_preflight.csv",
                columns=["output_id", "condition", "ablation", "prompt_id", "heldout_fidelity"],
                index=False)

    iters = int(cfg["analysis"]["bootstrap_iters"]); ci = float(cfg["analysis"]["ci"])
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    print("\n=== CERTIFIER known-groups means (within ablation=none, n=16/cell) ===")
    none = outs[outs.ablation == "none"]
    for c in COND_ORDER:
        v = none[none.condition == c]["heldout_fidelity"].to_numpy(float)
        lo, hi = _simple_ci(v, rng, iters, ci)
        print(f"  {c}: mean={v.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  n={len(v)}")

    groups = [none[none.condition == c]["heldout_fidelity"].to_numpy(float) for c in COND_ORDER]
    jt, z, p = jonckheere(groups)
    print(f"  Jonckheere-Terpstra (increasing C1<C2<C3, none): z={z:.3f}  p_one_sided={p:.5f}")

    print("\n=== CERTIFIER ordered contrasts (paired by prompt, cluster bootstrap) ===")
    for ca, cb in [("C1", "C2"), ("C2", "C3"), ("C1", "C3")]:
        paired = _paired_by_prompt(outs, "heldout_fidelity", ca, cb, "none")
        pt, lo, hi = cluster_bootstrap_diff(paired, rng, iters, ci)
        d = cliffs_delta(paired["b"].to_numpy(), paired["a"].to_numpy())
        print(f"  {cb}-{ca}(none): diff={pt:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  "
              f"delta={d:+.3f}  excl0={lo > 0 or hi < 0}")

    # style-card ablation full>none, pooled C2,C3, paired by (condition, prompt)
    sub = outs[outs.condition.isin(["C2", "C3"])]
    full = sub[sub.ablation == "full"].set_index(["condition", "prompt_id"])["heldout_fidelity"]
    nn = sub[sub.ablation == "none"].set_index(["condition", "prompt_id"])["heldout_fidelity"]
    paired = pd.DataFrame({"a": nn, "b": full}).dropna()
    pt, lo, hi = cluster_bootstrap_diff(paired, rng, iters, ci)
    d = cliffs_delta(paired["b"].to_numpy(), paired["a"].to_numpy())
    print("\n=== CERTIFIER style-card ablation (full-none, pooled C2/C3) ===")
    print(f"  diff={pt:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  delta={d:+.3f}  excl0={lo > 0 or hi < 0}")


def main() -> int:
    cfg = load_cfg()
    certifier_known_groups(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
