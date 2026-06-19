"""
Construct-validity analyses over the metric CSVs: known-groups (fidelity gradient,
discriminant dissociation, style-card ablation), MTMM convergent/discriminant gap, and
the Holm-corrected k=4 confirmatory family. The unit of analysis is the prompt and every
CI is a cluster bootstrap that resamples prompts; proxies are oriented so higher = more
of the construct.

Run: python construct_validity.py  (writes known_groups_*.csv, mtmm_*.csv, confirmatory.csv)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from analysis import holm

# proxies oriented so higher = more of the construct
FIDELITY = ["calibrated_luar", "neg_burrows", "llm_judge_fidelity"]
NOVELTY = ["dual", "distinct_2", "llm_judge_novelty"]
PROXIES = FIDELITY + NOVELTY
COND_ORDER = ["C1", "C2", "C3"]


def load_cfg(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_merged(cfg: dict) -> pd.DataFrame:
    """Merge outputs + fidelity + novelty + both judges into one oriented frame."""
    runs = Path(cfg["paths"]["runs_dir"])
    outs = pd.read_csv(runs / "outputs.csv")[["output_id", "condition", "ablation", "prompt_id"]]
    fid = pd.read_csv(runs / "fidelity_scores.csv")[["output_id", "calibrated_luar", "burrows_delta"]]
    nov = pd.read_csv(runs / "novelty_scores.csv")[
        ["output_id", "dual", "distinct_1", "distinct_2", "dist_c1_centroid"]
    ]
    jf = pd.read_csv(runs / "judge_fidelity.csv")[["output_id", "llm_judge_fidelity"]]
    jn = pd.read_csv(runs / "judge_novelty.csv")[["output_id", "llm_judge_novelty"]]

    df = outs.merge(fid, on="output_id", how="left").merge(nov, on="output_id", how="left")
    df = df.merge(jf, on="output_id", how="left").merge(jn, on="output_id", how="left")

    # Burrows delta is reverse-coded, so flip it
    df["neg_burrows"] = -df["burrows_delta"]
    return df


def cliffs_delta(b: np.ndarray, a: np.ndarray) -> float:
    """Cliff's delta for (b vs a): P(b>a) - P(a>b). Positive => b tends larger."""
    b = np.asarray(b, float); a = np.asarray(a, float)
    b = b[~np.isnan(b)]; a = a[~np.isnan(a)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((bb > a).sum() for bb in b)
    lt = sum((bb < a).sum() for bb in b)
    return (gt - lt) / (len(a) * len(b))


def jonckheere(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """Jonckheere-Terpstra trend test for an increasing ordered alternative.

    Returns (JT, z, one_sided_p) via the normal approximation with the tie-corrected
    variance (Hollander, Wolfe & Chicken). The tie correction matters for the discrete
    LLM-judge proxies; for near-continuous proxies the tie terms vanish.
    """
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    ns = [float(len(g)) for g in groups]
    N = float(sum(ns))
    jt = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            xi, xj = groups[i], groups[j]
            gt = sum((xj > x).sum() for x in xi)
            ties = sum((xj == x).sum() for x in xi)
            jt += gt + 0.5 * ties
    mean = (N ** 2 - sum(n ** 2 for n in ns)) / 4.0

    # tie-corrected null variance; tk = multiplicities of tied values in the pooled sample
    allvals = np.concatenate(groups) if N else np.array([])
    _, tcounts = np.unique(allvals, return_counts=True)
    tk = tcounts.astype(float)
    s_a = lambda arr: float(sum(x * (x - 1) * (2 * x + 5) for x in arr))
    s_b = lambda arr: float(sum(x * (x - 1) * (x - 2) for x in arr))
    s_c = lambda arr: float(sum(x * (x - 1) for x in arr))
    var = (N * (N - 1) * (2 * N + 5) - s_a(ns) - s_a(tk)) / 72.0
    if N > 2:
        var += (s_b(ns) * s_b(tk)) / (36.0 * N * (N - 1) * (N - 2))
    if N > 1:
        var += (s_c(ns) * s_c(tk)) / (8.0 * N * (N - 1))

    z = (jt - mean) / np.sqrt(var) if var > 0 else float("nan")
    p = 1.0 - stats.norm.cdf(z) if np.isfinite(z) else float("nan")
    return float(jt), float(z), float(p)


def _rng(cfg: dict) -> np.random.Generator:
    return np.random.default_rng(int(cfg.get("seed", 0)))


def _paired_by_prompt(df: pd.DataFrame, col: str, cond_a: str, cond_b: str,
                      ablation: str | None) -> pd.DataFrame:
    """Return a frame indexed by prompt with columns a,b for the two conditions (same ablation)."""
    sub = df if ablation is None else df[df["ablation"] == ablation]
    a = sub[sub["condition"] == cond_a].set_index("prompt_id")[col]
    b = sub[sub["condition"] == cond_b].set_index("prompt_id")[col]
    return pd.DataFrame({"a": a, "b": b}).dropna()


def cluster_bootstrap_diff(paired: pd.DataFrame, rng: np.random.Generator,
                           iters: int, ci: float) -> tuple[float, float, float]:
    """Cluster bootstrap (resample prompts) for mean(b) - mean(a) on a paired-by-prompt frame."""
    a = paired["a"].to_numpy(float); b = paired["b"].to_numpy(float)
    n = len(a)
    point = float(np.mean(b) - np.mean(a))
    boot = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        boot.append(float(np.mean(b[idx]) - np.mean(a[idx])))
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return point, lo, hi


def known_groups(df: pd.DataFrame, cfg: dict) -> dict:
    iters = int(cfg["analysis"]["bootstrap_iters"]); ci = float(cfg["analysis"]["ci"])
    rng = _rng(cfg)
    out: dict = {"means": [], "contrasts": [], "ablation": []}

    # means per proxy x condition x ablation, with per-cell bootstrap CI
    for proxy in PROXIES:
        for abl in ["none", "full"]:
            for cond in COND_ORDER:
                vals = df[(df.condition == cond) & (df.ablation == abl)][proxy].dropna().to_numpy()
                if len(vals) == 0:
                    continue
                pt, lo, hi = (float(np.mean(vals)), *(_simple_ci(vals, rng, iters, ci)))
                out["means"].append(dict(proxy=proxy, ablation=abl, condition=cond,
                                         n=len(vals), mean=round(pt, 4),
                                         ci_lo=round(lo, 4), ci_hi=round(hi, 4)))

    # fidelity gradient within ablation=none: JT trend + ordered contrasts
    for proxy in FIDELITY:
        conds = [c for c in COND_ORDER
                 if len(df[(df.condition == c) & (df.ablation == "none")][proxy].dropna())]
        groups = [df[(df.condition == c) & (df.ablation == "none")][proxy].dropna().to_numpy()
                  for c in conds]
        jt, z, p = jonckheere(groups) if len(groups) >= 2 else (float("nan"),) * 3
        out["contrasts"].append(dict(proxy=proxy, kind="JT_trend_increasing(none)",
                                     conditions="<".join(conds), z=round(z, 3), p_one_sided=round(p, 5)))
        for ca, cb in [("C1", "C2"), ("C2", "C3"), ("C1", "C3")]:
            if ca not in conds or cb not in conds:
                continue
            paired = _paired_by_prompt(df, proxy, ca, cb, "none")
            pt, lo, hi = cluster_bootstrap_diff(paired, rng, iters, ci)
            d = cliffs_delta(paired["b"].to_numpy(), paired["a"].to_numpy())
            out["contrasts"].append(dict(proxy=proxy, kind=f"{cb}-{ca}(none)", n=len(paired),
                                         diff=round(pt, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                         cliffs_delta=round(d, 3),
                                         ci_excludes_zero=bool(lo > 0 or hi < 0)))

    # discriminant dissociation: distinct_2 falls C1->C3 within none
    paired = _paired_by_prompt(df, "distinct_2", "C1", "C3", "none")
    pt, lo, hi = cluster_bootstrap_diff(paired, rng, iters, ci)
    d = cliffs_delta(paired["b"].to_numpy(), paired["a"].to_numpy())
    out["contrasts"].append(dict(proxy="distinct_2", kind="C3-C1(none) [novelty down]", n=len(paired),
                                 diff=round(pt, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                 cliffs_delta=round(d, 3), ci_excludes_zero=bool(lo > 0 or hi < 0)))

    # style-card ablation: LUAR full vs none, paired by prompt (C2, C3, pooled)
    for cond in ["C2", "C3", "pooled"]:
        if cond == "pooled":
            sub = df[df.condition.isin(["C2", "C3"])]
            full = sub[sub.ablation == "full"].set_index(["condition", "prompt_id"])["calibrated_luar"]
            none = sub[sub.ablation == "none"].set_index(["condition", "prompt_id"])["calibrated_luar"]
            paired = pd.DataFrame({"a": none, "b": full}).dropna()
        else:
            paired = pd.DataFrame({
                "a": df[(df.condition == cond) & (df.ablation == "none")].set_index("prompt_id")["calibrated_luar"],
                "b": df[(df.condition == cond) & (df.ablation == "full")].set_index("prompt_id")["calibrated_luar"],
            }).dropna()
        pt, lo, hi = cluster_bootstrap_diff(paired, rng, iters, ci)
        try:
            w_p = stats.wilcoxon(paired["b"], paired["a"], alternative="greater").pvalue
        except ValueError:
            w_p = float("nan")
        d = cliffs_delta(paired["b"].to_numpy(), paired["a"].to_numpy())
        out["ablation"].append(dict(proxy="calibrated_luar", scope=cond, n=len(paired),
                                    full_minus_none=round(pt, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                    cliffs_delta=round(d, 3), wilcoxon_p_one_sided=round(w_p, 5),
                                    ci_excludes_zero=bool(lo > 0 or hi < 0)))
    return out


def _simple_ci(vals: np.ndarray, rng, iters: int, ci: float) -> tuple[float, float]:
    boot = [float(np.mean(vals[rng.integers(0, len(vals), len(vals))])) for _ in range(iters)]
    return (float(np.percentile(boot, (1 - ci) / 2 * 100)),
            float(np.percentile(boot, (1 + ci) / 2 * 100)))


def _corr_matrix(data: pd.DataFrame, method: str) -> pd.DataFrame:
    return data[PROXIES].corr(method=method)


def _gap(cmat: pd.DataFrame) -> tuple[float, float, float]:
    """Return (mean|convergent|, mean|discriminant|, gap) from a correlation matrix."""
    conv_pairs = ([(FIDELITY[i], FIDELITY[j]) for i in range(3) for j in range(i + 1, 3)]
                  + [(NOVELTY[i], NOVELTY[j]) for i in range(3) for j in range(i + 1, 3)])
    disc_pairs = [(f, n) for f in FIDELITY for n in NOVELTY]
    conv = np.nanmean([abs(cmat.loc[a, b]) for a, b in conv_pairs])
    disc = np.nanmean([abs(cmat.loc[a, b]) for a, b in disc_pairs])
    return float(conv), float(disc), float(conv - disc)


def mtmm(df: pd.DataFrame, cfg: dict) -> dict:
    iters = int(cfg["analysis"]["bootstrap_iters"]); ci = float(cfg["analysis"]["ci"])
    rng = _rng(cfg)
    c3 = df[df.condition == "C3"].copy()

    sp = _corr_matrix(c3, "spearman")
    pe = _corr_matrix(c3, "pearson")
    conv, disc, gap = _gap(sp)

    # cluster bootstrap by prompt
    prompts = c3["prompt_id"].unique()
    boot_gaps = []
    for _ in range(iters):
        pick = rng.choice(prompts, len(prompts), replace=True)
        rows = pd.concat([c3[c3.prompt_id == p] for p in pick], ignore_index=True)
        try:
            _, _, g = _gap(_corr_matrix(rows, "spearman"))
            if np.isfinite(g):
                boot_gaps.append(g)
        except Exception:
            pass
    boot = np.array(boot_gaps)
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100)) if len(boot) else float("nan")
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100)) if len(boot) else float("nan")
    p_gap = float(np.mean(boot <= 0)) if len(boot) else float("nan")  # one-sided gap > 0

    return dict(spearman=sp, pearson=pe, n=len(c3),
                convergent_mean_abs=round(conv, 4), discriminant_mean_abs=round(disc, 4),
                gap=round(gap, 4), gap_ci_lo=round(lo, 4), gap_ci_hi=round(hi, 4),
                gap_p_one_sided=round(p_gap, 5))


def confirmatory_family(df: pd.DataFrame, cfg: dict, kg: dict, mt: dict) -> pd.DataFrame:
    # C1: LUAR JT trend (within none)
    c1_p = next(r["p_one_sided"] for r in kg["contrasts"]
                if r.get("kind", "").startswith("JT_trend") and r["proxy"] == "calibrated_luar")
    c1_excl = next(r["ci_excludes_zero"] for r in kg["contrasts"]
                   if r["proxy"] == "calibrated_luar" and r.get("kind") == "C3-C1(none)")
    # C2: LUAR ablation full>none (pooled)
    abl = next(r for r in kg["ablation"] if r["scope"] == "pooled")
    c2_p, c2_excl = abl["wilcoxon_p_one_sided"], abl["ci_excludes_zero"]
    # C3: distinct_2 decreases C1->C3 (within none); one-sided Wilcoxon 'less'
    paired = _paired_by_prompt(df, "distinct_2", "C1", "C3", "none")
    try:
        c3_p = float(stats.wilcoxon(paired["b"], paired["a"], alternative="less").pvalue)
    except ValueError:
        c3_p = float("nan")
    c3_excl = next(r["ci_excludes_zero"] for r in kg["contrasts"]
                   if r["proxy"] == "distinct_2" and "C3-C1(none)" in r.get("kind", ""))
    # C4: MTMM gap > 0
    c4_p, c4_excl = mt["gap_p_one_sided"], bool(mt["gap_ci_lo"] > 0)

    rows = [
        dict(id="C1", claim="LUAR recovers C1<C2<C3 (none)", p=c1_p, ci_excludes_null=bool(c1_excl)),
        dict(id="C2", claim="style card raises LUAR (full>none)", p=c2_p, ci_excludes_null=bool(c2_excl)),
        dict(id="C3", claim="distinct_n falls C1->C3 (none)", p=c3_p, ci_excludes_null=bool(c3_excl)),
        dict(id="C4", claim="MTMM convergent>discriminant (C3)", p=c4_p, ci_excludes_null=bool(c4_excl)),
    ]
    pvals = [r["p"] for r in rows]
    rejected = holm(pvals, k=len(pvals))
    for r, rej in zip(rows, rejected):
        r["holm_reject"] = bool(rej)
        r["confirmed"] = bool(rej and r["ci_excludes_null"])
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_cfg()
    runs = Path(cfg["paths"]["runs_dir"])
    df = load_merged(cfg)
    print(f"[cv] loaded {len(df)} outputs "
          f"({(df.condition=='C1').sum()} C1 / {(df.condition=='C2').sum()} C2 / {(df.condition=='C3').sum()} C3)")

    kg = known_groups(df, cfg)
    means_df = pd.DataFrame(kg["means"])
    contr_df = pd.DataFrame(kg["contrasts"])
    abl_df = pd.DataFrame(kg["ablation"])
    means_df.to_csv(runs / "known_groups_means.csv", index=False)
    contr_df.to_csv(runs / "known_groups_contrasts.csv", index=False)
    abl_df.to_csv(runs / "known_groups_ablation.csv", index=False)

    print("\n=== KNOWN-GROUPS: condition means (within ablation=none) ===")
    print(means_df[means_df.ablation == "none"].to_string(index=False))
    print("\n=== KNOWN-GROUPS: fidelity gradient + discriminant dissociation (none) ===")
    print(contr_df.to_string(index=False))
    print("\n=== STYLE-CARD ABLATION (LUAR, full - none) ===")
    print(abl_df.to_string(index=False))

    mt = mtmm(df, cfg)
    mt["spearman"].round(3).to_csv(runs / "mtmm_spearman.csv")
    mt["pearson"].round(3).to_csv(runs / "mtmm_pearson.csv")
    print(f"\n=== MTMM within C3 (n={mt['n']}), Spearman ===")
    print(mt["spearman"].round(3).to_string())
    print(f"\nconvergent|r|={mt['convergent_mean_abs']}  discriminant|r|={mt['discriminant_mean_abs']}  "
          f"gap={mt['gap']}  95% CI [{mt['gap_ci_lo']}, {mt['gap_ci_hi']}]  p(gap>0)={mt['gap_p_one_sided']}")

    conf = confirmatory_family(df, cfg, kg, mt)
    conf.to_csv(runs / "confirmatory.csv", index=False)
    print("\n=== CONFIRMATORY FAMILY (Holm k=4) ===")
    print(conf.to_string(index=False))
    print(f"\n[cv] wrote known_groups_*.csv, mtmm_*.csv, confirmatory.csv to {runs}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
