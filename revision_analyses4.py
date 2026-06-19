"""Selection-validity diagnostics: why best-of-N on calibrated-LUAR doesn't move the
certifier, and whether that's specific to this instrument pair. Covers a
law-of-total-covariance split, a within- vs cross-prompt flip experiment, all-pairs
agreement plus a fidelity transfer matrix, and an oracle-gain CI.

Reads runs/candidate_scores.csv; writes runs/selvalid_*.csv and the figures
fig_between_within.pdf, fig_flip.pdf to ../paper/tex/figures/.
Run:  .venv/bin/python revision_analyses4.py
"""
from __future__ import annotations

from math import lgamma
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

SEL = "calibrated_luar"
CERT = "heldout_fidelity"
METRICS = ["calibrated_luar", "neg_burrows", "heldout_fidelity", "distinct_2", "dual_repaired"]
FIDELITY3 = ["calibrated_luar", "neg_burrows", "heldout_fidelity"]
LABELS = {"calibrated_luar": "LUAR", "neg_burrows": "-Burrows", "heldout_fidelity": "certifier",
          "distinct_2": "distinct-2", "dual_repaired": "dual-div"}


def load_cfg(p="config.yaml") -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def _figdir() -> Path:
    d = Path("../paper/tex/figures"); d.mkdir(parents=True, exist_ok=True); return d


def _cells(df: pd.DataFrame):
    return df.groupby(["condition", "ablation", "prompt_id"], sort=True)


def decomposition(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Exact law-of-total-covariance split (equal cell sizes, population covariances)."""
    x = df[SEL].to_numpy(float); y = df[CERT].to_numpy(float)
    cell_ids = (df["condition"] + "_" + df["ablation"] + "_" + df["prompt_id"]).to_numpy()
    cells = sorted(set(cell_ids))
    mx = np.array([x[cell_ids == c].mean() for c in cells])
    my = np.array([y[cell_ids == c].mean() for c in cells])

    def pcov(a, b):  # population covariance (ddof=0)
        return float(np.mean((a - a.mean()) * (b - b.mean())))

    cov_total = pcov(x, y)
    cov_between = pcov(mx, my)
    cov_within = float(np.mean([pcov(x[cell_ids == c], y[cell_ids == c]) for c in cells]))
    # between-cell variance shares (ICC)
    icc_x = pcov(mx, mx) / pcov(x, x)
    icc_y = pcov(my, my) / pcov(y, y)
    dem_x = np.concatenate([x[cell_ids == c] - x[cell_ids == c].mean() for c in cells])
    dem_y = np.concatenate([y[cell_ids == c] - y[cell_ids == c].mean() for c in cells])
    r_pooled = float(stats.pearsonr(x, y).statistic)
    r_between = float(stats.pearsonr(mx, my).statistic)
    r_within = float(stats.pearsonr(dem_x, dem_y).statistic)
    recon = r_between * np.sqrt(icc_x * icc_y) + r_within * np.sqrt((1 - icc_x) * (1 - icc_y))

    sp_pooled = float(stats.spearmanr(x, y).statistic)
    sp_between = float(stats.spearmanr(mx, my).statistic)
    cell_sp = [float(stats.spearmanr(x[cell_ids == c], y[cell_ids == c]).statistic) for c in cells]
    sp_within_mean = float(np.mean(cell_sp))

    # cluster-bootstrap CI by prompt for the mean within-cell Spearman
    rng = np.random.default_rng(int(cfg["seed"]))
    iters = int(cfg["analysis"]["bootstrap_iters"]); ci = float(cfg["analysis"]["ci"])
    cell_prompt = np.array([c.split("_")[-1] for c in cells])
    prompts = sorted(set(cell_prompt))
    sp_arr = np.array(cell_sp)
    boot = []
    for _ in range(iters):
        pick = rng.choice(prompts, len(prompts), replace=True)
        vals = np.concatenate([sp_arr[cell_prompt == p] for p in pick])
        boot.append(float(np.mean(vals)))
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))

    out = pd.DataFrame([dict(
        cov_total=cov_total, cov_between=cov_between, cov_within_mean=cov_within,
        share_between=cov_between / cov_total, share_within=cov_within / cov_total,
        icc_luar=icc_x, icc_certifier=icc_y,
        r_pearson_pooled=r_pooled, r_pearson_between=r_between, r_pearson_within=r_within,
        r_pearson_reconstructed=recon,
        sp_pooled=sp_pooled, sp_between=sp_between,
        sp_within_mean=sp_within_mean, sp_within_mean_ci_lo=lo, sp_within_mean_ci_hi=hi,
    )]).round(4)
    return out


def _exact_cross_expectation(sel_vals: np.ndarray, out_vals: np.ndarray, n: int,
                             rng: np.random.Generator) -> float:
    """E[outcome of the argmax-selector item] over all size-n subsets of the pool.

    P(item at ascending selector-rank k is the sample max) = C(k-1, n-1) / C(P, n).
    Selector ties (only from bootstrap-duplicated rows) are jittered apart.
    """
    P = len(sel_vals)
    if n > P:
        return float("nan")
    s = sel_vals.astype(float)
    if len(np.unique(s)) < P:                      # duplicates from bootstrap resampling
        s = s + rng.normal(0.0, 1e-12, P)
    order = np.argsort(s)
    o = out_vals[order]
    k = np.arange(1, P + 1, dtype=float)           # ascending rank
    logC = np.full(P, -np.inf)
    valid = k >= n
    kk = k[valid]
    logC[valid] = (np.vectorize(lgamma)(kk) - np.vectorize(lgamma)(np.repeat(n, len(kk)))
                   - np.vectorize(lgamma)(kk - n + 1.0))
    logCPn = lgamma(P + 1.0) - lgamma(n + 1.0) - lgamma(P - n + 1.0)
    # weight_k = C(k-1, n-1) / C(P, n)
    w = np.exp(logC - logCPn)
    return float(np.sum(w * o))


def flip(cfg, df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Within-prompt vs cross-prompt best-of-n inside each condition x ablation stratum."""
    rng = np.random.default_rng(int(cfg["seed"]))
    iters = int(cfg["analysis"]["bootstrap_iters"]); ci = float(cfg["analysis"]["ci"])
    strata = sorted(df.groupby(["condition", "ablation"]).groups.keys())
    prompts = sorted(df["prompt_id"].unique())

    def stratum_stats(sub: pd.DataFrame) -> tuple[float, float, float, float]:
        """(random, within_bo8, cross_bo8, cross_oracle8) for one stratum's candidate pool."""
        sel = sub[SEL].to_numpy(float); cert = sub[CERT].to_numpy(float)
        random_pick = float(cert.mean())
        within = float(np.mean([
            g[CERT].to_numpy(float)[np.argmax(g[SEL].to_numpy(float))]
            for _, g in sub.groupby("prompt_id")
        ]))
        cross = _exact_cross_expectation(sel, cert, n, rng)
        cross_oracle = _exact_cross_expectation(cert, cert, n, rng)
        return random_pick, within, cross, cross_oracle

    points = {s: stratum_stats(df[(df.condition == s[0]) & (df.ablation == s[1])]) for s in strata}
    pooled_point = np.mean(np.array(list(points.values())), axis=0)

    # cluster bootstrap by prompt
    by = {(s, p): df[(df.condition == s[0]) & (df.ablation == s[1]) & (df.prompt_id == p)]
          for s in strata for p in prompts}
    boot = np.empty((iters, 4))
    for b in range(iters):
        pick = rng.choice(prompts, len(prompts), replace=True)
        vals = []
        for s in strata:
            sub = pd.concat([by[(s, p)] for p in pick], ignore_index=True)
            vals.append(stratum_stats(sub))
        boot[b] = np.mean(np.array(vals), axis=0)

    def ci_of(col_fn):
        v = col_fn(boot)
        return (float(np.percentile(v, (1 - ci) / 2 * 100)),
                float(np.percentile(v, (1 + ci) / 2 * 100)))

    rows = []
    names = ["random", "within_bo8", "cross_bo8", "cross_oracle8"]
    for i, name in enumerate(names):
        lo, hi = ci_of(lambda B, i=i: B[:, i])
        rows.append(dict(policy=name, mean=round(float(pooled_point[i]), 5),
                         ci_lo=round(lo, 5), ci_hi=round(hi, 5)))
    for i, j, name in [(1, 0, "gain_within_vs_random"), (2, 0, "gain_cross_vs_random"),
                       (2, 1, "gain_cross_vs_within")]:
        lo, hi = ci_of(lambda B, i=i, j=j: B[:, i] - B[:, j])
        rows.append(dict(policy=name, mean=round(float(pooled_point[i] - pooled_point[j]), 5),
                         ci_lo=round(lo, 5), ci_hi=round(hi, 5)))
    return pd.DataFrame(rows)


def matrix(cfg, df: pd.DataFrame, n: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_ids = (df["condition"] + "_" + df["ablation"] + "_" + df["prompt_id"]).to_numpy()
    cells = sorted(set(cell_ids))
    data = {m: df[m].to_numpy(float) for m in METRICS}
    means = {m: np.array([data[m][cell_ids == c].mean() for c in cells]) for m in METRICS}

    rows = []
    for i, a in enumerate(METRICS):
        for b_ in METRICS[i + 1:]:
            pooled = float(stats.spearmanr(data[a], data[b_]).statistic)
            between = float(stats.spearmanr(means[a], means[b_]).statistic)
            within = float(np.mean([
                stats.spearmanr(data[a][cell_ids == c], data[b_][cell_ids == c]).statistic
                for c in cells]))
            rows.append(dict(metric_a=LABELS[a], metric_b=LABELS[b_],
                             sp_pooled=round(pooled, 3), sp_between=round(between, 3),
                             sp_within_mean=round(within, 3)))
    agree = pd.DataFrame(rows)

    # best-of-8 transfer for every ordered fidelity selector -> certifier pair
    rng = np.random.default_rng(int(cfg["seed"]))
    iters = int(cfg["analysis"]["bootstrap_iters"]); ci = float(cfg["analysis"]["ci"])
    cell_prompt = np.array([c.split("_")[-1] for c in cells])
    prompts = sorted(set(cell_prompt))
    trows = []
    for s in FIDELITY3:
        for c_ in FIDELITY3:
            if s == c_:
                continue
            per_cell_gain = np.array([
                data[c_][cell_ids == k][np.argmax(data[s][cell_ids == k])]
                - data[c_][cell_ids == k].mean() for k in cells])
            per_cell_oracle = np.array([
                data[c_][cell_ids == k].max() - data[c_][cell_ids == k].mean() for k in cells])
            point = float(per_cell_gain.mean()); oracle = float(per_cell_oracle.mean())
            boot = []
            for _ in range(iters):
                pick = rng.choice(prompts, len(prompts), replace=True)
                vals = np.concatenate([per_cell_gain[cell_prompt == p] for p in pick])
                boot.append(float(np.mean(vals)))
            lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
            hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
            trows.append(dict(selector=LABELS[s], certifier=LABELS[c_],
                              bo8_gain=round(point, 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5),
                              ci_excludes_zero=bool(lo > 0 or hi < 0),
                              oracle_gain=round(oracle, 5),
                              capture_share=round(point / oracle, 3) if oracle else float("nan")))
    transfer = pd.DataFrame(trows)
    return agree, transfer


def oracle_ci(cfg, df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"]))
    iters = int(cfg["analysis"]["bootstrap_iters"]); ci = float(cfg["analysis"]["ci"])
    cell_ids = (df["condition"] + "_" + df["ablation"] + "_" + df["prompt_id"]).to_numpy()
    cells = sorted(set(cell_ids))
    cert = df[CERT].to_numpy(float)
    per_cell_mean = np.array([cert[cell_ids == c].mean() for c in cells])
    per_cell_max = np.array([cert[cell_ids == c].max() for c in cells])
    cell_prompt = np.array([c.split("_")[-1] for c in cells])
    prompts = sorted(set(cell_prompt))
    point_gain = float((per_cell_max - per_cell_mean).mean())
    point_ratio = float(per_cell_max.mean() / per_cell_mean.mean())
    boot_g, boot_r = [], []
    for _ in range(iters):
        pick = rng.choice(prompts, len(prompts), replace=True)
        mx = np.concatenate([per_cell_max[cell_prompt == p] for p in pick])
        mn = np.concatenate([per_cell_mean[cell_prompt == p] for p in pick])
        boot_g.append(float((mx - mn).mean())); boot_r.append(float(mx.mean() / mn.mean()))
    q = lambda v, p: float(np.percentile(v, p))
    a = (1 - ci) / 2 * 100
    return pd.DataFrame([dict(
        oracle_gain=round(point_gain, 5), gain_ci_lo=round(q(boot_g, a), 5),
        gain_ci_hi=round(q(boot_g, 100 - a), 5),
        oracle_ratio=round(point_ratio, 4), ratio_ci_lo=round(q(boot_r, a), 4),
        ratio_ci_hi=round(q(boot_r, 100 - a), 4))])


def fig_between_within(df: pd.DataFrame, dec: pd.DataFrame, out: Path):
    cell_ids = (df["condition"] + "_" + df["ablation"] + "_" + df["prompt_id"]).to_numpy()
    cells = sorted(set(cell_ids))
    x = df[SEL].to_numpy(float); y = df[CERT].to_numpy(float)
    mx = np.array([x[cell_ids == c].mean() for c in cells])
    my = np.array([y[cell_ids == c].mean() for c in cells])
    dx = np.concatenate([x[cell_ids == c] - x[cell_ids == c].mean() for c in cells])
    dy = np.concatenate([y[cell_ids == c] - y[cell_ids == c].mean() for c in cells])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].scatter(mx, my, s=22, alpha=0.7, color="#1f77b4")
    axes[0].set_title(f"Across prompts: the rulers agree\n(64 cell means, "
                      f"$\\rho$ = {dec.sp_between.iloc[0]:.2f})", fontsize=11)
    axes[0].set_xlabel("selector (calibrated LUAR), cell mean")
    axes[0].set_ylabel("held-out certifier, cell mean")
    axes[1].scatter(dx, dy, s=10, alpha=0.35, color="#d62728")
    axes[1].axhline(0, color="grey", lw=0.6, ls=":"); axes[1].axvline(0, color="grey", lw=0.6, ls=":")
    axes[1].set_title(f"Within a prompt: they don't\n(512 deviations, mean "
                      f"$\\rho$ = {dec.sp_within_mean.iloc[0]:.3f})", fontsize=11)
    axes[1].set_xlabel("selector deviation within cell")
    axes[1].set_ylabel("certifier deviation within cell")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out / "fig_between_within.pdf"); plt.close(fig)


def fig_flip(fl: pd.DataFrame, out: Path):
    pol = fl.set_index("policy")
    names = ["random", "within_bo8", "cross_bo8"]
    labels = ["random pick", "best-of-8\nwithin prompt", "best-of-8\nacross prompts"]
    means = [pol.loc[n, "mean"] for n in names]
    los = [pol.loc[n, "ci_lo"] for n in names]; his = [pol.loc[n, "ci_hi"] for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]
    ax.bar(labels, means, yerr=[np.subtract(means, los), np.subtract(his, means)],
           capsize=4, color=colors, width=0.6)
    orc = pol.loc["cross_oracle8", "mean"]
    ax.axhline(orc, color="#d62728", lw=1.2, ls="--")
    ax.text(0.02, orc, f"cross-prompt certifier oracle ({orc:.4f})",
            transform=ax.get_yaxis_transform(), va="bottom", fontsize=9, color="#d62728")
    lo_all = min(los + [orc]) * 0.985
    ax.set_ylim(lo_all, max(his + [orc]) * 1.01)
    ax.set_ylabel("held-out certifier of the pick")
    ax.set_title("Selection works only where the rulers agree", pad=10)
    fig.savefig(out / "fig_flip.pdf"); plt.close(fig)


def main() -> int:
    cfg = load_cfg(); runs = Path(cfg["paths"]["runs_dir"]); out = _figdir()
    df = pd.read_csv(runs / "candidate_scores.csv")
    print(f"[an4] {len(df)} candidates, {df.groupby(['condition','ablation','prompt_id']).ngroups} cells")

    print("\n=== H. VARIANCE DECOMPOSITION (selector <-> certifier) ===")
    dec = decomposition(cfg, df); dec.to_csv(runs / "selvalid_decomposition.csv", index=False)
    print(dec.T.to_string(header=False))

    print("\n=== I. FLIP EXPERIMENT (within-prompt vs cross-prompt best-of-8) ===")
    fl = flip(cfg, df); fl.to_csv(runs / "selvalid_flip.csv", index=False)
    print(fl.to_string(index=False))

    print("\n=== J. ALL-PAIRS AGREEMENT (pooled / between / mean within-cell Spearman) ===")
    agree, transfer = matrix(cfg, df)
    agree.to_csv(runs / "selvalid_agreement.csv", index=False)
    transfer.to_csv(runs / "selvalid_transfer.csv", index=False)
    print(agree.to_string(index=False))
    print("\n=== J. BEST-OF-8 TRANSFER (ordered fidelity selector -> certifier) ===")
    print(transfer.to_string(index=False))

    print("\n=== K. ORACLE GAIN CI ===")
    oc = oracle_ci(cfg, df); oc.to_csv(runs / "selvalid_oracle_ci.csv", index=False)
    print(oc.to_string(index=False))

    fig_between_within(df, dec, out)
    fig_flip(fl, out)
    print("\n[an4] wrote selvalid_*.csv and fig_between_within.pdf, fig_flip.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
