"""
Part 2 selection analysis over runs/candidate_scores.csv. Selector is calibrated_luar,
the held-out certifier is heldout_fidelity, and the novelty axis is distinct_2.
Covers dose-response (H6), the fidelity/novelty frontier (H7), retention policy (E2),
the within/between-cell mechanism, length robustness, and the copy gate summary.
Writes runs/partii_*.csv and figures to ../paper/tex/figures/. Run: python partii_analysis.py
"""
from __future__ import annotations

from itertools import combinations
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
N_GRID = [1, 2, 4, 8]


def load_cfg(p="config.yaml") -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def _figdir() -> Path:
    d = Path("../paper/tex/figures"); d.mkdir(parents=True, exist_ok=True); return d


def _cell_key(df):
    return df["condition"] + "_" + df["ablation"] + "_" + df["prompt_id"]


def _expected_bestofN(cell: pd.DataFrame, n: int, sel="calibrated_luar", cert="heldout_fidelity") -> float:
    vals = cell[[sel, cert]].dropna().to_numpy()
    m = len(vals)
    if m == 0 or n > m:
        return float("nan")
    picks = [vals[list(idx)][np.argmax(vals[list(idx), 0]), 1] for idx in combinations(range(m), n)]
    return float(np.mean(picks))


def dose_response(cfg, df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"]))
    iters = int(cfg["analysis"]["bootstrap_iters"]); ci = float(cfg["analysis"]["ci"])
    cells = {k: g for k, g in df.assign(cell=_cell_key(df)).groupby("cell")}
    prompts = sorted(df["prompt_id"].unique())
    rows = []
    percell = {k: {n: _expected_bestofN(g, n) for n in N_GRID} for k, g in cells.items()}
    cell_prompt = {k: g["prompt_id"].iloc[0] for k, g in cells.items()}
    for n in N_GRID:
        point = float(np.nanmean([percell[k][n] for k in cells]))
        boot = []
        for _ in range(iters):
            pick = rng.choice(prompts, len(prompts), replace=True)
            sel = [percell[k][n] for k in cells if cell_prompt[k] in pick]
            boot.append(float(np.nanmean(sel)))
        lo = float(np.percentile(boot, (1 - ci) / 2 * 100)); hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
        rows.append(dict(N=n, expected_heldout=round(point, 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5)))
    out = pd.DataFrame(rows)
    # gain at N=8 vs N=1 with cluster-bootstrap CI
    boot = []
    for _ in range(iters):
        pick = rng.choice(prompts, len(prompts), replace=True)
        g8 = np.nanmean([percell[k][8] for k in cells if cell_prompt[k] in pick])
        g1 = np.nanmean([percell[k][1] for k in cells if cell_prompt[k] in pick])
        boot.append(float(g8 - g1))
    gain = out[out.N == 8].expected_heldout.iloc[0] - out[out.N == 1].expected_heldout.iloc[0]
    out.attrs["gain_8_vs_1"] = round(float(gain), 5)
    out.attrs["gain_ci"] = (round(float(np.percentile(boot, 2.5)), 5), round(float(np.percentile(boot, 97.5)), 5))
    return out


def frontier(cfg, df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"]))
    iters = int(cfg["analysis"]["bootstrap_iters"])
    rows = []
    for cond in ["C2", "C3"]:
        for fid_axis in ["calibrated_luar", "neg_burrows"]:
            sub = df[df.condition == cond][[fid_axis, "distinct_2", "prompt_id"]].dropna()
            x = sub[fid_axis].to_numpy(); y = sub["distinct_2"].to_numpy()
            rho = float(stats.spearmanr(x, y).correlation)
            prompts = sub["prompt_id"].unique()
            boot = []
            for _ in range(iters):
                pick = rng.choice(prompts, len(prompts), replace=True)
                bs = pd.concat([sub[sub.prompt_id == p] for p in pick])
                if bs[fid_axis].std() > 0 and bs["distinct_2"].std() > 0:
                    boot.append(float(stats.spearmanr(bs[fid_axis], bs["distinct_2"]).correlation))
            lo = float(np.percentile(boot, 2.5)) if boot else float("nan")
            hi = float(np.percentile(boot, 97.5)) if boot else float("nan")
            rows.append(dict(condition=cond, fidelity_axis=fid_axis, n=len(sub),
                             spearman_fid_vs_novelty=round(rho, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             ci_excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def retention(cfg, df: pd.DataFrame) -> pd.DataFrame:
    cells = {k: g for k, g in df.assign(cell=_cell_key(df)).groupby("cell")}
    rnd, greedy, oracle = [], [], []
    for g in cells.values():
        v = g[["calibrated_luar", "heldout_fidelity"]].dropna().to_numpy()
        if len(v) == 0:
            continue
        rnd.append(float(np.mean(v[:, 1])))                      # random pick = mean
        greedy.append(float(v[np.argmax(v[:, 0]), 1]))          # best by LUAR
        oracle.append(float(np.max(v[:, 1])))                   # best possible held-out
    return pd.DataFrame([
        dict(policy="random (mean)", heldout=round(np.mean(rnd), 5)),
        dict(policy="greedy LUAR best-of-8", heldout=round(np.mean(greedy), 5)),
        dict(policy="oracle held-out best", heldout=round(np.mean(oracle), 5)),
    ])


def _spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[m], y[m]).correlation)


def _partial_spearman(x, y, z) -> float:
    """Spearman partial correlation of x,y controlling for z (rank then residualise)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if m.sum() < 4:
        return float("nan")
    rx, ry, rz = (stats.rankdata(v[m]) for v in (x, y, z))
    def resid(a, b):
        b1 = np.vstack([np.ones_like(b), b]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def mechanism(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Decompose selector/certifier agreement into within-cell (what best-of-N
    selects over), between-cell, and pooled rank correlations."""
    d = df.assign(cell=_cell_key(df))
    pooled = _spearman(d["calibrated_luar"], d["heldout_fidelity"])
    within = [_spearman(g["calibrated_luar"], g["heldout_fidelity"])
              for _, g in d.groupby("cell")]
    within = [w for w in within if not np.isnan(w)]
    within_mean = float(np.mean(within)) if within else float("nan")
    cm = d.groupby("cell")[["calibrated_luar", "heldout_fidelity"]].mean()
    between = _spearman(cm["calibrated_luar"], cm["heldout_fidelity"])
    return pd.DataFrame([
        dict(scope="pooled (all candidates)", spearman_luar_vs_heldout=round(pooled, 4)),
        dict(scope="within-cell (mean; what best-of-N selects over)", spearman_luar_vs_heldout=round(within_mean, 4)),
        dict(scope="between-cell (cell means)", spearman_luar_vs_heldout=round(between, 4)),
    ])


def length_robustness(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Length confound + length-controlled frontier. Joins candidate text for token length."""
    runs = Path(cfg["paths"]["runs_dir"])
    cand = pd.read_csv(runs / "candidates.csv")[["candidate_id", "text"]]
    cand["length"] = cand["text"].astype(str).str.split().apply(len)
    d = df.merge(cand[["candidate_id", "length"]], on="candidate_id", how="left")
    rows = []
    for cond in ["C2", "C3"]:
        s = d[d.condition == cond]
        rows.append(dict(
            condition=cond, n=len(s),
            rho_len_luar=round(_spearman(s["length"], s["calibrated_luar"]), 3),
            rho_len_negburrows=round(_spearman(s["length"], s["neg_burrows"]), 3),
            rho_len_distinct2=round(_spearman(s["length"], s["distinct_2"]), 3),
            # fidelity/novelty, raw and length-partialled
            rho_luar_distinct2=round(_spearman(s["calibrated_luar"], s["distinct_2"]), 3),
            partial_luar_distinct2=round(_partial_spearman(s["calibrated_luar"], s["distinct_2"], s["length"]), 3),
            rho_negburrows_distinct2=round(_spearman(s["neg_burrows"], s["distinct_2"]), 3),
            partial_negburrows_distinct2=round(_partial_spearman(s["neg_burrows"], s["distinct_2"], s["length"]), 3),
            rho_luar_negburrows=round(_spearman(s["calibrated_luar"], s["neg_burrows"]), 3),
        ))
    return pd.DataFrame(rows)


def gate_summary(df: pd.DataFrame) -> pd.DataFrame:
    c3 = df[df.condition == "C3"]
    if "copy_flag" not in c3 or len(c3) == 0:
        return pd.DataFrame([dict(group="C3 (no gate cols)", n=len(c3))])
    top = c3.sort_values("calibrated_luar", ascending=False).head(max(1, len(c3) // 10))
    return pd.DataFrame([
        dict(group="all C3 candidates", n=len(c3), flagged=int(c3.copy_flag.sum()),
             mean_copy_rate=round(float(c3.copy_rate.mean()), 5)),
        dict(group="top-10% fidelity winners", n=len(top), flagged=int(top.copy_flag.sum()),
             mean_copy_rate=round(float(top.copy_rate.mean()), 5)),
    ])


def fig_dose(dr: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.errorbar(dr.N, dr.expected_heldout,
                yerr=[dr.expected_heldout - dr.ci_lo, dr.ci_hi - dr.expected_heldout],
                marker="o", capsize=3, color="#1f77b4")
    ax.set_xscale("log", base=2); ax.set_xticks(N_GRID); ax.set_xticklabels(N_GRID)
    ax.set_xlabel("N (candidates; selected on LUAR)")
    ax.set_ylabel("held-out certifier\n(char-$n$-gram TF-IDF)")
    ax.set_title("Dose–response: no transfer to the held-out certifier (null)", pad=12)
    fig.savefig(out / "fig_dose_response.pdf"); plt.close(fig)


def fig_frontier(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for cond, color in [("C2", "#ff7f0e"), ("C3", "#1f77b4")]:
        s = df[df.condition == cond]
        ax.scatter(s.calibrated_luar, s.distinct_2, s=10, alpha=0.45, color=color, label=cond)
    ax.set_xlabel("fidelity (calibrated LUAR)"); ax.set_ylabel("novelty (distinct-2)")
    ax.set_title("Fidelity × novelty candidate cloud (the trade-off)")
    ax.legend(frameon=False)
    fig.savefig(out / "fig_frontier.pdf"); plt.close(fig)


def main() -> int:
    cfg = load_cfg(); runs = Path(cfg["paths"]["runs_dir"]); out = _figdir()
    path = runs / "candidate_scores.csv"
    if not path.exists():
        print(f"[an2] {path} not found; run partii_score.py first."); return 1
    df = pd.read_csv(path)
    print(f"[an2] {len(df)} candidates "
          f"({(df.condition=='C2').sum()} C2 / {(df.condition=='C3').sum()} C3)")

    dr = dose_response(cfg, df); dr.to_csv(runs / "partii_dose_response.csv", index=False)
    print("\n=== H6 DOSE-RESPONSE (held-out certifier vs N) ===")
    print(dr.to_string(index=False))
    print(f"gain N=8 vs N=1: {dr.attrs['gain_8_vs_1']}  95% CI {dr.attrs['gain_ci']}")

    fr = frontier(cfg, df); fr.to_csv(runs / "partii_frontier.csv", index=False)
    print("\n=== H7 FRONTIER trade-off: Spearman(fidelity, novelty) per condition ===")
    print(fr.to_string(index=False))

    rt = retention(cfg, df); rt.to_csv(runs / "partii_retention.csv", index=False)
    print("\n=== E2 RETENTION POLICY (mean held-out) ===")
    print(rt.to_string(index=False))

    me = mechanism(cfg, df); me.to_csv(runs / "partii_mechanism.csv", index=False)
    print("\n=== MECHANISM: selector<->certifier agreement (within vs between cell) ===")
    print(me.to_string(index=False))

    lr = length_robustness(cfg, df); lr.to_csv(runs / "partii_length.csv", index=False)
    print("\n=== LENGTH ROBUSTNESS (confound + length-partialled frontier) ===")
    print(lr.to_string(index=False))

    gs = gate_summary(df); gs.to_csv(runs / "partii_gate.csv", index=False)
    print("\n=== GATE (C3 candidates) ===")
    print(gs.to_string(index=False))

    fig_dose(dr, out); fig_frontier(df, out)
    print(f"\n[an2] wrote partii_*.csv and 2 figures to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
