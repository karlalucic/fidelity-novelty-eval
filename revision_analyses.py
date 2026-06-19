"""
Extra analyses for the peer-review revision, run over the existing run CSVs and
generation logs (pandas/numpy/scipy, CPU-only):

  A. retrieval_conditioning   — is the fidelity/novelty dissociation a RAG anchoring artifact?
  B. sensitivity_power        — bootstrap achieved power at the observed effects
  C. within_cell_distribution — per-cell Spearman(LUAR, certifier) across the 64 Part-2 cells
  D. burrows_shorttext_reliability — bootstrap SE of Burrows's Delta on short Senra windows

Run:  .venv/bin/python revision_analyses.py
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from construct_validity import jonckheere  # tie-corrected JT


def load_cfg(p="config.yaml") -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


_WORD = re.compile(r"[a-z0-9']+")


def _toks(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(toks: list[str], n: int) -> set:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)} if len(toks) >= n else set()


def _spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[m], y[m]).correlation)


def _partial_spearman(x, y, z) -> float:
    """Spearman partial corr of x,y controlling for z (rank -> residualise -> Pearson)."""
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


def _cluster_boot_spearman(df, xcol, ycol, groupcol, iters, rng):
    """Cluster-bootstrap (resample groups) CI for pooled Spearman(x,y)."""
    groups = df[groupcol].unique()
    boot = []
    by = {g: df[df[groupcol] == g] for g in groups}
    for _ in range(iters):
        pick = rng.choice(groups, len(groups), replace=True)
        bs = pd.concat([by[g] for g in pick])
        r = _spearman(bs[xcol], bs[ycol])
        if not np.isnan(r):
            boot.append(r)
    lo = float(np.percentile(boot, 2.5)) if boot else float("nan")
    hi = float(np.percentile(boot, 97.5)) if boot else float("nan")
    return lo, hi


# A. retrieval-conditioning confound

def _retrieved_by_prompt(cfg) -> dict:
    """prompt_id -> retrieved passage text (from C3 base-output generation logs)."""
    runs = Path(cfg["paths"]["runs_dir"])
    pat = re.compile(r"=== RETRIEVED[^=]*PASSAGES ===(.*?)=== END RETRIEVED PASSAGES ===", re.S)
    out = {}
    for line in (runs / "generation_log.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["condition"] != "C3":
            continue
        m = pat.search(d["prompt"])
        if m:
            out[d["prompt_id"]] = m.group(1).strip()
    return out


def _overlap_features(text: str, retrieved: str) -> dict:
    """How much of `text` echoes `retrieved` (higher = more lexical anchoring)."""
    ot = _toks(text); rt = _toks(retrieved)
    r2 = _ngrams(rt, 2); r3 = _ngrams(rt, 3)
    o2 = _ngrams(ot, 2); o3 = _ngrams(ot, 3)
    o1 = set(ot); r1 = set(rt)
    return dict(
        bigram_echo=(len(o2 & r2) / len(o2)) if o2 else np.nan,
        trigram_echo=(len(o3 & r3) / len(o3)) if o3 else np.nan,
        token_jaccard=(len(o1 & r1) / len(o1 | r1)) if (o1 | r1) else np.nan,
    )


def retrieval_conditioning(cfg, rng) -> dict:
    runs = Path(cfg["paths"]["runs_dir"])
    iters = int(cfg["analysis"]["bootstrap_iters"])
    retr = _retrieved_by_prompt(cfg)

    # candidate level (n=256 C3 candidates)
    cand = pd.read_csv(runs / "candidates.csv")
    cs = pd.read_csv(runs / "candidate_scores.csv")
    c3 = cand[cand.condition == "C3"].merge(
        cs[["candidate_id", "calibrated_luar", "distinct_2", "neg_burrows"]], on="candidate_id")
    feats = c3.apply(lambda r: _overlap_features(str(r["text"]), retr.get(r["prompt_id"], "")), axis=1)
    c3 = pd.concat([c3.reset_index(drop=True), pd.DataFrame(list(feats))], axis=1)
    c3["cell"] = c3["condition"] + "_" + c3["ablation"] + "_" + c3["prompt_id"]

    res = {"candidate": {}, "output": {}}
    for echo in ["bigram_echo", "trigram_echo", "token_jaccard"]:
        r_luar = _spearman(c3[echo], c3["calibrated_luar"])
        r_dist = _spearman(c3[echo], c3["distinct_2"])
        lo_l, hi_l = _cluster_boot_spearman(c3, echo, "calibrated_luar", "prompt_id", iters, rng)
        lo_d, hi_d = _cluster_boot_spearman(c3, echo, "distinct_2", "prompt_id", iters, rng)
        # within-cell mean correlation
        wl = np.nanmean([_spearman(g[echo], g["calibrated_luar"]) for _, g in c3.groupby("cell")])
        wd = np.nanmean([_spearman(g[echo], g["distinct_2"]) for _, g in c3.groupby("cell")])
        res["candidate"][echo] = dict(
            rho_echo_luar=round(r_luar, 3), ci_echo_luar=(round(lo_l, 3), round(hi_l, 3)),
            rho_echo_distinct2=round(r_dist, 3), ci_echo_distinct2=(round(lo_d, 3), round(hi_d, 3)),
            within_cell_echo_luar=round(float(wl), 3), within_cell_echo_distinct2=round(float(wd), 3),
        )

    # does controlling echo collapse the luar<->distinct2 trade-off?
    raw = _spearman(c3["calibrated_luar"], c3["distinct_2"])
    part_big = _partial_spearman(c3["calibrated_luar"], c3["distinct_2"], c3["bigram_echo"])
    part_tri = _partial_spearman(c3["calibrated_luar"], c3["distinct_2"], c3["trigram_echo"])
    res["partial"] = dict(raw_luar_distinct2=round(raw, 3),
                          partial_given_bigram_echo=round(part_big, 3),
                          partial_given_trigram_echo=round(part_tri, 3))

    # output level (within-none C3 base outputs)
    outs = pd.read_csv(runs / "outputs.csv")
    fid = pd.read_csv(runs / "fidelity_scores.csv")[["output_id", "calibrated_luar"]]
    nov = pd.read_csv(runs / "novelty_scores.csv")[["output_id", "distinct_2"]]
    o3 = outs[(outs.condition == "C3")].merge(fid, on="output_id").merge(nov, on="output_id")
    of = o3.apply(lambda r: _overlap_features(str(r["text"]), retr.get(r["prompt_id"], "")), axis=1)
    o3 = pd.concat([o3.reset_index(drop=True), pd.DataFrame(list(of))], axis=1)
    res["output"] = dict(
        n=len(o3),
        rho_bigramecho_luar=round(_spearman(o3["bigram_echo"], o3["calibrated_luar"]), 3),
        rho_bigramecho_distinct2=round(_spearman(o3["bigram_echo"], o3["distinct_2"]), 3),
        mean_bigram_echo=round(float(o3["bigram_echo"].mean()), 3),
        mean_trigram_echo=round(float(o3["trigram_echo"].mean()), 4),
    )
    c3.to_csv(runs / "rev_retrieval_overlap_candidates.csv", index=False)
    return res


# B. bootstrap achieved power at the observed effect

def _merged_none(cfg) -> pd.DataFrame:
    runs = Path(cfg["paths"]["runs_dir"])
    outs = pd.read_csv(runs / "outputs.csv")[["output_id", "condition", "ablation", "prompt_id"]]
    fid = pd.read_csv(runs / "fidelity_scores.csv")[["output_id", "calibrated_luar"]]
    nov = pd.read_csv(runs / "novelty_scores.csv")[["output_id", "distinct_2"]]
    df = outs.merge(fid, on="output_id").merge(nov, on="output_id")
    return df


def sensitivity_power(cfg, rng, B=2000) -> pd.DataFrame:
    df = _merged_none(cfg)
    none = df[df.ablation == "none"]
    prompts = sorted(none.prompt_id.unique())
    rows = []

    # C1: JT trend on calibrated_luar across C1<C2<C3 (within none), resample prompts
    def jt_p(sample_prompts):
        groups = []
        for c in ["C1", "C2", "C3"]:
            v = none[(none.condition == c) & (none.prompt_id.isin(sample_prompts))]["calibrated_luar"]
            vals = []
            for p in sample_prompts:
                vv = none[(none.condition == c) & (none.prompt_id == p)]["calibrated_luar"].values
                vals.extend(vv.tolist())
            groups.append(np.array(vals))
        _, _, p = jonckheere(groups)
        return p
    hits = 0
    for _ in range(B):
        pick = list(rng.choice(prompts, len(prompts), replace=True))
        if jt_p(pick) < 0.05:
            hits += 1
    rows.append(dict(test="C1 JT gradient (LUAR, none)", observed_p=0.00773, achieved_power=round(hits / B, 3)))

    # C3: dissociation distinct_2 C1>C3 (none), paired by prompt, one-sided Wilcoxon
    a = none[none.condition == "C1"].set_index("prompt_id")["distinct_2"]
    b = none[none.condition == "C3"].set_index("prompt_id")["distinct_2"]
    paired = pd.DataFrame({"c1": a, "c3": b}).dropna()
    hits = 0
    for _ in range(B):
        idx = rng.integers(0, len(paired), len(paired))
        s = paired.iloc[idx]
        try:
            if stats.wilcoxon(s["c3"], s["c1"], alternative="less").pvalue < 0.05:
                hits += 1
        except ValueError:
            pass
    rows.append(dict(test="C3 dissociation (distinct-2 down, none)", observed_p=7.6e-05,
                     achieved_power=round(hits / B, 3)))

    # C2: ablation full>none on LUAR (pooled C2+C3), paired by (cond,prompt)
    runs = Path(cfg["paths"]["runs_dir"])
    fid = pd.read_csv(runs / "fidelity_scores.csv")[["output_id", "calibrated_luar"]]
    outs = pd.read_csv(runs / "outputs.csv")[["output_id", "condition", "ablation", "prompt_id"]]
    m = outs.merge(fid, on="output_id")
    sub = m[m.condition.isin(["C2", "C3"])]
    full = sub[sub.ablation == "full"].set_index(["condition", "prompt_id"])["calibrated_luar"]
    nonev = sub[sub.ablation == "none"].set_index(["condition", "prompt_id"])["calibrated_luar"]
    pair = pd.DataFrame({"full": full, "none": nonev}).dropna()
    hits = 0
    for _ in range(B):
        idx = rng.integers(0, len(pair), len(pair))
        s = pair.iloc[idx]
        try:
            if stats.wilcoxon(s["full"], s["none"], alternative="greater").pvalue < 0.05:
                hits += 1
        except ValueError:
            pass
    rows.append(dict(test="C2 style-card ablation (LUAR full>none, pooled)", observed_p=1e-5,
                     achieved_power=round(hits / B, 3)))
    return pd.DataFrame(rows)


# C. within-cell Spearman distribution (LUAR vs certifier), 64 Part-2 cells

def within_cell_distribution(cfg) -> tuple[pd.DataFrame, np.ndarray]:
    runs = Path(cfg["paths"]["runs_dir"])
    df = pd.read_csv(runs / "candidate_scores.csv")
    df["cell"] = df["condition"] + "_" + df["ablation"] + "_" + df["prompt_id"]
    vals = []
    for _, g in df.groupby("cell"):
        r = _spearman(g["calibrated_luar"], g["heldout_fidelity"])
        if not np.isnan(r):
            vals.append(r)
    vals = np.array(vals)
    summary = pd.DataFrame([dict(
        n_cells=len(vals), mean=round(float(vals.mean()), 3), median=round(float(np.median(vals)), 3),
        sd=round(float(vals.std(ddof=1)), 3), min=round(float(vals.min()), 3), max=round(float(vals.max()), 3),
        q25=round(float(np.percentile(vals, 25)), 3), q75=round(float(np.percentile(vals, 75)), 3),
        frac_within_pm02=round(float(np.mean(np.abs(vals) <= 0.2)), 3),
        frac_negative=round(float(np.mean(vals < 0)), 3),
    )])
    return summary, vals


def fig_within_cell(vals: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(vals, bins=np.arange(-1.0, 1.01, 0.2), color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.axvline(float(vals.mean()), color="#d62728", lw=2, label=f"mean = {vals.mean():.3f}")
    ax.axvline(0, color="#555", lw=1, ls=":")
    ax.set_xlabel(r"within-cell Spearman($\rho$): selector (LUAR) vs certifier")
    ax.set_ylabel("number of prompt-cells")
    ax.set_title("Selector–certifier agreement is absent where selection happens", pad=8)
    ax.legend(frameon=False)
    out = Path("../paper/tex/figures"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_within_cell.pdf"); plt.close(fig)


# D. Burrows short-text reliability

def burrows_shorttext_reliability(cfg, rng, n_draws=200, words=370) -> pd.DataFrame:
    """Bootstrap SE of Burrows Delta-to-target on random ~`words`-word real-Senra windows."""
    try:
        from corpus import load_pool, build_guest_floor
        import fidelity as fid
    except Exception as e:
        return pd.DataFrame([dict(note=f"corpus load failed: {e}")])
    target = load_pool(cfg, "target")
    floor = build_guest_floor(cfg) + load_pool(cfg, "floor_solo")
    if not target or not floor:
        return pd.DataFrame([dict(note="empty target/floor")])
    big = _toks(" \n ".join(target))
    # raw concatenation for window slicing, preserving casing/punct for the burrows tokeniser
    raw = " ".join(target)
    raw_toks = raw.split()
    deltas = []
    for _ in range(n_draws):
        if len(raw_toks) <= words:
            seg = raw
        else:
            start = int(rng.integers(0, len(raw_toks) - words))
            seg = " ".join(raw_toks[start:start + words])
        d = fid.burrows_delta(seg, target, floor, cfg)
        if not np.isnan(d):
            deltas.append(d)
    deltas = np.array(deltas)
    return pd.DataFrame([dict(
        words=words, n_draws=len(deltas), mean_delta=round(float(deltas.mean()), 3),
        sd_delta=round(float(deltas.std(ddof=1)), 3),
        cv=round(float(deltas.std(ddof=1) / abs(deltas.mean())), 3),
        bootstrap_se=round(float(deltas.std(ddof=1) / np.sqrt(len(deltas))), 4),
    )])


def main() -> int:
    cfg = load_cfg()
    runs = Path(cfg["paths"]["runs_dir"])
    rng = np.random.default_rng(int(cfg["seed"]))

    print("=" * 78)
    print("A. RETRIEVAL-CONDITIONING (is the dissociation a RAG anchoring artifact?)")
    print("=" * 78)
    rc = retrieval_conditioning(cfg, rng)
    print("\n[candidate-level, n=256 C3 candidates] overlap-with-retrieved vs proxies:")
    for echo, d in rc["candidate"].items():
        print(f"  {echo:14s}: rho(echo,LUAR)={d['rho_echo_luar']:+.3f} {d['ci_echo_luar']}  "
              f"rho(echo,distinct2)={d['rho_echo_distinct2']:+.3f} {d['ci_echo_distinct2']}  "
              f"| within-cell: LUAR={d['within_cell_echo_luar']:+.3f} dist2={d['within_cell_echo_distinct2']:+.3f}")
    print(f"\n[decisive partial] LUAR<->distinct2 raw={rc['partial']['raw_luar_distinct2']:+.3f}  "
          f"| controlling bigram-echo={rc['partial']['partial_given_bigram_echo']:+.3f}  "
          f"| controlling trigram-echo={rc['partial']['partial_given_trigram_echo']:+.3f}")
    print(f"\n[output-level, n={rc['output']['n']} C3 base outputs] "
          f"rho(echo,LUAR)={rc['output']['rho_bigramecho_luar']:+.3f}  "
          f"rho(echo,distinct2)={rc['output']['rho_bigramecho_distinct2']:+.3f}  "
          f"mean bigram-echo={rc['output']['mean_bigram_echo']}  mean trigram-echo={rc['output']['mean_trigram_echo']}")
    pd.DataFrame(rc["candidate"]).T.to_csv(runs / "rev_retrieval_conditioning.csv")

    print("\n" + "=" * 78)
    print("B. BOOTSTRAP ACHIEVED POWER (at observed effect; resample prompts)")
    print("=" * 78)
    pw = sensitivity_power(cfg, rng)
    print(pw.to_string(index=False))
    pw.to_csv(runs / "rev_power.csv", index=False)

    print("\n" + "=" * 78)
    print("C. WITHIN-CELL SELECTOR<->CERTIFIER DISTRIBUTION (64 Part-2 cells)")
    print("=" * 78)
    summ, vals = within_cell_distribution(cfg)
    print(summ.to_string(index=False))
    summ.to_csv(runs / "rev_within_cell.csv", index=False)
    fig_within_cell(vals)
    print(f"[fig] wrote ../paper/tex/figures/fig_within_cell.pdf  ({len(vals)} cells)")

    print("\n" + "=" * 78)
    print("D. BURROWS SHORT-TEXT RELIABILITY (~370-word real-Senra windows)")
    print("=" * 78)
    br = burrows_shorttext_reliability(cfg, rng)
    print(br.to_string(index=False))
    br.to_csv(runs / "rev_burrows_reliability.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
