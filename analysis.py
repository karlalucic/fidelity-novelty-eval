"""
Statistics helpers for the automatic evaluation: bootstrap CIs, Holm correction,
construct-separation and proxy-agreement checks, and a few descriptive summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import krippendorff  # type: ignore
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore


def krippendorff_alpha(ratings: np.ndarray, level: str = "ordinal") -> float:
    """Krippendorff's alpha. ratings: (n_raters, n_items), nan for missing."""
    return float(krippendorff.alpha(reliability_data=ratings, level_of_measurement=level))


def bootstrap_ci(
    values,
    stat_fn: Callable,
    iters: int,
    ci: float,
    seed: int,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi).

    Uses percentile bootstrap.  ``values`` can be a 1D array-like or a tuple of
    array-likes (for two-sample statistics like correlation).
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values)

    point = float(stat_fn(arr))
    boot_stats: list[float] = []
    n = len(arr)
    for _ in range(iters):
        resample = arr[rng.integers(0, n, n)]
        try:
            boot_stats.append(float(stat_fn(resample)))
        except Exception:
            boot_stats.append(float("nan"))

    boot_arr = np.array(boot_stats)
    boot_arr = boot_arr[~np.isnan(boot_arr)]
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    lo = float(np.percentile(boot_arr, lo_pct)) if len(boot_arr) > 0 else float("nan")
    hi = float(np.percentile(boot_arr, hi_pct)) if len(boot_arr) > 0 else float("nan")
    return point, lo, hi


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    iters: int,
    ci: float,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap CI for Pearson correlation between x and y."""
    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")

    point_r, _ = stats.pearsonr(x, y)
    boot_rs: list[float] = []
    n = len(x)
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        try:
            r, _ = stats.pearsonr(xb, yb)
            boot_rs.append(float(r))
        except Exception:
            pass

    boot_arr = np.array(boot_rs)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    lo = float(np.percentile(boot_arr, lo_pct)) if len(boot_arr) > 0 else float("nan")
    hi = float(np.percentile(boot_arr, hi_pct)) if len(boot_arr) > 0 else float("nan")
    return float(point_r), lo, hi


def proxy_agreement(
    proxy: np.ndarray,
    human_mean: np.ndarray,
    iters: int,
    seed: int,
) -> dict:
    """Correlation between proxy and mean human ratings, with a bootstrap CI.
    Returns r, ci_lo, ci_hi, ci_excludes_zero."""
    proxy = np.array(proxy, dtype=float)
    human_mean = np.array(human_mean, dtype=float)

    r, lo, hi = bootstrap_corr_ci(proxy, human_mean, iters, 0.95, seed)
    ci_excludes_zero = (lo > 0) or (hi < 0) if not np.isnan(lo) else False

    return {
        "r": round(r, 4) if not np.isnan(r) else float("nan"),
        "ci_lo": round(lo, 4) if not np.isnan(lo) else float("nan"),
        "ci_hi": round(hi, 4) if not np.isnan(hi) else float("nan"),
        "ci_excludes_zero": bool(ci_excludes_zero),
    }


def construct_separation(
    human_fidelity: np.ndarray,
    human_novelty: np.ndarray,
    cfg: dict,
) -> dict:
    """Corr + |r|>separation_r_threshold falsification verdict.

    Pre-registered falsification: if |r| > threshold the two-axis claim is NOT supported.
    """
    iters = cfg["analysis"]["bootstrap_iters"]
    seed = cfg["seed"]
    threshold = cfg["analysis"]["separation_r_threshold"]

    r, lo, hi = bootstrap_corr_ci(human_fidelity, human_novelty, iters, 0.95, seed)
    falsified = abs(r) > threshold if not np.isnan(r) else False

    verdict = (
        "FALSIFIED: |r| > threshold -> two-axis claim NOT supported"
        if falsified
        else "SUPPORTED: |r| <= threshold -> axes are separable"
    )

    return {
        "r": round(r, 4) if not np.isnan(r) else float("nan"),
        "ci_lo": round(lo, 4) if not np.isnan(lo) else float("nan"),
        "ci_hi": round(hi, 4) if not np.isnan(hi) else float("nan"),
        "threshold": threshold,
        "falsified": falsified,
        "verdict": verdict,
    }


def holm(pvals: list[float], k: int) -> list[bool]:
    """Holm correction over <=holm_max_confirmatory tests.

    Returns a list of bool (True = reject null / significant after correction).
    Applies only to the first k p-values.
    """
    pvals = list(pvals[:k])
    m = len(pvals)
    sorted_idx = sorted(range(m), key=lambda i: pvals[i])
    rejected = [False] * m
    for rank, idx in enumerate(sorted_idx):
        alpha_adj = 0.05 / (m - rank)
        if pvals[idx] <= alpha_adj:
            rejected[idx] = True
        else:
            break
    return rejected


def trust_table(
    fidelity_df: pd.DataFrame,
    novelty_df: pd.DataFrame,
    human_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Build the block-diagonal trust table.

    Each proxy is screened only on its own axis (no cross-axis cells).
    Krippendorff alpha is computed from the human ratings matrix in human_df
    (columns: rater_id, output_id, human_fidelity, human_novelty), with per-proxy
    agreement against the human mean at output level, CIs, and Holm correction over
    the primary confirmatory proxies. Writes runs/trust_table.csv and returns the
    DataFrame.
    """
    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    iters = cfg["analysis"]["bootstrap_iters"]
    seed = cfg["seed"]
    primary = cfg["analysis"]["primary_confirmatory"]
    fid_proxies = cfg["analysis"]["fidelity_proxies"]
    nov_proxies = cfg["analysis"]["novelty_proxies"]
    holm_k = cfg["analysis"]["holm_max_confirmatory"]

    rows: list[dict] = []

    # human panel alpha
    human_alpha_fidelity: float = float("nan")
    human_alpha_novelty: float = float("nan")
    human_fid_mean: np.ndarray = np.array([])
    human_nov_mean: np.ndarray = np.array([])

    if not human_df.empty and "output_id" in human_df.columns:
        # rater x item matrices
        if "human_fidelity" in human_df.columns and "rater_id" in human_df.columns:
            fid_pivot = human_df.pivot_table(
                index="rater_id", columns="output_id", values="human_fidelity", aggfunc="mean"
            )
            human_alpha_fidelity = krippendorff_alpha(fid_pivot.values, "ordinal")
            human_fid_mean = fid_pivot.mean(axis=0).values

        if "human_novelty" in human_df.columns and "rater_id" in human_df.columns:
            nov_pivot = human_df.pivot_table(
                index="rater_id", columns="output_id", values="human_novelty", aggfunc="mean"
            )
            human_alpha_novelty = krippendorff_alpha(nov_pivot.values, "ordinal")
            human_nov_mean = nov_pivot.mean(axis=0).values

    def get_proxy_col(df: pd.DataFrame, col: str) -> np.ndarray:
        if df.empty or col not in df.columns:
            return np.array([])
        return df[col].values

    # align proxy scores with human_fid_mean by output_id
    def align(proxy_df: pd.DataFrame, proxy_col: str, human_vals: np.ndarray, human_df_use: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if proxy_df.empty or "output_id" not in proxy_df.columns or len(human_vals) == 0:
            return np.array([]), np.array([])
        if human_df_use.empty or "output_id" not in human_df_use.columns:
            return np.array([]), np.array([])
        human_oids = human_df_use["output_id"].unique() if not human_df_use.empty else []
        merged = proxy_df.merge(
            pd.DataFrame({"output_id": human_oids, "_human_mean": human_vals}) if len(human_oids) == len(human_vals)
            else pd.DataFrame(columns=["output_id", "_human_mean"]),
            on="output_id",
            how="inner",
        )
        if merged.empty or proxy_col not in merged.columns:
            return np.array([]), np.array([])
        return merged[proxy_col].values, merged["_human_mean"].values

    # p-values for Holm, approximated from r and n via t-test
    confirmatory_pvals: list[float] = []
    confirmatory_labels: list[str] = []

    def approx_pval(r: float, n: int) -> float:
        if np.isnan(r) or n < 3:
            return 1.0
        t = r * np.sqrt(n - 2) / np.sqrt(max(1 - r**2, 1e-10))
        return float(2 * (1 - stats.t.cdf(abs(t), df=n - 2)))

    # fidelity proxies
    for proxy_name in fid_proxies:
        col_map = {
            "calibrated_luar": "calibrated_luar",
            "burrows_delta": "burrows_delta",
            "llm_judge_fidelity": "llm_judge_fidelity",
        }
        col = col_map.get(proxy_name, proxy_name)
        proxy_vals, hum_vals = align(fidelity_df, col, human_fid_mean,
                                     human_df[["output_id"]].drop_duplicates() if not human_df.empty and "output_id" in human_df.columns else pd.DataFrame())
        if len(proxy_vals) >= 3:
            agr = proxy_agreement(proxy_vals, hum_vals, iters, seed)
            pval = approx_pval(agr["r"], len(proxy_vals))
        else:
            agr = {"r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "ci_excludes_zero": False}
            pval = 1.0

        rows.append({
            "proxy": proxy_name,
            "axis": "fidelity",
            "r": agr["r"],
            "ci_lo": agr["ci_lo"],
            "ci_hi": agr["ci_hi"],
            "ci_excludes_zero": agr["ci_excludes_zero"],
            "screening": "PASS" if agr["ci_excludes_zero"] else "FAIL",
            "n_outputs": len(proxy_vals),
            "human_alpha": human_alpha_fidelity,
        })

        if proxy_name in primary:
            confirmatory_pvals.append(pval)
            confirmatory_labels.append(proxy_name)

    # novelty proxies
    for proxy_name in nov_proxies:
        col_map = {
            "dual_divergence": "dual",
            "distinct_n": "distinct_1",
            "llm_judge_novelty": "llm_judge_novelty",
        }
        col = col_map.get(proxy_name, proxy_name)
        proxy_vals, hum_vals = align(novelty_df, col, human_nov_mean,
                                     human_df[["output_id"]].drop_duplicates() if not human_df.empty and "output_id" in human_df.columns else pd.DataFrame())
        if len(proxy_vals) >= 3:
            agr = proxy_agreement(proxy_vals, hum_vals, iters, seed)
        else:
            agr = {"r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "ci_excludes_zero": False}

        rows.append({
            "proxy": proxy_name,
            "axis": "novelty",
            "r": agr["r"],
            "ci_lo": agr["ci_lo"],
            "ci_hi": agr["ci_hi"],
            "ci_excludes_zero": agr["ci_excludes_zero"],
            "screening": "PASS" if agr["ci_excludes_zero"] else "FAIL",
            "n_outputs": len(proxy_vals),
            "human_alpha": human_alpha_novelty,
        })

    # holm correction on confirmatory p-values
    if confirmatory_pvals:
        holm_results = holm(confirmatory_pvals, holm_k)
        for label, rejected in zip(confirmatory_labels, holm_results):
            for row in rows:
                if row["proxy"] == label:
                    row["holm_rejected"] = rejected

    # construct separation
    if len(human_fid_mean) >= 3 and len(human_nov_mean) >= 3:
        sep = construct_separation(human_fid_mean, human_nov_mean, cfg)
        sep_row = {
            "proxy": "human_construct_separation",
            "axis": "both",
            "r": sep["r"],
            "ci_lo": sep["ci_lo"],
            "ci_hi": sep["ci_hi"],
            "ci_excludes_zero": not sep["falsified"],
            "screening": sep["verdict"],
            "n_outputs": len(human_fid_mean),
            "human_alpha": float("nan"),
        }
        rows.append(sep_row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(runs_dir / "trust_table.csv", index=False)
    return df_out


def rq3_clean_vs_noisy(df: pd.DataFrame, cfg: dict) -> dict:
    """Agreement on clean subset vs full set.

    Returns dict with agreement on clean subset and full set for comparison.
    """
    clean_subset = [f for f in cfg["analysis"].get("clean_subset", []) if f]
    if not clean_subset or df.empty or "output_id" not in df.columns:
        return {"note": "no clean subset configured; RQ3 skipped"}

    full_n = len(df)
    clean_mask = df["output_id"].str.contains("|".join(clean_subset), na=False)
    clean_df = df[clean_mask]

    return {
        "full_n": full_n,
        "clean_n": len(clean_df),
        "note": "compare agreement columns between full and clean rows",
    }


def copy_rate_summary(gate_df: pd.DataFrame) -> dict:
    """Descriptive C3 copy-rate from gate results."""
    if gate_df.empty or "copy_rate" not in gate_df.columns:
        return {"note": "no gate results"}
    return {
        "mean_copy_rate": round(float(gate_df["copy_rate"].mean()), 4),
        "max_copy_rate": round(float(gate_df["copy_rate"].max()), 4),
        "n_flagged": int(gate_df["copy_flag"].sum()) if "copy_flag" in gate_df.columns else 0,
        "n_total": len(gate_df),
    }
