"""
Audit the LLM judge from the existing CSVs. Per axis, reports the score
distribution and mode share, distinct-value count, normalized entropy,
per-condition means, and reason-text length stats. Writes
runs/judge_audit_dist.csv and runs/judge_audit_summary.csv.

Run: python judge_audit.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_cfg(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _entropy_norm(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    h = -(p * np.log2(p)).sum()
    hmax = np.log2(len(p)) if len(p) > 1 else 1.0
    return float(h / hmax) if hmax > 0 else 0.0


def audit(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = Path(cfg["paths"]["runs_dir"])
    outs = pd.read_csv(runs / "outputs.csv")[["output_id", "condition", "ablation", "prompt_id"]]
    jf = pd.read_csv(runs / "judge_fidelity.csv")
    jn = pd.read_csv(runs / "judge_novelty.csv")

    dist_rows, summ_rows = [], []
    for axis, jdf, score_col, reason_col in [
        ("fidelity", jf, "llm_judge_fidelity", "fidelity_reason"),
        ("novelty", jn, "llm_judge_novelty", "novelty_reason"),
    ]:
        m = outs.merge(jdf, on="output_id", how="inner")
        scores = m[score_col].dropna().to_numpy(float)
        vc = pd.Series(scores).value_counts().sort_index()
        for val, cnt in vc.items():
            dist_rows.append(dict(axis=axis, score=val, count=int(cnt),
                                  share=round(cnt / len(scores), 4)))
        mode_val = vc.idxmax(); mode_share = vc.max() / len(scores)
        reasons = m[reason_col].dropna().astype(str)
        wlens = reasons.str.split().apply(len)
        # per-condition means
        cond_means = {f"mean_{c}": round(float(m[m.condition == c][score_col].mean()), 3)
                      for c in ["C2", "C3"]}
        summ_rows.append(dict(
            axis=axis, n=len(scores), n_distinct_values=int(vc.size),
            mode=float(mode_val), mode_share=round(float(mode_share), 4),
            normalized_entropy=round(_entropy_norm(vc.to_numpy()), 4),
            mean=round(float(scores.mean()), 3), sd=round(float(scores.std(ddof=1)), 3),
            reason_words_mean=round(float(wlens.mean()), 1),
            reason_words_sd=round(float(wlens.std(ddof=1)), 1),
            **cond_means,
        ))
    return pd.DataFrame(dist_rows), pd.DataFrame(summ_rows)


def main() -> int:
    cfg = load_cfg()
    runs = Path(cfg["paths"]["runs_dir"])
    dist, summ = audit(cfg)
    dist.to_csv(runs / "judge_audit_dist.csv", index=False)
    summ.to_csv(runs / "judge_audit_summary.csv", index=False)
    print("=== JUDGE SCORE DISTRIBUTION ===")
    print(dist.to_string(index=False))
    print("\n=== JUDGE AUDIT SUMMARY ===")
    print(summ.to_string(index=False))
    print(f"\n[judge_audit] wrote judge_audit_dist.csv, judge_audit_summary.csv to {runs}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
