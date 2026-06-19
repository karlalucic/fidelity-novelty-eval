"""
Render the paper figures from the construct-validity result CSVs.

Writes four vector PDFs (dissociation, known-groups, MTMM heatmap, judge
distributions) to ../paper/tex/figures/. Run construct_validity.py and
judge_audit.py first, then: python figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from construct_validity import load_merged, FIDELITY, NOVELTY, PROXIES

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COND = ["C1", "C2", "C3"]


def _cfg() -> dict:
    return yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))


def _outdir(cfg) -> Path:
    d = Path("../paper/tex/figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def fig_dissociation(means: pd.DataFrame, out: Path) -> None:
    none = means[means.ablation == "none"]
    luar = none[none.proxy == "calibrated_luar"].set_index("condition").reindex(COND)
    dist = none[none.proxy == "distinct_2"].set_index("condition").reindex(COND)
    fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(3)
    ax1.errorbar(x - 0.04, luar["mean"], yerr=[luar["mean"] - luar["ci_lo"], luar["ci_hi"] - luar["mean"]],
                 marker="o", color="#1f77b4", capsize=3, label="fidelity (calibrated LUAR)")
    ax1.set_ylabel("calibrated LUAR (fidelity)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx(); ax2.spines["top"].set_visible(False)
    ax2.errorbar(x + 0.04, dist["mean"], yerr=[dist["mean"] - dist["ci_lo"], dist["ci_hi"] - dist["mean"]],
                 marker="s", color="#d62728", capsize=3, label="novelty (distinct-2)")
    ax2.set_ylabel("distinct-2 (lexical novelty)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xticks(x); ax1.set_xticklabels(["C1\n(generic)", "C2\n(style card)", "C3\n(RAG)"])
    ax1.set_title("Fidelity rises as novelty falls (within ablation=none)", pad=12)
    fig.savefig(out / "fig_dissociation.pdf"); plt.close(fig)


def fig_known_groups(means: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(3); w = 0.36
    for i, (abl, color) in enumerate([("none", "#7f7f7f"), ("full", "#2ca02c")]):
        sub = means[(means.proxy == "calibrated_luar") & (means.ablation == abl)].set_index("condition").reindex(COND)
        ax.bar(x + (i - 0.5) * w, sub["mean"], w, color=color, label=f"ablation={abl}",
               yerr=[sub["mean"] - sub["ci_lo"], sub["ci_hi"] - sub["mean"]], capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(["C1", "C2", "C3"])
    ax.set_ylabel("calibrated LUAR (fidelity)")
    ax.set_title("Known-groups: fidelity recovers the designed gradient", pad=12)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / "fig_known_groups.pdf"); plt.close(fig)


def fig_mtmm(runs: Path, out: Path) -> None:
    cmat = pd.read_csv(runs / "mtmm_spearman.csv", index_col=0).reindex(index=PROXIES, columns=PROXIES)
    labels = ["LUAR", "−Burrows", "judge-fid", "dual-div", "distinct-2", "judge-nov"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cmat.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(labels)
    for i in range(6):
        for j in range(6):
            v = cmat.to_numpy(float)[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=8)
    # block separator between fidelity (0-2) and novelty (3-5)
    ax.axhline(2.5, color="k", lw=1.2); ax.axvline(2.5, color="k", lw=1.2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    ax.set_title("MTMM within C3: convergent (blocks) vs discriminant (off-blocks)")
    fig.savefig(out / "fig_mtmm.pdf"); plt.close(fig)


def fig_judge_dist(runs: Path, out: Path) -> None:
    dist = pd.read_csv(runs / "judge_audit_dist.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)
    for ax, axis, color in [(axes[0], "fidelity", "#1f77b4"), (axes[1], "novelty", "#d62728")]:
        d = dist[dist.axis == axis]
        ax.bar(d["score"], d["share"], width=0.6, color=color)
        ax.set_title(f"judge {axis}"); ax.set_xlabel("score (1–5)"); ax.set_xticks([1, 2, 3, 4, 5])
    axes[0].set_ylabel("share of outputs")
    fig.suptitle("LLM-judge score distributions (degeneracy at 3.0)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig_judge_dist.pdf"); plt.close(fig)


def main() -> int:
    cfg = _cfg(); runs = Path(cfg["paths"]["runs_dir"]); out = _outdir(cfg)
    means = pd.read_csv(runs / "known_groups_means.csv")
    fig_dissociation(means, out)
    fig_known_groups(means, out)
    fig_mtmm(runs, out)
    fig_judge_dist(runs, out)
    print(f"[figures] wrote 4 PDFs to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
