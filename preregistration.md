# Pre-registration — Human-Free Construct Validity (Senra dual-axis)

> Locked 2026-06-07, **before** the confirmatory re-analysis and before any Part-2 generation.
> **Honesty note:** the descriptive pilot (2026-05-31) has already been run and some directional
> results are known (LUAR gradient, judge degeneracy, Burrows C1=C2 tie). This document therefore
> pre-registers the *confirmatory test definitions, directions, and decision rules* — it does not
> claim the analyst was blind to pilot direction. Confirmatory inference is reported separately from
> exploratory analysis throughout. Plan of record: `plan-human-free.md`.

## Constructs, proxies, and orientation

Two constructs. Each proxy is oriented so **higher = more of the construct** before any analysis:

- **Fidelity:** `calibrated_luar` (higher = more Senra ✓); `neg_burrows = −burrows_delta` (Burrows Δ is
  reverse-coded, so we flip it); `llm_judge_fidelity` 1–5 (✓; C2/C3 only).
- **Novelty:** `dual` (dual divergence, higher = more novel ✓; **meaningful only within C3** — C1/C2
  have placeholder `dist_nearest_chunk = 1.0`); `distinct_n` (lexical diversity; **`distinct_2` primary**,
  `distinct_1` robustness; meaningful all conditions); `llm_judge_novelty` 1–5 (✓; C2/C3 only).

## Design facts (fixed)

- Conditions C1 (unconditioned) / C2 (style card, no RAG) / C3 (RAG); 2-level style-card ablation
  (full / none); **n = 16 prompts per cell**, shared across conditions.
- **Unit of analysis = the prompt.** Every CI is a **cluster bootstrap that resamples prompts**
  (B = 10 000, percentile, seed = config.seed). Effect sizes report **Cliff's δ** (rank, robust to LUAR
  sd heterogeneity) alongside the mean contrast.
- Known-groups is reported **stratified by ablation**; the **within-ablation=none** stratum is primary
  (de-confounds the gradient from the ablation). MTMM primary stratum is **within-C3** (the only stratum
  where all six proxies are simultaneously meaningful).

## Confirmatory family (Holm-corrected, α = 0.05, k = 4)

All one-sided in the pre-specified direction.

| # | Confirmatory hypothesis | Test | Direction | Falsified if |
|---|------------------------|------|-----------|--------------|
| C1 | `calibrated_luar` recovers the designed gradient, within ablation=none | Jonckheere–Terpstra trend (normal approx) over C1<C2<C3 | increasing | JT not significant after Holm, OR the C3−C1 cluster-bootstrap CI includes 0 |
| C2 | Style card raises fidelity (`calibrated_luar`, full > none) | Wilcoxon signed-rank, paired by prompt, within C2∪C3 | full > none | Holm-corrected p ≥ .05 OR paired bootstrap CI of (full−none) includes 0 |
| C3 | Discriminant dissociation: novelty falls as fidelity rises | `distinct_2` C1 → C3 within ablation=none | decreasing (C3 < C1) | Holm-corrected p ≥ .05 OR the C1−C3 bootstrap CI includes 0 |
| C4 | MTMM convergent > discriminant (instrument has structure), within C3 | bootstrap of gap = mean\|convergent r\| − mean\|discriminant r\|; p = fraction of bootstrap gaps ≤ 0 | gap > 0 | Holm-corrected p ≥ .05 OR gap CI includes 0 |

Decision rule: a confirmatory claim stands only if it survives **Holm correction across these four**
*and* the corresponding bootstrap CI excludes the null. Anything else is reported as null/inconclusive.

## Pre-specified analysis choices (no post-hoc swaps)

- Correlations: **Spearman primary** (robust to the judge's ties/degeneracy), Pearson reported alongside.
- The LLM judge is an **audited object**, never ground truth and never a selector. Its degeneracy
  (~84% of fidelity scores at 3.0 in the pilot) is reported as a finding; α/agreement use ordinal stats.
- `dual` and `dist_nearest_chunk` are used **only within C3**; cross-condition novelty uses
  `dist_c1_centroid` / `distinct_n` only.
- Burrows is reported but **not** part of the confirmatory family for the gradient (pilot shows C1=C2 tie);
  it serves as the **C1-anchor robustness check** for Part-2 frontier/discriminant claims (no C1 term).

## Exploratory (explicitly not confirmatory)

Full 6×6 MTMM matrix; pooled (non-stratified) gradients; novelty/judge known-groups; per-cell
correlations; `distinct_1` and Pearson variants; gate copy-rate descriptives; all visualizations.

## Part 2 pre-commitments (selection study)

- **Three-role rule:** each metric is at most one of {selector, held-out evaluator, frontier axis} per claim.
- **Selector** = `calibrated_luar`; **held-out certifier** = an independent CPU-only style/authorship
  encoder distinct from LUAR-MUD (chosen + verified at build); **frontier axes** = fidelity (LUAR; Burrows
  robustness pass) × **`distinct_n`** (referent-disjoint).
- **Primary selection test** = dose-response: held-out-certifier gain vs N ∈ {1,2,4,8}, cluster-bootstrap CI.
- **Gate before select** on the full candidate pool; gate is exact-match-only (state this; no
  "non-plagiarized by construction" claim); report copy-rate of top-fidelity winners.
- A **null / hard-trade-off frontier is pre-committed as a publishable result** (it corroborates the
  novelty↔fidelity tension), not a failed experiment.

## Integrity

No fabricated citations or results; unverified citations marked TODO. All numbers measured, not assumed.
No data mismatch/corruption; pilot CSVs are the single source of truth. API key stays in `.env`.
