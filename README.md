# fidelity-novelty-eval

A code-only evaluation framework for measuring and validating automatic metrics of LLM
stylistic voice imitation, along two axes: voice fidelity (does generated text reproduce a
target author's idiolect) and creative novelty (does it diverge from the source rather than
echo it). The framework establishes metric validity without human labels, through known-groups
(manipulation) validity, a multitrait–multimethod (MTMM) convergent/discriminant analysis, and
an audit of an LLM judge treated as a studied object. It then tests whether a trusted fidelity
proxy transfers to best-of-N candidate selection under anti-circularity controls.

## What it measures

The corpus is built from a single target author (solo monologue = the target voice) and a
"floor" of other speakers (interview guests, extracted turn-by-turn). Generations are produced
under three known-groups conditions designed to span a fidelity gradient — C1 (generic, no
conditioning), C2 (style card), C3 (style card plus BM25 retrieval) — crossed with a two-level
style-card ablation (full vs. none).

Two constructs, each estimated by three method-independent proxies:

- **Voice fidelity**
  - *Calibrated authorship-embedding score* — LUAR-MUD author embeddings; cosine to the target
    centroid is rescaled against the guest-floor distribution and offset by the generic-LLM (C1)
    anchor, so the score reads as "more target-like than generic, in units of floor spread."
  - *Function-word stylometry (Burrows's Delta)* — a self-contained numpy Delta over the top
    function words, gated by a held-out real-target-vs-real-floor pre-flight that skips the proxy
    if it cannot separate the two classes.
  - *LLM-judge fidelity score* — a blind 1–5 voice-match rating.
- **Creative novelty**
  - *Embedding divergence* — sentence-embedding distance from the nearest source chunk and from
    the C1 centroid (the selection-study variant takes the nearest over the whole archive).
  - *Distinct-n* — lexical-diversity ratios of unique to total n-grams.
  - *LLM-judge novelty score* — a blind 1–5 originality rating.

The LLM judge is not treated as ground truth: a separate audit characterizes its score
distribution, mode share, and entropy, and a cross-family re-audit re-scores the same outputs
with a different model family to test whether that behaviour is a property of the task or of
using the generator's own family as the judge.

## Pipeline

The modules, grouped by role:

- **Corpus / vetting** — `vet.py` (single-speaker, clip-handoff, caption-quality, and dedup
  gates; writes a cleaned corpus and a manifest with exclusions recorded as data), `corpus.py`
  (manifest-driven loaders, guest-turn extraction, archive chunking), `fetch.py` (standalone
  caption fetcher).
- **Generation** — `generate.py` (seeded prompts across C1/C2/C3 × ablation), `judge.py` (the
  two blind LLM judges). Both incur API cost and run only behind explicit flags.
- **Retrieval** — `retrieve.py` (BM25 index and top-k retrieval over archive chunks).
- **Scoring** — `fidelity.py` (calibrated LUAR, Burrows's Delta), `novelty.py` (divergence,
  Distinct-n), `gate.py` (verbatim-overlap memorization safeguard, run before scoring).
- **Construct-validity analysis** — `construct_validity.py` (known-groups fidelity gradient via
  Jonckheere–Terpstra, discriminant dissociation, style-card ablation, the MTMM convergent/
  discriminant gap, and a Holm-corrected confirmatory family; all CIs are cluster bootstraps that
  resample prompts), `judge_audit.py` and `judge_crossfamily.py` (judge audits), `analysis.py`
  (bootstrap / Holm / reliability helpers), and `revision_analyses*.py` (robustness analyses).
- **Selection study** — `partii_generate.py` (best-of-N temperature-sampled candidates; resumable
  and spend-capped), `partii_score.py` (selector score plus a LUAR-independent held-out certifier
  and the memorization gate), `partii_analysis.py` (dose-response, fidelity × novelty frontier,
  retention-policy ablation, and mechanism / length-confound diagnostics). The selector
  (calibrated LUAR), the held-out certifier (char-n-gram TF-IDF), and the frontier novelty axis
  (Distinct-2) are kept referent-disjoint to avoid circularity.
- **Figures** — `figures.py`, plus figures emitted by `partii_analysis.py`.
- **Orchestration** — `pipeline.py` is the single entry point. It defaults to the no-cost stages
  over existing outputs; the API-calling steps run only with explicit flags. Everything is seeded
  from `config.yaml`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY
```

Models run on CPU. API keys are read from the environment or from `.env` (gitignored); generation
and the LLM judge are the only steps that call an API, and both run only behind explicit flags.

```bash
python pipeline.py                  # no-cost: score and analyze existing outputs
python pipeline.py --generate       # also generate (calls the API)
python pipeline.py --judge          # also run the LLM judges (calls the API)
python construct_validity.py        # known-groups + MTMM + confirmatory family
python partii_analysis.py           # best-of-N selection study
```

## Reproducing the numbers

The pre-registration is in `preregistration.md`. Every number behind the paper's tables and figures
is released as a result CSV in `data/`, alongside the per-output proxy scores. The construct-validity
statistics regenerate from these CSVs with no corpus required — set `paths.runs_dir: data` in
`config.yaml` (or copy `data/*.csv` into a `runs/` directory) and run:

```bash
python construct_validity.py     # known-groups, MTMM, confirmatory family
python judge_audit.py            # LLM-judge audit
python partii_analysis.py        # selection study
```

Steps that rebuild embeddings (the LUAR / MiniLM scoring) additionally require the corpus and do not
run without it.

## Data and reuse

The source corpus is third-party copyrighted material and is **not** included — neither are the
generated outputs or any derived corpus text. What ships here is the code, the hand-authored style
card (`style_card.md`), the pre-registration, and the numeric metric CSVs in `data/` (all free-text
columns removed). With your own transcripts under the path set in `config.yaml` (`transcripts_root`),
the full pipeline regenerates deterministically under the fixed seed. The code is shared for academic
reference and reproduction.

Author: Karla Lučić
