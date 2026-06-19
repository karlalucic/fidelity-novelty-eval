"""Evaluation pipeline: vet, corpus, retrieve, gate, fidelity, novelty, judges, metric CSVs.

By default it scores existing runs/ outputs; generation and LLM judging are opt-in via
--generate / --judge. Run `python pipeline.py` for the default scoring pass.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def seed_all(seed: int) -> None:
    """Seed numpy, random, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_cfg(config_path: str = "config.yaml") -> dict:
    """Load config.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_existing_outputs(cfg: dict) -> tuple[dict, dict, dict]:
    """Load generated outputs from runs/.

    Returns (all_outputs, c1_outputs, judged_outputs), each {output_id: text}.
    """
    runs_dir = Path(cfg["paths"]["runs_dir"])

    outputs_csv = runs_dir / "outputs.csv"
    if outputs_csv.exists():
        df = pd.read_csv(outputs_csv)
        all_outputs = {row["output_id"]: row["text"] for _, row in df.iterrows()
                       if "output_id" in df.columns and "text" in df.columns}
    else:
        # fallback: individual <condition>_<ablation>_<prompt_id>.txt files
        all_outputs = {}
        for fpath in sorted(runs_dir.glob("C[123]_*.txt")):
            output_id = fpath.stem
            all_outputs[output_id] = fpath.read_text(encoding="utf-8")

    c1_outputs = {k: v for k, v in all_outputs.items() if k.startswith("C1_")}
    judged = {k: v for k, v in all_outputs.items()
              if any(k.startswith(c + "_") for c in cfg["generation"]["judged_conditions"])}

    return all_outputs, c1_outputs, judged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Senra evaluation pipeline. Defaults: --no-generate --no-judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        default=False,
        help="call the Anthropic API to generate outputs (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="run the Anthropic LLM-judge per axis (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        default=False,
        help="Skip LUAR and sentence-transformer loading (for smoke testing without model files).",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # load .env (ANTHROPIC_API_KEY) without requiring a shell export
    try:
        import envload  # type: ignore
        envload.load_dotenv()
    except Exception:
        pass  # .env is optional; the key is validated at call time

    cfg = load_cfg(args.config)
    seed = cfg["seed"]
    seed_all(seed)

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] seed={seed}")
    print(f"[pipeline] generate={args.generate}, judge={args.judge}")

    print("\n[pipeline] Step 1: Vetting gate (vet.py)...")
    import vet  # type: ignore
    manifest_df = vet.vet_corpus(cfg)
    print(f"[pipeline] Manifest: {len(manifest_df)} files; "
          f"{manifest_df['pass'].sum()} passed, "
          f"{(~manifest_df['pass']).sum()} excluded.")

    print("\n[pipeline] Step 2: Loading corpus...")
    import corpus as corpus_mod  # type: ignore
    target_texts = corpus_mod.load_pool(cfg, "target")
    print(f"[pipeline] Target texts: {len(target_texts)}")
    archive_chunks = corpus_mod.chunk_archive(target_texts, cfg)
    print(f"[pipeline] Archive chunks: {len(archive_chunks)}")

    print("\n[pipeline] Step 3: Generation (gated)...")
    import generate as gen_mod  # type: ignore
    if args.generate:
        print("[pipeline] --generate flag set: calling Anthropic API...")
        gen_mod.generate_all(cfg, enabled=True)
    else:
        print("[pipeline] --no-generate (default): skipping generation.")

    all_outputs, c1_outputs, judged_outputs = load_existing_outputs(cfg)
    print(f"[pipeline] Loaded {len(all_outputs)} outputs "
          f"({len(c1_outputs)} C1, {len(judged_outputs)} judged C2/C3).")

    if not all_outputs:
        print("[pipeline] No outputs found in runs/. Run with --generate or add output files.")
        print("[pipeline] Pipeline complete (no outputs to score).")
        return 0

    # build index, attach retrieved chunks to C3 outputs
    print("\n[pipeline] Step 4: Building BM25 retrieval index...")
    import retrieve as ret_mod  # type: ignore
    index = ret_mod.build_index(archive_chunks, cfg)
    top_k = cfg["retrieval"]["top_k"]

    retrieved_per_output: dict[str, list[dict]] = {}
    c3_outputs = {k: v for k, v in all_outputs.items() if k.startswith("C3_")}
    for output_id, text in c3_outputs.items():
        # use output text as query proxy (no stored prompt)
        retrieved_per_output[output_id] = ret_mod.retrieve(index, archive_chunks, text[:500], top_k)

    print("\n[pipeline] Step 5: Memorisation gate (gate.py)...")
    import gate as gate_mod  # type: ignore
    if c3_outputs:
        archive_texts = [c["text"] for c in archive_chunks]
        gate_df = gate_mod.gate_batch(
            c3_outputs,
            {k: [c["text"] for c in v] for k, v in retrieved_per_output.items()},
            archive_texts,
            cfg,
        )
        gate_df.to_csv(runs_dir / "gate.csv", index=False)
        from analysis import copy_rate_summary  # type: ignore
        cr = copy_rate_summary(gate_df)
        print(f"[pipeline] Gate: mean copy_rate={cr.get('mean_copy_rate', 'N/A')}, "
              f"flagged={cr.get('n_flagged', 0)}/{cr.get('n_total', 0)}")
    else:
        gate_df = pd.DataFrame()
        print("[pipeline] No C3 outputs; gate skipped.")

    print("\n[pipeline] Step 6: Fidelity scores (fidelity.py)...")
    fidelity_df = pd.DataFrame()
    if not args.skip_models:
        import fidelity as fid_mod  # type: ignore
        try:
            fidelity_df = fid_mod.score_fidelity(all_outputs, cfg)
            fidelity_df.to_csv(runs_dir / "fidelity_scores.csv", index=False)
            print(f"[pipeline] Fidelity scores: {len(fidelity_df)} outputs.")
        except Exception as e:
            print(f"[pipeline] WARNING: fidelity scoring failed: {e}")
    else:
        print("[pipeline] --skip-models: fidelity scoring skipped.")

    print("\n[pipeline] Step 7: Novelty scores (novelty.py)...")
    novelty_df = pd.DataFrame()
    if not args.skip_models:
        import novelty as nov_mod  # type: ignore
        try:
            c1_embs = None
            if c1_outputs:
                c1_texts = list(c1_outputs.values())
                c1_embs = nov_mod.embed(c1_texts, cfg)
            else:
                print("[pipeline] WARNING: no C1 outputs for centroid; using zero vector.")
                c1_embs = np.zeros((1, 384))  # all-MiniLM-L6-v2 dim=384

            novelty_df = nov_mod.score_novelty(
                all_outputs,
                {k: [c["text"] for c in v] for k, v in retrieved_per_output.items()},
                c1_embs,
                cfg,
            )
            novelty_df.to_csv(runs_dir / "novelty_scores.csv", index=False)
            print(f"[pipeline] Novelty scores: {len(novelty_df)} outputs.")
        except Exception as e:
            print(f"[pipeline] WARNING: novelty scoring failed: {e}")
    else:
        print("[pipeline] --skip-models: novelty scoring skipped.")

    print("\n[pipeline] Step 8: LLM judges (gated)...")
    import judge as judge_mod  # type: ignore
    judge_fid_df = judge_mod.judge_fidelity(judged_outputs, cfg, enabled=args.judge)
    judge_nov_df = judge_mod.judge_novelty(judged_outputs, cfg, enabled=args.judge)
    if not args.judge:
        print("[pipeline] --no-judge (default): LLM judges skipped.")

    # merge judge results into metric frames
    if not judge_fid_df.empty and not fidelity_df.empty and "output_id" in fidelity_df.columns:
        fidelity_df = fidelity_df.merge(
            judge_fid_df[["output_id", "llm_judge_fidelity"]], on="output_id", how="left"
        )
    if not judge_nov_df.empty and not novelty_df.empty and "output_id" in novelty_df.columns:
        novelty_df = novelty_df.merge(
            judge_nov_df[["output_id", "llm_judge_novelty"]], on="output_id", how="left"
        )

    print("\n[pipeline] Step 9: RQ3 clean-vs-noisy (analysis.py)...")
    from analysis import rq3_clean_vs_noisy  # type: ignore
    rq3 = rq3_clean_vs_noisy(fidelity_df, cfg)
    if "note" not in rq3 or "skipped" not in rq3.get("note", ""):
        print(f"[pipeline] RQ3 clean-vs-noisy: {rq3}")

    print(f"\n[pipeline] Done. Metric CSVs in {runs_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
