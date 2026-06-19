"""
Best-of-N candidate generation for Part 2, concurrent with resume.

For each {condition} x {ablation} x {prompt} cell, draws N temperature-sampled candidates via a
thread pool, skipping candidate_ids already in runs/candidates_log.jsonl. Requires --generate to
hit the API and caps at --max-calls.

  python partii_generate.py --generate --n 2 --max-calls 4 --limit-prompts 1
  python partii_generate.py --generate --n 8 --max-calls 600 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from generate import _seed_prompts, build_prompt


def load_cfg(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _done_ids(log_path: Path) -> set:
    done = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["candidate_id"])
            except Exception:
                pass
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Best-of-N candidate generation (concurrent; calls the API).")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--conditions", nargs="+", default=["C2", "C3"])
    ap.add_argument("--ablations", nargs="+", default=["full", "none"])
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    cfg = load_cfg(args.config)
    runs = Path(cfg["paths"]["runs_dir"]); runs.mkdir(parents=True, exist_ok=True)
    log_path = runs / "candidates_log.jsonl"
    prompts = _seed_prompts(cfg)
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    done = _done_ids(log_path)
    tasks = []
    for c in args.conditions:
        for a in args.ablations:
            for i in range(len(prompts)):
                pid = f"p{i:02d}"
                for k in range(args.n):
                    cid = f"{c}_{a}_{pid}_k{k}"
                    if cid not in done:
                        tasks.append((c, a, pid, k, cid))
    tasks = tasks[: args.max_calls]
    print(f"[gen2] resume: {len(done)} already done; {len(tasks)} to generate "
          f"(workers={args.workers}, generate={args.generate})")
    if not args.generate:
        print("[gen2] dry plan only; no API calls."); return 0
    if not tasks:
        print("[gen2] nothing to do."); return 0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import envload; envload.load_dotenv(); api_key = os.environ.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set; cannot --generate.")

    import anthropic
    from corpus import load_pool, chunk_archive
    from retrieve import build_index, retrieve

    client = anthropic.Anthropic(api_key=api_key)
    model = cfg["models"]["generation"]
    max_tokens = cfg["generation"]["max_output_tokens"]
    temperature = cfg["generation"]["temperature"]
    top_k = cfg["retrieval"]["top_k"]
    style_card = Path(cfg["paths"]["style_card"])
    style_card_text = style_card.read_text(encoding="utf-8") if style_card.exists() else ""

    # cached per-prompt retrieval, shared read-only across threads
    target_texts = load_pool(cfg, "target")
    archive_chunks = chunk_archive(target_texts, cfg)
    index = build_index(archive_chunks, cfg)
    retr_cache = {}
    for i in range(len(prompts)):
        retr_cache[f"p{i:02d}"] = [c["text"] for c in retrieve(index, archive_chunks, prompts[i], top_k)]

    lock = threading.Lock()
    state = {"calls": 0, "in": 0, "out": 0, "fail": 0}
    logf = open(log_path, "a", encoding="utf-8")

    def work(task):
        c, a, pid, k, cid = task
        retrieved = retr_cache[pid] if c == "C3" else []
        prompt = build_prompt(pid, c, a, style_card_text, retrieved, cfg)
        for attempt in range(4):
            try:
                resp = client.messages.create(
                    model=model, max_tokens=max_tokens, temperature=temperature,
                    system=cfg["generation"].get("system", ""),
                    messages=[{"role": "user", "content": prompt}],
                )
                rec = dict(candidate_id=cid, condition=c, ablation=a, prompt_id=pid,
                           n_idx=k, text=resp.content[0].text)
                with lock:
                    logf.write(json.dumps({**rec, "model": model,
                                           "in_tok": resp.usage.input_tokens,
                                           "out_tok": resp.usage.output_tokens}) + "\n")
                    logf.flush()
                    state["calls"] += 1; state["in"] += resp.usage.input_tokens
                    state["out"] += resp.usage.output_tokens
                    if state["calls"] % 25 == 0:
                        print(f"[gen2] {state['calls']}/{len(tasks)} done "
                              f"(in={state['in']:,} out={state['out']:,})", flush=True)
                return True
            except Exception as e:
                if attempt == 3:
                    with lock:
                        state["fail"] += 1
                    print(f"[gen2] FAIL {cid}: {e}", flush=True)
                    return False
                time.sleep(2 ** attempt * 2)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(work, t) for t in tasks]))
    logf.close()

    # rebuild candidates.csv from the log, deduped by id
    recs = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line); recs[r["candidate_id"]] = r
        except Exception:
            pass
    df = pd.DataFrame([{kk: r[kk] for kk in ["candidate_id", "condition", "ablation", "prompt_id", "n_idx", "text"]}
                       for r in recs.values()])
    df.to_csv(runs / "candidates.csv", index=False)
    print(f"[gen2] done. new_calls={state['calls']} fails={state['fail']} "
          f"total_candidates={len(df)} tokens in={state['in']:,} out={state['out']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
