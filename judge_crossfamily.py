"""
Re-scores the 64 judged outputs (C2,C3 x {full,none} x 16 prompts) with an OpenAI judge,
using the same blind rubric prompts as the Sonnet judge, to see whether the fidelity-score
degeneracy is a property of the task or of using the generator's own model family as judge.

Reads OPENAI_API_KEY from env (or senra-eval/.env); needs the openai package.
Run:  .venv/bin/python judge_crossfamily.py --model gpt-5.5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import envload
envload.load_dotenv()

from judge import _FIDELITY_SYSTEM, _NOVELTY_SYSTEM
from judge_audit import load_cfg, _entropy_norm


def _judged_outputs(cfg) -> pd.DataFrame:
    runs = Path(cfg["paths"]["runs_dir"])
    outs = pd.read_csv(runs / "outputs.csv")
    jc = cfg["generation"]["judged_conditions"]   # [C2, C3]
    return outs[outs.condition.isin(jc)].copy()


def _parse(raw: str) -> tuple[float, str]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        i = raw.find("{")
        if i >= 0:
            raw = raw[i:]
    try:
        p = json.loads(raw)
        return float(p.get("score", float("nan"))), str(p.get("reason", ""))
    except Exception:
        m = re.search(r'"score"\s*:\s*([1-5])', raw) or re.search(r'\b([1-5])\b', raw)
        return (float(m.group(1)) if m else float("nan")), (f"parse:{raw[:80]}" if raw else "empty")


# probe the working kwarg shape once, then reuse it
_WORKING: dict | None = None


def _create(client, model: str, messages: list, max_out: int, temperature: float | None):
    global _WORKING
    base = dict(model=model, messages=messages)
    candidates = (
        [_WORKING] if _WORKING is not None else
        [
            {"max_completion_tokens": max_out},                  # gpt-5 family (default temperature)
            {"max_completion_tokens": max_out, "_temp": True},   # models that also allow temperature
            {"max_tokens": max_out, "_temp": True},              # gpt-4o family
            {"max_tokens": max_out},
        ]
    )
    last = None
    for shape in candidates:
        kw = dict(base)
        kw.update({k: v for k, v in shape.items() if k != "_temp" and not k.startswith("max_")})
        # carry the right token-budget key at the requested size
        if "max_completion_tokens" in shape:
            kw["max_completion_tokens"] = max_out
        if "max_tokens" in shape:
            kw["max_tokens"] = max_out
        if shape.get("_temp") and temperature is not None:
            kw["temperature"] = temperature
        try:
            resp = client.chat.completions.create(**kw)
            _WORKING = shape
            return resp
        except Exception as e:  # noqa: BLE001
            last = e
            _WORKING = None
            continue
    raise last


def _call(client, model: str, system: str, text: str, temperature: float | None) -> tuple[float, str]:
    user = f"Please evaluate the following passage:\n\n---\n{text}\n---"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    resp = _create(client, model, messages, max_out=4000, temperature=temperature)
    content = resp.choices[0].message.content or ""
    if not content.strip():                      # reasoning may have eaten the budget; retry larger once
        resp = _create(client, model, messages, max_out=12000, temperature=temperature)
        content = resp.choices[0].message.content or ""
    return _parse(content)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="applied only to gpt-4* models; gpt-5 family uses its fixed default")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    cfg = load_cfg()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not set (add it to senra-eval/.env).")
    from openai import OpenAI
    client = OpenAI(api_key=key)
    temp = args.temperature if args.model.startswith("gpt-4") else None

    runs = Path(cfg["paths"]["runs_dir"])
    sub = _judged_outputs(cfg)
    print(f"[xfam] judging {len(sub)} outputs x 2 axes with {args.model} "
          f"(temp={'default' if temp is None else temp}) = {len(sub) * 2} calls")

    rows, n_unparsed = [], 0
    for axis, system, col in [("fidelity", _FIDELITY_SYSTEM, "llm_judge_fidelity"),
                              ("novelty", _NOVELTY_SYSTEM, "llm_judge_novelty")]:
        recs = []
        for r in sub.itertuples():
            score, reason = _call(client, args.model, system, str(r.text), temp)
            if not np.isfinite(score):
                n_unparsed += 1
            recs.append(dict(output_id=r.output_id, condition=r.condition, ablation=r.ablation,
                             **{col: score}, reason=reason))
        df = pd.DataFrame(recs)
        df.to_csv(runs / f"judge_crossfamily_{axis}.csv", index=False)
        s = df[col].dropna().to_numpy(float)
        vc = pd.Series(s).value_counts().sort_index()
        mode_share = float(vc.max() / len(s)); ent = _entropy_norm(vc.to_numpy())
        cond_means = {f"mean_{c}": round(float(df[df.condition == c][col].mean()), 3) for c in ["C2", "C3"]}
        rows.append(dict(axis=axis, model=args.model, n=len(s), n_distinct=int(vc.size),
                         mode=float(vc.idxmax()), mode_share=round(mode_share, 4),
                         normalized_entropy=round(ent, 4), mean=round(float(s.mean()), 3),
                         sd=round(float(s.std(ddof=1)), 3), **cond_means))
        print(f"  [{axis}] dist={ {float(k): int(v) for k, v in vc.items()} }  "
              f"mode_share={mode_share:.4f}  entropy={ent:.4f}")

    summ = pd.DataFrame(rows)
    summ.to_csv(runs / "judge_crossfamily_audit.csv", index=False)
    print("\n=== CROSS-FAMILY JUDGE AUDIT ===")
    print(summ.to_string(index=False))
    print(f"\nunparsed responses: {n_unparsed}")
    print("(Sonnet-4.6 baseline -> fidelity: mode_share 0.8438, entropy 0.49; "
          "novelty: mode_share 0.625, entropy 0.64)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
