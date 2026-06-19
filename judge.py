"""
One Anthropic LLM-judge per axis (fidelity, novelty), blind to system, logging
prompts to runs/. Only runs when enabled; reads ANTHROPIC_API_KEY from env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


_FIDELITY_SYSTEM = """You are a blind evaluator assessing whether a text passage sounds like a
specific, distinctive podcast host. You do not know which system or condition produced this text.
Rate ONLY the voice/style match on a scale from 1 to 5:

1 = Sounds completely generic; no distinctive voice
2 = Slight hints of a distinctive voice, mostly generic
3 = Moderately distinctive; some signature features present
4 = Strongly distinctive; clearly a specific, recognizable voice
5 = Unmistakably captures a very specific, highly distinctive voice

Respond with ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}"""


_NOVELTY_SYSTEM = """You are a blind evaluator assessing the originality of a text passage.
You do not know which system or condition produced this text.
Rate ONLY originality/non-genericness on a scale from 1 to 5:

1 = Completely generic; sounds like default AI output or obvious restating
2 = Mostly generic with minor original touches
3 = Some original recombination of ideas; not entirely predictable
4 = Meaningfully original; recombines ideas in a non-obvious way
5 = Highly original; surprising and intelligent recombination of ideas

Respond with ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}"""


def _call_judge(
    client,
    model_name: str,
    system_prompt: str,
    text: str,
    output_id: str,
    axis: str,
    log_f,
    runs_dir: Path,
) -> dict:
    """Call the LLM judge for one output; return {output_id, score, reason}."""
    user_content = f"Please evaluate the following passage:\n\n---\n{text}\n---"

    log_entry = {
        "output_id": output_id,
        "axis": axis,
        "system": system_prompt,
        "user": user_content,
        "model": model_name,
    }
    log_f.write(json.dumps(log_entry) + "\n")
    log_f.flush()

    response = client.messages.create(
        model=model_name,
        max_tokens=200,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text.strip()

    try:
        parsed = json.loads(raw)
        score = float(parsed.get("score", float("nan")))
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        score = float("nan")
        reason = f"parse error: {raw[:100]}"

    return {"output_id": output_id, f"llm_judge_{axis}": score, f"{axis}_reason": reason}


def judge_fidelity(
    outputs: dict,
    cfg: dict,
    enabled: bool,
) -> pd.DataFrame:
    """One blind LLM-judge fidelity score per output; empty DataFrame if disabled."""
    if not enabled:
        return pd.DataFrame()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Export before passing --judge.")

    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    model_name = cfg["models"]["judge"]
    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / "judge_fidelity_log.jsonl"

    rows: list[dict] = []
    with open(log_path, "a", encoding="utf-8") as log_f:
        for output_id, text in outputs.items():
            row = _call_judge(
                client, model_name, _FIDELITY_SYSTEM, text, output_id, "fidelity", log_f, runs_dir
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(runs_dir / "judge_fidelity.csv", index=False)
    return df


def judge_novelty(
    outputs: dict,
    cfg: dict,
    enabled: bool,
) -> pd.DataFrame:
    """One blind LLM-judge novelty score per output; empty DataFrame if disabled."""
    if not enabled:
        return pd.DataFrame()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Export before passing --judge.")

    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    model_name = cfg["models"]["judge"]
    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / "judge_novelty_log.jsonl"

    rows: list[dict] = []
    with open(log_path, "a", encoding="utf-8") as log_f:
        for output_id, text in outputs.items():
            row = _call_judge(
                client, model_name, _NOVELTY_SYSTEM, text, output_id, "novelty", log_f, runs_dir
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(runs_dir / "judge_novelty.csv", index=False)
    return df
