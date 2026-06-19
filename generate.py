"""
Generate C1/C2/C3 outputs over the seeded prompt set across ablation levels.

Logs every prompt and response to runs/. Only runs when --generate is passed;
generate_all(cfg, enabled=False) returns immediately with no API call.
Reads ANTHROPIC_API_KEY from env.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pandas as pd


_PROMPT_TEMPLATES = [
    "Talk about the founding of {company} and what made {founder} different from everyone else.",
    "What can we learn from {founder}'s obsession with {topic}?",
    "Explain why {founder} succeeded where others failed when building {company}.",
    "What is the single most important lesson from {founder}'s biography?",
    "Describe {founder}'s approach to {topic} and how it shaped {company}.",
    "What would {founder} say about the importance of {topic} in building a great company?",
    "Tell me about the moment {founder} decided to {action} and what happened next.",
    "How did {founder} think about {topic} differently from everyone else in their era?",
    "What made {founder}'s approach to {company} so unusual and so effective?",
    "Describe {founder}'s relationship with {topic} and why it mattered so much.",
    "What is the core idea behind {founder}'s success with {company}?",
    "How did {founder} build {company} into something no one thought was possible?",
    "What does {founder}'s story teach us about the role of {topic} in business?",
    "Why did {founder} keep going when everyone said {company} would fail?",
    "Explain {founder}'s philosophy on {topic} as if teaching it to someone for the first time.",
    "What is the most counterintuitive thing about how {founder} built {company}?",
]

# coherent founder/company pairs that overlap the Senra solo episodes, so C3 retrieval has signal
_FOUNDER_COMPANY = [
    ("Steve Jobs", "Apple"),
    ("Jeff Bezos", "Amazon"),
    ("John D. Rockefeller", "Standard Oil"),
    ("Warren Buffett", "Berkshire Hathaway"),
    ("Enzo Ferrari", "Ferrari"),
    ("Akio Morita", "Sony"),
    ("Estee Lauder", "Estee Lauder"),
    ("Ken Griffin", "Citadel"),
    ("Bill Gates", "Microsoft"),
    ("Elon Musk", "Tesla"),
    ("Sam Walton", "Walmart"),
    ("Phil Knight", "Nike"),
    ("Walt Disney", "Disney"),
    ("Henry Ford", "Ford"),
    ("Bruce Springsteen", "his music"),
    ("Rick Rubin", "his records"),
]

_FILL_POOLS = {
    "topic": [
        "obsession", "simplicity", "long-term thinking", "constraints",
        "talent", "execution", "customer focus", "capital allocation",
        "iteration", "storytelling",
    ],
    "action": [
        "quit a steady job", "bet everything on one idea", "ignore conventional wisdom",
        "hire people smarter than themselves", "focus on a single market",
    ],
}


def _seed_prompts(cfg: dict) -> list[str]:
    """Build the seeded prompt set from config.seed."""
    rng = random.Random(cfg["seed"])
    templates = _PROMPT_TEMPLATES[: cfg["generation"]["n_prompts"]]
    prompts: list[str] = []
    for template in templates:
        founder, company = rng.choice(_FOUNDER_COMPANY)
        fills = {"founder": founder, "company": company}
        for key, pool in _FILL_POOLS.items():
            fills[key] = rng.choice(pool)
        prompts.append(template.format(**fills))
    return prompts


def build_prompt(
    prompt_id: str,
    condition: str,
    ablation: str,
    style_card: str,
    retrieved: list[str],
    cfg: dict,
) -> str:
    """Assemble the seeded prompt for a (condition, ablation) cell.

    C1: bare prompt, no style card, no retrieval.
    C2: prompt + style_card if ablation=='full'; no retrieval.
    C3: prompt + style_card if ablation=='full' + retrieved chunks.
    """
    prompts = _seed_prompts(cfg)
    # prompt_id is like "p00", "p01"
    idx = int(prompt_id[1:]) if len(prompt_id) > 1 and prompt_id[0] == "p" else 0
    base_prompt = prompts[idx] if idx < len(prompts) else prompts[0]

    parts: list[str] = []

    if condition == "C1":
        parts.append(
            "Write a long-form passage (200-400 words) about the following topic:\n\n"
            + base_prompt
        )

    elif condition in ("C2", "C3"):
        if ablation == "full" and style_card:
            parts.append(
                "You are imitating the voice of a specific podcast host. "
                "Study the following style guide carefully before writing.\n\n"
                "=== STYLE GUIDE ===\n"
                + style_card
                + "\n=== END STYLE GUIDE ===\n"
            )
        if condition == "C3" and retrieved:
            parts.append(
                "\n=== RETRIEVED ARCHIVE PASSAGES ===\n"
                + "\n---\n".join(retrieved)
                + "\n=== END RETRIEVED PASSAGES ===\n"
            )
        parts.append(
            "\nNow write a 200-400 word passage in that voice about:\n\n"
            + base_prompt
        )
    else:
        parts.append(base_prompt)

    return "\n".join(parts)


def generate_all(cfg: dict, enabled: bool) -> pd.DataFrame:
    """Generate C1/C2/C3 outputs and log each prompt/response to runs/ as JSON lines.

    When disabled, returns before constructing the client or reading the key.
    """
    if not enabled:
        return pd.DataFrame()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set in environment. "
            "Export the key before passing --generate."
        )

    import anthropic  # type: ignore
    from corpus import chunk_archive, load_pool  # type: ignore
    from retrieve import build_index, retrieve  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    model_name = cfg["models"]["generation"]
    max_tokens = cfg["generation"]["max_output_tokens"]
    temperature = cfg["generation"]["temperature"]

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    style_card_path = Path(cfg["paths"]["style_card"])
    style_card_text = style_card_path.read_text(encoding="utf-8") if style_card_path.exists() else ""

    target_texts = load_pool(cfg, "target")
    archive_chunks = chunk_archive(target_texts, cfg)
    index = build_index(archive_chunks, cfg)

    prompts = _seed_prompts(cfg)
    conditions = cfg["generation"]["conditions"]
    ablation_levels = cfg["generation"]["ablation_levels"]
    top_k = cfg["retrieval"]["top_k"]

    log_path = runs_dir / "generation_log.jsonl"
    rows: list[dict] = []

    with open(log_path, "a", encoding="utf-8") as log_f:
        for cond in conditions:
            # C1 has no style card to ablate
            levels = ["none"] if cond == "C1" else ablation_levels
            for ablation in levels:
                for idx, base_prompt in enumerate(prompts):
                    prompt_id = f"p{idx:02d}"

                    if cond == "C3":
                        top_chunks = retrieve(index, archive_chunks, base_prompt, top_k)
                        retrieved_texts = [c["text"] for c in top_chunks]
                    else:
                        retrieved_texts = []

                    full_prompt = build_prompt(
                        prompt_id, cond, ablation, style_card_text, retrieved_texts, cfg
                    )

                    # log the prompt before sending
                    log_entry = {
                        "output_id": f"{cond}_{ablation}_{prompt_id}",
                        "condition": cond,
                        "ablation": ablation,
                        "prompt_id": prompt_id,
                        "prompt": full_prompt,
                        "model": model_name,
                    }
                    log_f.write(json.dumps(log_entry) + "\n")
                    log_f.flush()

                    response = client.messages.create(
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=cfg["generation"].get("system", ""),
                        messages=[{"role": "user", "content": full_prompt}],
                    )
                    output_text = response.content[0].text

                    output_id = f"{cond}_{ablation}_{prompt_id}"
                    output_path = runs_dir / f"{output_id}.txt"
                    output_path.write_text(output_text, encoding="utf-8")

                    rows.append(
                        {
                            "output_id": output_id,
                            "condition": cond,
                            "ablation": ablation,
                            "prompt_id": prompt_id,
                            "text": output_text,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(runs_dir / "outputs.csv", index=False)
    return df
