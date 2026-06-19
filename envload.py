"""Minimal .env loader (no dependency on python-dotenv).

Reads KEY=VALUE lines from a .env file next to this module and injects any keys
not already in the environment. Lines starting with '#' and blank lines are ignored.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | None = None) -> None:
    p = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = val
