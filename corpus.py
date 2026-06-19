"""Manifest-driven corpus loader.

Reads manifest.csv (passing rows only), builds target / guest-floor / solo-floor
pools, extracts non-host turns from interview transcripts, and chunks the target
archive for RAG. Two speaker-label formats are handled: "Name: text" and "[Name] text".
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

# a speaker label is 1-4 Title-Case tokens, so ordinary sentences don't match
_INLINE_COLON = re.compile(r"^(?P<n>[A-Z][A-Za-z.'\-]*(?: [A-Z][A-Za-z.'\-]*){0,3}):\s+(?P<t>.+)$")
_INLINE_BRACK = re.compile(r"^\[(?P<n>[^\]]{1,40})\]\s*(?P<t>.*)$")
_TIMESTAMP = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$")
_META_LABELS = {"title", "date", "source", "url", "duration", "transcript"}


def load_manifest(cfg: dict) -> pd.DataFrame:
    """Read manifest.csv; return only rows with pass==True."""
    df = pd.read_csv(cfg["paths"]["manifest"])
    df["pass"] = df["pass"].astype(str).str.lower().map({"true": True, "false": False})
    return df[df["pass"] == True].copy()  # noqa: E712


def load_pool(cfg: dict, role: str) -> list[str]:
    """Return cleaned texts for a role (target | floor_solo) via the manifest clean_path."""
    df = load_manifest(cfg)
    texts: list[str] = []
    for _, row in df[df["role"] == role].iterrows():
        clean_path = str(row.get("clean_path", "") or "")
        if clean_path and Path(clean_path).exists():
            texts.append(Path(clean_path).read_text(encoding="utf-8"))
    return texts


def extract_speaker_turns(interview_text: str, host_label: str = "David Senra"):
    """Split an interview transcript into (guest_text, host_text).

    Format-agnostic over 'Name: text' and '[Name] text'. Metadata lines
    (Title/Date/Source) and stray timestamp lines are dropped; an unlabeled
    continuation line attaches to the current speaker.
    """
    host_key = host_label.strip().lower()
    guest, host = [], []
    current = None  # "host" | "guest" | None
    for ln in interview_text.splitlines():
        s = ln.strip()
        if not s:
            continue
        m = _INLINE_BRACK.match(s) or _INLINE_COLON.match(s)
        if m:
            name = m.group("n").strip()
            if name.lower() in _META_LABELS:
                current = None
                continue
            current = "host" if name.lower() == host_key else "guest"
            body = m.group("t").strip()
            if body:
                (host if current == "host" else guest).append(body)
            continue
        if _TIMESTAMP.match(s):
            continue
        if current == "host":
            host.append(s)
        elif current == "guest":
            guest.append(s)
    return " ".join(guest), " ".join(host)


def extract_guest_turns(interview_text: str, host_label: str = "David Senra") -> str:
    """Return only the guest (non-host) speech from an interview transcript."""
    guest, _host = extract_speaker_turns(interview_text, host_label)
    return guest


def build_guest_floor(cfg: dict) -> list[str]:
    """Extract each interview's guest (non-host) turns, one floor text per interview.

    Reads the raw interview file (labels intact), not the cleaned file. Every named
    speaker other than host_label is treated as the guest.
    """
    df = load_manifest(cfg)
    host_label = cfg["corpus"].get("host_label", "David Senra") if "corpus" in cfg \
        else cfg.get("host_label", "David Senra")
    transcripts_root = Path(cfg["paths"]["transcripts_root"])
    texts: list[str] = []
    for _, row in df[df["role"] == "floor_guest"].iterrows():
        raw_path = transcripts_root / str(row.get("folder", "")) / str(row["filename"])
        if not raw_path.exists():
            continue
        guest = extract_guest_turns(raw_path.read_text(encoding="utf-8", errors="replace"), host_label)
        if guest.strip():
            texts.append(guest)
    return texts


def chunk_archive(target_texts: list[str], cfg: dict) -> list[dict]:
    """Split target texts into overlapping word chunks for RAG."""
    chunk_words = cfg["retrieval"]["chunk_words"]
    overlap_words = cfg["retrieval"]["chunk_overlap_words"]
    step = max(1, chunk_words - overlap_words)
    chunks: list[dict] = []
    cid = 0
    for doc_idx, text in enumerate(target_texts):
        words = text.split()
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_words]
            if len(piece) >= 20:
                chunks.append({"id": f"doc{doc_idx}_chunk{cid}", "text": " ".join(piece)})
                cid += 1
            if start + chunk_words >= len(words):
                break
    return chunks


def held_out_passages(cfg: dict, n: int, words: int):
    """Return (target_passages, floor_passages), each ~``words`` words, seeded for reproducibility."""
    rng = random.Random(cfg["seed"])

    def passages(texts: list[str]) -> list[str]:
        allw: list[str] = []
        for t in texts:
            allw.extend(t.split())
        rng.shuffle(allw)
        out = []
        i = 0
        while i + words <= len(allw) and len(out) < n:
            out.append(" ".join(allw[i:i + words]))
            i += words
        return out

    target_texts = load_pool(cfg, "target")
    floor_texts = build_guest_floor(cfg)
    try:
        floor_texts = floor_texts + load_pool(cfg, "floor_solo")
    except Exception:
        pass
    return passages(target_texts)[:n], passages(floor_texts)[:n]
