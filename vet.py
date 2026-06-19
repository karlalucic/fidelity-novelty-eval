"""
Vetting gate. Scans the transcript tree, runs single-speaker, clip-handoff,
caption-quality and dedup checks against config thresholds, strips headers and
inline timestamps, and writes the cleaned corpus plus manifest.csv.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def clean_text(raw: str) -> str:
    """Strip the 4-line auto-caption header, inline timestamps, speaker markers,
    rolling-overlap duplicate lines; return canonical prose.

    The solo-Senra header is 4 lines:
        This transcript was generated automatically. Its accuracy may vary.
        Ads/promotional sections removed.
        Source: https://...
        Title: ...

    Inline timestamps look like ``2:53`` or ``10:02`` or ``1:01:45``.
    Speaker markers look like ``Speaker 1``, ``Speaker 2``, etc.
    The interview files also have bare timestamp lines (``0:02``) on their own line.
    """
    lines = raw.splitlines()

    # strip the 4-line auto-caption header if present (matched by first two lines)
    header_keywords = [
        "this transcript was generated automatically",
        "ads/promotional sections removed",
    ]
    start = 0
    if len(lines) >= 2:
        if (header_keywords[0] in lines[0].lower()
                or header_keywords[1] in lines[1].lower()):
            # skip the header block, then any blank line
            start = 4
            while start < len(lines) and lines[start].strip() == "":
                start += 1

    lines = lines[start:]

    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()

        # bare timestamp lines, e.g. "0:02", "10:32", "1:01:45"
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", stripped):
            continue

        # inline timestamps and speaker labels
        stripped = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", "", stripped)
        stripped = re.sub(r"\bSpeaker\s+\d+\b", "", stripped)

        stripped = stripped.strip()
        if stripped:
            cleaned.append(stripped)

    # drop YouTube rolling-overlap lines (it repeats the last sentence of a
    # segment at the start of the next)
    deduped: list[str] = []
    for line in cleaned:
        if deduped and line == deduped[-1]:
            continue
        deduped.append(line)

    return "\n".join(deduped)


def check_single_speaker(raw: str, cfg: dict) -> tuple[bool, str]:
    """Return (pass, reason) based on the ratio of non-primary-speaker labelled text.

    For solo files (no Speaker labels) this trivially passes.
    For interview files we count ``Speaker N`` occurrences; if any non-primary label
    accounts for more than ``single_speaker_max_other_label_ratio`` of all labelled
    lines, reject.
    """
    threshold = cfg["vetting"]["single_speaker_max_other_label_ratio"]
    label_pattern = re.compile(r"Speaker\s+(\d+)")
    matches = label_pattern.findall(raw)
    if not matches:
        return True, "no speaker labels (trivially single-speaker)"

    from collections import Counter
    counts = Counter(matches)
    total = sum(counts.values())
    primary = counts.most_common(1)[0][0]
    other = total - counts[primary]
    ratio = other / total if total > 0 else 0.0
    if ratio > threshold:
        return (
            False,
            f"multi-speaker: other-speaker ratio {ratio:.3f} > {threshold} "
            f"(primary=Speaker {primary}, counts={dict(counts)})",
        )
    return True, f"single-speaker check passed (other-speaker ratio {ratio:.3f})"


def check_clip_handoff(raw: str, cfg: dict) -> tuple[bool, str]:
    """Return (pass, reason); fail if any clip_handoff_markers present."""
    markers = cfg["vetting"]["clip_handoff_markers"]
    lower = raw.lower()
    for marker in markers:
        if marker in lower:
            return False, f"clip-handoff marker found: '{marker}'"
    return True, "no clip-handoff markers"


def check_caption_quality(raw: str, cfg: dict) -> tuple[bool, str]:
    """Return (pass, reason) from junk/duplication ratio + min_words floor.

    Junk ratio = fraction of tokens that are: repeated consecutive identical
    tokens, or pure punctuation fillers (e.g. '...', '---').
    """
    min_words = cfg["vetting"]["min_words"]
    max_junk = cfg["vetting"]["max_caption_junk_ratio"]

    tokens = raw.split()
    word_count = len(tokens)

    if word_count < min_words:
        return False, f"too short: {word_count} words < {min_words} minimum"

    junk = 0
    for i, tok in enumerate(tokens):
        if i > 0 and tok == tokens[i - 1]:
            junk += 1
        elif re.fullmatch(r"[^a-zA-Z0-9]+", tok):
            junk += 1

    ratio = junk / word_count if word_count > 0 else 0.0
    if ratio > max_junk:
        return False, f"caption junk ratio {ratio:.3f} > {max_junk}"
    return True, f"caption quality ok (words={word_count}, junk={ratio:.3f})"


def _ngrams(tokens: list[str], n: int) -> set[tuple]:
    """Return the set of n-gram tuples from a token list."""
    return {tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


def dedup_flags(texts: dict[str, str], cfg: dict) -> dict[str, str]:
    """5-gram Jaccard near-duplicate detection.

    Returns ``{filename: 'kept' | 'dup_of:<other>'}`` for all files. The longer
    of a near-duplicate pair is kept; the shorter is flagged. Short shared
    passages don't trip the whole-document Jaccard, which is dominated by
    unique content.
    """
    threshold = cfg["vetting"]["max_dup_jaccard"]
    filenames = list(texts.keys())
    token_sets: dict[str, set] = {}
    for fn, text in texts.items():
        tokens = text.lower().split()
        token_sets[fn] = _ngrams(tokens, 5)

    status: dict[str, str] = {fn: "kept" for fn in filenames}

    for i, fn_a in enumerate(filenames):
        if status[fn_a].startswith("dup_of"):
            continue
        for fn_b in filenames[i + 1:]:
            if status[fn_b].startswith("dup_of"):
                continue
            a_grams = token_sets[fn_a]
            b_grams = token_sets[fn_b]
            union = len(a_grams | b_grams)
            if union == 0:
                continue
            jaccard = len(a_grams & b_grams) / union
            if jaccard > threshold:
                # keep the longer one
                len_a = len(texts[fn_a].split())
                len_b = len(texts[fn_b].split())
                if len_a >= len_b:
                    status[fn_b] = f"dup_of:{fn_a}"
                else:
                    status[fn_a] = f"dup_of:{fn_b}"

    return status


def assign_role(
    filename: str,
    folder: str,
    passed: bool,
    cfg: dict,
) -> str:
    """Map to {target, floor_guest, floor_solo, interview, EXCLUDE}.

    - solo_dir + pass    -> target
    - interview_dir + in guest_speaker_map + pass -> floor_guest
    - interview_dir + not in guest_speaker_map  -> interview (then EXCLUDE)
    - floor_solo_dir + pass -> floor_solo
    - anything failing   -> EXCLUDE (reason is the check result)
    """
    if not passed:
        return "EXCLUDE"

    solo_dir = cfg["paths"]["solo_dir"]
    interview_dir = cfg["paths"]["interview_dir"]
    floor_solo_dir = cfg["paths"]["floor_solo_dir"]
    guest_map = cfg.get("guest_speaker_map", {})

    if folder == solo_dir:
        return "target"
    if folder == interview_dir:
        # every passing interview is a guest-floor source; host turns are
        # dropped at extraction time
        return "floor_guest"
    if folder == floor_solo_dir:
        return "floor_solo"
    return "EXCLUDE"


def vet_corpus(cfg: dict) -> pd.DataFrame:
    """Scan the transcript tree, run all checks, write cleaned text and
    manifest.csv, and return the manifest dataframe.

    Reads transcripts_root and config thresholds; writes clean_corpus_dir/*.txt
    and paths.manifest.
    """
    transcripts_root = Path(cfg["paths"]["transcripts_root"])
    clean_dir = Path(cfg["paths"]["clean_corpus_dir"])
    clean_dir.mkdir(parents=True, exist_ok=True)

    solo_dir = cfg["paths"]["solo_dir"]
    interview_dir = cfg["paths"]["interview_dir"]
    floor_solo_dir = cfg["paths"]["floor_solo_dir"]
    guest_map = cfg.get("guest_speaker_map", {})

    folders = [solo_dir, interview_dir, floor_solo_dir]

    candidates: list[dict] = []
    raw_texts: dict[str, str] = {}

    for folder in folders:
        folder_path = transcripts_root / folder
        if not folder_path.exists():
            continue
        for fpath in sorted(folder_path.glob("*.txt")):
            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            key = f"{folder}/{fpath.name}"
            raw_texts[key] = raw
            candidates.append({"filename": fpath.name, "folder": folder, "path": fpath})

    cleaned_texts: dict[str, str] = {}
    for cand in candidates:
        key = f"{cand['folder']}/{cand['filename']}"
        cleaned_texts[key] = clean_text(raw_texts.get(key, ""))

    # global dedup so near-duplicate episodes in the same role are caught
    dedup_input = {key: cleaned_texts[key] for key in cleaned_texts}
    dup_status = dedup_flags(dedup_input, cfg)

    rows: list[dict] = []

    for cand in candidates:
        filename = cand["filename"]
        folder = cand["folder"]
        key = f"{folder}/{filename}"
        raw = raw_texts.get(key, "")
        cleaned = cleaned_texts.get(key, "")

        ok_speaker, reason_speaker = check_single_speaker(raw, cfg)
        ok_handoff, reason_handoff = check_clip_handoff(raw, cfg)
        ok_quality, reason_quality = check_caption_quality(cleaned, cfg)

        dup_flag = dup_status.get(key, "kept")
        is_dup = dup_flag.startswith("dup_of")

        word_count = len(cleaned.split())

        # caption junk ratio, recomputed for the manifest
        tokens = cleaned.split()
        junk = sum(
            1 for i, t in enumerate(tokens)
            if (i > 0 and t == tokens[i - 1]) or re.fullmatch(r"[^a-zA-Z0-9]+", t)
        )
        junk_ratio = junk / len(tokens) if tokens else 0.0

        # other-speaker ratio
        label_pattern = re.compile(r"Speaker\s+(\d+)")
        matches = label_pattern.findall(raw)
        if matches:
            from collections import Counter
            counts = Counter(matches)
            total = sum(counts.values())
            primary_count = counts.most_common(1)[0][1]
            other_ratio = (total - primary_count) / total if total > 0 else 0.0
        else:
            other_ratio = 0.0

        passed = ok_speaker and ok_handoff and ok_quality and not is_dup

        if not ok_speaker:
            reason = reason_speaker
        elif not ok_handoff:
            reason = reason_handoff
        elif is_dup:
            reason = f"near-duplicate: {dup_flag}"
        elif not ok_quality:
            reason = reason_quality
        else:
            reason = "PASS"

        role = assign_role(filename, folder, passed, cfg)

        clean_path = ""
        if passed and role != "EXCLUDE":
            clean_filename = f"{folder.replace('/', '_')}_{filename}"
            clean_fpath = clean_dir / clean_filename
            clean_fpath.write_text(cleaned, encoding="utf-8")
            clean_path = str(clean_fpath)

        guest_speaker_id = ""
        if role == "floor_guest":
            guest_speaker_id = guest_map.get(filename, "")

        rows.append(
            {
                "filename": filename,
                "folder": folder,
                "role": role,
                "pass": passed and role != "EXCLUDE",
                "reason": reason,
                "word_count": word_count,
                "caption_junk_ratio": round(junk_ratio, 4),
                "other_speaker_ratio": round(other_ratio, 4),
                "dup_of": dup_flag if is_dup else "",
                "guest_speaker_id": guest_speaker_id,
                "clean_path": clean_path,
            }
        )

    df = pd.DataFrame(rows)

    manifest_path = cfg["paths"]["manifest"]
    df.to_csv(manifest_path, index=False)

    return df
