"""yt-dlp wrapper to fetch auto-captions into the transcript tree.

Standalone tool, not called by pipeline.py. After fetching, re-run pipeline.py
from the vet step to ingest the new files.

    python fetch.py <url> [--dest solo-david-senra|floor-solo] [--config config.yaml]
    python fetch.py --list urls.txt [--dest floor-solo]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def fetch(url: str, dest_dir: str, cfg: dict) -> str:
    """Download English auto-captions for url into dest_dir under transcripts_root.

    Returns the path to the saved transcript file, or an empty string on failure.
    """
    transcripts_root = Path(cfg["paths"]["transcripts_root"])
    dest_path = transcripts_root / dest_dir
    dest_path.mkdir(parents=True, exist_ok=True)

    output_template = str(dest_path / "%(title)s_transcript.%(ext)s")

    cmd = [
        "yt-dlp",
        "--write-auto-subs",
        "--skip-download",
        "--sub-lang", "en",
        "--convert-subs", "srt",
        "--output", output_template,
        url,
    ]

    print(f"[fetch] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"[fetch] ERROR: yt-dlp exited with code {result.returncode}", file=sys.stderr)
        return ""

    subtitles = sorted(dest_path.glob("*.srt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not subtitles:
        subtitles = sorted(dest_path.glob("*.vtt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if subtitles:
        # rename to .txt to match the existing corpus
        srt_path = subtitles[0]
        txt_path = srt_path.with_suffix(".txt")
        srt_path.rename(txt_path)
        print(f"[fetch] Saved transcript to: {txt_path}")
        return str(txt_path)

    print("[fetch] WARNING: no subtitle file found after download.", file=sys.stderr)
    return ""


def main(argv: list[str]) -> int:
    """Fetch one URL or a list of URLs into the destination subdirectory."""
    parser = argparse.ArgumentParser(
        description="Fetch YouTube auto-captions into the transcript tree (NETWORK)."
    )
    parser.add_argument("url", nargs="?", help="YouTube URL to fetch.")
    parser.add_argument(
        "--list",
        metavar="FILE",
        help="Text file with one URL per line.",
    )
    parser.add_argument(
        "--dest",
        default="floor-solo",
        help="Destination subdirectory under transcripts_root (default: floor-solo).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[fetch] ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    urls: list[str] = []
    if args.url:
        urls.append(args.url)
    if args.list:
        list_path = Path(args.list)
        if not list_path.exists():
            print(f"[fetch] ERROR: URL list not found: {list_path}", file=sys.stderr)
            return 1
        urls.extend(
            line.strip() for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    if not urls:
        print("[fetch] No URLs provided. Pass a URL or --list <file>.", file=sys.stderr)
        return 1

    for url in urls:
        fetch(url, args.dest, cfg)

    print(
        "\n[fetch] Done. Re-run pipeline.py (step 1: vet) to ingest new files into manifest.csv."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
