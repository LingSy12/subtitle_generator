"""Merge two SRT files into one, sorting cues by start time and renumbering.

Usage:
    python merge_srt.py HEAD.srt TAIL.srt OUTPUT.srt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TIMING_RE = re.compile(r"^(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})")


def parse_srt_time(text: str) -> float:
    text = text.strip().replace(",", ".")
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60_000) % 60
    h = total_ms // 3_600_000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def read_cues(path: Path) -> list[tuple[float, float, str]]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\r?\n+", raw.strip())
    cues: list[tuple[float, float, str]] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        idx = 1 if lines[0].strip().isdigit() else 0
        if idx >= len(lines):
            continue
        m = TIMING_RE.match(lines[idx].strip())
        if not m:
            continue
        start = parse_srt_time(m.group(1))
        end = parse_srt_time(m.group(2))
        body = "\n".join(lines[idx + 1 :]).strip()
        if not body:
            continue
        cues.append((start, end, body))
    return cues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge two SRT files by start time.")
    parser.add_argument("head", help="First SRT (typically the early-time partial)")
    parser.add_argument("tail", help="Second SRT (typically the resumed tail)")
    parser.add_argument("out", help="Output SRT path")
    parser.add_argument(
        "--dedupe-overlap",
        type=float,
        default=0.5,
        help="Drop tail cues whose start is within N seconds of an existing head cue end (default 0.5).",
    )
    args = parser.parse_args(argv)

    head_path = Path(args.head)
    tail_path = Path(args.tail)
    out_path = Path(args.out)

    head_cues = read_cues(head_path)
    tail_cues = read_cues(tail_path)
    print(f"head: {len(head_cues)} cues from {head_path}")
    print(f"tail: {len(tail_cues)} cues from {tail_path}")

    head_end = head_cues[-1][1] if head_cues else 0.0
    threshold = head_end - max(0.0, args.dedupe_overlap)
    filtered_tail = [c for c in tail_cues if c[0] >= threshold]
    if len(filtered_tail) != len(tail_cues):
        print(f"  dropped {len(tail_cues) - len(filtered_tail)} tail cues that overlap head")

    merged = head_cues + filtered_tail
    merged.sort(key=lambda c: c[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="\n") as fh:
        for index, (start, end, body) in enumerate(merged, start=1):
            fh.write(f"{index}\n")
            fh.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
            fh.write(f"{body}\n\n")

    print(f"merged: {len(merged)} cues -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
