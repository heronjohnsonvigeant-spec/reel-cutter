"""Break down a reference reel into a format spec Claude can rebuild from.

Usage (from repo root):
    .venv-reels/Scripts/python.exe -I scripts/reels/breakdown_ref.py VIDEO --out DIR [--every 2]

Writes to DIR:
    transcript.txt    timestamped spoken lines (faster-whisper base.en)
    cuts.txt          detected hard cuts / scene changes (ffmpeg scdet)
    sheetNN.png       contact sheets, one frame every --every seconds
    BREAKDOWN.md      duration, cut rate, transcript, and blanks for the
                      format notes (hook text, layout, card moments, CTA)

Borrow the format, never the words (hooks may be borrowed; scripts may not).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def scene_cuts(video: Path, threshold: float) -> list[float]:
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video), "-vf", f"scdet=threshold={threshold}", "-an", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stderr
    return [float(m.group(1)) for m in re.finditer(r"lavfi\.scd\.time:\s*([\d.]+)", err)]


def contact_sheets(video: Path, out: Path, every: float) -> list[Path]:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"fps=1/{every},scale=300:-1,tile=6x2", str(out / "sheet%02d.png")],
        check=True,
    )
    return sorted(out.glob("sheet*.png"))


def transcribe(video: Path) -> list[tuple[float, float, str]]:
    from faster_whisper import WhisperModel  # .venv-reels only

    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(video))
    return [(s.start, s.end, s.text.strip()) for s in segs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=float, default=2.0, help="seconds between contact-sheet frames")
    ap.add_argument("--threshold", type=float, default=10.0, help="scdet threshold (lower = more cuts)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    dur = duration(a.video)
    cuts = scene_cuts(a.video, a.threshold)
    sheets = contact_sheets(a.video, a.out, a.every)
    lines = transcribe(a.video)

    (a.out / "transcript.txt").write_text(
        "\n".join(f"[{s:5.1f}-{e:5.1f}] {t}" for s, e, t in lines), encoding="utf-8")
    (a.out / "cuts.txt").write_text("\n".join(f"{c:.2f}" for c in cuts), encoding="utf-8")

    words = sum(len(t.split()) for _, _, t in lines)
    rate = dur / (len(cuts) + 1)
    md = [
        f"# Reference breakdown: {a.video.name}",
        "",
        f"- Duration: {dur:.1f} s",
        f"- Visual changes detected: {len(cuts)} (one every {rate:.1f} s; scdet threshold {a.threshold}, "
        "screen recordings can over-count from UI flicker, check the sheets)",
        f"- Spoken words: {words} ({words / dur:.1f} words/s)",
        f"- Contact sheets: {', '.join(p.name for p in sheets)} (one frame every {a.every:g} s, left to right, top to bottom)",
        "",
        "## Transcript",
        "",
        *[f"- `{s:5.1f}` {t}" for s, _, t in lines],
        "",
        "## Format notes (fill in from the sheets)",
        "",
        "- On-screen hook text and where it sits:",
        "- Layout per section (talking head / split screen / full-screen card / B-roll):",
        "- Card or graphic moments (spoken phrase -> what the graphic shows):",
        "- Caption style:",
        "- CTA (keyword, what it delivers):",
        "- What to borrow (format only, never the script):",
        "",
    ]
    (a.out / "BREAKDOWN.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {a.out / 'BREAKDOWN.md'}")


if __name__ == "__main__":
    main()
