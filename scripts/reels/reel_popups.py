"""Reel pop-ups: animated cards over the talking head, timed to the words that trigger them.

A card in cards.json can carry cues, each anchored on a phrase you say:
  "popups": [{"at": "345 dials", "kind": "stat", "value": 345, "label": "Dials"},
             {"at": "how are you", "kind": "list", "n": 1, "text": "\"How are you?\"", "strike": true},
             {"at": "callbacks first", "kind": "tag", "text": "Callbacks first", "pos": "left"}]
Kinds and their motion live in remotion/src/Popups.tsx. The layer renders as transparent ProRes 4444
and reels_week.py overlays it under the captions.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import reel_captions

REMOTION = Path(__file__).resolve().parents[2] / "remotion"
KINDS = {"stat", "tag", "list", "image"}
DEFAULT_SECONDS = 1.7
PASSTHROUGH = ("text", "label", "value", "prefix", "suffix", "n", "strike", "image", "pos")


NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "ten": "10", "sixteen": "16"}


def _norm(word):
    # Whisper writes "2" or "two" depending on context; cues match either way.
    token = re.sub(r"[^a-z0-9']", "", str(word).lower())
    return NUMBER_WORDS.get(token, token)


def find_phrase(words, phrase):
    """Index of the first kept word where `phrase` starts, or None."""
    want = [t for t in (_norm(w) for w in str(phrase).split()) if t]
    have = [_norm(w["word"]) for w in words]
    for i in range(len(have) - len(want) + 1):
        if want and have[i:i + len(want)] == want:
            return i
    return None


def resolve(cues, words, ranges, duration):
    """Cues -> pop-ups on the cut timeline, plus notes for cues whose phrase was never said."""
    popups, notes = [], []
    for cue in cues or []:
        kind = cue.get("kind")
        if kind not in KINDS:
            notes.append(f"pop-up kind {kind!r} unknown (use {', '.join(sorted(KINDS))})")
            continue
        i = find_phrase(words, cue.get("at", ""))
        if i is None:
            notes.append(f"pop-up skipped: {cue.get('at')!r} was not said in the take")
            continue
        start = reel_captions.to_output(float(words[i]["start"]), ranges)
        if start is None:
            notes.append(f"pop-up skipped: {cue.get('at')!r} falls inside a cut")
            continue
        end = min(duration, start + float(cue.get("seconds", DEFAULT_SECONDS)))
        if end - start < 0.4:
            notes.append(f"pop-up skipped: {cue.get('at')!r} is too close to the end")
            continue
        item = {"start": round(start, 3), "end": round(end, 3), "kind": kind}
        item.update({k: cue[k] for k in PASSTHROUGH if k in cue})
        popups.append(item)
    popups.sort(key=lambda p: p["start"])
    for left, right in zip(popups, popups[1:]):  # one card at a time
        left["end"] = min(left["end"], right["start"])
    return [p for p in popups if p["end"] - p["start"] >= 0.4], notes


def render(popups, duration, out_mov, work):
    """Render the transparent pop-up layer with Remotion. Raises RuntimeError with the log path on failure."""
    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError("npx is not on PATH. Install Node.js (nodejs.org), then retry.")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    props = work / "popups-props.json"
    props.write_text(json.dumps({"durationInSeconds": round(duration, 3), "popups": popups}, indent=1), encoding="utf-8")
    log = work / "popups-render.log"
    cmd = [npx, "remotion", "render", "src/index.ts", "Popups", str(Path(out_mov).resolve()), f"--props={props.resolve()}",
           "--codec=prores", "--prores-profile=4444", "--pixel-format=yuva444p10le", "--image-format=png", "--log=error"]
    result = subprocess.run(cmd, cwd=REMOTION, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode or not Path(out_mov).is_file():
        raise RuntimeError(f"Remotion pop-up render failed. See {log}.")
    return Path(out_mov)
