"""Three read-only checks for a reel cut. Nothing here edits or overwrites any file.

  dropouts SOURCE.mp4                 wireless-mic dropouts in the raw take (before cutting)
  jumps    EDITED.mp4 PLAN.json       picture jump at every pause trim (join) in the edit
  repeats  EDITED.mp4                 duplicated starts the first transcript may have smoothed over

Run with the reels venv (numpy + faster-whisper):
  .venv-reels/Scripts/python scripts/reels/check_cut.py dropouts output/reels/<date>/raw/take.mp4
  .venv-reels/Scripts/python scripts/reels/check_cut.py jumps output/reels/<date>/edited/x.mp4 output/reels/<date>/analysis/x-plan.json
  .venv-reels/Scripts/python scripts/reels/check_cut.py repeats output/reels/<date>/edited/x.mp4

Ideas and the dropout / jump measurements adapted from Liam Johnston's ai-video-editor
(https://github.com/Liamrjohnston/ai-video-editor, MIT License, Copyright (c) 2026 Liam Johnston),
rewritten for this pipeline: the plan format is {"ranges": [[start_s, end_s], ...]} in source seconds,
and the edit is a single face-only 9:16 take with a hook line at the top.

An automated pass is not listening. Say what was measured and what was not.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------- mic dropouts (raw take) ----------
SR = 48000
WIN = 15          # ms window for the straight-line fit
RESID_DB = 0.8    # concealment is smoother than this; real speech is ragged (2-4 dB)
STEP_DB = 8.0     # abrupt fall / return size
FLOOR_DB = -75.0  # ignore digital silence


def load_audio(path):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "2",
                                   "-ar", str(SR), "-f", "f32le", "-"])
    return np.frombuffer(raw, "<f4").reshape(-1, 2)


def envelope_db(x):
    blk = SR // 1000
    a = np.abs(x).max(axis=1)
    n = len(a) // blk
    return 20 * np.log10(a[:n * blk].reshape(n, blk).max(axis=1) + 1e-9)


def fit_residual(e):
    n = len(e)
    res = np.full(n, 99.0)
    if n < WIN + 1:
        return res
    t = np.arange(WIN, dtype=float)
    A = np.stack([t, np.ones(WIN)], axis=1)
    proj = np.eye(WIN) - A @ np.linalg.pinv(A)
    segs = np.lib.stride_tricks.sliding_window_view(e, WIN)[: n - WIN]
    r = segs @ proj
    res[: n - WIN] = np.sqrt((r * r).mean(axis=1))
    return res


def find_dropouts(x):
    e = envelope_db(x)
    n = len(e)
    res = fit_residual(e)
    hits, i = [], 0
    while i < n - WIN:
        if res[i] < RESID_DB and e[i:i + WIN].max() > FLOOR_DB:
            j = i
            while j < n - WIN and res[j] < RESID_DB:
                j += 1
            a0, a1 = i, j + WIN
            if a0 + 1 < n and a1 - 2 >= 0 and a1 <= n:
                drop = e[max(0, a0 - 3):a0].max() - e[a0 + 1] if a0 > 0 else 0.0
                rise = e[a1:a1 + 3].max() - e[a1 - 2] if a1 < n else 0.0
                if drop >= STEP_DB and rise >= STEP_DB:
                    hits.append({"start_s": round(a0 / 1000, 3), "end_s": round(a1 / 1000, 3), "ms": int(a1 - a0),
                                 "fall_db": round(float(drop), 1), "return_db": round(float(rise), 1)})
            i = a1
        else:
            i += 1
    return hits, n / 1000.0


def cmd_dropouts(a):
    try:
        x = load_audio(a.source)
    except Exception as ex:  # noqa: BLE001
        print(f"could not read audio: {ex}", file=sys.stderr)
        return 2
    hits, dur = find_dropouts(x)
    for h in hits:
        print(f"dropout {h['start_s']:.3f}-{h['end_s']:.3f}s ({h['ms']} ms) fall {h['fall_db']} dB, return {h['return_db']} dB")
    print(f"{len(hits)} dropout(s) in {dur:.1f} s. Zero is a valid answer; listen to any hit before trusting it.")
    return 0


# ---------- picture jump at joins (edited video) ----------
TW, TH = 135, 240          # analysis size, 1/8 of 1080x1920
FACE_ROW0 = int(TH * 0.30)  # skip the hook text at the top; compare the face area and below
FLAG_AT, TARGET = 3.0, 2.0
# On near-still footage the ratio alone cries wolf: two joins on the 11-18 Larping edit read 4x but looked identical
# (absolute difference about 3.3 gray levels). A real head-position jump, built from two distant moments of the same
# take, measured 7.7. The floor is set between them; it comes from that one small sample, so revisit it with more edits.
ABS_FLOOR = 5.0


def probe_fps(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=r_frame_rate", "-of", "csv=p=0", str(path)]).decode().strip()
    n, d = out.split("/")
    return float(n) / float(d)


def load_frames(path):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                                   f"scale={TW}:{TH},format=gray", "-f", "rawvideo", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, TH, TW).astype(np.float32)


def cmd_jumps(a):
    try:
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        ranges = plan["ranges"]
        fps = probe_fps(a.video)
        X = load_frames(a.video)
    except Exception as ex:  # noqa: BLE001
        print(f"could not read inputs: {ex}", file=sys.stderr)
        return 2

    def D(k):
        return float(np.abs(X[k + 1, FACE_ROW0:] - X[k, FACE_ROW0:]).mean())

    cum, joins = 0.0, []
    for (s0, s1), nxt in zip(ranges, ranges[1:]):
        cum += s1 - s0
        j = int(round(cum * fps))
        if 1 <= j < len(X) - 1:
            ks = [k for k in range(j - 9, j + 8) if k != j - 1 and 0 <= k < len(X) - 1]
            base = max(float(np.median([D(k) for k in ks])), 0.05) if ks else 0.05
            joins.append({"out_s": round(j / fps, 2), "frame": j, "jump": round(D(j - 1) / base, 2),
                          "abs": round(D(j - 1), 1), "trimmed_s": round(nxt[0] - s1, 2)})
    for r in sorted(joins, key=lambda r: -r["jump"]):
        if r["jump"] >= FLAG_AT and r["abs"] >= ABS_FLOOR:
            mark = "FLAG"
        elif r["jump"] > TARGET:
            mark = "warn"
        else:
            mark = "ok"
        print(f"{mark:4} at {r['out_s']:6.2f}s (frame {r['frame']}) jump {r['jump']}x, {r['abs']} gray levels  [{r['trimmed_s']}s of pause removed here]")
    bad = sum(r["jump"] >= FLAG_AT and r["abs"] >= ABS_FLOOR for r in joins)
    print(f"{len(joins)} join(s) measured, {bad} flagged (>= {FLAG_AT}x and >= {ABS_FLOOR} gray levels; warn = ratio high but the picture barely changed). "
          "A flag means put back a few pause frames or cover it with a punch-in; a hard cut between sentences is normal.")
    return 1 if bad else 0


# ---------- duplicated starts (edited video, re-transcribed) ----------
def transcribe_words(path):
    from faster_whisper import WhisperModel
    wav = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "wav", "pipe:1"], stdout=subprocess.PIPE, check=True).stdout
    import io
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(io.BytesIO(wav), word_timestamps=True, language="en")
    words = []
    for s in segs:
        for w in s.words or []:
            words.append({"word": w.word.strip().lower().strip(".,!?;:\"'"), "start": w.start, "end": w.end})
    return words


def find_repeats(words, max_n=4):
    found = []
    for n in range(max_n, 0, -1):
        for i in range(len(words) - 2 * n + 1):
            a = [w["word"] for w in words[i:i + n]]
            b = [w["word"] for w in words[i + n:i + 2 * n]]
            if a == b and all(a):
                if n == 1 and a[0] in {"that", "had", "very", "really"}:
                    pass  # real English repeats still get listed below, flagged as weak
                found.append({"n": n, "text": " ".join(a), "at_s": round(words[i]["start"], 2),
                              "gap_s": round(words[i + n]["start"] - words[i + n - 1]["end"], 2)})
    # drop shorter repeats contained in a longer one at the same spot
    found.sort(key=lambda r: (r["at_s"], -r["n"]))
    out = []
    for r in found:
        if not any(o["at_s"] <= r["at_s"] < o["at_s"] + 2 * o["n"] * 0.6 and o["n"] > r["n"] for o in out):
            out.append(r)
    return out


def cmd_repeats(a):
    try:
        words = transcribe_words(a.video)
    except Exception as ex:  # noqa: BLE001
        print(f"could not transcribe: {ex}", file=sys.stderr)
        return 2
    reps = find_repeats(words)
    for r in reps:
        print(f"repeat at {r['at_s']:.2f}s: \"{r['text']} {r['text']}\" ({r['n']} word(s), {r['gap_s']}s apart)")
    print(f"{len(words)} words, {len(reps)} repeated sequence(s). Some are natural speech; listen to each at normal speed.")
    return 1 if reps else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dropouts"); s.add_argument("source"); s.set_defaults(fn=cmd_dropouts)
    s = sub.add_parser("jumps"); s.add_argument("video"); s.add_argument("plan"); s.set_defaults(fn=cmd_jumps)
    s = sub.add_parser("repeats"); s.add_argument("video"); s.set_defaults(fn=cmd_repeats)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
