"""Transcribe a reel take locally with word timestamps (free, no API).

Run with the reels venv:
  .venv-reels/Scripts/python scripts/reels/transcribe.py output/reels/<date>/raw/take-01.mov
Writes <dir>/../analysis/<name>.words.json and <name>.txt.
"""
import json
import io
import subprocess
import sys
from pathlib import Path


def transcribe(video: Path, model_name: str = "small.en", model=None) -> Path:
    from faster_whisper import WhisperModel

    out_dir = video.parent.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pipe the small mono WAV in memory: Windows sandbox tokens can lose access
    # to TemporaryDirectory's mode-0700 directory immediately after creation.
    wav = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video),
                          "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                         stdout=subprocess.PIPE, check=True).stdout
    if model is None:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(io.BytesIO(wav), word_timestamps=True, vad_filter=False)
    words, lines = [], []
    for seg in segments:
        lines.append(f"[{seg.start:6.2f}-{seg.end:6.2f}] {seg.text.strip()}")
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": round(w.start, 3),
                          "end": round(w.end, 3), "probability": round(w.probability, 3)})
    (out_dir / f"{video.stem}.words.json").write_text(json.dumps(words, indent=1), encoding="utf-8")
    (out_dir / f"{video.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return out_dir / f"{video.stem}.txt"


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        txt = transcribe(Path(arg))
        print(txt.read_text(encoding="utf-8"))
