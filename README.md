# reel-cutter

The scripts I use to cut my talking-head reels with AI. I film on my phone, pause about a second between lines, and these cut the pauses, burn in the hook and captions, mute lip smacks, check the joins and drop in full-screen cards.

Plain Python plus ffmpeg. No browser, no paid API. Transcription runs locally with faster-whisper.

## Setup

You need Python 3.10+ and ffmpeg / ffprobe on your PATH (`ffmpeg -version`).

```
python -m venv .venv-reels
.venv-reels/Scripts/python -m pip install -r scripts/reels/requirements.txt     # Windows
.venv-reels/bin/python -m pip install -r scripts/reels/requirements.txt         # Mac / Linux
```

The first run downloads a small Whisper model.

## What's in here

| Script | What it does |
| --- | --- |
| `transcribe.py VIDEO` | Word-level timestamps for a take, saved as JSON. |
| `reels_week.py FOLDER` | Cuts a folder of takes into vertical reels: trims pauses, picks the last clean take, burns in the hook and captions. See below. |
| `edit_reel.py` | Cut one take (or a mix of sentences from several takes) from its words file. |
| `reel_captions.py` | Burned-in captions, 2-3 words at a time, kept clear of the Reels UI. Used by the cutter. |
| `find_smacks.py VIDEO --fix OUT` | Finds and mutes lip smacks in the pauses. Run it until it reports none. |
| `check_cut.py` | Read-only checks: mic dropouts in the raw take, picture jumps at each join, repeated starts the transcript smoothed over. |
| `precards.py` | Builds full-screen cards from a JSON spec, then places each one where you say its anchor phrase. |
| `breakdown_ref.py VIDEO --out DIR` | Breaks down a reference reel: transcript, cut times, contact sheets. Borrow the format, not the words. |

## Cutting a folder of takes

```
my-week/
  cards.json      [{"reel": 1, "hook": "Your hook text"}]
  raw/            your phone takes
```

Start each take by saying the slate, "Reel 1, take 1", then pause and go. A flubbed take gets the same slate with the next take number, and the cutter keeps the last clean one.

```
.venv-reels/Scripts/python scripts/reels/reels_week.py my-week --no-popups
```

Finished files land in `my-week/final/` with a `REPORT.md`. `--dry-run` transcribes and writes the report without rendering.

`--no-popups`: the animated pop-ups need a Remotion project that isn't included here, so leave them off.

## Checks I run before calling a cut done

1. `find_smacks.py` until it finds none.
2. `check_cut.py jumps` and `repeats` on the edit.
3. Watch it with sound. The checks point at places to look; they don't listen for you.

## Credits

`check_cut.py` is adapted from Liam Johnston's [ai-video-editor](https://github.com/Liamrjohnston/ai-video-editor) (MIT). Fonts are under the SIL Open Font License. See `THIRD_PARTY_NOTICES.md`.

MIT licensed. Use it, change it, ship it.
