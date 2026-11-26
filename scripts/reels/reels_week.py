"""Cut a week of slated footage into vertical reels.

python scripts/reels/reels_week.py output/reels/week-2026-10-24
python scripts/reels/reels_week.py output/reels/week-2026-10-24 --only 3 17

Reads <week>/raw and <week>/cards.json. Writes <week>/final/reel-NN.mp4
and <week>/final/REPORT.md. See scripts/reels/README.md.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import edit_reel
import reel_captions
import reel_popups
import transcribe as transcribe_mod
from PIL import ImageFont

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
REEL_TOKENS = {"reel", "real", "reels", "riel", "rell"}
TAKE_TOKENS = {"take", "tape", "takes"}
NOISE = {"uh", "um", "uhm", "uhh", "umm", "ah", "er", "erm", "hmm", "mm", "mmm", "like", "number", "no"}
FILLERS = {"um", "uh", "uhh", "uhm", "umm", "erm", "er", "hmm", "mm", "mmm", "ah"}
ABORT_WORDS = {"cut", "wait", "sorry", "redo", "restart"}
ABORT_PHRASES = (
    ("start", "over"),
    ("let", "me"),
    ("my", "bad"),
    ("hold", "on"),
    ("hang", "on"),
    ("try", "again"),
    ("one", "more"),
)
ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
HOMOPHONES = {"to": 2, "too": 2, "for": 4, "won": 1, "ate": 8, "tree": 3}
LONG_PAUSE = 0.50
CONFIDENT = 0.50


class WeekError(Exception):
    """A problem the operator has to fix. The message is the whole report."""


def load_cards(path: Path) -> list:
    if not path.is_file():
        raise WeekError(
            f"missing cards file: {path}. Expected a JSON list of "
            '{"reel": N, "hook": "..."}. Example: scripts/reels/cards.example.json'
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise WeekError(
            f"cards.json is not valid JSON: {path} ({exc.msg} at line {exc.lineno})"
        ) from exc
    if not isinstance(data, list):
        raise WeekError(
            'cards.json must be a list of {"reel": N, "hook": "..."}, '
            f"got {type(data).__name__}: {path}"
        )
    if not data:
        raise WeekError(f"cards.json is empty: {path}")
    cards, seen = [], set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise WeekError(f"cards.json item {index} must be an object with reel and hook")
        extra = sorted(set(item) - {"reel", "hook", "popups"})
        if extra:
            raise WeekError(
                f"cards.json item {index} has unknown keys: {', '.join(extra)}. "
                'Use only "reel", "hook" and "popups".'
            )
        cues = item.get("popups", [])
        if not isinstance(cues, list) or any(
                not isinstance(c, dict) or not isinstance(c.get("at"), str) or c.get("kind") not in reel_popups.KINDS
                for c in cues):
            raise WeekError(
                f"cards.json item {index} popups must be a list of objects with \"at\" (a phrase you say) "
                f"and \"kind\" ({', '.join(sorted(reel_popups.KINDS))})"
            )
        if "reel" not in item or "hook" not in item:
            raise WeekError(f"cards.json item {index} needs reel and hook")
        reel, hook = item["reel"], item["hook"]
        if isinstance(reel, bool) or not isinstance(reel, int) or reel < 1:
            raise WeekError(f"cards.json item {index} reel must be a positive integer, got {reel!r}")
        if not isinstance(hook, str) or not hook.strip():
            raise WeekError(f"cards.json item {index} hook must be a non-empty string")
        if reel in seen:
            raise WeekError(f"cards.json has duplicate reel {reel}")
        seen.add(reel)
        card = {"reel": reel, "hook": " ".join(hook.split())}
        if cues:
            card["popups"] = cues
        cards.append(card)
    return cards


def list_raw_videos(raw: Path) -> list:
    if not raw.is_dir():
        raise WeekError(f"raw folder does not exist: {raw}")
    files = [path for path in raw.iterdir() if path.name.lower() not in IGNORED_NAMES and not path.name.startswith(".")]
    if not files:
        raise WeekError(f"raw folder is empty: {raw}")
    unknown = sorted(path.name for path in files if path.suffix.lower() not in VIDEO_EXT)
    if unknown:
        raise WeekError(
            "unrecognized file in raw/: " + ", ".join(unknown)
            + ". Expected video files (.mp4, .mov, .m4v, .mkv, .webm) only."
        )
    videos = sorted((path for path in files if path.suffix.lower() in VIDEO_EXT), key=lambda path: path.name.lower())
    if not videos:
        raise WeekError(f"raw folder has no video files: {raw}")
    return videos


def media_info(path: Path) -> dict:
    try:
        info = edit_reel.probe(path)
    except FileNotFoundError as exc:
        raise WeekError(f"ffprobe is not on PATH ({exc}). Install ffmpeg and retry.") from exc
    except RuntimeError as exc:
        raise WeekError(f"ffprobe failed on {path.name}: {exc}") from exc
    video = [stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"]
    audio = [stream for stream in info.get("streams", []) if stream.get("codec_type") == "audio"]
    if not video:
        raise WeekError(f"{path.name} has no video stream.")
    if not audio:
        raise WeekError(f"{path.name} has no audio stream.")
    try:
        float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WeekError(f"ffprobe did not report a duration for {path.name}.") from exc
    return info


def _validate_words(words, path: Path) -> list:
    if not isinstance(words, list):
        raise WeekError(f"transcript is not a list of words: {path}")
    for index, word in enumerate(words):
        if not isinstance(word, dict) or not {"word", "start", "end"} <= set(word):
            raise WeekError(f"transcript word {index} is missing word/start/end: {path}")
        try:
            start, end = float(word["start"]), float(word["end"])
        except (TypeError, ValueError) as exc:
            raise WeekError(f"transcript word {index} has non-numeric times: {path}") from exc
        if end < start:
            raise WeekError(f"transcript word {index} ends before it starts: {path}")
    return words


def transcript_for(video: Path, model_name: str, model) -> list:
    words_path = video.parent.parent / "analysis" / f"{video.stem}.words.json"
    if words_path.is_file() and words_path.stat().st_mtime >= video.stat().st_mtime:
        try:
            words = json.loads(words_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WeekError(
                f"transcript is not valid JSON: {words_path} ({exc.msg} at line {exc.lineno}). "
                "Delete it to transcribe again."
            ) from exc
        print(f"reusing transcript {words_path}", flush=True)
        return _validate_words(words, words_path)
    print(f"transcribing {video.name}", flush=True)
    try:
        text_path = transcribe_mod.transcribe(video, model_name=model_name, model=model)
    except FileNotFoundError as exc:
        raise WeekError(f"ffmpeg is not on PATH ({exc}). Install ffmpeg and retry.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr or ""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise WeekError(f"ffmpeg failed transcribing {video.name}: {detail[-2000:] or exc}") from exc
    except WeekError:
        raise
    except Exception as exc:
        raise WeekError(f"transcription failed for {video.name}: {exc}") from exc
    written = text_path.with_name(f"{video.stem}.words.json")
    if not written.is_file():
        raise WeekError(f"transcription did not write {written}")
    try:
        words = json.loads(written.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WeekError(f"transcript is not valid JSON: {written} ({exc.msg})") from exc
    words = _validate_words(words, written)
    if not words:
        raise WeekError(f"no spoken words transcribed in {video.name}. Check the audio.")
    return words


def load_model(model_name: str):
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise WeekError(
            "faster-whisper is not installed. Create .venv-reels and "
            "pip install -r scripts/reels/requirements.txt. See scripts/reels/README.md. "
            f"({exc})"
        ) from exc
    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as exc:
        raise WeekError(f"could not load faster-whisper model {model_name!r}: {exc}") from exc


def parse_number(tokens, index):
    if index >= len(tokens):
        return None
    token = tokens[index]
    if token.isdigit():
        value = int(token)
        if value < 1 or value > 99:
            return None
        return value, index + 1
    for ten_word, ten_value in TENS.items():
        if token.startswith(ten_word) and len(token) > len(ten_word) and token[len(ten_word):] in ONES:
            return ten_value + ONES[token[len(ten_word):]], index + 1
    if token in TENS:
        if index + 1 < len(tokens) and tokens[index + 1] in ONES:
            return TENS[token] + ONES[tokens[index + 1]], index + 2
        return TENS[token], index + 1
    if token in TEENS:
        return TEENS[token], index + 1
    if token in ONES:
        return ONES[token], index + 1
    if token in HOMOPHONES:
        return HOMOPHONES[token], index + 1
    return None


def skip_noise(tokens, index):
    skipped = 0
    while index < len(tokens) and skipped < 4 and (tokens[index] == "" or tokens[index] in NOISE):
        index += 1
        skipped += 1
    return index


def _probability(word) -> float:
    try:
        return float(word.get("probability", 0))
    except (TypeError, ValueError):
        return 0.0


def match_slate(tokens, words, index):
    if tokens[index] not in REEL_TOKENS:
        return None
    structural = [index]
    number_start = skip_noise(tokens, index + 1)
    parsed = parse_number(tokens, number_start)
    if not parsed:
        return None
    reel, after_reel = parsed
    structural.extend(range(number_start, after_reel))
    take_at = skip_noise(tokens, after_reel)
    if take_at >= len(tokens) or tokens[take_at] not in TAKE_TOKENS:
        return None
    structural.append(take_at)
    number_start = skip_noise(tokens, take_at + 1)
    parsed = parse_number(tokens, number_start)
    if not parsed:
        return None
    take, end = parsed
    structural.extend(range(number_start, end))
    uncertain = any(_probability(words[at]) < CONFIDENT for at in structural)
    return {
        "reel": reel,
        "take": take,
        "start_index": index,
        "end_index": end,
        "uncertain": uncertain,
    }


def _content_tokens(words):
    return [edit_reel.token(word["word"]) for word in words if edit_reel.token(word["word"])]


def ends_with_abort(words) -> bool:
    tokens = _content_tokens(words)
    tail = tokens[-6:]
    if any(token in ABORT_WORDS for token in tail):
        return True
    for first, second in ABORT_PHRASES:
        for index in range(len(tail) - 1):
            if tail[index] == first and tail[index + 1] == second:
                return True
    return False


def classify_take(content, slate_uncertain: bool) -> str:
    tokens = _content_tokens(content)
    span = 0.0 if not content else max(0.0, float(content[-1]["end"]) - float(content[0]["start"]))
    aborted = ends_with_abort(content)
    long = len(tokens) >= 10 and span >= 4.0
    tiny = len(tokens) < 4 or span < 1.2
    if tiny or (aborted and not long):
        kind = "aborted"
    elif aborted or not long or slate_uncertain:
        kind = "unsure"
    else:
        kind = "clean"
    return kind


def find_takes(words, file_index: int, source: str) -> list:
    tokens = [edit_reel.token(word["word"]) for word in words]
    matches, index = [], 0
    while index < len(tokens):
        if tokens[index] not in REEL_TOKENS:
            index += 1
            continue
        match = match_slate(tokens, words, index)
        if match is None:
            index += 1
            continue
        matches.append(match)
        index = match["end_index"]
    takes = []
    for number, match in enumerate(matches):
        next_index = matches[number + 1]["start_index"] if number + 1 < len(matches) else len(words)
        content = words[match["end_index"]:next_index]
        end_limit = float(words[next_index]["start"]) if number + 1 < len(matches) else None
        takes.append({
            "reel": match["reel"],
            "take": match["take"],
            "kind": classify_take(content, match["uncertain"]),
            "file_index": file_index,
            "source": source,
            "start": float(words[match["start_index"]]["start"]),
            "content": content,
            "end_limit": end_limit,
            "uncertain": match["uncertain"],
        })
    return takes


def choose_take(takes):
    ordered = sorted(takes, key=lambda take: (take["take"], take["file_index"], take["start"]))
    clean = [take for take in ordered if take["kind"] == "clean"]
    unsure = [take for take in ordered if take["kind"] == "unsure"]
    if clean:
        chosen, status = clean[-1], "ok"
    elif unsure:
        chosen, status = unsure[-1], "check it"
    elif ordered:
        chosen, status = ordered[-1], "check it"
    else:
        return None, "missing", 'No "Reel N, take M" slate in raw/.'
    notes = []
    if status == "check it" and chosen["kind"] == "aborted":
        notes.append("Every take looks aborted. Cut the last one anyway.")
    elif status == "check it":
        notes.append("Take may be unfinished or the slate was uncertain.")
    later = ordered[ordered.index(chosen) + 1:]
    if later:
        shown = ", ".join(f"take {take['take']} ({take['kind']})" for take in later)
        notes.append(f"Skipped later {shown}.")
    return chosen, status, " ".join(notes)


def boundary_like(words, index) -> bool:
    if edit_reel.token(words[index]["word"]) != "like":
        return False
    if index == 0 or index == len(words) - 1:
        return True
    before = float(words[index]["start"]) - float(words[index - 1]["end"])
    after = float(words[index + 1]["start"]) - float(words[index]["end"])
    return before >= 0.35 or after >= 0.35


def drop_fillers(words):
    kept, removed = [], []
    for index, word in enumerate(words):
        token = edit_reel.token(word["word"])
        if not token or token in FILLERS or boundary_like(words, index):
            removed.append(word)
            continue
        kept.append(word)
    return kept, removed


def keep_ranges(words, removed, duration, quiet):
    if not words:
        raise WeekError("no words left after removing the slate and fillers")
    notes = []
    leading = [word for word in removed if float(word["end"]) <= float(words[0]["start"]) + 0.02]
    cursor = max(0.0, float(words[0]["start"]) - 0.12)
    if leading:
        cursor = max(cursor, float(leading[-1]["end"]) + 0.02)
    ranges = []
    for left, right in zip(words, words[1:]):
        gap = float(right["start"]) - float(left["end"])
        extras = [word for word in removed
                  if float(left["end"]) - 0.02 <= float(word["start"]) < float(right["start"])]
        if gap <= LONG_PAUSE and not extras:
            continue
        if extras:
            lo = float(left["end"]) + 0.04
            hi = float(right["start"]) - 0.04
        else:
            lo, hi = float(left["end"]) + 0.08, float(right["start"]) - 0.05
            candidates = [start for start, end in quiet if end > lo and float(left["start"]) <= start < hi]
            if candidates:
                lo = max(lo, candidates[0] + 0.04)
        if hi <= lo:
            notes.append(
                f"Could not safely cut the gap between {left['word']!r} and {right['word']!r}."
            )
            continue
        ranges.append([round(cursor, 6), round(lo, 6)])
        cursor = hi
    end = min(float(duration), float(words[-1]["end"]) + 0.40)
    trailing = [word for word in removed if float(word["start"]) >= float(words[-1]["end"]) - 0.02]
    if trailing:
        end = min(end, float(trailing[0]["start"]) - 0.02)
    if end <= cursor:
        raise WeekError("keep range collapsed to nothing")
    ranges.append([round(cursor, 6), round(end, 6)])
    ranges = [[start, stop] for start, stop in ranges if stop - start >= 0.05]
    if not ranges:
        raise WeekError("keep range collapsed to nothing")
    return ranges, notes


def layout_hook(hook: str, font_path: Path):
    hook = " ".join(hook.replace("\n", " ").split())
    if not hook:
        raise WeekError("hook is empty")
    words = hook.split()
    for size in range(72, 35, -2):  # hook text caps at 72 px
        font = ImageFont.truetype(str(font_path), size)
        if any(font.getlength(word) > 880 for word in words):
            continue
        lines, current = [], ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if font.getlength(trial) <= 880:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) <= 4:
            return lines, size
    raise WeekError(f"hook does not fit in 4 lines: {hook!r}. Shorten the card hook.")


def render_portrait(source: Path, ranges, lines, font_size: int, out_mp4: Path, work: Path,
                    caption_words=None, popups=None, hook_until: float | None = None) -> float:
    work.mkdir(parents=True, exist_ok=True)
    try:
        font_path = edit_reel.resolve_font()
        edit_reel.burn_in_image(work / "hook.png", lines, font_path, font_size=font_size)
    except FileNotFoundError as exc:
        raise WeekError(str(exc)) from exc
    except ValueError as exc:
        raise WeekError(str(exc)) from exc
    graph = edit_reel.audio_graph(ranges)
    pass1 = graph + ";[edited]loudnorm=I=-14:TP=-1:LRA=11:print_format=json[normalized]"
    try:
        first = edit_reel.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", source, "-filter_complex",
             pass1, "-map", "[normalized]", "-f", "null", "-"],
            work / "loudnorm-pass1.log",
        )
    except FileNotFoundError as exc:
        raise WeekError(f"ffmpeg is not on PATH ({exc}). Install ffmpeg and retry.") from exc
    except RuntimeError as exc:
        raise WeekError(f"ffmpeg loudnorm failed for {out_mp4.name}: {exc}") from exc
    matches = re.findall(r'\{\s*"input_i".*?\}', first.stderr, re.S)
    if not matches:
        raise WeekError(
            f"ffmpeg loudnorm did not report levels for {out_mp4.name}. "
            f"See {work / 'loudnorm-pass1.log'}."
        )
    measured = json.loads(matches[-1])
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    missing = [key for key in required if key not in measured]
    if missing:
        raise WeekError(f"ffmpeg loudnorm result for {out_mp4.name} is missing {', '.join(missing)}.")
    norm = ("loudnorm=I=-14:TP=-1:LRA=11:linear=true:print_format=json:"
            f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}")
    select = "+".join(f"gte(t,{start})*lt(t,{stop})" for start, stop in ranges)
    shift = str(ranges[0][0]) + "".join(
        f"+gte(PTS*TB,{ranges[index][0]})*{ranges[index][0] - ranges[index - 1][1]:.6f}"
        for index in range(1, len(ranges)))
    frames = max(1, math.ceil(sum(stop - start for start, stop in ranges) * 30 - 1e-6))
    # Layers over the cut video, bottom to top: hook block, pop-ups, captions.
    layer_inputs, layer_labels = [], []
    if popups:
        try:
            mov = reel_popups.render(popups, frames / 30, work / "popups.mov", work)
        except RuntimeError as exc:
            raise WeekError(f"{out_mp4.name}: {exc}") from exc
        layer_inputs += ["-i", mov]
        layer_labels.append("pop")
    if caption_words:
        caps = reel_captions.track(caption_words, ranges)
        (work / "captions.json").write_text(json.dumps(caps, indent=1), encoding="utf-8")
        if caps:
            try:
                concat = reel_captions.write_concat(caps, work / "captions", frames / 30)
            except ValueError as exc:
                raise WeekError(f"{out_mp4.name}: {exc}") from exc
            layer_inputs += ["-f", "concat", "-safe", "0", "-i", concat]
            layer_labels.append("cap")
    # hook_until: hide the hook block from that second on (e.g. so a text-behind word can take its place)
    hook_enable = f":enable='lt(t,{hook_until:.3f})'" if hook_until is not None else ""
    chain, current = [f"[base][1:v]overlay=0:0:format=auto{hook_enable}[v0]"], "v0"
    for offset, label in enumerate(layer_labels):
        chain.append(f"[{offset + 2}:v]fps=30,format=rgba[{label}]")
        chain.append(f"[{current}][{label}]overlay=0:0:eof_action=pass:format=auto[v{offset + 1}]")
        current = f"v{offset + 1}"
    caption_inputs = layer_inputs
    caption_filter = ";".join(chain) + f";[{current}]format=yuv420p[vout]"
    filters = (
        graph + f";[edited]{norm},volume=-0.3dB,aresample=48000[aout];"
        f"[0:v]select='{select}',setpts='PTS-({shift})/TB',"
        "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=1080:1920,setsar=1,fps=30:start_time=0,"
        f"tpad=stop_mode=clone:stop_duration=0.04,trim=end_frame={frames}[base];"
        + caption_filter
    )
    (work / "render-filter.txt").write_text(filters, encoding="utf-8")
    (work / "cuts.json").write_text(json.dumps({
        "source": str(source), "keep_ranges_seconds": ranges, "frames": frames,
    }, indent=2), encoding="utf-8")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        edit_reel.run(
            ["ffmpeg", "-y", "-hide_banner", "-stats", "-stats_period", "5", "-threads", "4",
             "-i", source, "-i", work / "hook.png", *caption_inputs, "-filter_complex_threads", "2",
             "-filter_complex", filters, "-map", "[vout]", "-map", "[aout]", "-map_metadata", "-1",
             "-c:v", "libx264", "-threads", "4", "-preset", "veryfast", "-crf", "18",
             "-profile:v", "high", "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
             "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
             "-color_range", "tv", "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
             "-movflags", "+faststart", out_mp4],
            work / "render.log",
        )
    except FileNotFoundError as exc:
        raise WeekError(f"ffmpeg is not on PATH ({exc}). Install ffmpeg and retry.") from exc
    except RuntimeError as exc:
        if out_mp4.exists():
            out_mp4.unlink()
        raise WeekError(f"ffmpeg failed rendering {out_mp4.name}: {exc}") from exc
    info = media_info(out_mp4)
    video = next(stream for stream in info["streams"] if stream.get("codec_type") == "video")
    if (video.get("width"), video.get("height")) != (1080, 1920):
        raise WeekError(
            f"rendered {out_mp4.name} is {video.get('width')}x{video.get('height')}, expected 1080x1920."
        )
    if video.get("codec_name") != "h264":
        raise WeekError(
            f"rendered {out_mp4.name} video codec is {video.get('codec_name')}, expected h264."
        )
    return float(info["format"]["duration"])


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def load_prior(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WeekError(
            f"existing report is not valid JSON: {path} ({exc.msg} at line {exc.lineno}). "
            "Delete final/report.json to start over."
        ) from exc
    if not isinstance(data, list):
        raise WeekError(f"existing report is not a list: {path}")
    rows = {}
    for item in data:
        if not isinstance(item, dict) or "reel" not in item:
            raise WeekError(f"existing report has a row without a reel number: {path}")
        rows[int(item["reel"])] = item
    return rows


def write_report(week: Path, rows: list, dry_run: bool, only) -> Path:
    final_dir = week / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "check it": 0, "missing": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = f"{counts['ok']} ok, {counts['check it']} check it, {counts['missing']} missing."
    lines = ["# Week reel report", ""]
    if dry_run:
        lines.append("Dry run: no mp4s written. Durations are planned from the transcript.")
        lines.append("")
    if only:
        listed = ", ".join(str(number) for number in only)
        lines.append(f"This run recut reels {listed}. Other rows were left as they were.")
        lines.append("")
    lines.append(f"Week: `{week}`")
    lines.append("")
    lines.append("| Reel | Status | Source | Take | Duration | Hook | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append("| " + " | ".join(_cell(row[key]) for key in
                     ("reel", "status", "source", "take", "duration", "hook", "notes")) + " |")
    lines.extend(["", summary, ""])
    report = final_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (final_dir / "report.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)
    print(f"wrote {report}", flush=True)
    return report


def _blank_row(card, status, notes):
    return {
        "reel": card["reel"],
        "status": status,
        "source": "",
        "take": None,
        "duration": "",
        "hook": card["hook"],
        "notes": notes,
    }


def _unprocessed_row(card, prior, final_dir: Path):
    previous = prior.get(card["reel"])
    mp4 = final_dir / f"reel-{card['reel']:02d}.mp4"
    if previous and mp4.is_file():
        row = dict(previous)
        row["hook"] = card["hook"]
        row["reel"] = card["reel"]
        extra = "Left unchanged by --only."
        row["notes"] = (str(row.get("notes") or "") + " " + extra).strip()
        return row
    if mp4.is_file():
        return _blank_row(
            card, "check it",
            "An mp4 exists but this run did not recut it and there is no final/report.json entry.",
        )
    return _blank_row(card, "missing", "Not rendered. This reel was not included in --only.")


def _quiet_for(path: Path, cache: dict):
    if path not in cache:
        try:
            cache[path] = edit_reel.silences(path, -38)
        except FileNotFoundError as exc:
            raise WeekError(f"ffmpeg is not on PATH ({exc}). Install ffmpeg and retry.") from exc
        except RuntimeError as exc:
            raise WeekError(f"ffmpeg silence scan failed on {path.name}: {exc}") from exc
    return cache[path]


def run_week(args) -> None:
    week = args.week
    if not week.exists():
        raise WeekError(f"week folder does not exist: {week}")
    if not week.is_dir():
        raise WeekError(f"week folder is not a directory: {week}")
    cards = load_cards(week / "cards.json")
    card_map = {card["reel"]: card for card in cards}
    only = list(args.only) if args.only else None
    if only:
        bad = [number for number in only if number < 1]
        if bad:
            raise WeekError(f"--only reel must be a positive integer, got {bad[0]}")
        duplicates = sorted({number for number in only if only.count(number) > 1})
        if duplicates:
            raise WeekError("--only has duplicate reel numbers: " + ", ".join(str(number) for number in duplicates))
        unknown = [number for number in only if number not in card_map]
        if unknown:
            known = ", ".join(str(card["reel"]) for card in cards)
            raise WeekError(
                "--only " + " ".join(str(number) for number in unknown)
                + f" is not in cards.json (reels: {known})"
            )
    videos = list_raw_videos(week / "raw")
    for video in videos:
        media_info(video)
    needs_model = False
    for video in videos:
        cached = video.parent.parent / "analysis" / f"{video.stem}.words.json"
        if not (cached.is_file() and cached.stat().st_mtime >= video.stat().st_mtime):
            needs_model = True
            break
    model = load_model(args.model) if needs_model else None
    transcripts = [(video, transcript_for(video, args.model, model)) for video in videos]
    takes = []
    for index, (video, words) in enumerate(transcripts):
        if not words:
            raise WeekError(f"no spoken words transcribed in {video.name}. Check the audio.")
        takes.extend(find_takes(words, index, f"raw/{video.name}"))
    if not takes:
        raise WeekError(
            'no "Reel N, take M" slate found in raw footage. '
            "Say the slate, pause, then the hook. "
            f"Transcripts are in {week / 'analysis'}."
        )
    extras = sorted({take["reel"] for take in takes} - set(card_map))
    if extras:
        raise WeekError(
            "raw footage has slates for reel(s) "
            + ", ".join(str(number) for number in extras)
            + " but cards.json has no matching card. Add the card or the slate is wrong."
        )
    try:
        font_path = edit_reel.resolve_font()
    except FileNotFoundError as exc:
        raise WeekError(str(exc)) from exc
    final_dir = week / "final"
    prior = load_prior(final_dir / "report.json") if only else {}
    quiet_cache = {}
    rows = []
    for card in cards:
        if only and card["reel"] not in only:
            rows.append(_unprocessed_row(card, prior, final_dir))
            continue
        group = [take for take in takes if take["reel"] == card["reel"]]
        chosen, status, notes = choose_take(group)
        if chosen is None:
            row = _blank_row(card, "missing", notes)
            rows.append(row)
            print(f"reel {card['reel']:02d} missing", flush=True)
            continue
        source = week / chosen["source"]
        kept, removed = drop_fillers(chosen["content"])
        if not kept:
            raise WeekError(
                f"reel {card['reel']} take {chosen['take']} in {chosen['source']} "
                "has no speech after the slate and fillers were removed."
            )
        info = media_info(source)
        limit = float(info["format"]["duration"])
        if chosen["end_limit"] is not None:
            limit = min(limit, chosen["end_limit"] - 0.05)
        try:
            ranges, pause_notes = keep_ranges(kept, removed, limit, _quiet_for(source, quiet_cache))
        except WeekError as exc:
            raise WeekError(f"reel {card['reel']} take {chosen['take']} in {chosen['source']}: {exc}") from exc
        notes = " ".join(part for part in (notes, " ".join(pause_notes)) if part)
        lines, font_size = layout_hook(card["hook"], font_path)
        out_mp4 = final_dir / f"reel-{card['reel']:02d}.mp4"
        planned = sum(stop - start for start, stop in ranges)
        popups, popup_notes = ([], []) if args.no_popups else reel_popups.resolve(card.get("popups"), kept, ranges, planned)
        notes = " ".join(part for part in (notes, " ".join(popup_notes)) if part)
        if args.dry_run:
            duration = f"{sum(stop - start for start, stop in ranges):.2f}s planned"
        else:
            print(f"rendering reel {card['reel']:02d} from {chosen['source']} take {chosen['take']}", flush=True)
            seconds = render_portrait(
                source, ranges, lines, font_size, out_mp4, final_dir / "work" / f"reel-{card['reel']:02d}",
                caption_words=None if args.no_captions else kept,
                popups=popups,
            )
            duration = f"{seconds:.2f}s"
        row = {
            "reel": card["reel"],
            "status": status,
            "source": chosen["source"],
            "take": chosen["take"],
            "duration": duration,
            "hook": card["hook"],
            "notes": notes,
        }
        rows.append(row)
        print(f"reel {card['reel']:02d} {status} {chosen['source']} take {chosen['take']} {duration}", flush=True)
    write_report(week, rows, args.dry_run, only)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("week", type=Path, help="Week folder, for example output/reels/week-2026-10-24")
    parser.add_argument("--only", nargs="+", type=int, metavar="N",
                        help="Recut only these reel numbers. Other finished reels are left in place.")
    parser.add_argument("--model", default="small.en",
                        help="faster-whisper model name (default: small.en)")
    parser.add_argument("--no-popups", action="store_true",
                        help="Skip the Remotion pop-up cards even when a card lists them")
    parser.add_argument("--no-captions", action="store_true",
                        help="Skip the burned-in word captions (TikTok Sans, 2-3 words at a time)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Transcribe, choose takes, and write REPORT.md without rendering mp4s")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        run_week(parse_args(argv))
    except WeekError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
