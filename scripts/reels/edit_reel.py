"""Edit single takes or a sentence-pick mix; requires ffmpeg, ffprobe and Pillow.

python scripts/reels/edit_reel.py --take output/reels/2026-10-04/raw/take-02.mp4 \
    --words output/reels/2026-10-04/analysis/take-02.words.json \
    --out-dir output/reels/2026-10-04/final

Source files are read-only. Cuts use word timing with audio-silence guards for
Whisper's early ends and zero-duration words. No speed changes or transitions.

For a mix, use --mix picks.json --out-dir <final>. The JSON contains a picks
list: each pick has take and words paths (relative to the manifest), and
sentences as [first_word_index, exclusive_end_index] pairs. Optional reason
records an editorial swap. Mix artifacts have distinct names.
"""
import argparse
import array
import difflib
import json
import math
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SCRIPT = """You're not bad at sales. You just never check the door. Your brain would
rather keep the door shut than risk what's behind it, even if what's in front of
you is a life you don't want. So you say I'm not ready. I'm bad at talking to
people. And it stops feeling like belief. It starts feeling like a fact. I told
myself I wasn't a sales guy. Then I made 62 cold calls. The first ones were rough.
Now I'm not even nervous anymore. Belief drives action. Action creates proof.
Proof builds belief. You don't need to feel ready to start the loop. You just
need to try the handle."""
LINES = ["You're not bad at sales.", "You never checked the door."]
ROOT = Path(__file__).resolve().parents[2]
FONT = ROOT / "scripts/reels/fonts/Inter-Bold.ttf"


def run(args, log=None):
    if log:
        # Stream long render logs to disk so reruns can be monitored live.
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run([str(x) for x in args], stdout=subprocess.PIPE,
                                    stderr=stream, text=True, encoding="utf-8", errors="replace")
        result.stderr = log.read_text(encoding="utf-8", errors="replace")
    else:
        result = subprocess.run([str(x) for x in args], capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr[-8000:])
    return result


def probe(path):
    return json.loads(run(["ffprobe", "-v", "error", "-show_streams",
                           "-show_format", "-of", "json", path]).stdout)


def token(word):
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("\u2019", "'"))


def silences(take, db):
    result = run(["ffmpeg", "-hide_banner", "-nostats", "-i", take, "-vn",
                  "-af", f"silencedetect=noise={db}dB:d=0.08", "-f", "null", "-"])
    intervals, start = [], None
    for kind, value in re.findall(r"silence_(start|end): ([0-9.]+)", result.stderr):
        if kind == "start":
            start = float(value)
        elif start is not None:
            intervals.append((start, float(value)))
            start = None
    return intervals


def make_cuts(words, duration, quiet, onset_quiet):
    expected = [token(w) for w in SCRIPT.split()]
    actual = [token(w["word"]) for w in words]
    kept, removed = [], []
    # Align to the binding spoken script: grammatical "like" is retained;
    # extra fillers/repetitions/false starts are removed, never all "like"s.
    for op, a, b, c, d in difflib.SequenceMatcher(None, actual, expected,
                                                autojunk=False).get_opcodes():
        if op == "equal":
            kept.extend(words[a:b])
        elif op == "delete":
            removed.extend(words[a:b])
        else:
            raise ValueError(f"Script alignment needs review: {op}: {actual[a:b]} / {expected[c:d]}")
    exclusions, guards = [], []
    for left, right in zip(kept, kept[1:]):
        gap = right["start"] - left["end"]
        extras = [w for w in removed if left["end"] <= w["start"] < right["start"]]
        if gap <= 0.15 and not extras:
            continue
        lo, hi = left["end"] + 0.08, right["start"] - 0.05
        if not extras:
            # Keep trailing consonants when Whisper ended the previous word early.
            candidates = [(s, e) for s, e in quiet if e > lo and
                          left["start"] <= s < hi]
            if candidates:
                lo = max(lo, candidates[0][0] + 0.05)
            # A zero/near-zero word can actually begin well before its timestamp.
            if right["end"] - right["start"] <= 0.04:
                candidates = [(s, e) for s, e in onset_quiet
                              if e - s >= 0.1 and lo < e <= right["start"] + 0.08
                              and e >= right["start"] - 0.65]
                if candidates:
                    hi = min(hi, candidates[-1][1] - 0.05)
                if token(right["word"]) in {"it", "i", "a"}:
                    # Very short function words can be quieter than the silence
                    # threshold; a zero-length alignment is not a safe onset.
                    hi = min(hi, right["start"] - 0.25)
        else:
            lo = min(lo, extras[0]["start"])
            hi = max(hi, extras[-1]["end"])
        if hi <= lo:
            raise ValueError(f"Cannot safely cut boundary {left['word']} / {right['word']}")
        exclusions.append((lo, hi))
        guards.append({"after": left["word"], "before": right["word"],
                       "timestamp_gap": round(gap, 6),
                       "removed_start": round(lo, 6), "removed_end": round(hi, 6)})
    ranges, cursor = [], kept[0]["start"]
    for lo, hi in exclusions:
        ranges.append([round(cursor, 6), round(lo, 6)])
        cursor = hi
    end = min(duration, kept[-1]["end"] + 0.5)
    ranges.append([round(cursor, 6), round(end, 6)])
    return ranges, guards, removed


def resolve_font():
    """Bundled Inter Bold, then the legacy output-folder font. Never a surprise face."""
    bundled = Path(__file__).resolve().parent / "fonts" / "Inter-Bold.ttf"
    if bundled.is_file():
        return bundled
    if FONT.is_file():
        return FONT
    raise FileNotFoundError(
        "Hook font not found. Expected "
        f"{bundled} (bundled with scripts/reels) or {FONT} (legacy). "
        "See scripts/reels/README.md."
    )


def burn_in_image(path, lines, font_path=None, font_size=None, measure_line=None):
    # Instagram-native look: sentence case, white text,
    # thin black outline, no shadow, sitting just above the head. Drawn at 3x for clean edges.
    font_path = Path(font_path) if font_path else resolve_font()
    if not font_path.is_file():
        raise FileNotFoundError(
            f"Hook font not found: {font_path}. See scripts/reels/README.md."
        )
    lines = [str(line).strip() for line in lines]
    if not lines or any(not line for line in lines):
        raise ValueError("Hook text is empty.")
    k, width, top, stroke = 3, 880, 270, 5
    sizing = measure_line or max(lines, key=lambda line: ImageFont.truetype(str(font_path), 40).getlength(line))
    if font_size is None:
        size = 1
        while size < 160 and ImageFont.truetype(str(font_path), size + 1).getlength(sizing) <= width:
            size += 1
    else:
        size = int(font_size)
        if size < 1:
            raise ValueError(f"Hook font size must be positive, got {font_size}.")
    fitted = ImageFont.truetype(str(font_path), size)
    if fitted.getlength(sizing) > width or any(fitted.getlength(line) > width for line in lines):
        raise ValueError(
            f"Hook does not fit on screen at {size}px: {sizing!r}. Shorten the card hook."
        )
    font = ImageFont.truetype(str(font_path), size * k)
    canvas = Image.new("RGBA", (1080 * k, 1920 * k))
    draw = ImageDraw.Draw(canvas)
    # No block. White text with a dark outline, no box.
    # Text top sits at y 300 so it clears the Reels top bar (y < 288).
    top = 300
    boxes = [font.getbbox(line) for line in lines]
    gap = 12 * k
    y = top * k
    for line, box in zip(lines, boxes):
        x = (1080 * k - (box[2] - box[0])) / 2 - box[0]
        draw.text((x, y - box[1]), line, font=font, fill="white", stroke_width=stroke * k, stroke_fill="black")
        y += (box[3] - box[1]) + gap
    canvas = canvas.resize((1080, 1920), Image.Resampling.LANCZOS)
    canvas.save(path)
    return {"font_size": size, "top": top, "long_line_width": font.getlength(sizing) / k,
            "text_bbox": canvas.getbbox()}


def overlay(path):
    return burn_in_image(path, LINES, measure_line=LINES[1])


def audio_graph(ranges):
    parts = [f"[0:a]asplit={len(ranges)}" + "".join(f"[s{i}]" for i in range(len(ranges)))]
    for i, (start, end) in enumerate(ranges):
        parts.append(f"[s{i}]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.008,afade=t=out:st={end-start-0.008:.6f}:d=0.008[a{i}]")
    parts.append("".join(f"[a{i}]" for i in range(len(ranges))) +
                 f"concat=n={len(ranges)}:v=0:a=1[edited]")
    return ";".join(parts)


def contact_sheet(video, duration, out):
    sheet = Image.new("RGB", (1080, 1280), (20, 20, 20))
    times = [0.05 + i * (duration - 0.15) / 5 for i in range(6)]
    for i, moment in enumerate(times):
        frame = out / f"frame-{i+1:02d}.jpg"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{moment:.6f}",
             "-i", video, "-frames:v", "1", "-vf", "scale=360:640", "-q:v", "2", frame])
        with Image.open(frame) as im:
            sheet.paste(im, ((i % 3) * 360, (i // 3) * 640))
    sheet.save(out / "contact-sheet.jpg", quality=94)
    return times


def hiss_envelope(take):
    """20 ms high-pass (3.5 kHz) energy in dB. Trailing s/sh/f sounds are quiet and
    hissy, so silence detection and Whisper both end words before them (10-04: the
    hook's "sales" came out as "sale")."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(take), "-vn", "-ac", "1",
                          "-ar", "16000", "-af", "highpass=f=3500", "-f", "s16le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    samples = array.array("h", raw)
    n = 320
    env = [20 * math.log10(math.sqrt(sum(x * x for x in samples[i:i+n]) / n) + 1)
           for i in range(0, len(samples) - n, n)]
    floor = sorted(env)[len(env) // 5] if env else 0
    return {"db": env, "threshold": floor + 6}


def extend_for_hiss(end, limit, hiss):
    """Push a cut later while the high-frequency tail is still sounding."""
    if not hiss:
        return end
    env, thr = hiss["db"], hiss["threshold"]
    i = int(end / 0.02)
    quiet_run = 0
    while i < len(env) and (i + 1) * 0.02 <= limit:
        quiet_run = quiet_run + 1 if env[i] < thr else 0
        if quiet_run >= 2:
            return max(end, (i - 1) * 0.02 + 0.04)
        i += 1
    return max(end, min(limit, i * 0.02))


def mix_ranges(words, sentences, duration, quiet, onset_quiet, last=False, hiss=None):
    """Preserve word spans and pad every sentence, including within a pick."""
    ranges = []
    for first, stop in sentences:
        selected = words[first:stop]
        if not selected or not 0 <= first < stop <= len(words):
            raise ValueError(f"Invalid sentence word indices: {first}:{stop}")
        start = max(0, selected[0]["start"] - 0.12)
        # Whisper sometimes gives an initial word no duration. Include the
        # onset preceding that timestamp, without shortening the requested pad.
        if selected[0]["end"] - selected[0]["start"] <= 0.04:
            candidates = [e for s, e in onset_quiet if e-s >= 0.1 and
                          selected[0]["start"]-0.65 <= e <= selected[0]["start"]]
            if candidates:
                start = min(start, max(0, candidates[-1] - 0.12))
            if token(selected[0]["word"]) in {"it", "i", "a"}:
                start = min(start, max(0, selected[0]["start"] - 0.25))
        cursor = start
        for left, right in zip(selected, selected[1:]):
            lo, hi = left["end"] + 0.08, right["start"] - 0.05
            if right["start"] - left["end"] <= 0.15:
                continue
            candidates = [s for s, e in quiet if e > lo and left["start"] <= s < hi]
            if candidates:
                lo = max(lo, candidates[0] + 0.05)
            lo = extend_for_hiss(lo, min(left["end"] + 0.4, right["start"] - 0.03), hiss)
            if hi > lo:
                ranges.append([cursor, lo])
                cursor = hi
        tail = 0.5 if last and stop == sentences[-1][1] else 0.10
        nxt = words[stop]["start"] - 0.03 if stop < len(words) else duration
        end = extend_for_hiss(selected[-1]["end"] + tail,
                              min(duration, nxt, selected[-1]["end"] + 0.4), hiss)
        ranges.append([cursor, min(duration, end)])
    # Padding may overlap when consecutive sentences have no pause.
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([round(a, 6), round(b, 6)])
    for first, stop in sentences:
        for word in words[first:stop]:
            if not any(a <= word["start"] and b >= word["end"] for a, b in merged):
                raise ValueError(f"Word outside keep ranges: {word}")
    return merged


def loudness_measure(inputs, graph, label, log):
    result = run(["ffmpeg", "-hide_banner", "-nostats", *inputs,
                  "-filter_complex", graph + f";[{label}]loudnorm=I=-14:TP=-1:LRA=11:print_format=json[norm]",
                  "-map", "[norm]", "-f", "null", "-"], log)
    return json.loads(re.findall(r'\{\s*"input_i".*?\}', result.stderr, re.S)[-1])


def render_mix(manifest, out):
    config = json.loads(manifest.read_text(encoding="utf-8-sig"))
    picks = config["picks"]
    if not picks:
        raise ValueError("Mix needs at least one pick")
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for i, pick in enumerate(picks):
        take = (manifest.parent / pick["take"]).resolve()
        if str(take) not in sources:
            sources[str(take)] = {"probe": probe(take), "quiet": silences(take, -38),
                                   "onset": silences(take, -32), "hiss": hiss_envelope(take),
                                   "ranges": []}
        source = sources[str(take)]
        words = json.loads((manifest.parent / pick["words"]).read_text(encoding="utf-8-sig"))
        ranges = mix_ranges(words, pick["sentences"], float(source["probe"]["format"]["duration"]),
                            source["quiet"], source["onset"], i == len(picks)-1,
                            source["hiss"])
        source["ranges"].extend(ranges)
        pick.update(source=str(take), keep_ranges_seconds=ranges,
                    expected_words=[w for a, b in pick["sentences"] for w in words[a:b]])
        pick["frames"] = math.ceil(sum(b-a for a, b in ranges)*30 - 1e-7)
    duration = sum(p["frames"] for p in picks)/30
    if duration >= 35:
        raise ValueError(f"Mix exceeds 35 seconds: {duration:.3f}")
    # Measure the selected speech from each take, then apply a constant gain to
    # match takes before measuring and normalizing the complete edited program.
    for i, (take, source) in enumerate(sources.items()):
        measurement = loudness_measure(["-i", take], audio_graph(source["ranges"]),
                                       "edited", out / f"mix-take-{i+1}-loudness.log")
        source["loudness"] = measurement
        source["gain_db"] = -20 - float(measurement["input_i"])
    inputs, audio, video_filters = [], [], []
    for i, pick in enumerate(picks):
        ranges = pick["keep_ranges_seconds"]
        offset, end = ranges[0][0], ranges[-1][1]
        inputs += ["-threads", "2", "-ss", str(offset), "-t", str(end-offset), "-i", pick["source"]]
        local = [[round(a-offset, 6), round(b-offset, 6)] for a, b in ranges]
        ag = audio_graph(local).replace("[0:a]", f"[{i}:a]")
        ag = re.sub(r"\[(s\d+|a\d+|edited)\]", lambda m: f"[p{i}_{m[1]}]", ag)
        clip_duration = pick["frames"]/30
        gain = sources[pick["source"]]["gain_db"]
        audio += [ag, f"[p{i}_edited]volume={gain:.6f}dB,aresample=48000,"
                  f"apad,atrim=duration={clip_duration:.9f},asetpts=PTS-STARTPTS[pa{i}]"]
        select = "+".join(f"gte(t,{a})*lt(t,{b})" for a, b in local)
        shift = "0" + "".join(f"+gte(PTS*TB,{local[j][0]})*{local[j][0]-local[j-1][1]:.6f}"
                               for j in range(1, len(local)))
        video_filters.append(f"[{i}:v]select='{select}',setpts='PTS-({shift})/TB',"
            "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:1920,setsar=1,fps=30:start_time=0,"
            f"tpad=stop_mode=clone:stop_duration=0.1,trim=end_frame={pick['frames']},"
            f"settb=AVTB,setpts=PTS-STARTPTS[pv{i}]")
    audio.append("".join(f"[pa{i}]" for i in range(len(picks))) +
                 f"concat=n={len(picks)}:v=0:a=1[edited]")
    audio = ";".join(audio)
    measured = loudness_measure(inputs, audio, "edited", out / "mix-loudnorm-pass1.log")
    norm = ("loudnorm=I=-14:TP=-1:LRA=11:linear=true:print_format=json:"
            f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}")
    layout = overlay(out / "headline-mix.png")
    video_filters += ["".join(f"[pv{i}]" for i in range(len(picks))) +
                      f"concat=n={len(picks)}:v=1:a=0[base]",
                      f"[base][{len(picks)}:v]overlay=0:0:format=auto,format=yuv420p[vout]"]
    filters = audio + f";[edited]{norm},volume=-0.3dB,aresample=48000[aout];" + ";".join(video_filters)
    (out / "render-filter-mix.txt").write_text(filters, encoding="utf-8")
    (out / "cuts-mix.json").write_text(json.dumps({"picks": picks, "planned_duration": duration,
        "sentence_padding": [0.12, 0.10], "end_hold": 0.5, "audio_fade": 0.008,
        "take_gains_db": {k: v["gain_db"] for k, v in sources.items()}}, indent=2), encoding="utf-8")
    video = out / f"reel-{out.parent.name}-mix.mp4"
    if str(video.resolve()) in sources:
        raise ValueError("Output cannot overwrite a source")
    print(f"Rendering mix once: {duration:.3f}s, {len(picks)} picks", flush=True)
    run(["ffmpeg", "-y", "-hide_banner", "-stats", "-stats_period", "5", *inputs,
         "-i", out / "headline-mix.png", "-filter_complex_threads", "2", "-filter_complex", filters,
         "-map", "[vout]", "-map", "[aout]", "-map_metadata", "-1",
         "-c:v", "libx264", "-threads", "4", "-preset", "veryfast", "-crf", "18",
         "-profile:v", "high", "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
         "-color_range", "tv", "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
         "-movflags", "+faststart", video], out / "render-mix.log")
    final = probe(video)
    vs = next(s for s in final["streams"] if s["codec_type"] == "video")
    assert float(final["format"]["duration"]) < 35
    assert (vs["width"], vs["height"], vs["avg_frame_rate"]) == (1080, 1920, "30/1")
    sheet = Image.new("RGB", (1440, 1280), (20, 20, 20))
    cursor, times = 0, []
    for i, pick in enumerate(picks):
        moment = cursor + min(0.8, pick["frames"]/60)
        times.append(moment)
        frame = out / f"mix-frame-{i+1:02d}.jpg"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(moment), "-i", video,
             "-frames:v", "1", "-vf", "scale=360:640", "-q:v", "2", frame])
        with Image.open(frame) as im:
            sheet.paste(im, ((i % 4)*360, (i // 4)*640))
        cursor += pick["frames"]/30
    sheet.save(out / "contact-sheet-mix.jpg", quality=94)
    run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-vn", "-af",
         "loudnorm=I=-14:TP=-1:LRA=11:print_format=json", "-f", "null", "-"], out / "final-loudness-mix.log")
    (out / "render-summary-mix.json").write_text(json.dumps({"duration": float(final["format"]["duration"]),
        "text_layout": layout, "contact_sheet_times": times, "loudnorm_pass1": measured,
        "probe": final}, indent=2), encoding="utf-8")
    print(f"Rendered {video}: {final['format']['duration']}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", type=Path)
    parser.add_argument("--words", type=Path)
    parser.add_argument("--mix", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if args.mix:
        if args.take or args.words:
            parser.error("--mix cannot be combined with --take or --words")
        return render_mix(args.mix.resolve(), out)
    if not args.take or not args.words:
        parser.error("Provide --mix, or both --take and --words")
    take = args.take.resolve()
    out.mkdir(parents=True, exist_ok=True)
    video = out / f"reel-{out.parent.name}.mp4"
    if video == take:
        raise ValueError("Output cannot overwrite the source")
    source_probe = probe(take)
    words = json.loads(args.words.read_text(encoding="utf-8-sig"))
    ranges, guards, removed = make_cuts(words, float(source_probe["format"]["duration"]),
                                        silences(take, -38), silences(take, -32))
    duration = sum(b-a for a, b in ranges)
    frames = math.ceil(duration * 30)
    if frames / 30 >= 35:
        raise ValueError(f"Edit exceeds duration limit: {duration:.3f}s")
    cuts = {"source": str(take), "keep_ranges_seconds": ranges,
            "cut_count": len(ranges)-1, "segment_count": len(ranges),
            "planned_duration": duration, "end_hold_seconds": 0.5,
            "gap_threshold": 0.15, "start_padding": 0.05, "end_padding": 0.08,
            "audio_fade_seconds": 0.008, "boundary_guards": guards,
            "removed_words": removed}
    (out / "cuts.json").write_text(json.dumps(cuts, indent=2), encoding="utf-8")
    text_layout = overlay(out / "headline.png")
    graph = audio_graph(ranges)
    pass1 = graph + ";[edited]loudnorm=I=-14:TP=-1:LRA=11:print_format=json[normalized]"
    print(f"{len(ranges)-1} hard cuts, planned {duration:.3f}s. Measuring audio...", flush=True)
    first = run(["ffmpeg", "-hide_banner", "-nostats", "-i", take, "-filter_complex",
                 pass1, "-map", "[normalized]", "-f", "null", "-"], out / "loudnorm-pass1.log")
    measured = json.loads(re.findall(r'\{\s*"input_i".*?\}', first.stderr, re.S)[-1])
    norm = ("loudnorm=I=-14:TP=-1:LRA=11:linear=true:print_format=json:"
            f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}")
    select = "+".join(f"gte(t,{a})*lt(t,{b})" for a, b in ranges)
    # Compact retained frames onto the same continuous timeline as audio.
    shift = str(ranges[0][0]) + "".join(
        f"+gte(PTS*TB,{ranges[i][0]})*{ranges[i][0]-ranges[i-1][1]:.6f}"
        for i in range(1, len(ranges)))
    # AAC can overshoot the PCM true peak; reserve 0.3 dB of encoding headroom.
    filters = (graph + f";[edited]{norm},volume=-0.3dB,aresample=48000[aout];"
               f"[0:v]select='{select}',setpts='PTS-({shift})/TB',"
               "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
               "crop=1080:1920,setsar=1,fps=30:start_time=0,"
               f"tpad=stop_mode=clone:stop_duration=0.04,trim=end_frame={frames}[base];"
               "[base][1:v]overlay=0:0:format=auto,format=yuv420p[vout]")
    (out / "render-filter.txt").write_text(filters, encoding="utf-8")
    print("Rendering H.264 portrait video...", flush=True)
    run(["ffmpeg", "-y", "-hide_banner", "-stats", "-stats_period", "5", "-threads", "4", "-i", take,
         "-i", out / "headline.png", "-filter_complex_threads", "2", "-filter_complex", filters,
         "-map", "[vout]", "-map", "[aout]", "-map_metadata", "-1",
         "-c:v", "libx264", "-threads", "4", "-preset", "veryfast", "-crf", "18",
         "-profile:v", "high", "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
         "-color_range", "tv", "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
         "-movflags", "+faststart", video], out / "render.log")
    final_probe = probe(video)
    (out / "ffprobe.json").write_text(json.dumps(final_probe, indent=2), encoding="utf-8")
    actual_duration = float(final_probe["format"]["duration"])
    vs = next(s for s in final_probe["streams"] if s["codec_type"] == "video")
    assert actual_duration < 35 and (vs["width"], vs["height"]) == (1080, 1920)
    assert vs["avg_frame_rate"] == "30/1" and vs["pix_fmt"] == "yuv420p"
    assert vs["codec_name"] == "h264" and vs["profile"] == "High"
    times = contact_sheet(video, actual_duration, out)
    run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-vn", "-af",
         "loudnorm=I=-14:TP=-1:LRA=11:print_format=json", "-f", "null", "-"],
        out / "final-loudness.log")
    (out / "render-summary.json").write_text(json.dumps({"duration": actual_duration,
        "cut_count": len(ranges)-1, "text_layout": text_layout,
        "contact_sheet_times": times, "loudnorm_pass1": measured}, indent=2), encoding="utf-8")
    print(f"Rendered {video}: {actual_duration:.3f}s, {len(ranges)-1} hard cuts.", flush=True)


if __name__ == "__main__":
    main()
