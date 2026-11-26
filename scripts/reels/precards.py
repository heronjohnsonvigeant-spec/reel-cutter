"""Pre-built cards: build them from the approved script BEFORE filming, then
drop them onto the cut where the anchor phrase is spoken.

    python scripts/reels/precards.py build   SPEC.json
    python scripts/reels/precards.py place   SPEC.json --edit CUT.mp4 --words WORDS.json [--cuts cuts.json] --out OUT.mp4
    python scripts/reels/precards.py compare REFERENCE.mp4 OURS.mp4 --out SHEET.jpg

SPEC.json:
{
  "reel": "my-reel-v1",
  "script": "scripts/my-reel-v1.txt",
  "out_dir": "output/precards/my-reel-v1",
  "cards": [
    {"id": "c1", "anchor": "first 36 months", "template": "timeline", "fields": {...}, "hold": 3.5}
  ]
}

Templates (all #0836A8, Inter Bold, white / #D6E2FF / yellow #FFD23F):
  stat      kicker, value, label, source
  compare   kicker, left{title,value,sub}, right{title,value,sub}, bottom, source
  timeline  kicker, start, end, marker, progress (0-1), bottom, source
  steps     title, steps[] (never exactly 3), source

Drawn with Pillow, no browser needed.

Rules baked in: nothing in the first 3 s, 2.5 to 4 s
on screen (about 1 s per 3 words of card text), never two cards at
once, never overwrite an existing file. `--words` is either the cut's own
words file, or the raw take's words file plus `--cuts` (the cut's
keep_ranges_seconds) to map raw time onto cut time.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT = ROOT / "scripts/reels/fonts/Inter-Bold.ttf"
BLUE, SOFT, YELLOW, WHITE = "#0836A8", "#D6E2FF", "#FFD23F", "#FFFFFF"
GLASS = (255, 255, 255, 36)
FPS = 30
MIN_START = 3.0
MIN_HOLD = 2.5
MAX_HOLD = 4.0  # 11-23 rule: aim for 3-4 s holds
SETTLED = 2.4  # every entrance animation has finished by this second


def card_text(card: dict) -> str:
    parts: list[str] = []
    for key, v in card["fields"].items():
        if key in ("progress", "source"):
            continue
        if isinstance(v, dict):
            parts += [str(x) for x in v.values()]
        elif isinstance(v, list):
            parts += [str(x) for x in v]
        else:
            parts.append(str(v))
    return " ".join(parts)


def hold_for(card: dict) -> float:
    words = len(card_text(card).split())
    wanted = card.get("hold") or words / 3
    return round(min(MAX_HOLD, max(MIN_HOLD, wanted)), 2)


@lru_cache(maxsize=64)
def font(size: int):
    return ImageFont.truetype(str(FONT), size)


def wrap(d, text: str, f, width: int) -> list[str]:
    lines, cur = [], ""
    for word in str(text).split():
        trial = f"{cur} {word}".strip()
        if cur and d.textlength(trial, font=f) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def fit(d, text, size: int, width: int):
    """Largest font at or under `size` that keeps `text` on one line."""
    while size > 20 and d.textlength(str(text), font=font(size)) > width:
        size -= 4
    return font(size)


def ease(t: float, start: float, dur: float = .35) -> float:
    x = min(1.0, max(0.0, (t - start) / dur))
    return 1 - (1 - x) ** 3


def centred(d, y: int, text, size: int, fill=WHITE, width: int = 920, cx: int = 540) -> None:
    f = font(size)
    for line in wrap(d, text, f, width):
        d.text((cx, y), line, font=f, fill=fill, anchor="ma")
        y += int(size * 1.22)


def layer(im: Image.Image, alpha: float, paint, dx: float = 0, dy: float = 0) -> None:
    """Paint onto a transparent layer, then fade and shift it onto the frame."""
    if alpha <= 0:
        return
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    paint(ImageDraw.Draw(over))
    if alpha < 1:
        over.putalpha(over.getchannel("A").point(lambda v: int(v * alpha)))
    if dx or dy:
        moved = Image.new("RGBA", im.size, (0, 0, 0, 0))
        moved.paste(over, (int(dx), int(dy)))
        over = moved
    im.alpha_composite(over)


def render_frame(card: dict, t: float) -> Image.Image:
    kind, f = card["template"], card["fields"]
    im = Image.new("RGBA", (1080, 1920), BLUE)

    def text(start, y, value, size, fill=WHITE, rise=30):
        a = ease(t, start)
        layer(im, a, lambda d: centred(d, y, value, size, fill), dy=(1 - a) * rise)

    if f.get("source"):
        text(.1, 1440, str(f["source"]).upper(), 30, SOFT, rise=0)
    if kind != "steps" and f.get("kicker"):
        text(0, 430, str(f["kicker"]).upper(), 38, SOFT)

    if kind == "stat":
        a = ease(t, .15, .4)
        layer(im, a, lambda d: d.text((540, 800), str(f["value"]), font=fit(d, f["value"], 210, 920),
                                      fill=WHITE, anchor="mm"), dy=(1 - a) * 60)
        text(.6, 980, f.get("label", ""), 60)
    elif kind == "compare":
        for i, (key, start) in enumerate((("left", .15), ("right", .9))):
            col, x0, win = f[key], 70 + i * 490, i == 1

            def paint(d, col=col, x0=x0, win=win):
                ink, cx = (BLUE if win else WHITE), x0 + 225
                d.rounded_rectangle((x0, 580, x0 + 450, 1060), 36, fill=WHITE if win else GLASS)
                centred(d, 640, str(col.get("title", "")).upper(), 32, ink, 390, cx)
                d.text((cx, 820), str(col.get("value", "")), font=fit(d, col.get("value", ""), 110, 400),
                       fill=ink, anchor="mm")
                centred(d, 920, col.get("sub", ""), 40, ink, 390, cx)

            a = ease(t, start)
            layer(im, a, paint, dx=(1 - a) * (40 if win else -40))
        text(1.6, 1160, f.get("bottom", ""), 52)
    elif kind == "timeline":
        x = 100 + int(880 * float(f.get("progress", 1.0)) * ease(t, .2, 1.4))

        def paint(d):
            d.rounded_rectangle((100, 800, 980, 836), 18, fill=(255, 255, 255, 46))
            d.rounded_rectangle((100, 800, max(x, 136), 836), 18, fill=YELLOW)
            d.text((100, 866), str(f.get("start", "")), font=font(40), fill=SOFT, anchor="la")
            d.text((980, 866), str(f.get("end", "")), font=font(40), fill=SOFT, anchor="ra")
            marker = str(f.get("marker", ""))
            if marker:
                w = d.textlength(marker, font=font(40)) + 48
                left = min(max(x - w / 2, 70), 1010 - w)
                d.rounded_rectangle((left, 690, left + w, 766), 18, fill=WHITE)
                d.text((left + w / 2, 728), marker, font=font(40), fill=BLUE, anchor="mm")

        layer(im, ease(t, .1), paint)
        text(1.7, 1030, f.get("bottom", ""), 56)
    elif kind == "steps":
        steps = f["steps"]
        if len(steps) == 3:
            raise SystemExit(f"card {card['id']}: steps card has exactly 3 steps (rule: no lists of three)")
        text(.1, 400, f["title"], 66, YELLOW)
        y = 640
        probe = ImageDraw.Draw(im)
        for i, step in enumerate(steps):
            lines = wrap(probe, step, font(44), 760)
            h = 60 + 56 * len(lines)

            def paint(d, y=y, h=h, i=i, lines=lines):
                d.rounded_rectangle((70, y, 1010, y + h), 28, fill=GLASS)
                d.text((120, y + h / 2), str(i + 1), font=font(46), fill=YELLOW, anchor="mm")
                for j, line in enumerate(lines):
                    d.text((180, y + 32 + 56 * j), line, font=font(44), fill=WHITE, anchor="la")

            a = ease(t, .5 + i * .45)
            layer(im, a, paint, dx=(1 - a) * -30)
            y += h + 28
        if y > 1420:
            raise SystemExit(f"card {card['id']}: steps run past the safe area, shorten them")
    else:
        raise SystemExit(f"card {card['id']}: unknown template {kind!r}")
    return im.convert("RGB")


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["out_dir"] = ROOT / spec["out_dir"]
    return spec


def check_anchors(spec: dict) -> None:
    script = spec.get("script")
    if not script:
        return
    text = re.sub(r"\s+", " ", (ROOT / script).read_text(encoding="utf-8")).lower()
    for c in spec["cards"]:
        if c["anchor"].lower() not in text:
            raise SystemExit(f"card {c['id']}: anchor {c['anchor']!r} is not in the approved script")


def cmd_build(a) -> None:
    spec = load_spec(a.spec)
    check_anchors(spec)
    out = spec["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    old = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    manifest = {}
    for c in spec["cards"]:
        dur = hold_for(c)
        mp4 = out / f"{c['id']}.mp4"
        if mp4.exists() and not a.force:
            print(f"skip {mp4.name} (exists; --force to rebuild under a new name)")
            manifest[c["id"]] = old.get(c["id"], {"file": mp4.name, "hold": dur, "anchor": c["anchor"]})
            continue
        n = 2
        while mp4.exists():
            mp4 = out / f"{c['id']}-v{n}.mp4"
            n += 1
        render_frame(c, SETTLED).save(out / f"{mp4.stem}-proof.png")
        proc = subprocess.Popen(
            ["ffmpeg", "-n", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920",
             "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", "-pix_fmt", "yuv420p", str(mp4)], stdin=subprocess.PIPE)
        still = None
        for i in range(int(round(dur * FPS))):
            t = i / FPS
            if t < SETTLED:
                frame = render_frame(c, t).tobytes()
            else:
                still = still or render_frame(c, SETTLED).tobytes()
                frame = still
            proc.stdin.write(frame)
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed on {mp4}")
        manifest[c["id"]] = {"file": mp4.name, "hold": dur, "anchor": c["anchor"]}
        print(f"built {mp4.name} ({dur}s) + proof png", flush=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def to_cut_time(t: float, ranges: list[list[float]] | None) -> float | None:
    if not ranges:
        return t
    acc = 0.0
    for a, b in ranges:
        if a <= t <= b:
            return acc + (t - a)
        acc += b - a
    return None


def find_anchor(words: list[dict], anchor: str) -> float | None:
    target = [norm(w) for w in anchor.split() if norm(w)]
    toks = [norm(w["word"]) for w in words]
    for i in range(len(toks) - len(target) + 1):
        if toks[i:i + len(target)] == target:
            return words[i]["start"]
    return None


def media_duration(path: Path) -> float:
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]).strip())


def cmd_place(a) -> None:
    spec = load_spec(a.spec)
    out_dir = spec["out_dir"]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    words = json.loads(a.words.read_text(encoding="utf-8"))
    ranges = json.loads(a.cuts.read_text(encoding="utf-8"))["keep_ranges_seconds"] if a.cuts else None
    if a.out.exists():
        raise SystemExit(f"Refusing to overwrite {a.out}")
    cut_len = media_duration(a.edit)

    found, problems = [], []
    for c in spec["cards"]:
        raw_t = find_anchor(words, c["anchor"])
        if raw_t is None:
            problems.append(f"{c['id']}: anchor {c['anchor']!r} not heard in the take (reworded on camera?)")
            continue
        t = to_cut_time(raw_t, ranges)
        if t is None:
            problems.append(f"{c['id']}: anchor falls in a trimmed section of the cut")
            continue
        found.append((t, c["id"]))

    placed, last_end = [], 0.0
    for t, cid in sorted(found):
        m = manifest[cid]
        t = max(t, MIN_START, last_end + .3 if placed else 0)
        end = t + m["hold"]
        if end > cut_len - 1.5:
            problems.append(f"{cid}: would run into the last line (start {t:.2f}s)")
            continue
        placed.append((cid, out_dir / m["file"], round(t, 2), round(end, 2)))
        last_end = end
    if not placed:
        raise SystemExit("No cards could be placed, so nothing was written:\n" + "\n".join(problems))

    inputs, chain, prev = ["-i", str(a.edit)], [], "0:v"
    for i, (_cid, f, t, end) in enumerate(placed, 1):
        inputs += ["-i", str(f)]
        chain.append(f"[{i}:v]setpts=PTS-STARTPTS+{t}/TB[c{i}]")
        chain.append(f"[{prev}][c{i}]overlay=0:0:eof_action=pass:enable='between(t,{t},{end})'[v{i}]")
        prev = f"v{i}"
    graph = ";".join(chain) + f";[{prev}]format=yuv420p[vout]"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-n", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[vout]",
                    "-map", "0:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "copy",
                    "-movflags", "+faststart", str(a.out)], check=True)
    report = {"output": str(a.out), "placed": [{"id": p[0], "start": p[2], "end": p[3]} for p in placed],
              "problems": problems}
    a.out.with_suffix(".placement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cmd_compare(a) -> None:
    """Side by side at 8 matching points: reference left, ours right."""
    tmp = a.out.parent / (a.out.stem + "-frames")
    tmp.mkdir(parents=True, exist_ok=True)
    sources = ((a.reference, media_duration(a.reference)), (a.ours, media_duration(a.ours)))
    n, w, h = 8, 240, 426  # even height: ffmpeg pad rounds odd sizes down
    # 4 time points per row pair keeps the sheet readable on a phone
    sheet = Image.new("RGB", (4 * (w * 2 + 30), 2 * (h + 40) + 20), "#151515")
    d = ImageDraw.Draw(sheet)
    for i in range(n):
        frac = (i + .5) / n
        bx, by = (i % 4) * (w * 2 + 30), (i // 4) * (h + 40) + 20
        d.text((bx + 4, by - 16), f"{frac:.0%}   REFERENCE | OURS", fill="white")
        for col, (src, total) in enumerate(sources):
            t = frac * total
            fp = tmp / f"{col}-{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
                            str(fp)], check=True)
            with Image.open(fp) as im:
                x = bx + col * (w + 4)
                sheet.paste(im, (x, by))
                d.text((x + 4, by + h + 4), f"{t:.1f}s", fill="#bbbbbb")
    sheet.save(a.out, quality=90)
    print(f"wrote {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("spec", type=Path)
    b.add_argument("--force", action="store_true", help="rebuild as a new version file, never overwrite")
    p = sub.add_parser("place")
    p.add_argument("spec", type=Path)
    p.add_argument("--edit", type=Path, required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--cuts", type=Path)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("reference", type=Path)
    c.add_argument("ours", type=Path)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"build": cmd_build, "place": cmd_place, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
