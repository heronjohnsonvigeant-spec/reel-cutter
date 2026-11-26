"""Burned-in captions for reels (Mino Lee style: short chunks, hard cut on and off).

TikTok Sans 550 at 52 px, white, slight soft shadow, 2-3 words at a time, hard cut on and off,
one line centred around the mouth (y 1055), never inside the Reels/TikTok UI zones.
Each chunk is one transparent 1080x1920 PNG; an ffconcat file times them onto the cut timeline
so ffmpeg overlays them in one pass (no browser).
"""
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = Path(__file__).resolve().parent / "fonts" / "TikTokSans-latin.woff2"
WEIGHT, SIZE, TRACKING_EM = 550, 52, -0.035
CENTER_Y = 1420  # below the face (chest), not on it; still above the Reels bottom zone (1536)
MAX_WORDS, MAX_CHARS = 3, 18
LEAD_S, HOLD_S, GAP_BREAK_S = 0.04, 0.5, 0.28
SHADOW_ALPHA, SHADOW_BLUR = 51, 13  # 20% black, blur = 25% of the font size
# Reels/TikTok draw their UI here: top tabs, bottom caption/username, right-side button column.
SAFE_TOP, SAFE_BOTTOM, SAFE_RAIL_X, SAFE_RAIL_FROM_Y = 288, 1536, 900, 960
# Words drawn in the brand amber instead of white (set per reel, e.g. {"wait", "ready", "today"}); empty = all white.
HIGHLIGHT: set = set()
HIGHLIGHT_FILL = (242, 169, 59, 255)
CENSOR = {"fuck": "f*ck", "fucking": "f*cking", "fucked": "f*cked", "shit": "sh*t", "bitch": "b*tch"}
WEAK = set("a an the to of my your in for and i i've i'm that this from with like so but can at on how as our is "
           "it's you'll you're we we're they their his her its be was are while when if because or about into than "
           "just very really".split())


def _norm(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def _clean(word):
    text = re.sub(r"[,.?!;:]+$", "", word.strip())
    return CENSOR.get(_norm(text), text)


def _is_number(word):
    return bool(re.match(r"^\$?\d", word))


def chunk(words):
    """Group words into caption chunks in source time: [{"text", "start", "end"}]."""
    chunks, current = [], []

    def close(keep_tail=True):
        nonlocal current
        carry = []
        # Don't end a chunk on "to/the/I" or on a bare number when more speech follows.
        while keep_tail and len(current) > 1 and (_norm(current[-1]["word"]) in WEAK or _is_number(_clean(current[-1]["word"]))):
            carry.insert(0, current.pop())
        if current:
            chunks.append({"text": " ".join(_clean(w["word"]) for w in current),
                           "start": float(current[0]["start"]), "end": float(current[-1]["end"])})
        current = carry

    for index, word in enumerate(words):
        if not _clean(word["word"]):
            continue
        trial = " ".join(_clean(w["word"]) for w in current + [word])
        if current and (len(current) >= MAX_WORDS or len(trial) > MAX_CHARS):
            close()
        current.append(word)
        nxt = words[index + 1] if index + 1 < len(words) else None
        sentence_end = bool(re.search(r"[.?!,]$", word["word"].strip()))
        long_gap = nxt is not None and float(nxt["start"]) - float(word["end"]) > GAP_BREAK_S
        if nxt is None or sentence_end or long_gap:
            close(keep_tail=False)
    if current:
        close(keep_tail=False)
    return chunks


def to_output(t, ranges):
    """Source time -> time in the cut reel, or None if t falls in a removed gap."""
    offset = 0.0
    for start, stop in ranges:
        if start - 1e-6 <= t <= stop + 1e-6:
            return offset + (t - start)
        offset += stop - start
    return None


def _clamp_to_ranges(t, ranges):
    for start, stop in ranges:
        if t < start:
            return start
        if t <= stop:
            return t
    return ranges[-1][1]


def track(words, ranges):
    """Caption chunks on the output timeline: [{"t", "s", "e"}], non-overlapping."""
    caps = []
    for c in chunk(words):
        s = to_output(_clamp_to_ranges(c["start"], ranges), ranges)
        e = to_output(_clamp_to_ranges(c["end"], ranges), ranges)
        if s is None or e is None or e <= s:
            continue
        caps.append({"t": c["text"], "s": max(0.0, s - LEAD_S), "e": e + HOLD_S - LEAD_S})
    for left, right in zip(caps, caps[1:]):
        left["e"] = min(left["e"], right["s"])
    return [dict(c, s=round(c["s"], 3), e=round(c["e"], 3)) for c in caps if c["e"] > c["s"]]


def _font(size=SIZE):
    font = ImageFont.truetype(str(FONT), size)
    font.set_variation_by_axes([WEIGHT])
    return font


def render_png(text, path, font_path=None):
    """One caption chunk as a transparent 1080x1920 PNG. Returns the text bbox; raises if it leaves the safe area."""
    font = _font()
    track_px = TRACKING_EM * SIZE
    # Per-character x from the prefix width keeps kerning; tracking is added on top.
    xs = [font.getlength(text[:i]) + i * track_px for i in range(len(text))]
    width = (font.getlength(text) + (len(text) - 1) * track_px) if text else 0
    ascent, descent = font.getmetrics()
    x0, y0 = (1080 - width) / 2, CENTER_Y - (ascent + descent) / 2
    if x0 + width > SAFE_RAIL_X or x0 < 1080 - SAFE_RAIL_X:
        raise ValueError(f"Caption too wide for the safe area: {text!r}")

    hot = set()
    for m in re.finditer(r"\S+", text):
        if _norm(m.group()) in HIGHLIGHT:
            hot.update(range(m.start(), m.end()))

    def draw_text(fill, hot_fill=None):
        layer = Image.new("RGBA", (1080, 1920))
        d = ImageDraw.Draw(layer)
        for i, (ch, x) in enumerate(zip(text, xs)):
            d.text((x0 + x, y0), ch, font=font, fill=hot_fill if hot_fill and i in hot else fill)
        return layer

    shadow = draw_text((0, 0, 0, SHADOW_ALPHA)).filter(ImageFilter.GaussianBlur(SHADOW_BLUR / 2))
    canvas = Image.alpha_composite(shadow, draw_text((255, 255, 255, 255), HIGHLIGHT_FILL))
    canvas.save(path)
    box = draw_text((255, 255, 255, 255)).getbbox() or (0, 0, 0, 0)
    left, top, right, bottom = box
    if top < SAFE_TOP or bottom > SAFE_BOTTOM or (bottom > SAFE_RAIL_FROM_Y and right > SAFE_RAIL_X):
        raise ValueError(f"Caption leaves the safe area: {text!r} {box}")
    return box


def write_concat(caps, work, duration):
    """PNG per chunk + a blank, timed by an ffconcat file covering 0..duration. Returns its path."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    blank = work / "cap-blank.png"
    Image.new("RGBA", (1080, 1920)).save(blank)
    entries, cursor = [], 0.0
    for i, cap in enumerate(caps):
        if cap["s"] >= duration:
            break
        if cap["s"] > cursor:
            entries.append((blank, cap["s"] - cursor))
        png = work / f"cap-{i:03d}.png"
        render_png(cap["t"], png)
        end = min(cap["e"], duration)
        entries.append((png, end - max(cap["s"], cursor)))
        cursor = end
    if cursor < duration:
        entries.append((blank, duration - cursor))
    lines = ["ffconcat version 1.0"]
    for png, seconds in entries:
        lines += [f"file '{png.name}'", f"duration {seconds:.3f}"]
    lines.append(f"file '{entries[-1][0].name}'")  # concat demuxer needs the last file repeated for its duration
    path = work / "captions.ffconcat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
