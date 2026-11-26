"""Find and mute lip smacks. Usage: python find_smacks.py file.mp4 [--fix out.mp4]

A smack (measured on real phone takes): a jump of 20+ dB in one 10 ms frame to above -32 dB,
then a decay below -38 dB within 80 ms, and no speech (nothing above -35 dB) for the next 100 ms.
Speech onsets rise over several frames and stay loud, so they don't match.
--fix mutes each smack (from 20 ms before the jump until it falls under -48 dB) and copies the video.
"""
import subprocess
import sys

import numpy as np

src = sys.argv[1]
SR, WIN = 16000, 160
pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                     capture_output=True, check=True).stdout
x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
n = len(x) // WIN
db = 20 * np.log10(np.sqrt((x[: n * WIN].reshape(n, WIN) ** 2).mean(1) + 1e-12))

hits = []
p = 1
while p < n - 20:
    jump = db[p] - db[p - 1]
    if jump >= 20 and db[p] > -32 and db[p - 1] < -40:
        peak = p + int(np.argmax(db[p:p + 3]))
        decayed = db[peak + 8] < -38
        quiet_after = db[peak + 8:peak + 18].max() < -35
        if decayed and quiet_after:
            end = peak
            while end < n - 1 and db[end] > -48 and end < peak + 25:
                end += 1
            hits.append(((p - 2) * .01, (end + 2) * .01, float(db[peak])))
            p = end
    p += 1

name = src.replace("\\", "/").split("/")[-1]
print(name, "smacks:", ", ".join(f"{a:.2f}-{b:.2f}s ({pk:.0f} dB)" for a, b, pk in hits) or "none")
if "--fix" in sys.argv:
    out = sys.argv[sys.argv.index("--fix") + 1]
    if hits:
        vol = ",".join(f"volume=enable='between(t,{a:.3f},{b:.3f})':volume=0" for a, b, _ in hits)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-c:v", "copy", "-af", vol, "-c:a", "aac",
                        "-b:a", "192k", "-movflags", "+faststart", out], check=True)
        print("wrote", out)
