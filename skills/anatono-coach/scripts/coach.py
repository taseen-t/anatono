"""Coach one sung take against a song with Anatono (Gemma 4). Run from anywhere inside the Anatono repo.
Usage: python3 skills/anatono-coach/scripts/coach.py --song SONG --take TAKE [--start 0] [--dur 20] [--out out]
Writes: coach.json, chart.png, raw_mix.wav, tuned_mix.wav in --out, and prints coach.json."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # repo root: skills/anatono-coach/scripts/coach.py -> ../../..
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--song", required=True, help="the original song (mp3/wav/m4a/flac)")
p.add_argument("--take", required=True, help="the user singing the same part (any audio format)")
p.add_argument("--start", type=float, default=0.0, help="where the part starts in the song, seconds")
p.add_argument("--dur", type=float, default=20.0, help="length of the part, seconds (10-30)")
p.add_argument("--out", default="out", help="folder for the results")
a = p.parse_args()

for f in (a.song, a.take):
    if not Path(f).is_file():
        sys.exit(f"File not found: {f}")
if not 10 <= a.dur <= 30:
    sys.exit("--dur must be between 10 and 30 seconds")
if not (ROOT / ".env").is_file():
    sys.exit(f"No .env in {ROOT}: copy .env.example to .env and add GEMINI_API_KEY")

import librosa       # noqa: E402 - after the checks, so bad input fails fast
import soundfile as sf  # noqa: E402
import anatono as A  # noqa: E402

take, _ = librosa.load(a.take, sr=A.SR, mono=True)
r = A.run(a.song, take, a.start, a.dur, log=lambda m: print(m, file=sys.stderr))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
r["chart"].save(out / "chart.png")
sf.write(out / "raw_mix.wav", r["raw_mix"], A.SR)
sf.write(out / "tuned_mix.wav", r["tuned_mix"], A.SR)
result = {"coach": r["coach"], "measured": r["summary"], "take_starts_at_s": round(r["offset"], 2),
          "files": {k: str(out / k) for k in ("chart.png", "raw_mix.wav", "tuned_mix.wav")}}
(out / "coach.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
print(json.dumps(result, ensure_ascii=False, indent=1))
