---
name: anatono-coach
description: Coach someone's singing against an original song with Anatono and Gemma 4. Use when the user wants to know how their voice sounds on a song, how off-key they were, what to fix, or wants their take auto-tuned and mixed over the song's music. Splits the song into music and vocals (Demucs), aligns the user's recording, measures pitch against the original singer, has Gemma 4 read the pitch chart and decide the coaching and auto-tune, then mixes the result.
license: MIT
compatibility: Needs the Anatono repo (github.com/taseen-t/anatono) with requirements.txt installed, Python 3.11+, ffmpeg-readable audio, network access to the Gemini API, and GEMINI_API_KEY in the repo's .env.
metadata:
  model: gemma-4-26b-a4b-it
  repo: https://github.com/taseen-t/anatono
---

# Anatono coach

Gives singing feedback from two audio files: the original song and the user singing the same part.

## Steps

1. Get from the user: the **song file**, their **recorded take** (sung over the music, ideally with headphones), and
   **where the part starts** in the song (seconds) and **how long** it is (10–30 s). If they don't know the start,
   use 0 and the length of their take.
2. From the Anatono repo root, run:

   ```bash
   python3 skills/anatono-coach/scripts/coach.py --song SONG --take TAKE --start START --dur DUR --out out
   ```

   It takes about 1–2 minutes: Demucs separation, then one Gemma 4 call. Progress goes to stderr, and the result
   JSON goes to stdout and `out/coach.json`.
3. Report back to the user:
   - `coach.score` (0–100), `coach.praise`, and each item in `coach.fixes` (they name times in seconds),
   - `coach.urdu` if the user writes in Urdu or asks for it,
   - the auto-tune Gemma chose: `coach.autotune` (0–1) and `coach.fix_ranges`,
   - the files: `out/tuned_mix.wav` (auto-tuned over the beat), `out/raw_mix.wav` (as sung), `out/chart.png`.

## Example

Input: `--song songs/sample_song.wav --take my_take.m4a --start 0 --dur 20`

Output (shortened):
```json
{"coach": {"score": 78, "praise": "Good timing...", "fixes": ["At 6-8 s you drift flat on the long note..."],
 "urdu": "...", "autotune": 0.45, "fix_ranges": [[6.0, 8.0]]},
 "measured": {"avg_abs_cents": 38, "within_50_cents_pct": 71}, "take_starts_at_s": 0.42}
```

No song? `python3 make_sample.py` (macOS) creates a copyright-free `songs/sample_song.wav`.

## Edge cases

- **"Gemma did not return usable coaching"**: the API failed twice (it sometimes returns 500). Wait a minute and
  rerun.
- **`sang_with_singer_pct` is very low**: the take was probably recorded at the wrong part, or the music leaked into
  the mic. Ask the user to re-record with headphones, starting at the same point as `--start`.
- **The user sang an octave higher or lower**: that's fine. Pitch is compared octave-folded.
- Never commit or upload songs the user doesn't own. Only the chart image and numbers are sent to Gemma.
