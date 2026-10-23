# 🎤 Anatono

**Anatono helps anyone who loves singing hear how they'd sound on their favourite song, and how to sing it
better, using Gemma 4.**

Upload a song. Anatono takes the singer out, you sing the part over the music, and Gemma 4 looks at a chart of your
pitch against the original singer's. It tells you what to fix (in English and Urdu) and decides how much auto-tune
your take needs. Then you hear yourself over the beat, raw and auto-tuned.

## ✅ Verify in 30 seconds

| What | Where |
|---|---|
| **Source** | This repository (public) |
| **Licence** | [MIT](LICENSE) |
| **AI model** | **Gemma 4**, `gemma-4-26b-a4b-it` (open weights, 26B parameters, a *large* model by the challenge's definition), called **through the Gemini API**. Licence: [Apache 2.0](https://ai.google.dev/gemma/apache_2) |
| **The line that calls Gemma** | [`anatono.py`, `coach()`](anatono.py#L147): `client.models.generate_content(model=MODEL, contents=[prompt, img])` with `MODEL = "gemma-4-26b-a4b-it"` |
| **API key** | Bring your own: copy `.env.example` to `.env`. No key is in this repo (`.env` is git-ignored). |
| **Key dependencies** | See the table below, each with its licence |

## What we built, and what the AI does

```
song ──► Demucs ──► music + singer's vocal
                         │
you sing over the music ─┤
                         ▼
          code: line your take up with the song (onset rhythm), measure your pitch vs the singer's
                         │            (pYIN, cents off, every 2 s)
                         ▼
          code: draw the pitch chart  ──►  Gemma 4 looks at the chart + numbers and DECIDES:
                                            • score, what you did well, the 2–3 things to fix (with times)
                                            • the same advice in Urdu
                                            • auto-tune strength (0–1) and which seconds to correct
                         ▼
          code: checks Gemma's answer (clamps strength, drops impossible time ranges), applies auto-tune
                (pyworld), mixes you over the beat
```

**Gemma's part, in plain words:** Gemma 4 is the coach. It *reads* the pitch chart image (multimodal) the way a
vocal teacher would, explains what went wrong and where, and decides how much your take should be corrected. Code
does the measuring and the audio work, and **checks Gemma's decisions before using them**, so the coaching is
grounded in real numbers and the model can't break the audio.

**What the team wrote (all in this repo, built at the hackathon):**
- `anatono.py`: the pipeline: separation, singer-start detection, alignment, pitch comparison with octave folding,
  chart, Gemma coach with output validation and retry, auto-tune, mixing, plus a self-check.
- `app.py`: the one-screen web app (Streamlit).
- `make_sample.py`: a copyright-free sample song, so anyone can try it.

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your Gemini API key (free at aistudio.google.com)
streamlit run app.py
```

No song handy? `python3 make_sample.py` (macOS) makes `songs/sample_song.wav`.
Self-check (uses a fake take with a known answer): `python3 anatono.py songs/sample_song.wav`

**Use headphones** when you record, so the music doesn't leak into your take.

## Key dependencies

| Package | Used for | Licence |
|---|---|---|
| [Gemma 4](https://ai.google.dev/gemma) via [google-genai](https://pypi.org/project/google-genai/) 2.33 | The coach | Gemma: Apache 2.0 · SDK: Apache-2.0 |
| [Demucs](https://github.com/facebookresearch/demucs) 4.1.0 (`htdemucs`) | Splitting music and vocals (open-weight model) | MIT |
| [PyTorch](https://pytorch.org) 2.14 | Runs Demucs | BSD-style |
| [librosa](https://librosa.org) 1.0 | Pitch (pYIN), onsets, resampling | ISC |
| [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) 0.3.5 | Auto-tune (WORLD vocoder) | MIT |
| [Streamlit](https://streamlit.io) 1.66 | Web app | Apache-2.0 |
| soundfile, numpy, matplotlib, python-dotenv | Audio files, maths, chart, `.env` | BSD / PSF-style |

## Notes

- Songs you upload stay on your machine. Only the pitch **chart image** and the numbers are sent to Gemma, never
  your audio.
- Don't commit songs you don't own. `songs/` and audio files are git-ignored.
- What's next: lyrics on screen in time with the music, and a "sing it again" loop that tracks your progress.
