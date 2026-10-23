"""Anatono: sing over any song, and Gemma 4 coaches your voice.

Pipeline: Demucs splits the song into music + vocals -> you record yourself over the music -> code aligns your take
and measures your pitch against the original singer -> Gemma 4 reads the pitch chart and decides the coaching and
how much auto-tune to apply -> code checks Gemma's answer, applies auto-tune (pyworld) and mixes you over the beat.
"""
import io, json, re, time
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv(Path(__file__).with_name(".env"))
SR = 44100      # audio everywhere
PSR = 16000     # pitch analysis
HOP = 512
MODEL = "gemma-4-26b-a4b-it"
_demucs = None


def separate(path, start=0.0, dur=25.0):
    """Cut [start, start+dur] from the song and split it into (music, vocals), mono at SR."""
    global _demucs
    x, _ = librosa.load(path, sr=SR, mono=False, offset=start, duration=dur)
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    if _demucs is None:
        _demucs = get_model("htdemucs").eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    with torch.no_grad():
        out = apply_model(_demucs, torch.tensor(x, dtype=torch.float32)[None], device=dev, progress=False)[0]
    v = _demucs.sources.index("vocals")
    vocals = out[v].mean(0).numpy()
    music = (out.sum(0) - out[v]).mean(0).numpy()
    return music, vocals


def singing_starts(vocals):
    """Seconds into the clip where the original singer starts."""
    rms = librosa.feature.rms(y=vocals, hop_length=HOP)[0]
    on = np.flatnonzero(rms > 0.15 * rms.max())
    return float(on[0] * HOP / SR) if len(on) else 0.0


def align(take, vocals, max_lead=8.0):
    """Song time (s) at which the take's first sample sits, by matching onset rhythm (works an octave apart too)."""
    a = librosa.onset.onset_strength(y=take, sr=SR, hop_length=HOP)
    b = librosa.onset.onset_strength(y=vocals, sr=SR, hop_length=HOP)
    a, b = (a - a.mean()) / (a.std() + 1e-9), (b - b.mean()) / (b.std() + 1e-9)
    c = np.correlate(b, a, mode="full")          # lag k: take[i] ~ vocals[i + k]
    lags = np.arange(-len(a) + 1, len(b))
    ok = lags * HOP / SR >= -max_lead
    k = lags[ok][np.argmax(c[ok])]
    return float(k * HOP / SR)


def pitch(y):
    """Per-frame MIDI pitch (nan where not singing)."""
    f0, voiced, _ = librosa.pyin(librosa.resample(y, orig_sr=SR, target_sr=PSR), fmin=65, fmax=1000,
                                 sr=PSR, hop_length=HOP)
    m = librosa.hz_to_midi(f0)
    m[~voiced] = np.nan
    return m


def compare(take, vocals, offset):
    """Line the take up with the original singer and measure how far off you are, in cents."""
    shift = int(round(offset * SR))
    ref = vocals[max(shift, 0):][: len(take) - max(-shift, 0)]
    you = take[max(-shift, 0):][: len(ref)]
    mr, my = pitch(ref), pitch(you)
    n = min(len(mr), len(my))
    mr, my = mr[:n], my[:n]
    diff = (my - mr) * 100
    diff = (diff + 600) % 1200 - 600               # singing an octave lower/higher is fine
    t = np.arange(n) * HOP / PSR + max(offset, 0)  # song-clip time of each frame
    both = ~np.isnan(diff)
    segs = []
    for s in np.arange(t[0], t[-1], 2.0):
        w = both & (t >= s) & (t < s + 2)
        if w.sum() >= 5:
            segs.append({"from": round(float(s), 1), "to": round(float(s + 2), 1),
                         "avg_cents": int(np.mean(diff[w])), "avg_abs_cents": int(np.mean(np.abs(diff[w])))})
    summary = {
        "sang_with_singer_pct": int(100 * both.sum() / max(1, (~np.isnan(mr)).sum())),
        "avg_abs_cents": int(np.nanmean(np.abs(diff))) if both.any() else None,
        "within_50_cents_pct": int(100 * (np.abs(diff[both]) <= 50).mean()) if both.any() else 0,
        "segments": segs,
    }
    return t, mr, my, diff, summary


def chart(t, mr, my, diff):
    """Pitch chart Gemma looks at: original singer vs you, off-key stretches shaded red."""
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=110)
    my_fold = mr + diff / 100                     # your pitch, moved to the singer's octave
    ax.plot(t, mr, color="#888", lw=3, label="Original singer")
    ax.plot(t, my_fold, color="#1f6feb", lw=2, label="You")
    off = np.abs(diff) > 50
    ax.fill_between(t, np.nanmin(mr) - 1, np.nanmax(mr) + 1, where=off, color="red", alpha=0.15,
                    label="More than 50 cents off")
    lo, hi = int(np.nanmin(mr)) - 1, int(np.nanmax(mr)) + 2
    ax.set_yticks(range(lo, hi), [librosa.midi_to_note(m) for m in range(lo, hi)])
    ax.set_xlabel("seconds into the clip")
    ax.set_xticks(np.arange(np.floor(t[0]), t[-1] + 1, 1))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Pitch: original singer vs you")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Image.open(buf)


COACH_PROMPT = """You are Anatono, a friendly singing coach. The image is a pitch chart: grey = the original singer,
blue = the user singing the same part (moved to the singer's octave), red shading = more than 50 cents off.
Numbers measured by code: {summary}

Look at the chart and the numbers, then decide:
- what the user did well and the 2-3 most important things to fix, naming the time in seconds,
- the same advice in simple Urdu (Urdu script),
- how much auto-tune to apply: 0 (none, already good) to 1 (full snap to the note),
- which time ranges to correct (empty list = correct everywhere).
Return only JSON:
{{"score": 0-100, "praise": "...", "fixes": ["..."], "urdu": "...", "autotune": 0.0, "fix_ranges": [[start, end]]}}"""


def coach(img, summary, clip_len):
    """Gemma 4 reads the chart and decides the coaching + auto-tune. Code checks the answer before using it."""
    client = genai.Client()
    prompt = COACH_PROMPT.format(summary=json.dumps(summary))
    last = None
    for attempt in range(2):                       # the API sometimes returns 500 or loose JSON: retry once
        try:
            text = client.models.generate_content(model=MODEL, contents=[prompt, img]).text or ""
            d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            d["autotune"] = float(np.clip(float(d.get("autotune", 0.5)), 0, 1))
            d["score"] = int(np.clip(int(d.get("score", 0)), 0, 100))
            d["fix_ranges"] = [[max(0.0, float(a)), min(clip_len, float(b))] for a, b in d.get("fix_ranges", [])
                               if float(b) > float(a) and float(a) < clip_len]
            d["fixes"] = [str(f) for f in d.get("fixes", [])][:4]
            return d
        except Exception as e:                     # noqa: BLE001 - surface the last error after the retry
            last = e
            time.sleep(3)
    raise RuntimeError(f"Gemma did not return usable coaching: {last}")


def autotune(take, offset, strength, ranges):
    """Pull your pitch toward the nearest note by `strength`, only inside `ranges` (song-clip seconds) if given."""
    x = take.astype(np.float64)
    f0, sp, ap = pw.wav2world(x, SR)
    t = np.arange(len(f0)) * 0.005 + offset          # pyworld frame period is 5 ms
    m = librosa.hz_to_midi(np.where(f0 > 0, f0, 1))
    target = m + strength * (np.round(m) - m)
    sel = f0 > 0
    if ranges:
        sel &= np.any([(t >= a) & (t <= b) for a, b in ranges], axis=0)
    f0 = np.where(sel, librosa.midi_to_hz(target), f0)
    return pw.synthesize(f0, sp, ap, SR).astype(np.float32)[: len(take)]


def mix(music, take, offset):
    """Your voice over the beat, starting at `offset` seconds of the clip."""
    out = music.copy()
    s = int(round(offset * SR))
    a, b = max(s, 0), min(len(out), s + len(take))
    v = take[a - s: b - s]
    v = v * (0.9 * np.sqrt(np.mean(music ** 2)) / (np.sqrt(np.mean(v ** 2)) + 1e-9))
    out[a:b] += v
    return (out / (np.abs(out).max() + 1e-9) * 0.95).astype(np.float32)


def run(song, take, start=0.0, dur=25.0, log=print):
    """The whole thing, one step at a time. `log` lets the app show each step."""
    log("Splitting the song into music and vocals (Demucs)...")
    music, vocals = separate(song, start, dur)
    log(f"The singer starts {singing_starts(vocals):.1f} s into the clip.")
    return coach_take(music, vocals, take, dur, log)


def coach_take(music, vocals, take, dur, log=print):
    """Everything after separation: align, measure, Gemma coaches, auto-tune, mix."""
    log("Lining your take up with the song...")
    offset = align(take, vocals)
    log(f"Your recording starts at {offset:+.2f} s of the clip.")
    log("Measuring your pitch against the singer's...")
    t, mr, my, diff, summary = compare(take, vocals, offset)
    img = chart(t, mr, my, diff)
    log(f"Asking Gemma 4 ({MODEL}) to coach you...")
    c = coach(img, summary, dur)
    log(f"Gemma says auto-tune strength {c['autotune']:.2f} on {c['fix_ranges'] or 'the whole take'}.")
    tuned = autotune(take, offset, c["autotune"], c["fix_ranges"])
    return {"music": music, "vocals": vocals, "offset": offset, "summary": summary, "chart": img, "coach": c,
            "raw_mix": mix(music, take, offset), "tuned_mix": mix(music, tuned, offset)}


if __name__ == "__main__":
    # Self-check on a fake take with a known answer: the separated vocal, cut 0.7 s in and sung 40 cents flat.
    import sys
    import soundfile as sf
    song = sys.argv[1] if len(sys.argv) > 1 else "songs/sample_song.wav"
    music, vocals = separate(song, 0, 25)
    take = librosa.effects.pitch_shift(vocals[int(0.7 * SR):], sr=SR, n_steps=-0.4)
    off = align(take, vocals)
    assert abs(off - 0.7) < 0.1, f"alignment off: {off}"
    t, mr, my, diff, summary = compare(take, vocals, off)
    assert summary["avg_abs_cents"] is not None and 25 < summary["avg_abs_cents"] < 60, summary
    print("self-check ok: offset", round(off, 2), "s, avg off", summary["avg_abs_cents"], "cents")
    r = run(song, take, 0, 25)
    Path("out").mkdir(exist_ok=True)
    r["chart"].save("out/chart.png")
    sf.write("out/tuned_mix.wav", r["tuned_mix"], SR)
    print(json.dumps(r["coach"], ensure_ascii=False, indent=1))
