"""Anatono web app: streamlit run app.py"""
import io
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st

import anatono as A

st.set_page_config(page_title="Anatono", page_icon="🎤", layout="centered")
st.title("🎤 Anatono")
st.write("**Sing over any song, and Gemma 4 coaches your voice.** Upload a song, sing the part over the music, "
         "and get a pitch chart, coaching in English and Urdu, and your take auto-tuned over the beat.")
st.caption(f"AI: `{A.MODEL}` (Gemma 4, open weights, Apache 2.0) through the Gemini API · "
           "Demucs for splitting the song · open source, MIT")


def wav_bytes(y):
    buf = io.BytesIO()
    sf.write(buf, y, A.SR, format="WAV")
    return buf.getvalue()


# 1. Song -> music + vocals
st.header("1. Pick a song")
SONGS = Path(__file__).with_name("songs")
found = sorted(p.name for p in SONGS.glob("*") if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac", ".ogg"})
pick = st.selectbox("A song from the `songs/` folder", ["(upload one instead)"] + found,
                    index=1 if found else 0, help="Run `python3 make_sample.py` to get copyright-free samples.")
upload = pick.startswith("(")
song = st.file_uploader("Upload a song", type=["mp3", "wav", "m4a", "flac", "ogg"]) if upload else None
path = None if upload else str(SONGS / pick)
if song:
    with tempfile.NamedTemporaryFile(suffix="." + song.name.rsplit(".", 1)[-1], delete=False) as f:
        f.write(song.getvalue())
        path = f.name
if path:
    total = librosa.get_duration(path=path)
    c1, c2 = st.columns(2)
    start = c1.number_input("Start at (seconds)", 0.0, max(0.0, total - 10), 0.0, 1.0)
    dur = c2.slider("Length (seconds)", 10, 30, 20)
    if st.button("Split the song", type="primary"):
        with st.spinner("Demucs is separating the music from the vocals..."):
            music, vocals = A.separate(path, start, dur)
        st.session_state.update(music=music, vocals=vocals, dur=dur, starts=A.singing_starts(vocals))
        st.session_state.pop("result", None)

if "music" in st.session_state:
    st.audio(wav_bytes(st.session_state.music), format="audio/wav")
    st.caption("☝️ The music without the singer. Play it in your headphones.")
    with st.expander("Hear the singer's part on its own"):
        st.audio(wav_bytes(st.session_state.vocals), format="audio/wav")
    st.info(f"The singer comes in **{st.session_state.starts:.1f} s** into this clip. Start singing there.")

    # 2. Record
    st.header("2. Sing it")
    st.write("Headphones on. Press **record**, then press **play** on the music above, and sing along.")
    take = st.audio_input("Your take", sample_rate=A.SR)

    # 3. Coach
    if take and st.button("Coach me", type="primary"):
        y, sr = sf.read(io.BytesIO(take.getvalue()), dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        if sr != A.SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=A.SR)
        with st.status("Anatono is listening...", expanded=True) as s:
            try:
                st.session_state.result = A.coach_take(st.session_state.music, st.session_state.vocals, y,
                                                       st.session_state.dur, log=st.write)
                s.update(label="Done", state="complete")
            except Exception as e:  # noqa: BLE001 - show the user what failed
                s.update(label="Something went wrong", state="error")
                st.error(str(e))

if "result" in st.session_state:
    r = st.session_state.result
    c = r["coach"]
    st.header("3. Your coaching")
    st.metric("Score from Gemma", f"{c['score']}/100")
    st.success(c.get("praise", ""))
    for fix in c["fixes"]:
        st.write("•", fix)
    st.markdown(f"<div dir='rtl' style='font-size:1.15em'>{c.get('urdu', '')}</div>", unsafe_allow_html=True)
    st.image(r["chart"], caption="What Gemma looked at: grey = original singer, blue = you, red = more than 50 cents off")
    st.subheader("Hear yourself")
    st.write("Your take over the beat:")
    st.audio(wav_bytes(r["raw_mix"]), format="audio/wav")
    where = "everywhere" if not c["fix_ranges"] else ", ".join(f"{a:.0f}–{b:.0f} s" for a, b in c["fix_ranges"])
    st.write(f"Auto-tuned (Gemma chose strength **{c['autotune']:.2f}**, {where}):")
    st.audio(wav_bytes(r["tuned_mix"]), format="audio/wav")
    st.download_button("Download the auto-tuned mix", wav_bytes(r["tuned_mix"]), "anatono_mix.wav", "audio/wav")
    with st.expander("The numbers code measured (what Gemma was given)"):
        st.json(r["summary"])
