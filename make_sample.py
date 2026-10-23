"""Make a copyright-free test song (macOS): a sung vocal from the built-in "Good News" voice over a synth beat.
Usage: python3 make_sample.py  ->  songs/sample_song.wav (vocals start at 2.0 s)"""
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 44100
Path("songs").mkdir(exist_ok=True)
subprocess.run(["say", "-v", "Good News", "-o", "songs/_voice.aiff",
                "Jhol jhol jhol, mera dil hai tera, sun le meri sada, raat bhi hai, chaand bhi hai, aaja mere paas"],
               check=True)
v, _ = librosa.load("songs/_voice.aiff", sr=SR, mono=True)
n = len(v) + 4 * SR
t = np.arange(n) / SR
chords = [(220, 277.2, 329.6), (196, 246.9, 293.7), (174.6, 220, 261.6), (196, 246.9, 293.7)]
pad = np.zeros(n)
for i in range(n // (2 * SR) + 1):
    a, b = i * 2 * SR, min((i + 1) * 2 * SR, n)
    pad[a:b] = sum(np.sin(2 * np.pi * f * t[a:b]) for f in chords[i % 4]) / 3
kick = np.zeros(n)
for k in range(0, n, SR // 2):
    m = min(4000, n - k)
    kick[k:k + m] += np.sin(2 * np.pi * 60 * np.arange(m) / SR) * np.exp(-np.arange(m) / 600)
vocal = np.zeros(n)
vocal[2 * SR: 2 * SR + len(v)] = v
song = 0.25 * pad + 0.4 * kick + 0.8 * vocal
song /= np.abs(song).max() * 1.1
sf.write("songs/sample_song.wav", np.stack([song, song], 1), SR)
Path("songs/_voice.aiff").unlink()
print("songs/sample_song.wav:", round(n / SR, 1), "s, vocals start at 2.0 s")
