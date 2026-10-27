"""Synthesizes the reel's soundtrack (no licensed audio): drone, 100 BPM pulse, whooshes on cuts, hits on reveals."""
import numpy as np, wave, sys

SR = 44100
DUR = 38.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(7)
out = np.zeros(N)

def band(x, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))

def add(start, sig, gain=1.0):
    i = int(start * SR); j = min(N, i + len(sig))
    if i < N: out[i:j] += gain * sig[: j - i]

def env(n, a, d):  # attack/decay envelope, seconds
    x = np.arange(n) / SR
    return np.minimum(1, x / max(a, 1e-4)) * np.exp(-x / d)

# drone: A minor, brightening to A major for the "fix" section
def pad(freqs, t0, t1, g):
    n = int((t1 - t0) * SR); x = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * f * x + k) * (0.6 if f > 150 else 1) for k, f in enumerate(freqs))
    fade = np.minimum(1, np.minimum(x / 0.6, (t1 - t0 - x) / 0.6))
    add(t0, s * fade * (0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * x)), g)
pad([55, 82.41, 110, 130.81, 164.81], 0, 29.2, 0.05)
pad([55, 82.41, 110, 138.59, 164.81, 220], 28.8, DUR, 0.05)

# kick on the beat, hats on the off-beat
n = int(0.35 * SR); x = np.arange(n) / SR
kick = np.sin(2 * np.pi * (45 * x + (120 - 45) * 0.04 * (1 - np.exp(-x / 0.04)))) * np.exp(-x / 0.12)
hat = band(rng.standard_normal(int(0.05 * SR)), 6000, 14000) * env(int(0.05 * SR), 0.001, 0.012)
beat = 0.6
for b in np.arange(3.2, 34.0, beat):
    add(b, kick, 0.55)
    if b >= 10.8: add(b + beat / 2, hat, 0.5)

# whoosh into each cut
for cut in [3.2, 7.4, 10.8, 16.0, 20.5, 24.5, 29.0, 34.0]:
    n = int(0.4 * SR); x = np.arange(n) / SR
    w = band(rng.standard_normal(n), 300, 5000) * (x / 0.4) ** 2.5
    add(cut - 0.4, w, 0.12)

# impacts
def boom(at, g):
    n = int(1.2 * SR); x = np.arange(n) / SR
    s = np.sin(2 * np.pi * (38 * x + 60 * 0.08 * (1 - np.exp(-x / 0.08)))) * np.exp(-x / 0.45)
    s += band(rng.standard_normal(n), 80, 3000) * np.exp(-x / 0.07) * 0.5
    add(at, s, g)
boom(0.12, 0.9)    # 22%
boom(21.3, 0.7)    # $0
boom(24.6, 0.45)   # zoom to Mott Haven

def blip(at, f, g, d=0.09):
    n = int(0.3 * SR); x = np.arange(n) / SR
    add(at, np.sin(2 * np.pi * f * x) * env(n, 0.002, d), g)
# "...right?" down-bend
n = int(0.35 * SR); x = np.arange(n) / SR
add(1.5, np.sin(2 * np.pi * (660 * x - 500 * x * x)) * env(n, 0.005, 0.12), 0.25)
# one tick per district lighting up (15 rising notes)
scale = [440, 494, 523, 587, 659, 698, 784, 880, 988, 1047, 1175, 1319, 1397, 1568, 1760]
for i, f in enumerate(scale):
    blip(16.0 + 0.4 + i * 0.11 + 0.12, f, 0.12)
# trees: soft major arpeggio as the corridor draws
for i, f in enumerate([440, 554.37, 659.26, 880, 1108.73, 1318.5]):
    blip(29.5 + i * 0.22, f, 0.12, 0.35)

# gentle fade-out and limiting
out *= np.minimum(1, (DUR - t) / 1.5)
out = np.tanh(out * 1.4) / np.tanh(1.4)
out *= 0.89 / np.abs(out).max()
pcm = (np.repeat(out[:, None], 2, 1) * 32767).astype('<i2')
with wave.open(sys.argv[1] if len(sys.argv) > 1 else 'soundtrack.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
