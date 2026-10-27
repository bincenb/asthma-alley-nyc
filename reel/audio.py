"""Soundtrack for the Asthma Alley reel: an original phonk-style beat plus sound
effects, all synthesized here so the reel carries no licensed audio.

Timeline (120 BPM, one bar = 2 s) is shared with make_reel.py through CUES.
"""
import wave

import numpy as np
from scipy import signal

SR = 44100
BPM = 120
BEAT = 60 / BPM
STEP = BEAT / 4          # one 16th note
BAR = BEAT * 4
DUR = 32.0

rng = np.random.default_rng(7)


# ---------------------------------------------------------------- helpers
def tt(d):
    return np.arange(int(d * SR)) / SR


def mf(m):
    return 440.0 * 2 ** ((m - 69) / 12)


NOTE = {n: i for i, n in enumerate(
    ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"])}


def nf(name):
    """'Ab5' -> Hz."""
    return mf(12 * (int(name[-1]) + 1) + NOTE[name[:-1]])


def _sos(kind, f):
    return signal.butter(2, f, kind, fs=SR, output="sos")


def lp(x, f):
    return signal.sosfilt(_sos("lowpass", f), x, axis=0)


def hp(x, f):
    return signal.sosfilt(_sos("highpass", f), x, axis=0)


def bp(x, lo, hi):
    return signal.sosfilt(_sos("bandpass", [lo, hi]), x, axis=0)


def noise(d):
    return rng.standard_normal(int(d * SR))


class Bus:
    def __init__(self):
        self.x = np.zeros((int(DUR * SR), 2))

    def add(self, s, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i >= len(self.x) or i < 0:
            return
        s = s[: len(self.x) - i]
        if s.ndim == 1:
            a = (pan + 1) * np.pi / 4
            s = np.stack([s * np.cos(a), s * np.sin(a)], 1) * np.sqrt(2)
        self.x[i:i + len(s)] += s * gain


# ---------------------------------------------------------------- drums
def kick(d=0.32):
    t = tt(d)
    f = 48 + 170 * np.exp(-t / 0.025)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13)
    x += 0.35 * np.exp(-t / 0.003) * np.sin(2 * np.pi * 2200 * t)
    return np.tanh(2.5 * x) / np.tanh(2.5)


def b808(f0, d, f_from=None, glide=0.06):
    """Distorted 808 bass note; the saturation adds harmonics phones can play."""
    t = tt(d)
    f = np.full_like(t, f0)
    if f_from:
        f = f0 + (f_from - f0) * np.exp(-t / glide)
    f = f + 60 * np.exp(-t / 0.012)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    env = np.exp(-t / 0.9) * np.clip((d - t) / 0.04, 0, 1)
    return np.tanh(3.2 * x * env) * 0.8


def clap():
    t = tt(0.4)
    env = sum((t >= o) * np.exp(-np.clip(t - o, 0, None) / 0.006) * 0.7
              for o in (0, 0.009, 0.019))
    env = env + (t >= 0.027) * np.exp(-np.clip(t - 0.027, 0, None) / 0.13)
    return bp(noise(0.4) * env, 900, 5000) * 0.75


def hat(open_=False):
    d = 0.25 if open_ else 0.06
    t = tt(d)
    return hp(noise(d), 7500) * np.exp(-t / (0.09 if open_ else 0.018)) * 0.22


def snare():
    t = tt(0.25)
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04)
    return (body * 0.6 + bp(noise(0.25), 1500, 8000) * np.exp(-t / 0.07) * 0.35) * 0.9


def cowbell(f, d=0.4):
    t = tt(d)
    x = (signal.square(2 * np.pi * f * t) + signal.square(2 * np.pi * f * 1.4813 * t)) * 0.5
    x = bp(x, f * 0.75, min(f * 4.5, 15000))
    return x * (np.exp(-t / 0.11) * 0.75 + np.exp(-t / 0.01) * 0.25)


# ---------------------------------------------------------------- tonal
def pad(freqs, d, cutoff=1400):
    t = tt(d)
    x = np.zeros_like(t)
    for f in freqs:
        for det in (-0.13, 0.0, 0.12):
            x += signal.sawtooth(2 * np.pi * f * 2 ** (det / 12) * t + rng.uniform(0, 6))
    x = lp(x / (3 * len(freqs)), cutoff)
    return x * np.clip(t / 0.2, 0, 1) * np.clip((d - t) / 0.25, 0, 1)


def pluck(f, d=0.35):
    t = tt(d)
    x = sum(np.sin(2 * np.pi * f * k * t) * np.exp(-t * (6 + 5 * k)) / k for k in range(1, 6))
    return x * np.clip(t / 0.003, 0, 1)


def bell(f, d=1.2):
    t = tt(d)
    return sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / dec)
               for r, a, dec in ((1, 1, 0.5), (2.0, 0.5, 0.35), (2.76, 0.4, 0.25), (5.4, 0.25, 0.12)))


# ---------------------------------------------------------------- sfx
def whoosh(d=0.5, up=True):
    t = tt(d)
    env = np.sin(np.pi * t / d) ** 2
    x = bp(noise(d), 300, 4000) * env
    f = (400 if up else 2400) * (6 if up else 1 / 6) ** (t / d)
    x += 0.15 * np.sin(2 * np.pi * np.cumsum(f) / SR) * env
    return x * 0.9


def riser(d):
    t = tt(d)
    r = t / d
    f = 180 * 12 ** r
    tone = sum(np.sin(2 * np.pi * np.cumsum(f * k) / SR) / k for k in (1, 1.5, 2))
    return (hp(noise(d), 2500) * r ** 2 * 0.8 + 0.25 * tone * r ** 1.5)


def impact():
    t = tt(1.6)
    f = 32 + 60 * np.exp(-t / 0.08)
    boom = np.tanh(2 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.55))
    crack = lp(noise(1.6), 3000) * np.exp(-t / 0.08)
    return boom * 0.9 + crack * 0.6


def pop(f0=1300, f1=380):
    t = tt(0.09)
    f = f1 + (f0 - f1) * np.exp(-t / 0.015)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.03) * 0.8


def tick():
    t = tt(0.03)
    return bp(noise(0.03), 2500, 6000) * np.exp(-t / 0.004) * 1.2 + \
        0.3 * np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.006)


def buzz():
    t = tt(0.45)
    gate = ((t < 0.16) | ((t > 0.21) & (t < 0.42))).astype(float)
    x = signal.square(2 * np.pi * 98 * t) + signal.square(2 * np.pi * 104 * t)
    return lp(x * gate, 2500) * 0.35


def cha_ching():
    out = np.zeros(int(1.2 * SR))
    for o, f in ((0, 2093), (0.09, 2637)):
        b = bell(f, 1.0) * 0.5
        i = int(o * SR)
        out[i:i + len(b)] += b[: len(out) - i]
    t = tt(1.2)
    out += hp(noise(1.2), 6000) * np.exp(-t / 0.25) * 0.4
    return out


def sparkle():
    out = np.zeros(int(1.0 * SR))
    for i, m in enumerate([84, 86, 88, 91, 93, 96, 98, 100]):
        b = np.sin(2 * np.pi * mf(m) * tt(0.4)) * np.exp(-tt(0.4) / 0.12) * 0.3
        j = int(i * 0.035 * SR)
        out[j:j + len(b)] += b
    return out


def horn(d=0.65):
    t = tt(d)
    x = signal.sawtooth(2 * np.pi * 185 * t) + signal.sawtooth(2 * np.pi * 233 * t)
    return lp(x, 1600) * np.clip(t / 0.03, 0, 1) * np.clip((d - t) / 0.08, 0, 1) * 0.3


def stamp():
    t = tt(0.35)
    f = 50 + 70 * np.exp(-t / 0.03)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.1) +
            bp(noise(0.35), 400, 3000) * np.exp(-t / 0.03) * 0.7)


def scratch():
    """Back-and-forth playback of a short buzzy sample, like a record scratch."""
    src_t = tt(0.6)
    src = bp(signal.sawtooth(2 * np.pi * 150 * src_t) + signal.sawtooth(2 * np.pi * 226 * src_t)
             + 0.6 * noise(0.6), 300, 3500)
    t = tt(0.42)
    pos = 0.2 + 0.14 * np.sin(2 * np.pi * 4.8 * t) * np.exp(-t / 0.6)
    y = np.interp(pos * SR, np.arange(len(src)), src)
    speed = np.abs(np.gradient(pos)) * SR
    return y * np.clip(speed / 0.8, 0, 1) * 0.7


def crackle(d):
    x = np.zeros(int(d * SR))
    idx = rng.integers(0, len(x), int(d * 40))
    x[idx] = rng.uniform(-1, 1, len(idx))
    return hp(x, 1500) * 0.5 + lp(noise(d), 600) * 0.01


def reverb(x, secs=1.8, damp=4500):
    t = tt(secs)
    ir = np.stack([lp(noise(secs), damp) * np.exp(-t / (secs / 4)) for _ in range(2)], 1)
    ir /= np.sqrt((ir ** 2).sum(0))
    return np.stack([signal.fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)


# ---------------------------------------------------------------- cue sheet
# Times (s) where the visuals cut or hit. make_reel.py imports this.
CUES = dict(scratch=1.5, beat_in=2.0, drop=10.0, breakdown=22.0, huge=23.0,
            fix=24.0, build=28.25, cta=29.0)

MINOR = {  # F minor: i, VI, III, VII
    "Fm": (["F3", "Ab3", "C4", "F4"], "F2"),
    "Db": (["Db3", "F3", "Ab3", "Db4"], "Db2"),
    "Ab": (["Ab3", "C4", "Eb4", "Ab4"], "Ab2"),
    "Eb": (["Eb3", "G3", "Bb3", "Eb4"], "Eb2"),
}
COWBELL = [  # (16th step, note) per bar, four-bar loop over Fm Db Ab Eb
    [(0, "F5"), (3, "F5"), (6, "Ab5"), (8, "C6"), (10, "Ab5"), (11, "F5"), (14, "Eb5")],
    [(0, "Db5"), (3, "Db5"), (6, "F5"), (8, "Ab5"), (10, "F5"), (12, "Eb5"), (14, "C5")],
    [(0, "Ab5"), (3, "Ab5"), (6, "C6"), (8, "Eb6"), (10, "C6"), (11, "Ab5"), (14, "G5")],
    [(0, "G5"), (3, "G5"), (6, "Bb5"), (8, "Eb6"), (10, "Bb5"), (12, "G5"), (14, "F5")],
]
LOOP = ["Fm", "Db", "Ab", "Eb"]


def build(path):
    drums, bass, mel, pads, sfx, wet = (Bus() for _ in range(6))
    kicks = []

    def chord_at(bar_idx, prog=LOOP):
        return prog[bar_idx % len(prog)]

    # ---- intro: pad + vinyl, then a tape stop into the scratch
    pads.add(pad([nf(n) for n in MINOR["Fm"][0]], 1.5, 900), 0, 0.55)
    for i in range(6):
        mel.add(pluck(nf(["F5", "Ab5", "C6", "Ab5", "Eb5", "C5"][i])), i * STEP * 2, 0.25)
    sfx.add(crackle(1.5), 0, 0.5)

    # ---- verse (2-10 s) and drop (10-22 s)
    for b in range(1, 11):
        t0 = b * BAR
        drop = t0 >= CUES["drop"]
        name = chord_at(b - 1)
        notes, root = MINOR[name]
        if t0 + BAR <= 9.75 or drop:
            pads.add(pad([nf(n) for n in notes], BAR, 1800 if drop else 1100), t0, 0.5 if drop else 0.4)
        kpat = [0, 7, 10] if b % 2 else [0, 3, 8, 11]
        for s in kpat:
            t = t0 + s * STEP
            if 9.75 <= t < 10.0:
                continue
            drums.add(kick(), t, 1.0)
            kicks.append(t)
            nxt = [x for x in kpat if x > s]
            dur = ((nxt[0] if nxt else 16) - s) * STEP
            bass.add(b808(nf(root), dur, f_from=nf(root) * (1.5 if s == 0 else 1.0)), t, 0.85 if drop else 0.6)
        for s in (4, 12):
            t = t0 + s * STEP
            if not 9.75 <= t < 10.0:
                drums.add(clap(), t, 0.8)
                wet.add(clap(), t, 0.25)
        # hats: 8ths in the verse, 16ths with rolls in the drop
        for s in range(0, 16, 1 if drop else 2):
            t = t0 + s * STEP
            if t >= 9.75 and t < 10:
                continue
            drums.add(hat(), t, 0.35 if s % 4 else 0.5, pan=0.25)
        if drop and b % 2 == 0:
            for k in range(6):  # triplet roll on the last beat
                drums.add(hat(), t0 + 3 * BEAT + k * BEAT / 6, 0.3, pan=-0.2)
        if drop:
            for s, n in COWBELL[(b - 1) % 4]:
                t = t0 + s * STEP
                cb = cowbell(nf(n))
                mel.add(cb, t, 0.55, pan=0.1)
                mel.add(cowbell(nf(n) / 2), t, 0.25, pan=-0.1)
                wet.add(cb, t, 0.3)

    # snare roll + riser into the drop, then a gap
    for i in range(12):
        drums.add(snare(), 9.0 + i * (0.75 / 12), 0.25 + 0.05 * i)
    sfx.add(riser(2.0), 8.0, 0.55)

    # ---- breakdown (22-24): dark filtered beat, handled after mixing
    for t in (22.0, 23.0):
        bass.add(b808(nf("F2"), 0.95, f_from=nf("F3")), t, 0.9)
        drums.add(kick(), t, 0.9)
        kicks.append(t)

    # ---- uplift (24-30): Ab major I-V-vi-IV, bright plucks
    UP = {"Ab": (["Ab3", "C4", "Eb4"], "Ab2", ["Ab5", "C6", "Eb6", "C6"]),
          "Eb": (["Eb3", "G3", "Bb3"], "Eb2", ["G5", "Bb5", "Eb6", "Bb5"]),
          "Fm": (["F3", "Ab3", "C4"], "F2", ["F5", "Ab5", "C6", "Ab5"]),
          "Db": (["Db3", "F3", "Ab3"], "Db2", ["F5", "Ab5", "Db6", "Ab5"])}
    for i, name in enumerate(["Ab", "Eb", "Fm", "Db"]):
        t0 = 24.0 + i * BAR
        notes, root, arp = UP[name]
        last = name == "Db"
        pads.add(pad([nf(n) for n in notes] + [nf(notes[0]) * 2], BAR, 2600), t0, 0.5)
        for s in range(16):
            if last and s >= 12:
                break
            mel.add(pluck(nf(arp[s % 4])), t0 + s * STEP, 0.32, pan=0.3 if s % 2 else -0.3)
            wet.add(pluck(nf(arp[s % 4])), t0 + s * STEP, 0.2)
        for s in ([0, 10] if not last else [0]):
            drums.add(kick(), t0 + s * STEP, 0.85)
            kicks.append(t0 + s * STEP)
            bass.add(b808(nf(root), 0.9), t0 + s * STEP, 0.6)
        for s in (4, 12):
            if not (last and s == 12):
                drums.add(clap(), t0 + s * STEP, 0.65)
                wet.add(clap(), t0 + s * STEP, 0.3)
        for s in range(0, 16, 2):
            if not (last and s >= 12):
                drums.add(hat(s % 4 == 2), t0 + s * STEP, 0.25, pan=0.25)
    # tail back into the loop: the intro pad again
    pads.add(pad([nf(n) for n in MINOR["Fm"][0]], 1.4, 900), 30.6, 0.35)

    # ---- sound effects, synced to make_reel.py
    S = [
        (whoosh(0.35), 0.0, 0.35), (pop(), 0.05, 0.5),
        (scratch(), 1.5, 0.9), (impact(), 1.5, 0.35),
        (whoosh(0.6), 2.0, 0.6), (pop(900, 300), 2.55, 0.7),
        (whoosh(0.4), 3.0, 0.4), (bell(1568, 1.0), 3.9, 0.35),
        (whoosh(0.4), 5.0, 0.45), (pop(), 6.5, 0.4),
        (whoosh(0.5), 7.0, 0.5), (horn(), 7.6, 0.5),
        (pop(), 8.5, 0.5), (pop(), 8.85, 0.5),
        (impact(), 10.0, 1.0), (hp(noise(1.5), 4000) * np.exp(-tt(1.5) / 0.4), 10.0, 0.35),
        (whoosh(0.4), 12.0, 0.4), (riser(0.8), 12.2, 0.3), (stamp(), 13.0, 0.9),
        (whoosh(0.4), 14.0, 0.4), (stamp(), 14.5, 0.9),
        (whoosh(0.4), 16.0, 0.4), (riser(0.8), 16.2, 0.25), (bell(1318, 1.0), 17.0, 0.3),
        (whoosh(0.4), 18.0, 0.4), (bell(1760, 1.0), 19.75, 0.35),
        (whoosh(0.4), 20.5, 0.4), (buzz(), 21.0, 0.9),
        (impact(), 22.0, 0.5), (pop(700, 250), 22.5, 0.6),
        (impact(), 23.0, 0.9), (stamp(), 23.5, 0.8),
        (whoosh(0.6), 24.0, 0.5), (sparkle(), 24.1, 0.7),
        (pop(), 25.5, 0.55), (pop(1500, 500), 25.85, 0.55), (pop(1700, 600), 26.2, 0.55),
        (whoosh(0.4), 27.0, 0.4), (cha_ching(), 27.75, 0.6), (stamp(), CUES["build"], 1.0),
        (whoosh(0.5), 29.0, 0.45), (pop(), 29.5, 0.5), (bell(2093, 1.0), 30.25, 0.3),
        (whoosh(0.5, up=False), 31.5, 0.35),
    ]
    S += [(tick(), 3.1 + i * 0.08, 0.5) for i in range(10)]           # counter
    S += [(tick(), 5.5 + i * 0.11, 0.6) for i in range(9)]            # bridge tiles
    S += [(tick(), 14.05 + i * 0.008, 0.08) for i in range(59)]       # ranking bars
    S += [(tick(), 18.4 + i * 0.09, 0.6) for i in range(15)]          # 15 districts
    S += [(pop(1100 + 40 * i, 400), 24.4 + i * 0.07, 0.25) for i in range(8)]  # schools/route
    for s, t, g in S:
        sfx.add(s, t, g)
    wet.add(sparkle(), 24.1, 0.4)

    # ---- mix
    sc = np.ones(len(pads.x))  # sidechain pump on pads and melody
    tsec = np.arange(len(sc)) / SR
    for k in kicks:
        i = int(k * SR)
        seg = tsec[i:i + int(0.3 * SR)] - k
        sc[i:i + len(seg)] = np.minimum(sc[i:i + len(seg)], 1 - 0.55 * np.exp(-seg / 0.09))
    music = drums.x * 0.7 + bass.x * 0.32 + (mel.x * 0.8 + pads.x * 0.6) * sc[:, None]
    music += reverb(wet.x) * 0.45

    # tape stop on the intro (1.25-1.5 s) and silence until the scratch
    a, b = int(1.25 * SR), int(1.5 * SR)
    rate = np.linspace(1, 0, b - a) ** 1.5
    pos = a + np.cumsum(rate)
    music[a:b] = np.stack([np.interp(pos, np.arange(len(music)), music[:, c]) for c in range(2)], 1)
    music[b:int(2.0 * SR)] *= 0

    # breakdown: low-pass the beat for 22-24 s with short crossfades
    dark = lp(music, 380) * 1.6
    w = np.zeros(len(music))
    x0, x1 = int(22.0 * SR), int(24.0 * SR)
    f = int(0.05 * SR)
    w[x0:x1] = 1
    w[x0 - f:x0] = np.linspace(0, 1, f)
    w[x1 - f:x1] = np.linspace(1, 0, f)
    music = music * (1 - w[:, None]) + dark * w[:, None]

    mix = music * 0.8 + sfx.x * 0.9 + reverb(sfx.x * 0.15, 1.2) * 0.5
    fade = np.clip((DUR - tsec) / 0.4, 0, 1)
    mix *= fade[:, None]
    mix = np.tanh(mix * 1.1)
    mix /= np.abs(mix).max() / 0.95
    pcm = (mix * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "soundtrack.wav")
