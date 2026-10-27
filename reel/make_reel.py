"""Build the "Asthma Alley After the Toll" Instagram reel.

Renders a 1080x1920, 30 fps, ~34 s vertical video with an original
jersey-club style beat and synced sound effects, all generated here
(no samples, no licensed audio). Facts come from report.html and the
map data embedded in part1-map.html.

    python3 reel/make_reel.py          # writes reel/asthma-alley-reel.mp4

Needs numpy, scipy, pillow, pycairo and ffmpeg.
"""
import json
import math
import os
import re
import subprocess
import sys
import wave
from multiprocessing import Pool

import cairo
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "asthma-alley-reel.mp4")
COVER = os.path.join(HERE, "cover.png")
TMP = os.environ.get("REEL_TMP", HERE)

W, H, FPS = 1080, 1920, 30
BPM = 140
BEAT = 60 / BPM
TOTAL_BEATS = 80
DUR = TOTAL_BEATS * BEAT
SR = 44100

FONT_DIR = "/usr/share/fonts/opentype/inter/"
F_BLACK = FONT_DIR + "InterDisplay-Black.otf"
F_XBOLD = FONT_DIR + "InterDisplay-ExtraBold.otf"
F_BOLD = FONT_DIR + "InterDisplay-Bold.otf"
F_SEMI = FONT_DIR + "InterDisplay-SemiBold.otf"
F_EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

# palette
BG = (9, 12, 22)
INK = (245, 246, 250)
MUTED = (150, 160, 182)
YEL = (255, 210, 63)
RED = (255, 69, 58)
BLUE = (64, 156, 255)
GRN = (52, 211, 120)


def b2t(b):
    return b * BEAT


# ---------------------------------------------------------------- data

def load_data():
    s = open(os.path.join(ROOT, "part1-map.html"), encoding="utf-8").read()
    i = s.index("window.DATA=") + len("window.DATA=")
    d, _ = json.JSONDecoder().raw_decode(s[i:])
    return d


DATA = load_data()
LAT0 = 40.75
COSL = math.cos(math.radians(LAT0))


def proj(lon, lat):
    return (lon + 74.0) * COSL, lat - LAT0


def rings_of(geom):
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    else:
        polys = geom["coordinates"]
    out = []
    for poly in polys:
        for ring in poly:
            a = np.array([proj(x, y) for x, y in ring])
            out.append(a)
    return out


CD = {}
for f in DATA["cd"]["features"]:
    p = f["properties"]
    CD[p["id"]] = {"p": p, "rings": rings_of(f["geometry"])}
CRZ = [r for f in DATA["crz"]["features"] for r in rings_of(f["geometry"])]

TOP15 = sorted(CD, key=lambda k: -(CD[k]["p"]["asthma_ed_adult_aa"] or 0))[:15]
assert all(CD[k]["p"]["no2_pct_rel"] > 0 for k in TOP15)
# flagged districts outside the MTA's place-based mitigation areas (report, Part 1)
NO_MONEY = [110, 207, 303, 305, 316, 317, 414]
assert set(NO_MONEY) <= set(TOP15)
MOTT = 201
mott_p = CD[MOTT]["p"]
ASTHMA_X = mott_p["asthma_ed_adult_aa"] / DATA["city"]["asthma_ed_adult_aa"]  # ~3.8

BRUCK = np.array([proj(x, y) for x, y in DATA["corridor"]["bruck"]])
DEEGAN = np.array([proj(x, y) for x, y in DATA["corridor"]["deegan"]])
SCHOOLS = [proj(s["lon"], s["lat"]) for s in DATA["schools"] if s["d_hwy"] < 300]

# rough truck bypass (illustration only, like the map's own layer)
BYPASS = np.array([proj(*c) for c in [
    (-74.000, 40.858), (-73.9525, 40.8517), (-73.930, 40.8465), (-73.906, 40.8452),
    (-73.885, 40.8405), (-73.862, 40.8345), (-73.842, 40.8290), (-73.832, 40.8180),
    (-73.829, 40.8020), (-73.828, 40.7850), (-73.826, 40.7700)]])
SHORTCUT = np.array([proj(*c) for c in [
    (-74.040, 40.738), (-74.005, 40.750), (-73.975, 40.748), (-73.945, 40.744), (-73.910, 40.742)]])

MTA = {}
for m in DATA["mta"]:
    name = "RFK Bridge" if m["name"].startswith("RFK") else m["name"]
    MTA.setdefault(name, []).append(m["fy"])
MTA_TILES = sorted(((n, max(v)) for n, v in MTA.items()), key=lambda x: -x[1])
assert len(MTA_TILES) == 9 and sum(v > 0 for _, v in MTA_TILES) == 7


# ---------------------------------------------------------------- easing

def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 3 * x * x - 2 * x * x * x


def ease_out_back(x, s=2.2):
    x = clamp(x)
    x -= 1
    return 1 + x * x * ((s + 1) * x + s)


def pop(t, t0, dur=0.28):
    """scale/alpha for an element that pops in at t0"""
    if t < t0:
        return 0.0, 0.0
    u = (t - t0) / dur
    return max(0.001, ease_out_back(u)), clamp(u * 2.5)


# ---------------------------------------------------------------- audio

def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def tone(freq, dur, kind="sine", phase=0.0):
    t = np.arange(int(dur * SR)) / SR
    if callable(freq):
        ph = 2 * np.pi * np.cumsum(freq(t)) / SR + phase
    else:
        ph = 2 * np.pi * freq * t + phase
    if kind == "sine":
        return np.sin(ph)
    if kind == "saw":
        return 2 * ((ph / (2 * np.pi)) % 1.0) - 1
    if kind == "square":
        return np.sign(np.sin(ph))
    if kind == "tri":
        return 2 * np.abs(2 * ((ph / (2 * np.pi)) % 1.0) - 1) - 1
    raise ValueError(kind)


def lp(x, fc, order=2):
    b, a = signal.butter(order, min(fc, SR / 2 - 100) / (SR / 2), "low")
    return signal.lfilter(b, a, x)


def hp(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), "high")
    return signal.lfilter(b, a, x)


def bp(x, f1, f2, order=2):
    b, a = signal.butter(order, [f1 / (SR / 2), f2 / (SR / 2)], "band")
    return signal.lfilter(b, a, x)


def noise(dur, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, int(dur * SR))


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


class Track:
    def __init__(self):
        self.buf = np.zeros((int((DUR + 3) * SR), 2))

    def add(self, x, t, gain=1.0, pan=0.0):
        i = int(t * SR)
        if i < 0:
            x = x[-i:]
            i = 0
        n = min(len(x), len(self.buf) - i)
        if n <= 0:
            return
        l = math.cos((pan + 1) * math.pi / 4) * math.sqrt(2)
        r = math.sin((pan + 1) * math.pi / 4) * math.sqrt(2)
        self.buf[i:i + n, 0] += x[:n] * gain * l
        self.buf[i:i + n, 1] += x[:n] * gain * r


def kick(soft=False):
    d = 0.42
    f = lambda t: 42 + 160 * np.exp(-t * 32)
    x = tone(f, d) * env_exp(int(d * SR), 0.16)
    click = hp(noise(0.012, 1), 2500) * np.linspace(1, 0, int(0.012 * SR))
    x[:len(click)] += click * 0.5
    x = np.tanh(x * 1.8)
    return lp(x, 900) if soft else x


def clap():
    d = 0.35
    n = noise(d, 2)
    e = np.zeros(int(d * SR))
    for k, off in enumerate([0, 0.011, 0.022]):
        i = int(off * SR)
        e[i:] += env_exp(len(e) - i, 0.012 if k < 2 else 0.12)
    return bp(n * e, 900, 5000) * 1.6


def hat(open_=False):
    d = 0.25 if open_ else 0.05
    return hp(noise(d, 3), 7000) * env_exp(int(d * SR), 0.07 if open_ else 0.012)


def bass808(freq, dur):
    f = lambda t: freq * (1 + 0.5 * np.exp(-t * 40))
    x = tone(f, dur) * env_exp(int(dur * SR), 0.5)
    x *= np.minimum(1, (np.arange(len(x))[::-1]) / (0.02 * SR))
    return np.tanh(x * 2.2) * 0.8


def pluck(freq, dur=0.35, bright=1.0):
    t = np.arange(int(dur * SR)) / SR
    x = np.zeros_like(t)
    for n in range(1, 14):
        if freq * n > 12000:
            break
        x += np.sin(2 * np.pi * freq * n * t) / n * np.exp(-t * (6 + n * 5 / bright))
    return x * 0.6


def bell(freq, dur=1.2):
    t = np.arange(int(dur * SR)) / SR
    parts = [(1, 1, 2.2), (2.76, 0.5, 4), (5.4, 0.25, 7), (8.9, 0.12, 10)]
    return sum(a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t * k) for r, a, k in parts) * 0.5


def pad_chord(notes, dur, cutoff):
    x = np.zeros(int(dur * SR))
    for n in notes:
        for det in (-0.12, 0, 0.12):
            x += tone(midi(n + det), dur, "saw", phase=np.random.rand() * 6)
    x = lp(x, cutoff, 2) / (len(notes) * 3)
    a = int(0.04 * SR)
    x[:a] *= np.linspace(0, 1, a)
    x[-a:] *= np.linspace(1, 0, a)
    return x


# ---- sound effects

def sfx_whoosh(dur=0.5, up=True):
    n = noise(dur, 4)
    k = len(n)
    u = np.linspace(0, 1, k)
    if not up:
        u = 1 - u
    bands = [(300, 800), (800, 2000), (2000, 5000), (5000, 11000)]
    out = np.zeros(k)
    for j, (f1, f2) in enumerate(bands):
        centre = j / (len(bands) - 1)
        out += bp(n, f1, f2) * np.exp(-((u - centre) / 0.3) ** 2)
    return out * np.sin(np.pi * np.linspace(0, 1, k)) ** 2 * 1.6


def sfx_impact():
    d = 1.6
    f = lambda t: 30 + 70 * np.exp(-t * 8)
    sub = tone(f, d) * env_exp(int(d * SR), 0.45)
    crack = lp(noise(d, 5), 3000) * env_exp(int(d * SR), 0.08)
    return np.tanh((sub * 1.3 + crack * 0.7) * 1.5)


def sfx_riser(dur):
    k = int(dur * SR)
    u = np.linspace(0, 1, k)
    n = hp(noise(dur, 6), 800) * u ** 2
    f = lambda t: 200 + 1600 * (t / dur) ** 2
    s = tone(f, dur, "saw") * u ** 3 * 0.25
    return lp(n + s, 9000)


def sfx_reverse_swell(dur=0.6):
    x = sfx_impact()[: int(dur * SR)]
    c = lp(noise(dur, 7), 6000) * env_exp(int(dur * SR), 0.2)
    return (x * 0.5 + c * 0.6)[::-1]


def sfx_cash():
    x = bell(midi(88), 0.9) + bell(midi(93), 0.9) * 0.8
    out = np.zeros(int(1.0 * SR))
    out[: len(x)] += x * 0.5
    i = int(0.07 * SR)
    out[i:i + len(x)] += bell(midi(96), 0.9)[: len(out) - i] * 0.6
    drawer = bp(noise(0.12, 8), 1500, 6000) * env_exp(int(0.12 * SR), 0.03)
    out[: len(drawer)] += drawer * 0.8
    return out


def sfx_scratch():
    d = 0.45
    t = np.arange(int(d * SR)) / SR
    wob = 1 + 0.9 * np.sin(2 * np.pi * 9 * t) * np.exp(-t * 2)
    f = lambda tt: 380 * (1 + 0.9 * np.sin(2 * np.pi * 9 * tt))
    x = tone(f, d, "saw") * 0.3 + bp(noise(d, 9), 800, 4000) * 0.6 * np.abs(wob - 1)
    return lp(x, 5000) * env_exp(len(x), 0.25)


def sfx_pop(pitch=1.0):
    d = 0.09
    f = lambda t: (900 - 5000 * t) * pitch
    return tone(f, d) * env_exp(int(d * SR), 0.025)


def sfx_tick():
    return hp(noise(0.02, 10), 3000) * env_exp(int(0.02 * SR), 0.004) * 0.8


def sfx_buzz():
    d = 0.32
    x = tone(110, d, "square") * 0.5 + tone(116, d, "square") * 0.5
    return lp(x, 1800) * np.minimum(1, env_exp(int(d * SR), 0.3) * 3) * 0.6


def sfx_horn():
    out = np.zeros(int(0.75 * SR))
    for t0, d in [(0, 0.16), (0.24, 0.42)]:
        x = (tone(392, d, "square") + tone(494, d, "square") * 0.8)
        x = lp(x, 2200) * 0.35
        a = int(0.01 * SR)
        x[:a] *= np.linspace(0, 1, a)
        x[-a:] *= np.linspace(1, 0, a)
        i = int(t0 * SR)
        out[i:i + len(x)] += x
    return out


def sfx_chime():
    # two-tone "doors" chime, descending
    out = np.zeros(int(1.5 * SR))
    a, b = bell(midi(76), 0.9), bell(midi(72), 1.1)
    out[: len(a)] += a
    i = int(0.32 * SR)
    out[i:i + len(b)] += b
    return out * 0.8


def reverb(x, secs=1.6, mix=0.25):
    ir = noise(secs, 11) * env_exp(int(secs * SR), secs / 6)
    ir = lp(ir, 5000)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = signal.fftconvolve(x, ir)[: len(x)]
    return x * (1 - mix) + wet * mix


# Sound-effect cues, in beats. Visual animations key off the same beats.
SFX = [
    (0.05, "horn", 0.55), (1.5, "cash", 0.55), (4.0, "whoosh", 0.45),
    (7.6, "swell", 0.6), (8.0, "impact", 1.0), (11.0, "pop", 0.5),
    (13.0, "scratch", 0.75), (14.0, "impact", 0.6),
    (19.0, "whoosh", 0.5), (20.0, "impact", 0.7),
    (24.0, "whoosh", 0.5), (25.0, "cash", 0.45), (27.0, "whoosh_long", 0.55),
    (32.0, "whoosh", 0.45), (37.0, "impact", 0.85),
    (40.0, "buzz", 0.55), (40.0, "impact", 0.5),
    (45.0, "whoosh", 0.5), (45.7, "chime", 0.5),
    (54.0, "impact", 0.55), (60.0, "impact", 0.9), (61.0, "whoosh", 0.4),
    (67.0, "cash", 0.5), (70.0, "whoosh", 0.5), (74.0, "pop", 0.5),
    (76.0, "pop", 0.5), (78.0, "impact", 0.8),
]
# pops for the 9 crossing tiles and ticks for the 15 districts
TILE_BEATS = [14.6 + i * 0.36 for i in range(9)]
DISTRICT_BEATS = [33.0 + i * 0.27 for i in range(15)]
CARD_BEATS = [47.0, 49.0, 51.0]
for i, b in enumerate(TILE_BEATS):
    SFX.append((b, ("pop", 1.0 + i * 0.08), 0.45))
for b in DISTRICT_BEATS:
    SFX.append((b, "tick", 0.7))
for b in CARD_BEATS:
    SFX.append((b, "whoosh_short", 0.45))
    SFX.append((b + 0.2, ("pop", 1.2), 0.4))


def build_audio(path):
    np.random.seed(1)
    music = Track()
    fx = Track()

    minor = [[53, 56, 60], [49, 53, 56], [56, 60, 63], [51, 55, 58]]  # Fm Db Ab Eb
    minor_bass = [29, 37, 32, 39]
    major = [[56, 60, 63], [51, 55, 58], [53, 56, 60], [49, 53, 56]]  # Ab Eb Fm Db
    major_bass = [32, 39, 29, 37]

    k_full, k_soft, cl, hh, oh = kick(), kick(True), clap(), hat(), hat(True)
    kicks = []

    for bar in range(TOTAL_BEATS // 4):
        b0 = bar * 4
        t0 = b2t(b0)
        lift = b0 >= 60
        prog, bprog = (major, major_bass) if lift else (minor, minor_bass)
        chord, root = prog[bar % 4], bprog[bar % 4]
        section = ("intro" if b0 < 8 else "break" if 54 <= b0 < 60 else
                   "outro" if b0 >= 76 else "main")

        # pad (always), opens up after the drop
        cutoff = {"intro": 700 + 300 * bar, "break": 900, "outro": 2500}.get(section, 2600 if lift else 1900)
        music.add(pad_chord(chord + [chord[0] + 12], b2t(4), cutoff), t0, 0.42 if section != "main" else 0.32)

        if section == "intro":
            for q in range(4):
                if bar == 1 and q == 3:
                    continue  # leave a gap before the drop
                music.add(k_soft, t0 + b2t(q), 0.5)
            continue
        if section in ("break",):
            music.add(bass808(midi(root), b2t(4)), t0, 0.35)
            for q in (0, 2):
                music.add(k_soft, t0 + b2t(q), 0.35)  # heartbeat
            continue
        if section == "outro":
            if b0 == 76:
                for st in (0, 4, 8):
                    music.add(k_full, t0 + st * b2t(0.25), 0.9)
                    kicks.append(t0 + st * b2t(0.25))
                music.add(bass808(midi(root), b2t(2)), t0, 0.7)
            continue

        # jersey-club kick pattern in 16ths, with a triplet roll every 4th bar
        if bar % 4 == 3:
            steps = [0, 4, 8] + [12 + i * 4 / 3 for i in range(3)]
        elif bar % 2:
            steps = [0, 4, 8, 10, 12, 14]
        else:
            steps = [0, 4, 8, 11, 12]
        for st in steps:
            tk = t0 + st * b2t(0.25)
            music.add(k_full, tk, 0.95)
            kicks.append(tk)
        for q in (1, 3):
            music.add(cl, t0 + b2t(q), 0.55, pan=0.05)
        for e in range(8):
            music.add(hh if e % 2 else oh, t0 + b2t(e / 2 + 0.5 * 0), 0.18 if e % 2 else 0.1,
                      pan=0.3 if e % 2 else -0.3)
        if b0 >= 24:  # extra hat drive in the second half
            for s in range(16):
                if s % 4 == 2:
                    music.add(hh, t0 + s * b2t(0.25), 0.12, pan=-0.4)
        # 808 follows the first kicks of the bar
        music.add(bass808(midi(root), b2t(1.5)), t0, 0.75)
        music.add(bass808(midi(root), b2t(1.0)), t0 + b2t(2), 0.6)
        music.add(bass808(midi(root + (7 if bar % 2 else 12)), b2t(1.0)), t0 + b2t(3), 0.55)

        # chop-style plucks on the offbeats
        mel = chord + [chord[1] + 12, chord[2] + 12]
        pattern = [1, 3, 0, 2, 4, 2, 3, 1] if bar % 2 == 0 else [4, 2, 3, 1, 2, 0, 1, 3]
        for e in range(8):
            if e in (0, 4):
                continue
            n = mel[pattern[e]] + 12
            music.add(pluck(midi(n), 0.3, 1.6 if lift else 1.0), t0 + b2t(e / 2), 0.16,
                      pan=-0.35 if e % 2 else 0.35)
        if lift:
            top = chord[2] + 24
            for e, step in ((0, 0), (3, 2), (6, 4)):
                music.add(bell(midi(top + [0, -3, 2][e // 3]), 0.8), t0 + b2t(e / 2), 0.09, pan=0.2)

    # sidechain pump on everything musical except the kicks themselves
    gain = np.ones(len(music.buf))
    for tk in kicks:
        i = int(tk * SR)
        n = min(int(0.32 * SR), len(gain) - i)
        gain[i:i + n] = np.minimum(gain[i:i + n], 1 - 0.55 * np.exp(-np.arange(n) / (0.09 * SR)))
    drums = Track()
    for tk in kicks:
        drums.add(k_full, tk, 0.95)
    music.buf = (music.buf - drums.buf) * gain[:, None] + drums.buf

    # intro filter sweep already in pad; add riser into the drop
    fx.add(sfx_riser(b2t(3.5)), b2t(4.0), 0.35)
    fx.add(sfx_riser(b2t(2.0)), b2t(58.0), 0.35)

    gen = {
        "horn": sfx_horn, "cash": sfx_cash, "whoosh": lambda: sfx_whoosh(0.45),
        "whoosh_long": lambda: sfx_whoosh(1.1), "whoosh_short": lambda: sfx_whoosh(0.3),
        "swell": sfx_reverse_swell, "impact": sfx_impact, "scratch": sfx_scratch,
        "tick": sfx_tick, "buzz": sfx_buzz, "chime": sfx_chime,
    }
    for b, name, g in SFX:
        if g <= 0:
            continue
        if isinstance(name, tuple):
            x = sfx_pop(name[1])
        elif name == "pop":
            x = sfx_pop()
        else:
            x = gen[name]()
        offset = -0.6 if name == "swell" else 0.0
        fx.add(x, b2t(b) + offset, g)

    fxw = np.stack([reverb(fx.buf[:, c], 1.4, 0.2) for c in range(2)], 1)
    mix = music.buf * 0.8 + fxw * 0.9
    mix = mix[: int(DUR * SR) + int(0.5 * SR)]
    fade = int(0.5 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    mix /= np.percentile(np.abs(mix), 99.9)
    mix = np.tanh(mix * 0.9) / np.tanh(0.9)
    mix *= 0.89 / np.max(np.abs(mix))
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- text sprites

_fonts = {}
_sprites = {}


def font(path, size):
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]️?")


def emoji_img(ch, size):
    f = font(F_EMOJI, 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox())
    s = size * 1.05 / im.height
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)


def line_img(segs, size, path):
    """segs: list of (text, color, bg_or_None)"""
    f = font(path, size)
    asc, desc = f.getmetrics()
    lh = asc + desc
    pieces = []
    for text, col, bg in segs:
        parts = []
        pos = 0
        for m in EMOJI_RE.finditer(text):
            if m.start() > pos:
                parts.append(("t", text[pos:m.start()]))
            parts.append(("e", m.group(0)[0]))
            pos = m.end()
        if pos < len(text):
            parts.append(("t", text[pos:]))
        for kind, val in parts:
            if kind == "t":
                w = int(math.ceil(f.getlength(val)))
                im = Image.new("RGBA", (w + 4, lh), (0, 0, 0, 0))
                ImageDraw.Draw(im).text((2, 0), val, font=f, fill=col)
            else:
                e = emoji_img(val, size)
                im = Image.new("RGBA", (e.width + 4, lh), (0, 0, 0, 0))
                im.paste(e, (2, int(asc - e.height * 0.86)), e)
            pieces.append((im, bg))
    padx, pady = int(size * 0.22), int(size * 0.06)
    total = sum(im.width for im, _ in pieces) + 2 * padx
    out = Image.new("RGBA", (total + 8, lh + 2 * pady + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    x = padx + 4
    # merge runs that share a highlight into one pill
    runs = []
    for im, bg in pieces:
        if runs and runs[-1][0] == bg:
            runs[-1][1].append(im)
        else:
            runs.append((bg, [im]))
    for bg, ims in runs:
        w = sum(i.width for i in ims)
        if bg:
            d.rounded_rectangle([x - padx, 4 + int(lh * 0.06), x + w + padx, 4 + lh + 2 * pady - int(lh * 0.02)],
                                radius=int(size * 0.18), fill=bg)
        for i in ims:
            out.paste(i, (x, 4 + pady), i)
            x += i.width
    return out


def block(lines, size, path=F_BLACK, gap=0.0, shadow=True, align="c"):
    """lines: list of segs or str; returns cached RGBA sprite"""
    key = (repr(lines), size, path, gap, shadow, align)
    if key in _sprites:
        return _sprites[key]
    ims = []
    for ln in lines:
        if isinstance(ln, str):
            ln = [(ln, INK, None)]
        ims.append(line_img(ln, size, path))
    wmax = max(i.width for i in ims)
    step = [i.height + int(size * gap) for i in ims]
    hh = sum(step)
    m = 40
    out = Image.new("RGBA", (wmax + 2 * m, hh + 2 * m), (0, 0, 0, 0))
    y = m
    for i, s in zip(ims, step):
        x = m + (wmax - i.width) // 2 if align == "c" else m
        out.paste(i, (x, y), i)
        y += s
    if shadow:
        a = out.getchannel("A").filter(ImageFilter.GaussianBlur(14))
        sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
        sh.putalpha(a.point(lambda v: int(v * 0.85)))
        sh2 = Image.new("RGBA", out.size, (0, 0, 0, 0))
        sh2.paste(sh, (0, 6), sh)
        sh2.alpha_composite(out)
        out = sh2
    _sprites[key] = out
    return out


def put(img, spr, cx, cy, s=1.0, a=1.0):
    if s <= 0.01 or a <= 0.01:
        return
    if abs(s - 1) > 0.005:
        spr = spr.resize((max(1, int(spr.width * s)), max(1, int(spr.height * s))), Image.BILINEAR)
    if a < 0.995:
        spr = spr.copy()
        spr.putalpha(spr.getchannel("A").point(lambda v, a=a: int(v * a)))
    img.paste(spr, (int(cx - spr.width / 2), int(cy - spr.height / 2)), spr)


def put_pop(img, spr, cx, cy, t, t0, t1=None, fade=0.18):
    s, a = pop(t, t0)
    if t1 is not None and t > t1 - fade:
        a *= clamp((t1 - t) / fade)
    put(img, spr, cx, cy, s, a)


# ---------------------------------------------------------------- cairo drawing

def rgb(c, a=1.0):
    return (c[0] / 255, c[1] / 255, c[2] / 255, a)


class Cam:
    def __init__(self, lon, lat, k, cy=1000):
        self.X, self.Y = proj(lon, lat)
        self.k = k
        self.cy = cy

    def __call__(self, X, Y):
        return W / 2 + (X - self.X) * self.k, self.cy - (Y - self.Y) * self.k


def lerp_cam(a, b, u):
    u = ease_in_out(u)
    c = Cam(-74, LAT0, 1)
    c.X = a.X + (b.X - a.X) * u
    c.Y = a.Y + (b.Y - a.Y) * u
    c.k = math.exp(math.log(a.k) + (math.log(b.k) - math.log(a.k)) * u)
    c.cy = a.cy + (b.cy - a.cy) * u
    return c


def path_rings(ctx, rings, cam):
    for r in rings:
        xs, ys = cam(r[:, 0], r[:, 1])
        ctx.move_to(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            ctx.line_to(x, y)
        ctx.close_path()


def draw_map(ctx, cam, fills=None, base=(26, 34, 52), stroke=(48, 62, 92), alpha=1.0):
    fills = fills or {}
    for cid, c in CD.items():
        col, a = fills.get(cid, (base, 1.0))
        path_rings(ctx, c["rings"], cam)
        ctx.set_source_rgba(*rgb(col, a * alpha))
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(stroke, alpha))
        ctx.set_line_width(max(0.8, min(2.2, cam.k / 2500)))
        ctx.stroke()


def draw_crz(ctx, cam, glow=1.0, alpha=1.0):
    for w, a in ((26, 0.10), (14, 0.18), (5, 0.6)):
        path_rings(ctx, CRZ, cam)
        ctx.set_source_rgba(*rgb(YEL, a * glow * alpha))
        ctx.set_line_width(w)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.stroke()
    path_rings(ctx, CRZ, cam)
    ctx.set_source_rgba(*rgb(YEL, 0.28 * alpha))
    ctx.fill()


def draw_polyline(ctx, pts, cam, col, width, frac=1.0, glow=True, dash=None, alpha=1.0):
    if frac <= 0:
        return None
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    L = seg.sum() * frac
    xs, ys = cam(pts[:, 0], pts[:, 1])
    out = [(xs[0], ys[0])]
    acc = 0
    for i, s in enumerate(seg):
        if acc + s >= L:
            u = (L - acc) / s if s else 0
            out.append((xs[i] + (xs[i + 1] - xs[i]) * u, ys[i] + (ys[i + 1] - ys[i]) * u))
            break
        out.append((xs[i + 1], ys[i + 1]))
        acc += s
    layers = ((width * 4, 0.12), (width * 2.2, 0.25), (width, 1.0)) if glow else ((width, 1.0),)
    for w, a in layers:
        ctx.move_to(*out[0])
        for p in out[1:]:
            ctx.line_to(*p)
        ctx.set_source_rgba(*rgb(col, a * alpha))
        ctx.set_line_width(w)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        if dash:
            ctx.set_dash(dash)
        ctx.stroke()
        ctx.set_dash([])
    return out[-1]


BORO_LABELS = [("MANHATTAN", -73.968, 40.790), ("BRONX", -73.875, 40.852), ("BROOKLYN", -73.945, 40.648),
               ("QUEENS", -73.815, 40.718), ("STATEN IS.", -74.150, 40.585), ("NEW JERSEY", -74.075, 40.800)]


def label(ctx, txt, x, y, size=30, col=MUTED, a=0.75, spacing=4):
    ctx.select_font_face("Inter Display", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    widths = [ctx.text_extents(ch).x_advance for ch in txt]
    total = sum(widths) + spacing * (len(txt) - 1)
    cx = x - total / 2
    for ch, w in zip(txt, widths):
        ctx.move_to(cx, y + size * 0.35)
        ctx.set_source_rgba(*rgb(col, a))
        ctx.show_text(ch)
        cx += w + spacing


def boro_labels(ctx, cam, a=0.75, size=None):
    size = size or int(clamp(cam.k / 160, 22, 40))
    for name, lon, lat in BORO_LABELS:
        x, y = cam(*proj(lon, lat))
        if -200 < x < W + 200 and 0 < y < H:
            label(ctx, name, x, y, size, a=a)


def dot(ctx, x, y, r, col, a=1.0, glow=True):
    if glow:
        for rr, aa in ((r * 3.2, 0.12), (r * 2, 0.25)):
            ctx.arc(x, y, rr, 0, 2 * math.pi)
            ctx.set_source_rgba(*rgb(col, aa * a))
            ctx.fill()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.set_source_rgba(*rgb(col, a))
    ctx.fill()


def rrect(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()


def background(ctx, tint=(20, 26, 48), t=0.0):
    g = cairo.RadialGradient(W / 2, H * 0.42, 50, W / 2, H * 0.5, H * 0.75)
    g.add_color_stop_rgb(0, *[c / 255 for c in tint])
    g.add_color_stop_rgb(1, *[c / 255 for c in BG])
    ctx.set_source(g)
    ctx.paint()


# ---------------------------------------------------------------- scenes

CAM_NYC = Cam(-73.95, 40.705, 2350, 1010)
CAM_MAN = Cam(-73.975, 40.762, 5200, 1000)
CAM_BRONX = Cam(-73.900, 40.800, 6200, 930)
CAM_MOTT = Cam(-73.915, 40.812, 21000, 1040)
CAM_BRUCK = Cam(-73.912, 40.806, 30000, 1060)

RNG = np.random.default_rng(3)
PARTICLES = [(RNG.uniform(0, 2 * math.pi), RNG.uniform(0.4, 1.0), RNG.uniform(0, 1)) for _ in range(90)]


def crz_center():
    allp = np.vstack(CRZ)
    return allp[:, 0].mean(), allp[:, 1].mean()


CRZ_C = crz_center()
CD_CENTER = {cid: (np.vstack(c["rings"])[:, 0].mean(), np.vstack(c["rings"])[:, 1].mean()) for cid, c in CD.items()}


def scene_hook(ctx, img_cb, t):
    """beats 0-8"""
    b = t / BEAT
    cam = lerp_cam(CAM_NYC, CAM_MAN, clamp(b / 8) * 0.55)
    cam.cy += 230
    background(ctx)
    draw_map(ctx, cam)
    draw_crz(ctx, cam, glow=0.7 + 0.3 * math.sin(t * 8))
    boro_labels(ctx, cam, 0.55)
    if b >= 4:
        u = (b - 4) / 4
        cx, cy = cam(*CRZ_C)
        for ang, spd, ph in PARTICLES:
            r = ((u * spd + ph) % 1.0)
            rr = 30 + r * 520
            x, y = cx + math.cos(ang) * rr, cy + math.sin(ang) * rr * 1.1
            dot(ctx, x, y, 5, RED if ph > 0.5 else YEL, a=(1 - r) * clamp(u * 3), glow=False)

    def text(img):
        if b < 4:
            put_pop(img, block(["NYC now charges"], 84, F_XBOLD), W / 2, 330, t, 0.05, b2t(4))
            put_pop(img, block([[("$9", YEL, None)]], 280), W / 2, 540, t, b2t(1.5), b2t(4))
            put_pop(img, block(["to drive into Manhattan 🚕"], 72, F_XBOLD), W / 2, 740, t, b2t(2.3), b2t(4))
        else:
            put_pop(img, block(["So where did all"], 92), W / 2, 360, t, b2t(4))
            put_pop(img, block([[("that traffic go? ", INK, None), ("👀", INK, None)]], 92), W / 2, 480, t, b2t(5))
    return text


def scene_drop(ctx, t):
    """beats 8-13"""
    tl = t - b2t(8)
    background(ctx, tint=(52, 44, 10))
    u = clamp(tl / 0.6)
    val = int(round(11 * ease_out_cubic(u)))

    def text(img):
        put(img, block([[(f"−{val}%", YEL, None)]], 330), W / 2, 760, 1 + 0.15 * (1 - ease_out_cubic(tl / 0.35)))
        put_pop(img, block(["fewer vehicles entering", "the toll zone 🚗"], 66, F_XBOLD), W / 2, 1010, t, b2t(8.5))
        put_pop(img, block([[("It worked… right?", (20, 20, 20), YEL)]], 60, F_BLACK), W / 2, 1200, t, b2t(11))
    return text


def scene_but(ctx, t):
    """beats 13-19"""
    b = t / BEAT
    background(ctx, tint=(60, 14, 16))
    # 3x3 crossing tiles
    tiles = []
    for i, (name, v) in enumerate(MTA_TILES):
        col, row = i % 3, i // 3
        x, y = 90 + col * 310, 820 + row * 205
        s, a = pop(t, b2t(TILE_BEATS[i]))
        if a <= 0:
            continue
        up = v > 0
        ctx.save()
        ctx.translate(x + 145, y + 90)
        ctx.scale(s, s)
        rrect(ctx, -140, -88, 280, 176, 26)
        ctx.set_source_rgba(*rgb(RED if up else BLUE, 0.22 * a))
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(RED if up else BLUE, 0.95 * a))
        ctx.set_line_width(4)
        ctx.stroke()
        ctx.restore()
        tiles.append((x + 145, y + 90, name, up, s, a))

    def text(img):
        if b < 14:
            put(img, block([[("BUT", RED, None)]], 300), W / 2, 760, 1 + 0.25 * (1 - ease_out_cubic((t - b2t(13)) / 0.25)))
            return
        put_pop(img, block([[("7", RED, None), (" of ", INK, None), ("9", INK, None)]], 190), W / 2, 380, t, b2t(14))
        put_pop(img, block(["MTA bridges & tunnels", [("got ", INK, None), ("BUSIER", INK, RED), (" in 2025", INK, None)]],
                           60, F_XBOLD), W / 2, 600, t, b2t(14.3))
        for x, y, name, up, s, a in tiles:
            short = (name.replace(" Bridge", "").replace(" Tunnel", "").replace("Verrazzano-Narrows", "Verrazzano")
                     .replace("Bronx-Whitestone", "Whitestone").replace("Hugh L. ", ""))
            lbl = block([[("▲" if up else "▼", RED if up else BLUE, None)], short], 36, F_BOLD, gap=-0.1, shadow=False)
            put(img, lbl, x, y, s, a)
        if b >= 17.6:
            put_pop(img, block([[("▼ = the only 2 that go ", MUTED, None), ("INTO", BLUE, None), (" the zone", MUTED, None)]],
                               40, F_BOLD), W / 2, 1490, t, b2t(17.6))
    return text


def scene_fdr(ctx, t):
    """beats 19-24"""
    tl = t - b2t(20)
    background(ctx, tint=(54, 16, 20))
    val = 8.9 * ease_out_cubic(tl / 0.6) if tl > 0 else 0

    def text(img):
        put_pop(img, block(["The toll-FREE FDR Drive"], 70, F_XBOLD), W / 2, 520, t, b2t(19))
        if tl > 0:
            put(img, block([[(f"+{val:.1f}%", RED, None)]], 290), W / 2, 790,
                1 + 0.15 * (1 - ease_out_cubic(tl / 0.35)))
        put_pop(img, block([[("≈12,000", INK, RED), (" more vehicles a day", INK, None)]], 58, F_XBOLD),
                W / 2, 1040, t, b2t(21))
    return text


def scene_trucks(ctx, t):
    """beats 24-32"""
    b = t / BEAT
    cam = lerp_cam(CAM_NYC, CAM_BRONX, (b - 24) / 3)
    background(ctx)
    draw_map(ctx, cam)
    draw_crz(ctx, cam, 0.6)
    boro_labels(ctx, cam)
    u_old = clamp((b - 25) / 1.2)
    if u_old > 0:
        draw_polyline(ctx, SHORTCUT, cam, MUTED, 9, u_old, glow=False, dash=[18, 14],
                      alpha=1 - 0.6 * clamp((b - 27) / 1))
    if b >= 26.6:
        mx, my = cam(*SHORTCUT[2])
        s, a = pop(t, b2t(26.6))
        ctx.save()
        ctx.translate(mx, my)
        ctx.scale(s, s)
        for dx in (1, -1):
            ctx.move_to(-32 * dx, -32)
            ctx.line_to(32 * dx, 32)
        ctx.set_source_rgba(*rgb(RED, a))
        ctx.set_line_width(13)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.stroke()
        ctx.restore()
    u = clamp((b - 27) / 3.2)
    end = draw_polyline(ctx, BYPASS, cam, RED, 11, ease_in_out(u))
    if end:
        dot(ctx, end[0], end[1], 16, YEL)

    def text(img):
        put_pop(img, block([[("Trucks pay up to ", INK, None), ("$21.60", (20, 20, 20), YEL), (" 🚚", INK, None)]], 62),
                W / 2, 300, t, b2t(25), b2t(27))
        put_pop(img, block([[("so it's cheaper to go ", INK, None), ("AROUND", INK, RED)]], 66), W / 2, 300, t, b2t(27))
        put_pop(img, block([[("…right through the ", INK, None), ("Bronx", INK, RED)]], 66), W / 2, 1460, t, b2t(29.2))
        put(img, block([[("route drawn roughly, for illustration", MUTED, None)]], 28, F_SEMI, shadow=False),
            W / 2, 1560, 1, 0.8 * clamp((b - 27) * 2))
    return text


def scene_districts(ctx, t):
    """beats 32-45"""
    b = t / BEAT
    cam = lerp_cam(CAM_BRONX, Cam(-73.93, 40.70, 2550, 1020), (b - 32) / 1.5)
    background(ctx)
    fills = {}
    for i, cid in enumerate(TOP15):
        if b >= DISTRICT_BEATS[i]:
            flash = clamp(1 - (t - b2t(DISTRICT_BEATS[i])) / 0.25)
            col = RED
            a = 0.85
            if b >= 40 and cid not in NO_MONEY:
                col, a = (70, 80, 104), 0.9
            fills[cid] = (tuple(min(255, int(c + (255 - c) * flash)) for c in col), a)
    draw_map(ctx, cam, fills)
    boro_labels(ctx, cam, 0.6)
    if b >= 40:
        pulse = 0.5 + 0.5 * math.sin(t * 9)
        for cid in NO_MONEY:
            path_rings(ctx, CD[cid]["rings"], cam)
            ctx.set_source_rgba(*rgb(YEL, 0.5 + 0.5 * pulse))
            ctx.set_line_width(4)
            ctx.stroke()
    n_lit = sum(b >= x for x in DISTRICT_BEATS)

    def text(img):
        if b < 40:
            put_pop(img, block(["NYC's 15 highest-", "asthma neighborhoods"], 66), W / 2, 330, t, b2t(32), b2t(37))
            if b < 37:
                put(img, block([[(f"{n_lit}/15", RED, None)]], 90), W / 2, 1480, 1, clamp((b - 33) * 3))
            put_pop(img, block([[("ALL 15", INK, RED)], "did worse than the city", [("on NO₂ in 2025", INK, None)]],
                               64, gap=0.05), W / 2, 330, t, b2t(37))
        else:
            put_pop(img, block([[("7", YEL, None), (" of them get ", INK, None), ("$0", INK, RED)]], 96), W / 2, 300, t, b2t(40))
            put_pop(img, block(["of the MTA's $105M", "mitigation money"], 56, F_XBOLD), W / 2, 450, t, b2t(41))
    return text


def scene_mott(ctx, t):
    """beats 45-54"""
    b = t / BEAT
    cam = lerp_cam(Cam(-73.93, 40.70, 2550, 1020), CAM_MOTT, (b - 45) / 1.6)
    cam.cy = 1000 - 260 * ease_in_out((b - 46.6) / 1.2)
    background(ctx)
    draw_map(ctx, cam, {MOTT: (RED, 0.55 + 0.25 * math.sin(t * 7))})
    path_rings(ctx, CD[MOTT]["rings"], cam)
    ctx.set_source_rgba(*rgb(RED, 1))
    ctx.set_line_width(5)
    ctx.stroke()
    draw_polyline(ctx, BRUCK, cam, (255, 120, 80), 5, 1, glow=False, alpha=0.6)
    draw_polyline(ctx, DEEGAN, cam, (255, 120, 80), 5, 1, glow=False, alpha=0.6)
    # pin
    if b >= 45.6:
        px, py = cam(*CD_CENTER[MOTT])
        drop = 1 - ease_out_back((t - b2t(45.6)) / 0.35, 1.5)
        py -= drop * 300
        ctx.arc(px, py - 70, 34, 0, 2 * math.pi)
        ctx.move_to(px - 30, py - 56)
        ctx.line_to(px, py)
        ctx.line_to(px + 30, py - 56)
        ctx.set_source_rgba(*rgb(YEL, 1))
        ctx.fill()
        dot(ctx, px, py - 70, 13, BG, glow=False)

    def text(img):
        put_pop(img, block([[("Meet ", INK, None), ("Mott Haven", (20, 20, 20), YEL)]], 84), W / 2, 270, t, b2t(45))
        put_pop(img, block([[("South Bronx · #1 of 59 on our priority index", MUTED, None)]], 34, F_SEMI),
                W / 2, 375, t, b2t(45.4))
        cards = [
            [[("Asthma ER visits", MUTED, None)], [(f"{ASTHMA_X:.1f}×", RED, None), (" the city rate", INK, None)]],
            [[("Tree canopy", MUTED, None)], [("7.1%", RED, None), ("  vs city 21.1%", INK, None)]],
            [[("Schools near the highways", MUTED, None)], [("7", RED, None), (" within 150 m", INK, None)]],
        ]
        for i, (cb, card) in enumerate(zip(CARD_BEATS, cards)):
            y = 1140 + i * 150
            if b >= cb:
                s, a = pop(t, b2t(cb))
                x = W / 2 + (1 - ease_out_cubic((t - b2t(cb)) / 0.3)) * 500
                put(img, card_sprite(card), x, y, 1, a)
    return text


def card_sprite(card):
    key = ("card", repr(card))
    if key in _sprites:
        return _sprites[key]
    l1 = block([card[0]], 34, F_SEMI, shadow=False)
    l2 = block([card[1]], 64, F_BLACK, shadow=False)
    w, h = 880, 140
    im = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([20, 20, 20 + w, 20 + h], radius=30, fill=(14, 18, 32, 235), outline=(255, 69, 58, 255), width=4)
    im.paste(l1, (20 + w // 2 - l1.width // 2, 20 - 30), l1)
    im.paste(l2, (20 + w // 2 - l2.width // 2, 20 + 6), l2)
    _sprites[key] = im
    return im


def scene_break(ctx, t):
    """beats 54-60"""
    b = t / BEAT
    background(ctx, tint=(16, 20, 34))

    def text(img):
        put_pop(img, block(["Is the toll to blame?"], 76), W / 2, 640, t, b2t(54))
        put_pop(img, block([[("Not proven yet.", (20, 20, 20), INK)]], 70), W / 2, 800, t, b2t(55.5))
        put_pop(img, block(["The city found no significant", "change at its highway monitors."], 42, F_SEMI),
                W / 2, 960, t, b2t(56.3))
        put_pop(img, block([[("So measure where it hurts most.", YEL, None)]], 52, F_XBOLD), W / 2, 1130, t, b2t(57.6))
    return text


TREES = None


def scene_fix(ctx, t):
    """beats 60-70"""
    global TREES
    b = t / BEAT
    background(ctx, tint=(10, 46, 30))
    cam = lerp_cam(CAM_MOTT, CAM_BRUCK, (b - 60) / 3)
    cam.cy = 1080
    draw_map(ctx, cam, {MOTT: ((22, 60, 44), 1.0)}, base=(20, 30, 40), stroke=(40, 70, 60))
    u = clamp((b - 61) / 2.5)
    draw_polyline(ctx, BRUCK, cam, GRN, 16, ease_in_out(u))
    draw_polyline(ctx, DEEGAN, cam, GRN, 12, ease_in_out(clamp((b - 62) / 2)))
    if TREES is None:
        pts = []
        for line in (BRUCK, DEEGAN):
            for i in range(len(line) - 1):
                for v in np.linspace(0, 1, 6, endpoint=False):
                    p = line[i] + (line[i + 1] - line[i]) * v
                    for side in (-1, 1):
                        d = line[i + 1] - line[i]
                        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)
                        pts.append(p + nrm * side * 0.00075)
        TREES = pts
    n = int(len(TREES) * clamp((b - 62.5) / 2.5))
    for i, p in enumerate(TREES[:n]):
        x, y = cam(*p)
        dot(ctx, x, y, 7, (120, 235, 150), glow=False)
    if b >= 62:
        a = clamp((b - 62) * 2)
        x, y = cam(*BRUCK[len(BRUCK) * 2 // 3])
        label(ctx, "BRUCKNER BLVD", x + 20, y + 110, 30, GRN, a)
        x, y = cam(*DEEGAN[len(DEEGAN) // 3])
        label(ctx, "I-87 / MAJOR DEEGAN", x + 230, y - 10, 28, GRN, a)
    for sx, sy in SCHOOLS:
        x, y = cam(sx, sy)
        if 0 < x < W and 0 < y < H and b >= 64:
            dot(ctx, x, y, 11, YEL, a=clamp((b - 64) * 2))

    def text(img):
        put_pop(img, block([[("THE FIX ", GRN, None), ("🌳", INK, None)]], 120), W / 2, 300, t, b2t(60))
        put_pop(img, block(["A green corridor on", "Bruckner Boulevard"], 62), W / 2, 470, t, b2t(61))
        if b < 67:
            put_pop(img, block([[("~360 trees", (10, 30, 18), GRN), (" + 2.1 km", INK, None)], "evergreen barrier"],
                               58, F_BLACK), W / 2, 1450, t, b2t(64), b2t(67))
        put_pop(img, block([[("$10–19M", (10, 30, 18), GRN)], [("MTA already set aside ", INK, None), ("$24M", YEL, None)],
                            "for Mott Haven"], 56, F_BLACK), W / 2, 1420, t, b2t(67))
        if b >= 64:
            put(img, block([[("● ", YEL, None), ("schools", MUTED, None)]], 30, F_SEMI, shadow=False),
                170, 1590, 1, clamp((b - 64) * 2))
    return text


def scene_cta(ctx, t):
    """beats 70-80"""
    b = t / BEAT
    background(ctx, tint=(18, 30, 40))
    cam = Cam(-73.93, 40.70, 2300 + 120 * (b - 70), 1300)
    fills = {cid: (RED, 0.8) for cid in TOP15}
    fills[MOTT] = (GRN, 0.9)
    draw_map(ctx, cam, fills, alpha=0.35 * clamp((b - 70) / 2))

    def text(img):
        put_pop(img, block(["Clean air", "shouldn't depend", [("on your ", INK, None), ("zip code.", INK, RED)]], 96,
                           gap=0.02), W / 2, 640, t, b2t(70))
        put_pop(img, block([[("Send this to a New Yorker ", INK, None), ("🗽", INK, None)]], 54, F_XBOLD),
                W / 2, 1000, t, b2t(74))
        put_pop(img, block([[("Full map + report → ", MUTED, None), ("link in bio", YEL, None)]], 46, F_BOLD),
                W / 2, 1110, t, b2t(76))
        put_pop(img, block([[("CUNY Datathon 2026 · public data", MUTED, None)]], 30, F_SEMI, shadow=False),
                W / 2, 1200, t, b2t(76.5))
    return text


SCENES = [
    (0, 8, scene_hook), (8, 13, scene_drop), (13, 19, scene_but), (19, 24, scene_fdr),
    (24, 32, scene_trucks), (32, 45, scene_districts), (45, 54, scene_mott),
    (54, 60, scene_break), (60, 70, scene_fix), (70, 80, scene_cta),
]
CUTS = [s[0] for s in SCENES[1:]]
IMPACTS = [b for b, n, g in SFX if n == "impact"]

# ---------------------------------------------------------------- frame pipeline

_vignette = None
_grain = None


def overlays():
    global _vignette, _grain
    if _vignette is None:
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2)
        _vignette = np.clip(1 - 0.55 * d ** 2.2, 0.35, 1).astype(np.float32)[..., None]
        rng = np.random.default_rng(0)
        _grain = [np.repeat(np.repeat(rng.normal(0, 3, (H // 2, W // 2)), 2, 0), 2, 1).astype(np.float32)]
    return _vignette, _grain


def render(fi):
    t = fi / FPS
    b = t / BEAT
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(surf)
    for b0, b1, fn in SCENES:
        if b0 <= b < b1:
            break
    if fn is scene_hook:
        text = fn(ctx, None, t)
    else:
        text = fn(ctx, t)
    buf = np.ndarray((H, W, 4), np.uint8, surf.get_data())
    img = Image.fromarray(buf[:, :, [2, 1, 0]].copy())
    text(img)

    # top progress bar (retention) and brand tag
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([60, 120, W - 60, 128], radius=4, fill=(255, 255, 255, 40))
    d.rounded_rectangle([60, 120, 60 + (W - 120) * clamp(t / DUR), 128], radius=4,
                        fill=GRN if b >= 60 else YEL)
    tag = block([[("ASTHMA ALLEY", INK, None), ("  ·  AFTER THE TOLL", MUTED, None)]], 28, F_BOLD, shadow=False)
    put(img, tag, W / 2, 168)

    # beat-synced camera punch, shake on impacts, flash on cuts
    zoom = 1.0
    if 8 <= b < 54 or 60 <= b < 78:
        ph = (b % 1.0)
        zoom += 0.012 * math.exp(-ph * 6)
    shake = 0.0
    for ib in IMPACTS:
        dt = t - b2t(ib)
        if 0 <= dt < 0.35:
            shake = max(shake, 22 * (1 - dt / 0.35))
            zoom += 0.05 * (1 - dt / 0.35) ** 2
    arr = np.asarray(img).astype(np.float32)
    if zoom > 1.001 or shake > 0:
        rng = np.random.default_rng(fi)
        ox, oy = rng.uniform(-shake, shake, 2) if shake else (0, 0)
        cw, ch = W / zoom, H / zoom
        x0, y0 = (W - cw) / 2 + ox, (H - ch) / 2 + oy
        img = img.transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32)
    vig, grain = overlays()
    arr *= vig
    # static grain: adds texture without costing bitrate the way per-frame noise does
    arr += grain[0][..., None]
    for c in CUTS:
        dt = t - b2t(c)
        if 0 <= dt < 0.12:
            arr += (1 - dt / 0.12) * 160
    if t > DUR - 0.4:
        arr *= clamp((DUR - t) / 0.4)
    return np.clip(arr, 0, 255).astype(np.uint8).tobytes()


def main():
    wav = os.path.join(TMP, "reel_audio.wav")
    print("audio…", flush=True)
    build_audio(wav)
    n = int(DUR * FPS)
    print(f"video: {n} frames…", flush=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "19",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
           "-movflags", "+faststart", "-shortest", OUT]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as pool:
        for i, frame in enumerate(pool.imap(render, range(n), chunksize=4)):
            ff.stdin.write(frame)
            if i % 60 == 0:
                print(f"  {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    # cover frame: the "ALL 15" beat
    fi = int(b2t(38.5) * FPS)
    Image.frombytes("RGB", (W, H), render(fi)).save(COVER)
    print("wrote", OUT, COVER)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "frames":
        for bt in map(float, sys.argv[2:]):
            fi = int(b2t(bt) * FPS)
            Image.frombytes("RGB", (W, H), render(fi)).save(os.path.join(TMP, f"f_{bt:05.1f}.png"))
    else:
        main()
