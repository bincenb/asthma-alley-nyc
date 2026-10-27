"""Render the Asthma Alley Instagram reel (1080x1920, 30 fps, 32 s).

    python3 reel/make_reel.py            # writes reel/asthma-alley-reel.mp4 + cover.png

Map geometry and numbers come from window.DATA in part1-map.html; the
soundtrack comes from audio.py. Needs numpy, scipy, pillow and ffmpeg.
"""
import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio  # noqa: E402

W, H, FPS = 1080, 1920, 30
DUR = audio.DUR
SS = 2  # map supersampling

FD = "/usr/share/fonts/opentype/inter/"
BLACK, XBOLD, BOLD, SEMI, MED = (FD + f"InterDisplay-{w}.otf" for w in
                                 ("Black", "ExtraBold", "Bold", "SemiBold", "Medium"))
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

BG = (11, 12, 16)
WHITE = (245, 245, 240)
GREY = (150, 152, 160)
RED = (255, 59, 48)
ORANGE = (255, 140, 40)
YELLOW = (255, 214, 10)
GREEN = (52, 211, 120)
MINT = (120, 255, 170)


# ---------------------------------------------------------------- data
def load_data():
    html = open(os.path.join(HERE, "..", "part1-map.html"), encoding="utf-8").read()
    i = html.index("window.DATA=") + len("window.DATA=")
    obj, _ = json.JSONDecoder().raw_decode(html[i:])
    return obj


D = load_data()
LAT0 = 40.7
KX = math.cos(math.radians(LAT0))


def proj(lon, lat):
    return ((lon + 74.05) * KX * 1000.0, (40.95 - lat) * 1000.0)


def rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [np.array([proj(*p) for p in poly[0]]) for poly in polys]


CD = [dict(p=f["properties"], rings=rings(f["geometry"])) for f in D["cd"]["features"]]
BORO = [dict(name=f["properties"]["name"], rings=rings(f["geometry"])) for f in D["boro"]["features"]]
CRZ = rings(D["crz"]["features"][0]["geometry"])
BRUCK = np.array([proj(*p) for p in D["corridor"]["bruck"]])
DEEGAN = np.array([proj(*p) for p in D["corridor"]["deegan"]])
SCHOOLS = [proj(s["lon"], s["lat"]) for s in D["schools"] if s["d_hwy"] <= 300]
CITY = D["city"]
MOTT = next(c for c in CD if c["p"]["id"] == 201)["p"]
TOP15 = sorted(CD, key=lambda c: -(c["p"]["asthma_ed_adult_aa"] or 0))[:15]
TOP15_IDS = [c["p"]["id"] for c in TOP15]
assert all(c["p"]["no2_pct_rel"] > 0 for c in TOP15)  # the "15 of 15" claim

# Rough truck bypass: GWB -> Cross Bronx -> Bruckner -> Whitestone / Throgs Neck
ROUTE = np.array([proj(*p) for p in [
    (-74.02, 40.853), (-73.953, 40.851), (-73.93, 40.846), (-73.905, 40.845), (-73.887, 40.840),
    (-73.867, 40.834), (-73.848, 40.831), (-73.835, 40.826), (-73.829, 40.812), (-73.829, 40.795),
    (-73.83, 40.78)]])
ROUTE2 = np.array([proj(*p) for p in [(-73.835, 40.826), (-73.818, 40.822), (-73.794, 40.805),
                                      (-73.79, 40.795)]])

MTA_FY = [("Whitestone", 1.9), ("Cross Bay", 0.9), ("Henry Hudson", 0.6), ("Marine Pkwy", 1.8),
          ("RFK", 0.8), ("Throgs Neck", 0.4), ("Verrazzano", 0.5),
          ("Queens-Midtown\nTunnel", -1.1), ("Carey\nTunnel", -2.6)]
assert sum(v > 0 for _, v in MTA_FY) == 7


def P(lon, lat):
    return np.array(proj(lon, lat))


MANHATTAN_C = P(-73.97, 40.76)
NYC_C = P(-73.94, 40.70)
BRONX_C = P(-73.91, 40.82)
MOTT_C = P(-73.918, 40.808)


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x)
    return 3 * x * x - 2 * x * x * x


def pop_scale(dt, dur=0.18):
    """0 before dt=0, overshoots to ~1.18 then settles at 1."""
    if dt < 0:
        return 0.0
    x = clamp(dt / dur)
    return 1 + 0.25 * math.sin(math.pi * x) * (1 - x) if x < 1 else 1.0


def lerp(a, b, x):
    return a + (b - a) * x


def mix(c1, c2, x):
    return tuple(int(lerp(a, b, x)) for a, b in zip(c1, c2))


# ---------------------------------------------------------------- text
@lru_cache(None)
def font(path, size):
    return ImageFont.truetype(path, size)


def is_emoji(s):
    return any(ord(ch) > 0x2190 for ch in s) and not any(ch.isalnum() for ch in s)


@lru_cache(4096)
def word_img(text, size, path, color, stroke=0, scolor=(0, 0, 0)):
    if is_emoji(text):
        f = font(EMOJI, 109)
        im = Image.new("RGBA", (160 * len(text), 160))
        ImageDraw.Draw(im).text((10, 10), text, font=f, embedded_color=True)
        im = im.crop(im.getbbox())
        h = int(size * 0.95)
        return im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS)
    f = font(path, size)
    asc, desc = f.getmetrics()
    w = int(f.getlength(text)) + 2 * stroke + 4
    im = Image.new("RGBA", (w, asc + desc + 2 * stroke))
    ImageDraw.Draw(im).text((stroke + 2, stroke + asc), text, font=f, fill=color, anchor="ls",
                            stroke_width=stroke, stroke_fill=scolor)
    return im


def paste_center(frame, im, cx, cy, scale=1.0, alpha=1.0):
    if scale <= 0.01 or alpha <= 0.01:
        return
    if abs(scale - 1) > 0.01:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if alpha < 0.99:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    frame.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


class Caption:
    """Word-by-word kinetic caption. Markup: *word* = accent colour, [word] = highlight box."""

    def __init__(self, lines, t0, size=88, path=BLACK, y=360, gap=0.07, accent=YELLOW,
                 color=WHITE, hl_bg=YELLOW, hl_fg=(15, 15, 15), stroke=6, line_h=1.16, times=None):
        self.items = []
        k = 0
        ytop = y
        for li, line in enumerate(lines):
            words = []
            for raw in line.split(" "):
                style = "n"
                if raw.startswith("*") and raw.endswith("*"):
                    style, raw = "a", raw[1:-1]
                elif raw.startswith("[") and raw.endswith("]"):
                    style, raw = "h", raw[1:-1]
                raw = raw.replace("_", " ")
                fg = {"n": color, "a": accent, "h": hl_fg}[style]
                img = word_img(raw, size, path, fg, 0 if style == "h" else stroke)
                if style == "h":
                    pad = int(size * 0.18)
                    box = Image.new("RGBA", (img.width + 2 * pad, img.height + pad // 2))
                    ImageDraw.Draw(box).rounded_rectangle((0, 0, box.width - 1, box.height - 1),
                                                          radius=int(size * 0.18), fill=hl_bg)
                    box.alpha_composite(img, (pad, pad // 4))
                    img = box
                words.append(img)
            space = int(size * 0.26)
            total = sum(w.width for w in words) + space * (len(words) - 1)
            x = W / 2 - total / 2
            cy = ytop + size * line_h * li + size * 0.6
            for img in words:
                t = times[k] if times else t0 + k * gap
                self.items.append((img, x + img.width / 2, cy, t))
                x += img.width + space
                k += 1
        self.bottom = ytop + size * line_h * len(lines)

    def draw(self, frame, t, out=None):
        a = 1.0 if out is None else clamp(1 - (t - out) / 0.12)
        for img, cx, cy, tw in self.items:
            s = pop_scale(t - tw)
            if s > 0:
                paste_center(frame, img, cx, cy, s, a * clamp((t - tw) / 0.06))


def text(frame, s, cx, cy, size, path=BOLD, color=WHITE, scale=1.0, alpha=1.0, stroke=0):
    paste_center(frame, word_img(s, size, path, color, stroke), cx, cy, scale, alpha)


def chip(frame, label, cx, cy, size, fg, bg, t, alpha=1.0):
    s = pop_scale(t)
    if s <= 0:
        return
    img = word_img(label, size, BOLD, fg)
    pad = int(size * 0.55)
    box = Image.new("RGBA", (img.width + 2 * pad, int(img.height + pad * 0.9)))
    ImageDraw.Draw(box).rounded_rectangle((0, 0, box.width - 1, box.height - 1), radius=box.height // 2,
                                          fill=bg + (235,))
    box.alpha_composite(img, (pad, int(pad * 0.45)))
    paste_center(frame, box, cx, cy, s, alpha)


def rich_chip(frame, parts, cx, cy, size, bg, t):
    """Chip made of [(text, colour)] parts, emoji allowed."""
    s = pop_scale(t)
    if s <= 0:
        return
    imgs = [word_img(p, size, BOLD, c) for p, c in parts]
    gap = int(size * 0.3)
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    h = max(i.height for i in imgs)
    pad = int(size * 0.6)
    box = Image.new("RGBA", (w + 2 * pad, h + pad))
    ImageDraw.Draw(box).rounded_rectangle((0, 0, box.width - 1, box.height - 1), radius=int(size * 0.5),
                                          fill=bg + (235,))
    x = pad
    for i in imgs:
        box.alpha_composite(i, (x, (box.height - i.height) // 2))
        x += i.width + gap
    paste_center(frame, box, cx, cy, s)


# ---------------------------------------------------------------- backgrounds
@lru_cache(None)
def background(tint=(0, 0, 0), strength=0.0):
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt(((x - W / 2) / W) ** 2 + ((y - H * 0.42) / H) ** 2)
    glow = np.clip(1 - r * 1.6, 0, 1)[..., None] ** 1.5
    base = np.array(BG, float) + glow * np.array(tint, float) * strength
    arr = np.dstack([np.clip(base, 0, 255).astype(np.uint8), np.full((H, W), 255, np.uint8)])
    return Image.fromarray(arr, "RGBA")


def bg(tint=None, s=0.35):
    return background(tint or (0, 0, 0), s if tint else 0).copy()


VIGNETTE = None
GRAIN = None


def init_post():
    global VIGNETTE, GRAIN
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.75)) ** 2)
    VIGNETTE = np.clip(1.15 - 0.55 * r ** 2, 0.45, 1)[..., None].astype(np.float32)
    g = np.random.default_rng(3)
    GRAIN = [g.normal(0, 7, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]


# ---------------------------------------------------------------- map
class Cam:
    def __init__(self, c, z):
        self.c, self.z = np.asarray(c, float), z  # z = screen px per map unit

    def xy(self, pts, ss=1):
        return (np.asarray(pts) - self.c) * self.z * ss + np.array([W * ss / 2, H * ss / 2])


def cam_lerp(a, b, x):
    # zoom interpolates geometrically so the move feels constant
    return Cam(a.c + (b.c - a.c) * x, a.z * (b.z / a.z) ** x)


def asthma_color(v, k=1.0):
    x = clamp((v - 20) / 230)
    stops = [(28, 30, 38), (110, 40, 40), (220, 60, 40), (255, 140, 40), (255, 220, 120)]
    i = min(int(x * 4), 3)
    c = mix(stops[i], stops[i + 1], x * 4 - i)
    return mix((28, 30, 38), c, k)


def draw_map(frame, cam, fill=None, outline=(46, 50, 62), crz=0.0, glow_ids=(), boro_line=True):
    """fill: callable(props)->RGB or None. crz: 0..1 toll zone highlight."""
    cv = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)
    lw = max(1, int(1.2 * SS * min(2.5, cam.z / 2.2)))
    for c in CD:
        col = fill(c["p"]) if fill else (28, 30, 38)
        for r in c["rings"]:
            pts = [tuple(p) for p in cam.xy(r, SS)]
            d.polygon(pts, fill=col + (255,), outline=outline + (255,), width=lw)
    if boro_line:
        for b in BORO:
            for r in b["rings"]:
                d.line([tuple(p) for p in cam.xy(r, SS)], fill=(80, 86, 104, 255), width=lw + SS)
    if crz > 0:
        for r in CRZ:
            pts = [tuple(p) for p in cam.xy(r, SS)]
            d.polygon(pts, fill=GREEN + (int(150 * crz),), outline=MINT + (int(255 * crz),), width=lw * 2)
    small = cv.resize((W, H), Image.LANCZOS)
    if glow_ids or crz > 0:
        g = small.filter(ImageFilter.GaussianBlur(18))
        frame.alpha_composite(g)
    frame.alpha_composite(small)


def draw_line(frame, cam, pts, progress, color, width, glow=True):
    """Polyline revealed by arc length."""
    xy = cam.xy(pts, SS)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = seg.sum() * clamp(progress)
    out = [tuple(xy[0])]
    acc = 0
    for i, L in enumerate(seg):
        if acc + L >= total:
            f = (total - acc) / L if L else 0
            out.append(tuple(xy[i] + (xy[i + 1] - xy[i]) * f))
            break
        out.append(tuple(xy[i + 1]))
        acc += L
    if len(out) < 2:
        return None
    cv = Image.new("RGBA", (W * SS, H * SS))
    d = ImageDraw.Draw(cv)
    d.line(out, fill=color + (255,), width=width * SS, joint="curve")
    r = width * SS / 2
    for p in (out[0], out[-1]):
        d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=color + (255,))
    small = cv.resize((W, H), Image.LANCZOS)
    if glow:
        frame.alpha_composite(small.filter(ImageFilter.GaussianBlur(14)))
        frame.alpha_composite(small.filter(ImageFilter.GaussianBlur(5)))
    frame.alpha_composite(small)
    return np.array(out[-1]) / SS


def pin(frame, x, y, t, color=RED, size=46):
    s = pop_scale(t, 0.25)
    if s <= 0:
        return
    drop = (1 - ease(t / 0.25)) * 120
    d = ImageDraw.Draw(frame)
    r = size * s
    cy = y - r * 1.6 - drop
    d.polygon([(x - r * 0.62, cy + r * 0.6), (x + r * 0.62, cy + r * 0.6), (x, y - drop)], fill=color)
    d.ellipse((x - r, cy - r, x + r, cy + r), fill=color)
    d.ellipse((x - r * 0.4, cy - r * 0.4, x + r * 0.4, cy + r * 0.4), fill=WHITE)
    if t > 0.25:  # ripple
        k = ((t - 0.25) % 0.8) / 0.8
        rr = 20 + 90 * k
        d.ellipse((x - rr, y - rr * 0.45, x + rr, y + rr * 0.45), outline=color + (int(255 * (1 - k)),),
                  width=4)


# ---------------------------------------------------------------- smog
PART = np.random.default_rng(11).random((46, 4))


@lru_cache(None)
def blob(r, color):
    im = Image.new("RGBA", (r * 4, r * 4))
    ImageDraw.Draw(im).ellipse((r, r, r * 3, r * 3), fill=color + (90,))
    return im.filter(ImageFilter.GaussianBlur(r * 0.6))


def smog(frame, t, color=(255, 90, 60), n=46):
    for i in range(n):
        x0, y0, sp, sz = PART[i]
        x = (x0 * W + math.sin(t * 0.7 + i) * 40 + t * 30 * (sp - 0.5)) % (W + 200) - 100
        y = (y0 * H - t * (60 + 120 * sp)) % (H + 200) - 100
        r = int(30 + sz * 70) // 10 * 10
        im = blob(r, color)
        frame.alpha_composite(im, (int(x - im.width / 2), int(y - im.height / 2)))


# ---------------------------------------------------------------- scenes
CAM_NYC = Cam(NYC_C, 2.15)
CAM_MAN = Cam(P(-73.95, 40.77), 3.4)
CAM_BX = Cam(BRONX_C + np.array([0, 10]), 9.5)
CAM_ROUTE = Cam(P(-73.895, 40.815), 6.2)
CAM_MOTT = Cam(MOTT_C + np.array([0, 4]), 30.0)


def tag(frame, label="ASTHMA ALLEY · NYC", color=GREY):
    text(frame, label, W / 2, 210, 30, SEMI, color)


def s_hook(t):  # 0 - 1.5
    f = bg()
    cam = cam_lerp(CAM_NYC, CAM_MAN, ease_io(t / 1.5) * 0.35)
    draw_map(f, cam, crz=0.6 + 0.4 * math.sin(t * 9) ** 2)
    x, y = cam.xy(P(-73.99, 40.74))
    chip(f, "TOLL ZONE", x - 170, y + 40, 34, (10, 30, 18), MINT, t - 0.35)
    CAP_HOOK.draw(f, t)
    return f


def s_wait(t):  # 1.5 - 2.0
    f = s_hook(1.5)
    arr = np.asarray(f).astype(np.int16)
    arr[..., 0] = np.clip(arr[..., 0] + 40, 0, 255)
    f = Image.fromarray(arr.astype(np.uint8), "RGBA")
    text(f, "WAIT.", W / 2, H * 0.52, 260, BLACK, RED, pop_scale(t, 0.15), stroke=10)
    return f


def s_nothere(t):  # 2.0 - 3.0
    f = bg((255, 40, 40), 0.18)
    k = ease(t / 0.7)
    cam = cam_lerp(CAM_MAN, CAM_BX, k)
    bx = lambda p: (70, 28, 30) if p["id"] // 100 == 2 else (28, 30, 38)
    draw_map(f, cam, fill=lambda p: mix((28, 30, 38), bx(p), k))
    x, y = cam.xy(MOTT_C)
    pin(f, x, y, t - 0.5)
    CAP_NOTHERE.draw(f, t)
    if t > 0.6:
        chip(f, "THE SOUTH BRONX", x, y + 70, 34, WHITE, (120, 20, 20), t - 0.6)
    return f


def s_toll(t):  # 3.0 - 5.0
    f = bg(GREEN, 0.25)
    tag(f, "CARS ENTERING THE TOLL ZONE, 2025", MINT)
    k = ease((t - 0.1) / 0.8)
    v = round(11 * k)
    s = 1.0 + (0.12 * pop_scale(t - 0.9) if t > 0.9 else 0)
    text(f, f"−{v}%", W / 2, 820, 330, BLACK, MINT if t > 0.9 else WHITE, s)
    text(f, "fewer vehicles", W / 2, 1040, 64, BOLD, WHITE, alpha=clamp(t / 0.3))
    if t > 1.3:
        CAP_WIN.draw(f, t)
    return f


def s_bridges(t):  # 5.0 - 7.0
    f = bg((255, 60, 40), 0.2)
    CAP_BRIDGES.draw(f, t)
    d = ImageDraw.Draw(f)
    tw, th, gx, gy = 290, 200, 22, 22
    x0 = W / 2 - (3 * tw + 2 * gx) / 2
    y0 = 720
    for i, (name, v) in enumerate(MTA_FY):
        r, c = divmod(i, 3)
        tl = t - (0.5 + i * 0.11)
        s = pop_scale(tl) if tl > 0 else 0
        x, y = x0 + c * (tw + gx), y0 + r * (th + gy)
        lit = tl > 0
        col = (RED if v > 0 else GREEN) if lit else (40, 42, 52)
        cx, cy = x + tw / 2, y + th / 2
        sw, sh = tw * (0.92 + 0.08 * s if lit else 1), th * (0.92 + 0.08 * s if lit else 1)
        d.rounded_rectangle((cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2), radius=26,
                            fill=mix(BG, col, 0.28 if lit else 1), outline=col, width=4)
        lines = name.split("\n")
        for j, ln in enumerate(lines):
            text(f, ln, cx, cy - 40 + j * 34 - (len(lines) - 1) * 14, 30, SEMI, WHITE)
        if lit:
            text(f, f"{v:+.1f}%", cx, cy + 45, 62, BLACK, col, s)
    if t > 1.5:
        text(f, "full-year change in traffic vs 2024 · MTA", W / 2, 1450, 30, MED, GREY)
    return f


def s_bypass(t):  # 7.0 - 10.0
    f = bg((255, 120, 30), 0.15)
    zoom = ease_io((t - 2.0) / 0.75) if t > 2.0 else 0
    cam = cam_lerp(CAM_ROUTE, Cam(MOTT_C, 16), zoom * 0.5)
    draw_map(f, cam, fill=lambda p: (34, 32, 36) if p["id"] // 100 == 2 else (26, 28, 34))
    for nm, lon, lat in (("MANHATTAN", -73.955, 40.785), ("BRONX", -73.875, 40.86), ("QUEENS", -73.875, 40.745),
                         ("NJ", -74.0, 40.86)):
        x, y = cam.xy(P(lon, lat))
        text(f, nm, x, y, 30, BOLD, (120, 124, 138))
    p = ease_io((t - 0.15) / 1.4)
    head = draw_line(f, cam, ROUTE, p, ORANGE, 16)
    if t > 1.2:
        draw_line(f, cam, ROUTE2, ease((t - 1.2) / 0.5), ORANGE, 16)
    if head is not None and p < 1:
        paste_center(f, word_img("🚚", 90, BOLD, WHITE).transpose(Image.FLIP_LEFT_RIGHT), head[0], head[1] - 40)
    CAP_BYPASS.draw(f, t)
    rich_chip(f, [("Whitestone trucks", WHITE), ("+4.1%", ORANGE)], W / 2, 1330, 40, (40, 22, 10), t - 1.5)
    rich_chip(f, [("Throgs Neck trucks", WHITE), ("+2.1%", ORANGE)], W / 2, 1440, 40, (40, 22, 10), t - 1.85)
    if t > 2.75:  # pre-drop blackout
        f = bg()
        text(f, "welcome to…", W / 2, H * 0.5, 70, SEMI, WHITE, alpha=clamp((t - 2.75) / 0.08))
    return f


def s_drop(t):  # 10.0 - 12.0
    f = bg((255, 30, 20), 0.55)
    cam = Cam(BRONX_C + np.array([t * 3, 0]), 8.5 + t * 0.6)
    draw_map(f, cam, fill=lambda p: asthma_color(p["asthma_ed_adult_aa"] or 0, 0.85))
    dark = Image.new("RGBA", (W, H), (10, 4, 6, 120))
    f.alpha_composite(dark)
    smog(f, t + 10)
    beat = (t % 0.5) / 0.5
    s = 1 + 0.06 * (1 - ease(beat / 0.4))
    text(f, "ASTHMA", W / 2, H * 0.40, 230, BLACK, WHITE, s * pop_scale(t), stroke=10)
    text(f, "ALLEY", W / 2, H * 0.40 + 230, 270, BLACK, RED, s * pop_scale(t - 0.08), stroke=10)
    if t > 0.5:
        rich_chip(f, [("📍", WHITE), ("South Bronx, NYC", WHITE)], W / 2, H * 0.40 + 450, 44, (90, 10, 10),
                  t - 0.5)
    return f


def s_ratio(t):  # 12.0 - 14.0
    f = bg((255, 40, 40), 0.25)
    CAP_RATIO.draw(f, t)
    d = ImageDraw.Draw(f)
    k = ease((t - 0.25) / 0.75)
    maxw = 820
    x0 = (W - maxw) / 2
    bars = [("NYC average", CITY["asthma_ed_adult_aa"], (110, 114, 128)),
            ("Mott Haven", MOTT["asthma_ed_adult_aa"], RED)]
    for i, (lab, v, col) in enumerate(bars):
        y = 820 + i * 230
        text(f, lab, x0 + 4 + word_img(lab, 44, BOLD, WHITE).width / 2, y - 50, 44, BOLD, WHITE)
        w = maxw * v / 260 * k
        d.rounded_rectangle((x0, y, x0 + max(w, 30), y + 110), radius=22, fill=col)
        if k > 0.05:
            text(f, f"{v * k:.0f}", x0 + max(w, 30) - 80 if w > 200 else x0 + w + 80, y + 55, 58, BLACK,
                 WHITE if w > 200 else col)
    text(f, "adult asthma ER visits per 10,000 people", W / 2, 1300, 34, MED, GREY)
    if t > 1.0:
        s = pop_scale(t - 1.0, 0.2)
        text(f, "3.8×", W / 2, 1480, 210, BLACK, YELLOW, s * (1 + 0.6 * (1 - ease((t - 1.0) / 0.12))), stroke=8)
    return f


def s_rank(t):  # 14.0 - 16.0
    f = bg((255, 60, 30), 0.2)
    CAP_RANK.draw(f, t)
    d = ImageDraw.Draw(f)
    vals = sorted((c["p"]["asthma_ed_u5"] or 0 for c in CD), reverse=True)
    n = len(vals)
    bw = 14
    x0 = (W - n * bw) / 2
    base = 1350
    for i, v in enumerate(vals):
        k = ease((t - i * 0.008) / 0.35)
        h = 520 * v / vals[0] * k
        col = RED if i == 0 else (70, 72, 84)
        d.rectangle((x0 + i * bw + 2, base - h, x0 + (i + 1) * bw - 2, base), fill=col)
    if t > 0.5:
        s = pop_scale(t - 0.5, 0.2) * (1 + 0.7 * (1 - ease((t - 0.5) / 0.12)))
        text(f, "#1", x0 + 160, base - 640, 230, BLACK, RED, s, stroke=8)
        text(f, "Mott Haven", x0 + 160, base - 470, 44, BOLD, WHITE, alpha=clamp((t - 0.6) / 0.2))
    text(f, "all 59 community districts, ranked", W / 2, base + 60, 34, MED, GREY)
    return f


def ring(frame, cx, cy, r, frac, color, width=46):
    d = ImageDraw.Draw(frame)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(40, 44, 52), width=width)
    if frac > 0:
        d.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * frac, fill=color, width=width)


def s_trees(t):  # 16.0 - 18.0
    f = bg((40, 200, 90), 0.12)
    CAP_TREES.draw(f, t)
    k = ease((t - 0.2) / 0.8)
    for cx, val, lab, col in ((290, 7.1, "Mott Haven", RED), (790, 21.1, "NYC average", GREEN)):
        ring(f, cx, 900, 200, val / 100 * k * 2.2, col)
        text(f, f"{val * k:.1f}%", cx, 900, 96, BLACK, WHITE)
        text(f, lab, cx, 1160, 46, BOLD, col)
    text(f, "tree canopy cover", W / 2, 1240, 34, MED, GREY)
    if t > 1.0:
        CAP_TREES2.draw(f, t)
    return f


def s_fifteen(t):  # 18.0 - 20.5
    f = bg((255, 40, 40), 0.15)
    cam = Cam(P(-73.92, 40.72), 2.35)
    lit = {cid: t - (0.4 + i * 0.09) for i, cid in enumerate(TOP15_IDS)}

    def fill(p):
        dt = lit.get(p["id"])
        if dt is None or dt < 0:
            return (30, 32, 40)
        return mix(YELLOW, RED, clamp(dt / 0.25))
    draw_map(f, cam, fill=fill, glow_ids=tuple(TOP15_IDS))
    CAP_15.draw(f, t)
    n = sum(1 for v in lit.values() if v >= 0)
    if t > 0.4:
        chip(f, f"{n}/15", W - 190, 1250, 70, WHITE, (140, 20, 20), 0.5)
    if t > 1.75:
        CAP_15B.draw(f, t)
    return f


def s_zero(t):  # 20.5 - 22.0
    f = bg((255, 40, 40), 0.25)
    CAP_ZERO.draw(f, t)
    d = ImageDraw.Draw(f)
    # 15 dots: 8 funded, 7 not
    for i in range(15):
        r_, c = divmod(i, 5)
        x, y = W / 2 + (c - 2) * 120, 640 + r_ * 120
        s = pop_scale(t - 0.05 * i)
        if s <= 0:
            continue
        rr = 40 * s
        if i < 8:
            d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=GREEN)
            text(f, "$", x, y, 44, BLACK, (10, 40, 20), s)
        else:
            d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=RED, width=6)
    if t > 0.5:
        s = pop_scale(t - 0.5, 0.2) * (1 + 0.8 * (1 - ease((t - 0.5) / 0.1)))
        shake = (math.sin(t * 90) * 14 * clamp(1 - (t - 0.5) / 0.4)) if t < 0.9 else 0
        text(f, "$0", W / 2 + shake, 1190, 300, BLACK, RED, s, stroke=10)
    if t > 0.8:
        text(f, "of the MTA's mitigation money", W / 2, 1400, 50, BOLD, WHITE, alpha=clamp((t - 0.8) / 0.15))
    return f


def s_honest(t):  # 22.0 - 24.0
    f = bg()
    text(f, "The toll's effect on the air?", W / 2, 620, 64, BOLD, WHITE, pop_scale(t))
    if t > 0.5:
        text(f, "Small.", W / 2 - 60, 760, 120, BLACK, GREY, pop_scale(t - 0.5))
        text(f, "🤏", W / 2 + 230, 760, 110, scale=pop_scale(t - 0.6))
        text(f, "most changes weren't statistically significant", W / 2, 860, 30, MED, (110, 112, 120),
             alpha=clamp((t - 0.6) / 0.2))
    if t > 1.0:
        text(f, "The asthma gap?", W / 2, 1060, 64, BOLD, WHITE, pop_scale(t - 1.0))
        sc = pop_scale(t - 1.0, 0.2) * (1 + 0.5 * (1 - ease((t - 1.0) / 0.15)))
        sh = math.sin(t * 80) * 10 * clamp(1 - (t - 1.0) / 0.5)
        text(f, "HUGE.", W / 2 + sh, 1240, 250, BLACK, RED, sc, stroke=10)
    return f


def s_fix(t):  # 24.0 - 27.0
    f = bg(GREEN, 0.3)
    k = ease_io(t / 0.8)
    cam = cam_lerp(Cam(MOTT_C, 12), CAM_MOTT, k)
    draw_map(f, cam, fill=lambda p: (30, 44, 36) if p["id"] == 201 else (26, 28, 34),
             outline=(50, 70, 58))
    draw_line(f, cam, DEEGAN, ease((t - 0.3) / 0.7), (90, 200, 255), 12)
    draw_line(f, cam, BRUCK, ease((t - 0.15) / 0.9), MINT, 18)
    d = ImageDraw.Draw(f)
    for i, s in enumerate(SCHOOLS[:8]):
        x, y = cam.xy(s)
        sc = pop_scale(t - 0.4 - i * 0.07)
        if sc > 0:
            r = 14 * sc
            d.ellipse((x - r, y - r, x + r, y + r), fill=YELLOW, outline=(30, 30, 30), width=3)
    x, y = cam.xy(P(-73.905, 40.8045))
    if t > 0.9:
        chip(f, "Bruckner Blvd", x - 20, y + 70, 32, (10, 30, 18), MINT, t - 0.9)
    CAP_FIX.draw(f, t)
    for i, (em, a, b) in enumerate((("🌳", "262", "new trees"), ("🌲", "2.1 km", "evergreen barrier"),
                                    ("🏫", "4", "schools shielded"))):
        rich_chip(f, [(em, WHITE), (a, MINT), (b, WHITE)], W / 2, 1180 + i * 125, 50, (12, 36, 22),
                  t - 1.5 - i * 0.35)
    return f


def s_cost(t):  # 27.0 - 29.0
    f = bg(GREEN, 0.25)
    tag(f, "STAGE 1 COST", MINT)
    text(f, "$10–19M", W / 2, 560, 180, BLACK, WHITE, pop_scale(t))
    text(f, "to build", W / 2, 690, 54, BOLD, GREY, alpha=clamp(t / 0.2))
    if t > 0.5:
        text(f, "already set aside", W / 2, 860, 54, BOLD, WHITE, pop_scale(t - 0.5))
        text(f, "for Mott Haven:", W / 2, 930, 54, BOLD, WHITE, pop_scale(t - 0.55))
        k = ease((t - 0.6) / 0.25)
        text(f, f"${24 * k:.0f}M", W / 2 - 60, 1070, 170, BLACK, MINT, pop_scale(t - 0.6))
        text(f, "✅", W / 2 + 300, 1070, 120, scale=pop_scale(t - 0.75))
    if t > 1.25:
        s = pop_scale(t - 1.25, 0.2) * (1 + 0.8 * (1 - ease((t - 1.25) / 0.1)))
        stamp = word_img("BUILD IT.", 170, BLACK, (10, 30, 18))
        pad = 40
        box = Image.new("RGBA", (stamp.width + 2 * pad, stamp.height + pad))
        ImageDraw.Draw(box).rounded_rectangle((0, 0, box.width - 1, box.height - 1), radius=30, fill=MINT)
        box.alpha_composite(stamp, (pad, pad // 2))
        box = box.rotate(-6, expand=True, resample=Image.BICUBIC)
        paste_center(f, box, W / 2, 1340, s)
    return f


def s_cta(t):  # 29.0 - 32.0
    f = bg()
    back = ease_io((t - 2.4) / 0.6)  # glide back to the hook framing so the loop is seamless
    cam = cam_lerp(Cam(P(-73.92, 40.73), 2.3), CAM_NYC, back)
    k = 1 - back
    draw_map(f, cam, fill=lambda p: mix((28, 30, 38), asthma_color(p["asthma_ed_adult_aa"] or 0, 0.9), k * 0.8),
             crz=back * 0.6)
    f.alpha_composite(Image.new("RGBA", (W, H), (8, 8, 10, int(130 * k))))
    a = clamp(1 - (t - 2.5) / 0.25)
    CAP_CTA.draw(f, t, out=2.5)
    if a > 0:
        text(f, "Asthma Alley After the Toll", W / 2, 1180, 50, XBOLD, WHITE, alpha=a * clamp((t - 0.8) / 0.2))
        text(f, "CUNY Datathon 2026 · all data public", W / 2, 1250, 34, MED, GREY, alpha=a * clamp((t - 0.9) / 0.2))
        if a > 0.99:
            rich_chip(f, [("🔖", WHITE), ("Save it", (15, 15, 15)), ("·", (15, 15, 15)), ("📤", WHITE),
                          ("Send it", (15, 15, 15))], W / 2, 1390, 40, YELLOW, t - 1.25)
    return f


SCENES = [(0.0, 1.5, s_hook), (1.5, 2.0, s_wait), (2.0, 3.0, s_nothere), (3.0, 5.0, s_toll),
          (5.0, 7.0, s_bridges), (7.0, 10.0, s_bypass), (10.0, 12.0, s_drop), (12.0, 14.0, s_ratio),
          (14.0, 16.0, s_rank), (16.0, 18.0, s_trees), (18.0, 20.5, s_fifteen), (20.5, 22.0, s_zero),
          (22.0, 24.0, s_honest), (24.0, 27.0, s_fix), (27.0, 29.0, s_cost), (29.0, 32.0, s_cta)]

CAP_HOOK = Caption(["They said the", "congestion toll", "*cleaned* NYC's air ✅"], 0.0, 92, gap=0.05)
CAP_NOTHERE = Caption(["Not up here."], 0.05, 110, y=330, accent=RED, gap=0.12)
CAP_WIN = Caption(["Win, right? 👀"], 1.3, 80, y=1260, gap=0.1)
CAP_BRIDGES = Caption(["But 7 of 9 MTA", "bridges & tunnels got", "[MORE] traffic 📈"], 0.0, 80, y=300,
                      hl_bg=RED, hl_fg=WHITE, gap=0.06)
CAP_BYPASS = Caption(["Trucks skip Manhattan", "and go *around…*", "through the Bronx 🚚"], 0.0, 80, y=290,
                     accent=ORANGE, times=[0, .06, .12, 0.4, .46, .52, 1.0, 1.06, 1.12, 1.18])
CAP_RATIO = Caption(["Mott Haven, Bronx 📍", "asthma ER visits:"], 0.0, 84, y=330, gap=0.06)
CAP_RANK = Caption(["Toddler asthma", "ER visits (age 0–4)"], 0.0, 84, y=330, gap=0.07)
CAP_TREES = Caption(["Tree cover 🌳"], 0.0, 96, y=330, gap=0.1)
CAP_TREES2 = Caption(["*⅓* of the city's shade"], 1.0, 70, y=1340, accent=RED, gap=0.08)
CAP_15 = Caption(["The 15 districts with", "the most asthma ER visits…"], 0.0, 72, y=300, gap=0.05)
CAP_15B = Caption(["[ALL_15] got worse NO₂", "than the city in 2025"], 1.75, 72, y=1380, hl_bg=RED, hl_fg=WHITE,
                  gap=0.06)
CAP_ZERO = Caption(["7 of those 15 get…"], 0.0, 80, y=330, gap=0.08)
CAP_FIX = Caption(["So we designed", "a fix 🌱"], 0.0, 96, y=290, accent=MINT, gap=0.08)
CAP_CTA = Caption(["Send this to", "someone who breathes", "*Bronx_air* 📍"], 0.1, 80, y=560, accent=RED, gap=0.08)


# ---------------------------------------------------------------- post FX
IMPACTS = [1.5, 10.0, 13.0, 14.5, 21.0, 23.0, 28.25]
PUNCH = [s for s, _, _ in SCENES] + [10.5, 11.0, 11.5]
FLASH = [10.0, 24.0]
GLITCH = [(1.5, 0.25), (9.95, 0.12), (10.0, 0.3), (23.0, 0.15)]


def post(frame, t, i):
    # zoom punch on cuts
    for s in PUNCH:
        if 0 <= t - s < 0.2:
            z = 1 + 0.07 * (1 - ease((t - s) / 0.2))
            w, h = int(W / z), int(H / z)
            frame = frame.crop(((W - w) // 2, (H - h) // 2, (W - w) // 2 + w, (H - h) // 2 + h)).resize((W, H))
            break
    arr = np.asarray(frame.convert("RGB")).astype(np.float32)
    amp = 0
    for s in IMPACTS:
        if 0 <= t - s < 0.4:
            amp = max(amp, 26 * (1 - (t - s) / 0.4))
    if amp:
        g = np.random.default_rng(i)
        dx, dy = (g.uniform(-amp, amp, 2)).astype(int)
        arr = np.roll(arr, (dy, dx), (0, 1))
    for s, d in GLITCH:
        if 0 <= t - s < d:
            g = np.random.default_rng(i + 99)
            k = int(18 * (1 - (t - s) / d)) + 4
            arr[..., 0] = np.roll(arr[..., 0], k, 1)
            arr[..., 2] = np.roll(arr[..., 2], -k, 1)
            for _ in range(6):
                y0 = g.integers(0, H - 80)
                h = g.integers(10, 80)
                arr[y0:y0 + h] = np.roll(arr[y0:y0 + h], g.integers(-60, 60), 1)
    arr *= VIGNETTE
    gr = GRAIN[i % len(GRAIN)]
    arr += np.kron(gr, np.ones((2, 2), np.float32))[..., None]
    for s in FLASH:
        if 0 <= t - s < 0.15:
            a = 1 - (t - s) / 0.15
            arr = arr * (1 - a) + 255 * a
    return np.clip(arr, 0, 255).astype(np.uint8)


def frame_at(t, i=0):
    for s, e, fn in SCENES:
        if s <= t < e:
            return post(fn(t - s), t, i)
    return post(SCENES[-1][2](SCENES[-1][1] - SCENES[-1][0] - 1e-3), t, i)


def _render_one(i):
    return frame_at(i / FPS, i).tobytes()


def render(out_mp4, wav, start=0.0, end=DUR):
    init_post()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-ss", str(start), "-i", wav, "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-maxrate", "6M", "-bufsize", "12M", "-pix_fmt", "yuv420p", "-profile:v", "high",
           "-c:a", "aac", "-b:a", "192k", "-af", "loudnorm=I=-14:TP=-1.0:LRA=11", "-ar", "48000",
           "-movflags", "+faststart", "-shortest", out_mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n0, n1 = int(start * FPS), int(end * FPS)
    with Pool(os.cpu_count(), initializer=init_post) as pool:
        for i, buf in enumerate(pool.imap(_render_one, range(n0, n1), chunksize=4), n0):
            p.stdin.write(buf)
            if i % 60 == 0:
                print(f"frame {i}/{n1}", flush=True)
    p.stdin.close()
    p.wait()


def cover(path):
    init_post()
    f = s_drop(1.2)
    text(f, "the toll didn't fix this.", W / 2, H * 0.40 - 230, 64, XBOLD, WHITE, stroke=4)
    Image.fromarray(post(f, 99, 0)).save(path)


if __name__ == "__main__":
    out = os.path.join(HERE, "asthma-alley-reel.mp4")
    wav = os.path.join(HERE, "soundtrack.wav")
    if "--frames" in sys.argv:  # quick stills for checking layout
        init_post()
        dst = sys.argv[sys.argv.index("--frames") + 1]
        os.makedirs(dst, exist_ok=True)
        for t in [float(x) for x in sys.argv[sys.argv.index("--frames") + 2].split(",")]:
            Image.fromarray(frame_at(t, int(t * FPS))).save(os.path.join(dst, f"t{t:05.2f}.png"))
        sys.exit()
    audio.build(wav)
    render(out, wav)
    cover(os.path.join(HERE, "cover.png"))
    print("wrote", out)
