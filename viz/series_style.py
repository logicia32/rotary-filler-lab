"""series_style.py -- the shared visual language for the five-machine twin series.

An IDENTICAL copy lives in each machine's viz directory (like paths.py). It is viz-only:
no physics, no params values, no local imports -- only stdlib + numpy + PIL, so it drops
cleanly into a flat src/ or into rotary's viz/. Every machine renders its own frames; this
module stamps the SHARED IDENTITY on them so the five read as one series at a glance:

  * three semantic colours, and only three -- NEUTRAL (normal), WARN (the named failure is
    happening / the failing part), CORRECT (the fix is active: shaper / DLS / comp ...);
  * one corner badge, same place every machine: "n/5 . <machine> . <category>";
  * one caption grammar (title band, small tags) in one font;
  * one aspect pipeline (16:9 for Zenn, 1:1 and 4:5 for LinkedIn) on a neutral letterbox;
  * one seamless-loop, palette-stable, <8 MB GIF writer.

The frame contract is deliberately dumb: a frame is an (H, W, 3) uint8 numpy array (what
every machine's render already produces via plotter.screenshot / np.asarray(PIL)). Machines
hand frames in; this module composes, reframes, badges and writes. In-scene recolouring of
the failing part uses the colour constants below; everything else is a post-hoc overlay, so
a machine whose scene builder cannot take a colour can still carry the semantics by a WARN
tag and the A/B layout alone.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- colours
# int RGB (PIL overlay) is the source of truth; *_F is the 0..1 float form for pyvista mesh.
NEUTRAL = (140, 143, 150)      # normal operation (a neutral mesh grey)
WARN    = (196, 64, 44)        # warm -- the NAMED FAILURE is happening / the failing part
CORRECT = (40, 110, 168)       # cool -- the FIX is active (shaper / DLS / backlash comp ...)
BG      = (247, 247, 249)      # neutral background: pyvista bg AND letterbox pad
INK     = (28, 32, 40)         # primary text
MUT     = (108, 114, 124)      # secondary text
HOT     = (150, 24, 16)        # a clipped/pinned cap (a bar past full scale, a hard edge)
PANEL   = (247, 247, 249)      # header/footer band fill (== BG, on purpose)
HAIR    = (208, 210, 216)      # hairlines, dividers


def as_float(rgb):
    """(r,g,b) 0..255 -> (r,g,b) 0..1 for pyvista add_mesh(color=...)."""
    return tuple(c / 255.0 for c in rgb)


NEUTRAL_F, WARN_F, CORRECT_F, BG_F = (as_float(c) for c in (NEUTRAL, WARN, CORRECT, BG))

# --------------------------------------------------------------------------- series id
# Generic, OPSEC-safe machine names (no manufacturer / product) and a one-word physics
# category. The badge that makes the series recognisable is built ONLY from this table.
SEP = "\u00b7"                         # middle dot, kept as an escape so the .py stays ASCII
SERIES = {
    1: ("rotary filler",     "conservation"),
    2: ("egg packer",        "constraint"),
    3: ("xy stage",          "feedback"),
    4: ("chip mounter",      "dynamics"),
    5: ("articulated robot", "kinematics"),
}

# --------------------------------------------------------------------------- fonts
_FONT_CACHE = {}
_REG = ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
_BLD = ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size, bold=False):
    """A cached DejaVu face at `size`. DejaVu carries the middle dot and arrows we use."""
    key = (int(size), bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    for nm in (_BLD if bold else _REG):
        try:
            f = ImageFont.truetype(nm, int(size))
            _FONT_CACHE[key] = f
            return f
        except Exception:
            continue
    f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def _img(frame):
    """Accept an (H,W,3) array or a PIL image; return a PIL RGB image (a COPY for arrays)."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    return Image.fromarray(np.asarray(frame)[..., :3].astype("uint8"), "RGB")


def _arr(im):
    return np.asarray(im.convert("RGB"))


def _text_wh(d, text, fnt):
    l, t, r, b = d.textbbox((0, 0), text, font=fnt)
    return r - l, b - t


# --------------------------------------------------------------------------- overlays
def badge(frame, n, corner="br", margin_frac=0.018, height_frac=0.052,
          min_h=22, max_h=40):
    """Stamp the series badge -- "n/5 . name . category" -- in a corner. THIS is the mark of
    the series: same grammar, same corner, every machine. Drawn as an opaque light pill with
    a small accent chip so it reads on a grey 3D scene or a white panel alike. Returns a new
    (H,W,3) uint8 array. `corner` in {br, bl, tr, tl}."""
    im = _img(frame)
    W, H = im.size
    d = ImageDraw.Draw(im)
    h = int(round(min(max(H * height_frac, min_h), max_h)))
    fs = int(round(h * 0.52))
    fb = font(fs, bold=True)
    fr = font(fs, bold=False)
    name, cat = SERIES[int(n)]
    idx = "%d/5" % int(n)
    padx = int(round(h * 0.42))
    gap = int(round(h * 0.34))
    wi, _ = _text_wh(d, idx, fb)
    wn, _ = _text_wh(d, "%s %s %s" % (SEP, name, SEP), fr)
    wc, _ = _text_wh(d, cat, fb)
    pill_w = padx + wi + gap + wn + gap + wc + padx
    m = int(round(min(W, H) * margin_frac))
    if "r" in corner:
        x0 = W - m - pill_w
    else:
        x0 = m
    if "b" in corner:
        y0 = H - m - h
    else:
        y0 = m
    x1, y1 = x0 + pill_w, y0 + h
    rad = int(round(h * 0.28))
    d.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=(255, 255, 255),
                        outline=HAIR, width=1)
    ty = y0 + (h - fs) / 2 - 1
    x = x0 + padx
    d.text((x, ty), idx, font=fb, fill=INK)
    x += wi + gap
    d.text((x, ty), "%s %s %s" % (SEP, name, SEP), font=fr, fill=MUT)
    x += wn + gap
    d.text((x, ty), cat, font=fb, fill=(WARN if int(n) else INK))
    return _arr(im)


def banner(frame, title, sub=None, side="top", accent=None, pad_frac=0.015):
    """A light title band across the top (or bottom) with a bold `title` and a muted `sub`.
    `accent` tints the title (pass WARN/CORRECT to colour a state). The band GROWS the image
    (it is not painted over the scene), so nothing in the render is hidden. Returns a new
    (H,W,3) uint8 array."""
    im = _img(frame)
    W, H = im.size
    pad = int(round(W * pad_frac))
    ts = max(14, int(round(W * 0.022)))
    ss = max(11, int(round(W * 0.015)))
    band = ts + pad // 2 + (ss + 4 if sub else 0) + pad
    out = Image.new("RGB", (W, H + band), PANEL)
    out.paste(im, (0, band if side == "top" else 0))
    d = ImageDraw.Draw(out)
    y0 = 0 if side == "top" else H
    d.rectangle([0, y0, W, y0 + band], fill=PANEL)
    ty = y0 + pad // 2
    d.text((pad, ty), title, font=font(ts, bold=True), fill=accent or INK)
    if sub:
        d.text((pad, ty + ts + 4), sub, font=font(ss, bold=False), fill=MUT)
    return _arr(out)


def tag(frame, xy, text, kind="warn", anchor="lt", pad_frac=0.010):
    """A small pill label at image coords `xy`, coloured by `kind` in {warn, correct,
    neutral}. Use it to name the failing part ("axes 4,6 align") or the fix ("DLS"). `anchor`
    is a PIL-style two-char (l/m/r + t/m/b) for which corner of the pill sits at `xy`."""
    im = _img(frame)
    W, H = im.size
    col = {"warn": WARN, "correct": CORRECT, "neutral": MUT}[kind]
    fs = max(11, int(round(W * 0.016)))
    d = ImageDraw.Draw(im)
    tw, th = _text_wh(d, text, font(fs, bold=True))
    px = int(round(W * pad_frac))
    pw, ph = tw + 2 * px, th + int(px * 1.1)
    x, y = xy
    if "r" in anchor:
        x -= pw
    elif "m" in anchor[:1]:
        x -= pw // 2
    if "b" in anchor:
        y -= ph
    elif anchor[-1] == "m":
        y -= ph // 2
    d.rounded_rectangle([x, y, x + pw, y + ph], radius=int(ph * 0.32), fill=col)
    d.text((x + px, y + ph / 2), text, font=font(fs, bold=True), fill=(255, 255, 255),
           anchor="lm")
    return _arr(im)


def caption_band(frame, text, kind="warn", side="bottom", margin_frac=0.035,
                 height_frac=0.135, min_h=32, max_h=104):
    """THE one caption treatment the whole series shares: a single filled, rounded band --
    WARN warm-red when the named failure is on screen, CORRECT cool-blue when the fix is --
    with white bold text, inset from the edges. Every machine states its failure/fix through
    THIS (not ad-hoc pills or plain text), so the five read as one series and it stays legible
    at thumbnail size. The text auto-fits the band width; the band is painted over a bottom
    (or top) strip, so keep the subject centred. Returns a new (H,W,3) uint8 array."""
    im = _img(frame)
    W, H = im.size
    d = ImageDraw.Draw(im)
    col = {"warn": WARN, "correct": CORRECT, "neutral": MUT}[kind]
    m = int(round(W * margin_frac))
    h = int(round(min(max(H * height_frac, min_h), max_h)))
    y1 = (H - m) if side == "bottom" else (m + h)
    y0 = y1 - h
    x0, x1 = m, W - m
    rad = int(round(h * 0.24))
    d.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=col)
    avail = (x1 - x0) - 2 * rad - int(W * 0.02)
    fs = max(11, int(h * 0.44))
    while fs > 11:
        tw, _ = _text_wh(d, text, font(fs, bold=True))
        if tw <= avail:
            break
        fs -= 1
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=font(fs, bold=True),
           fill=(255, 255, 255), anchor="mm")
    return _arr(im)


# --------------------------------------------------------------------------- A/B compose
def _match_h(a, b):
    """Resize two RGB images to a common height (the smaller), keeping aspect."""
    ha, hb = a.height, b.height
    h = min(ha, hb)
    if a.height != h:
        a = a.resize((round(a.width * h / a.height), h), Image.LANCZOS)
    if b.height != h:
        b = b.resize((round(b.width * h / b.height), h), Image.LANCZOS)
    return a, b


def side_by_side(frames_a, frames_b, gap_frac=0.010, label_a=None, label_b=None,
                 color_a=WARN, color_b=CORRECT, head_frac=0.060):
    """Compose two equal-length frame lists into one wide A|B clip with a divider and small
    coloured headers. A is LEFT (the failure), B is RIGHT (the fix). Returns a list of
    (H,W,3) uint8 arrays. This is the Zenn-inline layout (wide)."""
    n = min(len(frames_a), len(frames_b))
    out = []
    for i in range(n):
        la, lb = _match_h(_img(frames_a[i]), _img(frames_b[i]))
        gap = int(round((la.width + lb.width) * 0.5 * gap_frac)) + 2
        head = int(round(la.height * head_frac)) if (label_a or label_b) else 0
        W = la.width + gap + lb.width
        cv = Image.new("RGB", (W, la.height + head), (255, 255, 255))
        cv.paste(la, (0, head))
        cv.paste(lb, (la.width + gap, head))
        d = ImageDraw.Draw(cv)
        if head:
            d.rectangle([0, 0, W, head], fill=PANEL)
            fs = max(12, int(round(head * 0.52)))
            if label_a:
                d.text((6, head / 2), label_a, font=font(fs, bold=True), fill=color_a,
                       anchor="lm")
            if label_b:
                d.text((la.width + gap + 6, head / 2), label_b, font=font(fs, bold=True),
                       fill=color_b, anchor="lm")
        d.line([(la.width + gap // 2, head), (la.width + gap // 2, la.height + head)],
               fill=HAIR, width=2)
        out.append(_arr(cv))
    return out


def sequential(frames_fail, frames_fix, hold=2, label_fail=None, label_fix=None):
    """Concatenate FAIL then FIX into one single-pane clip (fail first, so a LinkedIn
    thumbnail = the failure). A small state tag rides in the top-left -- WARN over the fail
    block, CORRECT over the fix block -- and each block's last frame is held `hold` extra
    frames so the eye lands before the cut. Returns a list of (H,W,3) uint8 arrays; the two
    inputs must already share a size (reframe them first)."""
    def block(frames, kind, label):
        seq = list(frames) + [frames[-1]] * max(0, hold)
        stamped = []
        for f in seq:
            im = _img(f)
            W, _ = im.size
            xy = (int(W * 0.02), int(W * 0.02))
            stamped.append(tag(im, xy, label or kind.upper(), kind=kind, anchor="lt"))
        return stamped
    return block(frames_fail, "warn", label_fail) + block(frames_fix, "correct", label_fix)


def boomerang(frames):
    """frames + reversed-interior -> a clip whose last frame flows back into the first, so a
    non-periodic motion still loops seamlessly (forward, then back)."""
    frames = list(frames)
    if len(frames) < 3:
        return frames
    return frames + frames[-2:0:-1]


# --------------------------------------------------------------------------- aspect
_ASPECT = {"16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "4:5": 4 / 5, "9:16": 9 / 16,
           "3:2": 3 / 2}


def reframe(frame, aspect, bg=BG, mode="pad"):
    """Letterbox (`mode="pad"`, the default -- never lose content) or centre-crop
    (`mode="crop"`) a frame to an exact aspect ratio on a neutral background. `aspect` is a
    key of _ASPECT or a float W/H. Returns a new (H,W,3) uint8 array."""
    im = _img(frame)
    W, H = im.size
    ar = _ASPECT[aspect] if isinstance(aspect, str) else float(aspect)
    if mode == "crop":
        cur = W / H
        if cur > ar:                                   # too wide -> trim sides
            nw = int(round(H * ar))
            x0 = (W - nw) // 2
            im = im.crop((x0, 0, x0 + nw, H))
        else:                                          # too tall -> trim top/bottom
            nh = int(round(W / ar))
            y0 = (H - nh) // 2
            im = im.crop((0, y0, W, y0 + nh))
        return _arr(im)
    # pad
    if W / H > ar:                                     # too wide -> pad top/bottom
        nh = int(round(W / ar))
        cv = Image.new("RGB", (W, nh), bg)
        cv.paste(im, (0, (nh - H) // 2))
    else:                                              # too tall -> pad sides
        nw = int(round(H * ar))
        cv = Image.new("RGB", (nw, H), bg)
        cv.paste(im, ((nw - W) // 2, 0))
    return _arr(cv)


def reframe_all(frames, aspect, bg=BG, mode="pad"):
    return [reframe(f, aspect, bg=bg, mode=mode) for f in frames]


def fit_width(frames, width):
    """Downscale (never upscale) every frame to `width` px, keeping aspect. Handy to hit a
    known deliverable width before writing the GIF."""
    out = []
    for f in frames:
        im = _img(f)
        if im.width <= width:
            out.append(_arr(im))
            continue
        h = round(im.height * width / im.width)
        out.append(_arr(im.resize((int(width), h), Image.LANCZOS)))
    return out


# --------------------------------------------------------------------------- GIF export
def _common_palette(frames, colors):
    """One palette for the whole clip, chosen from a montage of sampled frames so small,
    saturated accents (a red bar, a blue tip -- few pixels, high chroma) are not rounded to
    grey the way a frame-0-only median cut would. dither=NONE keeps inter-frame diffs sparse
    (small GIF)."""
    n = len(frames)
    k = min(12, n)
    picks = [frames[round(i * (n - 1) / max(k - 1, 1))] for i in range(k)]
    ims = [_img(p) for p in picks]
    w = ims[0].width
    ims = [im if im.width == w else im.resize((w, round(im.height * w / im.width)),
                                              Image.LANCZOS) for im in ims]
    sheet = Image.new("RGB", (w, sum(im.height for im in ims)))
    y = 0
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.height
    return sheet.quantize(colors=max(2, colors), method=Image.MEDIANCUT,
                          dither=Image.NONE)


def _write_once(frames, path, fps, colors):
    pal = _common_palette(frames, colors)
    q = [_img(f).quantize(palette=pal, dither=Image.NONE) for f in frames]
    dur = max(20, int(round(1000.0 / float(fps))))
    q[0].save(path, save_all=True, append_images=q[1:], duration=dur, loop=0,
              optimize=True, disposal=2)
    return os.path.getsize(path)


def write_gif(frames, path, fps=16, max_mb=8.0, colors=160, loop_boomerang=False,
              verbose=True):
    """Write a seamless, palette-stable, loop=0 GIF and keep it under `max_mb`. If the first
    write is over budget, step the palette down (160->128->96->72->56) and then, if still
    over, scale the frames by 0.9 repeatedly -- re-quantising each time -- until it fits. The
    loop is seamless by construction: pass an already-periodic clip, or loop_boomerang=True
    to append the reversed interior. Returns a dict describing what was written."""
    if loop_boomerang:
        frames = boomerang(frames)
    if not frames:
        raise ValueError("write_gif got no frames")
    fps = max(float(fps), 1.0)                        # a GIF frame lasts <= 1000 ms anyway
    frames = [_img(f) for f in frames]
    if not os.path.dirname(path):
        raise ValueError("write_gif needs a path with a directory (use paths.fig(name))")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    budget = max_mb * 1e6
    scale = 1.0
    tried = []
    for col in (colors, 128, 96, 72, 56):
        if col > colors:
            continue
        size = _write_once(frames, path, fps, col)
        tried.append((round(scale, 3), col, size))
        if size <= budget:
            break
    else:
        col = 56
    # still over: shrink and retry at the lean palette
    while os.path.getsize(path) > budget and frames[0].width > 360:
        scale *= 0.9
        w = int(frames[0].width * 0.9)
        frames = [im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
                  for im in frames]
        size = _write_once(frames, path, fps, col)
        tried.append((round(scale, 3), col, size))

    W, H = frames[0].size
    mb = os.path.getsize(path) / 1e6
    info = dict(path=path, size_mb=round(mb, 2), w=W, h=H, frames=len(frames),
                fps=fps, colors=col, scale=round(scale, 3),
                over_budget=mb > max_mb, seconds=round(len(frames) / float(fps), 1))
    if verbose:
        flag = "  !! OVER" if info["over_budget"] else ""
        print("  %-42s %dx%d  %d frames  %.1fs  %d col  %.2f MB%s"
              % (os.path.basename(path), W, H, len(frames), info["seconds"], col, mb, flag))
    return info


# --------------------------------------------------------------------------- self-check
def _selftest(outdir):
    """Render synthetic frames through the whole pipeline: a red disc (fail) and a blue disc
    (fix), composed A|B and sequentially, badged and reframed to every deliverable aspect,
    written as seamless GIFs. Proves the module in isolation with no machine dependency."""
    import math
    os.makedirs(outdir, exist_ok=True)
    N = 24
    H0, W0 = 300, 380

    def disc(i, col):
        im = Image.new("RGB", (W0, H0), BG)
        d = ImageDraw.Draw(im)
        cx = W0 * 0.5 + 70 * math.sin(2 * math.pi * i / N)
        cy = H0 * 0.5
        d.ellipse([cx - 34, cy - 34, cx + 34, cy + 34], fill=col, outline=INK)
        return _arr(im)

    fail = [disc(i, WARN) for i in range(N)]
    fix = [disc(i, CORRECT) for i in range(N)]

    # Zenn: A|B side by side -> 16:9 -> badge #5
    zz = side_by_side(fail, fix, label_a="naive", label_b="fixed")
    zz = [banner(f, "Named failure vs the fix", "synthetic self-test") for f in zz]
    zz = [badge(reframe(f, "16:9"), 5) for f in zz]
    a = write_gif(zz, os.path.join(outdir, "selftest_zenn.gif"), fps=16)

    # LinkedIn 1:1: sequential fail-first -> caption_band (the one shared treatment) -> badge
    lf = [caption_band(reframe(f, "1:1"), "Failure: it runs away", "warn") for f in fail]
    lx = [caption_band(reframe(f, "1:1"), "Fixed: it settles", "correct") for f in fix]
    li = sequential(lf, lx, hold=3)
    li = [badge(f, 5) for f in li]
    b = write_gif(li, os.path.join(outdir, "selftest_li_1x1.gif"), fps=14)

    # LinkedIn 4:5 via boomerang of the fail clip
    v = [badge(reframe(f, "4:5"), 5) for f in fail]
    c = write_gif(v, os.path.join(outdir, "selftest_li_4x5.gif"), fps=16,
                  loop_boomerang=True)
    return a, b, c


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "series_style_selftest"
    print("series_style self-test ->", out)
    for info in _selftest(out):
        assert not info["over_budget"], info
    print("OK: all deliverable aspects wrote seamless GIFs under budget")
