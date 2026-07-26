"""make_hook.py -- the series "hook" GIFs for machine #1 (rotary filler).

VIZ-ONLY. Nothing here touches the physics core, the reference, the params, the
analysis or the sensors: it only *reuses already-rendered assets* and stamps the
shared visual language on them.

  * MOTION comes from the already-rendered figs/anim_filler.gif (the 8-station
    intermittent rotation). We load its frames with PIL and subsample one seamless
    loop -- we do NOT re-run the heavy 3D pipeline.
  * SIGNAL comes from viz/panels.bottle_cross_section(...), used read-only. It is
    exact geometry: the liquid surface, the wall dz = R*tan(tilt), and a readout.
    tilt = 0 is the flat/normal surface; tilt = +/-0.068 rad (3.9 deg, the real
    slosh tilt) is the sloshing surface. We also re-plot that same tilt as a small
    signal trace so "the signal spikes" is literal -- it is the very tilt we feed
    the panel, nothing invented.

The named failure is LIQUID SLOSH (category: conservation). A = abnormal fill
(WARN, warm): the surface sloshes, the tilt signal spikes. B = normal fill
(CORRECT, cool): flat surface, quiet signal. The machine motion is the SAME on
both sides (it is the same machine), so the A/B is carried by the surface, the
signal trace, and the WARN/CORRECT tags.

Honesty: the surface tilt is drawn TO SCALE (3.9 deg peak, no exaggeration -- see
panels.bottle_cross_section); only the playback is slowed so the eye can read it.

Run:  cd rotary_filler && .venv/bin/python viz/make_hook.py
Writes: figs/hook_zenn.gif (16:9), figs/hook_li_1x1.gif (1:1), figs/hook_li_4x5.gif (4:5)
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

# viz/ is the script dir, so these two neighbours import directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panels          # noqa: E402  (read-only: signal panels, exact geometry)
import series_style as ss  # noqa: E402  (the shared visual language; imported, never edited)

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(LAB_ROOT, "figs")
ANIM = os.path.join(FIGS, "anim_filler.gif")

# --------------------------------------------------------------------------- copy
# Burned caption strings. ASCII only (middle dot as an escape). No product / maker
# names in the burned text; the byline lives in LICENSE and the README, not here.
DOT = "\u00b7"     # middle dot, kept as an escape so this .py stays ASCII
CAP = {
    "zenn_title": "Liquid slosh " + DOT + " abnormal fill vs normal fill",
    "zenn_sub": ("one machine, two outcomes: the abnormal bottle sloshes "
                 "and the tilt signal spikes; the normal bottle stays flat"),
    "li_title": "Liquid slosh",
    "machine": "the machine (shared)",
    "trace": "surface tilt at wall  [deg]",
    "trace_short": "tilt at wall [deg]",
    "hdr_abn": "ABNORMAL " + DOT + " slosh",
    "hdr_nrm": "NORMAL " + DOT + " flat",
    # THE shared failure/fix statement (rendered through ss.caption_band):
    "fail": "Abnormal fill: surface sloshes",
    "fix": "Normal fill: surface stays flat",
    "scale": "tilt drawn to scale (3.9 deg peak) " + DOT + " playback slowed to read it",
    "scale_small": "tilt drawn to scale (3.9 deg)",
}

# --------------------------------------------------------------------------- physics-free constants
LEVEL_MM = 120.5          # ~400 mL fill (498 mL body); matches fill.target_volume_mL
AMP_RAD = 0.068           # the real slosh tilt (3.9 deg); NOT exaggerated
TILT_MAX_DEG = 5.0        # trace y-range, so the 3.9 deg peak reads without clipping


# --------------------------------------------------------------------------- small PIL helpers
def _pil(a):
    if isinstance(a, Image.Image):
        return a.convert("RGB")
    return Image.fromarray(np.asarray(a)[..., :3].astype("uint8"), "RGB")


def paste_fit(canvas, img, box):
    """Paste `img` into `box`=(x0,y0,x1,y1), centered, aspect kept (never distorted)."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    im = _pil(img)
    iw, ih = im.size
    s = min(bw / iw, bh / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    im = im.resize((nw, nh), Image.LANCZOS)
    canvas.paste(im, (x0 + (bw - nw) // 2, y0 + (bh - nh) // 2))


def pill(d, box, text, fill, text_fill=(255, 255, 255)):
    """A filled rounded header pill with centered white text."""
    x0, y0, x1, y1 = box
    d.rounded_rectangle([x0, y0, x1, y1], radius=(y1 - y0) // 2, fill=fill)
    f = ss.font(int((y1 - y0) * 0.46), bold=True)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=f, fill=text_fill, anchor="mm")


def make_trace(W, H, i, N, traces, label):
    """A compact SCROLLING strip chart of the surface-tilt trace(s). `traces` is a list
    of (values_deg, color): the abnormal channel is a WARN oscillation, the normal one a
    CORRECT flat line -- the SAME tilt values fed to the bottle panel, so it is honest.

    The current sample sits at a FIXED "now" line on the right edge and the (periodic)
    history scrolls left. Because the tilt loop is periodic, display[N-1] scrolls one
    sample into display[0] with NO cursor jump at the seam -- an oscilloscope-style sweep
    would instead snap the cursor from far-right back to far-left, which we avoid."""
    im = Image.new("RGB", (W, H), ss.BG)
    d = ImageDraw.Draw(im)
    pl, pr, pt, pb = 10, 10, 24, 12
    x0, y0, x1, y1 = pl, pt, W - pr, H - pb
    cy = (y0 + y1) / 2.0
    half = (y1 - y0) / 2.0
    d.rectangle([x0, y0, x1, y1], outline=ss.HAIR, width=1)
    d.line([x0, cy, x1, cy], fill=ss.HAIR, width=1)

    def colx(c):
        return x0 + (c / (N - 1)) * (x1 - x0)

    def py(v):
        return cy - (v / TILT_MAX_DEG) * half

    # column c shows sample (i - (N-1) + c) mod N; rightmost column c=N-1 is the current
    # sample i, oldest is on the left. periodic vals -> seamless scroll across the seam.
    for vals, col in traces:
        pts = [(colx(c), py(vals[(i - (N - 1) + c) % N])) for c in range(N)]
        d.line(pts, fill=col, width=3, joint="curve")
    d.line([x1, y0, x1, y1], fill=ss.HOT, width=2)          # fixed "now" line
    for vals, col in traces:
        d.ellipse([x1 - 4, py(vals[i]) - 4, x1 + 4, py(vals[i]) + 4], fill=col)
    d.text((x0 + 2, 3), label, font=ss.font(13, bold=True), fill=ss.INK)
    return im


# --------------------------------------------------------------------------- asset loaders
def load_motion():
    """All frames of the already-rendered rotary animation as (H,W,3) uint8 arrays."""
    im = Image.open(ANIM)
    n = getattr(im, "n_frames", 1)
    out = []
    for k in range(n):
        im.seek(k)
        out.append(np.asarray(im.convert("RGB")))
    return out


def subsample(frames, N):
    """N evenly spaced frames over one seamless period (start at 0, wrap-safe)."""
    n = len(frames)
    return [frames[int(round(i * n / N)) % n] for i in range(N)]


def bottle_png(tmp, params, tilt_rad, tag):
    """Render one exact-geometry bottle cross-section (read-only) and load it."""
    p = os.path.join(tmp, "bottle_%s.png" % tag)
    panels.bottle_cross_section(params, LEVEL_MM, tilt_rad, p,
                                size_px=(560, 760), dpi=100)
    return Image.open(p).convert("RGB")


def tilt_schedule(N, cycles):
    """Seamless surface-tilt loop. cos start so frame 0 is a PEAK (good thumbnail)."""
    tilt_rad = np.array([AMP_RAD * math.cos(2 * math.pi * cycles * i / N)
                         for i in range(N)])
    return tilt_rad, np.degrees(tilt_rad), np.zeros(N)


# --------------------------------------------------------------------------- Zenn 16:9
def build_zenn(motion, bottles_a, bottle_n, ya_deg, yn_deg):
    """One instrument: machine (left) + tilt trace + the abnormal|normal bottles."""
    N = len(motion)
    W, H = 1280, 720
    frames = []
    for i in range(N):
        cv = Image.new("RGB", (W, H), ss.BG)
        d = ImageDraw.Draw(cv)
        d.text((22, 12), CAP["zenn_title"], font=ss.font(28, bold=True), fill=ss.INK)
        d.text((22, 49), CAP["zenn_sub"], font=ss.font(15), fill=ss.MUT)
        # machine motion (shared, neutral)
        paste_fit(cv, motion[i], (16, 84, 470, 430))
        d.text((18, 434), CAP["machine"], font=ss.font(14, bold=True), fill=ss.MUT)
        # signal trace: both channels, the abnormal one spiking
        cv.paste(make_trace(454, 150, i, N,
                            [(yn_deg, ss.CORRECT), (ya_deg, ss.WARN)], CAP["trace"]),
                 (16, 460))
        d.text((18, 618), CAP["scale"], font=ss.font(12), fill=ss.MUT)
        # the two bottles
        pill(d, (500, 72, 884, 100), CAP["hdr_abn"], ss.WARN)
        pill(d, (892, 72, 1276, 100), CAP["hdr_nrm"], ss.CORRECT)
        paste_fit(cv, bottles_a[i], (500, 104, 884, 664))
        paste_fit(cv, bottle_n, (892, 104, 1276, 664))
        frames.append(ss.badge(np.asarray(cv), 1, corner="tr"))   # uniform: top-right
    return frames


# --------------------------------------------------------------------------- LinkedIn square
def build_li_state(motion, bottle_list, kind, N):
    """One square block (900x900) for a single state. The bottle cross-section -- the
    tilted surface + the dz readout -- is the LARGE hero so the failure reads at a glance
    at frame 0; the machine render and the tilt trace are small ancillaries in a narrow
    left column. The failure/fix statement is the shared ss.caption_band (the sole state
    caption -- no ad-hoc pill or plain-text line). bottle_list is length N (abnormal
    varies; normal repeats). Returns pre-badge (H,W,3) arrays. Tilt stays 3.9 deg to
    scale -- the panel is only made bigger, never exaggerated."""
    W = 900
    col = ss.WARN if kind == "warn" else ss.CORRECT
    state = CAP["fail"] if kind == "warn" else CAP["fix"]
    # trace shows the CURRENT state's signal: WARN oscillation (abnormal) or flat (normal)
    if kind == "warn":
        vals = np.degrees(np.array([AMP_RAD * math.cos(2 * math.pi * 2 * i / N)
                                    for i in range(N)]))
    else:
        vals = np.zeros(N)
    traces = [(vals, col)]
    frames = []
    for i in range(N):
        cv = Image.new("RGB", (W, W), ss.BG)
        d = ImageDraw.Draw(cv)
        d.text((30, 18), CAP["li_title"], font=ss.font(30, bold=True), fill=ss.INK)
        # small ancillaries, narrow left column
        paste_fit(cv, motion[i], (28, 84, 250, 250))
        d.text((30, 254), CAP["machine"], font=ss.font(14, bold=True), fill=ss.MUT)
        cv.paste(make_trace(224, 118, i, N, traces, CAP["trace_short"]), (28, 300))
        d.text((30, 424), CAP["scale_small"], font=ss.font(13), fill=ss.MUT)
        # the hero: a large bottle cross-section (surface tilt + dz), clear of the band
        paste_fit(cv, bottle_list[i], (272, 58, 884, 760))
        # the ONE shared caption treatment (fills the bottom strip)
        frames.append(ss.caption_band(np.asarray(cv), state, kind=kind))
    return frames


# --------------------------------------------------------------------------- main
def main():
    params = panels.load_params()
    motion_all = load_motion()

    N_Z, FPS_Z, K_Z = 60, 10, 3          # 6.0 s Zenn loop, 3 slosh cycles
    N_L, FPS_L = 30, 12                  # LinkedIn per-state block

    with tempfile.TemporaryDirectory(prefix="rf_hook_") as tmp:
        # -- Zenn assets
        motion_z = subsample(motion_all, N_Z)
        tilt_z, ya_z, yn_z = tilt_schedule(N_Z, K_Z)
        bottles_a_z = [bottle_png(tmp, params, float(t), "z%02d" % i)
                       for i, t in enumerate(tilt_z)]
        bottle_n = bottle_png(tmp, params, 0.0, "flat")

        zenn = build_zenn(motion_z, bottles_a_z, bottle_n, ya_z, yn_z)
        info_z = ss.write_gif(zenn, os.path.join(FIGS, "hook_zenn.gif"), fps=FPS_Z)

        # -- LinkedIn assets (fail-first: abnormal block, then normal block)
        motion_l = subsample(motion_all, N_L)
        tilt_l, _, _ = tilt_schedule(N_L, 2)
        bottles_a_l = [bottle_png(tmp, params, float(t), "l%02d" % i)
                       for i, t in enumerate(tilt_l)]

        abn = build_li_state(motion_l, bottles_a_l, "warn", N_L)
        nrm = build_li_state(motion_l, [bottle_n] * N_L, "correct", N_L)

        hold = 3
        square = abn + [abn[-1]] * hold + nrm + [nrm[-1]] * hold  # FAIL FIRST
        # uniform badge: top-right on every variant
        li_1x1 = [ss.badge(f, 1, corner="tr") for f in square]
        info_1 = ss.write_gif(li_1x1, os.path.join(FIGS, "hook_li_1x1.gif"), fps=FPS_L)

        # 4:5 pads the square top/bottom; badge stays top-right (in the top letterbox)
        li_4x5 = [ss.badge(ss.reframe(f, "4:5"), 1, corner="tr") for f in square]
        info_4 = ss.write_gif(li_4x5, os.path.join(FIGS, "hook_li_4x5.gif"), fps=FPS_L)

    print("\nwrote:")
    for tag, info in (("zenn 16:9", info_z), ("li 1:1", info_1), ("li 4:5", info_4)):
        print("  %-10s %dx%d  %d frames  %.1fs  %.2f MB  over=%s"
              % (tag, info["w"], info["h"], info["frames"], info["seconds"],
                 info["size_mb"], info["over_budget"]))
    return info_z, info_1, info_4


if __name__ == "__main__":
    main()
