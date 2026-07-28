"""anim_article.py -- 記事用の「動く GIF」3 本。

VIZ 専用。物理コア・参照実装・params・センサ・解析には一切触らない。ここは
`py/ref.py` の運動学とスロッシングを **読むだけ** で呼び、series_style で
シリーズ共通の見た目（角のバッジ・3 色）を載せて GIF を書き出す。
figs/ には他の担当の図もあるので `art_` で始まる名前しか書かない。

  * art_cam_motion  -- 入力軸は一定速で回り続け、テーブルは動いて止まる（機構）
  * art_resonance   -- タクト 2.4 s（拍子が合う）と 3.0 s（ずらす）で液の揺れを並べる
  * art_alias       -- 速い振動を遅いセンサで読むと、本当は無い遅い山が湧く

Run: cd rotary_filler && .venv/bin/python -m viz.anim_article [--only cam|res|alias]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Wedge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                 # series_style, panels
sys.path.insert(0, os.path.join(ROOT, "py"))  # ref, params
import series_style as ss    # noqa: E402
import params as params_mod  # noqa: E402
import ref                   # noqa: E402

FIGS = os.path.join(ROOT, "figs")

plt.rcParams["font.family"] = ["IPAGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _c(rgb):
    """series_style の 0..255 タプルを matplotlib の 0..1 に。"""
    return tuple(v / 255.0 for v in rgb)


NEU, WARN, COR, INK, MUT, BG, HAIR = (_c(c) for c in (
    ss.NEUTRAL, ss.WARN, ss.CORRECT, ss.INK, ss.MUT, ss.BG, ss.HAIR))
GLASS = (0.86, 0.88, 0.91)
LIQ = (0.28, 0.55, 0.82)


def fig_to_array(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


# =====================================================================
# GIF 1: カムの割出し -- 一定入力 -> 間欠出力
# =====================================================================

def _draw_dial_input(ax, psi):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.35)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=MUT, lw=2.0))
    ax.plot([0, math.cos(psi)], [0, math.sin(psi)], color=INK, lw=3.0,
            solid_capstyle="round")
    ax.add_patch(Circle((0, 0), 0.07, color=INK))
    ax.text(0, 1.20, "入力軸（一定の速さで回り続ける）", ha="center",
            fontsize=11, color=INK)


def _draw_dial_table(ax, th, in_index):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.35)
    ax.add_patch(Circle((0, 0), 1.0, fill=True, fc=(0.95, 0.96, 0.97),
                        ec=MUT, lw=2.0, zorder=1))
    # 8 つの持ち場（ボトル）。45 度対称なので 1 タクトで見た目が戻る。
    for k in range(8):
        a = th + k * math.pi / 4.0
        x, y = 0.80 * math.cos(a), 0.80 * math.sin(a)
        ax.add_patch(Circle((x, y), 0.11, fc=LIQ, ec=INK, lw=1.0, zorder=3))
    ax.add_patch(Circle((0, 0), 0.06, color=INK, zorder=4))
    # 世界に固定のノズル（真上）。停留のあいだ、下に来たボトルに注ぐ。
    ax.add_patch(Polygon([[-0.10, 1.18], [0.10, 1.18], [0.0, 0.98]],
                         closed=True, fc=INK, zorder=5))
    if not in_index:
        ax.add_patch(Circle((0, 0.80), 0.11, fc=COR, ec=INK, lw=1.4, zorder=6))
        ax.text(0.62, 1.14, "充填中", color=COR, fontsize=11, ha="left")
    lab = "止まっている（充填）" if not in_index else "回っている（割出し）"
    col = COR if not in_index else WARN
    ax.text(0, -1.18, lab, ha="center", fontsize=12, color=col, fontweight="bold")
    ax.text(0, 1.20, "テーブル（動いて、止まる）", ha="center",
            fontsize=11, color=INK)


def build_cam(p, N=54):
    tact = p.tact
    tt = np.linspace(0, tact, 500)
    th_curve = np.array([math.degrees(ref.table_motion(ref.cam_input_angle(t, p), p)[0])
                         for t in tt])
    frames = []
    for i in range(N):
        t = tact * i / N
        psi = ref.cam_input_angle(t, p)
        th = ref.table_motion(psi, p)[0]
        in_index = ref.is_index_phase(psi, p)
        fig = plt.figure(figsize=(9.2, 5.4), dpi=110)
        fig.patch.set_facecolor(BG)
        axL = fig.add_axes([0.02, 0.34, 0.44, 0.60]); axL.set_facecolor(BG)
        axR = fig.add_axes([0.50, 0.34, 0.46, 0.60]); axR.set_facecolor(BG)
        _draw_dial_input(axL, psi)
        _draw_dial_table(axR, th, in_index)
        axB = fig.add_axes([0.09, 0.10, 0.86, 0.19]); axB.set_facecolor(BG)
        axB.axvspan(p.index_time, tact, color=(0.90, 0.91, 0.93), zorder=0)
        axB.plot(tt, th_curve, color=MUT, lw=1.6)
        axB.plot([t], [math.degrees(th)], "o", color=INK, ms=8, zorder=5)
        axB.set_xlim(0, tact); axB.set_ylim(-4, 50)
        axB.set_yticks([0, 45]); axB.set_yticklabels(["0°", "45°"], fontsize=9)
        axB.set_xlabel("時間 [秒]（1 タクト = 3.0 秒）", fontsize=10, color=INK)
        axB.text(p.index_time * 0.5, 47, "回す 1.5 秒", ha="center", fontsize=9,
                 color=WARN)
        axB.text(p.index_time + (tact - p.index_time) * 0.5, 47, "止める 1.5 秒",
                 ha="center", fontsize=9, color=COR)
        for s in ("top", "right"):
            axB.spines[s].set_visible(False)
        frames.append(fig_to_array(fig))
        plt.close(fig)
    return frames


# =====================================================================
# GIF 2: 共振 -- タクト 2.4 s（拍子が合う）と 3.0 s（ずらす）
# =====================================================================

def sim_tilt(p, tact, n_cycles):
    """タクト tact でスロッシングを n_cycles まわし、(サイクル番号, サイクル最大傾き[度])。

    生の傾きは 3.75 Hz で往復するので連続では読めない。1 サイクル（止まる 1 回）ごとの
    最大傾きだけを取ると、拍子が合うほど積み上がる様子が 1 本の線になる。
    """
    p.tact = tact
    p.dwell_time = tact - p.index_time
    h = p.height_from_volume(p.target_volume)
    w1 = ref.slosh_omega(p.R, h, p.g)
    st = ref.SloshState()
    dt = p.dt
    n_steps = int(round(tact / dt))
    peaks = []
    for cyc in range(n_cycles):
        cyc_peak = 0.0
        for k in range(n_steps):
            t = k * dt
            psi = ref.cam_input_angle(t, p)
            _th, omega, alpha = ref.table_motion(psi, p)
            a_t, a_r = ref.bottle_accelerations(omega, alpha, p)
            ref.step_slosh(st, a_t, a_r, omega, alpha, w1, p.zeta, dt, p.g, True)
            cyc_peak = max(cyc_peak, st.tilt())
        peaks.append(math.degrees(cyc_peak))
    return np.arange(1, n_cycles + 1), np.array(peaks)


def _draw_bottle(ax, tilt_deg, level_frac=0.80, face=LIQ, R=1.0, body=2.0):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-0.35, body + 0.7)
    # ガラス（角丸の胴）
    ax.add_patch(FancyBboxPatch((-R, 0), 2 * R, body,
                                boxstyle="round,pad=0,rounding_size=0.18",
                                fc=GLASS, ec=(0.6, 0.63, 0.67), lw=1.6, zorder=1))
    lvl = level_frac * body
    dz = R * math.tan(math.radians(tilt_deg))
    surf = [(-R, lvl - dz), (R, lvl + dz)]
    poly = [(-R, 0), (R, 0), surf[1], surf[0]]
    ax.add_patch(Polygon(poly, closed=True, fc=face, ec="none", zorder=2))
    ax.plot([-R, R], [lvl - dz, lvl + dz], color=(0.12, 0.30, 0.5), lw=2.6,
            zorder=3, solid_capstyle="round")
    ax.axhline(lvl, xmin=0.12, xmax=0.88, color=MUT, lw=0.8, ls=":", zorder=2)


def build_resonance(p, frames_per_cycle=10, n_cycles=6):
    pa = params_mod.load(); ta, ya = sim_tilt(pa, 2.4, n_cycles)
    pb = params_mod.load(); tb, yb = sim_tilt(pb, 3.0, n_cycles)
    ymax = max(ya.max(), yb.max()) * 1.18
    axA = np.concatenate([[0.0], ta.astype(float)])   # 補間の起点 (0,0) を足す
    ayA = np.concatenate([[0.0], ya])
    ayB = np.concatenate([[0.0], yb])
    N = frames_per_cycle * n_cycles
    frames = []
    for i in range(N + 1):
        g = n_cycles * i / N                       # 現在のサイクル進行度
        aA = float(np.interp(g, axA, ayA))         # いまのサイクル最大傾き[度]
        aB = float(np.interp(g, axA, ayB))
        rock = math.cos(2.0 * math.pi * 2.5 * i / N)   # 見せるための緩い往復
        fig = plt.figure(figsize=(9.4, 5.6), dpi=110)
        fig.patch.set_facecolor(BG)
        axL = fig.add_axes([0.02, 0.34, 0.30, 0.60]); axL.set_facecolor(BG)
        axR = fig.add_axes([0.34, 0.34, 0.30, 0.60]); axR.set_facecolor(BG)
        _draw_bottle(axL, aA * rock); _draw_bottle(axR, aB * rock)
        axL.set_title("拍子が合う（タクト 2.4 秒）", color=WARN, fontsize=12,
                      fontweight="bold")
        axR.set_title("ずらす（タクト 3.0 秒）", color=COR, fontsize=12,
                      fontweight="bold")
        axL.text(0, -0.28, f"揺れ {aA:.1f}°", ha="center", color=WARN, fontsize=11)
        axR.text(0, -0.28, f"揺れ {aB:.1f}°", ha="center", color=COR, fontsize=11)
        # 右側: サイクルごとの最大傾きの積み上がり
        axS = fig.add_axes([0.70, 0.16, 0.28, 0.74]); axS.set_facecolor(BG)
        fx = np.linspace(0, max(g, 1e-6), 160)
        axS.plot(fx, np.interp(fx, axA, ayA), color=WARN, lw=2.4)
        axS.plot(fx, np.interp(fx, axA, ayB), color=COR, lw=2.4)
        mA = ta <= g + 1e-6
        axS.plot(ta[mA], ya[mA], "o", color=WARN, ms=6)
        axS.plot(tb[mA], yb[mA], "o", color=COR, ms=6)
        axS.set_xlim(0, n_cycles + 0.2); axS.set_ylim(0, ymax)
        axS.set_xticks(range(0, n_cycles + 1))
        axS.set_xlabel("止まった回数", fontsize=10, color=INK)
        axS.set_ylabel("液面の傾き [度]", fontsize=10, color=INK)
        for s in ("top", "right"):
            axS.spines[s].set_visible(False)
        fig.text(0.02, 0.95, "止まるたびに、拍子が合う左のほうが大きく揺れていく",
                 fontsize=12, color=INK)
        frames.append(fig_to_array(fig))
        plt.close(fig)
    return ss.boomerang(frames)


# =====================================================================
# GIF 3: エイリアシング -- 速い振動を遅いセンサで読む
# =====================================================================

def build_alias(p, N=48):
    x = np.linspace(0, 1, 1400)
    f_fast = 9.5                     # 窓の中の速い振動の回数（速い振動の見立て）
    fast = np.sin(2 * math.pi * f_fast * x)
    n_samp = 11                      # 遅いセンサの読み取り点。alias = |9.5-11| = 1.5 山
    xs = (np.arange(n_samp) + 0.5) / n_samp
    ys_true = np.sin(2 * math.pi * f_fast * xs)         # 落とさずに読む
    ys_filt = 0.06 * np.sin(2 * math.pi * f_fast * xs)  # 事前に落として読む（1/20 以下）
    frames = []
    for i in range(N + 1):
        r = i / N                    # 左から右へ読み進む割合
        fig = plt.figure(figsize=(9.4, 5.4), dpi=110)
        fig.patch.set_facecolor(BG)
        panels = [(0.55, ys_true, 0.06 * fast, "そのまま読む", WARN, True,
                   "つなぐと見える遅い山（本当は無い）"),
                  (0.06, ys_filt, 0.06 * fast, "速い分を先に落としてから読む", COR, False,
                   "つないでもほぼ平ら")]
        for ax0, ys, resid, title, tcol, show_fast, line_lab in panels:
            ax = fig.add_axes([0.07, ax0, 0.88, 0.35]); ax.set_facecolor(BG)
            mx = x <= r
            if show_fast:
                ax.plot(x[mx], fast[mx], color=(0.72, 0.74, 0.78), lw=1.0, zorder=1,
                        label="本当の速い振動")
            else:
                ax.plot(x[mx], resid[mx], color=(0.80, 0.82, 0.85), lw=1.0, zorder=1,
                        label="落としたあとの残り")
            ms = xs <= r
            xk, yk = xs[ms], ys[ms]
            ax.vlines(xk, 0, yk, color=(0.62, 0.64, 0.68), lw=0.8, zorder=2)
            if len(xk) >= 2:
                ax.plot(xk, yk, "-", color=tcol, lw=2.6, zorder=3, label=line_lab)
            ax.plot(xk, yk, "o", color=tcol, ms=7, zorder=4, label="読み取り点")
            ax.axvline(r, color=HAIR, lw=1.0, zorder=0)
            ax.set_xlim(0, 1); ax.set_ylim(-1.25, 1.25)
            ax.set_yticks([]); ax.set_xticks([])
            ax.set_title(title, color=tcol, fontsize=12, fontweight="bold", loc="left")
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=3)
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_visible(False)
        fig.text(0.07, 0.955, "速い振動を、遅い読み取りで見ると", fontsize=12,
                 color=INK)
        frames.append(fig_to_array(fig))
        plt.close(fig)
    return ss.boomerang(frames)


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cam", "res", "alias"], default=None)
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    if a.only in (None, "cam"):
        p = params_mod.load()
        info = ss.write_gif(build_cam(p), os.path.join(FIGS, "art_cam_motion.gif"),
                            fps=16)
        print("cam  ", info["w"], info["h"], info["frames"], f'{info["size_mb"]}MB')
    if a.only in (None, "res"):
        p = params_mod.load()
        info = ss.write_gif(build_resonance(p), os.path.join(FIGS, "art_resonance.gif"),
                            fps=15)
        print("res  ", info["w"], info["h"], info["frames"], f'{info["size_mb"]}MB')
    if a.only in (None, "alias"):
        p = params_mod.load()
        info = ss.write_gif(build_alias(p), os.path.join(FIGS, "art_alias.gif"),
                            fps=16)
        print("alias", info["w"], info["h"], info["frames"], f'{info["size_mb"]}MB')


if __name__ == "__main__":
    main()
