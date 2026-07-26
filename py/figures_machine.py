"""前半（機械を組む）の図。`py/ref.py` の物理から直に描く。

`figs/` には他の担当の図もあるので、`machine_` で始まる名前しか書かない。

    .venv/bin/python -m py.figures_machine
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import params as params_mod  # noqa: E402
import ref  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent.parent / "figs"
PREFIX = "machine_"

plt.rcParams["font.family"] = ["IPAGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def _save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{PREFIX}{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig] {path}")
    return path


# ---------------------------------------------------------------------------

def fig_cam_motion(p):
    """カムの形が決めるテーブルの動き。角度・速さ・加速度を 1 タクトぶん。

    入力軸は一定の速さで回り続けていて、テーブルが動いて止まるのはカムの形による。
    割出しと停留がはっきり分かれることを見せる。
    """
    n = 2000
    t = np.linspace(0.0, p.tact, n)
    th = np.empty(n)
    om = np.empty(n)
    al = np.empty(n)
    for k, tk in enumerate(t):
        psi = ref.cam_input_angle(tk, p)
        th[k], om[k], al[k] = ref.table_motion(psi, p)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True)

    axes[0].plot(t, np.degrees(th), color="tab:blue")
    axes[0].set_ylabel("角度 [度]")
    axes[0].set_yticks([0, 15, 30, 45])

    axes[1].plot(t, om, color="tab:blue")
    axes[1].set_ylabel("速さ [rad/s]")
    axes[1].axhline(ref.table_omega_max(p), color="gray", lw=0.8, ls=":")
    axes[1].annotate(f"最大 {ref.table_omega_max(p):.3f}",
                     xy=(0.02, 0.80), xycoords="axes fraction",
                     fontsize=9, color="gray")

    axes[2].plot(t, al, color="tab:blue")
    axes[2].set_ylabel("加速度 [rad/s$^2$]")
    axes[2].set_xlabel("時間 [s]")
    axes[2].axhline(0.0, color="gray", lw=0.6)
    a_max = ref.table_alpha_max(p)
    axes[2].annotate(f"最大 {a_max:.2f}（テーブル縁で {ref.tangential_accel_max(p):.2f} m/s$^2$）",
                     xy=(0.30, 0.88), xycoords="axes fraction",
                     fontsize=9, color="gray")

    for ax in axes:
        ax.axvspan(p.index_time, p.tact, color="0.92", zorder=0)
        ax.grid(alpha=0.3)
        ax.set_xlim(0.0, p.tact)

    axes[0].annotate("回っている（1.5 s）", xy=(p.index_time * 0.5, 42),
                     ha="center", fontsize=10)
    axes[0].annotate("止まっている（1.5 s）",
                     xy=(p.index_time + p.dwell_time * 0.5, 42),
                     ha="center", fontsize=10)

    axes[0].set_title("カムの形が決めるテーブルの動き（1 タクト = 3.0 s）")
    fig.tight_layout()
    return _save(fig, "cam_motion")


def fig_tact_resonance(p):
    """タクトの整数倍と、液の揺れやすい周波数の位置関係。2.4 s と 3.0 s の比較。

    2.4 s では 9 倍がちょうど重なる。3.0 s では最寄りが外れる。
    """
    h = p.height_from_volume(p.target_volume)
    w1 = ref.slosh_omega(p.R, h, p.g)
    f1 = w1 / (2.0 * math.pi)
    half = ref.resonance_half_width_hz(w1, p.zeta)

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.6), sharex=True)
    x_hi = 5.0

    for ax, tact, label in ((axes[0], 2.4, "タクト 2.4 s（最初に置いた値）"),
                            (axes[1], 3.0, "タクト 3.0 s（いまの値）")):
        f_tact = 1.0 / tact
        orders = np.arange(1, int(x_hi / f_tact) + 1)
        freqs = orders * f_tact

        # 液が揺れやすい帯
        ax.axvspan(f1 - half, f1 + half, color="tab:red", alpha=0.18, zorder=0)
        ax.axvline(f1, color="tab:red", lw=1.4, zorder=2)

        ax.vlines(freqs, 0, 1, color="tab:blue", lw=1.2, zorder=3)

        # 最寄りの整数倍
        k = int(np.argmin(np.abs(freqs - f1)))
        f_near = freqs[k]
        detune = abs(f_near - f1)
        ax.plot([f_near], [1.0], "o", color="tab:blue", ms=6, zorder=4)
        ax.annotate(f"{orders[k]} 倍 = {f_near:.4f}",
                    xy=(f_near, 1.0), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color="tab:blue")
        ratio = detune / half
        text = (f"ずれは幅の {ratio * 100:.0f} %" if ratio < 1.0
                else f"ずれは幅の {ratio:.1f} 倍")
        ax.annotate(text,
                    xy=(0.98, 0.72), xycoords="axes fraction",
                    ha="right", fontsize=10,
                    color="tab:red" if ratio < 1.0 else "0.35")

        ax.set_ylim(0, 1.5)
        ax.set_yticks([])
        ax.set_xlim(0, x_hi)
        ax.set_title(label, fontsize=11, loc="left")
        ax.grid(axis="x", alpha=0.3)

    axes[0].annotate(f"液が揺れやすい {f1:.4f} 回/秒",
                     xy=(f1, 1.35), xytext=(10, 0), textcoords="offset points",
                     fontsize=10, color="tab:red")
    axes[1].set_xlabel("1 秒あたりの回数")
    fig.tight_layout()
    return _save(fig, "tact_resonance")


def main():
    p = params_mod.load()
    fig_cam_motion(p)
    fig_tact_resonance(p)


if __name__ == "__main__":
    main()
