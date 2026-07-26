"""3D では見えないものを 2D の線図で見せるパネル。matplotlib だけで描く。

3D の絵（`scene.py`）と左右に並べ、同じ時刻で再生する前提なので、
どのパネルも「ある時刻 t の 1 枚」を返すことだけを仕事にしてある。
PNG のパスを受け、書いた `Path` を返す。画素数は `size_px` で受ける。

| 関数 | 何を見せるか |
|---|---|
| `bottle_cross_section()` | ボトル 1 本の断面。液深と液面の傾き。記事の主役 |
| `cam_diagram()` | カム入力軸角とテーブル角の対応。割出しと停留の割付 |
| `sensor_panel()` | 波形。いまの時刻に縦線を引いて 3D と揃える |
| `cycle_timing()` | 1 タクト 3.0s のタイムチャート。停留の中の充填と余裕 |

物理はここに書かない
--------------------
スロッシングの応答も充填の流量計算もこのファイルの仕事ではない。
パネルは渡された値を絵にするだけで、値を作るのは物理コア側。
例外はカム曲線で、これは指令でも応答でもなく幾何そのものなので
`cam_diagram()` の中で計算している。

図の中の文字
------------
matplotlib の既定フォントに日本語が無く、そのまま書くと豆腐になる。
図中は英字と数値だけにしてある。説明はコード側のコメントに書く。

使い方
------
    .venv/bin/python viz/panels.py --outdir <書き出し先>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = LAB_ROOT / "params.json"

# 画素数を figsize に直すだけなので savefig で bbox を切らない。
# bbox="tight" にすると余白が詰まって、指定した画素数と出来上がりがずれる。
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLOR = {
    "glass": "0.82",
    "glass_edge": "0.35",
    "liquid": "tab:blue",
    "surface": "#12467b",
    "ref": "0.45",
    "index": "tab:orange",
    "dwell": "tab:green",
    "flow": "tab:blue",
    "delay": "0.55",
    "margin": "tab:red",
    "now": "tab:red",
    "curve": "tab:blue",
}


def load_params(path=PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _figure(size_px, dpi):
    """画素数で figure を作る。size_px = (幅, 高さ) [px]。"""
    w, h = int(size_px[0]), int(size_px[1])
    return plt.figure(figsize=(w / float(dpi), h / float(dpi)), dpi=dpi)


def _save(fig, out_path, dpi) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 変形正弦カム曲線
#
# 加速度が 3 区間の正弦で、位相の進み方だけが変わる。
#   0    .. 1/8 : 角速度 4pi   （0 から正の山へ）
#   1/8  .. 7/8 : 角速度 4pi/3 （正の山から負の谷へ）
#   7/8  .. 1   : 角速度 4pi   （負の谷から 0 へ）
# 変位が 0 から 1 になるよう振幅を決めると、その振幅がそのまま無次元最大
# 加速度 Ca = 4pi^2/(pi+4) = 5.5279 になり、無次元最大速度は Ca/pi = 1.7596。
# どちらも params.json の indexer.curve_Ca / curve_Cv と一致する。
# --------------------------------------------------------------------------
MS_CA = 4.0 * np.pi ** 2 / (np.pi + 4.0)
MS_CV = MS_CA / np.pi


def modified_sine(x):
    """変形正弦の無次元 変位・速度・加速度。x は 0..1 に正規化した時間（角度）。

    戻り値 (s, v, a) はいずれも x と同じ形。区間外は s を 0 / 1 に張り付け、
    v と a は 0 にする（停留のつながりがそのまま出る）。
    """
    x = np.asarray(x, dtype=float)
    A = MS_CA
    v1 = A / (4.0 * np.pi)                       # x=1/8 と x=7/8 での速度
    s1 = v1 * (1.0 / 8.0 - 1.0 / (4.0 * np.pi))  # x=1/8 での変位
    s2 = s1 + v1 * 0.75 + 9.0 * A / (8.0 * np.pi ** 2)   # x=7/8 での変位

    s = np.zeros_like(x)
    v = np.zeros_like(x)
    a = np.zeros_like(x)

    m1 = (x >= 0.0) & (x < 1.0 / 8.0)
    m2 = (x >= 1.0 / 8.0) & (x < 7.0 / 8.0)
    m3 = (x >= 7.0 / 8.0) & (x <= 1.0)
    over = x > 1.0

    u = x[m1]
    s[m1] = v1 * (u - np.sin(4.0 * np.pi * u) / (4.0 * np.pi))
    v[m1] = v1 * (1.0 - np.cos(4.0 * np.pi * u))
    a[m1] = A * np.sin(4.0 * np.pi * u)

    u = x[m2] - 1.0 / 8.0
    w = 4.0 * np.pi / 3.0
    s[m2] = s1 + v1 * u + (9.0 * A / (16.0 * np.pi ** 2)) * (1.0 - np.cos(w * u))
    v[m2] = v1 + (3.0 * A / (4.0 * np.pi)) * np.sin(w * u)
    a[m2] = A * np.cos(w * u)

    u = x[m3] - 7.0 / 8.0
    s[m3] = s2 + v1 * u + (A / (16.0 * np.pi ** 2)) * (np.cos(4.0 * np.pi * u) - 1.0)
    v[m3] = v1 - (A / (4.0 * np.pi)) * np.sin(4.0 * np.pi * u)
    a[m3] = -A * np.cos(4.0 * np.pi * u)

    s[over] = 1.0
    return s, v, a


def curve_check(params: dict = None, n: int = 200001) -> dict:
    """変形正弦の検算。無次元の最大速度・最大加速度と、両端の閉じ具合を返す。

    params を渡すと indexer.curve_Ca / curve_Cv との差も入れる。
    曲線の式を触ったら必ずここを通す。合わなければ式が間違っている。
    """
    x = np.linspace(0.0, 1.0, n)
    s, v, a = modified_sine(x)
    out = {
        "Cv": float(np.max(np.abs(v))),
        "Ca": float(np.max(np.abs(a))),
        "s_end": float(s[-1]),
        "v_end": float(v[-1]),
        "a_end": float(a[-1]),
        # 速度を台形則で積分した変位と解析式のずれ。式の取り違えがあれば出る
        "s_int_err": float(abs(np.trapezoid(v, x) - 1.0)),
    }
    if params is not None:
        out["Cv_ref"] = float(params["indexer"]["curve_Cv"])
        out["Ca_ref"] = float(params["indexer"]["curve_Ca"])
        out["Cv_err"] = out["Cv"] - out["Cv_ref"]
        out["Ca_err"] = out["Ca"] - out["Ca_ref"]
    return out


# --------------------------------------------------------------------------
# ボトル断面
# --------------------------------------------------------------------------
def _bottle_outline(b: dict):
    """ガラスの断面をひと筆で囲む閉じた輪郭。左外側 → 左内側 → 底 → 右内側 → 右外側。

    寸法は params.json の bottle だけを読む。ここに数値を直書きしない。
    """
    wt = float(b["wall_thickness_mm"])
    ri = float(b["inner_diameter_mm"]) / 2.0
    ro = ri + wt
    rni = float(b["neck_diameter_mm"]) / 2.0
    rno = rni + wt
    y_body = float(b["body_height_mm"])
    y_sh = y_body + float(b["shoulder_height_mm"])
    y_top = y_sh + float(b["neck_height_mm"])

    pts = [
        (-ro, -wt), (-ro, y_body), (-rno, y_sh), (-rno, y_top),
        (-rni, y_top), (-rni, y_sh), (-ri, y_body), (-ri, 0.0),
        (ri, 0.0), (ri, y_body), (rni, y_sh), (rni, y_top),
        (rno, y_top), (rno, y_sh), (ro, y_body), (ro, -wt),
    ]
    return np.array(pts), dict(ri=ri, ro=ro, y_body=y_body, y_sh=y_sh, y_top=y_top)


def bottle_cross_section(params: dict, level_mm: float, tilt_rad: float, out_path,
                         size_px=(640, 720), dpi=100, t_s=None,
                         annotate=True) -> Path:
    """ボトル 1 本の断面。液深 `level_mm` と液面の傾き `tilt_rad` を絵にする。

    横軸は傾きの向きに取る。`tilt_rad` が正のとき右側で液面が上がる
    （`scene.tilt_to_world()` と同じ約束で、正なら「その向きの側で上がる」）。
    液面は傾いた直線で、回した中心を液面の中心に置いてあるので、傾けても
    液量は変わらない（3D 側の `parts.liquid()` と同じ作り）。

    揺れの実際の傾きは 0.068 rad（3.9 度）しかない。誇張はしないので、
    液面の線を太くし、傾き 0 の基準線を薄く重ね、壁での液面の上下 dz を
    寸法として添えて読めるようにしてある。
    """
    b = params["bottle"]
    poly, dim = _bottle_outline(b)
    ri, y_body, y_top = dim["ri"], dim["y_body"], dim["y_top"]

    level = float(np.clip(level_mm, 0.0, y_body))
    tilt = float(tilt_rad)
    dz = ri * np.tan(tilt)          # 壁での液面の上下。MODEL.md 2 節の dz = R tan(tilt)

    fig = _figure(size_px, dpi)
    ax = fig.add_axes([0.13, 0.07, 0.84, 0.86])

    # ガラス
    ax.fill(poly[:, 0], poly[:, 1], facecolor=COLOR["glass"],
            edgecolor=COLOR["glass_edge"], lw=1.4, zorder=2)

    # 液。傾いた液面で切った台形。塗りは胴の中だけを考える約束なので上下で切る。
    # 線のほうは切らない。切ると傾きが実際より寝て見えるので、縁を越えたときは
    # 越えたまま描いて、そのことが分かるようにしておく
    xs = np.array([-ri, ri])
    zs = level + xs * np.tan(tilt)
    zc = np.clip(zs, 0.0, y_body)
    ax.fill([-ri, ri, ri, -ri], [0.0, 0.0, zc[1], zc[0]],
            facecolor=COLOR["liquid"], alpha=0.32, edgecolor="none", zorder=1)

    # 傾き 0 の基準線。これが無いと 3.9 度は目で拾えない
    ax.plot([-ri, ri], [level, level], ls=(0, (5, 4)), lw=1.0,
            color=COLOR["ref"], zorder=3)
    # 液面そのもの
    ax.plot(xs, zs, lw=3.0, color=COLOR["surface"], solid_capstyle="round", zorder=4)

    # 壁での上下を寸法で見せる
    if abs(dz) > 1e-6:
        ax.annotate("", xy=(ri * 0.92, zs[1]), xytext=(ri * 0.92, level),
                    arrowprops=dict(arrowstyle="<->", color=COLOR["surface"], lw=1.1),
                    zorder=5)
        ax.text(ri * 1.10, (level + zs[1]) / 2.0, f"dz = {dz:+.2f} mm",
                fontsize=9, color=COLOR["surface"], va="center", ha="left")
        ax.annotate("", xy=(-ri * 0.92, zs[0]), xytext=(-ri * 0.92, level),
                    arrowprops=dict(arrowstyle="<->", color=COLOR["surface"], lw=1.1),
                    zorder=5)
        ax.text(-ri * 1.10, (level + zs[0]) / 2.0, f"{-dz:+.2f}",
                fontsize=9, color=COLOR["surface"], va="center", ha="right")

    if annotate:
        # 首の上端（195mm）に文字がかからないよう、箱の幅は詰めてある
        lines = []
        if t_s is not None:
            lines.append(f"t     {float(t_s):7.3f} s")
        lines += [
            f"tilt  {np.degrees(tilt):+7.2f} deg",
            f"      {tilt:+7.4f} rad",
            f"depth {level:7.1f} mm",
            f"dz    {dz:+7.2f} mm",
        ]
        ax.text(0.985, 0.985, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
        ax.text(0.03, 0.02, "dashed line: level at tilt = 0", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, color=COLOR["ref"])

    ax.set_aspect("equal")
    ax.set_xlim(-95.0, 95.0)
    ax.set_ylim(-20.0, y_top + 20.0)
    ax.set_xlabel("x [mm]  (+ = direction of positive tilt)", fontsize=9)
    ax.set_ylabel("height [mm]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_title("Bottle cross-section", fontsize=11)
    return _save(fig, out_path, dpi)


# --------------------------------------------------------------------------
# カム線図
# --------------------------------------------------------------------------
def cam_diagram(params: dict, psi_deg: float, out_path,
                size_px=(640, 480), dpi=100) -> Path:
    """入力軸角 psi とテーブル角の対応。3 段で 変位・速度・加速度。

    入力軸は 20rpm で回りっぱなしで、割付 180 度でテーブルが 45 度進み、
    残り 180 度は停留する。停留区間を塗り分けてあり、そこが「止まっている
    時間」= 充填できる時間だと一目で分かるようにしてある。

    速度と加速度は無次元で描く（最大が Cv / Ca そのものになるので、
    カタログの値と図の上で突き合わせられる）。
    """
    idx = params["indexer"]
    tbl = params["table"]
    beta = float(idx["index_angle_input_deg"])
    dwell = float(idx["dwell_angle_input_deg"])
    d_th = float(tbl["index_angle_deg"])
    ca_ref = float(idx["curve_Ca"])
    cv_ref = float(idx["curve_Cv"])

    psi = np.linspace(0.0, beta + dwell, 1441)
    x = np.clip(psi / beta, 0.0, 1.0)
    s, v, a = modified_sine(x)
    theta = d_th * s
    v[psi > beta] = 0.0
    a[psi > beta] = 0.0

    psi_now = float(psi_deg) % (beta + dwell)
    x_now = min(psi_now / beta, 1.0)
    s_n, v_n, a_n = modified_sine(np.array([x_now]))
    if psi_now > beta:
        v_n[0] = 0.0
        a_n[0] = 0.0

    fig = _figure(size_px, dpi)
    axes = fig.subplots(3, 1, sharex=True,
                        gridspec_kw=dict(height_ratios=[1.5, 1.0, 1.0],
                                         hspace=0.12, left=0.13, right=0.97,
                                         top=0.90, bottom=0.12))
    series = [
        (theta, "table angle [deg]", COLOR["curve"], d_th * s_n[0]),
        (v, "vel [-]", "tab:purple", v_n[0]),
        (a, "accel [-]", "tab:brown", a_n[0]),
    ]
    for ax, (y, lab, col, y_now) in zip(axes, series):
        ax.axvspan(beta, beta + dwell, color=COLOR["dwell"], alpha=0.13, lw=0)
        ax.plot(psi, y, lw=1.8, color=col)
        ax.axvline(psi_now, color=COLOR["now"], lw=1.2)
        ax.plot([psi_now], [y_now], "o", ms=6, color=COLOR["now"], zorder=5)
        ax.set_ylabel(lab, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlim(0.0, beta + dwell)

    axes[0].set_ylim(-4.0, d_th + 8.0)
    axes[0].text(beta * 0.5, d_th * 0.12, "INDEX", ha="center", fontsize=10,
                 color=COLOR["index"], weight="bold")
    axes[0].text(beta + dwell * 0.5, d_th * 0.12, "DWELL (table stopped)",
                 ha="center", fontsize=10, color="tab:green", weight="bold")
    axes[0].text(0.02, 0.95, f"psi = {psi_now:6.1f} deg", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=9, family="monospace",
                 color=COLOR["now"])

    # 最大値をカタログ値の線で重ねる。曲線が違えば線から外れて見える
    for ax, ref, tag in ((axes[1], cv_ref, "Cv"), (axes[2], ca_ref, "Ca")):
        ax.axhline(ref, ls=":", lw=1.0, color="0.4")
        ax.axhline(-ref, ls=":", lw=1.0, color="0.4")
        ax.text(beta + dwell * 0.02, ref, f"{tag} = {ref:.4f}", fontsize=8,
                va="bottom", color="0.3")
        ax.set_ylim(-ref * 1.6, ref * 1.6)

    axes[-1].set_xlabel("cam input shaft angle psi [deg]", fontsize=9)
    axes[-1].set_xticks(np.arange(0.0, beta + dwell + 1.0, 45.0))
    axes[0].set_title(f"Cam index unit, modified sine  "
                      f"({beta:.0f} deg index / {dwell:.0f} deg dwell)", fontsize=11)
    return _save(fig, out_path, dpi)


# --------------------------------------------------------------------------
# 波形パネル
# --------------------------------------------------------------------------
def fixed_ylim(channels: dict, margin: float = 0.08) -> dict:
    """全区間から軸の範囲を一度だけ決める。コマ送りはこれを毎コマ渡す。

    軸がコマごとに伸び縮みすると、アニメでちらついて波形が読めなくなる。
    """
    lim = {}
    for name, y in channels.items():
        y = np.asarray(y, dtype=float)
        lo, hi = float(np.min(y)), float(np.max(y))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            lo, hi = lo - 1.0, hi + 1.0
        pad = (hi - lo) * margin
        lim[name] = (lo - pad, hi + pad)
    return lim


def _pick_ylim(ylim, name, i):
    """ylim は dict / (lo,hi) の並び / 全段共通の (lo,hi) のどれでも受ける。"""
    if ylim is None:
        return None
    if isinstance(ylim, dict):
        return ylim.get(name)
    seq = list(ylim)
    if len(seq) == 2 and np.isscalar(seq[0]):
        return (float(seq[0]), float(seq[1]))
    try:
        return seq[i]
    except IndexError:
        return None


def sensor_panel(t, channels: dict, t_now: float, out_path,
                 size_px=(960, 480), dpi=100, ylabels=None, title=None,
                 ylim=None) -> Path:
    """波形パネル。段数はチャンネル数に合わせる。

    channels は {"accel_lf [m/s2]": 配列, "strain [ue]": 配列, ...}。
    キーがそのまま段のラベルになる（ylabels を渡せば差し替え）。
    `t_now` の縦線が 3D の絵と時刻を揃えるための線。

    `ylim` は全コマで軸を固定するための引数。dict（チャンネル名 → (lo,hi)）、
    段ごとの (lo,hi) の並び、全段共通の (lo,hi) のいずれでも受ける。
    アニメを作るときは `fixed_ylim()` で一度決めた dict を毎コマ渡す。
    """
    names = list(channels.keys())
    n = max(len(names), 1)
    t = np.asarray(t, dtype=float)

    fig = _figure(size_px, dpi)
    axes = fig.subplots(n, 1, sharex=True, squeeze=False,
                        gridspec_kw=dict(hspace=0.14, left=0.145, right=0.98,
                                         top=0.90 if title else 0.96, bottom=0.13))
    axes = [a[0] for a in axes]

    for i, name in enumerate(names):
        ax = axes[i]
        y = np.asarray(channels[name], dtype=float)
        ax.plot(t, y, lw=0.9, color=COLOR["curve"])
        ax.axvline(float(t_now), color=COLOR["now"], lw=1.4)
        lab = name if ylabels is None else list(ylabels)[i]
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        lim = _pick_ylim(ylim, name, i)
        if lim is not None:
            ax.set_ylim(lim)
        if t.size:
            ax.set_xlim(float(t[0]), float(t[-1]))

    axes[-1].set_xlabel("t [s]", fontsize=9)
    axes[0].text(0.995, 1.02, f"t = {float(t_now):.3f} s", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=9, family="monospace",
                 color=COLOR["now"])
    if title:
        fig.suptitle(title, fontsize=11)
    return _save(fig, out_path, dpi)


# --------------------------------------------------------------------------
# タイムチャート
# --------------------------------------------------------------------------
def cycle_timing(params: dict, t_now: float, out_path,
                 size_px=(960, 240), dpi=100) -> Path:
    """1 タクトのタイムチャート。割出し・停留と、その中の充填弁の開閉。

    停留 1.5s に対して充填が何秒使い、余裕がどれだけ残るかを読ませる図。
    余裕は params.json の fill._timing_note と同じ数え方
    （弁の開き遅れ + 充填 + 閉じ遅れ を停留から引く）。start_delay_s は
    この数え方には入っていないので、図でも積まない。
    """
    cyc = params["cycle"]
    fil = params["fill"]
    tact = float(cyc["tact_s"])
    t_idx = float(cyc["index_time_s"])
    t_dwell = float(cyc["dwell_time_s"])
    t_open = float(fil["valve_open_delay_s"])
    t_close = float(fil["valve_close_delay_s"])
    t_fill = float(fil["target_volume_mL"]) / float(fil["flow_rate_mL_s"])
    t_used = t_open + t_fill + t_close
    t_margin = t_dwell - t_used

    x0 = t_idx
    x_flow = x0 + t_open
    x_shut = x_flow + t_fill
    x_end = x_shut + t_close

    fig = _figure(size_px, dpi)
    ax = fig.add_axes([0.10, 0.27, 0.88, 0.55])

    bars = [
        # (段, 開始, 長さ, 色, alpha, ラベル)
        (1, 0.0, t_idx, COLOR["index"], 0.75, "INDEX  45 deg"),
        (1, t_idx, t_dwell, COLOR["dwell"], 0.45, "DWELL  (table stopped)"),
        (0, x0, t_open, COLOR["delay"], 0.85, ""),
        (0, x_flow, t_fill, COLOR["flow"], 0.75, f"FILL  {t_fill:.3f} s"),
        (0, x_shut, t_close, COLOR["delay"], 0.85, ""),
        (0, x_end, t_margin, COLOR["margin"], 0.22, ""),
    ]
    for row, x, w, col, al, lab in bars:
        ax.broken_barh([(x, w)], (row - 0.34, 0.68), facecolors=col, alpha=al,
                       edgecolor="0.35", lw=0.8)
        if lab:
            ax.text(x + w / 2.0, row, lab, ha="center", va="center", fontsize=9,
                    color="black")

    # 弁の開閉遅れは 30ms しかなく帯の中に文字が入らないので下へ引き出す。
    # 上（テーブルの帯の側）へ出すと INDEX / DWELL の文字と重なる
    ax.annotate(f"valve open {t_open * 1e3:.0f} ms", xy=(x_flow, -0.36),
                xytext=(x_flow - 0.26, -0.62), fontsize=8, ha="center",
                va="center", arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
    ax.annotate(f"close {t_close * 1e3:.0f} ms", xy=(x_shut, -0.36),
                xytext=(x_shut + 0.20, -0.62), fontsize=8, ha="center",
                va="center", arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))

    # 余裕。ここが読めることがこの図の目的
    ax.annotate("", xy=(x_end, -1.02), xytext=(tact, -1.02),
                arrowprops=dict(arrowstyle="<->", color=COLOR["margin"], lw=1.4))
    ax.text((x_end + tact) / 2.0, -1.10, f"margin {t_margin:.3f} s",
            ha="center", va="top", fontsize=9, color=COLOR["margin"])

    ax.axvline(float(t_now) % tact, color=COLOR["now"], lw=1.6, zorder=5)
    ax.set_xlim(0.0, tact)
    ax.set_ylim(-1.45, 1.55)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["VALVE", "TABLE"], fontsize=9)
    ax.set_xticks(np.arange(0.0, tact + 1e-9, 0.25))
    ax.set_xlabel("t in one tact [s]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_title(f"Tact {tact:.1f} s   index {t_idx:.1f} s / dwell {t_dwell:.1f} s"
                 f"   valve busy {t_used:.3f} s", fontsize=10)
    return _save(fig, out_path, dpi)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # ここで作る波形は仮のもの。形と軸の見え方を確かめるためだけに置いてある。
    # 物理コアの書き直しが終わったら、core のダンプを読んでそれに差し替える。
    # 液の反力も充填の流量計算も入っていないので、この波形の値そのものに
    # 意味は無い（カムから出る接線加速度だけは厳密）。

    def _table_kinematics(t, params):
        """カム曲線からテーブルの角度・角加速度・接線加速度を出す。ここは厳密。"""
        tact = float(params["cycle"]["tact_s"])
        t_idx = float(params["cycle"]["index_time_s"])
        d_th = np.radians(float(params["table"]["index_angle_deg"]))
        rp = float(params["table"]["pitch_circle_diameter_mm"]) / 2000.0
        x = np.clip((t % tact) / t_idx, 0.0, 1.0)
        s, v, a = modified_sine(x)
        alpha = a * d_th / t_idx ** 2
        alpha[(t % tact) > t_idx] = 0.0
        return d_th * s, alpha, rp * alpha

    def _pendulum(t, drive, f_hz, zeta):
        """1 自由度の振り子。phi'' + 2 zeta w phi' + w^2 phi = -a/L1（MODEL.md 2 節）。"""
        w = 2.0 * np.pi * f_hz
        dt = float(t[1] - t[0])
        phi = np.zeros_like(t)
        vel = 0.0
        p = 0.0
        for i in range(t.size):
            phi[i] = p
            acc = -drive[i] * w ** 2 / 9.80665 - 2.0 * zeta * w * vel - w ** 2 * p
            vel += acc * dt
            p += vel * dt
        return phi

    ap = argparse.ArgumentParser(description="2D パネルの確認用に 1 組焼く")
    ap.add_argument("--outdir", type=Path, default=LAB_ROOT / "figs",
                    help="PNG の書き出し先")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = load_params()

    chk = curve_check(params)
    print("modified sine:")
    print(f"  Cv = {chk['Cv']:.4f}  (params {chk['Cv_ref']:.4f}, "
          f"diff {chk['Cv_err']:+.2e})")
    print(f"  Ca = {chk['Ca']:.4f}  (params {chk['Ca_ref']:.4f}, "
          f"diff {chk['Ca_err']:+.2e})")
    print(f"  s(1) = {chk['s_end']:.6f}  v(1) = {chk['v_end']:.2e}  "
          f"a(1) = {chk['a_end']:.2e}  int err = {chk['s_int_err']:.2e}")

    # 仮の時系列。1 タクト 3.0s を 2 周ぶん
    fs = 2000.0
    tact = float(params["cycle"]["tact_s"])
    tt = np.arange(0.0, 2.0 * tact, 1.0 / fs)
    theta, alpha, a_t = _table_kinematics(tt, params)
    f_slosh = 3.7514
    zeta = float(params["liquid"]["slosh_damping_ratio"])
    phi_t = _pendulum(tt, a_t, f_slosh, zeta)

    # 液量は停留の中で増やすだけの仮の作り。流量計算はしていない
    fil = params["fill"]
    t_fill = float(fil["target_volume_mL"]) / float(fil["flow_rate_mL_s"])
    t0 = float(params["cycle"]["index_time_s"]) + float(fil["valve_open_delay_s"])
    frac = np.clip(((tt % tact) - t0) / t_fill, 0.0, 1.0)
    vol = 400.0 * frac

    ri = float(params["bottle"]["inner_diameter_mm"]) / 2.0
    level = vol * 1000.0 / (np.pi * ri * ri)

    channels = {
        "accel_lf [m/s2]": a_t,
        "tilt phi_t [mrad]": phi_t * 1e3,
        "strain [ue]": 12.0 * a_t + 3.0 * phi_t * 1e3,
        "fill volume [mL]": vol,
    }
    ylim = fixed_ylim(channels)

    made = []
    made.append(bottle_cross_section(params, 120.5, 0.0, outdir / "panel_bottle_p00.png"))
    made.append(bottle_cross_section(params, 120.5, +0.068,
                                     outdir / "panel_bottle_pos.png", t_s=1.62))
    made.append(bottle_cross_section(params, 120.5, -0.068,
                                     outdir / "panel_bottle_neg.png", t_s=1.75))
    made.append(cam_diagram(params, 72.0, outdir / "panel_cam_072.png"))
    made.append(cam_diagram(params, 250.0, outdir / "panel_cam_250.png"))
    made.append(cycle_timing(params, 1.85, outdir / "panel_cycle.png"))

    # 軸固定の確認。5 コマとも同じ ylim を渡す
    for k in range(5):
        t_now = 1.20 + 0.15 * k
        made.append(sensor_panel(tt, channels, t_now,
                                 outdir / f"panel_sensor_{k}.png", ylim=ylim,
                                 title="virtual sensors (placeholder waveform)"))

    print(f"max |a_t| = {np.max(np.abs(a_t)):.5f} m/s2  "
          f"(params {params['cycle']['tangential_accel_max_m_s2']})")
    print(f"max |phi_t| = {np.degrees(np.max(np.abs(phi_t))):.3f} deg")
    print(f"level at 400 mL = {level.max():.1f} mm")
    for p in made:
        print(f"  [png] {p}")
