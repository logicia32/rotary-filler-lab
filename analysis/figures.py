"""図。`run_matrix.py` が残した材料（figdata.npz）と判定結果から描く。

`figs/` には他の担当の図もあるので、`analysis_` で始まる名前しか書かない。

    .venv/bin/python -m analysis.figures --workdir <作業ディレクトリ>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config, features  # noqa: E402
from .dataset import CONDITIONS  # noqa: E402

FIG_DIR = config.LAB_ROOT / "figs"
PREFIX = "analysis_"

# 日本語が出るフォントを先頭に置く。無ければ英語のまま化けずに落ちるより、
# 見出しだけ英語になるほうがましなので、フォールバックを並べておく。
plt.rcParams["font.family"] = ["IPAGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

COLORS = {
    "normal": "tab:blue",
    "missing": "tab:red",
    "valve": "tab:orange",
    "cam": "tab:purple",
    "bearing": "tab:red",
}


def _save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{PREFIX}{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig] {path}")
    return path


def _get(d, key, default=None):
    return d[key] if key in d else default


# ---------------------------------------------------------------------------

def fig_order_spectrum(fd, res):
    """次数比スペクトル。次数 1 は真値には立つが、センサ出力では雑音に埋もれる。"""
    la = res.get("long_average", {})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    panels = [("truth", "① 真値（センサ連鎖に入る前）"),
              ("out", "② センサ出力（雑音 0.5 ustrain rms を通したあと）")]
    for ax, (tag, title) in zip(axes, panels):
        for cond in ("normal", "missing"):
            x = _get(fd, f"ord1_{cond}_{tag}_x")
            y = _get(fd, f"ord1_{cond}_{tag}_y")
            if x is None:
                continue
            v = la.get(cond, {})
            lab = CONDITIONS[cond]["label"]
            if tag == "truth" and v:
                lab += f"（次数 1 = {v['truth_order1_1rev']:.3g}）"
            ax.semilogy(x[1:], y[1:], lw=0.9, color=COLORS[cond], label=lab)
            k1 = int(np.argmin(np.abs(x - 1.0)))
            ax.plot([x[k1]], [y[k1]], "o", ms=7, mfc="none", mew=1.6,
                    color=COLORS[cond])
        ax.axvline(1.0, color="tab:green", ls="--", lw=1.2)
        ax.text(1.6, 0.97, "次数 1（回転同期）", transform=ax.get_xaxis_transform(),
                va="top", fontsize=9, color="tab:green")
        for k in (7, 9, 15, 17, 23, 25):
            ax.axvline(k, color="0.75", ls=":", lw=0.8)
        ax.text(9.4, 0.02, "次数 1±8k（割出しでできる側帯波）",
                transform=ax.get_xaxis_transform(), fontsize=8, color="0.45")
        ax.set_xlim(0, 30)
        ax.set_xlabel("次数 [回転 1 回あたりの周期数]")
        ax.set_ylabel("ひずみの振幅 [ustrain]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower left")
    v = la.get("missing", {})
    if v:
        axes[1].axhline(v["out_order1_1rev"], color="0.35", ls="-.", lw=1.2)
        axes[1].text(14, v["out_order1_1rev"] * 1.15,
                     f"次数 1 のビンの雑音床 {v['out_order1_1rev']:.3g}",
                     fontsize=8, color="0.35")
    fig.suptitle("次数比スペクトル（ひずみゲージ・定常の 1 回転）: "
                 "欠品は回転に同期し次数 1 に出る（1 テーブル回転に 1 回）。"
                 "ただしこの諸元では雑音の下",
                 fontsize=11)
    return _save(fig, "order_spectrum")


def fig_envelope(fd, res):
    """包絡線: 軸受はモータ軸（定速 1500 rpm）にあり、欠陥通過 BPFO は時間的に等間隔。

    時間波形には 1/BPFO 間隔の衝撃が並び、時間領域のまま包絡線をスペクトルにすると
    BPFO とその倍数に線が立つ。テーブル角の次数へは移さない（軸受はテーブルと同期
    しないので、割出し次数に立てても意味がない）。
    """
    bpfo = _bpfo_hz()
    period_ms = 1.0e3 / bpfo
    t_show = 12.0 / bpfo                       # 衝撃 12 発ぶんの時間窓
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))

    ax = axes[0]
    for cond in ("normal", "bearing"):        # 衝撃をあとに描いて前に出す
        t = _get(fd, f"envwave_{cond}_t")
        y = _get(fd, f"envwave_{cond}_y")
        if t is None:
            continue
        m = t <= t_show
        ax.plot(t[m], y[m], lw=0.9, color=COLORS[cond], label=CONDITIONS[cond]["label"])
    ev = _get(fd, "envwave_bearing_events")
    if ev is not None and len(ev):
        ev = ev[ev <= t_show]
        for i, te in enumerate(ev):
            ax.axvline(te, color="tab:green", ls="--", lw=0.8,
                       label="真の衝撃時刻" if i == 0 else None)
    ax.set_xlim(0, t_show)
    ax.set_xlabel("時刻 [s]")
    ax.set_ylabel("包絡線 [m/s^2]")
    ax.set_title(f"① accel_hf の包絡線。衝撃は {period_ms:.1f} ms 間隔で等間隔に並ぶ",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for cond in ("normal", "bearing"):
        t = _get(fd, f"envwave_{cond}_t")
        y = _get(fd, f"envwave_{cond}_y")
        if t is None:
            continue
        fs = 1.0 / float(np.median(np.diff(t)))
        freq, amp = features.amplitude_spectrum(y, fs, window="hann")
        ax.semilogy(freq, amp, lw=0.9, color=COLORS[cond],
                    label=CONDITIONS[cond]["label"])
    for k in range(1, 4):
        ax.axvline(bpfo * k, color="tab:green", ls="--", lw=1.0,
                   label=f"BPFO {bpfo:.1f} Hz とその倍数" if k == 1 else None)
    ax.set_xlim(0, bpfo * 3.6)
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("包絡線の振幅 [m/s^2]")
    ax.set_title("② 時間軸のまま包絡線を FFT。BPFO の線が立つ", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("包絡線解析: 軸受はモータ軸（定速 1500 rpm）にあり欠陥通過は時間的に一定。"
                 "時間領域の包絡線復調で BPFO に立つ", fontsize=11)
    return _save(fig, "envelope")


def _bpfo_hz() -> float:
    """軸受外輪の欠陥通過周波数 BPFO [Hz]。機械の定数（故障の有無に依らない）。

    BPFO = 欠陥次数 × モータ回転数。ダンプヘッダの bearing_defect_freq_hz は
    故障無効だと 0 になるので、正常と故障で同じ帯を見るために params から出す。
    """
    params, _ = config.load_machine_params()
    fault = params["faults"]["bearing_outer_race"]
    motor_rev_s = float(params["drive"]["motor_rpm_at_operating_point"]) / 60.0
    return float(fault["defect_freq_ratio"]) * motor_rev_s


def _slosh_hz() -> float:
    params, _ = config.load_machine_params()
    from sensors.virtual import slosh_frequency
    R = float(params["bottle"]["inner_diameter_mm"]) * 1e-3 / 2.0
    V = float(params["fill"]["target_volume_mL"]) * 1e-6
    h = V / (np.pi * R ** 2)
    return slosh_frequency(h, R, float(params["sim"]["gravity_m_s2"]))


def fig_slosh(fd, res):
    """スロッシングの線を、真値とセンサ出力で見比べる。

    揺れは故障ではないので「異常として出るか」ではなく
    「そのセンサでそもそも観測できるか」を見る。
    """
    f1 = _slosh_hz()
    sv = res.get("startup", {}).get("slosh_visibility", {})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    panels = [("strain", "ひずみ [ustrain]", "① ひずみゲージ"),
              ("accel_lf_tangential", "加速度 [m/s^2]", "② accel_lf 接線（接地・250 Hz）")]
    for ax, (chan, ylab, title) in zip(axes, panels):
        f = _get(fd, f"strut_missing_{chan}_f")
        a = _get(fd, f"strut_missing_{chan}_a")
        if f is not None:
            ax.semilogy(f, a, lw=0.9, color="tab:red", label="真値（連鎖に入る前）")
        fo = _get(fd, f"samp_missing_{chan}_f")
        ao = _get(fd, f"samp_missing_{chan}_a")
        if fo is not None:
            ax.semilogy(fo, ao, lw=0.6, color="0.6", label="センサ出力")
        v = sv.get(chan)
        if v and np.isfinite(v.get("floor", float("nan"))):
            ax.axhline(v["floor"], color="tab:blue", ls="-.", lw=1.3,
                       label="雑音床（力が 0 のときの出力）")
            ax.set_title(f"{title}   S/N {v['snr_db']:+.1f} dB", fontsize=10)
        else:
            ax.set_title(title, fontsize=10)
        ax.axvline(f1, color="tab:green", ls="--", lw=1.0)
        ax.text(f1 * 1.05, 0.03, f"{f1:.3f} Hz", transform=ax.get_xaxis_transform(),
                fontsize=9, color="tab:green")
        ax.set_xlim(0, 12)
        ax.set_xlabel("周波数 [Hz]")
        ax.set_ylabel(f"振幅 {ylab}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    tacts = config.ANALYSIS["startup"]["tact_variants_s"]
    for c, tact in zip(("tab:purple", "tab:orange"), tacts):
        f = _get(fd, f"tacttruth_{tact:g}_f")
        a = _get(fd, f"tacttruth_{tact:g}_a")
        if f is None:
            continue
        ax.semilogy(f, a, lw=0.9, color=c, label=f"タクト {tact:g} s（真値）")
        for k in range(1, 40):
            fk = k / tact
            if fk > 12:
                break
            ax.axvline(fk, color=c, ls=":", lw=0.6, alpha=0.45)
    ax.axvline(f1, color="tab:green", ls="--", lw=1.4)
    ax.text(f1 * 0.985, 0.03, f"{f1:.3f} Hz は動かない", ha="right",
            transform=ax.get_xaxis_transform(), fontsize=9, color="tab:green")
    ax.set_xlim(2.6, 5.0)
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("ひずみ [ustrain]")
    ax.set_title("③ タクトを変える（点線 = タクトの高調波・3〜5 Hz を拡大）", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("スロッシング（立ち上げの 1 回転・欠品）: "
                 "揺れは確かにあるが、雑音床から出るかはセンサで決まる", fontsize=11)
    return _save(fig, "slosh_band")


def fig_structure(fd, res, cond="missing"):
    """構造モデル 3 通り。検出できるかどうかがここだけで裏返る。"""
    f1 = _slosh_hz()
    struct = res.get("startup", {}).get("structure", {}).get(cond, {})
    keys = [k for k in struct if not k.startswith("_")]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax = axes[0]
    cols = ["tab:blue", "tab:orange", "tab:red"]
    floor = None
    for c, key in zip(cols, keys):
        f = _get(fd, f"struct_{cond}_{key}_ftruth")
        p = _get(fd, f"struct_{cond}_{key}_atruth")
        if f is None:
            continue
        ax.semilogy(f, p, lw=0.9, color=c, label=struct[key]["label"])
        floor = struct[key]["channels"]["accel_lf_tangential"]["floor_amp"]
    if floor and np.isfinite(floor):
        ax.axhline(floor, color="0.4", ls="-.", lw=1.4,
                   label="センサの雑音床（同じ窓・同じ分解能）")
    ax.axvline(f1, color="tab:green", ls="--", lw=1.0)
    ax.set_xlim(0, 12)
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("加速度の真値の振幅 [m/s^2]")
    ax.set_title("accel_lf 接線の真値（センサ連鎖に入る前）", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    labels, vals = [], []
    for key in keys:
        c = struct[key]["channels"]["accel_lf_tangential"]
        labels.append(struct[key]["label"].replace("（", "\n（"))
        vals.append(c["snr_db"])
    bars = ax.bar(range(len(vals)), vals, color=cols[:len(vals)])
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("スロッシング帯の S/N [dB]")
    ax.set_title("雑音床に対する余裕。0 dB を割ると埋もれる", fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (2 if v > 0 else -6),
                f"{v:+.1f} dB", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    lo = min(vals) if vals else -70
    ax.set_ylim(lo - 15, max(max(vals) + 15, 15))

    fig.suptitle(f"架台の構造モデルだけで {abs(max(vals) - min(vals)):.0f} dB 動く"
                 f"（{CONDITIONS[cond]['label']}・params.json は書き換えずメモリ上で差し替え）",
                 fontsize=11)
    return _save(fig, "structure_model")


def fig_angle_domain(fd, res, cond="missing"):
    """停止区間の扱い。時間軸と角度軸で同じ信号を並べる。"""
    t = _get(fd, f"excerpt_{cond}_t")
    y = _get(fd, f"excerpt_{cond}_y")
    th = _get(fd, f"excerpt_{cond}_th")
    keep = _get(fd, f"excerpt_{cond}_keep")
    if t is None:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=False)

    ax = axes[0]
    ax.plot(t, th, lw=1.2, color="tab:blue")
    moving = np.zeros(len(t), dtype=bool)
    moving[keep] = True
    ax.fill_between(t, 0, th.max(), where=~moving, color="0.85", step="mid",
                    label="停止（角度が進まない）")
    ax.set_ylabel("テーブル角 [deg]")
    ax.set_title("① テーブル角。停止のあいだ角度は進まない", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    stopped_pct = 100.0 * float(np.mean(~moving))
    yo = _get(fd, f"excerpt_{cond}_out_y")
    ax.plot(t, y, lw=0.9, color="tab:red", label="真値（連鎖に入る前）")
    if yo is not None:
        span = float(np.max(np.abs(yo))) / max(float(np.max(np.abs(y))), 1e-12)
        ax.text(0.02, 0.06, f"センサ出力はこの縦軸の約 {span:.0f} 倍の幅で振れている"
                            "（雑音 0.5 ustrain rms）",
                transform=ax.transAxes, fontsize=8, color="0.35")
    ax.fill_between(t, y.min() * 3, y.max() * 3, where=~moving, color="0.9", step="mid")
    ax.set_xlabel("時刻 [s]")
    ax.set_ylabel("ひずみ [ustrain]")
    ax.set_ylim(y.min() * 3, y.max() * 3)
    ax.set_title(f"② 時間軸のまま。停止区間が信号の {stopped_pct:.0f} % を占める", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    thk, yk = th[keep], y[keep]
    ax.plot(thk, yk, lw=0.9, color="tab:red")
    ax.set_xlabel("テーブル角 [deg]")
    ax.set_ylabel("ひずみ [ustrain]")
    ax.set_title("③ 角度軸に並べ直したあと。停止区間は消え、割出しどうしが直につながる", fontsize=10)
    ax.grid(alpha=0.3)
    for k in range(1, 3):
        ax.axvline(45.0 * k, color="tab:green", ls=":", lw=1.0)
    ax.text(45.4, 0.95, "継ぎ目（45 度ごと）", transform=ax.get_xaxis_transform(),
            va="top", fontsize=8, color="tab:green")

    fig.suptitle("停止区間をどう扱うか: 角度領域には入れない、という取り決め"
                 "（②③ の赤はセンサ連鎖に入る前の真値）", fontsize=11)
    fig.tight_layout()
    return _save(fig, "angle_domain")


def make_all(workdir, res=None):
    workdir = Path(workdir)
    fd = np.load(workdir / "figdata.npz")
    if res is None:
        res = json.loads((workdir / "results.json").read_text(encoding="utf-8"))
    fig_order_spectrum(fd, res)
    fig_envelope(fd, res)
    fig_slosh(fd, res)
    fig_structure(fd, res)
    fig_angle_domain(fd, res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    make_all(a.workdir)
