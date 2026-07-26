"""仮想センサ層の実演。図は figs/sensor_*.png に出る。

    .venv/bin/python -m sensors.demo

やること

1. 物理コア（core/filler が吐く RFILL004）を回して連続量のセンサ信号を作る。
2. 軸受とカムフォロワの故障を入れた回もまわし、衝撃を**イベント列**から合成して足す。
3. 連鎖の各段が効いていることを図で見せる。
   - 段ごとの効き方（帯域制限・標本化と量子化・ノイズ）
   - 折り返し（帯域制限が無いとナイキストを超える成分が化ける）
   - **加速度センサが 2 つ要る理由**（低速側では衝撃が見えず、高速側では揺れが見えない）
4. スロッシング成分が量子化ノイズの床からどれだけ上にあるかを数字で出す。

連続量に依存する図は物理コアから作る。コアが無いときは、コア不要で描ける
段ごとの効き方（stages）と折り返し（alias）だけ描いて、残りは省く。
黙って代用データを作らない。
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):  # 直接叩かれたとき用
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sensors"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import chain  # noqa: E402
from . import virtual  # noqa: E402
from .read_dump import load_run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "figs")
SEED = 20260722
DURATION_S = 12.0     # 数回転ぶん。液が入ってスロッシングが立ち上がるところまで見たい


# ---------------------------------------------------------------------------
# スペクトル
# ---------------------------------------------------------------------------

def amplitude_density(x, fs):
    """片側の振幅スペクトル密度 [単位/rtHz]。ハン窓、雑音の床が読める正規化。"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    scale = np.sqrt(2.0 / (fs * np.sum(w ** 2)))
    return f, np.abs(X) * scale


def band_level(f, a, f0, half_width=0.5):
    """f0 の周りでいちばん高いところの値と、その周波数。"""
    m = (f > f0 - half_width) & (f < f0 + half_width)
    if not np.any(m):
        return np.nan, np.nan
    i = int(np.argmax(a[m]))
    return float(a[m][i]), float(f[m][i])


def floor_level(f, a, f0, f1):
    """f0〜f1 の中央値を床の目安として返す。"""
    m = (f >= f0) & (f <= f1)
    return float(np.median(a[m])) if np.any(m) else np.nan


def quantization_floor(ch):
    """量子化だけで決まる振幅密度の床 = (LSB/sqrt(12)) / sqrt(fs/2)。"""
    return ch.info["lsb"] / np.sqrt(12.0) / np.sqrt(ch.fs_hz / 2.0)


def db(x):
    return 20.0 * np.log10(x)


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

def fig_timeseries(truth, chans, path, title, window=4.8, lp_hz=8.0):
    """真値・センサ出力・センサ出力を低域通過したもの を重ねる。

    生の出力はノイズで振り切れるので、縦軸は真値の大きさに合わせてある。
    低域通過は連鎖の一部ではなく、後段（検出側）が何を見るかの目安。
    """
    items = [
        ("accel_lf_radial", "架台 柱 半径 加速度 [m/s^2]"),
        ("strain", "支持軸 根元 曲げひずみ [ustrain]"),
        ("current", "モータ電流 [A]"),
    ]
    fig, axes = plt.subplots(len(items), 1, figsize=(10, 9), sharex=True)
    t1 = float(truth.t[-1])
    t0 = max(float(truth.t[0]), t1 - window)   # 液が入ったあとを見たいので後ろ側
    for ax, (name, label) in zip(axes, items):
        ch = chans[name]
        m = (ch.truth_t >= t0) & (ch.truth_t <= t1)
        ms = (ch.t >= t0) & (ch.t <= t1)
        lp = chain.butter2_lowpass(ch.y, ch.fs_hz, lp_hz)
        ax.plot(ch.t[ms], ch.y[ms], lw=0.3, color="tab:red", alpha=0.16,
                label=f"センサ出力 {ch.fs_hz:.0f} Hz / {ch.info['bits']} bit"
                      "（縦軸の外まで振れている）")
        ax.plot(ch.truth_t[m], ch.truth[m], lw=1.1, color="0.25", label="真値")
        ax.plot(ch.t[ms], lp[ms], lw=1.0, color="tab:blue", alpha=0.9,
                label=f"センサ出力を {lp_hz:.0f} Hz で低域通過（参考）")
        span = max(float(np.max(np.abs(ch.truth[m]))),
                   float(np.max(np.abs(lp[ms])))) * 1.4
        if span > 0:
            ax.set_ylim(-span, span)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("時刻 [s]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_stages(params, path):
    """段ごとに何が起きるかを、素性の分かった試験信号で見せる（accel_lf の諸元）。"""
    g = float(params["sim"]["gravity_m_s2"])
    spec = virtual.spec_from_params(params, "accel_lf", g)
    spec.channel = "probe"
    fs_in = float(params["sim"]["log_rate_hz"])
    n = int(fs_in * 1.0)
    t = np.arange(n) / fs_in
    x = 0.06 * np.sin(2 * np.pi * 3.75 * t)   # スロッシングに相当する成分

    aa = chain.butter2_lowpass(x, fs_in, spec.bandwidth_hz)
    ys, _ = chain.resample(aa, fs_in, spec.sample_rate_hz)
    ts = chain.timebase(len(ys), spec.sample_rate_hz)
    yq = chain.quantize(ys, spec.full_scale, spec.bits)          # ノイズ無し
    yn = chain.add_noise(ys, spec.noise_rms, SEED, "probe")
    yf = chain.saturate(chain.quantize(yn, spec.full_scale, spec.bits),
                        spec.range_amplitude)
    lsb = chain.lsb_of(spec.full_scale, spec.bits)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    ax = axes[0]
    fc = spec.bandwidth_hz
    probe_f = np.geomspace(1.0, 0.45 * fs_in, 45)
    gain = []
    for f0 in probe_f:
        nn = max(int(fs_in * max(20.0 / f0, 0.2)), 2048)
        tt = np.arange(nn) / fs_in
        yy = chain.butter2_lowpass(np.sin(2 * np.pi * f0 * tt), fs_in, fc)
        half = yy[nn // 2:]          # 過渡が抜けたところで振幅を測る
        gain.append(np.sqrt(2.0) * np.sqrt(np.mean(half ** 2)))
    gain = np.array(gain)
    theory = 1.0 / np.sqrt(1.0 + (probe_f / fc) ** 4)
    b, a = chain.biquad_lowpass(fc, fs_in)
    z = np.exp(-2j * np.pi * probe_f / fs_in)
    digital = np.abs((b[0] + b[1] * z + b[2] * z ** 2)
                     / (a[0] + a[1] * z + a[2] * z ** 2))
    ax.loglog(probe_f, gain, "o", ms=3.5, color="tab:red", label="測った振幅比")
    ax.loglog(probe_f, digital, lw=1.2, color="tab:blue", label="係数から出した離散系の応答")
    ax.loglog(probe_f, theory, lw=1.0, ls="--", color="0.5",
              label="連続系 1/sqrt(1+(f/fc)^4)")
    ax.axvline(fc, color="tab:green", ls="--", lw=1, label=f"fc = {fc:.0f} Hz")
    ax.axhline(1 / np.sqrt(2), color="0.5", ls=":", lw=1, label="-3.01 dB")
    ax.axvline(2 * fc, color="tab:orange", ls=":", lw=1,
               label=f"2fc = {2*fc:.0f} Hz（-12.3 dB のはず）")
    ax.set_title("① 帯域制限: 2 次バターワースの振幅特性"
                 "（双一次変換なのでナイキストで 0 に落ちる）", fontsize=10)
    ax.set_ylabel("振幅比", fontsize=9)
    ax.set_xlabel("周波数 [Hz]", fontsize=9)
    ax.set_ylim(1e-2, 2)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    m2 = (ts >= 0.0) & (ts <= 0.012)
    mm = (t >= 0.0) & (t <= 0.012)
    ax.plot(t[mm], aa[mm], lw=1.0, color="tab:blue", label="帯域制限後（連続とみなす）")
    ax.step(ts[m2], yq[m2], where="post", lw=1.0, color="tab:green",
            label=f"標本化 {spec.sample_rate_hz:.0f} Hz + 量子化 {spec.bits} bit")
    ax.plot(ts[m2], yq[m2], ".", ms=4, color="tab:green")
    lo = float(np.min(yq[m2]))
    ax.set_ylim(lo - 2 * lsb, lo + 12 * lsb)
    ax.set_title(f"② 標本化と量子化: 段差 = LSB = {lsb:.4g} m/s^2"
                 "（縦軸は 14 LSB ぶんに拡大）", fontsize=10)
    ax.set_ylabel("加速度 [m/s^2]", fontsize=9)

    ax = axes[2]
    ax.plot(t[mm], aa[mm], lw=1.2, color="tab:blue", label="帯域制限後")
    ax.plot(ts[m2], yf[m2], ".-", ms=3, lw=0.7, color="tab:red",
            label=f"ノイズ込みの最終出力（rms {spec.noise_rms:.3g}"
                  f" = {spec.noise_rms/lsb:.0f} LSB）")
    ax.set_title("③ ノイズ: この諸元では量子化よりノイズのほうがずっと大きい", fontsize=10)
    ax.set_ylabel("加速度 [m/s^2]", fontsize=9)
    ax.set_xlabel("時刻 [s]")

    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        if ax is not axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("信号連鎖の段ごとの効き方（accel_lf の諸元）", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_spectrum(truth, chans, chans_quiet, path, f_slosh):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ["accel_lf_radial", "strain"]):
        ch = chans[name]
        chq = chans_quiet[name]
        ft, at = amplitude_density(ch.truth, truth.fs_hz)
        fy, ay = amplitude_density(ch.y, ch.fs_hz)
        fq, aq = amplitude_density(chq.y, chq.fs_hz)
        ax.loglog(ft, at, lw=0.8, color="0.4", label="真値（ログ周波数）")
        ax.loglog(fy, ay, lw=0.8, color="tab:red", alpha=0.8, label="センサ出力")
        if np.any(chq.y != 0.0):
            ax.loglog(fq, aq, lw=0.8, color="tab:blue", alpha=0.7,
                      label="ノイズ抜き（量子化だけ）")
        else:
            # 信号が 1 LSB より小さいと、ノイズが無い場合の出力は全部 0 になる。
            # 実センサではノイズがディザとして働き、平均すれば情報が残る。
            ax.text(0.30, 0.90,
                    "ノイズを 0 にすると出力は全て 0。\n"
                    f"信号（〜{np.max(np.abs(ch.truth)):.1g}）が"
                    f" 1 LSB（{ch.info['lsb']:.2g}）より小さいため。\n"
                    "実センサではノイズがディザとして働く",
                    transform=ax.transAxes, fontsize=7, color="tab:blue",
                    va="top", bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
        bw = ch.info["bandwidth_hz"]
        if bw:
            ax.axvline(bw, color="tab:green", ls="--", lw=1, label=f"帯域制限 {bw:.0f} Hz")
        ax.axvline(f_slosh, color="tab:orange", ls=":", lw=1.2,
                   label=f"スロッシング {f_slosh:.2f} Hz")
        qf = quantization_floor(ch)
        ax.axhline(qf, color="tab:blue", ls="-.", lw=1, label=f"量子化の床 {qf:.2g}")
        ax.set_xlabel("周波数 [Hz]")
        ax.set_ylabel(f"振幅密度 [{ch.unit}/rtHz]")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_xlim(0.3, max(fy[-1], ft[-1]))
    fig.suptitle("スペクトル: 帯域制限の肩と、量子化ノイズの床", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_alias(path):
    """ナイキストを超える成分を入れて、アンチエイリアスの有無を比べる。"""
    fs_in, fs_out = 5000.0, 500.0
    fc = 200.0
    f_sig = 2200.0          # 出力側ナイキスト 250 Hz の遥か上
    f_alias = abs(f_sig - round(f_sig / fs_out) * fs_out)
    n = int(fs_in * 2.0)
    t = np.arange(n) / fs_in
    x = np.sin(2 * np.pi * 40.0 * t) + np.sin(2 * np.pi * f_sig * t)

    y_no, _ = chain.resample(x, fs_in, fs_out)
    y_aa, _ = chain.resample(chain.butter2_lowpass(x, fs_in, fc), fs_in, fs_out)

    f0, a0 = amplitude_density(y_no, fs_out)
    f1, a1 = amplitude_density(y_aa, fs_out)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    tt = chain.timebase(len(y_no), fs_out)
    m = tt < 0.1
    ax.plot(t[t < 0.1], x[t < 0.1], lw=0.7, color="0.6", label="入力 40 Hz + 2200 Hz")
    ax.plot(tt[m], y_no[m], "o-", ms=3, lw=0.9, color="tab:red",
            label="帯域制限なしで 500 Hz に間引き")
    ax.plot(tt[m], y_aa[m], "s-", ms=3, lw=0.9, color="tab:blue",
            label="2 次バターワース 200 Hz を通してから間引き")
    ax.set_xlabel("時刻 [s]")
    ax.set_ylabel("振幅")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("時間波形", fontsize=10)

    ax = axes[1]
    ax.semilogy(f0, a0, lw=0.9, color="tab:red", label="帯域制限なし")
    ax.semilogy(f1, a1, lw=0.9, color="tab:blue", label="帯域制限あり")
    ax.axvline(f_alias, color="tab:orange", ls=":", lw=1.3,
               label=f"折り返し先 {f_alias:.0f} Hz")
    ax.axvline(40.0, color="0.4", ls="--", lw=1, label="本来の 40 Hz")
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("振幅密度 [1/rtHz]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(f"2200 Hz は 500 Hz 標本化で {f_alias:.0f} Hz に化ける", fontsize=10)
    fig.suptitle("アンチエイリアスの有無", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

    lvl_no, _ = band_level(f0, a0, f_alias, 3.0)
    lvl_aa, _ = band_level(f1, a1, f_alias, 3.0)
    return lvl_no, lvl_aa


def fig_two_accels(dump, params, events, chans, path, f_slosh, seed=SEED):
    """加速度センサが 2 つ要る理由を 4 枚で見せる。

    左上: 衝撃 1 発を accel_hf で見たところ（3 kHz のリンギングが乗る）
    右上: 同じ衝撃を accel_lf で見たところ（帯域制限があれば消え、無ければ化ける）
    左下: 衝撃まわりのスペクトル
    右下: スロッシングの帯域。低速側は見え、高速側は取り付けと交流結合で消える
    """
    g = float(params["sim"]["gravity_m_s2"])
    fs_hi = virtual.impact_rate_hz(events.ring_freq_hz)
    t0 = float(dump.t[0])
    dur = float(dump.t[-1] - dump.t[0])
    raw = virtual.impact_train(events, t0, dur, fs_hi)

    # 帯域制限を外した accel_lf（アンチエイリアス無しでどうなるか）
    p_noaa = copy.deepcopy(params)
    spec_lf = virtual.spec_from_params(params, "accel_lf", g)
    spec_lf.channel = "accel_lf_radial"
    spec_lf.allow_upsample = True
    spec_noaa = virtual.spec_from_params(p_noaa, "accel_lf", g)
    spec_noaa.bandwidth_hz = None
    spec_noaa.channel = "accel_lf_radial"
    spec_noaa.allow_upsample = True

    zeros = np.zeros_like(dump.t)
    y_lf, _ = chain.run_chain(zeros, dump.log_rate_hz, spec_lf, seed,
                              extra=raw, extra_fs=fs_hi)
    y_noaa, _ = chain.run_chain(zeros, dump.log_rate_hz, spec_noaa, seed,
                                extra=raw, extra_fs=fs_hi)
    t_lf = chain.timebase(len(y_lf), spec_lf.sample_rate_hz, t0)

    ch_hf = chans["accel_hf_radial"]
    te = float(events.times[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0][0]
    m = (ch_hf.t >= te - 0.002) & (ch_hf.t <= te + 0.012)
    ax.plot(ch_hf.t[m] - te, ch_hf.y[m], lw=0.8, color="tab:red",
            label=f"accel_hf 出力 {ch_hf.fs_hz/1000:.1f} kHz")
    ax.set_title(f"accel_hf: 衝撃が {events.ring_freq_hz/1000:.0f} kHz で鳴る"
                 f"（振幅 {events.amplitudes[0]:.1f} m/s^2）", fontsize=10)
    ax.set_xlabel("衝撃からの時間 [s]")
    ax.set_ylabel("加速度 [m/s^2]")

    ax = axes[0][1]
    # 欠陥通過 89.5 Hz で衝撃が密に出るので、重ねるのは先頭数発に限る。
    # 衝撃の減衰は約 7 ms、欠陥通過の周期は約 11 ms なので、1 発ぶんの窓に切れる。
    n_overlay = 8
    peaks_no, peaks_aa = [], []
    for k, tev in enumerate(events.times[:n_overlay]):
        m = (t_lf >= tev - 0.002) & (t_lf <= tev + 0.009)
        if not np.any(m):
            continue
        peaks_no.append(float(np.max(np.abs(y_noaa[m]))))
        peaks_aa.append(float(np.max(np.abs(y_lf[m]))))
        ax.plot(t_lf[m] - tev, y_noaa[m], "-", lw=0.9, color="tab:orange", alpha=0.8,
                label="帯域制限を外した場合（偽の低周波）" if k == 0 else None)
        ax.plot(t_lf[m] - tev, y_lf[m], "-", lw=0.9, color="tab:blue", alpha=0.8,
                label=f"規定どおり {spec_lf.bandwidth_hz:.0f} Hz で帯域制限"
                if k == 0 else None)
    ax.set_title(f"accel_lf ({spec_lf.sample_rate_hz:.0f} Hz): 同じ衝撃を"
                 f"先頭 {len(peaks_no)} 発ぶん重ねた", fontsize=10)
    ax.set_xlabel("衝撃からの時間 [s]")
    ax.set_ylabel("加速度 [m/s^2]")
    ax.text(0.02, 0.03,
            "3000 Hz は 1000 Hz 標本化でちょうど 0 Hz に落ちるので、\n"
            "見かけの大きさが標本化の位相しだいで毎回変わる",
            transform=ax.transAxes, fontsize=7, color="0.3")

    ax = axes[1][0]
    win_hf = (ch_hf.t >= te - 0.01) & (ch_hf.t <= te + 0.09)
    f_hf, a_hf = amplitude_density(ch_hf.y[win_hf], ch_hf.fs_hz)
    win_lf = (t_lf >= te - 0.01) & (t_lf <= te + 0.09)
    f_l, a_l = amplitude_density(y_lf[win_lf], spec_lf.sample_rate_hz)
    f_n, a_n = amplitude_density(y_noaa[win_lf], spec_noaa.sample_rate_hz)
    ax.loglog(f_hf, a_hf, lw=0.8, color="tab:red", label="accel_hf")
    ax.loglog(f_n, a_n, lw=0.8, color="tab:orange", label="accel_lf 帯域制限なし")
    ax.loglog(f_l, a_l, lw=0.8, color="tab:blue", label="accel_lf 規定どおり")
    ax.axvline(events.ring_freq_hz, color="0.3", ls=":", lw=1.2,
               label=f"{events.ring_freq_hz:.0f} Hz")
    ax.axvline(0.5 * spec_lf.sample_rate_hz, color="tab:green", ls="--", lw=1,
               label=f"accel_lf のナイキスト {0.5*spec_lf.sample_rate_hz:.0f} Hz")
    ax.set_title("衝撃まわりのスペクトル（衝撃 1 発の窓）", fontsize=10)
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("振幅密度 [m/s^2/rtHz]")
    ax.set_ylim(1e-8, 1e-1)

    ax = axes[1][1]
    # 単位が違うので、各チャネルの真値を「そのチャネル自身の床」で割って比べる。
    # 1 を超えていれば、そのセンサでは信号が雑音より上にある。
    for name, color, lab in (
            ("accel_lf_radial", "tab:blue", "accel_lf（DC 結合・柱）"),
            ("accel_hf_radial", "tab:red", "accel_hf（交流結合・軸受箱）"),
            ("strain", "tab:green", "strain（支持軸の曲げ）")):
        ch = chans[name]
        f, a = amplitude_density(ch.y, ch.fs_hz)
        fl = floor_level(f, a, 8.0, 30.0)
        ft, at = amplitude_density(ch.truth, ch.truth_t.size /
                                   (ch.truth_t[-1] - ch.truth_t[0]))
        sel = (ft > 0.3) & (ft < 40.0)
        ax.loglog(ft[sel], at[sel] / fl, lw=0.8, color=color, label=lab)
    ax.axhline(1.0, color="0.3", ls="-", lw=1.2, label="そのチャネルの床")
    ax.axvline(f_slosh, color="tab:orange", ls=":", lw=1.3,
               label=f"スロッシング {f_slosh:.2f} Hz")
    ax.set_title("低周波側: 揺れが床より上に出るのはどれか", fontsize=10)
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("真値 ÷ そのチャネルの床")

    for row in axes:
        for ax in row:
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7)
    fig.suptitle("加速度センサが 2 つ要る理由: 低速側では衝撃が見えず、高速側では揺れが見えない",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

    # 数字も返す（報告用）
    ring = events.ring_freq_hz
    hf_pk, _ = band_level(f_hf, a_hf, ring, 200.0)
    hf_floor = floor_level(f_hf, a_hf, 6000.0, 9000.0)
    peak_no = float(np.max(peaks_no)) if peaks_no else float("nan")
    peak_aa = float(np.max(peaks_aa)) if peaks_aa else float("nan")
    # 折り返し先: 3 kHz は 1 kHz 標本化でちょうど 0 Hz に落ちるが、減衰振動は
    # zeta*f だけ幅があるので、0 〜 zeta*f 付近の帯に化ける
    lo, hi = 5.0, max(20.0, 2.0 * events.ring_damping * ring)
    fake_no = float(np.max(a_n[(f_n >= lo) & (f_n <= hi)]))
    fake_aa = float(np.max(a_l[(f_l >= lo) & (f_l <= hi)]))
    return {"hf_peak": hf_pk, "hf_floor": hf_floor,
            "lf_peak_noaa": peak_no, "lf_peak_aa": peak_aa,
            "lf_peaks_noaa": peaks_no,
            "fake_band": (lo, hi), "fake_no": fake_no, "fake_aa": fake_aa}


# ---------------------------------------------------------------------------

def _use_japanese_font():
    for name in ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic",
                 "VL Gothic", "Noto Sans JP"]:
        try:
            from matplotlib import font_manager
            font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        matplotlib.rcParams["font.family"] = name
        return name
    return None


def run_core(tmpdir: str, name: str, duration: float, extra_args=()):
    """core/filler を回して RFILL004 ダンプを作る。出力先は一時ディレクトリ。

    イベント列は ``<out>.events``（RFEVT002）へ自動で出る。物理コアが無い・
    失敗したときは None を返す（黙って代用データを作らない）。
    """
    exe = os.path.join(ROOT, "core", "filler")
    if not os.path.exists(exe):
        return None
    out = os.path.join(tmpdir, name)
    cmd = [exe, "--params", os.path.join(ROOT, "params.json"),
           "--duration", str(duration), "--out", out, "--quiet", *extra_args]
    print(f"  core/filler {' '.join(extra_args)} を {duration:.0f} s ぶん回す")
    try:
        r = subprocess.run(cmd, cwd=os.path.join(ROOT, "core"),
                           capture_output=True, timeout=900)
    except Exception as exc:
        print(f"  物理コアを回せなかった: {exc}")
        return None
    if r.returncode != 0 or not os.path.exists(out):
        print(f"  物理コアが失敗した (rc={r.returncode}): "
              f"{r.stderr.decode('utf-8', 'replace').strip()[:300]}")
        return None
    return out


def report_slosh_pickup(chans, truth, f_slosh, label=""):
    """スロッシング成分が、実測の床と量子化の床からどれだけ上にあるか。

    センサ出力のピークだけ見ると、ノイズの山を拾って「見えている」と誤読するので、
    真値のピーク（＝信号そのもの）も並べて出す。判定は真値ピーク / 実測の床。
    """
    print(f"\nスロッシング（{f_slosh:.2f} Hz）の見え方 {label}")
    print("  {:<20} {:>10} {:>10} {:>10} {:>10} {:>9} {:>10} {}".format(
        "チャネル", "真値ピーク", "出力ピーク", "実測の床", "量子化の床",
        "S/N[dB]", "量子化余裕", "判定"))
    rows = {}
    for name in ("accel_lf_radial", "accel_lf_tangential", "accel_hf_radial", "strain"):
        ch = chans[name]
        f, a = amplitude_density(ch.y, ch.fs_hz)
        ft, at = amplitude_density(ch.truth, truth.fs_hz)
        pk, _ = band_level(f, a, f_slosh, 0.3)
        pkt, _ = band_level(ft, at, f_slosh, 0.3)
        fl = floor_level(f, a, 8.0, 30.0)
        qf = quantization_floor(ch)
        snr = db(pkt / fl)
        verdict = "見える" if snr > 6.0 else ("際どい" if snr > 0.0 else "床に埋もれる")
        rows[name] = {"truth_peak": pkt, "peak": pk, "floor": fl, "qfloor": qf,
                      "snr_db": snr, "q_db": db(pkt / qf), "verdict": verdict}
        print("  {:<20} {:>10.3g} {:>10.3g} {:>10.3g} {:>10.3g} {:>9.1f} {:>9.1f}dB  {}"
              .format(name, pkt, pk, fl, qf, snr, db(pkt / qf), verdict))
    print("  ※ 床は 8〜30 Hz の中央値。S/N は真値ピーク / 実測の床。"
          "量子化余裕は真値ピーク / 量子化だけの床")
    return rows


def report_mount_sensitivity(dump, params, f_slosh, seed=SEED):
    """架台の伝達をどう置くかで、加速度センサの結論がどれだけ動くかを見る。

    「揺れが加速度センサで見えるか」は取り付け部のモデルにかなり効く。
    ここを断らずに結論だけ書くと誇張になるので、3 通り並べておく。
    """
    cases = [
        ("grounded 250 Hz", "grounded", 250.0, "既定。柱の局所モードを想定"),
        ("grounded  78 Hz", "grounded", 78.0, "架台全体の揺れモード（柔らかめ）"),
        ("free  (剛体)", "free", 250.0, "接地を無視して a = F/m とした場合"),
    ]
    print("\n架台の伝達モデルによる違い（accel_lf 接線、スロッシング成分）")
    print("  {:<18} {:>11} {:>11} {:>9} {}".format(
        "モデル", "真値ピーク", "実測の床", "S/N[dB]", "意味"))
    for label, model, fn, note in cases:
        p = copy.deepcopy(params)
        p["sensors"]["accel_lf"]["structure_model"] = model
        p["sensors"]["accel_lf"]["structure_freq_hz"] = fn
        ch, tr = virtual.synthesize(dump, p, seed=seed,
                                    channels=("accel_lf_tangential",))
        c = ch["accel_lf_tangential"]
        f, a = amplitude_density(c.y, c.fs_hz)
        ft, at = amplitude_density(c.truth, tr.fs_hz)
        pkt, _ = band_level(ft, at, f_slosh, 0.3)
        fl = floor_level(f, a, 8.0, 30.0)
        print("  {:<18} {:>11.3g} {:>11.3g} {:>9.1f}  {}".format(
            label, pkt, fl, db(pkt / fl), note))


def main() -> int:
    os.makedirs(FIGS, exist_ok=True)
    if _use_japanese_font() is None:
        print("注意: 日本語フォントが見つからないので図のラベルは化ける")

    params, added = virtual.load_params(os.path.join(ROOT, "params.json"))
    print("params.json に無くて既定値で埋めた項目（PARAMS_ADDED.md 参照）:")
    for k in added:
        print(f"  {k}")

    R = float(params["bottle"]["inner_diameter_mm"]) * 1e-3 / 2.0
    h_full = (float(params["fill"]["target_volume_mL"]) * 1e-6) / (np.pi * R ** 2)
    f_slosh = virtual.slosh_frequency(h_full, R, float(params["sim"]["gravity_m_s2"]))
    print(f"\n満量 {params['fill']['target_volume_mL']} mL → 液深 {h_full*1e3:.1f} mm,"
          f" スロッシング {f_slosh:.3f} Hz")

    # 段ごとの効き方と折り返しは、素性の分かった試験信号だけで描ける（コア不要）
    fig_stages(params, os.path.join(FIGS, "sensor_stages.png"))
    lvl_no, lvl_aa = fig_alias(os.path.join(FIGS, "sensor_alias.png"))
    print(f"折り返し: 帯域制限なし {lvl_no:.3g} → あり {lvl_aa:.3g}"
          f" ({db(lvl_aa/lvl_no):.1f} dB)")

    tmpdir = tempfile.mkdtemp(prefix="sensors_demo_")
    try:
        # --- 正常時 ------------------------------------------------------
        path = run_core(tmpdir, "normal.bin", DURATION_S, ("--no-faults",))
        if path is None:
            print("\n物理コア（core/filler）が使えないので、連続量に依存する図"
                  "（時系列・スペクトル・加速度 2 種）は省いた。"
                  "コア不要の stages と alias は描いてある")
            print(f"\n図: {FIGS}/sensor_*.png")
            return 0

        dump, _ = load_run(path)
        print(f"  物理コア RFILL004: {dump.summary()}")
        for w in dump.warnings:
            print(f"  注意: {w}")
        err = virtual.verify_reaction(dump, params)
        print(f"  コアの f_tab_* と、こちらで組み直した合力の差: 相対 {err:.3g}"
              "（層またぎ不変量。符号ズレなら 2 前後に跳ねる）")
        print(f"  水平合力 最大 {np.max(np.hypot(dump.f_tab_x, dump.f_tab_y)):.4g} N"
              f" / T_slosh 最大 {np.abs(dump.torque_slosh).max():.4g} Nm")

        chans, truth = virtual.synthesize(dump, params, seed=SEED)
        for note in truth.notes:
            print(f"  {note}")
        print("\n各チャネル:")
        for name, ch in chans.items():
            info = ch.info
            print(f"  {name}: {ch.fs_hz:.0f} Hz {info['bits']} bit,"
                  f" 間引き {info['resample']['mode']} 比 {info['resample']['ratio']:.3g},"
                  f" LSB {info['lsb']:.4g} {ch.unit}, rms {ch.rms():.4g},"
                  f" 飽和 {info['saturated_samples']} 点")
            for w in info.get("bandlimit_skipped", []):
                print(f"      {w}")

        # ノイズ抜き（量子化の床を見せるため）
        p_quiet = copy.deepcopy(params)
        p_quiet["sensors"]["accel_lf"]["noise_density_ug_rthz"] = 0.0
        p_quiet["sensors"]["accel_hf"]["noise_density_ug_rthz"] = 0.0
        p_quiet["sensors"]["strain"]["noise_rms_ustrain"] = 0.0
        p_quiet["sensors"]["current"]["noise_rms_A"] = 0.0
        chans_quiet, _ = virtual.synthesize(dump, p_quiet, seed=SEED)

        fig_timeseries(truth, chans, os.path.join(FIGS, "sensor_timeseries.png"),
                       "物理コア RFILL004: 真値とセンサ出力（末尾数秒）")
        fig_spectrum(truth, chans, chans_quiet,
                     os.path.join(FIGS, "sensor_spectrum.png"), f_slosh)
        report_slosh_pickup(chans, truth, f_slosh, "（正常時）")
        report_mount_sensitivity(dump, params, f_slosh)

        # 再現性
        again, _ = virtual.synthesize(dump, params, seed=SEED)
        other, _ = virtual.synthesize(dump, params, seed=SEED + 1)
        print(f"\n再現性: 同じ種で一致"
              f" {np.array_equal(again['strain'].y, chans['strain'].y)} /"
              f" 違う種で相違 {not np.array_equal(other['strain'].y, chans['strain'].y)}")

        # --- 故障（軸受外輪傷＋カムフォロワ当たり）------------------------
        bpath = run_core(tmpdir, "fault.bin", DURATION_S,
                         ("--fault-bearing", "--fault-cam"))
        if bpath is None:
            print("\n故障の回を作れないので、衝撃の図は省いた")
        else:
            bd, bev = load_run(bpath)
            for w in bd.warnings:
                print(f"  注意: {w}")
            # 衝撃はイベント列（RFEVT002）から取り出す。軸受=kind0 / カム当たり=kind1
            ev = virtual.bearing_events(bev, params)
            cam = virtual.cam_events(bev, params)
            print(f"\n軸受の衝撃: {len(ev)} 発 / {ev.ring_freq_hz:.0f} Hz"
                  f" zeta={ev.ring_damping} / {ev.source}")
            print(f"カムフォロワ当たり: {len(cam)} 件 / {cam.source}"
                  "（トルクの衝撃なので加速度リンギングは合成しない。検出は発生時刻で）")
            if len(ev):
                gap = np.diff(ev.times)
                if gap.size:
                    print(f"  発生間隔 [s]: 中央 {float(np.median(gap)):.4f}"
                          "（軸受はモータ軸に置いたので時間軸でほぼ等間隔）")
                bchans, btruth = virtual.synthesize(bd, params, seed=SEED, impacts=ev)
                for name in ("accel_hf_radial", "accel_lf_radial"):
                    ch = bchans[name]
                    print(f"  {name}: rms {ch.rms():.4g} {ch.unit},"
                          f" ピーク {np.abs(ch.y).max():.4g}, 飽和 {ch.info['saturated_samples']} 点")
                res = fig_two_accels(bd, params, ev, bchans,
                                     os.path.join(FIGS, "sensor_two_accels.png"), f_slosh)
                print(f"  accel_hf: {ev.ring_freq_hz:.0f} Hz のピーク"
                      f" {res['hf_peak']:.3g} m/s^2/rtHz、6〜9 kHz の床"
                      f" {res['hf_floor']:.3g} → {db(res['hf_peak']/res['hf_floor']):.1f} dB 上")
                nrms = bchans["accel_lf_radial"].info["noise_rms"]
                pk = np.array(res["lf_peaks_noaa"])
                print(f"  accel_lf（1 kHz 標本化）で衝撃 1 発ぶんの窓のピーク:"
                      f" 帯域制限なし {res['lf_peak_noaa']:.3g} →"
                      f" 規定どおり {res['lf_peak_aa']:.3g} m/s^2"
                      f"（{db(res['lf_peak_aa']/res['lf_peak_noaa']):.1f} dB）")
                print(f"  ノイズ rms {nrms:.3g} に対して、帯域制限なしなら"
                      f" {res['lf_peak_noaa']/nrms:.0f} 倍の偽イベントに見える"
                      f"（規定どおりなら {res['lf_peak_aa']/nrms:.1f} 倍。"
                      "これは折り返しではなく、帯域制限そのものが衝撃で鳴った残り）")
                if pk.size:
                    print(f"  3000 Hz は 1000 Hz 標本化でちょうど 0 Hz に落ちるので、"
                          f"見かけの大きさは標本化の位相しだい。先頭 {pk.size} 発のピークは"
                          f" {np.array2string(np.round(pk, 3))} とばらつく")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n図: {FIGS}/sensor_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
