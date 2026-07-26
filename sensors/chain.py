"""センサ信号連鎖の部品。

SENSORS.md 2 節の順序をそのまま関数に割った。

    真値 → 取り付け位置の伝達 → 帯域制限 → サンプリング → ノイズ → 量子化 → レンジ飽和

各段を単体で試験できるように、状態を持たない関数として分けてある。
まとめて通したいときは :func:`run_chain` を使う。

方針
----
* フィルタは因果（片方向）で回す。実機のアナログ／ディジタルフィルタは位相を戻せない
  ので、`filtfilt` のような零位相化はしない。そのぶん群遅延が出るが、それも含めて
  「センサが返す信号」とみなす。
* 乱数は種を引数で受ける。同じ種・同じチャネル名なら、何度回しても同じ波形が出る。
* 量子化の LSB は `full_scale / 2**bits`。両振りのセンサでは full_scale に
  「レンジの 2 倍（±2 g なら 4 g）」を渡す。この換算は :mod:`sensors.virtual` 側で行う。

依存は numpy だけ。scipy があれば `lfilter` を借りて速くするが、無くても動く。
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

try:  # scipy があれば畳み込みを任せる（無くても結果は同じ）
    from scipy.signal import lfilter as _scipy_lfilter
except Exception:  # pragma: no cover - scipy が無い環境用
    _scipy_lfilter = None


# ---------------------------------------------------------------------------
# 2 次のディジタルフィルタ（双一次変換）
# ---------------------------------------------------------------------------

def biquad_lowpass(fc_hz: float, fs_hz: float, q: float = 1.0 / np.sqrt(2.0)):
    """2 次低域通過の係数 (b, a) を返す。

    既定の Q = 1/sqrt(2) がバターワース。双一次変換の周波数歪みは fc で補正して
    あるので、fc での振幅は -3.01 dB ちょうどになる。
    """
    _check_cutoff(fc_hz, fs_hz)
    w0 = 2.0 * np.pi * fc_hz / fs_hz
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b = np.array([(1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0])
    a = np.array([1.0 + alpha, -2.0 * cw, 1.0 - alpha])
    return b / a[0], a / a[0]


def biquad_highpass(fc_hz: float, fs_hz: float, q: float = 1.0 / np.sqrt(2.0)):
    """2 次高域通過の係数 (b, a)。伝達関数は s^2 / (s^2 + (w0/Q) s + w0^2)。

    接地された構造の「力 → 加速度」（アクセレランス）がこの形になるので、
    架台の伝達に使う。低域では加速度が f^2 で落ちる。
    """
    _check_cutoff(fc_hz, fs_hz)
    w0 = 2.0 * np.pi * fc_hz / fs_hz
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b = np.array([(1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0])
    a = np.array([1.0 + alpha, -2.0 * cw, 1.0 - alpha])
    return b / a[0], a / a[0]


def _check_cutoff(fc_hz: float, fs_hz: float) -> None:
    if not (fc_hz > 0.0):
        raise ValueError("カットオフは正の値であること")
    if fc_hz >= 0.5 * fs_hz:
        raise ValueError(
            f"カットオフ {fc_hz} Hz が入力側のナイキスト {0.5 * fs_hz} Hz 以上。"
            " 物理コアのログ周波数が足りない"
        )


def apply_biquad(b, a, x) -> np.ndarray:
    """差分方程式を 1 回だけ通す（因果）。初期値は 0。"""
    x = np.asarray(x, dtype=float)
    if _scipy_lfilter is not None:
        return np.asarray(_scipy_lfilter(b, a, x), dtype=float)
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for i, xi in enumerate(x):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return y


def butter2_lowpass(x, fs_hz: float, fc_hz: float) -> np.ndarray:
    """2 次バターワース低域通過（アンチエイリアス段）。"""
    b, a = biquad_lowpass(fc_hz, fs_hz)
    return apply_biquad(b, a, x)


def butter2_highpass(x, fs_hz: float, fc_hz: float) -> np.ndarray:
    """2 次バターワース高域通過（圧電センサの交流結合）。

    IEPE の加速度センサは電荷が抜けるので直流を返さない。params の
    `highpass_hz` がその折れ点。ここが効いて低周波の情報が落ちる。
    """
    b, a = biquad_highpass(fc_hz, fs_hz)
    return apply_biquad(b, a, x)


def second_order(x, fs_hz: float, fn_hz: float, zeta: float, form: str = "lowpass") -> np.ndarray:
    """構造の共振を 1 つだけ入れる 2 次系。

    form="lowpass"  : 直流利得 1。自由な剛体としての伝達（低域はそのまま通る）
    form="highpass" : 直流利得 0。接地した構造のアクセレランス（低域は f^2 で落ちる）

    どちらも共振点で 1/(2*zeta) 倍に持ち上がり、衝撃が fn で鳴る。
    """
    q = 1.0 / (2.0 * zeta)
    if form == "lowpass":
        b, a = biquad_lowpass(fn_hz, fs_hz, q=q)
    elif form == "highpass":
        b, a = biquad_highpass(fn_hz, fs_hz, q=q)
    else:
        raise ValueError(f"form は lowpass か highpass: {form!r}")
    return apply_biquad(b, a, x)


# ---------------------------------------------------------------------------
# サンプリング（間引き / 再標本化）
# ---------------------------------------------------------------------------

def resample(x, fs_in: float, fs_out: float, allow_upsample: bool = False):
    """物理コアのログ周波数からセンサのサンプリング周波数へ落とす。

    返り値は (y, info)。info は何をしたかの記録（dict）。

    比が整数のときは単純な間引き（先頭を残す）。
    整数比にならないときは、直前の帯域制限で既に帯域が絞られている前提で、
    出力側の時刻に線形補間する。帯域制限後の信号は標本間で滑らかなので、
    補間による誤差は帯域とサンプリング比の比で決まり、
    ここで扱う範囲（bandwidth << fs_in/2）では量子化段より十分小さい。
    厳密にやるなら多相分解 + sinc 補間だが、その差はこの用途では見えない。

    fs_out > fs_in は「無い情報を作る」ことになるので既定では拒否する。
    どうしても要る場合（accel_hf の 51.2 kHz に対してログが 4 kHz、など）は
    allow_upsample=True を明示すること。補間で埋めるだけで、
    fs_in/2 より上に本来あるはずの成分は出てこない。
    """
    x = np.asarray(x, dtype=float)
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("サンプリング周波数は正の値であること")
    if fs_out > fs_in and not allow_upsample:
        raise ValueError(
            f"センサのサンプリング {fs_out} Hz が入力 {fs_in} Hz より高い。"
            " 補間で埋めるだけで情報は増えないので、意図するなら allow_upsample=True"
        )

    ratio = fs_in / fs_out
    n_in = len(x)
    if abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1:
        step = int(round(ratio))
        y = x[::step].copy()
        info = {"mode": "decimate", "step": step, "ratio": ratio}
        return y, info

    n_out = int(np.floor((n_in - 1) * fs_out / fs_in)) + 1
    t_in = np.arange(n_in) / fs_in
    t_out = np.arange(n_out) / fs_out
    y = np.interp(t_out, t_in, x)
    info = {
        "mode": "interpolate" if fs_out < fs_in else "upsample",
        "step": None,
        "ratio": ratio,
    }
    return y, info


# ---------------------------------------------------------------------------
# ノイズ
# ---------------------------------------------------------------------------

def make_rng(seed: int, channel: str = "") -> np.random.Generator:
    """種とチャネル名から乱数生成器を作る。

    チャネル名は crc32 で整数に落とす。組み込みの hash() は実行ごとに変わるので
    使わない（再現性が消える）。
    """
    tag = zlib.crc32(channel.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(tag)]))


def noise_rms_from_density(density_ug_rthz: float, bandwidth_hz: float,
                           neb_factor: float = 1.11) -> float:
    """加速度センサのノイズ密度 [ug/rtHz] と帯域から rms [g] を出す。

    rms = density * sqrt(bandwidth * neb_factor)

    neb_factor は雑音等価帯域幅の係数。2 次バターワースでは 1.11
    （1 次の 1.57 に対して、切れが良いぶん小さい）。
    """
    if density_ug_rthz < 0 or bandwidth_hz <= 0:
        raise ValueError("ノイズ密度は 0 以上、帯域は正の値であること")
    return density_ug_rthz * 1e-6 * np.sqrt(bandwidth_hz * neb_factor)


def add_noise(x, rms: float, seed: int, channel: str = "") -> np.ndarray:
    """白色ガウス雑音を加える。rms は加える雑音の実効値（信号と同じ単位）。

    雑音はサンプリング後の系列に乗せる（SENSORS.md 2 節の順序）。したがって
    指定した rms は 0 〜 fs/2 に一様に広がる。帯域制限より上のナイキストまで
    平らになるぶん、帯域内の密度は真値よりわずかに低く出る。ここは近似。
    """
    x = np.asarray(x, dtype=float)
    if rms < 0:
        raise ValueError("rms は 0 以上であること")
    if rms == 0:
        return x.copy()
    rng = make_rng(seed, channel)
    return x + rng.normal(0.0, rms, size=x.shape)


# ---------------------------------------------------------------------------
# 量子化とレンジ飽和
# ---------------------------------------------------------------------------

def lsb_of(full_scale: float, bits: int) -> float:
    """量子化ステップ。full_scale は「取りうる幅の全体」。

    両振りのセンサ（±2 g など）では full_scale = 2 * range を渡す。
    """
    if bits < 1:
        raise ValueError("bits は 1 以上であること")
    if full_scale <= 0:
        raise ValueError("full_scale は正の値であること")
    return full_scale / float(2 ** bits)


def quantize(x, full_scale: float, bits: int) -> np.ndarray:
    """一様量子化（中央値丸め）。段差は full_scale / 2**bits。"""
    lsb = lsb_of(full_scale, bits)
    return np.round(np.asarray(x, dtype=float) / lsb) * lsb


def saturate(x, limit: float) -> np.ndarray:
    """レンジ飽和。±limit で頭打ちにする。"""
    if limit <= 0:
        raise ValueError("limit は正の値であること")
    return np.clip(np.asarray(x, dtype=float), -limit, limit)


# ---------------------------------------------------------------------------
# 連鎖まとめ
# ---------------------------------------------------------------------------

@dataclass
class ChainSpec:
    """1 チャネルぶんの連鎖の設定。

    range_amplitude : 片振り幅（±この値まで）。量子化の full_scale はこの 2 倍。
    bandwidth_hz    : アンチエイリアスのカットオフ。None なら帯域制限しない。
    highpass_hz     : 交流結合の折れ点。None なら DC 結合（そのまま通す）。
    noise_rms       : 加える雑音の実効値。信号と同じ単位。
    """

    sample_rate_hz: float
    range_amplitude: float
    bits: int
    noise_rms: float = 0.0
    bandwidth_hz: float | None = None
    highpass_hz: float | None = None
    allow_upsample: bool = False
    channel: str = ""

    @property
    def full_scale(self) -> float:
        return 2.0 * self.range_amplitude


def _front_end(x, fs: float, spec: ChainSpec, info: dict) -> np.ndarray:
    """交流結合 → 帯域制限。標本化より前の段をまとめたもの。"""
    y = np.asarray(x, dtype=float)
    if spec.highpass_hz is not None:
        y = butter2_highpass(y, fs, spec.highpass_hz)
    if spec.bandwidth_hz is not None:
        if spec.bandwidth_hz >= 0.5 * spec.sample_rate_hz:
            info["antialias_warning"] = (
                f"帯域 {spec.bandwidth_hz} Hz がセンサのナイキスト"
                f" {0.5 * spec.sample_rate_hz} Hz 以上。折り返しが残る"
            )
        if spec.bandwidth_hz >= 0.5 * fs:
            # 入口の信号がすでにそれより狭い（例: 10 kHz 帯域のセンサに 4 kHz のログ）。
            # 落とすものが無いので通さない。無理に掛けると係数が不正になる。
            info.setdefault("bandlimit_skipped", []).append(
                f"帯域 {spec.bandwidth_hz:.0f} Hz は入力 {fs:.0f} Hz のナイキストより上なので"
                " この段では何も落ちない")
        else:
            y = butter2_lowpass(y, fs, spec.bandwidth_hz)
    return y


def run_chain(x, fs_in: float, spec: ChainSpec, seed: int,
              extra=None, extra_fs: float | None = None) -> tuple[np.ndarray, dict]:
    """真値（取り付け伝達まで済んだもの）を通して、センサ出力を返す。

    extra は「別の刻みで合成した成分」（衝撃の減衰振動など）。
    連続量とは別に、同じ交流結合・帯域制限・間引きを **その刻みで** 通してから足す。
    こうしないと、ログのナイキストを超えるリンギングが入口で折り返してしまう。

    返り値は (y, info)。info には各段の記録が入る。
    """
    x = np.asarray(x, dtype=float)
    info: dict = {"channel": spec.channel, "fs_in": fs_in, "seed": seed}

    # 1. 交流結合と帯域制限（アンチエイリアス）
    y = _front_end(x, fs_in, spec, info)
    info["bandwidth_hz"] = spec.bandwidth_hz
    info["highpass_hz"] = spec.highpass_hz

    # 2. サンプリング
    y, rinfo = resample(y, fs_in, spec.sample_rate_hz, allow_upsample=spec.allow_upsample)
    info["resample"] = rinfo
    info["fs_out"] = spec.sample_rate_hz

    # 2b. 別刻みで合成した成分を、同じ前段を通してから足す
    if extra is not None:
        if extra_fs is None:
            raise ValueError("extra を渡すなら extra_fs も要る")
        e = _front_end(extra, extra_fs, spec, info)
        e, einfo = resample(e, extra_fs, spec.sample_rate_hz,
                            allow_upsample=spec.allow_upsample)
        if len(e) < len(y):
            e = np.pad(e, (0, len(y) - len(e)))
        y = y + e[:len(y)]
        info["extra"] = {"fs": extra_fs, "resample": einfo}

    # 3. ノイズ
    y = add_noise(y, spec.noise_rms, seed, spec.channel)
    info["noise_rms"] = spec.noise_rms

    # 4. 量子化
    y = quantize(y, spec.full_scale, spec.bits)
    info["lsb"] = lsb_of(spec.full_scale, spec.bits)
    info["bits"] = spec.bits

    # 5. レンジ飽和
    n_sat = int(np.count_nonzero(np.abs(y) > spec.range_amplitude))
    y = saturate(y, spec.range_amplitude)
    info["range_amplitude"] = spec.range_amplitude
    info["saturated_samples"] = n_sat

    return y, info


def timebase(n: int, fs_hz: float, t0: float = 0.0) -> np.ndarray:
    """サンプル数と周波数から時刻列を作る。"""
    return t0 + np.arange(n) / fs_hz
