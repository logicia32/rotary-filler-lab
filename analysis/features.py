"""特徴量。回転角リサンプル・次数比分析・包絡線解析・帯域パワー・衝撃検出。

センサ層（`sensors/`）が作った信号から、異常監視で見る量を取り出す。
判定は `analysis/detect.py`、総当たりは `analysis/run_matrix.py`。

停止区間の扱い（この解析でいちばん効く取り決め）
------------------------------------------------
このテーブルは割出しと停止を繰り返す。停止している 0.9 s のあいだ、
テーブル角は進まない。次数比分析は「回転角で等間隔に並べ直してから FFT する」
手法なので、角度が進まない区間をどう扱うかで結果が変わる。

ここでは角度領域には停止区間を入れない。理由は単純で、角度が進まない以上、
その区間に対応する角度が無いから。実装としては、テーブル角の系列から
「角度が同じまま並んでいる区間（プラトー）」を 1 点に潰し、狭義単調増加にしてから
等角度の格子へ補間する。潰すときはプラトーの最後の点を残す
（次の割出しが引き継ぐ状態がそこにあるため。`config.ANALYSIS["angle_resample"]["dwell_rule"]`）。

この取り決めの代償と、それでも成り立つこと:

* 停止中に続いている液の自由振動は角度領域から落ちる。落として構わない。
  むしろ回転に同期しない成分を持ち込まないので、次数比側の見通しが良くなる。
* 割出しの終わりと次の割出しの始まりが直接つながるので、継ぎ目に段差が残る。
  段差は 45 度ごとに 1 回、つまり 1 回転あたり 8 回きっちり出るので、
  漏れ込む先は次数 8 の倍数に限られる。狙っている次数 1（回転同期のアンバランス）
  とは重ならない。
* 窓は必ず整数回転に取る。そうすると次数の刻みが 1/回転数になり、
  整数次が DFT のビン中心に乗る。矩形窓のままで整数次どうしが漏れずに分離できる。

回転しない成分（スロッシング 3.75 Hz）は、角度領域では瞬時の回転速度で割った
次数に写るので、速度が変わるあいだ次数が動いて広がる。時間領域のスペクトルでは
一本に立つ。同じ信号を時間軸と角度軸の両方で見れば、両者を分けられる。

依存は numpy と scipy。センサ層と違って、ここは後処理なので零位相の
`sosfiltfilt` を使う（実機のフィルタと違い、あとから群遅延を戻してよい）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# 回転角リサンプル
# ---------------------------------------------------------------------------

def monotone_angle(theta) -> np.ndarray:
    """テーブル角を非減少に均す。

    ログは f32 なので、角度が 60 rad を超えるあたりでは 1 LSB が 3.8e-6 rad ある。
    停止中の値はその刻みで前後に 1 LSB 揺れて、差分が負になることがある。
    物理としてはテーブルは戻らないので、累積最大で均す。
    """
    return np.maximum.accumulate(np.asarray(theta, dtype=float))


def plateau_collapse(theta, rule: str = "last") -> np.ndarray:
    """停止区間（角度が変わらない並び）を 1 点に潰した index を返す。

    rule="last"  プラトーの最後の点を残す（既定）
    rule="first" プラトーの最初の点を残す

    返り値の index で拾った角度は狭義単調増加になる。
    """
    th = monotone_angle(theta)
    if len(th) < 2:
        return np.arange(len(th))
    grows = th[1:] > th[:-1]
    keep = np.zeros(len(th), dtype=bool)
    if rule == "last":
        keep[:-1] = grows          # 次で角度が増える点 = プラトーの最後
        keep[-1] = True
    elif rule == "first":
        keep[1:] = grows           # 前から増えて来た点 = プラトーの最初
        keep[0] = True
    else:
        raise ValueError(f"rule は last か first: {rule!r}")
    idx = np.flatnonzero(keep)
    # 末尾がプラトーのままだと最後の 1 点が重複するので落とす
    while len(idx) >= 2 and th[idx[-1]] <= th[idx[-2]]:
        idx = idx[:-1]
    return idx


def order_antialias(y, fs: float, samples_per_rev: int, peak_rev_per_s: float,
                    margin: float = 0.8, order: int = 4):
    """角度リサンプルの前に掛ける低域通過。

    角度領域のナイキストは `samples_per_rev / 2` 次。回転がいちばん速い瞬間の
    速度 `peak_rev_per_s` [回転/s] を使うと、それは時間軸で

        f_max = (samples_per_rev / 2) * peak_rev_per_s   [Hz]

    にあたる。ここより上の成分は角度領域へ折り返す。この段を省くと、
    センサの帯域いっぱいに広がっている雑音がまるごと折り返してきて、
    次数のビンの雑音床が桁で持ち上がる（既定の諸元では 12 dB ぶん損をする）。

    リサンプル自体は補間なので、間引きのように雑音を平均してくれない。
    平均が効くのはこの低域通過を通したときだけ。

    返り値は (フィルタ後の信号, カットオフ [Hz])。落とすものが無ければそのまま返す。
    """
    y = np.asarray(y, dtype=float)
    fc = margin * 0.5 * float(samples_per_rev) * float(peak_rev_per_s)
    if not (fc > 0) or fc >= 0.5 * fs:
        return y, float("nan")
    sos = butter(order, fc / (0.5 * fs), btype="lowpass", output="sos")
    return sosfiltfilt(sos, y), fc


@dataclass
class AngleSignal:
    """等角度に並べ直した信号。"""

    theta: np.ndarray            # [rad] 等間隔
    y: np.ndarray
    samples_per_rev: int
    n_rev: int
    info: dict = field(default_factory=dict)


def angle_at(t_query, t_log, theta_log) -> np.ndarray:
    """別の刻みの時刻列に対するテーブル角を補間で出す。

    センサのサンプリング（1 kHz など）はログ（4 kHz）と違うので、
    角度リサンプルの前にここで揃える。停止中は角度が一定なので補間して問題ない。
    """
    return np.interp(np.asarray(t_query, dtype=float),
                     np.asarray(t_log, dtype=float), monotone_angle(theta_log))


def angle_resample(theta, y, samples_per_rev: int, n_rev: int,
                   theta0: float | None = None, rule: str = "last") -> AngleSignal:
    """回転角で等間隔にリサンプルする。

    theta と y は同じ時刻列の上にあること（`angle_at` で揃えてから渡す）。
    窓は必ず整数回転（n_rev 回転ちょうど）にする。
    """
    th = monotone_angle(theta)
    y = np.asarray(y, dtype=float)
    if len(th) != len(y):
        raise ValueError(f"theta と y の長さが違う: {len(th)} と {len(y)}")
    idx = plateau_collapse(th, rule=rule)
    thk, yk = th[idx], y[idx]
    if theta0 is None:
        theta0 = float(thk[0])
    n = int(samples_per_rev) * int(n_rev)
    dtheta = TWO_PI / float(samples_per_rev)
    grid = theta0 + np.arange(n) * dtheta
    if grid[-1] > thk[-1] + 1e-12:
        raise ValueError(
            f"{n_rev} 回転ぶんの角度が足りない"
            f"（要求 {grid[-1] - theta0 + dtheta:.3f} rad、"
            f"ある {thk[-1] - theta0:.3f} rad）")
    yg = np.interp(grid, thk, yk)
    info = {
        "dropped_fraction": 1.0 - len(idx) / len(th),   # 停止区間が占めていた割合
        "dtheta_rad": dtheta,
        "theta0": theta0,
        "rule": rule,
        "max_dtheta_in_source_rad": float(np.max(np.diff(thk))) if len(thk) > 1 else 0.0,
    }
    # 元の角度の刻みが格子の刻みより粗いと、補間ではなく引き伸ばしになる
    info["oversampled"] = info["max_dtheta_in_source_rad"] < dtheta
    return AngleSignal(theta=grid, y=yg, samples_per_rev=int(samples_per_rev),
                       n_rev=int(n_rev), info=info)


# ---------------------------------------------------------------------------
# スペクトル
# ---------------------------------------------------------------------------

def _window(n: int, kind: str) -> np.ndarray:
    if kind in ("rect", "none", None):
        return np.ones(n)
    if kind == "hann":
        return np.hanning(n + 1)[:n]      # 周期的 Hann（FFT 用）
    raise ValueError(f"窓は rect か hann: {kind!r}")


def amplitude_spectrum(y, fs: float, window: str = "hann", detrend: bool = True):
    """片側の振幅スペクトルを返す (freq, amp)。

    ビン中心にちょうど乗った振幅 A の正弦波が A になるよう、
    コヒーレントゲイン（窓の平均）で割ってある。
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if detrend:
        y = y - y.mean()
    w = _window(n, window)
    cg = w.mean()
    spec = np.fft.rfft(y * w)
    amp = np.abs(spec) / (n * cg)
    amp[1:] *= 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, amp


def order_spectrum(sig: AngleSignal, window: str = "rect", detrend: bool = True):
    """次数比スペクトルを返す (orders, amp)。

    窓が整数回転なので、次数の刻みは 1/n_rev。既定を矩形窓にしてあるのは、
    整数次のあいだで漏れが出ないため（整数回転ちょうどの窓では直交する）。
    非整数次（軸受の欠陥次数など）を見るときは hann にする。
    """
    y = sig.y
    n = len(y)
    if detrend:
        y = y - y.mean()
    w = _window(n, window)
    cg = w.mean()
    spec = np.fft.rfft(y * w)
    amp = np.abs(spec) / (n * cg)
    amp[1:] *= 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0
    orders = np.arange(len(amp)) / float(sig.n_rev)
    return orders, amp


def order_amplitude(sig: AngleSignal, order: float, window: str = "rect") -> float:
    """指定した次数の振幅。整数次なら 1 ビンをそのまま読む。"""
    orders, amp = order_spectrum(sig, window=window)
    k = int(round(order * sig.n_rev))
    if k < 0 or k >= len(amp):
        return float("nan")
    return float(amp[k])


def band_peak(freq, amp, lo: float, hi: float):
    """帯域 [lo, hi] の最大値とその位置を返す (peak_freq, peak_amp)。"""
    sel = (freq >= lo) & (freq <= hi)
    if not np.any(sel):
        return float("nan"), float("nan")
    k = int(np.argmax(amp[sel]))
    return float(freq[sel][k]), float(amp[sel][k])


def line_amplitude(y, fs: float, lo: float, hi: float, window: str = "hann"):
    """帯域 [lo, hi] に立つ線の振幅を返す (peak_freq, peak_amp)。

    スロッシングのように、狭い帯域に一本立つ成分を探すときはこちら。
    帯域を積分する :func:`band_rms` だと、帯域の幅ぶんの雑音を一緒に集めてしまう
    （既定の諸元では 0.75 Hz 幅で 14 ビンぶん）。線の振幅は分解能に依らず一定なのに、
    ビンあたりの雑音は窓を長くするほど下がるので、線を探すなら
    窓いっぱいの分解能で振幅スペクトルを取って、その最大値を読むほうが素直。
    """
    freq, amp = amplitude_spectrum(y, fs, window=window)
    return band_peak(freq, amp, lo, hi)


def band_rms(y, fs: float, lo: float, hi: float, resolution_hz: float = 0.125):
    """帯域 [lo, hi] の実効値と、その帯域のピーク周波数を返す。

    Welch の平均ピリオドグラムで PSD を出し、帯域を積分する。
    振幅スペクトルの和ではなく PSD の積分にしてあるのは、
    線スペクトルでも雑音でも同じ意味（その帯域が持つ実効値）になるため。
    """
    y = np.asarray(y, dtype=float)
    nperseg = int(2 ** np.ceil(np.log2(max(fs / resolution_hz, 8))))
    nperseg = min(nperseg, len(y))
    freq, psd = welch(y - y.mean(), fs=fs, nperseg=nperseg,
                      noverlap=nperseg // 2, window="hann", detrend=False)
    sel = (freq >= lo) & (freq <= hi)
    if not np.any(sel):
        return float("nan"), float("nan"), (freq, psd)
    power = float(np.trapezoid(psd[sel], freq[sel]))
    k = int(np.argmax(psd[sel]))
    return float(np.sqrt(max(power, 0.0))), float(freq[sel][k]), (freq, psd)


# ---------------------------------------------------------------------------
# 包絡線
# ---------------------------------------------------------------------------

@dataclass
class Envelope:
    """包絡線と、その刻み。"""

    t: np.ndarray
    y: np.ndarray
    fs_hz: float
    info: dict = field(default_factory=dict)


class BandOutOfRange(ValueError):
    """帯域通過の帯域が、そのセンサのナイキストに入らない。"""


def envelope(y, fs: float, band: tuple[float, float], lowpass_hz: float,
             order: int = 4, method: str = "abs",
             decimate_to_hz: float | None = None, t0: float = 0.0) -> Envelope:
    """帯域通過、絶対値（またはヒルベルト）、低域通過、間引きの順に通す。

    軸受の欠陥は、構造を叩いて高い周波数で鳴る。その鳴りの振幅の変化に
    欠陥の通過周期が乗るので、搬送波を落として包絡線だけを見る。

    method="abs"     整流して低域通過。素朴だが実機の包絡線検波と同じ形
    method="hilbert" 解析信号の絶対値。整流の高調波が出ない
    """
    y = np.asarray(y, dtype=float)
    lo, hi = float(band[0]), float(band[1])
    nyq = 0.5 * fs
    if hi >= nyq:
        raise BandOutOfRange(
            f"帯域通過 {lo:.0f}〜{hi:.0f} Hz が、このセンサのナイキスト {nyq:.0f} Hz を超える")
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    band_y = sosfiltfilt(sos, y)
    if method == "abs":
        rect = np.abs(band_y)
    elif method == "hilbert":
        rect = np.abs(hilbert(band_y))
    else:
        raise ValueError(f"method は abs か hilbert: {method!r}")
    if lowpass_hz >= nyq:
        raise BandOutOfRange(f"包絡線の低域通過 {lowpass_hz:.0f} Hz がナイキストを超える")
    sos_lp = butter(order, lowpass_hz / nyq, btype="lowpass", output="sos")
    env = sosfiltfilt(sos_lp, rect)

    fs_out = fs
    step = 1
    if decimate_to_hz is not None and decimate_to_hz < fs:
        step = int(np.floor(fs / decimate_to_hz))
        if step > 1:
            # 低域通過済みなので単純間引きでよい（残っているのは lowpass_hz まで）
            if lowpass_hz >= 0.5 * fs / step:
                step = max(int(np.floor(fs / (2.5 * lowpass_hz))), 1)
            env = env[::step]
            fs_out = fs / step
    t = t0 + np.arange(len(env)) / fs_out
    return Envelope(t=t, y=env, fs_hz=fs_out,
                    info={"band_hz": (lo, hi), "lowpass_hz": lowpass_hz,
                          "method": method, "decimate_step": step, "order": order})


def robust_sigma(y) -> float:
    """中央絶対偏差から出した標準偏差の推定。外れ値（衝撃そのもの）に引きずられない。"""
    y = np.asarray(y, dtype=float)
    med = np.median(y)
    return float(1.4826 * np.median(np.abs(y - med)))


def impact_times(env: Envelope, k_sigma: float = 5.0,
                 min_separation_s: float = 0.0) -> tuple[np.ndarray, float]:
    """包絡線のしきい値越えから衝撃の時刻を拾う。

    しきい値は中央値 + k_sigma × (中央絶対偏差から出した σ)。
    平均と標準偏差にすると衝撃自身が基準を持ち上げてしまう。

    返り値は (時刻, しきい値)。
    """
    y = env.y
    thr = float(np.median(y) + k_sigma * robust_sigma(y))
    above = y > thr
    onset = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if len(onset) == 0:
        return np.zeros(0), thr
    times = env.t[onset]
    if min_separation_s > 0:
        keep = [times[0]]
        for tt in times[1:]:
            if tt - keep[-1] >= min_separation_s:
                keep.append(tt)
        times = np.asarray(keep)
    return times, thr


def match_events(found, truth, tolerance_s: float):
    """検出した時刻と真の時刻を突き合わせる。

    返り値は (検出できた真イベント数, 真イベント数, 余分に出た数, 時刻差の実効値)。
    """
    found = np.asarray(found, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if len(truth) == 0:
        return 0, 0, len(found), float("nan")
    used = np.zeros(len(found), dtype=bool)
    hits, errs = 0, []
    for te in truth:
        if len(found) == 0:
            break
        d = np.abs(found - te)
        d[used] = np.inf
        k = int(np.argmin(d))
        if d[k] <= tolerance_s:
            used[k] = True
            hits += 1
            errs.append(found[k] - te)
    extra = int(np.count_nonzero(~used))
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    return hits, len(truth), extra, rms
