"""特徴量の検証。答えが分かっている合成信号だけで確かめる。

外部データ（物理コアのダンプ）には一切依存しない。ここが通らないうちは、
`run_matrix.py` が出す数字を信じてはいけない。

    .venv/bin/python -m pytest -q analysis/test_features.py
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis import detect, features

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# スペクトルの振幅が合うか
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", ["rect", "hann"])
def test_amplitude_spectrum_recovers_known_sine(window):
    """ビン中心に乗せた正弦波は、窓に関係なく振幅そのものが返ること。"""
    fs, n = 1000.0, 4000            # 4 s -> 分解能 0.25 Hz
    f0, amp = 25.0, 0.7             # 25 Hz はビン中心（100 番目）
    t = np.arange(n) / fs
    y = amp * np.sin(TWO_PI * f0 * t + 0.3) + 2.0   # 直流も混ぜる
    freq, spec = features.amplitude_spectrum(y, fs, window=window)
    k = int(np.argmax(spec))
    assert freq[k] == pytest.approx(f0)
    assert spec[k] == pytest.approx(amp, rel=1e-6)


def test_amplitude_spectrum_two_tones():
    fs, n = 2000.0, 8000
    t = np.arange(n) / fs
    y = 1.0 * np.sin(TWO_PI * 50 * t) + 0.25 * np.sin(TWO_PI * 137.5 * t)
    freq, spec = features.amplitude_spectrum(y, fs, window="rect")
    assert spec[np.argmin(np.abs(freq - 50.0))] == pytest.approx(1.0, rel=1e-9)
    assert spec[np.argmin(np.abs(freq - 137.5))] == pytest.approx(0.25, rel=1e-9)


def test_band_rms_of_sine_is_amplitude_over_root_two():
    fs, n = 2000.0, 40000           # 20 s
    t = np.arange(n) / fs
    amp = 0.4
    y = amp * np.sin(TWO_PI * 3.75 * t)
    rms, peak, _ = features.band_rms(y, fs, 3.4, 4.1, resolution_hz=0.125)
    assert rms == pytest.approx(amp / np.sqrt(2.0), rel=0.05)
    assert peak == pytest.approx(3.75, abs=0.2)


def test_band_rms_of_white_noise_scales_with_bandwidth():
    """白色雑音なら、帯域の実効値は全体の実効値 × sqrt(帯域 / ナイキスト)。"""
    fs, n = 2000.0, 400000
    rng = np.random.default_rng(4)
    y = rng.normal(0.0, 1.0, n)
    lo, hi = 100.0, 300.0
    rms, _, _ = features.band_rms(y, fs, lo, hi, resolution_hz=1.0)
    expect = 1.0 * np.sqrt((hi - lo) / (fs / 2.0))
    assert rms == pytest.approx(expect, rel=0.05)


# ---------------------------------------------------------------------------
# 角度と停止区間の扱い
# ---------------------------------------------------------------------------

def test_monotone_angle_fixes_one_lsb_dips():
    th = np.array([1.0, 2.0, 2.0 - 1e-7, 2.0, 3.0])
    out = features.monotone_angle(th)
    assert np.all(np.diff(out) >= 0)
    assert out[2] == 2.0


def test_plateau_collapse_keeps_last_point_of_each_stop():
    th = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    idx = features.plateau_collapse(th, rule="last")
    # 1.0 のプラトーからは index 3、2.0 からは index 5 が残る
    assert list(idx) == [0, 3, 5, 6]
    assert np.all(np.diff(th[idx]) > 0)


def test_plateau_collapse_first_rule():
    th = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    idx = features.plateau_collapse(th, rule="first")
    assert list(idx) == [0, 1, 4]


def _synthetic_index_profile(n_cycles: int, fs: float, index_time: float,
                             dwell: float, index_angle: float):
    """割出しと停止を繰り返すテーブル角を作る（台形速度、加速 1/3）。"""
    dt = 1.0 / fs
    t, th = [], []
    now, ang = 0.0, 0.0
    for _ in range(n_cycles):
        m = int(round(index_time * fs))
        tau = np.arange(m) * dt / index_time          # 0..1
        # 台形速度（加速 1/3・定速 1/3・減速 1/3）を積分した角度の形
        a = 1.0 / 3.0
        s = np.empty_like(tau)
        v_peak = 1.0 / (1.0 - a)                      # 平均速度 1 になるピーク
        for i, x in enumerate(tau):
            if x < a:
                s[i] = 0.5 * v_peak / a * x ** 2
            elif x < 1.0 - a:
                s[i] = 0.5 * v_peak * a + v_peak * (x - a)
            else:
                d = 1.0 - x
                s[i] = 1.0 - 0.5 * v_peak / a * d ** 2
        t.extend(now + np.arange(m) * dt)
        th.extend(ang + index_angle * s)
        now += m * dt
        ang += index_angle
        k = int(round(dwell * fs))
        t.extend(now + np.arange(k) * dt)
        th.extend(np.full(k, ang))
        now += k * dt
    return np.asarray(t), np.asarray(th)


def test_angle_resample_recovers_orders_of_an_angle_locked_signal():
    """回転角の関数として作った信号は、角度リサンプル後に整数次へ立つこと。

    テーブルは停止を挟むので、時間軸のままではこの信号は一本にならない。
    """
    fs = 2000.0
    n_stations, index_deg = 8, 45.0
    n_cycles = 8 * 3                                  # 3 回転
    t, th = _synthetic_index_profile(n_cycles, fs, 1.5, 0.9, np.deg2rad(index_deg))
    y = 1.0 * np.cos(1.0 * th) + 0.5 * np.cos(3.0 * th + 0.7) + 0.25 * np.cos(8.0 * th)
    sig = features.angle_resample(th, y, 1024, 3, theta0=0.0)
    orders, amp = features.order_spectrum(sig, window="rect")
    assert amp[np.argmin(np.abs(orders - 1.0))] == pytest.approx(1.0, rel=0.02)
    assert amp[np.argmin(np.abs(orders - 3.0))] == pytest.approx(0.5, rel=0.02)
    assert amp[np.argmin(np.abs(orders - 8.0))] == pytest.approx(0.25, rel=0.02)
    # 停止区間ぶんが落ちていること（停止 0.9 / タクト 2.4 = 0.375）
    assert sig.info["dropped_fraction"] == pytest.approx(0.9 / 2.4, abs=0.02)


def test_angle_resample_rejects_too_short_window():
    fs = 1000.0
    t, th = _synthetic_index_profile(8, fs, 1.5, 0.9, np.deg2rad(45.0))
    with pytest.raises(ValueError):
        features.angle_resample(th, np.zeros_like(th), 256, 2, theta0=0.0)


def test_order_spectrum_bins_are_exact_integers_for_integer_revolutions():
    sig = features.AngleSignal(
        theta=np.arange(4096) * TWO_PI / 1024,
        y=np.cos(np.arange(4096) * TWO_PI / 1024 * 5.0),
        samples_per_rev=1024, n_rev=4)
    orders, amp = features.order_spectrum(sig, window="rect")
    assert orders[1] == pytest.approx(0.25)           # 4 回転なので刻みは 1/4 次
    assert features.order_amplitude(sig, 5.0) == pytest.approx(1.0, rel=1e-9)
    assert features.order_amplitude(sig, 4.0) < 1e-9


def test_fixed_frequency_tone_lands_far_from_order_one():
    """回転に同期しない成分（一定周波数の揺れ）が、角度領域のどこに写るか。

    「広がって消える」のではない。角度への写し方は割出しの速度で決まるので、
    いちばん速い瞬間の回転速度で決まる次数

        次数 = 揺れの周波数 / 最大回転速度 [回転/s]

    のあたりに山ができて、そこから上へ尾を引く。台形速度（加速 1/3）で
    45 度を 1.5 s なら最大 0.125 回転/s なので、3.751 Hz は次数 30 のあたり。
    狙っている次数 1 とは重ならない、というのがここで確かめたいこと。
    """
    fs = 2000.0
    index_time, index_deg, accel_frac = 1.5, 45.0, 1.0 / 3.0
    t, th = _synthetic_index_profile(8 * 3, fs, index_time, 0.9, np.deg2rad(index_deg))
    f0 = 3.751
    peak_rev_per_s = (index_deg / 360.0) / index_time / (1.0 - accel_frac)
    expect_order = f0 / peak_rev_per_s
    y = np.sin(TWO_PI * f0 * t)
    sig = features.angle_resample(th, y, 1024, 3, theta0=0.0)
    orders, amp = features.order_spectrum(sig, window="rect")
    peak_order = orders[int(np.argmax(amp))]
    assert expect_order == pytest.approx(30.0, abs=0.5)
    assert peak_order == pytest.approx(expect_order, rel=0.15)
    # 次数 1 のところには、ほとんど何も落ちてこない
    assert features.order_amplitude(sig, 1.0) < 0.05


# ---------------------------------------------------------------------------
# 包絡線と衝撃検出
# ---------------------------------------------------------------------------

def _ring_train(fs: float, duration: float, times, amp: float,
                ring_hz: float, damping: float) -> np.ndarray:
    """既知の間隔で並ぶ減衰振動列（SENSORS.md 4.2 と同じ式）。"""
    n = int(round(duration * fs))
    y = np.zeros(n)
    decay = TWO_PI * damping * ring_hz
    m = int(round(np.log(1000.0) / decay * fs))
    tt = np.arange(m) / fs
    shape = np.exp(-decay * tt) * np.sin(TWO_PI * ring_hz * tt)
    for te in times:
        i = int(round(te * fs))
        if 0 <= i < n:
            b = min(i + m, n)
            y[i:b] += amp * shape[: b - i]
    return y


def test_envelope_of_amplitude_modulated_carrier_shows_the_modulation():
    fs, n = 51200.0, 51200 * 4
    t = np.arange(n) / fs
    fm, fc = 7.0, 3000.0
    y = (1.0 + 0.8 * np.sin(TWO_PI * fm * t)) * np.sin(TWO_PI * fc * t)
    env = features.envelope(y, fs, (2100.0, 4200.0), 600.0,
                            decimate_to_hz=1600.0, method="abs")
    freq, spec = features.amplitude_spectrum(env.y, env.fs_hz, window="hann")
    k = int(np.argmax(spec))
    assert freq[k] == pytest.approx(fm, abs=0.5)


def test_envelope_hilbert_and_abs_find_the_same_modulation():
    fs, n = 51200.0, 51200 * 2
    t = np.arange(n) / fs
    y = (1.0 + 0.5 * np.sin(TWO_PI * 11.0 * t)) * np.sin(TWO_PI * 3000.0 * t)
    peaks = []
    for method in ("abs", "hilbert"):
        env = features.envelope(y, fs, (2100.0, 4200.0), 600.0, method=method,
                                decimate_to_hz=1600.0)
        freq, spec = features.amplitude_spectrum(env.y, env.fs_hz, window="hann")
        peaks.append(freq[int(np.argmax(spec))])
    assert peaks[0] == pytest.approx(peaks[1], abs=0.5)


def test_envelope_rejects_a_band_above_nyquist():
    fs = 1000.0
    y = np.zeros(4000)
    with pytest.raises(features.BandOutOfRange):
        features.envelope(y, fs, (2100.0, 4200.0), 600.0)


def test_impact_times_finds_a_known_ring_train():
    fs, duration = 51200.0, 4.0
    spacing = 0.37
    truth = np.arange(0.2, duration - 0.3, spacing)
    rng = np.random.default_rng(11)
    y = _ring_train(fs, duration, truth, amp=2.0, ring_hz=3000.0, damping=0.05)
    y = y + rng.normal(0.0, 0.05, len(y))
    env = features.envelope(y, fs, (2100.0, 4200.0), 600.0, decimate_to_hz=1600.0)
    found, thr = features.impact_times(env, k_sigma=5.0, min_separation_s=0.25 * spacing)
    hits, n_truth, extra, rms = features.match_events(found, truth, tolerance_s=0.005)
    assert n_truth == len(truth)
    assert hits == n_truth
    assert extra == 0
    assert rms < 2.0e-3          # 立ち上がりぶんの遅れだけ


def test_envelope_order_spectrum_of_an_angle_locked_ring_train():
    """回転角で等間隔に打つ衝撃は、包絡線を角度領域へ移すと欠陥次数に立つ。"""
    fs = 51200.0
    t_prof, th_prof = _synthetic_index_profile(8 * 4, 2000.0, 1.5, 0.9, np.deg2rad(45.0))
    duration = float(t_prof[-1])
    defect_order = 3.58
    step = TWO_PI / defect_order
    targets = np.arange(step, th_prof[-1], step)
    times = np.interp(targets, features.monotone_angle(th_prof), t_prof)
    y = _ring_train(fs, duration, times, amp=2.0, ring_hz=3000.0, damping=0.05)
    env = features.envelope(y, fs, (2100.0, 4200.0), 600.0, decimate_to_hz=1600.0)
    th_env = features.angle_at(env.t, t_prof, th_prof)
    sig = features.angle_resample(th_env, env.y, 4096, 3, theta0=float(th_prof[0]))
    orders, amp = features.order_spectrum(sig, window="hann")
    band = (orders > 0.5) & (orders < 12.5)
    floor = float(np.median(amp[band]))
    # 鋭い衝撃の列なので、基本波と高調波がだいたい同じ高さで並ぶ。
    # 「基本波が最大」ではなく「欠陥次数の倍数だけが立つ」ことを確かめる。
    for k in (1, 2, 3):
        _, a = features.band_peak(orders, amp, defect_order * k * 0.95, defect_order * k * 1.05)
        assert a > 4.0 * floor, f"{k} 倍の欠陥次数が立っていない"
    for q in (2.0, 5.0, 6.0, 9.0):     # 欠陥次数の倍数から外れたところ
        _, a = features.band_peak(orders, amp, q - 0.05, q + 0.05)
        assert a < 2.0 * floor


def test_robust_sigma_matches_std_for_gaussian_noise():
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 2.5, 200000)
    assert features.robust_sigma(y) == pytest.approx(2.5, rel=0.02)


def test_robust_sigma_ignores_outliers():
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 1.0, 20000)
    y[::200] = 50.0                      # 0.5 % を外れ値にする
    assert features.robust_sigma(y) == pytest.approx(1.0, rel=0.05)
    assert np.std(y) > 2.0               # 標準偏差のほうは引きずられる


def test_match_events_counts_extras_and_misses():
    truth = np.array([1.0, 2.0, 3.0])
    found = np.array([1.001, 2.9, 5.0])
    hits, n, extra, rms = features.match_events(found, truth, tolerance_s=0.01)
    assert (hits, n, extra) == (1, 3, 2)
    assert rms == pytest.approx(0.001, rel=0.05)


# ---------------------------------------------------------------------------
# 判定側
# ---------------------------------------------------------------------------

def test_baseline_and_threshold():
    rng = np.random.default_rng(7)
    vals = rng.normal(10.0, 0.5, 40)
    bl = detect.build_baseline(vals, name="x", unit="ustrain")
    assert bl.n == 40
    assert bl.mean == pytest.approx(10.0, abs=0.2)
    assert bl.threshold(6.0) == pytest.approx(bl.mean + 6.0 * bl.std)
    v = detect.judge(bl.mean + 10.0 * bl.std, bl, 6.0)
    assert v.detected and v.z == pytest.approx(10.0, rel=1e-6)
    v2 = detect.judge(bl.mean + 1.0 * bl.std, bl, 6.0)
    assert not v2.detected


def test_sigma_for_false_alarm_is_monotone_in_the_target():
    k_strict, p_strict = detect.sigma_for_false_alarm(22, 4500.0, 0.1)
    k_loose, p_loose = detect.sigma_for_false_alarm(22, 4500.0, 10.0)
    assert p_strict < p_loose
    assert k_strict > k_loose
    assert 3.0 < k_loose < k_strict < 12.0


def test_false_alarm_rate_round_trip():
    k, p = detect.sigma_for_false_alarm(22, 4500.0, 1.0)
    assert detect.false_alarm_rate(k, 22) == pytest.approx(p, rel=1e-6)


def test_snr_db_is_a_ratio_of_amplitudes():
    bl = detect.build_baseline(np.full(10, 2.0))
    assert bl.snr_db(20.0) == pytest.approx(20.0)
    assert bl.snr_db(2.0) == pytest.approx(0.0)
