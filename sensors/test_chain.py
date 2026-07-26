"""信号連鎖そのものの検証。外部データには一切依存しない。

    .venv/bin/python -m pytest -q sensors/test_chain.py

確かめること

* 帯域内は振幅がほぼ変わらず、カットオフの 2 倍では設計どおり落ちる
* 交流結合（圧電センサ）が直流を落とし、帯域内は通す
* 量子化ステップが range/2^bits と一致する
* ノイズの rms が指定値と一致する
* 同じ種なら同じ出力、違う種なら違う出力
* レンジを超えたら飽和する
* ナイキストを超える成分は、アンチエイリアスが無ければ折り返し、あれば落ちる
* 衝撃を別の刻みで合成して連鎖に足せる（低速センサでは折り返し、高速センサでは残る）
* 物理コアのバイナリ読み込みが、想定と違うものを黙って読まない（版 001 / 002）
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors import chain, virtual                      # noqa: E402
from sensors.read_dump import (                         # noqa: E402
    CanonicalDump, DumpFormatError, Events, load_run, read_dump)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "params.json")
FILLER = os.path.join(ROOT, "core", "filler")


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------

def steady_amplitude(y, fs, f0):
    """定常部（後ろ半分）で、周波数 f0 の成分の振幅を取り出す。"""
    y = np.asarray(y, dtype=float)
    half = y[len(y) // 2:]
    t = np.arange(len(half)) / fs
    c = np.mean(half * np.exp(-2j * np.pi * f0 * t))
    return 2.0 * abs(c)


def tone(f0, fs, seconds, amp=1.0):
    n = int(round(fs * seconds))
    t = np.arange(n) / fs
    return t, amp * np.sin(2 * np.pi * f0 * t)


def line_amplitude(y, fs, f0):
    """整数周期ぶんの矩形窓で、f0 の線スペクトルの振幅を読む。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    Y = np.fft.rfft(y - y.mean())
    f = np.fft.rfftfreq(n, 1.0 / fs)
    i = int(np.argmin(np.abs(f - f0)))
    return 2.0 * abs(Y[i]) / n


# ---------------------------------------------------------------------------
# 帯域制限
# ---------------------------------------------------------------------------

def test_passband_amplitude_is_preserved():
    """帯域内（カットオフの 1/10 以下）では振幅がほぼ変わらない。"""
    fs, fc = 20000.0, 500.0
    for f0 in (1.0, 3.75, 10.0, 50.0):
        _, x = tone(f0, fs, 4.0)
        y = chain.butter2_lowpass(x, fs, fc)
        g = steady_amplitude(y, fs, f0)
        assert abs(g - 1.0) < 5e-3, f"{f0} Hz で振幅が {g}"


def test_cutoff_is_minus_3db():
    """カットオフちょうどで 1/sqrt(2)。双一次変換の歪みは補正済み。"""
    fs, fc = 20000.0, 500.0
    _, x = tone(fc, fs, 4.0)
    y = chain.butter2_lowpass(x, fs, fc)
    assert abs(steady_amplitude(y, fs, fc) - 1.0 / math.sqrt(2.0)) < 5e-3


def test_two_times_cutoff_falls_as_designed():
    """2 次バターワースなので 2fc で 1/sqrt(1+2^4) = 0.2425（-12.3 dB）。"""
    fs, fc = 20000.0, 500.0
    expect = 1.0 / math.sqrt(1.0 + 2.0 ** 4)
    y = chain.butter2_lowpass(tone(2 * fc, fs, 4.0)[1], fs, fc)
    got = steady_amplitude(y, fs, 2 * fc)
    assert abs(got - expect) / expect < 0.02, f"2fc で {got}（期待 {expect}）"
    assert abs(20 * math.log10(got) + 12.30) < 0.3


def test_rolloff_is_second_order():
    """遮断域の傾きが -40 dB/dec（2 次）であること。"""
    fs, fc = 40000.0, 200.0
    g1 = steady_amplitude(chain.butter2_lowpass(tone(2000.0, fs, 2.0)[1], fs, fc),
                          fs, 2000.0)
    g2 = steady_amplitude(chain.butter2_lowpass(tone(4000.0, fs, 2.0)[1], fs, fc),
                          fs, 4000.0)
    slope_db = 20 * math.log10(g2 / g1)   # 1 オクターブぶん
    assert abs(slope_db - (-12.04)) < 0.5


def test_cutoff_above_nyquist_is_rejected():
    with pytest.raises(ValueError):
        chain.butter2_lowpass(np.zeros(100), 1000.0, 600.0)


@pytest.mark.parametrize("form,f0,expect", [("lowpass", 0.01, 1.0), ("highpass", 0.01, 0.0)])
def test_second_order_dc_gain(form, f0, expect):
    """構造の 2 次系: 直流利得は lowpass で 1、highpass（アクセレランス）で 0。"""
    fs, fn = 5000.0, 250.0
    _, x = tone(f0 * fn, fs, 40.0)
    y = chain.second_order(x, fs, fn, 0.03, form=form)
    g = steady_amplitude(y, fs, f0 * fn)
    assert abs(g - expect) < 2e-3


def test_highpass_blocks_dc_and_passes_band():
    """交流結合（IEPE）は直流を返さず、折れ点の 10 倍より上はそのまま通す。"""
    fs, fc = 51200.0, 2.0
    x = np.ones(int(fs * 4))
    y = chain.butter2_highpass(x, fs, fc)
    assert abs(float(np.mean(y[-1000:]))) < 1e-6          # 直流は残らない
    _, s = tone(20.0, fs, 4.0)
    assert abs(steady_amplitude(chain.butter2_highpass(s, fs, fc), fs, 20.0) - 1.0) < 0.02


def test_highpass_at_cutoff_is_minus_3db():
    fs, fc = 20000.0, 100.0
    y = chain.butter2_highpass(tone(fc, fs, 4.0)[1], fs, fc)
    assert abs(steady_amplitude(y, fs, fc) - 1.0 / math.sqrt(2.0)) < 5e-3


def test_second_order_peaks_at_resonance():
    """共振点で 1/(2*zeta) 倍に持ち上がる（衝撃が鳴る理由）。"""
    fs, fn, zeta = 20000.0, 250.0, 0.03
    y = chain.second_order(tone(fn, fs, 6.0)[1], fs, fn, zeta, form="lowpass")
    assert abs(steady_amplitude(y, fs, fn) - 1.0 / (2 * zeta)) / (1 / (2 * zeta)) < 0.02


@pytest.mark.skipif(pytest.importorskip("scipy", reason="scipy 無し") is None, reason="")
def test_coefficients_match_scipy():
    """自前の双一次変換が scipy.signal.butter と一致する（係数の裏取り）。"""
    from scipy.signal import butter
    fs, fc = 5000.0, 800.0
    b, a = chain.biquad_lowpass(fc, fs)
    bs, as_ = butter(2, fc / (fs / 2), btype="low")
    assert np.allclose(b, bs, atol=1e-12)
    assert np.allclose(a, as_, atol=1e-12)


# ---------------------------------------------------------------------------
# サンプリングと折り返し
# ---------------------------------------------------------------------------

def test_integer_ratio_is_plain_decimation():
    x = np.arange(100, dtype=float)
    y, info = chain.resample(x, 1000.0, 250.0)
    assert info["mode"] == "decimate" and info["step"] == 4
    assert np.array_equal(y, x[::4])


def test_non_integer_ratio_interpolates():
    """5000 Hz → 2000 Hz（比 2.5）。整数にならないので補間になる。"""
    fs_in, fs_out, f0 = 5000.0, 2000.0, 20.0
    _, x = tone(f0, fs_in, 1.0)
    y, info = chain.resample(x, fs_in, fs_out)
    assert info["mode"] == "interpolate"
    assert len(y) == int(np.floor((len(x) - 1) * fs_out / fs_in)) + 1
    # 帯域に対して十分速い標本化なので、補間しても振幅は保たれる
    assert abs(line_amplitude(y, fs_out, f0) - 1.0) < 5e-3


def test_upsampling_needs_explicit_permission():
    """センサのほうが速いとき（accel_hf 51.2 kHz > ログ 4 kHz）は既定で拒否。"""
    x = np.zeros(100)
    with pytest.raises(ValueError):
        chain.resample(x, 4000.0, 51200.0)
    y, info = chain.resample(x, 4000.0, 51200.0, allow_upsample=True)
    assert info["mode"] == "upsample" and len(y) > len(x)


def test_aliasing_with_and_without_antialias():
    """ナイキストを超える成分は、帯域制限が無ければ折り返して化ける。

    5000 Hz で作った 2200 Hz を 500 Hz に間引くと、出力側では 200 Hz に見える。
    2 次バターワース 200 Hz を通してから間引けば、その偽の 200 Hz は落ちる。
    """
    fs_in, fs_out, fc = 5000.0, 500.0, 200.0
    f_in_band, f_out_band = 40.0, 2200.0
    f_alias = abs(f_out_band - round(f_out_band / fs_out) * fs_out)
    assert f_alias == 200.0

    t = np.arange(int(fs_in * 1.0)) / fs_in     # 1 s ちょうど = 1 Hz 刻み
    x = np.sin(2 * np.pi * f_in_band * t) + np.sin(2 * np.pi * f_out_band * t)

    y_no, _ = chain.resample(x, fs_in, fs_out)
    y_aa, _ = chain.resample(chain.butter2_lowpass(x, fs_in, fc), fs_in, fs_out)

    a_no = line_amplitude(y_no, fs_out, f_alias)
    a_aa = line_amplitude(y_aa, fs_out, f_alias)
    # 帯域制限しないと、そこに無いはずの 200 Hz がほぼ元の振幅で立つ
    assert a_no > 0.9
    # 帯域制限すれば 2 桁以上落ちる
    assert a_aa < a_no / 100.0
    # 帯域内の 40 Hz は、どちらでも残っている
    assert abs(line_amplitude(y_no, fs_out, f_in_band) - 1.0) < 0.05
    assert abs(line_amplitude(y_aa, fs_out, f_in_band) - 1.0) < 0.05


# ---------------------------------------------------------------------------
# ノイズ
# ---------------------------------------------------------------------------

def test_noise_rms_matches_specification():
    n = 400000
    for rms in (0.01, 0.5, 3.0):
        y = chain.add_noise(np.zeros(n), rms, seed=7, channel="x")
        got = float(np.sqrt(np.mean(y ** 2)))
        assert abs(got - rms) / rms < 0.01, f"rms {got}（指定 {rms}）"


def test_noise_is_zero_mean_and_additive():
    n = 200000
    x = np.linspace(-1.0, 1.0, n)
    y = chain.add_noise(x, 0.1, seed=3, channel="y")
    assert abs(float(np.mean(y - x))) < 5e-4


def test_noise_density_conversion():
    """100 ug/rtHz を 1000 Hz 帯域で受けると 3.33 mg rms（雑音等価帯域 1.11 込み）。"""
    got = chain.noise_rms_from_density(100.0, 1000.0)
    assert abs(got - 100e-6 * math.sqrt(1000.0 * 1.11)) < 1e-12
    assert abs(got - 3.331e-3) < 1e-5
    # 帯域を 4 倍にすると rms は 2 倍
    assert abs(chain.noise_rms_from_density(100.0, 4000.0) / got - 2.0) < 1e-12


def test_same_seed_same_waveform():
    a = chain.add_noise(np.zeros(1000), 1.0, seed=42, channel="accel")
    b = chain.add_noise(np.zeros(1000), 1.0, seed=42, channel="accel")
    assert np.array_equal(a, b)


def test_different_seed_different_waveform():
    a = chain.add_noise(np.zeros(1000), 1.0, seed=42, channel="accel")
    c = chain.add_noise(np.zeros(1000), 1.0, seed=43, channel="accel")
    assert not np.array_equal(a, c)
    assert abs(float(np.corrcoef(a, c)[0, 1])) < 0.1


def test_channels_are_independent():
    """同じ種でも、チャネルが違えば別の波形（軸どうしで相関しない）。"""
    a = chain.add_noise(np.zeros(20000), 1.0, seed=5, channel="accel_tangential")
    b = chain.add_noise(np.zeros(20000), 1.0, seed=5, channel="accel_radial")
    assert not np.array_equal(a, b)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.05


def test_seed_is_stable_across_processes():
    """チャネル名の整数化に組み込み hash() を使っていない（実行ごとに変わらない）。"""
    import zlib
    got = chain.add_noise(np.zeros(3), 1.0, seed=1, channel="strain")
    expect = np.random.default_rng(
        np.random.SeedSequence([1, zlib.crc32(b"strain")])).normal(0, 1, 3)
    assert np.allclose(got, expect)


# ---------------------------------------------------------------------------
# 量子化と飽和
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("full_scale,bits", [(10.0, 16), (1000.0, 24), (40.0, 12)])
def test_quantization_step_equals_range_over_2_pow_bits(full_scale, bits):
    lsb = chain.lsb_of(full_scale, bits)
    assert lsb == full_scale / 2 ** bits
    # 段差そのものを見たいので、1 LSB より細かい刻みで入力を振る
    x = np.linspace(0.0, 30.0 * lsb, 6001)
    y = chain.quantize(x, full_scale, bits)
    codes = np.round(y / lsb)
    assert np.allclose(y, codes * lsb)                 # 段差の整数倍しか出ない
    steps = np.unique(np.round(np.diff(np.unique(y)) / lsb))
    assert steps.tolist() == [1.0]                      # 段差は 1 LSB で一定
    assert np.max(np.abs(y - x)) <= lsb / 2 + 1e-12     # 誤差は ±LSB/2
    # レンジいっぱいまで振っても、出力は必ず段差の整数倍
    big = chain.quantize(np.linspace(-full_scale / 2, full_scale / 2, 5001),
                         full_scale, bits)
    assert np.allclose(big / lsb, np.round(big / lsb))


def test_quantization_error_is_uniform():
    """量子化誤差の rms が LSB/sqrt(12) になる（床の見積もりの根拠）。"""
    full_scale, bits = 10.0, 12
    lsb = chain.lsb_of(full_scale, bits)
    rng = np.random.default_rng(0)
    x = rng.uniform(-4.0, 4.0, 200000)
    err = chain.quantize(x, full_scale, bits) - x
    assert abs(float(np.sqrt(np.mean(err ** 2))) - lsb / math.sqrt(12)) / (lsb / math.sqrt(12)) < 0.02


def test_saturation_clips_to_range():
    x = np.array([-100.0, -5.1, -5.0, 0.0, 3.0, 5.0, 5.1, 100.0])
    y = chain.saturate(x, 5.0)
    assert np.array_equal(y, np.array([-5.0, -5.0, -5.0, 0.0, 3.0, 5.0, 5.0, 5.0]))


def test_chain_saturates_and_counts():
    """レンジ ±5 に対して振幅 8 を入れたら、頭が潰れて飽和数が上がる。"""
    fs_in = 5000.0
    spec = chain.ChainSpec(sample_rate_hz=1000.0, range_amplitude=5.0, bits=12,
                           noise_rms=0.0, bandwidth_hz=100.0, channel="sat")
    _, x = tone(10.0, fs_in, 2.0, amp=8.0)
    y, info = chain.run_chain(x, fs_in, spec, seed=1)
    assert info["saturated_samples"] > 0
    assert float(np.max(np.abs(y))) <= 5.0 + 1e-12
    assert float(np.max(y)) == pytest.approx(5.0, abs=chain.lsb_of(10.0, 12))


# ---------------------------------------------------------------------------
# 連鎖まとめ
# ---------------------------------------------------------------------------

def test_run_chain_is_reproducible_and_ordered():
    fs_in = 5000.0
    spec = chain.ChainSpec(sample_rate_hz=2500.0, range_amplitude=2.0, bits=14,
                           noise_rms=0.01, bandwidth_hz=500.0, channel="c")
    _, x = tone(20.0, fs_in, 2.0, amp=1.0)
    y1, info = chain.run_chain(x, fs_in, spec, seed=11)
    y2, _ = chain.run_chain(x, fs_in, spec, seed=11)
    y3, _ = chain.run_chain(x, fs_in, spec, seed=12)
    assert np.array_equal(y1, y2)
    assert not np.array_equal(y1, y3)
    # 量子化が飽和より前なので、出力は必ず LSB の整数倍
    lsb = info["lsb"]
    assert np.allclose(y1 / lsb, np.round(y1 / lsb))
    assert len(y1) == len(x) // 2
    assert abs(line_amplitude(y1, spec.sample_rate_hz, 20.0) - 1.0) < 0.02


def test_run_chain_warns_when_bandwidth_exceeds_nyquist():
    spec = chain.ChainSpec(sample_rate_hz=500.0, range_amplitude=1.0, bits=12,
                           noise_rms=0.0, bandwidth_hz=400.0, channel="w")
    _, info = chain.run_chain(np.zeros(5000), 5000.0, spec, seed=1)
    assert "antialias_warning" in info


def test_run_chain_skips_bandlimit_that_cannot_do_anything():
    """帯域 10 kHz のセンサに 4 kHz のログを入れても、その段では何も落ちない。"""
    spec = chain.ChainSpec(sample_rate_hz=51200.0, range_amplitude=10.0, bits=16,
                           noise_rms=0.0, bandwidth_hz=10000.0, allow_upsample=True,
                           channel="hf")
    _, info = chain.run_chain(np.zeros(4000), 4000.0, spec, seed=1)
    assert info.get("bandlimit_skipped")


# ---------------------------------------------------------------------------
# 衝撃をイベント列から合成する
# ---------------------------------------------------------------------------

def make_events(times, amp=2.0, ring=3000.0, damping=0.05):
    return virtual.ImpactEvents(np.asarray(times, float),
                                np.full(len(times), amp), ring, damping)


def test_impact_waveform_has_the_specified_frequency_and_decay():
    """合成した減衰振動が a = A*exp(-2*pi*zeta*f*t)*sin(2*pi*f*t) になっている。"""
    ring, zeta, amp = 3000.0, 0.05, 2.0
    fs = virtual.impact_rate_hz(ring)
    ev = make_events([0.0], amp=amp, ring=ring, damping=zeta)
    y = virtual.impact_train(ev, 0.0, 0.05, fs)
    t = np.arange(len(y)) / fs
    expect = amp * np.exp(-2 * np.pi * zeta * ring * t) * np.sin(2 * np.pi * ring * t)
    n = int(0.005 * fs)      # 尾は 1/1000 まで落ちたところで切ってある
    assert np.allclose(y[:n], expect[:n], atol=1e-9)
    # 包絡線が 1 周期で exp(-2*pi*zeta) になる
    ratio = np.max(np.abs(y[:int(fs / ring)])) / np.max(
        np.abs(y[int(fs / ring):int(2 * fs / ring)]))
    assert abs(ratio - math.exp(2 * np.pi * zeta)) / math.exp(2 * np.pi * zeta) < 0.05


def test_impact_train_places_events_at_the_right_times():
    ring = 3000.0
    fs = virtual.impact_rate_hz(ring)
    ev = make_events([0.1, 0.37])
    y = virtual.impact_train(ev, 0.0, 0.5, fs)
    peaks = []
    for te in ev.times:
        i0 = int(te * fs)
        seg = np.abs(y[i0:i0 + int(0.002 * fs)])
        peaks.append(i0 + int(np.argmax(seg)))
    assert all(abs(p / fs - te) < 5e-4 for p, te in zip(peaks, ev.times))
    # イベントの前は静か
    assert np.max(np.abs(y[:int(0.09 * fs)])) < 1e-12


def test_impact_rate_must_be_fast_enough():
    ev = make_events([0.0], ring=3000.0)
    with pytest.raises(ValueError):
        virtual.impact_train(ev, 0.0, 0.1, 4000.0)


def test_impact_survives_the_fast_chain_but_not_the_slow_one():
    """同じ衝撃が、高速センサでは残り、低速センサでは帯域制限で落ちる。

    これが「加速度センサが 2 つ要る」ことの数値的な裏づけ。
    """
    ring = 3000.0
    fs_hi = virtual.impact_rate_hz(ring)
    ev = make_events([0.05, 0.2], amp=2.0, ring=ring)
    extra = virtual.impact_train(ev, 0.0, 0.4, fs_hi)
    zeros = np.zeros(int(0.4 * 4000.0))

    fast = chain.ChainSpec(sample_rate_hz=51200.0, range_amplitude=50 * 9.80665,
                           bits=16, noise_rms=0.0, bandwidth_hz=10000.0,
                           allow_upsample=True, channel="hf")
    slow = chain.ChainSpec(sample_rate_hz=1000.0, range_amplitude=2 * 9.80665,
                           bits=16, noise_rms=0.0, bandwidth_hz=200.0, channel="lf")
    y_fast, _ = chain.run_chain(zeros, 4000.0, fast, 1, extra=extra, extra_fs=fs_hi)
    y_slow, _ = chain.run_chain(zeros, 4000.0, slow, 1, extra=extra, extra_fs=fs_hi)

    assert np.max(np.abs(y_fast)) > 1.5           # 振幅 2 のうち大半が残る
    assert np.max(np.abs(y_slow)) < 0.3           # 200 Hz の帯域制限で落ちる
    # 高速側では 3 kHz にピークが立つ
    f = np.fft.rfftfreq(len(y_fast), 1.0 / fast.sample_rate_hz)
    a = np.abs(np.fft.rfft(y_fast))
    assert abs(f[int(np.argmax(a))] - ring) < 60.0


def test_extra_needs_its_rate():
    spec = chain.ChainSpec(sample_rate_hz=1000.0, range_amplitude=1.0, bits=12)
    with pytest.raises(ValueError):
        chain.run_chain(np.zeros(100), 1000.0, spec, 1, extra=np.zeros(100))


# ---------------------------------------------------------------------------
# 真値合成の式（連鎖の入口が正しいこと）
# ---------------------------------------------------------------------------

def test_slosh_frequency_matches_notes():
    """py/NOTES.md の 3.751 Hz と一致する（R=32.5mm, h=120.5mm）。"""
    f1 = virtual.slosh_frequency(0.12054, 0.0325)
    assert abs(f1 - 3.751) < 0.002


def test_empty_bottle_does_not_slosh():
    w1, L1, ratio = virtual.slosh_terms(np.array([0.0, -1.0]), 0.0325)
    assert np.all(w1 == 0) and np.all(L1 == 0) and np.all(ratio == 0)


def test_slosh_mass_ratio_matches_notes():
    """液深 120.5 mm で m1/m_liq = 0.123（py/NOTES.md 3 節）。"""
    _, _, ratio = virtual.slosh_terms(np.array([0.12054]), 0.0325)
    assert abs(float(ratio[0]) - 0.1231) < 0.001


def test_slosh_steady_state_recovers_tank_accel():
    """定常（phi = -a/g, phi' = 0）では、揺れる質量の絶対加速度が容器と同じになる。

    a1 = -(g*phi + 2*zeta*w1*L1*phi') という書き換えが正しいことの確認。
    """
    g = 9.80665
    R, h = 0.0325, 0.12054
    w1, L1, ratio = virtual.slosh_terms(np.array([h]), R, g)
    a = 0.8
    phi = -a / g
    m_liq = np.array([0.4])
    F = virtual.liquid_force(np.array([a]), np.array([phi]), np.array([0.0]),
                             m_liq, w1, L1, ratio, zeta=0.005, g=g)
    assert abs(float(F[0]) - float(m_liq[0]) * a) < 1e-9


def test_section_modulus_formula():
    """中実丸棒（内径 0）では Z = pi*D^3/32 に戻る。"""
    assert abs(virtual.section_modulus_mm3(40.0, 0.0) - math.pi * 40 ** 3 / 32) < 1e-9
    with pytest.raises(ValueError):
        virtual.section_modulus_mm3(30.0, 40.0)




def test_added_params_do_not_touch_the_file(tmp_path):
    """params.json は読むだけ。既定値の埋め込みはメモリ上だけで起きる。

    いまの params.json には取り付けの定数が入っているので、`added` は空になる。
    埋め込みのほうを試すために、キーを抜いた写しを作ってそちらで確かめる。
    どちらの場合もファイルが書き換わらないことが、この試験の本体。
    """
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "params.json")
    before = open(src, "rb").read()
    params, added = virtual.load_params(src)
    assert params["sensors"]["accel_lf"]["frame_equiv_mass_kg"] > 0
    assert open(src, "rb").read() == before

    # 取り付けの定数を抜いた写しでは、既定値が補われて `added` に名前が残る
    stripped = json.loads(before.decode("utf-8"))
    del stripped["sensors"]["accel_lf"]["frame_equiv_mass_kg"]
    del stripped["sensors"]["accel_hf"]["structure_freq_hz"]
    copy = tmp_path / "params.json"
    copy.write_text(json.dumps(stripped, ensure_ascii=False), encoding="utf-8")
    copy_before = copy.read_bytes()

    params2, added2 = virtual.load_params(str(copy))
    assert "sensors.accel_lf.frame_equiv_mass_kg" in added2
    assert "sensors.accel_hf.structure_freq_hz" in added2
    assert params2["sensors"]["accel_lf"]["frame_equiv_mass_kg"] > 0
    assert copy.read_bytes() == copy_before


def test_spec_from_params_reads_each_sensor():
    """params.json の各センサから、連鎖の設定が矛盾なく作れる。"""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "params.json")
    params, _ = virtual.load_params(src)
    g = params["sim"]["gravity_m_s2"]

    lf = virtual.spec_from_params(params, "accel_lf", g)
    assert lf.highpass_hz is None                     # DC 結合
    assert lf.range_amplitude == pytest.approx(2.0 * g)
    assert lf.bandwidth_hz < 0.5 * lf.sample_rate_hz  # ナイキストの内側

    hf = virtual.spec_from_params(params, "accel_hf", g)
    assert hf.highpass_hz == 2.0                      # 交流結合
    assert hf.sample_rate_hz > 10 * lf.sample_rate_hz

    st = virtual.spec_from_params(params, "strain", g)
    assert st.bits == 24 and st.noise_rms == 0.5


# ---------------------------------------------------------------------------
# バイナリ読み込み（作り物のファイルで、想定外を弾けるかだけ見る）
# ---------------------------------------------------------------------------

LAYOUT = {"001": (64, 6, 5), "002": (128, 13, 9)}


def make_dump_bytes(path, version="001", n_scalars=None, n_per_station=None,
                    n_stations=2, n_records=8, header_bytes=None, record_bytes=None,
                    log_dt=1e-3, truncate=0, bottle_radius=0.0325, magic=None,
                    fault_flags=0, bearing_ratio=0.0, bearing_ring=0.0,
                    omega=0.0, real_dt=None):
    """FORMAT.md どおりの並びで、作り物のダンプを書く。

    omega を入れると th_t が一定速度で増える（軸受のイベント時刻の試験用）。
    real_dt を入れると、時刻の刻みをヘッダの log_dt とわざとずらせる。
    """
    hb0, nsc0, nps0 = LAYOUT[version]
    header_bytes = hb0 if header_bytes is None else header_bytes
    n_scalars = nsc0 if n_scalars is None else n_scalars
    n_per_station = nps0 if n_per_station is None else n_per_station
    magic = ("RFILL" + version).encode() if magic is None else magic
    real_dt = log_dt if real_dt is None else real_dt

    ncol = n_scalars + n_per_station * n_stations
    rb = record_bytes if record_bytes is not None else 4 * ncol
    head = bytearray(header_bytes)
    head[0:8] = magic
    struct.pack_into("<6I", head, 8, header_bytes, rb, n_stations,
                     n_scalars, n_per_station, n_records)
    struct.pack_into("<2d", head, 32, 2e-5, log_dt)
    struct.pack_into("<4f", head, 48, 0.225, bottle_radius, 0.15, 5.0)
    if header_bytes >= 128:
        struct.pack_into("<f", head, 64, 0.09)        # max_dz_over_R
        struct.pack_into("<I", head, 68, 0)           # range_exceeded
        struct.pack_into("<f", head, 72, 0.20)        # range_limit
        struct.pack_into("<I", head, 76, fault_flags)
        struct.pack_into("<8f", head, 80, 2.4, 1.5, 0.9, 20.0,
                         bearing_ring, bearing_ratio, 4.0e-4, 5.0e-4)
        struct.pack_into("<I", head, 112, 0)
        struct.pack_into("<2f", head, 116, 1.0, 23.57)

    body = bytearray()
    for i in range(n_records):
        rec = [0.0] * ncol
        rec[0] = i * real_dt                      # t
        rec[1] = omega * i * real_dt              # th_t
        rec[2] = omega                            # omega
        for s in range(n_stations):
            base = n_scalars + n_per_station * s
            V = 1.0e-4 * (s + 1)
            rec[base + 0] = V
            rec[base + 1] = V / (math.pi * bottle_radius ** 2)   # h
        body += struct.pack("<%df" % ncol, *rec)
    raw = bytes(head) + bytes(body)
    if truncate:
        raw = raw[:-truncate]
    with open(path, "wb") as fp:
        fp.write(raw)
    return path


def test_reads_a_well_formed_001_dump(tmp_path):
    d = read_dump(make_dump_bytes(tmp_path / "ok1.bin", "001"))
    assert d.n_records == 8 and d.n_stations == 2
    assert d.header.version == "001"
    assert not d.has_reaction          # 版 001 に反力は無い
    # 時刻は f32 なので、実測のログ周波数はぴったりにはならない
    assert d.log_rate_hz == pytest.approx(1000.0, rel=1e-5)
    assert d.warnings == []


def test_reads_a_well_formed_002_dump(tmp_path):
    d = read_dump(make_dump_bytes(tmp_path / "ok2.bin", "002"))
    assert d.header.version == "002"
    assert d.has_reaction
    for arr in (d.F_tab_t, d.F_tab_r, d.T_slosh, d.a_bear, d.J_load,
                d.F_liq_t, d.dz_over_R, d.range_flag):
        assert arr is not None
    assert d.header.tact_s == pytest.approx(2.4)
    assert d.header.index_time_s == pytest.approx(1.5)
    assert d.header.gear_ratio == pytest.approx(20.0)


def test_actual_log_rate_wins_over_the_nominal_one(tmp_path):
    """コアは dt の整数倍で間引くので、公称のログ周波数とは一致しない。

    ここを取り違えるとスペクトルの周波数軸が数 % ずれる（FORMAT.md 6 節）。
    """
    p = make_dump_bytes(tmp_path / "rate.bin", "002",
                        log_dt=1.0 / 4000.0, real_dt=13 * 2e-5)
    d = read_dump(p)
    assert d.header.nominal_log_rate_hz == pytest.approx(4000.0)
    assert d.log_rate_hz == pytest.approx(1.0 / (13 * 2e-5), rel=1e-4)
    assert any("実際のログ周波数" in w for w in d.warnings)


def test_rejects_foreign_file(tmp_path):
    p = make_dump_bytes(tmp_path / "bad.bin", magic=b"NOTMINE!")
    with pytest.raises(DumpFormatError, match="magic"):
        read_dump(p)


def test_rejects_unknown_version(tmp_path):
    p = make_dump_bytes(tmp_path / "v9.bin", magic=b"RFILL009")
    with pytest.raises(DumpFormatError, match="未対応"):
        read_dump(p)


def test_rejects_changed_field_count(tmp_path):
    """版に対して項目数が合わなければ、既定では黙って読まずに落ちる。"""
    p = make_dump_bytes(tmp_path / "more.bin", "002", n_scalars=15, n_per_station=11)
    with pytest.raises(DumpFormatError, match="項目数"):
        read_dump(p)
    d = read_dump(p, strict=False)      # 明示すれば先頭の既知ぶんだけ読む
    assert d.n_records == 8 and any("項目数" in w for w in d.warnings)


def test_rejects_missing_fields(tmp_path):
    p = make_dump_bytes(tmp_path / "less.bin", "001", n_scalars=4, n_per_station=3)
    with pytest.raises(DumpFormatError, match="足りない"):
        read_dump(p, strict=False)


def test_rejects_wrong_header_length_for_the_version(tmp_path):
    p = make_dump_bytes(tmp_path / "hb.bin", "002", header_bytes=64)
    with pytest.raises(DumpFormatError, match="ヘッダ長"):
        read_dump(p)


def test_rejects_inconsistent_record_bytes(tmp_path):
    p = make_dump_bytes(tmp_path / "rb.bin", record_bytes=999)
    with pytest.raises(DumpFormatError, match="record_bytes"):
        read_dump(p)


def test_rejects_truncated_file(tmp_path):
    p = make_dump_bytes(tmp_path / "cut.bin", truncate=7)
    with pytest.raises(DumpFormatError, match="割り切れない"):
        read_dump(p)


def test_counts_records_from_size_when_header_says_zero(tmp_path):
    """標準出力へ吐くと n_records を書き戻せないので 0 のまま（FORMAT.md 1 節）。"""
    p = make_dump_bytes(tmp_path / "zero.bin", n_records=8)
    raw = bytearray(open(p, "rb").read())
    struct.pack_into("<I", raw, 28, 0)
    (tmp_path / "zero.bin").write_bytes(bytes(raw))
    d = read_dump(tmp_path / "zero.bin")
    assert d.n_records == 8 and any("n_records" in w for w in d.warnings)


def test_unknown_version_is_validated_against_invariants(tmp_path):
    """未対応の版でも、h = V/(pi R^2) が合えば読む。合わなければ落とす。"""
    p = make_dump_bytes(tmp_path / "v3ok.bin", "002", magic=b"RFILL003",
                        n_scalars=15, n_per_station=11)
    d = read_dump(p, allow_unknown_version=True)
    assert d.n_records == 8
    assert any("記述が無い" in w for w in d.warnings)
    assert not d.has_reaction      # 並びが保証できないので拡張欄は取り出さない

    raw = bytearray(open(p, "rb").read())
    off = 128 + 4 * 15 + 4         # 1 レコード目 ステーション 0 の h
    struct.pack_into("<f", raw, off, 12345.0)
    q = tmp_path / "v3ng.bin"
    q.write_bytes(bytes(raw))
    with pytest.raises(DumpFormatError, match="食い違う"):
        read_dump(q, allow_unknown_version=True)


# ---------------------------------------------------------------------------
# 衝撃イベント（RFEVT002）の取り出し — 新シグネチャ（events 経由）
# ---------------------------------------------------------------------------
# rev.3 では衝撃を連続ログに載せず、発生時刻と振幅をイベント列（サイドカー）で出す。
# bearing_events / cam_events は Dump ではなく Events を受け取る。ここでは Events を
# 手で組んで、取り出しの規則だけを純粋に確かめる（core/filler は要らない）。

def make_event_log(bearing_times, cam_times=(), accel=2.0,
                   ring=3000.0, damping=0.05, cam_torque=0.2):
    """RFEVT002 相当の Events を手で組む（kind=0 軸受 / kind=1 カム当たり）。

    d0 に軸受は加速度 [m/s^2]、カムは衝撃トルク [N m] を入れる（FORMAT.md 5.3）。
    """
    tb = np.asarray(bearing_times, float)
    tc = np.asarray(cam_times, float)
    t = np.concatenate([tb, tc])
    kind = np.concatenate([np.zeros(len(tb), int),
                           np.ones(len(tc), int)]).astype(np.int64)
    order = np.argsort(t, kind="stable")
    t, kind = t[order], kind[order]
    d = np.zeros((len(t), 8))
    d[kind == 0, 0] = accel          # 軸受: d0 = 加速度 [m/s^2]
    d[kind == 1, 0] = cam_torque     # カム: d0 = 衝撃トルク [N m]
    header = {"bearing_ring_freq_hz": ring, "bearing_ring_damping": damping,
              "cam_impact_torque_Nm": cam_torque}
    return Events(header=header, t=t, kind=kind,
                  station=np.full(len(t), -1, np.int64),
                  th_t=np.zeros(len(t)), th_m=np.zeros(len(t)), d=d)


def test_bearing_events_come_from_the_event_log():
    """軸受はモータ軸（連続回転）に置いたので、イベントは時間軸で等間隔。

    旧版はテーブル軸で回転角等間隔だった（そのぶん bearing_events も dump から角度で
    作っていた）。rev.3 では RFEVT002 kind=0 をそのまま取り出す。混ざった kind=1
    （カム当たり）は拾わない。
    """
    step = 1.0 / 89.5                       # 外輪傷通過 89.5 Hz
    times = np.arange(20) * step + 0.01
    ev = make_event_log(times, cam_times=[0.05, 0.4], accel=2.0, ring=3000.0)
    out = virtual.bearing_events(ev, {"faults": {}})
    assert len(out) == 20                    # カム 2 件は入らない
    assert out.ring_freq_hz == 3000.0
    assert np.allclose(out.amplitudes, 2.0)
    assert np.allclose(np.diff(out.times), step, atol=1e-9)
    # 減衰振動に合成できる（高速センサでは残り、低速センサでは折り返す、は別テスト）
    fs = virtual.impact_rate_hz(out.ring_freq_hz)
    assert virtual.impact_train(out, 0.0, float(times[-1] + 0.05), fs).max() > 0.0


def test_bearing_impulse_can_be_overridden():
    """impulse_accel_m_s2 を渡すと各衝撃の振幅を上書きする。"""
    ev = make_event_log([0.1, 0.2, 0.3], accel=2.0)
    out = virtual.bearing_events(ev, {"faults": {}}, impulse_accel_m_s2=5.0)
    assert len(out) == 3
    assert np.allclose(out.amplitudes, 5.0)


def test_no_bearing_events_when_the_log_has_none():
    """軸受イベントが無ければ空。impact_train も 0。events=None でも空。"""
    ev = make_event_log([], cam_times=[0.1, 0.2])       # カム当たりだけ
    out = virtual.bearing_events(ev, {"faults": {}})
    assert len(out) == 0
    assert virtual.impact_train(out, 0.0, 1.0, 51200.0).max() == 0.0
    assert len(virtual.bearing_events(None)) == 0


def test_cam_events_replace_backlash():
    """バックラッシュ（旧 T_bl）は無く、カムフォロワ当たり（kind=1）が代わり。"""
    ev = make_event_log([0.1, 0.2, 0.3], cam_times=[0.05, 0.5], cam_torque=0.2)
    out = virtual.cam_events(ev, {})
    assert len(out) == 2                     # 軸受 3 件は入らない
    assert np.allclose(out.times, [0.05, 0.5])
    assert np.allclose(out.amplitudes, 0.2)
    assert out.ring_freq_hz == 0.0           # トルク衝撃なので加速度リンギングは合成しない
    assert len(virtual.cam_events(None)) == 0


# ---------------------------------------------------------------------------
# RFILL004（現行形式）を core/filler で 1 本生成して層を通す
# ---------------------------------------------------------------------------
# 読み層（canonical ルート）・真値層（build_truth）・イベント層（bearing/cam_events）・
# 層またぎ不変量（verify_reaction）を、実ダンプ 1 本でまとめて確かめる。
# core/filler が無い環境では skip（作り物のバイト合成はしない）。

@pytest.fixture(scope="module")
def dump004(tmp_path_factory):
    if not os.path.exists(FILLER):
        pytest.skip("core/filler が無い（RFILL004 を生成できない）")
    out = tmp_path_factory.mktemp("rfill004") / "fault.bin"
    cmd = [FILLER, "--params", PARAMS, "--cycles", "1",
           "--fault-bearing", "--fault-cam", "--out", str(out), "--quiet"]
    r = subprocess.run(cmd, cwd=os.path.dirname(FILLER),
                       capture_output=True, timeout=300)
    if r.returncode != 0 or not out.exists():
        pytest.skip(f"core/filler が失敗した (rc={r.returncode}): "
                    f"{r.stderr.decode('utf-8', 'replace')[:200]}")
    return str(out)


def test_004_routes_to_canonical(dump004):
    """RFILL004 は magic 振り分けで canonical（dumpio.Dump）に回る。旧列は持たない。"""
    d = read_dump(dump004)
    assert isinstance(d, CanonicalDump)
    # canonical な小文字列は属性で引ける
    for col in ("t", "th_t", "omega", "alpha", "j_load", "torque_input",
                "torque_slosh", "f_tab_x", "f_tab_y", "V", "h", "present"):
        assert hasattr(d, col), col
    # 旧名（motor_current / T_slosh / F_tab_t）は持たない
    for old in ("motor_current", "T_slosh", "F_tab_t", "F_tab_r", "a_bear"):
        with pytest.raises(AttributeError):
            getattr(d, old)


def test_004_build_truth_runs(dump004):
    """build_truth が通り、電流はトルクから再構成されて励磁電流付近でほぼ一定。"""
    d = read_dump(dump004)
    params, _ = virtual.load_params(PARAMS)
    truth = virtual.build_truth(d, params)
    n = d.n_records
    for arr in (truth.current_A, truth.strain_ustrain,
                truth.accel_lf_radial, truth.accel_hf_radial, truth.force_t):
        assert arr.shape == (n,)
        assert np.all(np.isfinite(arr))
    i0 = float(params["drive"]["no_load_current_A"])
    assert abs(float(truth.current_A.mean()) - i0) < 0.02
    assert float(truth.current_A.std()) < 0.02   # トルクの弱い観測量なのでほぼ一定


def test_004_event_layers_return_counts(dump004):
    """load_run でイベント列を取り、bearing_events / cam_events が件数を返す。"""
    params, _ = virtual.load_params(PARAMS)
    dump, events = load_run(dump004)
    assert events is not None
    bev = virtual.bearing_events(events, params)
    cev = virtual.cam_events(events, params)
    assert len(bev) > 0 and bev.ring_freq_hz == 3000.0
    assert len(cev) > 0
    # 軸受の衝撃は減衰振動に合成できる（時刻は連続ログの範囲に入っている）
    fs = virtual.impact_rate_hz(bev.ring_freq_hz)
    y = virtual.impact_train(bev, float(dump.t[0]),
                             float(dump.t[-1] - dump.t[0]), fs)
    assert y.max() > 0.0


def test_004_verify_reaction_is_small(dump004):
    """層またぎ不変量。符号や軸を取り違えると 2 前後に跳ねる。正常なら ~0.055。"""
    d = read_dump(dump004)
    params, _ = virtual.load_params(PARAMS)
    err = virtual.verify_reaction(d, params)
    assert err < 0.15


def test_004_verify_reaction_catches_sign_flip(dump004):
    """不変量が符号の取り違えを本当に捕まえることを確かめる。

    f_tab_x/y を反転すると table_reaction 側だけが逆相になり
    （_reaction_from_stations はステーションから組むので影響なし）、
    成分残差は 2 前後に跳ねる。大きさを先に取る実装だと跳ねないので、
    ここが回帰しないよう固定する。
    """
    import copy
    d = read_dump(dump004)
    params, _ = virtual.load_params(PARAMS)
    flipped = copy.copy(d)
    flipped.f_tab_x = -d.f_tab_x
    flipped.f_tab_y = -d.f_tab_y
    assert virtual.verify_reaction(flipped, params) > 1.5
