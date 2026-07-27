"""仮想センサ。真値の合成と、信号連鎖を通した出力。

物理コアの出力（:mod:`sensors.read_dump` が返す :class:`Dump`）から
SENSORS.md 3 節・5.5 節のとおりに真値を作り、:mod:`sensors.chain` の連鎖に通す。

センサは 4 系統（SENSORS.md 5.5）
--------------------------------
* ``accel_lf`` 架台の柱、DC 結合 MEMS。接線・半径の 2 軸。揺れと回転同期成分を見る
* ``accel_hf`` 軸受箱、交流結合の圧電。半径 1 軸。軸受の衝撃とリンギングを見る
* ``strain``   支持軸の根元の曲げひずみ
* ``current``  モータ電流

**1 つのセンサで全部は見られない。** 低速側に 3 kHz のリンギングを通せば折り返し、
高速側は取り付けが硬いので 3.75 Hz の揺れが出てこない。demo.py がその 2 枚を描く。

連続量と衝撃を分けて扱う（SENSORS.md 5.5）
------------------------------------------
連続量（テーブル角・液面・反力・電流）は ``log_rate_hz`` の連続ログから読む。
衝撃（軸受・バックラッシュ）はログに乗せず、**発生時刻と振幅のイベント列**として扱い、
センサ層が各センサの刻みで減衰振動に合成して足す。

いまのところ物理コアはイベント列を別途は出していない（core/FORMAT.md 4 節の CLI にも
その出力は無い）。代わりに、FORMAT.md 2.5 に書かれた発生規則
「テーブル角が ``2*pi / bearing_defect_ratio`` 進むごとに 1 回」から
:func:`bearing_events` がイベント列を作り直す。ログの ``a_bear`` は
``ring_freq_hz`` がログのナイキストを超えていて折り返しているので、波形としては使わない
（時刻の突き合わせにだけ使える）。コアがイベント列を出すようになったら、
:func:`bearing_events` を差し替えるだけで残りはそのまま動く。

液の力の書き換え
----------------
SENSORS.md 1 節の ``a1 = a_tank + L1*phi''`` は phi の二階微分が要る。
運動方程式 ``phi'' + 2*zeta*w1*phi' + w1^2*phi = -a_tank/L1`` を代入して phi'' を消すと、
``L1*w1^2 = g`` なので

    a1 = -( g*phi + 2*zeta*w1*L1*phi' )

となり、一階微分だけで済む（core/FORMAT.md 2.4 も同じ書き方をしている）。

params.json に無い定数は :data:`ADDED_DEFAULTS` に持ち、読み込み時に不足ぶんだけ埋める。
根拠は PARAMS_ADDED.md。params.json 自体は書き換えない。
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import chain
from .read_dump import Dump, Events

EPS1 = 1.8412  # J1'(x) = 0 の第 1 根（MODEL.md 2 節）
G_STD = 9.80665

# params.json に足すべき定数。値の根拠は PARAMS_ADDED.md。
ADDED_DEFAULTS = {
    "sensors": {
        "accel_lf": {
            "frame_equiv_mass_kg": 60.0,
            "structure_freq_hz": 250.0,
            "structure_damping": 0.03,
            "structure_model": "grounded",
        },
        "accel_hf": {
            "frame_equiv_mass_kg": 25.0,
            "structure_freq_hz": 1200.0,
            "structure_damping": 0.04,
            "structure_model": "grounded",
        },
        "strain": {
            "arm_length_mm": 150.0,
            "shaft_outer_diameter_mm": 45.0,
            "shaft_inner_diameter_mm": 39.0,
            "youngs_modulus_GPa": 193.0,
            "gauge_angle_deg": 0.0,
            "rotating_gauge": False,
        },
        "current": {
            "bandwidth_hz": 1500.0,
        },
    }
}

# 衝撃を合成するときの内部刻み。リンギングに対して十分速く、かつ
# いちばん速いセンサ（51.2 kHz）以上であること。
IMPACT_RATE_MIN_HZ = 51200.0
IMPACT_RATE_RING_FACTOR = 16.0


def _deep_fill(dst: dict, src: dict, added: list, prefix: str = "") -> None:
    """dst に無いキーだけ src から埋める。埋めたキー名を added に残す。"""
    for k, v in src.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            node = dst.setdefault(k, {})
            if not isinstance(node, dict):
                raise TypeError(f"{path} が dict ではない")
            _deep_fill(node, v, added, prefix=path + ".")
        elif k not in dst:
            dst[k] = v
            added.append(path)


def load_params(path: str | os.PathLike = "params.json") -> tuple[dict, list]:
    """params.json を読み、足りない定数を既定値で埋めて返す。

    返り値は (params, 埋めたキーの一覧)。ファイルは読むだけ。
    """
    with open(path, "r", encoding="utf-8") as fp:
        params = json.load(fp)
    added: list[str] = []
    _deep_fill(params, copy.deepcopy(ADDED_DEFAULTS), added)
    return params, added


def section_modulus_mm3(outer_mm: float, inner_mm: float) -> float:
    """中空丸軸の断面係数 Z = pi*(D^4 - d^4) / (32*D)  [mm^3]"""
    if outer_mm <= inner_mm or inner_mm < 0:
        raise ValueError("外径は内径より大きいこと")
    return np.pi * (outer_mm ** 4 - inner_mm ** 4) / (32.0 * outer_mm)


# ---------------------------------------------------------------------------
# スロッシングの補助量
# ---------------------------------------------------------------------------

def slosh_terms(h_m, R_m: float, g: float = G_STD):
    """液深 h からスロッシングの諸量を出す。

    返り値 (w1, L1, m1_ratio)。h <= 0 のところは 0 を返す（液が無ければ揺れない）。
    """
    h = np.asarray(h_m, dtype=float)
    ok = h > 1e-9
    hs = np.where(ok, h, 1.0)  # 0 割り回避用のダミー
    th = np.tanh(EPS1 * hs / R_m)
    w1sq = (g * EPS1 / R_m) * th
    w1 = np.where(ok, np.sqrt(w1sq), 0.0)
    L1 = np.where(ok, g / np.where(w1sq > 0, w1sq, 1.0), 0.0)
    m1_ratio = np.where(ok, (R_m / (2.2 * hs)) * th, 0.0)
    return w1, L1, m1_ratio


def slosh_frequency(h_m: float, R_m: float, g: float = G_STD) -> float:
    """液深 h でのスロッシング固有振動数 [Hz]。"""
    w1, _, _ = slosh_terms(np.array([h_m]), R_m, g)
    return float(w1[0] / (2.0 * np.pi))


def liquid_force(a_tank, phi, phidot, m_liq, w1, L1, m1_ratio,
                 zeta: float, g: float = G_STD):
    """液がボトルに及ぼす水平力 [N]（方向ごと）。

    F_liq = m0*a_tank + m1*a1、a1 = -(g*phi + 2*zeta*w1*L1*phi')。
    """
    m1 = m_liq * m1_ratio
    m0 = m_liq - m1
    a1 = -(g * phi + 2.0 * zeta * w1 * L1 * phidot)
    return m0 * a_tank + m1 * a1


# ---------------------------------------------------------------------------
# 衝撃イベント
# ---------------------------------------------------------------------------

@dataclass
class ImpactEvents:
    """衝撃の発生時刻と振幅。波形にはなっていない。

    times      : 発生時刻 [s]
    amplitudes : 振幅（加速度なら m/s^2）
    ring_freq_hz, ring_damping : 減衰振動の諸元
    source     : どこから作ったか（報告用）
    """

    times: np.ndarray
    amplitudes: np.ndarray
    ring_freq_hz: float
    ring_damping: float
    source: str = ""

    def __len__(self) -> int:
        return len(self.times)


def bearing_events(events: Events, params: dict = None,
                   impulse_accel_m_s2: float | None = None) -> ImpactEvents:
    """軸受外輪傷の衝撃を、コアのイベント列（RFEVT002 kind=0）から取り出す。

    rev.3 では衝撃を連続ログに載せず、発生時刻と振幅をイベント列で出す
    （core/FORMAT.md 5.3）。軸受はモータ軸（1500rpm 連続回転）に置いたので、
    欠陥通過は **時間軸で等間隔**（旧版のテーブル軸・回転角等間隔とは違う）。

    d0 が加速度振幅 [m/s^2]、d1 がリンギング周波数、d2 が減衰比（FORMAT.md 5.3）。
    リンギングの周波数と減衰はイベント列ヘッダの代表値を使う（1 記録では一定）。
    ``impulse_accel_m_s2`` を渡すと各衝撃の加速度振幅を上書きする。
    """
    if events is None or len(events) == 0:
        return ImpactEvents(np.zeros(0), np.zeros(0), 0.0, 0.0,
                            source="イベント列が無い")
    bev = events.of(0)
    h = events.header
    ring = float(h.get("bearing_ring_freq_hz", 0.0))
    damping = float(h.get("bearing_ring_damping", 0.05))
    if len(bev) == 0 or ring <= 0.0:
        return ImpactEvents(np.zeros(0), np.zeros(0), ring, damping,
                            source="軸受の衝撃イベントが無い")
    amp = bev.d[:, 0].copy()                       # d0 加速度 [m/s^2]
    if impulse_accel_m_s2 is not None:
        amp = np.full(bev.t.shape, float(impulse_accel_m_s2))
    return ImpactEvents(
        times=bev.t.copy(),
        amplitudes=amp,
        ring_freq_hz=ring,
        ring_damping=damping,
        source=f"RFEVT002 kind=0（{len(bev)} 件・時間軸で等間隔）",
    )


def cam_events(events: Events, params: dict = None) -> ImpactEvents:
    """カムフォロワの当たりの衝撃を、イベント列（RFEVT002 kind=1）から取り出す。

    旧版のバックラッシュ（``T_bl``）を置き換える。カムは停留部で予圧されるので
    バックラッシュは無く、代わりにフォロワの摩耗・当たりが衝撃源になる。
    d0 が衝撃トルク [N m]、d1 がすきま [rad]（FORMAT.md 5.3）。
    トルクの衝撃なので加速度リンギングの合成には使わず（``ring_freq_hz=0``）、
    発生時刻と振幅を**検出**に使う。
    """
    if events is None or len(events) == 0:
        return ImpactEvents(np.zeros(0), np.zeros(0), 0.0, 0.0,
                            source="イベント列が無い")
    cev = events.of(1)
    if len(cev) == 0:
        return ImpactEvents(np.zeros(0), np.zeros(0), 0.0, 0.0,
                            source="カム当たりのイベントが無い")
    return ImpactEvents(
        times=cev.t.copy(),
        amplitudes=cev.d[:, 0].copy(),             # d0 衝撃トルク [N m]
        ring_freq_hz=0.0,
        ring_damping=0.0,
        source=f"RFEVT002 kind=1（{len(cev)} 件）",
    )


def impact_rate_hz(ring_freq_hz: float) -> float:
    """衝撃を合成するときの内部刻み。"""
    return max(IMPACT_RATE_MIN_HZ, IMPACT_RATE_RING_FACTOR * float(ring_freq_hz))


def impact_train(events: ImpactEvents, t0: float, duration: float, fs_hz: float) -> np.ndarray:
    """イベント列から減衰振動の列を合成する（SENSORS.md 4.2）。

        a(t) = A * exp(-2*pi*zeta*f*(t-t0)) * sin(2*pi*f*(t-t0))

    fs_hz はリンギングに対して十分速いこと（:func:`impact_rate_hz`）。
    ここで作った波形を、各センサの帯域制限と間引きに通す。
    """
    n = int(round(duration * fs_hz)) + 1
    y = np.zeros(n)
    if len(events) == 0 or events.ring_freq_hz <= 0:
        return y
    if fs_hz < 4.0 * events.ring_freq_hz:
        raise ValueError(
            f"合成の刻み {fs_hz} Hz がリンギング {events.ring_freq_hz} Hz に対して粗すぎる")
    f = events.ring_freq_hz
    zeta = events.ring_damping
    decay = 2.0 * np.pi * zeta * f
    # 振幅が 1/1000 に落ちるまでを 1 発ぶんの長さとする
    tail = min(np.log(1000.0) / decay if decay > 0 else duration, duration)
    m = int(round(tail * fs_hz)) + 1
    tt = np.arange(m) / fs_hz
    shape = np.exp(-decay * tt) * np.sin(2.0 * np.pi * f * tt)
    for te, amp in zip(events.times, events.amplitudes):
        i = int(round((te - t0) * fs_hz))
        if i >= n or i + m <= 0:
            continue
        a, b = max(i, 0), min(i + m, n)
        y[a:b] += amp * shape[a - i:b - i]
    return y


# ---------------------------------------------------------------------------
# 真値の合成
# ---------------------------------------------------------------------------

@dataclass
class Truth:
    """物理コアのログから作った、まだセンサを通していない量。"""

    t: np.ndarray
    fs_hz: float
    force_t: np.ndarray            # センサ方位で分解した水平合力 [N]（接線成分）
    force_r: np.ndarray            # 同 半径成分
    accel_lf_tangential: np.ndarray    # 架台の柱 [m/s^2]（構造共振まで通した）
    accel_lf_radial: np.ndarray
    accel_hf_radial: np.ndarray        # 軸受箱 [m/s^2]
    strain_ustrain: np.ndarray         # 支持軸根元の曲げひずみ [ustrain]
    current_A: np.ndarray              # モータ電流 [A]
    torque_slosh_Nm: np.ndarray        # 液が割出し軸に返すトルク [Nm]（参考）
    notes: list = field(default_factory=list)


def table_reaction(dump: Dump, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """テーブルに掛かる水平合力を、センサ方位（接線・半径）で分解して返す。

    版 004 のログはコアが `f_tab_x` / `f_tab_y`（世界座標の力）を出す。
    センサ方位 az（世界角）で半径・接線へ分解する。
    版 002 は `F_tab_t` / `F_tab_r`（センサ方位で分解済）。
    版 001 のログには無いので、ステーションごとの状態から組み直す。
    """
    az = np.deg2rad(float(params["sensors"]["accel_lf"].get("azimuth_deg", 0.0)))
    if hasattr(dump, "f_tab_x"):                   # 004: 世界座標 x/y の力
        fx, fy = dump.f_tab_x, dump.f_tab_y
        c, s = np.cos(az), np.sin(az)
        # 符号は _reaction_from_stations（ステーションから独立に組んだ反力）と
        # verify_reaction の成分残差で突き合わせて確定した（負）。+符号は残差 2.06
        # （完全逆相）、−符号は 0.055。コアの f_tab_* は液・ボトルがテーブルに
        # 及ぼす力で、架台が受ける反力は逆向き。
        f_r = -(fx * c + fy * s)
        f_t = -(-fx * s + fy * c)
        return f_t, f_r
    if dump.has_reaction:
        # コアの分解は世界角 0 基準。センサ方位が違うなら回してから分解し直す。
        fx, fy = dump.F_tab_r, dump.F_tab_t       # x = 半径, y = 接線（FORMAT.md 2.3）
        if abs(az) > 1e-12:
            c, s = np.cos(az), np.sin(az)
            f_r = fx * c + fy * s
            f_t = -fx * s + fy * c
            return f_t, f_r
        return dump.F_tab_t.copy(), dump.F_tab_r.copy()
    return _reaction_from_stations(dump, params, az)


def _reaction_from_stations(dump: Dump, params: dict, az: float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """ステーションごとの状態から水平合力を組み立てる（版 001 用・突き合わせ用）。

    版 002 の `F_tab_*` と一致することは test_chain.py で確認している。
    """
    g = float(params["sim"]["gravity_m_s2"])
    rho = float(params["liquid"]["density_kg_m3"])
    zeta = float(params["liquid"]["slosh_damping_ratio"])
    m_bottle = float(params["bottle"]["empty_mass_kg"])
    Rp = float(dump.header.pitch_radius_m)
    R = float(dump.header.bottle_radius_m)

    a_tank_t = Rp * dump.alpha
    a_tank_r = -Rp * dump.omega ** 2
    w1, L1, ratio = slosh_terms(dump.h, R, g)
    m_liq = rho * dump.V
    phidot_t = np.gradient(dump.phi_t, dump.t, axis=0)
    phidot_r = np.gradient(dump.phi_r, dump.t, axis=0)
    F_liq_t = liquid_force(a_tank_t[:, None], dump.phi_t, phidot_t,
                           m_liq, w1, L1, ratio, zeta, g)
    F_liq_r = liquid_force(a_tank_r[:, None], dump.phi_r, phidot_r,
                           m_liq, w1, L1, ratio, zeta, g)
    F_t = m_bottle * a_tank_t[:, None] + F_liq_t
    F_r = m_bottle * a_tank_r[:, None] + F_liq_r

    nst = dump.n_stations
    th = dump.th_t[:, None] + np.arange(nst)[None, :] * (2.0 * np.pi / nst)
    c, s = np.cos(th), np.sin(th)
    fx = np.sum(-F_t * s + F_r * c, axis=1)      # 接線 e_t = (-sin, cos)
    fy = np.sum(F_t * c + F_r * s, axis=1)       # 半径 e_r = (cos, sin)
    ca, sa = np.cos(az), np.sin(az)
    f_r = fx * ca + fy * sa
    f_t = -fx * sa + fy * ca
    return f_t, f_r


def build_truth(dump: Dump, params: dict) -> Truth:
    """ログから各センサの真値（連続量ぶん）を合成する。衝撃はここには入らない。"""
    g = float(params["sim"]["gravity_m_s2"])
    fs = dump.log_rate_hz
    lf = params["sensors"]["accel_lf"]
    hf = params["sensors"]["accel_hf"]
    stn = params["sensors"]["strain"]
    notes: list[str] = []

    F_t, F_r = table_reaction(dump, params)
    notes.append("水平合力は " + (
        "コアの f_tab_*（版 004・符号反転して整合）" if hasattr(dump, "f_tab_x")
        else "コアの F_tab_*（版 002）" if getattr(dump, "has_reaction", False)
        else "ステーションごとの状態から再構成（版 001）"))

    # --- 加速度センサ 2 種 ------------------------------------------------
    def frame_accel(force, sec, name):
        m = float(sec["frame_equiv_mass_kg"])
        fn = float(sec["structure_freq_hz"])
        zs = float(sec["structure_damping"])
        model = str(sec.get("structure_model", "grounded"))
        form = {"grounded": "highpass", "free": "lowpass"}.get(model)
        if form is None:
            raise ValueError(f"structure_model は grounded か free: {model!r}")
        if model == "grounded":
            notes.append(
                f"{name}: 取り付け部を接地とみなし、力→加速度を 2 次系のアクセレランス"
                f"（{fn:.0f} Hz, zeta={zs}）で通した。共振より下は f^2 で落ちる")
        else:
            notes.append(
                f"{name}: 自由な剛体とみなし、2 次系（{fn:.0f} Hz, zeta={zs}）を"
                "直流利得 1 で通した")
        # 架台が受けるのは反力なので符号を反転する
        return chain.second_order(-force / m, fs, fn, zs, form=form)

    a_lf_t = frame_accel(F_t, lf, "accel_lf 接線")
    a_lf_r = frame_accel(F_r, lf, "accel_lf 半径")
    a_hf_r = frame_accel(F_r, hf, "accel_hf 半径")
    # 水平 2 軸なので重力は乗らない（センサは傾かない前提。SENSORS.md 3 節）

    # --- ひずみゲージ（テーブル支持軸の根元）-----------------------------
    Z_mm3 = section_modulus_mm3(float(stn["shaft_outer_diameter_mm"]),
                                float(stn["shaft_inner_diameter_mm"]))
    Z = Z_mm3 * 1e-9                      # m^3
    L_arm = float(stn["arm_length_mm"]) * 1e-3
    E = float(stn["youngs_modulus_GPa"]) * 1e9
    gauge = np.deg2rad(float(stn["gauge_angle_deg"]))
    if bool(stn.get("rotating_gauge", False)):
        ang = dump.th_t + gauge
        notes.append("ひずみゲージは回転する軸に貼った前提（感度方向がテーブルと共に回る）")
    else:
        ang = np.full_like(dump.th_t, gauge)
        notes.append("ひずみゲージは静止側（軸受箱・支持部）に貼った前提。感度方向は世界固定")
    # 感度方向は「センサ方位からの角度」。半径方向を 0 deg とする。
    az_gauge = np.deg2rad(float(params["sensors"]["accel_lf"].get("azimuth_deg", 0.0)))
    # (1) 水平反力による曲げ。力をゲージ方位へ射影して仮の腕を掛ける。
    M_horiz = (F_r * np.cos(ang) + F_t * np.sin(ang)) * L_arm
    # (2) 垂直偏荷重による曲げ。MODEL.md 6 節のとおり、これが支持部の曲げの主成分で
    #     水平反力の 30 倍以上あり、テーブル角に同期して回る（静止ゲージには次数 1 で出る）。
    #     欠品はこの経路に直接乗るので、入れないとひずみは欠品にほぼ盲目になる。
    #     m_bend_x/y はコアが出す世界座標の偏り方向成分（C↔Python 一致を検証済み）。
    #     センサ方位へ回してゲージ方位へ射影する。面上の引張方向＝偏り方向なので、
    #     偏り成分をそのまま射影してよい（物理モーメントは 90 度回した量・MODEL.md 6.1）。
    if hasattr(dump, "m_bend_x"):
        m_r = dump.m_bend_x * np.cos(az_gauge) + dump.m_bend_y * np.sin(az_gauge)
        m_t = -dump.m_bend_x * np.sin(az_gauge) + dump.m_bend_y * np.cos(az_gauge)
        M_vert = m_r * np.cos(ang) + m_t * np.sin(ang)
        notes.append("ひずみは垂直偏荷重の曲げ（m_bend・支持部の主成分）と水平反力の曲げの和")
    else:
        M_vert = np.zeros_like(M_horiz)
        notes.append("この版は m_bend 列が無いので、ひずみは水平反力の曲げのみ（主成分が抜ける）")
    strain = ((M_horiz + M_vert) / Z) / E * 1e6   # ustrain

    if hasattr(dump, "torque_slosh"):              # 004
        T_slosh = dump.torque_slosh.copy()
    elif getattr(dump, "T_slosh", None) is not None:   # 002
        T_slosh = dump.T_slosh.copy()
    else:                                          # 001
        T_slosh = np.zeros_like(dump.t)

    # --- モータ電流 -------------------------------------------------------
    if hasattr(dump, "torque_input"):              # 004: トルクから再構成する
        dr = params["drive"]
        gr = float(dump.header.gear_ratio) or float(dr["gear_ratio"])
        i0 = float(dr["no_load_current_A"])
        kt = float(dr["torque_current_constant_Nm_per_A"])
        T_motor = dump.torque_input / gr           # モータ軸トルク [N m]
        current = np.sqrt(i0 ** 2 + (T_motor / kt) ** 2)
        notes.append(
            f"電流はトルクから再構成した（誘導モータ 1 次近似・励磁 {i0} A・"
            f"k_T {kt} Nm/A）。この運転点ではトルク電流が励磁電流よりずっと小さく、"
            "線電流はほぼ励磁電流で一定になる（電流はトルクの弱い観測量）")
    else:                                          # 001 / 002: コアが出した電流
        current = dump.motor_current.copy()

    return Truth(
        t=dump.t, fs_hz=fs, force_t=F_t, force_r=F_r,
        accel_lf_tangential=a_lf_t, accel_lf_radial=a_lf_r, accel_hf_radial=a_hf_r,
        strain_ustrain=strain, current_A=current,
        torque_slosh_Nm=T_slosh, notes=notes,
    )


def verify_reaction(dump: Dump, params: dict) -> float:
    """コアの反力（``f_tab_*`` / ``F_tab_*``）と、ステーションから独立に組み直した
    合力の食い違い（相対値）。**層をまたいだ整合の不変量**。

    参照とセンサ方位ぶんを**成分ごと**（接線・半径）に引いてから大きさを取る。
    こうすると全体符号の反転（残差 ~2.06・完全逆相）や接線・半径の取り違え
    （~1.4）が残差に残り、捕まえられる。大きさ（hypot）を先に取ってから引くと
    回転不変になって符号も軸も消えてしまう（どちらでも ~0.055 を返す）ので、
    必ず成分で引くこと。正常なら ~0.055 で、この残差は ``np.gradient`` の数値
    微分・離散化・空瓶ホルダの慣性の既知省略ぶんを含む（`整合` ではなく
    「符号・軸を取り違えていない範囲で 5.5% 以内に一致」）。
    版 001（反力を出さない）では NaN。
    """
    has = hasattr(dump, "f_tab_x") or getattr(dump, "has_reaction", False)
    if not has:
        return float("nan")
    az = np.deg2rad(float(params["sensors"]["accel_lf"].get("azimuth_deg", 0.0)))
    ref_t, ref_r = _reaction_from_stations(dump, params, az)
    mine_t, mine_r = table_reaction(dump, params)
    scale = float(np.max(np.hypot(ref_t, ref_r)))
    if scale <= 0:
        return float("nan")
    return float(np.max(np.hypot(mine_t - ref_t, mine_r - ref_r)) / scale)


# ---------------------------------------------------------------------------
# 連鎖を通す
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    """1 チャネルぶんの結果。truth は入力側（ログ周波数）の連続量の真値。"""

    name: str
    unit: str
    fs_hz: float
    t: np.ndarray
    y: np.ndarray
    truth_t: np.ndarray
    truth: np.ndarray
    info: dict

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.y ** 2)))


# チャネル名 → (params のセクション, 単位, Truth の属性)
CHANNELS = {
    "accel_lf_tangential": ("accel_lf", "m/s^2", "accel_lf_tangential"),
    "accel_lf_radial": ("accel_lf", "m/s^2", "accel_lf_radial"),
    "accel_hf_radial": ("accel_hf", "m/s^2", "accel_hf_radial"),
    "strain": ("strain", "ustrain", "strain_ustrain"),
    "current": ("current", "A", "current_A"),
}
IMPACT_CHANNELS = ("accel_lf_tangential", "accel_lf_radial", "accel_hf_radial")


def spec_from_params(params: dict, section: str, g: float) -> chain.ChainSpec:
    """params.json の 1 セクションから :class:`~sensors.chain.ChainSpec` を作る。"""
    p = params["sensors"][section]
    bw = p.get("bandwidth_hz")
    hp = None if p.get("dc_coupled", True) else float(p.get("highpass_hz", 0.0))
    if hp is not None and hp <= 0:
        hp = None
    if "range_g" in p:
        rms = chain.noise_rms_from_density(
            float(p["noise_density_ug_rthz"]), float(bw)) * g
        return chain.ChainSpec(
            sample_rate_hz=float(p["sample_rate_hz"]),
            range_amplitude=float(p["range_g"]) * g,   # ±range_g を m/s^2 に直す
            bits=int(p["bits"]), noise_rms=rms,
            bandwidth_hz=float(bw), highpass_hz=hp)
    if "range_ustrain" in p:
        return chain.ChainSpec(
            sample_rate_hz=float(p["sample_rate_hz"]),
            range_amplitude=float(p["range_ustrain"]),
            bits=int(p["bits"]), noise_rms=float(p["noise_rms_ustrain"]),
            bandwidth_hz=float(bw), highpass_hz=hp)
    if "range_A" in p:
        return chain.ChainSpec(
            sample_rate_hz=float(p["sample_rate_hz"]),
            range_amplitude=float(p["range_A"]),
            bits=int(p["bits"]), noise_rms=float(p["noise_rms_A"]),
            bandwidth_hz=float(bw), highpass_hz=hp)
    raise ValueError(f"レンジの指定が読み取れない: sensors.{section}")


def synthesize(dump: Dump, params: dict, seed: int = 12345,
               truth: Truth | None = None,
               impacts: ImpactEvents | None = None,
               channels: tuple | None = None,
               allow_upsample: bool = True) -> tuple[dict[str, Channel], Truth]:
    """ログ 1 本から各チャネルのセンサ出力を作る。

    同じ seed・同じ入力なら、何度呼んでも同じ配列が返る。
    impacts を渡すと、加速度チャネルにだけ衝撃を合成して足す
    （合成は内部の高い刻みで行い、各センサの帯域制限と間引きを通してから足す）。
    """
    g = float(params["sim"]["gravity_m_s2"])
    if truth is None:
        truth = build_truth(dump, params)
    fs_in = truth.fs_hz
    names = tuple(CHANNELS) if channels is None else tuple(channels)

    extra = None
    fs_hi = None
    if impacts is not None and len(impacts) > 0:
        fs_hi = impact_rate_hz(impacts.ring_freq_hz)
        extra = impact_train(impacts, float(truth.t[0]),
                             float(truth.t[-1] - truth.t[0]), fs_hi)
        truth.notes.append(
            f"衝撃 {len(impacts)} 発を {fs_hi/1000:.1f} kHz で合成して加速度に足した"
            f"（{impacts.ring_freq_hz:.0f} Hz, zeta={impacts.ring_damping}）。{impacts.source}")

    out: dict[str, Channel] = {}
    for name in names:
        section, unit, attr = CHANNELS[name]
        spec = spec_from_params(params, section, g)
        spec.channel = name
        spec.allow_upsample = allow_upsample
        x = getattr(truth, attr)
        if spec.sample_rate_hz > fs_in and allow_upsample:
            truth.notes.append(
                f"{name}: センサ {spec.sample_rate_hz:.0f} Hz > ログ {fs_in:.0f} Hz。"
                " 連続量は補間で埋めた（ログのナイキストより上は入っていない）")
        use_extra = extra if name in IMPACT_CHANNELS else None
        y, info = chain.run_chain(x, fs_in, spec, seed,
                                  extra=use_extra, extra_fs=fs_hi)
        out[name] = Channel(
            name=name, unit=unit, fs_hz=spec.sample_rate_hz,
            t=chain.timebase(len(y), spec.sample_rate_hz, t0=float(truth.t[0])),
            y=y, truth_t=truth.t, truth=x, info=info)
    return out, truth
