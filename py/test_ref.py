"""参照実装そのものの妥当性テスト。

外部の実装には一切依存しない。ここで見るのは「ref.py が params.json rev.3 の
機械（カム式インデックスユニット＋誘導ギヤモータ）の物理をしているか」だけ。
標準ライブラリと pytest しか使わない。

  .venv/bin/python -m pytest py/ -q

期待値は諸元から一意に決まる確定値をそのまま置いてある。
数字が動いたら、params.json を触ったか、式を壊したかのどちらか。
"""

from __future__ import annotations

import math

import pytest

import params as params_mod
import ref


@pytest.fixture(scope="module")
def p():
    return params_mod.load()


# =======================================================================
# 1. カム曲線（変形正弦）
# =======================================================================


def test_ca_cv_are_the_closed_form(p):
    """Ca = 4pi^2/(pi+4)、Cv = Ca/pi。json の丸めた値とも一致すること。"""
    assert ref.MS_CA == pytest.approx(4.0 * math.pi ** 2 / (math.pi + 4.0), rel=1e-15)
    assert ref.MS_CV == pytest.approx(ref.MS_CA / math.pi, rel=1e-15)
    assert ref.MS_CA == pytest.approx(5.528, rel=1e-4)
    assert ref.MS_CV == pytest.approx(1.7596, rel=1e-4)
    # params.json の値（正典）とも一致する
    assert ref.MS_CA == pytest.approx(p.curve_Ca_ref, rel=1e-4)
    assert ref.MS_CV == pytest.approx(p.curve_Cv_ref, rel=1e-4)


def test_modified_sine_ends_are_closed():
    """両端で 変位 0/1、速度 0、加速度 0。停留と滑らかにつながること。"""
    s0, v0, a0 = ref.modified_sine(0.0)
    assert (s0, v0, a0) == (0.0, 0.0, 0.0)
    s1, v1, a1 = ref.modified_sine(1.0)
    assert s1 == pytest.approx(1.0, abs=1e-12)
    assert v1 == 0.0 and a1 == 0.0
    # 端に寄っても速度・加速度が 0 に落ちていく（段差が無い）
    for x in (1e-4, 1.0 - 1e-4):
        _s, v, a = ref.modified_sine(x)
        assert abs(v) < 1e-3
        assert abs(a) < 1e-2
    # 区間外は張り付く
    assert ref.modified_sine(-0.5) == (0.0, 0.0, 0.0)
    assert ref.modified_sine(1.5) == (1.0, 0.0, 0.0)


def test_modified_sine_peaks_are_ca_and_cv():
    """曲線の実際の最大速度・最大加速度が Cv / Ca になること。"""
    n = 200001
    v_max = 0.0
    a_max = 0.0
    for k in range(n):
        _s, v, a = ref.modified_sine(k / (n - 1))
        v_max = max(v_max, abs(v))
        a_max = max(a_max, abs(a))
    assert v_max == pytest.approx(ref.MS_CV, rel=1e-6)
    assert a_max == pytest.approx(ref.MS_CA, rel=1e-6)


def test_modified_sine_is_self_consistent():
    """速度の積分が変位、加速度の積分が速度になること（式の取り違えが出る）。"""
    n = 20000
    dx = 1.0 / n
    s = 0.0
    v = 0.0
    for k in range(n):
        x = (k + 0.5) * dx
        _s, vv, aa = ref.modified_sine(x)
        s += vv * dx
        v += aa * dx
    assert s == pytest.approx(1.0, rel=1e-6)
    assert abs(v) < 1e-6          # 端で速度 0 に戻る = 加速度の積分は 0


def test_modified_sine_is_point_symmetric():
    """s(x) + s(1-x) = 1（加速と減速が対称）。"""
    for x in (0.05, 0.125, 0.3, 0.5, 0.7, 0.875, 0.95):
        s_a, _v, _a = ref.modified_sine(x)
        s_b, _v, _a = ref.modified_sine(1.0 - x)
        assert s_a + s_b == pytest.approx(1.0, abs=1e-12)


# =======================================================================
# 2. 割出しの運動学（カム入力軸角が独立変数）
# =======================================================================


def test_cycle_split_comes_from_the_cam(p):
    """割付 180deg / 停留 180deg なので、割出し時間 = 停留時間 = タクト/2。"""
    ratio = p.index_angle_input / (2.0 * math.pi)
    assert p.index_time == pytest.approx(p.tact * ratio)
    assert p.dwell_time == pytest.approx(p.tact - p.index_time)
    assert p.index_time == pytest.approx(1.5)
    assert p.dwell_time == pytest.approx(1.5)
    # 入力軸は 1 タクトで 1 回転（20 rpm）
    assert p.input_shaft_speed * p.tact == pytest.approx(2.0 * math.pi)
    assert p.input_shaft_speed == pytest.approx(p.output_speed, rel=1e-12)


def test_table_kinematic_peaks(p):
    """諸元の確定値: 最大角速度 0.92132 / 角加速度 1.92964 / 接線加速度 0.43417。"""
    assert ref.table_omega_max(p) == pytest.approx(0.92132, rel=1e-4)
    assert ref.table_alpha_max(p) == pytest.approx(1.92964, rel=1e-4)
    assert ref.tangential_accel_max(p) == pytest.approx(0.43417, rel=1e-4)
    # 44.3 mg。満量ボトルの転倒しきい値からは 1 桁以上遠い
    assert ref.tangential_accel_max(p) / p.g == pytest.approx(0.0443, rel=1e-2)
    # json 側が持っている値とも一致する
    assert ref.table_omega_max(p) == pytest.approx(p.table_omega_max_ref, rel=1e-4)
    assert ref.table_alpha_max(p) == pytest.approx(p.table_alpha_max_ref, rel=1e-4)
    assert ref.tangential_accel_max(p) == pytest.approx(p.tangential_accel_max_ref,
                                                        rel=1e-4)


def test_table_motion_matches_the_peaks(p):
    """時間を刻んで拾った最大値が、閉じた式の最大値と一致すること。"""
    n = 20000
    om = 0.0
    al = 0.0
    for k in range(n + 1):
        psi = 2.0 * math.pi * k / n
        _th, o, a = ref.table_motion(psi, p)
        om = max(om, abs(o))
        al = max(al, abs(a))
    assert om == pytest.approx(ref.table_omega_max(p), rel=1e-5)
    assert al == pytest.approx(ref.table_alpha_max(p), rel=1e-5)


def test_table_stops_during_dwell(p):
    """停留区間ではテーブルが完全に止まっていること（割出し角に到達したまま）。"""
    for psi in (math.pi, 1.2 * math.pi, 1.9 * math.pi, 2.0 * math.pi - 1e-9):
        th, om, al = ref.table_motion(psi, p)
        assert th == pytest.approx(p.index_angle, abs=1e-12)
        assert om == 0.0
        assert al == 0.0
    assert ref.is_index_phase(0.0, p)
    assert ref.is_index_phase(0.99 * math.pi, p)
    assert not ref.is_index_phase(math.pi, p)


def test_table_angle_is_a_function_of_psi_not_time(p):
    """同じ psi なら、何サイクル目でもテーブルの相対角・速度・加速度が同じであること。"""
    for frac in (0.1, 0.37, 0.5, 0.8):
        t1 = frac * p.tact
        t2 = t1 + 5.0 * p.tact
        psi1 = ref.cam_input_angle(t1, p)
        psi2 = ref.cam_input_angle(t2, p)
        assert psi1 == pytest.approx(psi2, abs=1e-9)
        assert ref.table_motion(psi1, p) == pytest.approx(ref.table_motion(psi2, p))


def test_motor_shaft_keeps_turning(p):
    """モータ軸は減速比ぶん速く、テーブルが止まっていても回り続けること。"""
    assert p.motor_speed == pytest.approx(p.input_shaft_speed * p.gear_ratio, rel=1e-3)
    # 1500 rpm = 25 Hz。軸受の故障周波数の土台になる
    assert p.motor_speed / (2.0 * math.pi) == pytest.approx(25.0, rel=1e-3)
    f = p.fault("bearing_outer_race")
    assert f.rotation_freq_hz == pytest.approx(25.0)
    assert f.defect_freq_hz == pytest.approx(f.rotation_freq_hz * f.defect_freq_ratio,
                                             rel=1e-3)
    # 停留のまん中でもモータは進んでいる
    th_a = ref.motor_angle(p.index_time + 0.1, p)
    th_b = ref.motor_angle(p.index_time + 0.2, p)
    assert th_b - th_a == pytest.approx(p.motor_speed * 0.1, rel=1e-12)


def test_bottle_accelerations(p):
    """a_t = Rp*alpha, a_r = -Rp*omega^2。向心加速度は内向きなので a_r は負。"""
    a_t, a_r = ref.bottle_accelerations(2.0, 3.0, p)
    assert a_t == pytest.approx(p.Rp * 3.0)
    assert a_r == pytest.approx(-p.Rp * 4.0)


# =======================================================================
# 3. 慣性とトルク
# =======================================================================


def test_plate_mass_and_polar_inertia(p):
    """テーブル板 15.566 kg、極慣性 0.6102 kg m^2。直径まわりの 1/4 ではないこと。"""
    assert p.plate_mass_from_geometry == pytest.approx(15.566, rel=1e-4)
    assert p.plate_mass_from_geometry == pytest.approx(p.plate_mass, rel=1e-4)
    assert p.polar_inertia_from_geometry == pytest.approx(0.6102, rel=1e-4)
    assert p.bare_inertia == pytest.approx(p.polar_inertia_from_geometry, rel=1e-4)
    # 前版の取り違え（(1/4) m r^2 = 直径まわり）は半分になる。番人として置いておく
    r = p.plate_diameter / 2.0
    wrong = 0.25 * p.plate_mass_from_geometry * r * r
    assert wrong == pytest.approx(0.5 * p.bare_inertia, rel=1e-3)


def test_residence_comes_from_the_process_angles(p):
    """滞留回数が工程角から出ること。供給 315 / 排出 225 なら 6 割出し = 18 秒。

    ホルダ番号を固定で書くと、工程角を動かした瞬間に幾何と食い違う。
    """
    assert math.degrees(p.infeed_angle) == pytest.approx(315.0)
    assert math.degrees(p.fill_angle) == pytest.approx(0.0)
    assert math.degrees(p.discharge_angle) == pytest.approx(225.0)
    assert ref.residence_moves(p) == 6
    assert ref.residence_moves(p) * p.tact == pytest.approx(18.0)
    # 供給ステーションにいるホルダの齢は 0、そこから 1 ピッチごとに 1 ずつ増える
    i_infeed = p.holder_at(p.infeed_angle, 0.0)
    assert ref.holder_age(i_infeed, p) == 0
    assert ref.holder_age((i_infeed + 1) % p.n_stations, p) == 1
    # 排出ステーションにいるホルダの齢は滞留回数と同じ（この停留で抜かれる）
    assert ref.holder_age(p.holder_at(p.discharge_angle, 0.0), p) == 6


def test_steady_state_inertia_and_peak_torque(p):
    """定常状態（満量5 + 空瓶1 + 空ホルダ2）で J_load = 0.71904、ピークトルク 1.38748 Nm。

    数え方は工程配置から一意に決まる。315 に供給されたばかりの空瓶、
    0〜180 が満量 5 本、225 と 270 は排出済みでボトルが無い。
    """
    holders = ref.steady_holders(p)
    assert sum(1 for h in holders if h.has_bottle and h.volume > 0) == 5
    assert sum(1 for h in holders if h.has_bottle and h.volume == 0) == 1
    assert sum(1 for h in holders if not h.has_bottle) == 2
    # 世界角で見ても同じ（テーブル角 0）
    full_angles = sorted(round(math.degrees(p.station_world_angle(i, 0.0)) % 360.0)
                         for i, h in enumerate(holders) if h.volume > 0)
    assert full_angles == [0, 45, 90, 135, 180]
    bare_angles = sorted(round(math.degrees(p.station_world_angle(i, 0.0)) % 360.0)
                         for i, h in enumerate(holders) if not h.has_bottle)
    assert bare_angles == [225, 270]
    empty_bottle = [i for i, h in enumerate(holders)
                    if h.has_bottle and h.volume == 0.0]
    assert len(empty_bottle) == 1
    assert round(math.degrees(
        p.station_world_angle(empty_bottle[0], 0.0)) % 360.0) == 315

    volumes, present = ref.steady_load(p)
    j = ref.load_inertia(volumes, p, present)
    # 内訳: 素の慣性 + (満量5本 + 空瓶1本 = 2.150 kg) * Rp^2
    mass = 5 * (p.bottle_empty_mass + p.rho * p.target_volume) + p.bottle_empty_mass
    assert mass == pytest.approx(2.150, rel=1e-9)
    assert j == pytest.approx(p.bare_inertia + mass * p.Rp ** 2, rel=1e-12)
    assert j == pytest.approx(0.71904, rel=1e-4)
    assert j * ref.table_alpha_max(p) == pytest.approx(1.38748, rel=1e-4)


def test_steady_state_follows_the_process_angles(p):
    """排出角を動かすと定常状態の並びも動くこと。番号を固定していないことの確認。

    排出を 270deg（供給の 1 ステーション手前）に戻すと滞留 7 割出しになり、
    満量 6 本 + 空瓶 1 本 + 空ホルダ 1 つに変わる。前版の固定の並びはこの配置専用だった。
    """
    keep = p.discharge_angle
    try:
        p.discharge_angle = 270.0 * math.pi / 180.0
        assert ref.residence_moves(p) == 7
        holders = ref.steady_holders(p)
        assert sum(1 for h in holders if h.volume > 0) == 6
        assert sum(1 for h in holders if not h.has_bottle) == 1
        volumes, present = ref.steady_load(p)
        assert ref.load_inertia(volumes, p, present) == pytest.approx(0.7406, rel=1e-3)
    finally:
        p.discharge_angle = keep
    assert ref.residence_moves(p) == 6


def test_the_seven_full_bottle_count_is_wrong(p):
    """「満量7 + 空1」で数えると J_load が 6.0% 大きく出ること。

    全ホルダにボトルが載っている前提の数え方で、工程配置とは合わない。
    選定の上限としてなら使えるが、運動方程式には入れない。
    もう一度同じ穴を掘らないように番人として置いておく。
    """
    wrong_volumes = [p.target_volume] * p.n_stations
    wrong_volumes[-1] = 0.0
    j_wrong = ref.load_inertia(wrong_volumes, p)
    assert j_wrong == pytest.approx(0.7621, rel=1e-3)

    volumes, present = ref.steady_load(p)
    j = ref.load_inertia(volumes, p, present)
    assert j / j_wrong == pytest.approx(0.9435, rel=1e-3)
    assert j_wrong * ref.table_alpha_max(p) == pytest.approx(1.4705, rel=1e-3)


def test_load_inertia_counts_only_the_bottles_that_are_there(p):
    """空ホルダは慣性に効かない。中身が増えれば増えること。"""
    empty = ref.load_inertia([0.0] * p.n_stations, p)
    assert empty == pytest.approx(
        p.bare_inertia + p.n_stations * p.bottle_empty_mass * p.Rp ** 2)
    none_there = ref.load_inertia([0.0] * p.n_stations, p,
                                  [False] * p.n_stations)
    assert none_there == pytest.approx(p.bare_inertia)
    full = ref.load_inertia([p.target_volume] * p.n_stations, p)
    assert full > empty


def test_load_inertia_can_leave_the_sloshing_mass_out(p):
    """反力を別に足すときのために、液の剛体分だけ数えられること。"""
    volumes, present = ref.steady_load(p)
    ratio = ref.slosh_mass_ratio(p.R, p.fill_height)
    j_all = ref.load_inertia(volumes, p, present)
    j_rigid = ref.load_inertia(volumes, p, present,
                               liquid_rigid_fraction=1.0 - ratio)
    assert j_rigid < j_all
    # 満量 5 本ぶんの m1 が抜けた分だけ小さい
    lost = 5 * p.rho * p.target_volume * ratio * p.Rp ** 2
    assert j_all - j_rigid == pytest.approx(lost, rel=1e-12)
    # rigid_load_inertia でも同じ値になる（こちらが simulate の使う側）
    assert ref.rigid_load_inertia(volumes, p, present) == pytest.approx(j_rigid,
                                                                       rel=1e-12)
    assert j_rigid == pytest.approx(0.70664, rel=1e-4)


def test_input_shaft_torque_peak(p):
    """入力軸トルクのピークは 慣性負荷だけで 0.40290 Nm、psi = 51.7deg。

    トルクの最大（psi = 22.5deg）と速度の最大（psi = 90deg）は同時に起きない。
    2 つの最大値を掛け合わせると 8 割ほど過大になる。
    """
    bare, psi_bare = ref.input_shaft_torque_peak(p, viscous=False, drag=False)
    assert bare == pytest.approx(0.40290, rel=1e-4)
    assert math.degrees(psi_bare) == pytest.approx(51.7, abs=0.2)

    # 掛け合わせるとこうなる、という比較（この機械では成り立たない数え方）
    volumes, present = ref.steady_load(p)
    j = ref.load_inertia(volumes, p, present)
    naive = (j * ref.table_alpha_max(p) * ref.table_omega_max(p)
             / (p.cam_efficiency * p.input_shaft_speed))
    assert naive / bare == pytest.approx(1.78, rel=1e-2)

    # ピーク機械出力もこの積で決まる
    power = ref.peak_mechanical_power(p, viscous=False)
    assert power == pytest.approx(0.7173, rel=1e-3)
    assert bare == pytest.approx(power / (p.cam_efficiency * p.input_shaft_speed),
                                 rel=1e-3)


def test_friction_is_four_tenths_of_the_input_torque(p):
    """摩擦（効率・引きずり・粘性）の寄与が 4 割あること。全部仮置きの値。"""
    bare, _psi = ref.input_shaft_torque_peak(p, viscous=False, drag=False)
    with_viscous, _psi = ref.input_shaft_torque_peak(p, viscous=True, drag=False)
    full, psi_at = ref.input_shaft_torque_peak(p)

    assert with_viscous == pytest.approx(0.41548, rel=1e-4)
    assert full == pytest.approx(0.71548, rel=1e-4)
    # 引きずり 0.30 Nm だけで全体の 4 割を超える
    assert p.input_drag_torque / full == pytest.approx(0.419, rel=1e-2)
    # 効率 0.85 は負荷ぶんを 1/0.85 に膨らませる
    assert bare * p.cam_efficiency == pytest.approx(0.34247, rel=1e-4)
    # ピークは alpha の最大点（psi = 22.5deg）ではない
    assert math.degrees(psi_at) == pytest.approx(52.4, abs=0.3)
    # 仮置きであることが params.json に書いてあること（消えたら気づけるように）
    assert p.indexer.get("_friction_grade") == "assumed"


def test_table_torque_has_a_viscous_term(p):
    """粘性項が入っていること（前版は摩擦がどこにも無かった）。"""
    j = 0.71904
    t_no_speed = ref.table_torque(j, 0.0, 1.0, p)
    t_with_speed = ref.table_torque(j, 1.0, 1.0, p)
    assert t_with_speed - t_no_speed == pytest.approx(p.table_viscous, rel=1e-12)
    assert p.table_viscous > 0.0


def test_slosh_torque_is_subtracted_not_added(p):
    """準静的極限で、揺動質量が慣性として素直に足されること。

    液がタンクに追従する（a1_t = Rp*alpha）極限では

        T_slosh = -Rp * sum(m1 * a1_t) = -sum(m1) * Rp^2 * alpha

    になる。テーブル軸のトルクはこれを **引く** ので

        T = J_rigid*alpha - T_slosh = (J_rigid + sum(m1)*Rp^2) * alpha = J_all*alpha

    と、液を全量剛体として数えた慣性に戻る。足す向き（前版）だと揺動質量ぶんが
    引かれて J_all より小さくなり、物理として成立しない。
    """
    volumes, present = ref.steady_load(p)
    j_all = ref.load_inertia(volumes, p, present)
    j_rigid = ref.rigid_load_inertia(volumes, p, present)
    m1_total = sum(p.rho * v * ref.slosh_mass_ratio(p.R, p.height_from_volume(v))
                   for v, here in zip(volumes, present) if here and v > 0.0)
    assert j_all - j_rigid == pytest.approx(m1_total * p.Rp ** 2, rel=1e-12)

    alpha = ref.table_alpha_max(p)
    t_slosh = -p.Rp * m1_total * (p.Rp * alpha)
    assert t_slosh < 0.0
    t = ref.table_torque(j_rigid, 0.0, alpha, p, t_slosh=t_slosh)
    assert t == pytest.approx(j_all * alpha, rel=1e-12)
    assert t == pytest.approx(1.38748, rel=1e-4)
    # 符号を逆にすると揺動質量ぶんが 2 回引かれる。ここが前版の誤り
    wrong = ref.table_torque(j_rigid, 0.0, alpha, p, t_slosh=-t_slosh)
    assert wrong < j_rigid * alpha < t


def test_input_torque_efficiency_flips_with_the_power_flow(p):
    """効率の掛け方が T_table*omega の符号で切り替わること（MODEL.md 4.7）。

    カムが負荷を駆動している間は損失ぶん余計に要り（/eta）、負荷がカムを回している
    間は損失ぶん減って伝わる（*eta）。前版は絶対値を取って常に /eta だった。
    """
    r = 0.4399                      # dtheta/dpsi の最大値付近
    driving = ref.input_shaft_torque(1.0, r, p, omega=0.5)
    assert driving == pytest.approx(1.0 * r / p.cam_efficiency + p.input_drag_torque)
    braking = ref.input_shaft_torque(-1.0, r, p, omega=0.5)
    assert braking == pytest.approx(-1.0 * r * p.cam_efficiency + p.input_drag_torque)
    # 減速側を絶対値で数えると 1/0.85^2 = 1.38 倍過大になる
    assert abs(driving - p.input_drag_torque) / abs(braking - p.input_drag_torque) \
        == pytest.approx(1.0 / p.cam_efficiency ** 2, rel=1e-12)
    # omega = 0（停留）では場合分けの境目。変速比が 0 なのでどちらでも引きずりだけ
    assert ref.input_shaft_torque(1.0, 0.0, p, omega=0.0) == pytest.approx(
        p.input_drag_torque)


def test_load_inertia_rate_is_analytic_and_zero_in_practice(p):
    """dJ_load/dt が解析式で出ること。公称条件では omega = 0 なので寄与は厳密に 0。"""
    # 満量近くでは揺れる分の増え方が cosh で潰れるので、ほぼ rho*Q*Rp^2 になる
    dj = ref.load_inertia_rate(p.target_volume, p)
    assert dj == pytest.approx(p.rho * p.flow_rate * p.Rp ** 2, rel=1e-4)
    assert dj == pytest.approx(0.016706, rel=1e-3)
    # 数値差分（J_load を体積で振って差を取る）と一致すること
    d_v = 1.0e-9
    j_a = ref.rigid_mass(p.target_volume - d_v, True, p)
    j_b = ref.rigid_mass(p.target_volume + d_v, True, p)
    num = (j_b - j_a) / (2.0 * d_v) * p.flow_rate * p.Rp ** 2
    assert dj == pytest.approx(num, rel=1e-5)
    # 液が浅いうちは揺れる分の割合が大きく、剛体として増える分は少ない
    assert ref.load_inertia_rate(1.0e-6, p) < 0.2 * dj
    # 停留中（omega = 0）なら項そのものが消える
    assert ref.table_torque(0.7, 0.0, 1.0, p, dj_dt=dj) == pytest.approx(
        ref.table_torque(0.7, 0.0, 1.0, p), rel=1e-15)


def test_input_torque_is_zero_load_during_dwell(p):
    """停留中は変速比が 0 なので、入力軸に残るのは引きずりだけ。"""
    psi = 1.5 * math.pi
    assert ref.table_ratio(psi, p) == 0.0
    t_in = ref.input_shaft_torque(1.0, ref.table_ratio(psi, p), p)
    assert t_in == pytest.approx(p.input_drag_torque)


def test_cam_ratio_integrates_to_the_index_angle(p):
    """dtheta/dpsi を割出し区間で積分すると割出し角になること。"""
    n = 20000
    dpsi = p.index_angle_input / n
    total = 0.0
    for k in range(n):
        total += ref.table_ratio((k + 0.5) * dpsi, p) * dpsi
    assert total == pytest.approx(p.index_angle, rel=1e-6)


# =======================================================================
# 4. スロッシング
# =======================================================================


def bessel_j(order: int, x: float, terms: int = 40) -> float:
    """J0 / J1 を級数で。x が 2 程度なら十分すぎるほど収束する。"""
    half = x / 2.0
    total = 0.0
    for k in range(terms):
        term = ((-1.0) ** k) / (math.factorial(k) * math.factorial(k + order))
        total += term * half ** (2 * k + order)
    return total


def j1_prime(x: float) -> float:
    """J1'(x) = J0(x) - J1(x)/x。"""
    return bessel_j(0, x) - bessel_j(1, x) / x


def test_eps1_is_first_root_of_j1_prime():
    """eps1 = 1.8412 が J1'(x) = 0 の第 1 根であること。"""
    assert abs(j1_prime(ref.EPS1)) < 1.0e-5
    lo, hi = 1.0, 2.5
    assert j1_prime(lo) > 0.0
    assert j1_prime(hi) < 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if j1_prime(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    root = 0.5 * (lo + hi)
    assert root == pytest.approx(1.8411837813, abs=1.0e-9)
    assert abs(ref.EPS1 - root) / root < 1.0e-5


def test_full_bottle_slosh_numbers(p):
    """満量（400 mL, 液深 120.5 mm）で f1 = 3.75136 Hz、T1 = 0.26657 s、L1 = 17.652 mm。

    周期は 1/f1 = 1/3.751360 = 0.2665700 s。0.26661 s は誤り（MODEL.md 2.1 も 0.26657）。
    """
    assert p.fill_height == pytest.approx(0.1205, rel=1e-3)
    w1 = ref.slosh_omega(p.R, p.fill_height, p.g)
    assert w1 == pytest.approx(23.5705, rel=1e-4)
    f1 = w1 / (2.0 * math.pi)
    assert f1 == pytest.approx(3.75136, rel=1e-5)
    period = 2.0 * math.pi / w1
    assert period == pytest.approx(1.0 / f1, rel=1e-15)
    assert period == pytest.approx(0.26657, rel=1e-4)
    assert ref.pendulum_length(w1, p.g) == pytest.approx(17.652e-3, rel=1e-4)


def test_slosh_mass_split(p):
    """揺れる質量 m1 = 0.049 kg（12.3%）、残り m0 = 0.351 kg。"""
    ratio = ref.slosh_mass_ratio(p.R, p.fill_height)
    m_liq = p.rho * p.target_volume
    assert m_liq == pytest.approx(0.400, rel=1e-9)
    assert ratio == pytest.approx(0.123, rel=2e-2)
    assert m_liq * ratio == pytest.approx(0.049, rel=2e-2)
    assert m_liq * (1.0 - ratio) == pytest.approx(0.351, rel=2e-2)


def test_free_oscillation_period_and_decay(p):
    """回転が無ければ、与えた w1 の周期で振動し、zeta どおりに減衰すること。"""
    w1 = ref.slosh_omega(p.R, p.fill_height, p.g)
    zeta = p.zeta
    dt = 1.0e-4
    period = 2.0 * math.pi / (w1 * math.sqrt(1.0 - zeta ** 2))
    st = ref.SloshState(phi_t=0.05)
    series = [st.phi_t]
    for _ in range(int(round(12.0 * period / dt))):
        ref.step_slosh(st, 0.0, 0.0, 0.0, 0.0, w1, zeta, dt, p.g)
        series.append(st.phi_t)

    peaks = [(i * dt, series[i]) for i in range(1, len(series) - 1)
             if series[i] > series[i - 1] and series[i] >= series[i + 1] and series[i] > 0]
    assert len(peaks) >= 10
    gaps = [peaks[i + 1][0] - peaks[i][0] for i in range(len(peaks) - 1)]
    assert sum(gaps) / len(gaps) == pytest.approx(period, rel=1e-3)

    n = len(peaks) - 1
    delta = math.log(peaks[0][1] / peaks[n][1]) / n
    zeta_back = delta / math.sqrt(4.0 * math.pi ** 2 + delta ** 2)
    assert zeta_back == pytest.approx(zeta, rel=5e-3)


def test_static_tilt_is_a_over_g(p):
    """回転していないときのステップ応答は a/g に落ち着くこと（静的には厳密）。"""
    w1 = ref.slosh_omega(p.R, p.fill_height, p.g)
    dt = 1.0e-4
    st = ref.SloshState()
    a = 2.0
    for _ in range(int(round(60.0 * 2 * math.pi / w1 / dt))):
        ref.step_slosh(st, a, 0.0, 0.0, 0.0, w1, 0.2, dt, p.g)
    assert st.phi_t == pytest.approx(-a / p.g, rel=1e-4)


def test_coriolis_splits_the_mode_by_omega(p):
    """回転座標系で解くと固有振動数が w1 ± Omega の 2 本に割れること。

    z = phi_r + i*phi_t と置くと z'' + 2i*Omega*z' + (w1^2 - Omega^2) z = 0。
    根は lambda = w1 - Omega と -(w1 + Omega)。片方だけを励起して周期を測る。
    """
    w1 = ref.slosh_omega(p.R, p.fill_height, p.g)
    omega = ref.table_omega_max(p)
    dt = 2.0e-5

    for sign in (+1.0, -1.0):
        lam = w1 + sign * omega            # +1 -> 逆回転側、-1 -> 共回転側
        st = ref.SloshState(phi_r=1.0e-3, dphi_t=-sign * lam * 1.0e-3)
        series = [st.phi_r]
        n = int(round(8.0 * 2 * math.pi / lam / dt))
        for _ in range(n):
            ref.step_slosh(st, 0.0, 0.0, omega, 0.0, w1, 0.0, dt, p.g)
            series.append(st.phi_r)
        peaks = [i * dt for i in range(1, len(series) - 1)
                 if series[i] > series[i - 1] and series[i] >= series[i + 1]
                 and series[i] > 0]
        assert len(peaks) >= 5
        gaps = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
        measured = sum(gaps) / len(gaps)
        assert measured == pytest.approx(2.0 * math.pi / lam, rel=2e-3)


def test_mode_split_is_wider_than_the_resonance(p):
    """分裂幅が共振の半値幅より大きいこと（＝分裂が結果として見える）。

    分裂幅は Omega の関数で、割出しの間ずっと変わる（停留で 0、ピークで 0.9213 rad/s）。
    したがって「±何 %」という 1 つの数字は、どの Omega を指すかを言わないと決まらない。
    ここでは幅の両端を押さえる。
      Omega = 0            -> 分裂 0
      Omega = 平均 0.5236  -> 0.0833 Hz = 半値幅の 4.4 倍
      Omega = ピーク 0.9213 -> 0.1466 Hz = 半値幅の 7.8 倍
    """
    w1 = ref.slosh_omega(p.R, p.fill_height, p.g)
    half = ref.resonance_half_width_hz(w1, p.zeta)
    assert half == pytest.approx(0.0188, rel=1e-2)

    assert ref.mode_split_hz(0.0) == 0.0

    omega_mean = p.index_angle / p.index_time
    split_mean = ref.mode_split_hz(omega_mean)
    assert omega_mean == pytest.approx(0.5236, rel=1e-3)
    assert split_mean / half == pytest.approx(4.4, rel=2e-2)

    omega_max = ref.table_omega_max(p)
    split_max = ref.mode_split_hz(omega_max)
    assert ref.mode_split_fraction(w1, omega_max) == pytest.approx(0.0391, rel=1e-2)
    assert split_max == pytest.approx(0.1466, rel=1e-2)
    assert split_max / half == pytest.approx(7.8, rel=2e-2)

    # どの Omega をとっても半値幅より広い（＝コリオリを落とすと結果が変わる）
    for omega in (0.3 * omega_max, 0.6 * omega_max, omega_max):
        assert ref.mode_split_hz(omega) > half


def test_accidental_resonance_is_gone_at_tact_3s(p):
    """タクト 3.0 秒では、最寄りの高調波との離調が半値幅の 4.5 倍あること。"""
    w1 = ref.slosh_omega(p.R, p.fill_height, p.g)
    f1 = w1 / (2.0 * math.pi)
    half = ref.resonance_half_width_hz(w1, p.zeta)

    n, f_h, detune = ref.nearest_harmonic_detuning(f1, p.tact)
    assert n == 11
    assert f_h == pytest.approx(3.6667, rel=1e-4)
    assert detune == pytest.approx(0.0847, rel=1e-2)
    assert detune / half == pytest.approx(4.5, rel=2e-2)

    # 前版のタクト 2.4 秒では 9 次が 3.75000 Hz で、f1 とほぼ完全に一致していた。
    # 動作点が偶然の共振で決まっていた状態。
    n_old, f_old, detune_old = ref.nearest_harmonic_detuning(f1, 2.4)
    assert n_old == 9
    assert f_old == pytest.approx(3.75, rel=1e-9)
    assert detune_old < 0.002
    assert detune_old / half < 0.1
    assert detune / detune_old > 50.0


def test_index_slosh_peak_and_residual(p):
    """割出し 1 回で ピーク 68.07 mrad、停留に入ってからの残留が残ること。

    残留はコリオリを入れるかどうかで変わる。
      入れない（前版と同じ 2 方向独立）: 38.45 mrad
      入れる（こちらが正）:             35.31 mrad
    ピークのほうは割出し中の強制応答が支配的なので、両者でほとんど動かない。

    コリオリ入りの残留は、別の担当の再積分では 35.29 mrad。こちらは 35.31 mrad で、
    差は 0.06%（積分の刻みと減衰項の掛け方の違いの範囲）。合わせ込みはしていない。
    """
    peak_c, dwell_c, _end = ref.index_slosh_response(p, coriolis=True, dt=1e-4)
    peak_n, dwell_n, _end = ref.index_slosh_response(p, coriolis=False, dt=1e-4)

    assert peak_c * 1e3 == pytest.approx(68.07, rel=2e-3)
    assert peak_n * 1e3 == pytest.approx(68.07, rel=2e-3)
    assert abs(peak_c - peak_n) < 1e-4          # ピークは 5e-5 rad しか違わない
    assert dwell_n * 1e3 == pytest.approx(38.45, rel=2e-3)
    assert dwell_c * 1e3 == pytest.approx(35.30, rel=3e-3)
    # コリオリで共振から外れるぶん、残留は 8% 小さくなる
    assert dwell_c < dwell_n
    assert dwell_c / dwell_n == pytest.approx(0.918, rel=1e-2)


def test_peak_tilt_stays_inside_the_linear_range(p):
    """ピーク傾き 68.07 mrad は dz/R = 0.068 で、限界 0.2 に十分収まること。"""
    peak, _dwell, _end = ref.index_slosh_response(p, dt=1e-4)
    dz_over_r = math.tan(peak)
    assert dz_over_r == pytest.approx(0.068, rel=1e-2)
    assert dz_over_r < ref.TILT_INVALID_RATIO
    assert ref.linearity_flag(peak) == "ok"
    # 壁での液面上下は 2.2 mm。頭上空間 29.5 mm に対して余裕がある
    assert ref.wall_rise(p.R, peak) == pytest.approx(2.2e-3, rel=5e-2)


def test_static_tilt_alone_would_be_44_mrad(p):
    """静的つり合いだけなら 44.3 mrad。動的な行き過ぎで 1.54 倍になること。"""
    static = ref.tangential_accel_max(p) / p.g
    assert static * 1e3 == pytest.approx(44.27, rel=1e-3)
    peak, _dwell, _end = ref.index_slosh_response(p, dt=1e-4)
    assert peak / static == pytest.approx(1.54, rel=2e-2)


def test_slosh_reaction_force_uses_m1_and_m0(p):
    """液の水平力が -(m0*a_tank + m1*a1) になっていること。

    返すのは反作用（テーブルが受ける側。MODEL.md 2.4 の F_react）で、
    液を動かすのに要る力 F_hold とは符号が逆。T_slosh と向きが揃う。
    """
    w1 = ref.slosh_omega(p.R, p.fill_height, p.g)
    L1 = ref.pendulum_length(w1, p.g)
    st = ref.SloshState(phi_t=0.05, phi_r=-0.01)
    a_t, a_r = 0.4, -0.05
    f_t, f_r = ref.liquid_force(st, p.target_volume, a_t, a_r, w1, p.zeta, L1, p)

    m_liq = p.rho * p.target_volume
    m1 = m_liq * ref.slosh_mass_ratio(p.R, p.fill_height)
    m0 = m_liq - m1
    a1_t, a1_r = ref.slosh_accel(st, w1, p.zeta, L1, p.g)
    assert f_t == pytest.approx(-(m0 * a_t + m1 * a1_t))
    assert f_r == pytest.approx(-(m0 * a_r + m1 * a1_r))
    # 揺れていなければ剛体と同じ大きさで、向きは加速度と逆
    quiet = ref.SloshState()
    f_t0, _f_r0 = ref.liquid_force(quiet, p.target_volume, a_t, a_r, w1, p.zeta, L1, p)
    assert f_t0 == pytest.approx(-m0 * a_t)
    assert f_t0 * a_t < 0.0

    # 反力トルクと同じ向きに揃っていること。準静的に追従している液
    # （a1 = a_tank）なら、接線の反力は -m_liq*a_t、トルクは -Rp*m1*a_t で同符号
    t_slosh = -p.Rp * m1 * a_t
    f_follow, _r = ref.liquid_force(ref.SloshState(phi_t=-a_t / p.g), p.target_volume,
                                    a_t, a_r, w1, p.zeta, L1, p)
    assert f_follow < 0.0 and t_slosh < 0.0


# =======================================================================
# 5. 垂直荷重経路
# =======================================================================


def test_uniform_load_cancels(p):
    """全ホルダが同じなら、重量の偏りは打ち消し合って 0 になること。"""
    volumes = [p.target_volume] * p.n_stations
    _mx, _my, m = ref.vertical_bending_moment(volumes, 0.0, p)
    assert m < 1e-12


def test_one_missing_bottle_leaves_a_full_bottle_of_moment(p):
    """欠品 1 本で満量 1 本ぶん（0.883 Nm）の曲げが残ること。"""
    volumes = [p.target_volume] * p.n_stations
    present = [True] * p.n_stations
    present[3] = False
    _mx, _my, m = ref.vertical_bending_moment(volumes, 0.0, p, present)
    expected = (p.bottle_empty_mass + p.rho * p.target_volume) * p.g * p.Rp
    assert m == pytest.approx(expected, rel=1e-9)
    assert m == pytest.approx(0.938, rel=1e-2)      # 空瓶 25g 込み
    # 液だけで数えれば 0.883 Nm
    liquid_only = p.rho * p.target_volume * p.g * p.Rp
    assert liquid_only == pytest.approx(0.8826, rel=1e-3)


def test_four_consecutive_full_bottles_give_2306_mNm(p):
    """満量 4 本が連続したときの曲げモーメント 2.306 Nm。"""
    volumes = [0.0] * p.n_stations
    present = [True] * p.n_stations
    for i in (0, 1, 2, 3):
        volumes[i] = p.target_volume
    # 4 本の重心方位を基準にとる（-67.5, -22.5, +22.5, +67.5 deg）
    th_t = -1.5 * p.station_pitch
    _mx, _my, m = ref.vertical_bending_moment(volumes, th_t, p, present)
    # 空瓶の質量は 8 方位に均等なので和に残らない。液だけが残る
    weight = 2.0 * (math.cos(math.radians(22.5)) + math.cos(math.radians(67.5)))
    assert weight == pytest.approx(2.613, rel=1e-3)
    assert m == pytest.approx(weight * p.rho * p.target_volume * p.g * p.Rp, rel=1e-9)
    assert m == pytest.approx(2.306, rel=1e-3)


def test_vertical_path_is_32_times_the_horizontal_one(p):
    """重量の曲げが水平反力の曲げの 31.8 倍あること。

    比較の土俵は「テーブル軸に働く曲げモーメント」。同じ通し（2 タクト、dt = 0.5 ms）
    から、重量経路のピーク 2.3795 N·m と水平合力のピーク 0.4988 N を取り、
    水平のほうにはアーム長 150 mm（params.json の sensors.strain.arm_length_mm）を掛ける。

    **アーム長はセンサ側の値で、機械の重心高さとして確かめたものではない。**
    取り付けも未決なので、この比は桁の話として読むこと。
    以前ここに書いていた 38.4 倍は、定常が「満量 4 本 + 空 4 本」だった旧機の値で、
    いまの工程配置（満量 5 本 + 空瓶 1 本 + 空ホルダ 2 つ）には当てはまらない。
    """
    arm = p.raw["sensors"]["strain"]["arm_length_mm"] * 1e-3
    assert arm == pytest.approx(0.150)

    run = ref.simulate(p, n_cycles=2, dt=5.0e-4, log_rate=200.0)
    m_weight = max(r.m_bend for r in run.records)
    f_peak = max(math.hypot(r.f_tab_x, r.f_tab_y) for r in run.records)
    assert m_weight == pytest.approx(2.3795, rel=1e-3)
    assert f_peak == pytest.approx(0.4987, rel=1e-3)
    assert m_weight / (f_peak * arm) == pytest.approx(31.8, rel=1e-2)

    # 曲げは 0 に落ちない。定常のアンバランスがずっと乗っている
    assert min(r.m_bend for r in run.records) == pytest.approx(2.2249, rel=1e-3)

    # モデルから出るスロッシング力だけ（満量 1 本、ピーク傾き）で見るとさらに 2 桁小さい。
    # 水平合力の大半は液の m0（凍った分）と空瓶の慣性で、揺れそのものではない。
    m1 = p.rho * p.target_volume * ref.slosh_mass_ratio(p.R, p.fill_height)
    peak, _d, _e = ref.index_slosh_response(p, dt=2e-4)
    f_one = m1 * p.g * peak
    assert f_one == pytest.approx(0.0327, rel=1e-2)
    assert m_weight / (f_one * arm) == pytest.approx(485.0, rel=2e-2)


def test_bending_moment_direction_follows_the_table(p):
    """偏りの向きがテーブルと一緒に回ること（回転同期成分になる）。"""
    volumes = [p.target_volume] * p.n_stations
    volumes[0] = 0.0
    mx0, my0, m0 = ref.vertical_bending_moment(volumes, 0.0, p)
    mx1, my1, m1 = ref.vertical_bending_moment(volumes, p.station_pitch, p)
    assert m0 == pytest.approx(m1, rel=1e-12)
    ang0 = math.atan2(my0, mx0)
    ang1 = math.atan2(my1, mx1)
    assert (ang1 - ang0) % (2 * math.pi) == pytest.approx(p.station_pitch, rel=1e-9)


def test_jet_load_adds_to_the_moment(p):
    """充填ジェットを鉛直荷重として足せること。"""
    volumes = [p.target_volume] * p.n_stations
    f = ref.jet_force(p)
    _mx, _my, m = ref.vertical_bending_moment(
        volumes, 0.0, p, extra=[(p.fill_angle, f)])
    assert m == pytest.approx(f * p.Rp, rel=1e-9)


# =======================================================================
# 6. 可変質量の運動量
# =======================================================================


def test_nozzle_velocity_and_jet_force(p):
    """流速 2.144 m/s、液柱の力 rho*Q*v = 0.707 N。"""
    assert p.nozzle_velocity == pytest.approx(2.144, rel=1e-3)
    assert p.nozzle_velocity == pytest.approx(p.nozzle_velocity_ref, rel=1e-3)
    f = ref.jet_force(p)
    assert f == pytest.approx(p.rho * p.flow_rate * p.nozzle_velocity, rel=1e-12)
    assert f == pytest.approx(0.7074, rel=1e-3)
    # 満量ボトル 1 本の重量 3.92 N に対して 18%。無視できる大きさではない
    assert f / (p.rho * p.target_volume * p.g) == pytest.approx(0.180, rel=1e-2)


def test_jet_force_grows_with_fall_height(p):
    """落差を入れれば着液速度が上がって力が増えること（既定は落差 0）。"""
    f0 = ref.jet_force(p, 0.0)
    f1 = ref.jet_force(p, 0.05)
    assert f1 > f0
    v = math.sqrt(p.nozzle_velocity ** 2 + 2 * p.g * 0.05)
    assert f1 == pytest.approx(p.rho * p.flow_rate * v, rel=1e-12)


# =======================================================================
# 7. 充填のタイミング
# =======================================================================


def test_fill_fits_in_the_dwell(p):
    """充填 1.212 s + 弁の開閉遅れ 60 ms が停留 1.5 s に収まること。"""
    assert p.fill_duration == pytest.approx(1.2121, rel=1e-3)
    needed = p.start_delay + p.valve_open_delay + p.fill_duration + p.valve_close_delay
    assert needed < p.dwell_time
    assert p.dwell_time - needed == pytest.approx(0.178, rel=1e-2)


def test_close_command_anticipates_only_the_nominal_delay(p):
    """公称の閉じ遅れだけを先読みすること。故障ぶんの遅れは先読みしない。

    制御側は弁が劣化して遅くなったことを知らないので、先読みできるのは
    fill.valve_close_delay_s だけ。ここを故障ぶんまで先読みしていると、
    閉じ遅れ故障を入れても充填量が変わらず、故障として成立しない。
    """
    v_cmd = ref.close_command_volume(p)
    assert v_cmd < p.target_volume
    assert v_cmd + p.flow_rate * p.valve_close_delay == pytest.approx(p.target_volume)
    # 引数は Params だけ。追加遅れを渡す口が無いこと自体が仕様
    with pytest.raises(TypeError):
        ref.close_command_volume(p, 0.15)


def test_fill_rate_matches_flow_rate(p):
    dt = 1.0e-4
    volume = 0.0
    n = int(round(0.5 * p.fill_duration / dt))
    for _ in range(n):
        volume = ref.step_fill(volume, True, dt, p)
    assert volume == pytest.approx(p.flow_rate * n * dt, rel=1e-12)
    assert ref.step_fill(volume, False, dt, p) == volume


def test_target_volume_leaves_headroom(p):
    """400 mL が円筒部に収まり、頭上空間が 29 mm 残ること。"""
    h = p.height_from_volume(p.target_volume)
    assert h < p.body_height
    assert p.body_height - h == pytest.approx(0.0295, rel=1e-2)


def test_spill_needs_a_tilt_far_outside_the_linear_range(p):
    """こぼれるには線形範囲を大きく外れた傾きが要ること。"""
    _left, spilled = ref.apply_spill(p.target_volume, 0.07, p.R,
                                     p.cross_section, p.body_height)
    assert spilled == 0.0
    headroom = p.body_height - p.height_from_volume(p.target_volume)
    tilt_needed = math.atan(headroom / p.R)
    assert tilt_needed > ref.TILT_INVALID_RATIO
    _left, spilled = ref.apply_spill(p.target_volume, tilt_needed * 1.01, p.R,
                                     p.cross_section, p.body_height)
    assert spilled > 0.0


def test_linearity_flag_thresholds():
    assert ref.linearity_flag(0.0) == "ok"
    assert ref.linearity_flag(0.068) == "ok"
    assert ref.linearity_flag(0.15) == "warn"
    assert ref.linearity_flag(0.5) == "invalid"


def test_dt_warning_triggers_only_when_coarse(p):
    assert ref.dt_warning(p) is None
    fine = p.dt
    try:
        p.dt = 0.05
        assert ref.dt_warning(p) is not None
    finally:
        p.dt = fine


# =======================================================================
# 8. 受け渡しと故障（イベント）
# =======================================================================


def test_transfer_events_are_at_the_right_stations(p):
    """停留の先頭で 225deg のホルダが排出、315deg のホルダが受け取ること。"""
    result = ref.simulate(p, n_cycles=2, dt=5.0e-4, log_rate=100.0)
    kinds = [e.kind for e in result.events]
    assert kinds.count("discharge") == 2
    assert kinds.count("infeed") == 2

    for e in result.events:
        # 受け渡しは停留の先頭（割出しが終わった直後）に起きる
        phase = e.t % p.tact
        assert phase == pytest.approx(p.index_time, abs=1.0e-3)
        th_t = (int(e.t // p.tact) + 1) * p.index_angle
        angle = p.station_world_angle(e.station, th_t) % (2 * math.pi)
        if e.kind == "discharge":
            assert angle == pytest.approx(p.discharge_angle, abs=1e-6)
            assert e.data["volume_m3"] == pytest.approx(p.target_volume, rel=1e-3)
            # 排出は連続量ではなくイベント。傾きと角速度を持ち出す
            assert "tilt_rad" in e.data and "dphi_t_rad_s" in e.data
        if e.kind == "infeed":
            assert angle == pytest.approx(p.infeed_angle, abs=1e-6)
            assert e.data["mass_kg"] == pytest.approx(p.bottle_empty_mass)


def test_residence_is_six_index_moves(p):
    """1 本のボトルが供給されてから排出されるまで割出し 6 回 = 18 秒。

    供給 315deg -> 充填 0deg で 1 回、そこから排出 225deg まで 5 回。
    工程角から出る値（residence_moves）と、通しで測った値が一致すること。
    """
    assert ref.residence_moves(p) == 6
    result = ref.simulate(p, n_cycles=9, dt=2.0e-3, log_rate=20.0)
    infeeds = {e.station: e.t for e in result.events if e.kind == "infeed"}
    discharges = [(e.t, e.station) for e in result.events if e.kind == "discharge"]
    matched = 0
    for t_out, station in discharges:
        if station in infeeds and infeeds[station] < t_out:
            assert t_out - infeeds[station] == pytest.approx(6.0 * p.tact, abs=1e-2)
            assert t_out - infeeds[station] == pytest.approx(18.0, abs=1e-2)
            matched += 1
    assert matched >= 1


def test_bearing_impulses_are_on_the_motor_shaft(p):
    """外輪傷のイベントが 89.5 Hz で出て、1 タクトに 268 回入ること。"""
    times = ref.bearing_impulse_times(p, 0.0, p.tact)
    f = p.fault("bearing_outer_race")
    assert len(times) == pytest.approx(f.defect_freq_hz * p.tact, abs=1)
    assert len(times) == pytest.approx(268, abs=1)
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    assert min(gaps) == pytest.approx(1.0 / f.defect_freq_hz, rel=1e-9)
    # テーブルが止まっている停留の間にも出続ける（前版はここが止まっていた）
    dwell_hits = [t for t in times if t > p.index_time]
    assert len(dwell_hits) > 100


def test_removed_faults_are_gone(p):
    """この機構では成り立たない故障が params.json から消えていること。"""
    assert "backlash" not in p.faults
    assert "fill_imbalance" not in p.faults
    assert p.indexer.backlash_deg == 0.0
    assert set(p.faults) == {"missing_bottle", "valve_close_delay",
                             "cam_follower_wear", "bearing_outer_race"}


def test_missing_bottle_is_a_whole_full_bottle(p):
    """欠品は満量 400 g が丸ごと欠けること（前版の 30% 不足の 3.3 倍）。"""
    f = p.fault("missing_bottle")
    missing = p.rho * p.target_volume
    assert missing == pytest.approx(0.400, rel=1e-9)
    old_style = 0.30 * missing
    assert missing / old_style == pytest.approx(3.33, rel=1e-2)
    assert 0 <= int(f.station) < p.n_stations


def test_valve_close_delay_overfills_every_bottle(p):
    """閉じ遅れ 0.15 s + 液垂れ 3 mL で、全ボトルが等しく過充填になること。

    通しで回して実際に 449.5 mL になることまで見る。ここが 400 mL のままなら、
    閉じ命令が故障ぶんの遅れまで先読みしてしまっていて、故障として成立していない。
    """
    f = p.fault("valve_close_delay")
    extra = p.flow_rate * f.extra_delay_s
    assert extra == pytest.approx(49.5e-6, rel=1e-3)
    assert (extra + f.drip_volume_mL * 1e-6) / p.target_volume == pytest.approx(
        0.131, rel=1e-2)

    fault = p.raw["faults"]["valve_close_delay"]
    keep = fault["enabled"]
    try:
        fault["enabled"] = True
        p_fault = params_mod.Params(p.raw, p.path)
        bad = ref.simulate(p_fault, n_cycles=3, dt=2.0e-4, log_rate=50.0)
    finally:
        fault["enabled"] = keep

    filled = [v for v in bad.records[-1].volume if v > 0.5 * p.target_volume]
    overfilled = [v for v in filled if v > 1.05 * p.target_volume]
    # 故障を入れてから充填されたボトルだけが過充填。3 タクトぶんで 3 本
    assert len(overfilled) == 3
    for v in overfilled:
        assert v == pytest.approx(p.target_volume + extra, rel=2e-3)
        assert v * 1e6 == pytest.approx(449.5, rel=2e-3)
    # 液垂れイベントも出る（割出し中のテーブル上に落ちる分）
    assert sum(1 for e in bad.events if e.kind == "valve_drip") == 3


# =======================================================================
# 9. 通しで回す
# =======================================================================


def test_simulate_runs_and_logs(p):
    """1 サイクル通して回り、ログが出て、諸元どおりの値が並ぶこと。"""
    result = ref.simulate(p, n_cycles=1, dt=2.0e-4, log_rate=200.0)
    assert len(result.records) > 500
    assert result.warnings == []
    last = result.records[-1]
    assert last.t == pytest.approx(p.tact, abs=0.01)

    # テーブルは 1 サイクルで 45deg 進んで止まっている
    assert last.th_t == pytest.approx(p.index_angle, rel=1e-9)
    assert last.omega == 0.0
    # モータ軸は回り続けている（1 タクトで 25 回転）
    assert last.th_m / (2 * math.pi) == pytest.approx(25.0 * p.tact, rel=1e-2)
    # 停留中の入力軸トルクは引きずりだけ
    assert last.torque_input == pytest.approx(p.input_drag_torque, rel=1e-6)
    assert len(last.volume) == p.n_stations
    assert len(last.present) == p.n_stations
    assert all(isinstance(x, bool) for x in last.present)


def test_simulate_torque_peaks(p):
    """通しで拾ったトルクのピークが、閉じた式の値と合うこと。

    定常（2 タクト目以降）のテーブル軸ピークは 1.42007 N·m。剛体換算の
    1.38748 N·m（液を全量剛体として数えた J_load * alpha_max）より 2.3% 大きい。
    内訳はピークの瞬間で

        J_load(m0 だけ) * alpha = 1.36353
        c_visc * omega          = 0.01147
        -T_slosh                = 0.04507
        合計                    = 1.42007

    J から抜いた m1 ぶん（0.02393 N·m）が -T_slosh として 0.04507 で戻る。
    1.9 倍になっているのは振り子が行き過ぎるぶんで、準静的なら同じ値に戻る。
    """
    result = ref.simulate(p, n_cycles=2, dt=2.0e-4, log_rate=1000.0)
    t_table = max(abs(r.torque_table) for r in result.records)
    t_input = max(r.torque_input for r in result.records)
    volumes, present = ref.steady_load(p)
    j = ref.load_inertia(volumes, p, present)
    rigid_peak = j * ref.table_alpha_max(p)
    assert rigid_peak == pytest.approx(1.38748, rel=1e-4)
    assert t_table == pytest.approx(1.42007, rel=1e-4)
    # 剛体換算より大きい。符号を逆にすると小さくなるので、ここで向きが効く
    assert t_table > rigid_peak
    assert t_table / rigid_peak == pytest.approx(1.0235, rel=1e-3)

    peak_rec = max(result.records, key=lambda r: abs(r.torque_table))
    assert (peak_rec.j_load * peak_rec.alpha
            + p.table_viscous * peak_rec.omega
            - peak_rec.torque_slosh) == pytest.approx(peak_rec.torque_table,
                                                      rel=1e-12)
    assert peak_rec.torque_slosh < 0.0        # 減速側ではなく加速側のピーク

    peak, _psi = ref.input_shaft_torque_peak(p, volumes, present)
    assert t_input == pytest.approx(peak, rel=1e-1)


def test_simulate_fills_to_target(p):
    """停留のあいだに 1 本ぶんが目標体積まで入ること。

    1 サイクルで、満量 1 本が抜け（排出）、空 1 本が入り（供給）、
    前のサイクルで入った空 1 本が満量になる。満量の数は 5 本のまま釣り合う。
    """
    result = ref.simulate(p, n_cycles=1, dt=1.0e-4, log_rate=100.0)
    last = result.records[-1]
    filled = [v for v in last.volume if v > 0.5 * p.target_volume]
    assert len(filled) == 5
    for v in filled:
        assert v == pytest.approx(p.target_volume, rel=2e-3)
    # 在荷は 6（満量 5 + 空瓶 1）。空ホルダ 2 つとは V = 0 では区別できない
    assert sum(1 for x in last.present if x) == 6
    empty_bottles = [i for i, x in enumerate(last.present)
                     if x and last.volume[i] == 0.0]
    no_bottle = [i for i, x in enumerate(last.present) if not x]
    assert len(empty_bottles) == 1
    assert len(no_bottle) == 2
    # 空瓶と欠品はどちらも V = 0。present が無いと分けられない
    assert last.volume[empty_bottles[0]] == 0.0
    assert all(last.volume[i] == 0.0 for i in no_bottle)


def test_simulate_keeps_the_vertical_load_path(p):
    """曲げモーメントが常に出ていて、桁が 1 Nm 台であること。"""
    result = ref.simulate(p, n_cycles=1, dt=5.0e-4, log_rate=100.0)
    m = [r.m_bend for r in result.records]
    assert min(m) > 1.0
    assert max(m) < 3.0
    # 前版はこの経路自体が無く、水平力しか無かった。水平力より 1 桁以上大きい
    f_h = max(math.hypot(r.f_tab_x, r.f_tab_y) for r in result.records)
    assert max(m) > 10.0 * f_h * 0.150


def test_simulate_stays_in_the_linear_range(p):
    """定常状態を数サイクル回しても、傾きが適用範囲に収まること。

    前のサイクルの残りが次の割出しに乗るので、1 回だけの 68 mrad より大きくなる。
    それでも dz/R は 0.1 に届かない。
    """
    result = ref.simulate(p, n_cycles=3, dt=5.0e-4, log_rate=100.0)
    assert result.linearity == "ok"
    assert result.max_tilt > 0.068          # 単発より大きい
    assert math.tan(result.max_tilt) < ref.TILT_INVALID_RATIO
    assert result.max_tilt * 1e3 == pytest.approx(89.6, rel=5e-2)


def test_faster_index_leaves_the_linear_range(p):
    """割出しを詰めると適用範囲を外れ、警告に出ること。

    接線加速度は割出し時間の 2 乗で効くので、速くすれば必ずどこかで外れる。
    カム機構では割出し時間はタクトの半分に固定なので、これはタクトを詰める話になる。
    """
    keep = (p.tact, p.index_time)
    try:
        p.tact = 0.8
        p.index_time = 0.4
        result = ref.simulate(p, n_cycles=1, dt=5.0e-5, log_rate=100.0)
    finally:
        p.tact, p.index_time = keep
    assert math.tan(result.max_tilt) > ref.TILT_INVALID_RATIO
    assert result.linearity == "invalid"
    assert any("線形" in w for w in result.warnings)


def test_missing_bottle_fault_shows_up_in_the_moment(p):
    """欠品を有効にすると、曲げモーメントの偏りが大きくなること。

    対象ホルダが供給位置（315deg）に来るのは 4 サイクル目なので、そこまで回す。
    """
    n_cycles = 5
    fault = p.raw["faults"]["missing_bottle"]
    keep = fault["enabled"]
    try:
        fault["enabled"] = True
        p_fault = params_mod.Params(p.raw, p.path)
        bad = ref.simulate(p_fault, n_cycles=n_cycles, dt=1.0e-3, log_rate=50.0)
    finally:
        fault["enabled"] = keep
    good = ref.simulate(p, n_cycles=n_cycles, dt=1.0e-3, log_rate=50.0)

    m_bad = max(r.m_bend for r in bad.records)
    m_good = max(r.m_bend for r in good.records)
    assert m_bad > m_good
    assert any(e.kind == "infeed_missed" for e in bad.events)
