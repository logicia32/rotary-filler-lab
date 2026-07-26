"""カム式ロータリー充填機の参照実装。

このファイルがラボの仕様書。C コア（core/twin.c）はここに突き合わせて検証する。
速度は一切考えていない。純 Python のループで、式が目で追えることだけを優先する。
数式の解釈が割れたらこのファイルが正。

方針:
  - 途中式をまとめない。1 行に 1 つの意味だけ置く。
  - 数値は params.py 経由でしか入れない。ここに物性やパラメータを直書きしない。
    例外は「式そのものから出る定数」（変形正弦の Ca/Cv、ベッセルの根 eps1）だけで、
    どれも params.json の値と一致することをテストで見ている。
  - params.json が決めていない所は「取り決め」と書いて既定値をここに置く。
    C 側と数字を合わせないと突き合わせが成立しないので、動かすときは両方直すこと。

機械の姿（params.json rev.3）:
  - 割出しはカム式インデックスユニット。入力軸 1 回転でテーブルが 1 ステーション進む。
    入力軸は 20 rpm で連続回転し、割付 180deg / 停留 180deg。起動停止が無い。
  - テーブル角は入力軸角 psi の関数で、時間の関数ではない。カム曲線は変形正弦。
  - 駆動は誘導ギヤモータ + インバータの開ループ。位置・速度・電流ループは無い。
  - 単ノズル。供給 315deg / 充填 0deg / 排出 225deg。滞留は割出し 6 回 = 18 秒。
    排出を供給の 2 ステーション先に置くのは、1 ステーション（45deg）だと供給側と
    排出側の星車が抱えるボトルどうしが当たるため（params.json stations._fix_note）。

主張の範囲:
  実機の実測ではない。合わせ込みもしていない。言えるのは
  「この物理とこの諸元の下ではこうなる」までで、それ以上は書かない。
  摩擦（カム効率・引きずり・粘性）は出所が無い仮置きの値で、結果への寄与は小さくない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import params as params_mod

# --- スロッシングの定数 -------------------------------------------------

# J1'(x) = 0 の第 1 根。
# 厳密値は 1.8411837813406595 で、1.8412 との差は相対 8.8e-6。
# w1 は eps1 の平方根で効くので、固有振動数への影響は相対 4.4e-6（3.75 Hz に対して 1.6e-5 Hz）。
EPS1 = 1.8412

# 線形スロッシング理論が使える振幅の目安（NOTES.md に出典）。
# 壁での液面上下動 dz が R の何倍まで、という形で持つ。
TILT_WARN_RATIO = 0.1    # これを超えると減衰の振幅非依存性が崩れ始める
TILT_INVALID_RATIO = 0.2 # これを超えると波が砕ける・回るなどで線形モデルの外


# =======================================================================
# 1. カム曲線（変形正弦）
# =======================================================================
#
# 加速度を 3 区間の正弦でつないだ曲線。無次元時間 x = 0..1 に対し
#
#   0    .. 1/8 : a = Ca*sin(4pi*x)          （0 から正の山へ）
#   1/8  .. 7/8 : a = Ca*cos(4pi(x-1/8)/3)   （正の山から負の谷へ）
#   7/8  .. 1   : a = -Ca*cos(4pi(x-7/8))    （負の谷から 0 へ）
#
# 変位が 0 から 1 になるよう振幅を決めると、その振幅がそのまま無次元最大加速度
#
#   Ca = 4*pi^2 / (pi + 4) = 5.52796
#   Cv = Ca / pi           = 1.75960
#
# になる。両端で 変位 0/1、速度 0、加速度 0。だから停留と滑らかにつながる。
# params.json の indexer.curve_Ca / curve_Cv（5.528 / 1.7596）はこれを丸めたもの。

MS_CA = 4.0 * math.pi ** 2 / (math.pi + 4.0)
MS_CV = MS_CA / math.pi


def modified_sine(x: float) -> tuple[float, float, float]:
    """変形正弦の無次元 (変位, 速度, 加速度)。x は 0..1 に正規化した割出し進行度。

    区間外は変位を 0 / 1 に張り付け、速度と加速度は 0 にする（停留がそのまま出る）。
    """
    A = MS_CA
    v1 = A / (4.0 * math.pi)                       # x = 1/8 と 7/8 での速度
    s1 = v1 * (1.0 / 8.0 - 1.0 / (4.0 * math.pi))  # x = 1/8 での変位
    s2 = s1 + v1 * 0.75 + 9.0 * A / (8.0 * math.pi ** 2)   # x = 7/8 での変位

    if x <= 0.0:
        return 0.0, 0.0, 0.0
    if x >= 1.0:
        return 1.0, 0.0, 0.0

    if x < 1.0 / 8.0:
        u = x
        s = v1 * (u - math.sin(4.0 * math.pi * u) / (4.0 * math.pi))
        v = v1 * (1.0 - math.cos(4.0 * math.pi * u))
        a = A * math.sin(4.0 * math.pi * u)
        return s, v, a

    if x < 7.0 / 8.0:
        u = x - 1.0 / 8.0
        w = 4.0 * math.pi / 3.0
        s = s1 + v1 * u + (9.0 * A / (16.0 * math.pi ** 2)) * (1.0 - math.cos(w * u))
        v = v1 + (3.0 * A / (4.0 * math.pi)) * math.sin(w * u)
        a = A * math.cos(w * u)
        return s, v, a

    u = x - 7.0 / 8.0
    s = s2 + v1 * u + (A / (16.0 * math.pi ** 2)) * (math.cos(4.0 * math.pi * u) - 1.0)
    v = v1 - (A / (4.0 * math.pi)) * math.sin(4.0 * math.pi * u)
    a = -A * math.cos(4.0 * math.pi * u)
    return s, v, a


# =======================================================================
# 2. 割出しの運動学
# =======================================================================
#
# 独立変数はカム入力軸角 psi。1 タクトで 0 -> 2pi を一定速度で回る。
#   0 <= psi < index_angle_input   : 割出し
#   それ以降                        : 停留
# テーブル角は psi の関数で、時間の関数ではない。


def cam_input_angle(t: float, p: params_mod.Params) -> float:
    """時刻 t のカム入力軸角 psi [rad]（0 <= psi < 2pi に畳む）。"""
    return (p.input_shaft_speed * t) % (2.0 * math.pi)


def cam_progress(psi: float, p: params_mod.Params) -> float:
    """割出し区間の進行度 x = 0..1。停留区間では 1 を返す。"""
    if psi <= 0.0:
        return 0.0
    if psi >= p.index_angle_input:
        return 1.0
    return psi / p.index_angle_input


def is_index_phase(psi: float, p: params_mod.Params) -> bool:
    """割出し中かどうか。"""
    return 0.0 <= psi < p.index_angle_input


def table_motion(psi: float, p: params_mod.Params) -> tuple[float, float, float]:
    """入力軸角 psi のときの (テーブル相対角 [rad], 角速度 [rad/s], 角加速度 [rad/s^2])。

        th   = index_angle * s(x)
        dth  = index_angle * v(x) * (dx/dt)
        ddth = index_angle * a(x) * (dx/dt)^2

    dx/dt は割出し区間を進む速さ = 1 / index_time。入力軸が一定速度で回るので定数。
    停留区間では s = 1、v = a = 0 になり、テーブルは止まる。
    """
    x = cam_progress(psi, p)
    s, v, a = modified_sine(x)
    rate = 1.0 / p.index_time          # dx/dt [1/s]
    th = p.index_angle * s
    omega = p.index_angle * v * rate
    alpha = p.index_angle * a * rate * rate
    return th, omega, alpha


def table_ratio(psi: float, p: params_mod.Params) -> float:
    """カムの瞬時変速比 dtheta/dpsi [-]。入力軸トルクへの換算に使う。

        dtheta/dpsi = index_angle * v(x) / index_angle_input
    """
    x = cam_progress(psi, p)
    _s, v, _a = modified_sine(x)
    return p.index_angle * v / p.index_angle_input


def table_omega_max(p: params_mod.Params) -> float:
    """テーブルの最大角速度 [rad/s] = Cv * index_angle / index_time。"""
    return MS_CV * p.index_angle / p.index_time


def table_alpha_max(p: params_mod.Params) -> float:
    """テーブルの最大角加速度 [rad/s^2] = Ca * index_angle / index_time^2。"""
    return MS_CA * p.index_angle / (p.index_time * p.index_time)


def tangential_accel_max(p: params_mod.Params) -> float:
    """ボトル位置の最大接線加速度 [m/s^2] = Rp * alpha_max。"""
    return p.Rp * table_alpha_max(p)


def bottle_accelerations(omega: float, alpha: float,
                         p: params_mod.Params) -> tuple[float, float]:
    """ボトルが受ける水平加速度 (接線, 半径) [m/s^2]。

        a_t =  Rp * alpha
        a_r = -Rp * omega^2      （向心加速度。半径方向は外向きを正にとる）

    これは容器（ボトル）そのものの慣性系での加速度で、回転基底で成分表示したもの。
    """
    a_t = p.Rp * alpha
    a_r = -p.Rp * omega * omega
    return a_t, a_r


def motor_angle(t: float, p: params_mod.Params) -> float:
    """モータ軸角 [rad]。減速比ぶん入力軸より速く、連続回転する。

    軸受の故障をモータ側に置くために要る。1500 rpm = 25 Hz。
    """
    return p.motor_speed * t


# =======================================================================
# 3. 慣性とトルク
# =======================================================================


def holder_mass(volume: float, has_bottle: bool, p: params_mod.Params) -> float:
    """ホルダ 1 個に載っている質量 [kg]。ボトルが無ければ 0。"""
    if not has_bottle:
        return 0.0
    return p.bottle_empty_mass + p.rho * volume


def load_inertia(volumes: list[float], p: params_mod.Params,
                 present: list[bool] | None = None,
                 liquid_rigid_fraction: float = 1.0) -> float:
    """テーブル側の慣性 J_load [kg m^2]。

        J_load = bare_inertia + sum_i( m_i * Rp^2 )

    bare_inertia は円板の極慣性 (1/2) m r^2。直径まわりの (1/4) m r^2 ではない。

    liquid_rigid_fraction は「液のうち剛体として一緒に回る割合」。
    既定の 1.0 は液を全部凍った剛体として数える。定常状態でこちらは 0.71904 kg m^2。
    スロッシングの反力トルクを別に足す場合は m0 の割合 (1 - m1/m_liq = 0.8774) を渡す。
    両方やると揺れる分を二重に数えることになる（ピークトルクで 1.0% の差）。
    """
    j = p.bare_inertia
    for i, v in enumerate(volumes):
        here = True if present is None else present[i]
        if not here:
            continue
        m = p.bottle_empty_mass + p.rho * v * liquid_rigid_fraction
        j = j + m * p.Rp * p.Rp
    return j


def rigid_mass(volume: float, has_bottle: bool, p: params_mod.Params) -> float:
    """ホルダ 1 個の質量のうち、剛体として一緒に回る分 [kg]。

    液の揺れる分 m1 は振り子として別に動くので抜いてある。
    スロッシングの反力トルクを軸に足すときは、慣性側はこちらを使う。
    """
    if not has_bottle:
        return 0.0
    if volume <= 0.0:
        return p.bottle_empty_mass
    h = p.height_from_volume(volume)
    m_liq = p.rho * volume
    m1 = m_liq * slosh_mass_ratio(p.R, h)
    return p.bottle_empty_mass + m_liq - m1


def rigid_load_inertia(volumes: list[float], p: params_mod.Params,
                       present: list[bool] | None = None) -> float:
    """液の m1 を抜いた J_load [kg m^2]。反力トルクと組にして使う。

    定常状態で 0.70663 kg m^2（全量剛体の 0.71904 より 1.7% 小さい）。
    """
    j = p.bare_inertia
    for i, v in enumerate(volumes):
        here = True if present is None else present[i]
        j = j + rigid_mass(v, here, p) * p.Rp * p.Rp
    return j


def residence_moves(p: params_mod.Params) -> int:
    """供給から排出までの割出し回数。工程角だけから決まる。

        n_res = ((discharge_angle - infeed_angle) / pitch) mod N

    供給 315deg / 排出 225deg / 8 ステーションなら 6 回 = 18.0 秒。
    ホルダ番号を固定で書かないこと。工程角を動かすとここが動く。
    """
    k = (p.discharge_angle - p.infeed_angle) / p.station_pitch
    return int(round(k)) % p.n_stations


def holder_age(i: int, p: params_mod.Params) -> int:
    """ホルダ i に載っているボトルが供給されてから何回割出しされたか [-]。

        k = ((i*pitch - infeed_angle) / pitch) mod N

    テーブル角 0（最初の割出しに入る直前）での数え方。k = 0 が
    「供給ステーションにいる = 受け取ったばかり」で、k が増えるほど古い。
    """
    k = (i * p.station_pitch - p.infeed_angle) / p.station_pitch
    return int(round(k)) % p.n_stations


def steady_holders(p: params_mod.Params) -> list["Holder"]:
    """定常状態のホルダ列（simulate の初期値）。

    最初の割出しに入る直前の並びとして置く。**工程配置から作る。**
    ホルダ i の齢 k = holder_age(i) を見て

      - k >= n_res : ボトルが無い（排出済みで、まだ供給ステーションに来ていない）
      - k == 0     : 空のボトル（供給されたばかり。次の停留で充填される）
      - それ以外   : 満量

    供給 315deg / 充填 0deg / 排出 225deg・8 ステーションでは、テーブル角 0 のとき

      - 世界角 0/45/90/135/180deg : 満量 5 本
      - 世界角 225 / 270deg       : 空ホルダ 2 つ（排出済み）
      - 世界角 315deg             : 空瓶 1 本

    載っている質量は 5*0.425 + 0.025 = 2.150 kg で、
    J_load = 0.6102 + 2.150*Rp^2 = 0.71904 kg m^2（液を全量剛体として数えた場合）。
    ボトル 1 本の滞留は割出し 6 回 = 18 秒。

    前版はホルダ N-2 / N-1 を固定で見ていて、排出角が「供給角 - 1 ピッチ」の
    ときにしか合わなかった。排出が 225deg に動いた時点で幾何と食い違う。
    「満量 7 本 + 空 1 本（J_load = 0.7621）」は全ホルダにボトルが載っている前提の
    数え方で、選定の上限としては使えるが工程配置とは合わない。
    """
    n_res = residence_moves(p)
    holders: list[Holder] = []
    for i in range(p.n_stations):
        k = holder_age(i, p)
        if k >= n_res:
            holders.append(Holder(has_bottle=False))
        elif k == 0:
            holders.append(Holder(has_bottle=True, volume=0.0, filled=False))
        else:
            holders.append(Holder(has_bottle=True, volume=p.target_volume,
                                  filled=True))
    return holders


def steady_load(p: params_mod.Params) -> tuple[list[float], list[bool]]:
    """定常状態の (体積の並び, ボトルが載っているか) を取り出す。"""
    holders = steady_holders(p)
    return ([h.volume for h in holders], [h.has_bottle for h in holders])


def slosh_mass_rate(volume: float, p: params_mod.Params,
                    eps1: float = EPS1) -> float:
    """揺れる質量 m1 の体積についての微係数 dm1/dV [kg/m^3]。

        m1 = rho * 2*R*A*tanh(eps1*h/R) / (eps1*(eps1^2-1))     （h = V/A）
        dm1/dV = 2*rho / ( (eps1^2 - 1) * cosh^2(eps1*h/R) )

    V が消えた形になるので厳密に微分できる。数値差分は取らない。
    液深が深いほど cosh が効いて 0 に落ちる（満量では rho の 1e-5 倍）。
    """
    h = p.height_from_volume(volume)
    c = math.cosh(eps1 * h / p.R)
    return 2.0 * p.rho / ((eps1 * eps1 - 1.0) * c * c)


def load_inertia_rate(volume: float, p: params_mod.Params) -> float:
    """充填中の dJ_load/dt [kg m^2/s]。液は m0 だけ数える側の J_load の時間微分。

        dJ_load/dt = Rp^2 * (dm0/dV) * flow_rate
        dm0/dV     = rho - dm1/dV

    ノズルは世界座標に固定なので、入ってくる液の接線速度は 0。角運動量の収支から
    テーブル軸には (dJ_load/dt)*omega の項が要る（MODEL.md 7.4）。

    **公称条件ではこの項は厳密に 0。** 充填は停留中に終わり、停留では omega = 0 で
    あって、丸めですら 0 にならない量ではない。それでも式には入れる。タクトを詰めて
    充填が割出しにはみ出したときに効き始めるため。
    """
    return (p.rho - slosh_mass_rate(volume, p)) * p.flow_rate * p.Rp * p.Rp


def table_torque(j_load: float, omega: float, alpha: float,
                 p: params_mod.Params, t_slosh: float = 0.0,
                 dj_dt: float = 0.0) -> float:
    """テーブル軸に要るトルク [Nm]。

        T = J_load * alpha + (dJ_load/dt) * omega + c_visc * omega - T_slosh

    **T_slosh は引く。** T_slosh は「テーブルが受ける側」の符号で定義してある
    （MODEL.md 2.4）ので、駆動側に要るトルクにするには符号を返す。

        T_slosh = -Rp * sum( m1 * a1_t )
        準静的極限（液がタンクに追従して a1_t = Rp*alpha）では T_slosh = -m1*Rp^2*alpha
        -> T = J*alpha + m1*Rp^2*alpha        揺動質量が慣性として素直に足される

    足す向き（前版）だと、この極限で揺動質量ぶんの慣性が引かれてしまい物理にならない。
    テーブル軸ピークトルクで 5.8% の差が出る。

    c_visc（table_viscous_Nms_rad）は仮置き。出所が無い。
    旋回軸受の摩擦トルクは params.json に無いので入っていない。そのぶん、
    定速区間が厳密に c_visc*omega になるのはモデルの都合であって物理ではない。
    """
    return (j_load * alpha + dj_dt * omega + p.table_viscous * omega - t_slosh)


def input_shaft_torque(t_table: float, dtheta_dpsi: float,
                       p: params_mod.Params, omega: float = 0.0) -> float:
    """カム入力軸に要るトルク [Nm]。

        T_in  = T_cam + drag
        T_cam = T_table * (dtheta/dpsi) / efficiency   （T_table*omega >= 0）
        T_cam = T_table * (dtheta/dpsi) * efficiency   （T_table*omega <  0）

    効率の掛け方は流れの向きで逆になる（MODEL.md 4.7）。カムが負荷を駆動している間は
    損失ぶん余計に要り、負荷がカムを回している間は損失ぶん減って伝わる。
    前版は絶対値を取って常に /efficiency だったので、減速側を 1/0.85^2 = 1.38 倍
    過大に見ていた。dtheta/dpsi は常に 0 以上なので、向きを決めるのは T_table*omega。

    efficiency（0.85）と drag（0.30 Nm）はどちらも仮置きで出所が無い。
    引きずりトルクは負荷ピーク（0.40 Nm）と同じ桁あるので、入力軸トルクの 4 割を占める。
    ここでは入力軸が常に正方向へ回っている前提で、引きずりは常に正の負荷とする。
    """
    t_cam = t_table * dtheta_dpsi
    if t_table * omega >= 0.0:
        t_cam = t_cam / p.cam_efficiency
    else:
        t_cam = t_cam * p.cam_efficiency
    return t_cam + p.input_drag_torque


def input_shaft_torque_peak(p: params_mod.Params, volumes: list[float] | None = None,
                            present: list[bool] | None = None, n: int = 20000,
                            viscous: bool = True, drag: bool = True
                            ) -> tuple[float, float]:
    """割出し 1 回の中での入力軸トルクの最大値 [Nm] と、そのときの psi [rad] を返す。

    トルクの最大（psi = 22.5deg）と速度の最大（psi = 90deg）は同時に起きない。
    入力軸に返るのは T_table に dtheta/dpsi（速度に比例）を掛けたあとなので、
    最大は両者の積が最大になる位置（psi = 51.7deg 付近）に来る。
    2 つの最大値を掛け合わせると 8 割ほど過大になるので、この関数で走査する。

    viscous / drag を False にすると、その項を外した値になる（仮置きの寄与を見るため）。
    効率 0.85 は常に掛かる（向きは input_shaft_torque と同じ場合分け）。
    スロッシングの反力トルクは入れていない。ここは慣性と粘性だけの走査。
    """
    if volumes is None or present is None:
        volumes, present = steady_load(p)
    j = load_inertia(volumes, p, present)
    best = 0.0
    best_psi = 0.0
    for k in range(n + 1):
        psi = p.index_angle_input * k / n
        _th, omega, alpha = table_motion(psi, p)
        t_tab = j * alpha + (p.table_viscous * omega if viscous else 0.0)
        t_cam = t_tab * table_ratio(psi, p)
        t_in = (t_cam / p.cam_efficiency if t_tab * omega >= 0.0
                else t_cam * p.cam_efficiency)
        if drag:
            t_in = t_in + p.input_drag_torque
        if t_in > best:
            best = t_in
            best_psi = psi
    return best, best_psi


def peak_mechanical_power(p: params_mod.Params, volumes: list[float] | None = None,
                          present: list[bool] | None = None,
                          n: int = 20000, viscous: bool = True) -> float:
    """テーブル軸のピーク機械出力 [W] = max(T_table * omega)。

    負荷そのものはこれしか食わない。モータの定格出力はここからは決まっていない
    （params.json の drive._sizing_note）。
    viscous=False にすると慣性負荷だけの値になる。
    """
    if volumes is None or present is None:
        volumes, present = steady_load(p)
    j = load_inertia(volumes, p, present)
    best = 0.0
    for k in range(n + 1):
        psi = p.index_angle_input * k / n
        _th, omega, alpha = table_motion(psi, p)
        t_tab = j * alpha + (p.table_viscous * omega if viscous else 0.0)
        best = max(best, abs(t_tab * omega))
    return best


# =======================================================================
# 4. スロッシング（回転座標系・コリオリ結合）
# =======================================================================


def slosh_omega(R: float, h: float, g: float, eps1: float = EPS1) -> float:
    """円筒容器 1 次反対称モードの固有角周波数 w1 [rad/s]。

        w1^2 = (g * eps1 / R) * tanh(eps1 * h / R)

    h <= 0（液が無い）のときは 0 を返す。呼ぶ側で 0 割りを避けること。
    """
    if h <= 0.0:
        return 0.0
    w1_squared = (g * eps1 / R) * math.tanh(eps1 * h / R)
    return math.sqrt(w1_squared)


def pendulum_length(w1: float, g: float) -> float:
    """等価振り子の長さ L1 = g / w1^2 [m]。"""
    if w1 <= 0.0:
        return float("inf")
    return g / (w1 * w1)


def slosh_mass_ratio(R: float, h: float, eps1: float = EPS1) -> float:
    """液のうち揺れる分の割合 m1 / m_liq [-]。

        m1/m_liq = 2R * tanh(eps1*h/R) / ( eps1 * (eps1^2 - 1) * h )

    残り m0 = m_liq - m1 は容器と一緒に動く「凍った液」として扱う。
    この寸法・満量では m1/m_liq = 0.123（液 400 g のうち 49 g）。
    """
    if h <= 0.0:
        return 0.0
    denom = eps1 * (eps1 * eps1 - 1.0) * h
    return 2.0 * R * math.tanh(eps1 * h / R) / denom


@dataclass
class SloshState:
    """1 本のボトルの液面。回転座標系（接線 t / 半径 r）で持つ等価振り子の角。

    2 方向は独立ではない。コリオリ項で結合する（step_slosh を見ること）。
    """

    phi_t: float = 0.0      # 接線方向の振り子角 [rad]
    dphi_t: float = 0.0     # その角速度（回転座標系での微分）[rad/s]
    phi_r: float = 0.0      # 半径方向の振り子角 [rad]
    dphi_r: float = 0.0

    def tilt(self) -> float:
        """合成傾き [rad]。"""
        return math.sqrt(self.phi_t * self.phi_t + self.phi_r * self.phi_r)

    def reset(self) -> None:
        self.phi_t = self.dphi_t = self.phi_r = self.dphi_r = 0.0


def step_slosh(state: SloshState, a_t: float, a_r: float,
               omega: float, alpha: float,
               w1: float, zeta: float, dt: float, g: float,
               coriolis: bool = True) -> None:
    """スロッシングを 1 ステップ進める（state をその場で書き換える）。

    慣性系での等価振り子は phi'' + 2*zeta*w1*phi' + w1^2*phi = -a/L1（a は容器の加速度）。
    これを回転する基底（r 方向・t 方向）で書くと、基底の回転ぶんの項が出る。
    r^ , t^ が dr^/dt = omega*t^, dt^/dt = -omega*r^ で回るので、ベクトル u に対し

        u'' = (u_r'' - 2*omega*u_t' - alpha*u_t - omega^2*u_r) r^
            + (u_t'' + 2*omega*u_r' + alpha*u_r - omega^2*u_t) t^

    これを振り子の式に入れて整理すると:

        phi_r'' =  2*omega*phi_t' + alpha*phi_t + omega^2*phi_r
                   - 2*zeta*w1*phi_r' - w1^2*phi_r - a_r/L1
        phi_t'' = -2*omega*phi_r' - alpha*phi_r + omega^2*phi_t
                   - 2*zeta*w1*phi_t' - w1^2*phi_t - a_t/L1

    2*omega*phi' がコリオリ、alpha*phi がオイラー、omega^2*phi が遠心。
    コリオリは 1 次モードを共回転／逆回転の 2 本に ±omega だけ割る（mode_split_hz）。
    この機械では分裂幅が共振の半値幅より桁で大きいので、入れるかどうかで結果が変わる。

    **取り決め**: 減衰は容器に対する相対速度（回転座標系での微分）に掛けている。
    減衰の実体は壁の境界層なので相対のほうが素直で、この選び方だと下の a1 の式が
    厳密に閉じる。慣性系の微分に掛ける流儀との差は 2*zeta*omega/w1 = 4e-4 相当で、
    数値としては見えない（テストで両者の差を確認している）。

    積分は半陰的オイラー。加速度を今の状態から作り、速度を更新し、その新しい速度で
    角度を更新する。結合項は今のステップの速度を使う（陽的）。

    coriolis=False にすると回転基底の 3 項を落とし、前版と同じ「2 方向が独立な振り子」に
    戻る。比較のために残してある。既定は True で、こちらが正。
    """
    if w1 <= 0.0:
        return

    L1 = pendulum_length(w1, g)

    if not coriolis:
        omega = 0.0
        alpha = 0.0

    ddphi_r = (2.0 * omega * state.dphi_t
               + alpha * state.phi_t
               + omega * omega * state.phi_r
               - 2.0 * zeta * w1 * state.dphi_r
               - w1 * w1 * state.phi_r
               - a_r / L1)

    ddphi_t = (-2.0 * omega * state.dphi_r
               - alpha * state.phi_r
               + omega * omega * state.phi_t
               - 2.0 * zeta * w1 * state.dphi_t
               - w1 * w1 * state.phi_t
               - a_t / L1)

    state.dphi_r = state.dphi_r + ddphi_r * dt
    state.dphi_t = state.dphi_t + ddphi_t * dt
    state.phi_r = state.phi_r + state.dphi_r * dt
    state.phi_t = state.phi_t + state.dphi_t * dt


def mode_split_hz(omega: float) -> float:
    """コリオリによるモード分裂の片側幅 [Hz]。

    回転座標系の自由振動（減衰なし・omega 一定）を z = phi_r + i*phi_t で書くと

        z'' + 2i*omega*z' + (w1^2 - omega^2) z = 0

    となり、根は z = exp(i*lambda*t) の lambda = w1 - omega と -(w1 + omega)。
    つまり固有振動数が w1 ± omega の 2 本に割れる。幅は w1 に依らず omega だけで決まる。
    割合で言いたいときは mode_split_fraction を使うこと。
    """
    return abs(omega) / (2.0 * math.pi)


def mode_split_fraction(w1: float, omega: float) -> float:
    """モード分裂の割合 omega/w1 [-]。

    omega は割出しの間ずっと変わる（停留で 0、ピークで 0.9213 rad/s）ので、
    この割合も 1 つの数字には決まらない。どの omega で評価したかを必ず添えること。
    """
    if w1 <= 0.0:
        return 0.0
    return abs(omega) / w1


def resonance_half_width_hz(w1: float, zeta: float) -> float:
    """共振の半値半幅 [Hz] = zeta * f1。離調が意味を持つかどうかの物差し。"""
    return zeta * w1 / (2.0 * math.pi)


def nearest_harmonic_detuning(f1: float, tact: float) -> tuple[int, float, float]:
    """タクトの高調波のうち f1 に最も近いものを返す。

    戻り値は (次数 n, その周波数 [Hz], 離調 |f1 - n/tact| [Hz])。
    離調が共振の半値幅と同じ桁なら、動作点が偶然の共振で決まっていることになる。
    """
    f0 = 1.0 / tact
    n = max(1, int(round(f1 / f0)))
    f_h = n * f0
    return n, f_h, abs(f1 - f_h)


def index_slosh_response(p: params_mod.Params, volume: float | None = None,
                         coriolis: bool = True, dt: float | None = None
                         ) -> tuple[float, float, float]:
    """静止した液面に割出しを 1 回かけたときの傾き。1 本ぶんだけを見る。

    戻り値は (割出し中の最大傾き, 停留中の最大傾き, 割出し終端の傾き) [rad]。

    充填も受け渡しもしない。液深は volume（既定は満量）から決めた一定値。
    諸元の期待値（ピーク 68.07 mrad / 残留 35.3 mrad）はこの条件のもの。
    タクト 1 回ぶんだけ回すので、前のサイクルの残りが乗った場合は
    これより大きくなる（simulate のほうで見ること）。
    """
    if volume is None:
        volume = p.target_volume
    if dt is None:
        dt = p.dt

    h = p.height_from_volume(volume)
    w1 = slosh_omega(p.R, h, p.g)
    st = SloshState()

    peak_index = 0.0
    peak_dwell = 0.0
    tilt_end = 0.0
    n_steps = int(round(p.tact / dt))
    for k in range(n_steps):
        t = k * dt
        psi = cam_input_angle(t, p)
        _th, omega, alpha = table_motion(psi, p)
        a_t, a_r = bottle_accelerations(omega, alpha, p)
        step_slosh(st, a_t, a_r, omega, alpha, w1, p.zeta, dt, p.g, coriolis)
        tilt = st.tilt()
        if is_index_phase(psi, p):
            peak_index = max(peak_index, tilt)
            tilt_end = tilt
        else:
            peak_dwell = max(peak_dwell, tilt)
    return peak_index, peak_dwell, tilt_end


def slosh_accel(state: SloshState, w1: float, zeta: float, L1: float,
                g: float) -> tuple[float, float]:
    """揺れる質量 m1 の慣性系での加速度 (接線, 半径) [m/s^2]。

        a1 = -( g*phi + 2*zeta*w1*L1*phi' )

    振り子の運動方程式を代入すると容器の加速度が消えてこの形になる（w1^2*L1 = g）。
    回転座標系で解いていても、減衰を相対速度に掛けている限りこの形のまま成り立つ。
    """
    a1_t = -(g * state.phi_t + 2.0 * zeta * w1 * L1 * state.dphi_t)
    a1_r = -(g * state.phi_r + 2.0 * zeta * w1 * L1 * state.dphi_r)
    return a1_t, a1_r


def liquid_force(state: SloshState, volume: float, a_t: float, a_r: float,
                 w1: float, zeta: float, L1: float,
                 p: params_mod.Params) -> tuple[float, float]:
    """液がボトルに及ぼす水平力 (接線, 半径) [N]。**返すのは反作用のほう。**

        F_hold  =   m0 * a_tank + m1 * a1      ホルダが中身に及ぼす力（液を動かす力）
        F_react = -(m0 * a_tank + m1 * a1)     中身がホルダに返す力。返り値はこちら

    符号は MODEL.md 2.4 の取り決めに合わせて「テーブルが受ける側」に揃えてある。
    T_slosh（= -Rp * sum(m1*a1_t)）と同じ向き。前版は力だけ逆向きで、
    力とトルクで向きが揃っていなかった。

    m0 は一緒に動く分、m1 は揺れる分。m0 の分は J_load 側にも入っているので、
    軸トルクに足すときは m1 の分だけにすること（二重計上になる）。
    """
    if volume <= 0.0:
        return 0.0, 0.0
    h = p.height_from_volume(volume)
    m_liq = p.rho * volume
    m1 = m_liq * slosh_mass_ratio(p.R, h)
    m0 = m_liq - m1
    a1_t, a1_r = slosh_accel(state, w1, zeta, L1, p.g)
    return -(m0 * a_t + m1 * a1_t), -(m0 * a_r + m1 * a1_r)


# =======================================================================
# 5. 垂直荷重経路
# =======================================================================
#
# ボトルの重量はピッチ円半径 Rp に載る。8 方位に均等なら打ち消し合うが、
# 空きホルダや充填途中があると偏りが残り、テーブル軸に曲げモーメントとして入る。
# 前版はこの経路が丸ごと無く、水平力だけを見ていた。桁が 1 つ半違う。


def bending_moment_from_loads(loads: list[tuple[float, float]],
                              p: params_mod.Params) -> tuple[float, float, float]:
    """(世界角 [rad], 鉛直下向きの力 [N]) の並びから、テーブル軸の曲げモーメントを出す。

    戻り値は (Mx, My, |M|) [Nm]。
    各荷重は軸から Rp 離れた点に載るので、モーメントの大きさは Rp * |sum(F_i * u_i)|。
    Mx / My は「どちらへ偏っているか」を世界座標で持つための成分で、
    曲げの中立軸はこのベクトルに直交する。

    **どこで受けるかは決めない。** 取り付け位置と荷重経路が未決なので、
    ここは軸に働くモーメントまで。ひずみへの換算（断面係数・ヤング率・ゲージ方位）は
    センサ層の仕事にする。
    """
    sx = 0.0
    sy = 0.0
    for angle, force in loads:
        sx = sx + force * math.cos(angle)
        sy = sy + force * math.sin(angle)
    mx = p.Rp * sx
    my = p.Rp * sy
    return mx, my, math.hypot(mx, my)


def vertical_bending_moment(volumes: list[float], th_t: float,
                            p: params_mod.Params,
                            present: list[bool] | None = None,
                            extra: list[tuple[float, float]] | None = None
                            ) -> tuple[float, float, float]:
    """ボトル重量によるテーブル軸の曲げモーメント (Mx, My, |M|) [Nm]。

    全ホルダに同じ質量が載っていれば 8 方位の和が 0 になって消える。
    残るのは中身の偏りだけ。空ホルダ 1 個の欠品でも、満量 1 本ぶん（0.883 Nm）が残る。
    extra は充填ジェットなど、重量以外の鉛直力を (世界角, 力) で足したいとき。
    """
    loads: list[tuple[float, float]] = []
    for i, v in enumerate(volumes):
        here = True if present is None else present[i]
        m = holder_mass(v, here, p)
        loads.append((p.station_world_angle(i, th_t), m * p.g))
    if extra:
        loads.extend(extra)
    return bending_moment_from_loads(loads, p)


# =======================================================================
# 6. 充填と可変質量の運動量
# =======================================================================


def jet_force(p: params_mod.Params, fall_height: float = 0.0) -> float:
    """充填流がテーブルに与える鉛直下向きの力 [N]。

        F = rho * Q * v_impact          v_impact = sqrt(v_nozzle^2 + 2*g*fall_height)

    質量が静かに増えるだけではなく、注入の運動量ぶんだけ余分に押す。
    液柱が落ちる間は液の重量がテーブルに載らない一方、着液した瞬間に運動量が渡る。

    **fall_height（ノズル出口から液面までの距離）は params.json に無い。** 既定は 0 で、
    その場合はノズル出口の流速がそのまま着液速度になる。落差を入れれば増える。
    """
    v_impact = math.sqrt(p.nozzle_velocity ** 2 + 2.0 * p.g * max(0.0, fall_height))
    return p.rho * p.flow_rate * v_impact


def step_fill(volume: float, filling: bool, dt: float,
              p: params_mod.Params) -> float:
    """充填を 1 ステップ進めて、新しい体積 [m^3] を返す。

        V' = flow_rate  (充填中)

    行き過ぎの扱いは呼ぶ側（弁の閉じ判断）に任せる。ここは流量を積むだけ。
    """
    if not filling:
        return volume
    return volume + p.flow_rate * dt


def close_command_volume(p: params_mod.Params) -> float:
    """弁を閉じろと命じる体積 [m^3]。

        V_cmd = target_volume - flow_rate * valve_close_delay

    弁は命じてから valve_close_delay だけ遅れて閉じる。その間も流れるので、
    遅れぶんを先読みして早めに命じる、という取り決めにした
    （params.json は閉じ命令の出し方を決めていない）。

    **先読みするのは公称の遅れ（fill.valve_close_delay_s）だけ。** 制御側は
    弁が劣化して遅くなったことを知らないので、故障ぶんの遅れは先読みできない。
    だから正常時はちょうど target_volume で止まり、閉じ遅れ故障
    （extra_delay_s = 0.15 s）では flow_rate * 0.15 = 49.5 mL の過充填になる。
    前版は extra_delay まで先読みしていたので、故障を入れても 400 mL のままだった。
    """
    return p.target_volume - p.flow_rate * p.valve_close_delay


# =======================================================================
# 7. こぼれと適用範囲
# =======================================================================


def wall_rise(R: float, tilt: float) -> float:
    """壁での液面の上下動 dz = R * tan(tilt) [m]。"""
    return R * math.tan(tilt)


def apply_spill(volume: float, tilt: float, R: float, cross_section: float,
                body_height: float) -> tuple[float, float]:
    """こぼれを判定して (残った体積, こぼれた体積) を返す。

    h + dz が body_height を超えたら、超過分の高さに断面積を掛けたものを引く。
    実際に縁を越えるのは傾いた液面が縁を切る楔だけなので、この扱いはこぼれを
    多めに見積もる（NOTES.md「こぼれ判定の保守性」参照）。
    この諸元では傾き 68 mrad に対し頭上空間 29.5 mm あるので、そもそも届かない。
    """
    if volume <= 0.0:
        return 0.0, 0.0
    h = volume / cross_section
    dz = wall_rise(R, tilt)
    over = h + dz - body_height
    if over <= 0.0:
        return volume, 0.0
    spilled = over * cross_section
    if spilled > volume:
        spilled = volume
    return volume - spilled, spilled


def linearity_flag(tilt: float) -> str:
    """傾きが線形理論の範囲に収まっているかの表示を返す。

    dz/R = tan(tilt) を、壁での波高が R の何倍かとみなして判定する。
    """
    ratio = abs(math.tan(tilt))
    if ratio > TILT_INVALID_RATIO:
        return "invalid"
    if ratio > TILT_WARN_RATIO:
        return "warn"
    return "ok"


def dt_warning(p: params_mod.Params, h: float | None = None,
               steps_per_period: float = 20.0) -> str | None:
    """dt がスロッシング周期に対して粗すぎないか見て、粗ければ文言を返す。"""
    if h is None:
        h = p.fill_height
    w1 = slosh_omega(p.R, h, p.g)
    if w1 <= 0.0:
        return None
    period = 2.0 * math.pi / w1
    if p.dt * steps_per_period > period:
        return (f"dt={p.dt:.2e}s はスロッシング周期 {period:.4f}s に対して粗い"
                f"（1 周期あたり {period / p.dt:.1f} ステップ）")
    return None


# =======================================================================
# 8. イベント（連続量に乗せないもの）
# =======================================================================
#
# 受け渡しの当たり、カムフォロワの当たり、軸受の衝撃は、いずれも
# 連続ログ（4 kHz）には乗せない。3 kHz のリンギングが折り返すため。
# 発生時刻と振幅の列として出し、波形への合成はセンサ層が各センサの刻みで行う。


@dataclass
class Event:
    """イベント 1 件。data の中身は kind ごとに違う。"""

    t: float
    kind: str            # infeed / discharge / cam_impact / bearing_impulse / valve_drip
    station: int
    data: dict = field(default_factory=dict)


def bearing_impulse_times(p: params_mod.Params, t0: float, t1: float) -> list[float]:
    """外輪傷の衝撃が起きる時刻 [s] の列（t0 <= t < t1）。

    欠陥通過周波数はモータ軸の回転周波数の定数倍。モータは連続回転しているので、
    テーブルが止まっている間も等間隔で出続ける。ここが前版（テーブル軸に置いていた）
    との一番の違いで、1 サイクルあたりの衝撃回数が 0.44 回から 268 回になる。
    """
    f = p.faults.get("bearing_outer_race")
    if f is None:
        return []
    freq = f.get("defect_freq_hz")
    if not freq:
        return []
    period = 1.0 / freq
    k0 = math.ceil(t0 / period)
    times = []
    k = k0
    while k * period < t1:
        times.append(k * period)
        k += 1
    return times


# =======================================================================
# 9. 通しで回す
# =======================================================================


@dataclass
class Holder:
    """ホルダ 1 個。ボトルが載っていないこともある。"""

    has_bottle: bool = False
    volume: float = 0.0          # 入っている液の体積 [m^3]
    filled: bool = False         # 充填済みか（充填ステーションで 1 回だけ入れる）
    spilled: float = 0.0         # こぼれた累積 [m^3]
    slosh: SloshState = field(default_factory=SloshState)

    def height(self, p: params_mod.Params) -> float:
        return p.height_from_volume(self.volume)


@dataclass
class Record:
    """ログ 1 行。C 側のダンプ（core/FORMAT.md）と同じ並びにしてある。

    水平力の 2 列は **世界座標の x / y 成分**（MODEL.md 10.1 の F_tab_x / F_tab_y）。
    前版は f_tab_t / f_tab_r という名前で中身が世界座標の y / x だったので、
    名前を中身に合わせて直した。回転基底（接線・半径）への分解はしない。
    センサの取り付け方位が未決なので、世界座標のまま出してセンサ層に任せる。

    符号は **テーブルが受ける側**（MODEL.md 2.4 の F_react）。液の反作用と
    空瓶の慣性反力の和で、前版とは向きが逆になっている。

    torque_slosh も同じ「テーブルが受ける側」の符号（MODEL.md 2.4 の T_slosh）。
    **torque_table にはこれが既に引かれて入っている**（T = J*al + c*om - T_slosh）ので、
    ログから足し直さないこと。

    present は在荷フラグ。V = 0 だけでは「空瓶が載っている」と「ボトルが無い」を
    区別できないので列として持つ（C 側のダンプでは 0/1 の 1 バイト）。
    """

    t: float
    psi: float           # カム入力軸角 [rad]
    th_t: float          # テーブル角 [rad]
    omega: float
    alpha: float
    th_m: float          # モータ軸角 [rad]。軸受故障の置き場
    j_load: float
    torque_table: float
    torque_input: float
    torque_slosh: float
    m_bend: float        # テーブル軸の曲げモーメントの大きさ [Nm]
    m_bend_x: float
    m_bend_y: float
    f_tab_x: float       # テーブルが受ける水平合力の世界座標 x 成分 [N]
    f_tab_y: float
    volume: list[float]
    height: list[float]
    phi_t: list[float]
    phi_r: list[float]
    spill: list[float]
    present: list[bool]  # ホルダにボトルが載っているか


@dataclass
class Result:
    records: list[Record]
    events: list[Event]
    max_tilt: float
    max_tilt_index: float     # 割出し中の最大傾き [rad]
    max_tilt_dwell: float     # 停留中の最大傾き [rad]
    linearity: str
    warnings: list[str]


def simulate(p: params_mod.Params, n_cycles: int = 1,
             log_rate: float | None = None,
             dt: float | None = None,
             prime: bool = True,
             jet_fall_height: float = 0.0) -> Result:
    """割出し + 停留を n_cycles 回まわす。

    1 サイクルは「割出し（入力軸 0..180deg）-> 停留（180..360deg）」の順。
    停留に入った瞬間に受け渡し（排出 discharge_deg / 供給 infeed_deg）を行い、
    そのあと start_delay を置いて充填ステーション（fill_deg）で弁を開ける。

    prime=True なら、工程配置から作った定常状態（steady_holders）から始める。
    False なら空のテーブルから始める。

    **取り決め**（params.json が決めていないので、ここで決めた。C 側と揃えること）:
      - 受け渡しは停留の先頭で一瞬に起きる。星車の当たりはイベントにするだけで、
        連続量には乗せない。
      - 排出されたボトルの液は、揺れたまま外界へ持ち出される。イベントに
        そのときの傾きと角速度を残す（テーブル側の状態からは消す）。
      - 弁は公称の閉じ遅れだけを先読みして命じる（close_command_volume）。
        故障で伸びた遅れは先読みしないので、そのぶん過充填になる。

    **入れていないもの**（MODEL.md にはあるが、ここでは落とした。理由つき）:
      - PLC の 5 ms 格子（MODEL.md 7.1）。指令時刻を 5 ms に丸める取り決めだが、
        格子の位相を t = 0 に固定すると公称の開指令 t_d + 0.050 s はもともと格子点に
        乗るので、開弁側は 1 ステップも動かない。閉じ側だけが最大 1.65 mL（0.41%）
        量子化されることになり、弁の遅れモデルと区別がつかない差を足すだけになる。
        指令の刻みを主張に使うならセンサ層の取り決めとして別に置く。
      - カムフォロワ摩耗のテーブル角オフセット（MODEL.md 8.4）。テーブル角は
        カムから運動学で与えていて積分していないので、押している側が入れ替わる
        瞬間に ±clearance/2 の段差が入る。速度を伴わない角度の跳びになり、
        まともにやるには接触剛性が要る。params.json に剛性が無いので入れない。
        当たりそのものはイベント（cam_impact）で出している。
      - 水平反力による曲げ（MODEL.md 6.1 の M_horiz）。腕の長さがセンサ側の
        arm_length_mm しか無く、機械側の重心高さとして確かめていない。
    """
    if log_rate is None:
        log_rate = p.log_rate
    if dt is None:
        dt = p.dt

    warnings: list[str] = []
    w = dt_warning(p)
    if w:
        warnings.append(w)

    holders = (steady_holders(p) if prime
               else [Holder() for _ in range(p.n_stations)])

    # 故障の設定
    missing_station = None
    if p.fault_enabled("missing_bottle"):
        missing_station = int(p.fault("missing_bottle").station) % p.n_stations
    extra_close_delay = 0.0
    drip_volume = 0.0
    if p.fault_enabled("valve_close_delay"):
        f = p.fault("valve_close_delay")
        extra_close_delay = f.get("extra_delay_s", 0.0)
        drip_volume = f.get("drip_volume_mL", 0.0) * 1.0e-6
    cam_wear = p.fault_enabled("cam_follower_wear")

    events: list[Event] = []
    records: list[Record] = []
    log_interval = 1.0 / log_rate
    next_log = 0.0
    max_tilt = 0.0
    max_tilt_index = 0.0
    max_tilt_dwell = 0.0

    n_steps = int(round(p.tact / dt))
    th_base = 0.0          # 割出しの積み上げぶんのテーブル角
    t = 0.0

    for cycle in range(n_cycles):
        t0 = cycle * p.tact
        dwell_start = t0 + p.index_time
        valve_open_t = dwell_start + p.start_delay + p.valve_open_delay
        closing = False        # 弁の閉じ命令を出したあとか
        close_t = 0.0

        # --- 停留の先頭で起きる受け渡し（この時刻のテーブル角で数える）---
        th_now = th_base + p.index_angle
        i_out = p.holder_at(p.discharge_angle, th_now)
        i_in = p.holder_at(p.infeed_angle, th_now)

        # --- カムフォロワの当たり（割出しの入口と出口）---
        if cam_wear:
            f = p.fault("cam_follower_wear")
            for t_hit in (t0, t0 + p.index_time):
                events.append(Event(t=t_hit, kind="cam_impact", station=-1, data={
                    "torque_Nm": f.get("impact_torque_Nm", 0.0),
                    "clearance_rad": f.get("clearance_deg", 0.0) * math.pi / 180.0,
                }))

        # --- 軸受の衝撃（モータ軸。テーブルの停止と無関係に出続ける）---
        if p.fault_enabled("bearing_outer_race"):
            f = p.fault("bearing_outer_race")
            for t_hit in bearing_impulse_times(p, t0, t0 + p.tact):
                events.append(Event(t=t_hit, kind="bearing_impulse", station=-1, data={
                    "accel_m_s2": f.get("impulse_accel_m_s2", 0.0),
                    "ring_freq_hz": f.get("ring_freq_hz", 0.0),
                    "ring_damping": f.get("ring_damping", 0.0),
                    "motor_angle_rad": motor_angle(t_hit, p),
                }))

        transferred = False

        for k in range(n_steps):
            t_in_cycle = k * dt
            t = t0 + t_in_cycle
            # psi はサイクル内時刻から作る。通し時刻に剰余を取ると、サイクルの境目で
            # 丸め次第で 2pi の直前に落ちて、テーブル角が 1 ステップだけ飛ぶ。
            psi = cam_input_angle(t_in_cycle, p)
            in_index = is_index_phase(psi, p)

            th_rel, omega, alpha = table_motion(psi, p)
            th_t = th_base + th_rel
            a_t, a_r = bottle_accelerations(omega, alpha, p)

            # --- 受け渡し（停留に入った最初のステップで 1 回だけ）---
            if not in_index and not transferred:
                transferred = True
                out = holders[i_out]
                if out.has_bottle:
                    events.append(Event(t=t, kind="discharge", station=i_out, data={
                        "volume_m3": out.volume,
                        "tilt_rad": out.slosh.tilt(),
                        "phi_t_rad": out.slosh.phi_t,
                        "phi_r_rad": out.slosh.phi_r,
                        "dphi_t_rad_s": out.slosh.dphi_t,
                        "dphi_r_rad_s": out.slosh.dphi_r,
                        "mass_kg": holder_mass(out.volume, True, p),
                    }))
                    out.has_bottle = False
                    out.volume = 0.0
                    out.filled = False
                    out.slosh.reset()
                inn = holders[i_in]
                if not inn.has_bottle and i_in != missing_station:
                    inn.has_bottle = True
                    inn.volume = 0.0
                    inn.filled = False
                    inn.slosh.reset()
                    events.append(Event(t=t, kind="infeed", station=i_in, data={
                        "mass_kg": p.bottle_empty_mass,
                        # 星車との当たりの大きさは params.json に無い。
                        # 速度も剛性も置き場が無いので、ここでは発生時刻だけ渡す。
                        "impact_Ns": None,
                    }))
                elif i_in == missing_station:
                    events.append(Event(t=t, kind="infeed_missed", station=i_in, data={}))

            # --- 充填（充填ステーションのホルダに 1 回だけ）---
            i_fill = p.holder_at(p.fill_angle, th_t)
            fill_h = holders[i_fill]
            filling = False
            if (not in_index and fill_h.has_bottle and not fill_h.filled
                    and t >= valve_open_t):
                if not closing:
                    filling = True
                    fill_h.volume = step_fill(fill_h.volume, True, dt, p)
                    if fill_h.volume >= close_command_volume(p):
                        closing = True
                        close_t = t
                elif t < close_t + p.valve_close_delay + extra_close_delay:
                    filling = True
                    fill_h.volume = step_fill(fill_h.volume, True, dt, p)
                else:
                    fill_h.filled = True
                    if drip_volume > 0.0:
                        events.append(Event(t=t, kind="valve_drip", station=i_fill,
                                            data={"volume_m3": drip_volume}))

            # --- 各ホルダのスロッシング ---
            f_world_x = 0.0
            f_world_y = 0.0
            torque_slosh = 0.0
            for i, hd in enumerate(holders):
                if not hd.has_bottle or hd.volume <= 0.0:
                    continue
                h = hd.height(p)
                # 充填で h が上がると w1 が変わる。毎ステップ計算し直す。
                w1 = slosh_omega(p.R, h, p.g, EPS1)
                L1 = pendulum_length(w1, p.g)
                step_slosh(hd.slosh, a_t, a_r, omega, alpha, w1, p.zeta, dt, p.g)

                tilt = hd.slosh.tilt()
                if tilt > max_tilt:
                    max_tilt = tilt
                if in_index:
                    max_tilt_index = max(max_tilt_index, tilt)
                else:
                    max_tilt_dwell = max(max_tilt_dwell, tilt)

                hd.volume, spilled = apply_spill(
                    hd.volume, tilt, p.R, p.cross_section, p.body_height)
                hd.spilled = hd.spilled + spilled

                # 液がボトルに返す力（局所座標）を世界座標へ回して合成する。
                # liquid_force は反作用（テーブルが受ける側）を返すので、
                # 空瓶の慣性力も同じ向きに揃える（テーブルが受けるのは -m*a）。
                fl_t, fl_r = liquid_force(hd.slosh, hd.volume, a_t, a_r,
                                          w1, p.zeta, L1, p)
                m_empty = p.bottle_empty_mass
                loc_t = fl_t - m_empty * a_t
                loc_r = fl_r - m_empty * a_r
                ang = p.station_world_angle(i, th_t)
                f_world_x += loc_r * math.cos(ang) - loc_t * math.sin(ang)
                f_world_y += loc_r * math.sin(ang) + loc_t * math.cos(ang)

                # 揺れる分がテーブル軸に返すトルク（m0 の分は J_load 側にある）
                m_liq = p.rho * hd.volume
                m1 = m_liq * slosh_mass_ratio(p.R, h)
                a1_t, _a1_r = slosh_accel(hd.slosh, w1, p.zeta, L1, p.g)
                torque_slosh += -p.Rp * m1 * a1_t

            # --- 軸のトルク（液の揺れる分は反力として別に足す）---
            volumes = [hd.volume for hd in holders]
            present = [hd.has_bottle for hd in holders]
            # 揺れる分 m1 は反力トルク（torque_slosh）として別に入るので、
            # 慣性側からは抜く。両方入れると二重計上になる。
            j_load = rigid_load_inertia(volumes, p, present)
            # 充填中は液が増える。公称条件では停留中なので omega = 0 で項は消える。
            dj_dt = load_inertia_rate(fill_h.volume, p) if filling else 0.0
            t_table = table_torque(j_load, omega, alpha, p,
                                   t_slosh=torque_slosh, dj_dt=dj_dt)
            t_input = input_shaft_torque(t_table, table_ratio(psi, p), p, omega)

            # --- 垂直荷重（充填中はジェットの運動量も足す）---
            extra_loads = []
            if filling:
                extra_loads.append((p.station_world_angle(i_fill, th_t),
                                    jet_force(p, jet_fall_height)))
            m_x, m_y, m_abs = vertical_bending_moment(volumes, th_t, p, present,
                                                      extra_loads)

            if t >= next_log:
                records.append(Record(
                    t=t,
                    psi=psi,
                    th_t=th_t,
                    omega=omega,
                    alpha=alpha,
                    th_m=motor_angle(t, p),
                    j_load=j_load,
                    torque_table=t_table,
                    torque_input=t_input,
                    torque_slosh=torque_slosh,
                    m_bend=m_abs,
                    m_bend_x=m_x,
                    m_bend_y=m_y,
                    f_tab_x=f_world_x,
                    f_tab_y=f_world_y,
                    volume=list(volumes),
                    height=[hd.height(p) for hd in holders],
                    phi_t=[hd.slosh.phi_t for hd in holders],
                    phi_r=[hd.slosh.phi_r for hd in holders],
                    spill=[hd.spilled for hd in holders],
                    present=list(present),
                ))
                next_log = next_log + log_interval

        th_base = th_base + p.index_angle

    flag = linearity_flag(max_tilt)
    if flag != "ok":
        warnings.append(
            f"最大傾き {math.degrees(max_tilt):.1f} deg は線形スロッシングの"
            f"適用範囲を超えている（判定: {flag}）")

    events.sort(key=lambda e: e.t)
    return Result(records=records, events=events, max_tilt=max_tilt,
                  max_tilt_index=max_tilt_index, max_tilt_dwell=max_tilt_dwell,
                  linearity=flag, warnings=warnings)


# =======================================================================
# 手で確かめるとき用の出力
# =======================================================================


def summary(p: params_mod.Params) -> list[str]:
    """諸元から一意に決まる量を並べる。テストの期待値はここと同じ式から出る。"""
    lines = []
    lines.append(f"カム曲線 {p.cam_curve}: Ca = {MS_CA:.5f}, Cv = {MS_CV:.5f}"
                 f"  (json: {p.curve_Ca_ref} / {p.curve_Cv_ref})")
    lines.append(f"タクト {p.tact:.1f}s = 割出し {p.index_time:.1f}s + 停留 "
                 f"{p.dwell_time:.1f}s、入力軸 {p.input_shaft_speed:.4f} rad/s")
    lines.append(f"テーブル最大角速度   {table_omega_max(p):.5f} rad/s")
    lines.append(f"テーブル最大角加速度 {table_alpha_max(p):.5f} rad/s^2")
    lines.append(f"最大接線加速度       {tangential_accel_max(p):.5f} m/s^2 "
                 f"({tangential_accel_max(p) / p.g * 1e3:.1f} mg)")

    volumes, present = steady_load(p)
    j = load_inertia(volumes, p, present)
    n_full = sum(1 for v, h in zip(volumes, present) if h and v > 0.0)
    n_empty_bottle = sum(1 for v, h in zip(volumes, present) if h and v == 0.0)
    n_bare = sum(1 for h in present if not h)
    lines.append(f"テーブル板 {p.plate_mass:.3f} kg、素の慣性 {p.bare_inertia:.4f} kg m^2 "
                 f"（幾何から {p.polar_inertia_from_geometry:.4f}）")
    lines.append(f"工程 供給 {math.degrees(p.infeed_angle):.0f}deg / 充填 "
                 f"{math.degrees(p.fill_angle):.0f}deg / 排出 "
                 f"{math.degrees(p.discharge_angle):.0f}deg、"
                 f"滞留 {residence_moves(p)} 割出し = "
                 f"{residence_moves(p) * p.tact:.1f} s")
    lines.append(f"定常状態（満量{n_full} + 空瓶{n_empty_bottle} + 空ホルダ{n_bare}）"
                 f"の J_load = {j:.5f} kg m^2")
    lines.append(f"テーブル軸ピークトルク（剛体換算 J_load*alpha_max）"
                 f" = {j * table_alpha_max(p):.5f} Nm")
    run = simulate(p, n_cycles=2, dt=2.0e-4, log_rate=1000.0)
    t_tab_peak = max(abs(r.torque_table) for r in run.records)
    lines.append(f"  通しで拾うと {t_tab_peak:.5f} Nm"
                 f"（剛体換算の {t_tab_peak / (j * table_alpha_max(p)):.4f} 倍。"
                 f"粘性 + スロッシング反力ぶん。2 タクト・dt = 0.2 ms）")
    lines.append(f"ピーク機械出力 = {peak_mechanical_power(p):.4f} W")
    t_bare, psi_bare = input_shaft_torque_peak(p, viscous=False, drag=False)
    t_in, psi_at = input_shaft_torque_peak(p)
    lines.append(f"入力軸ピークトルク: 慣性負荷だけ {t_bare:.4f} Nm "
                 f"(psi = {math.degrees(psi_bare):.1f}deg) -> 粘性・引きずり込み "
                 f"{t_in:.4f} Nm (psi = {math.degrees(psi_at):.1f}deg)")

    h = p.fill_height
    w1 = slosh_omega(p.R, h, p.g)
    f1 = w1 / (2.0 * math.pi)
    lines.append(f"満量 液深 {h * 1e3:.1f} mm、w1 = {w1:.4f} rad/s -> f1 = {f1:.5f} Hz、"
                 f"T = {2 * math.pi / w1:.5f} s")
    lines.append(f"L1 = {pendulum_length(w1, p.g) * 1e3:.3f} mm、"
                 f"揺れる質量比 {slosh_mass_ratio(p.R, h):.4f}")

    om = table_omega_max(p)
    split = mode_split_hz(om)
    half = resonance_half_width_hz(w1, p.zeta)
    lines.append(f"共振半値幅 {half:.4f} Hz。コリオリ分裂は Omega の関数で、"
                 f"割出し中 0 -> ±{split:.4f} Hz (±{om / w1 * 100:.2f}%) と変わる "
                 f"= 半値幅の 0 -> {split / half:.1f} 倍")
    n, f_h, det = nearest_harmonic_detuning(f1, p.tact)
    lines.append(f"最寄り高調波 {n}/{p.tact:.1f}s = {f_h:.4f} Hz、"
                 f"離調 {det:.4f} Hz = 半値幅の {det / half:.1f} 倍")

    lines.append(f"充填ジェット {jet_force(p):.4f} N（流速 {p.nozzle_velocity:.3f} m/s）")

    peak_c, dwell_c, _e = index_slosh_response(p, coriolis=True, dt=1.0e-4)
    peak_n, dwell_n, _e = index_slosh_response(p, coriolis=False, dt=1.0e-4)
    lines.append(f"割出し 1 回の傾き: ピーク {peak_c * 1e3:.2f} mrad "
                 f"(dz/R = {math.tan(peak_c):.4f})、停留の残留 {dwell_c * 1e3:.2f} mrad")
    lines.append(f"  コリオリを落とすと ピーク {peak_n * 1e3:.2f} / 残留 "
                 f"{dwell_n * 1e3:.2f} mrad（残留が {dwell_n / dwell_c:.3f} 倍に増える）")
    return lines


if __name__ == "__main__":
    p = params_mod.load()
    for line in summary(p):
        print(line)
    for warn in (dt_warning(p),):
        if warn:
            print("警告:", warn)
