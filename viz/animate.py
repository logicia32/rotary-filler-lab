"""物理コアのダンプを読んで、コマ送りの PNG を書き出す。

絵にする値はダンプから来る。テーブル角は `th_t`、液量は `V_i`、
液面の傾きは `phi_t_i` / `phi_r_i` をそのまま渡す。見栄えのために数字を
作ることはしない。

例外は `--source cam`。カム式に書き直した物理コアがまだ無いので、その間だけ
カム曲線から作った仮の値で絵の作りを確かめられるようにしてある（下の
「仮の作り」の節）。**コアが出来たらこの道は消して `--source dump` に戻す。**

視点は 1 つに限らない。`viz/cameras.py` の表にある名前を並べて渡すと、
同じ 1 回のシミュレーションを複数の視点から同時に焼く。3D の視点と
2D のパネル（`viz/panels.py`）を混ぜて渡してもよく、どちらも同じ時刻列で
出るので、`viz/compose.py` で並べればそのまま合成アニメになる。

使い方
------
    .venv/bin/python viz/animate.py --dump core/out.bin
    .venv/bin/python viz/animate.py --start 7.2 --duration 4.8 --jobs 4
    .venv/bin/python viz/animate.py --camera top,iso,nozzle --out-dir frames
    .venv/bin/python viz/animate.py --camera iso,sensors --source cam --duration 3.0
    .venv/bin/python viz/animate.py --turntable 36 --turntable-mode hold

出力は `frames/` に `f0000.png` から連番。視点が 2 つ以上のときは
`frames/<視点名>/f0000.png` のように視点ごとのフォルダに分ける
（`--subdirs` で always / never にもできる）。`viz/make_gif.py` に渡すと GIF になる。

速さ
----
1 コマ 15〜25 秒（`scene.render()` を毎回呼ぶ場合）では動画にならない。
効いた順に:

1. **描画器を使い回す。** `pv.Plotter` を作るたびに環境マップの前計算と
   シェーダの構築が走る。ここが 1 コマぶんの大半で、使い回すと消える。
2. **回らない部品を作り直さない。** 架台・カバー・ノズル・中心柱は 1 回。
   テーブルと一緒に回る剛体（テーブル板・ボトル・接地部の暗がり）も 1 回作り、
   コマごとにアクタの変換行列を差し替えて回すだけにする。
   毎コマ組み直すのは液と液面の縁と充填中の液柱だけ。
3. **1 コマぶんのメッシュを組んだら、視点だけ変えて続けて焼く。** 液の組み直しは
   1 コマ 0.2〜0.5 s あるので、視点ごとに別々に流すとそこが視点の数だけ重複する。
   視点の切り替えはカメラを当てて撮るだけで、メッシュには触らない。
4. 解像度を下げる。
5. コマを分割して並列に回す。プロセスあたり描画器 1 つ。

同じ状態からは同じ絵が出る性質は保ってある（`scene.render_state()` が出す
PNG とバイト単位で一致することを確かめてある）。連番でちらつかない。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cameras  # noqa: E402
import panels  # noqa: E402
import scene  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))
from sensors.read_dump import read_dump  # noqa: E402

DEFAULT_DUMP = LAB_ROOT / "core" / "out.bin"
DEFAULT_OUT_DIR = LAB_ROOT / "frames"
DEFAULT_SIZE = (960, 720)
DEFAULT_CAMERA = "iso"

# 描く順は scene.DRAW_ORDER が唯一の表。ここには持たない。
# 以前はこちらにも同じ並びを写していて、部品を足したときに片方だけ古くなった。
# **テーブルと一緒に回る群も同じで、scene.rotating_groups() が唯一の表。**
# 自分の軸で回る群（スターホイール）は scene.spin_matrices() が別に当てる。

# 毎コマ中身が変わるグループ。
DYNAMIC = ("liquid", "rim", "stream")

# 液面の揺れの 1 次固有振動数 [Hz]。**params.json には無い。**
# params.json は機械と液の諸元表で、そこに載るのは水の物性値（密度・動粘度・
# 減衰比）まで。この値は容器の内半径と液深から出る「いまの液量での」揺れ方
# なので、機械の諸元ではなく計算結果であり、諸元表には置かない。物理コアを
# 書き直したら、ここではなくコアが液量から出した値を使う。
# 直書きは 1 箇所だけ（以前は animate と panels の 2 箇所に同じ数字があった）。
SLOSH_FREQ_HZ = 3.7514

# 2D パネルの既定の画素数。3D と並べる前提の縦横比（cameras.py の aspect）に合わせる。
PANEL_SIZE = {
    "bottle_xs": (540, 720),
    "cam": (768, 480),
    "sensors": (1056, 480),
}
# 波形パネルの 1 段あたりの高さ [px]。段が増えたぶんだけ縦に伸ばす。
PANEL_ROW_PX = 120


# --------------------------------------------------------------------------
# 描画器（1 プロセスに 1 つ持つ）
# --------------------------------------------------------------------------
class FrameRenderer:
    """描画器を 1 つ抱えて、状態を渡されるたびに視点の数だけ焼く。

    回らない部品と、テーブルと一緒に回る剛体は最初に 1 回だけ組む。
    以後のコマでは、回る側はアクタの変換行列を差し替え、液まわりだけ
    メッシュを差し替える。**メッシュを組み直すのは 1 コマにつき 1 回で、
    視点はそのあとカメラを当て替えて撮るだけ。**
    """

    def __init__(self, params: dict, lay: dict, size=DEFAULT_SIZE,
                 with_cover=False, with_floor=True, aa="ssaa"):
        self.params = params
        self.lay = lay

        meshes = dict(scene.build_static(params, lay, with_cover=with_cover,
                                         with_floor=with_floor))
        # 回る金属は別のアクタに分ける（変換行列が違うだけで材質は同じ）。
        # 群の読み替えは scene 側の表（CAROUSEL_ALIAS）に任せる。
        meshes.update(scene.build_carousel_meshes(params, lay, table_angle_rad=0.0))
        # 液まわりのアクタは下地の状態で作っておく。中身はコマごとに差し替える
        # ので形は何でもよいが、空だとアクタ自体が立たないので、液も液柱も
        # 必ず出る状態にする（ノズルの下のステーションは半分だけ入れておく）。
        n = lay["stations"]
        prime = scene.MachineState(
            table_angle_rad=0.0,
            volumes_mL=[params["fill"]["target_volume_mL"]] * n,
            tilt_t=[0.0] * n, tilt_r=[0.0] * n,
            filling_index=scene.station_under_nozzle(n, 0.0, lay.get("fill_deg")))
        prime.volumes_mL[prime.filling_index] *= 0.5
        meshes.update(scene.build_liquid(params, lay, prime, table_angle_rad=0.0))
        meshes["stream"] = scene.build_stream(params, lay, prime)

        # テーブルと一緒に回る群と、自分の軸で回る群。どちらも表は scene 側。
        self.rotating = scene.rotating_groups()
        self.spin_centers = dict(scene.SPIN_CENTERS)
        # 画角は機械の実測外形に合わせる。組んだメッシュをそのまま測るので、
        # ここで測っておけば以後どの視点でも組み直しは要らない。
        scene.ensure_extent(lay, meshes, params)

        self.plotter = scene.new_plotter(size)

        self.actors: dict = {}
        for name, material in scene.DRAW_ORDER:
            mesh = meshes.get(name)
            if mesh is None or mesh.n_points == 0:
                continue
            self.actors[name] = self.plotter.add_mesh(
                mesh, smooth_shading=True, split_sharp_edges=True,
                feature_angle=35.0, **scene.MATERIAL[material])

        scene.set_camera(self.plotter, lay, DEFAULT_CAMERA, params, meshes=meshes)
        # SSAA は内部で 2 倍の解像度に描くので、そのぶん時間も倍々で効く。
        # 輪郭の階段を我慢できるなら none がいちばん速い。
        if aa in (None, "none"):
            self.plotter.disable_anti_aliasing()
        else:
            self.plotter.enable_anti_aliasing(aa)

    # -- 中身の差し替え ----------------------------------------------------
    def _swap(self, name: str, mesh) -> None:
        """アクタを消さずに、中身のメッシュだけ入れ替える。

        `add_mesh` は法線を作る算法（`vtkAlgorithm`）を挟んでからマッパへ
        つなぐので、その入口を差し替えれば陰影の作り直しまで面倒を見てくれる。
        アクタを消して足し直すと描く順が変わってしまうので、それはしない。
        """
        actor = self.actors.get(name)
        if actor is None:
            return
        if mesh is None or mesh.n_points == 0:
            actor.SetVisibility(False)
            return
        actor.SetVisibility(True)
        actor.mapper.GetInputAlgorithm().SetInputDataObject(0, mesh)

    def set_state(self, state: scene.MachineState) -> None:
        """1 コマぶんのメッシュを組む。視点には触らない。"""
        liquid = scene.build_liquid(self.params, self.lay, state, table_angle_rad=0.0)
        self._swap("liquid", liquid.get("liquid"))
        self._swap("rim", liquid.get("rim"))
        # 液柱はノズルの下に立つので、テーブルとは一緒に回らない
        self._swap("stream", scene.build_stream(self.params, self.lay, state))

        mat = scene.rotation_matrix(state.table_angle_rad)
        for name in self.rotating:
            actor = self.actors.get(name)
            if actor is not None:
                actor.user_matrix = mat

        # 自分の軸で回る群（スターホイール）。テーブル軸まわりの回転では
        # 動かないので、中心へ寄せて回して戻す変換を別に当てる。
        # 姿勢の出どころは state.cam_angle_rad（カム入力軸角）。
        if self.spin_centers:
            for name, spin in scene.spin_matrices(self.params, self.lay,
                                                  state).items():
                actor = self.actors.get(name)
                if actor is not None:
                    actor.user_matrix = spin

    def shoot(self, cam, out_path) -> Path:
        """いま組んである状態を、指定の視点で 1 枚撮る。

        cam は視点の名前（`cameras.CAMERAS` のキー）か、`cameras.resolve()` が
        返した dict。ターンテーブルは後者を 1 コマずつ渡す。
        """
        if isinstance(cam, str):
            cameras.apply(self.plotter, self.lay, cam, self.params)
        else:
            cameras.apply_resolved(self.plotter, cam)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.render()
        self.plotter.screenshot(str(out_path))
        return out_path

    def draw(self, state: scene.MachineState, targets) -> list:
        """状態を 1 つ受け取って、視点の数だけ焼く。

        targets は (視点, 出力パス) の並び。パス 1 つだけを渡す古い呼び方
        （既定の視点で 1 枚）も受ける。
        """
        if isinstance(targets, (str, Path)):
            targets = [(DEFAULT_CAMERA, targets)]
        self.set_state(state)
        return [self.shoot(cam, path) for cam, path in targets]

    def close(self) -> None:
        self.plotter.close()


# --------------------------------------------------------------------------
# ダンプ -> コマ
# --------------------------------------------------------------------------
def pick_frames(dump, fps: float, start=None, stop=None) -> np.ndarray:
    """ログを表示コマ数まで間引く。返すのはレコード番号の列。

    ログは 4000 Hz、コマは 30 Hz なので 133 本に 1 本まで落ちる。補間はせず、
    各コマの時刻にいちばん近いレコードをそのまま使う。

    区間がダンプの外だと 1 コマも取れない。そのまま空を返すと呼び先が
    `states[0]` で落ちて、原因が分からない。ここで区間を添えて止める。
    """
    t = dump.t
    have = f"ダンプにあるのは {float(t[0]):.3f} 〜 {float(t[-1]):.3f} s"
    t0 = float(t[0]) if start is None else float(start)
    t1 = float(t[-1]) if stop is None else float(stop)
    if t1 <= t0:
        raise ValueError(f"区間が空: 開始 {t0:.3f} s に対し終了 {t1:.3f} s。{have}")
    if t0 > float(t[-1]) or t1 < float(t[0]):
        raise ValueError(f"区間 {t0:.3f} 〜 {t1:.3f} s がダンプの外。{have}")
    n = int(np.floor((t1 - t0) * float(fps))) + 1
    want = t0 + np.arange(n) / float(fps)
    want = want[want <= t[-1] + 1e-12]
    if want.size == 0:
        raise ValueError(f"区間 {t0:.3f} 〜 {t1:.3f} s から 1 コマも取れない"
                         f"（{fps:g} コマ/s）。{have}")
    idx = np.searchsorted(t, want)
    idx = np.clip(idx, 1, len(t) - 1)
    left = np.abs(t[idx - 1] - want) <= np.abs(t[idx] - want)
    idx[left] -= 1
    return idx


def filling_stations(dump) -> np.ndarray:
    """各レコードで「いま注がれている」ステーション番号。入っていなければ -1。

    液量が増えているかどうかだけで決める。充填の時刻をこちらで組み立てる
    のではなく、物理の出力から読む。
    """
    dV = np.diff(dump.V, axis=0, prepend=dump.V[:1])
    who = np.argmax(dV, axis=1)
    return np.where(dV[np.arange(len(who)), who] > 0.0, who, -1)


def states_from_dump(dump, indices: np.ndarray, tact_s=None) -> list:
    """レコード番号の列から、コマごとの `MachineState` を作る。

    液面の傾きは MODEL.md 2 節の `tilt_t = phi_t`, `tilt_r = phi_r` をそのまま
    使う（接線・半径の局所量。正でその向きの側が上がる）。

    カム入力軸の角はダンプに入っていないので、時刻から出す（入力軸は一定回転
    で、1 タクトで 1 回転する）。tact_s にはそのダンプのタクトを渡すこと。
    省くと自転する部品は割出しの始まりの姿勢で止まる。
    """
    fill = filling_stations(dump)
    out = []
    for k in indices:
        k = int(k)
        who = int(fill[k])
        psi = 0.0 if not tact_s else 2.0 * np.pi * float(dump.t[k]) / float(tact_s)
        out.append(scene.MachineState(
            table_angle_rad=float(dump.th_t[k]),
            volumes_mL=(dump.V[k] * 1e6).tolist(),
            tilt_t=dump.phi_t[k].tolist(),
            tilt_r=dump.phi_r[k].tolist(),
            filling_index=None if who < 0 else who,
            cam_angle_rad=psi))
    return out


# --------------------------------------------------------------------------
# 仮の作り（カム式の物理コアが出来るまでの間に合わせ）
#
# **物理コアの書き直しが終わったら、この節はまるごと core のダンプへ差し替える。**
# いま `core/out.bin` に入っているのは前の版（サーボ駆動・タクト 2.4 s）の
# 記録で、params.json のカム式・タクト 3.0 s とは別の機械のもの。カム曲線から
# 出る運動学だけは幾何そのもので正しいので、それを土台に置いてある。
# 液の反力も充填の流量計算も入っていないので、ここで作った値そのものに
# 物理的な意味は無い。パネルの見え方と時刻合わせを確かめるための足場でしかない。
# --------------------------------------------------------------------------
def cam_table_motion(params: dict, t):
    """カム曲線からテーブルの角度・角加速度・接線加速度を出す。ここは厳密。

    入力軸は一定回転で、割付 180 度でテーブルが 45 度進み、残り 180 度は停留。
    タクトをまたいで角度は積み上がる（`th_t` と同じ、回りっぱなしの角度）。

    **式は scene 側が持っている。** カム入力軸角 psi からテーブル角を出す道は
    静止画・コマ送り・スターホイールの自転で共通で、出どころは 1 つにしてある。
    ここは時刻を psi に直して渡すだけ。
    """
    psi = scene.cam_angle_from_time(params, t)
    pitch_r_m = float(params["table"]["pitch_circle_diameter_mm"]) / 2000.0
    theta = scene.table_angle_from_cam(params, psi)
    alpha = scene.table_alpha_from_cam(params, psi)
    return theta, alpha, pitch_r_m * alpha


def slosh_pendulum(t, drive, f_hz: float, zeta: float):
    """1 自由度の振り子。phi'' + 2 zeta w phi' + w^2 phi = -a/L1（MODEL.md 2 節）。

    `panels.py` の `__main__` に置いてある仮の応答と同じ式。前進オイラーで、
    刻みが揃っている前提。
    """
    t = np.asarray(t, dtype=float)
    w = 2.0 * np.pi * float(f_hz)
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


def cam_station_volumes(params: dict, lay: dict, t, theta):
    """仮の液量。ノズルの下に来たステーションを停留の中で満たしていく。

    流量計算はしていない。弁が開いてから閉じるまで一定の割合で増やし、
    排出ステーションを通ったら空にするだけ。液の入ったボトルが充填から
    排出へ運ばれていくことが絵で分かればよい、という程度の作り。
    """
    n = int(lay["stations"])
    step = 2.0 * np.pi / n
    tact = float(params["cycle"]["tact_s"])
    fil = params["fill"]
    target = float(fil["target_volume_mL"])
    t_fill = target / float(fil["flow_rate_mL_s"])
    t_open = float(params["cycle"]["index_time_s"]) + float(fil["valve_open_delay_s"])

    # 工程配置の控えは scene.station_deg() が 1 つだけ持っている
    fill_rad = np.radians(scene.station_deg(params, "fill"))
    disc_rad = np.radians(scene.station_deg(params, "discharge"))
    # 充填から排出まで何回の割出しがあるか
    steps_to_discharge = int(round(((disc_rad - fill_rad) % (2.0 * np.pi)) / step))

    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)
    frac = np.clip(((t % tact) - t_open) / t_fill, 0.0, 1.0)

    vol = np.zeros((t.size, n))
    for i in range(n):
        ang = theta + i * step
        # 充填ステーションを離れてから何回割り出したか（0 = いまノズルの下）
        since = np.round(((ang - fill_rad) % (2.0 * np.pi)) / step).astype(int) % n
        vol[:, i] = np.where(since == 0, target * frac,
                             np.where(since < steps_to_discharge, target, 0.0))
    return vol


def cam_run(params: dict, lay: dict, times):
    """仮のダンプ相当。コマの時刻列から、状態とパネル用の値をまとめて作る。

    返り値は (states, dense_t, ch, level_mm, tilt_rad)。
    dense_t と ch は波形パネル用の細かい時系列で、level / tilt はコマごとの値。
    """
    times = np.asarray(times, dtype=float)
    n = int(lay["stations"])
    zeta = float(params["liquid"]["slosh_damping_ratio"])
    f_slosh = float(params["liquid"].get("slosh_freq_hz", SLOSH_FREQ_HZ))
    fill_deg = lay.get("fill_deg")

    # 波形は 2000 Hz で作る。振り子を前進オイラーで解くので刻みは細かく取る
    fs = 2000.0
    dense_t = np.arange(times[0], times[-1] + 1.0 / fs, 1.0 / fs)
    _th_d, _al_d, a_t = cam_table_motion(params, dense_t)
    phi_d = slosh_pendulum(dense_t, a_t, f_slosh, zeta)
    vol_d = cam_station_volumes(params, lay, dense_t, _th_d)

    theta, _alpha, _a = cam_table_motion(params, times)
    phi = np.interp(times, dense_t, phi_d)
    vol = np.stack([np.interp(times, dense_t, vol_d[:, i]) for i in range(n)], axis=1)

    psi = scene.cam_angle_from_time(params, times)
    states = []
    for k, tk in enumerate(times):
        who = scene.station_under_nozzle(n, float(theta[k]), fill_deg)
        # 停留の中で液量が増えているコマだけ液柱を描く
        pouring = (k > 0) and (vol[k, who] > vol[k - 1, who] + 1e-9)
        states.append(scene.MachineState(
            table_angle_rad=float(theta[k]),
            volumes_mL=vol[k].tolist(),
            tilt_t=[float(phi[k])] * n,
            tilt_r=[0.0] * n,
            filling_index=who if pouring else None,
            cam_angle_rad=float(psi[k])))

    # ノズルの下のステーションの液量を波形にする（どのボトルかは時刻で変わるので、
    # 「いま注いでいる 1 本ぶん」として拾う）
    who_d = np.array([scene.station_under_nozzle(n, float(a), fill_deg)
                      for a in _th_d])
    vol_now = vol_d[np.arange(dense_t.size), who_d]
    ri_mm = float(params["bottle"]["inner_diameter_mm"]) / 2.0
    ch = {
        "accel_t [m/s2]": a_t,
        "phi_t [mrad]": phi_d * 1e3,
        "dz [mm]": ri_mm * np.tan(phi_d),
        "volume [mL]": vol_now,
    }
    level = np.interp(times, dense_t, vol_now) * 1000.0 / (np.pi * ri_mm ** 2)
    return states, dense_t, ch, level, phi


# --------------------------------------------------------------------------
# 2D パネル
# --------------------------------------------------------------------------
def dump_panel_feed(dump, indices: np.ndarray, params: dict, lay: dict, station: int):
    """ダンプから、パネルに流し込む値を取り出す。

    波形は 1 本のステーションを追う（割出しのたびに別のボトルが来るので、
    どのボトルを見るかを決めないと波形にならない）。
    """
    times = dump.t[indices]
    lo, hi = int(indices[0]), int(indices[-1]) + 1
    # 波形の点数は 4000 点までに落とす。これ以上あっても線図では潰れる
    stride = max(1, (hi - lo) // 4000)
    sl = slice(lo, hi, stride)
    dense_t = dump.t[sl]
    pitch_r_m = float(params["table"]["pitch_circle_diameter_mm"]) / 2000.0
    ri_mm = float(params["bottle"]["inner_diameter_mm"]) / 2.0

    ch = {
        "accel_t [m/s2]": pitch_r_m * dump.alpha[sl],
        "phi_t [mrad]": dump.phi_t[sl, station] * 1e3,
        "dz [mm]": ri_mm * np.tan(dump.phi_t[sl, station]),
        "volume [mL]": dump.V[sl, station] * 1e6,
    }
    if dump.motor_current is not None:
        ch["current [A]"] = dump.motor_current[sl]

    level = dump.V[indices, station] * 1e6 * 1000.0 / (np.pi * ri_mm ** 2)
    tilt = dump.phi_t[indices, station]
    return times, dense_t, ch, level, tilt


def render_panels(params: dict, lay: dict, name: str, times, dense_t, channels,
                  level_mm, tilt_rad, out_dir: Path, size=None, prefix="f",
                  ylim=None, title=None) -> list:
    """2D パネルの連番を焼く。3D と同じ時刻列を渡すこと。

    軸の範囲は `panels.fixed_ylim()` で一度だけ決めて全コマに渡す。コマごとに
    決め直すと軸が伸び縮みして、アニメがちらついて読めなくなる。
    """
    # 大きさの下限は視点ごとの表。**`--panel-size` を渡されたときも同じ下限を掛ける。**
    # 以前はそこが素通しで、指定したとたんに波形の軸名がタイトルへ重なり、
    # カム線図のタイトルが右端で切れた。パネルの字は px 数で決まらない
    # （matplotlib の点数が固定）ので、小さくすると字だけが相対的に大きくなって
    # はみ出す。段が増える波形パネルは、段数ぶんの高さも足す。
    floor = PANEL_SIZE.get(name, (960, 480))
    if name == "sensors":
        floor = (floor[0], max(floor[1], PANEL_ROW_PX * len(channels)))
    size = tuple(size) if size else floor
    if size[0] < floor[0] or size[1] < floor[1]:
        grown = (max(size[0], floor[0]), max(size[1], floor[1]))
        sys.stderr.write(
            f"注意: {name} パネルの {size[0]}x{size[1]} px では字がはみ出す。"
            f"{grown[0]}x{grown[1]} px に広げる"
            + (f"（{len(channels)} 段 x {PANEL_ROW_PX} px）\n" if name == "sensors"
               else "\n"))
        size = grown
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tact = float(params["cycle"]["tact_s"])
    paths = []
    t0 = time.time()
    for i, tk in enumerate(times):
        path = out_dir / f"{prefix}{i:04d}.png"
        if name == "bottle_xs":
            panels.bottle_cross_section(params, float(level_mm[i]), float(tilt_rad[i]),
                                        path, size_px=size, t_s=float(tk))
        elif name == "cam":
            # 入力軸は 1 タクトで 1 回転。時刻をそのまま入力軸角に直す
            panels.cam_diagram(params, 360.0 * (float(tk) % tact) / tact, path,
                               size_px=size)
        elif name == "sensors":
            panels.sensor_panel(dense_t, channels, float(tk), path, size_px=size,
                                ylim=ylim, title=title)
        else:
            raise ValueError(f"2D パネルではない視点: {name!r}")
        paths.append(path)
        el = time.time() - t0
        sys.stderr.write(f"\r  [{name}] {i + 1}/{len(times)} コマ  "
                         f"{el / (i + 1):.2f} s/コマ   ")
        sys.stderr.flush()
    sys.stderr.write("\n")
    return paths


# --------------------------------------------------------------------------
# 並列化
# --------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(params, lay, size, with_cover, with_floor, aa):
    _WORKER["renderer"] = FrameRenderer(params, lay, size=size, with_cover=with_cover,
                                        with_floor=with_floor, aa=aa)


def _render_one(job):
    number, state, targets = job
    _WORKER["renderer"].draw(state, targets)
    return number, len(targets)


def render_sequence(params, lay, states, out_dir: Path, size=DEFAULT_SIZE,
                    jobs=1, with_cover=False, with_floor=True, prefix="f",
                    aa="ssaa", cams=(DEFAULT_CAMERA,), dirs=None,
                    per_frame_cams=None) -> dict:
    """コマの列を焼く。視点ごとの PNG の並びを返す。進み具合は標準エラーへ。

    cams           : 視点の名前の並び。1 コマぶんのメッシュを組んだら、この順に
                     カメラだけ当て替えて焼く。
    dirs           : 視点ごとの出力先。省略すると out_dir をそのまま使う
                     （視点が 1 つのときの既存の呼び方と互換）。
    per_frame_cams : ターンテーブル用。コマ番号ごとの resolve 済みカメラの並び
                     を渡すと、cams の代わりにそれを当てる。
    """
    cams = list(cams)
    if dirs is None:
        dirs = {c: Path(out_dir) for c in cams}
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)

    out_paths = {c: [Path(dirs[c]) / f"{prefix}{i:04d}.png" for i in range(len(states))]
                 for c in cams}
    work = []
    for i, st in enumerate(states):
        if per_frame_cams is None:
            targets = [(c, out_paths[c][i]) for c in cams]
        else:
            targets = [(per_frame_cams[i], out_paths[cams[0]][i])]
        work.append((i, st, targets))

    total = len(work)
    per_frame = 1 if per_frame_cams is not None else len(cams)
    t0 = time.time()
    done = 0

    def report(done):
        el = time.time() - t0
        per = el / max(done, 1)
        eta = per * (total - done)
        sys.stderr.write(
            f"\r  {done}/{total} コマ  {per:.2f} s/コマ  "
            f"{per / per_frame:.2f} s/枚  経過 {el:6.1f}s  残り {eta:6.1f}s   ")
        sys.stderr.flush()

    if jobs <= 1:
        _init_worker(params, lay, size, with_cover, with_floor, aa)
        for job in work:
            _render_one(job)
            done += 1
            report(done)
        _WORKER["renderer"].close()
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(jobs, initializer=_init_worker,
                      initargs=(params, lay, size, with_cover, with_floor, aa)) as pool:
            for _ in pool.imap_unordered(_render_one, work, chunksize=1):
                done += 1
                report(done)
    sys.stderr.write("\n")
    return out_paths


# --------------------------------------------------------------------------
def split_cameras(names) -> tuple:
    """視点の名前を 3D とパネルに振り分ける。知らない名前はここで弾く。"""
    known3d = cameras.names("3d")
    known2d = cameras.names("2d")
    three, two = [], []
    for n in names:
        if n in known3d:
            three.append(n)
        elif n in known2d:
            two.append(n)
        else:
            raise SystemExit(f"知らない視点: {n}（ある視点: "
                             f"{', '.join(known3d + known2d)}）")
    return three, two


def _clear(out_dir: Path, prefix: str) -> int:
    old = sorted(Path(out_dir).glob(f"{prefix}*.png"))
    for p in old:
        p.unlink()
    return len(old)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ダンプを読んでコマ送りの PNG を書く")
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP, help="物理コアの出力")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="PNG の置き場")
    ap.add_argument("--camera", action="append", default=None,
                    help="視点。カンマ区切りか、複数回渡す。既定は iso。"
                         f"3D: {', '.join(cameras.names('3d'))} / "
                         f"パネル: {', '.join(cameras.names('2d'))}")
    ap.add_argument("--subdirs", choices=("auto", "always", "never"), default="auto",
                    help="視点ごとにフォルダを分けるか。auto は視点が 2 つ以上のとき分ける")
    ap.add_argument("--source", choices=("dump", "cam"), default="dump",
                    help="値の出どころ。dump は物理コアのダンプ、"
                         "cam はカム曲線から作る仮の値（コアの書き直し待ちの間に合わせ）")
    ap.add_argument("--fps", type=float, default=None,
                    help="コマ数 [Hz]。既定は params.json の sim.frame_rate_hz")
    ap.add_argument("--start", type=float, default=None, help="開始時刻 [s]")
    ap.add_argument("--stop", type=float, default=None, help="終了時刻 [s]")
    ap.add_argument("--duration", type=float, default=None,
                    help="長さ [s]（--stop の代わり）")
    ap.add_argument("--size", type=int, nargs=2, default=list(DEFAULT_SIZE),
                    metavar=("W", "H"), help="画素数")
    ap.add_argument("--panel-size", type=int, nargs=2, default=None,
                    metavar=("W", "H"),
                    help="2D パネルの画素数。既定は視点ごとの縦横比。"
                         "字が潰れる大きさは受け付けず、下限まで伸ばして標準エラーに出す"
                         f"（波形は 1 段 {PANEL_ROW_PX} px、カム線図は幅 "
                         f"{PANEL_SIZE['cam'][0]} px）")
    ap.add_argument("--panel-station", type=int, default=None,
                    help="波形と断面で追うステーション番号。既定は開始時刻に"
                         "ノズルの下にいるステーション")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                    help="並列数。プロセスあたり描画器 1 つ")
    ap.add_argument("--aa", choices=("ssaa", "fxaa", "none"), default="ssaa",
                    help="アンチエイリアス。none は 3 倍近く速いが輪郭が階段になる")
    ap.add_argument("--cover", action="store_true", help="安全カバーを付ける")
    ap.add_argument("--no-floor", action="store_true", help="床面を描かない")
    ap.add_argument("--prefix", default="f", help="連番の頭")
    ap.add_argument("--keep", action="store_true",
                    help="出力先に残っている古い PNG を消さない")
    ap.add_argument("--turntable", type=int, default=None, metavar="N",
                    help="視点を N コマで 1 周させる。3D の --camera は使わない")
    ap.add_argument("--turntable-base", default="iso",
                    help="ターンテーブルで回す元の視点")
    ap.add_argument("--turntable-mode", choices=("spin", "hold"), default="spin",
                    help="spin は機械を動かしながら回す / hold は 1 つの状態で止めて回す")
    ap.add_argument("--turntable-at", type=float, default=None,
                    help="hold で止める時刻 [s]。既定は開始時刻")
    args = ap.parse_args(argv)

    params = scene.load_params()
    lay = scene.derive_layout(params)
    fps = args.fps if args.fps is not None else float(params["sim"]["frame_rate_hz"])

    raw = args.camera if args.camera else [DEFAULT_CAMERA]
    names = [n.strip() for item in raw for n in str(item).split(",") if n.strip()]
    cams3d, cams2d = split_cameras(names)
    if args.turntable and cams3d:
        sys.stderr.write("注意: ターンテーブルなので --camera の 3D 視点は使わない\n")
        cams3d = []

    # ---- 時刻列と状態 ----------------------------------------------------
    dump = None
    indices = None
    if args.source == "dump":
        # 既定の道はここ。ダンプが無いまま素で叩かれることがいちばん多いので、
        # 生のトレースを出さずに次の一手を書く（他の行き止まりは全部そうしてある）。
        if not Path(args.dump).exists():
            raise SystemExit(
                f"{args.dump} が無い。物理コアの出力を読む道（--source dump）はここから始まる。\n"
                "  先に物理コアを回してダンプを作る（core/ で make && make run）か、\n"
                "  絵の作りだけ確かめるなら --source cam を付ける\n"
                "    .venv/bin/python viz/animate.py --source cam --camera iso,sensors "
                "--duration 3.0\n"
                "  （カム曲線から作った仮の値で、値そのものに物理的な意味は無い）")
        try:
            dump = read_dump(args.dump)
        except OSError as exc:                        # 読めるが壊れている・権限が無い
            raise SystemExit(f"{args.dump} が読めない（{exc}）") from None
        for w in dump.warnings:
            sys.stderr.write(f"注意: {w}\n")
        stop = args.stop
        if stop is None and args.duration is not None:
            stop = (dump.t[0] if args.start is None else args.start) + args.duration
        try:
            indices = pick_frames(dump, fps, start=args.start, stop=stop)
        except ValueError as exc:                     # 区間がダンプの外
            raise SystemExit(f"{args.dump}: {exc}") from None
        times = np.asarray(dump.t[indices], dtype=float)
        # カム入力軸の角は時刻から出す。ダンプ自身のタクトで割ること
        # （params.json のタクトと違う記録なら、そちらに合わせないとずれる）。
        tact_dump = float(getattr(dump.header, "tact_s", 0.0)) \
            or float(params["cycle"]["tact_s"])
        states = states_from_dump(dump, indices, tact_s=tact_dump)
        sys.stderr.write(
            f"{args.dump}: {dump.summary()}\n"
            f"  {times[0]:.3f} 〜 {times[-1]:.3f} s を {fps:g} コマ/s で "
            f"{len(states)} コマ（ログ {dump.log_rate_hz:.0f} Hz から "
            f"1/{dump.log_rate_hz / fps:.0f} に間引き）\n")
    else:
        t0 = 0.0 if args.start is None else float(args.start)
        dur = args.duration
        if dur is None:
            dur = (float(args.stop) - t0) if args.stop is not None \
                else float(params["cycle"]["tact_s"])
        times = t0 + np.arange(int(round(dur * fps)) + 1) / float(fps)
        states, dense_t, ch, level, tilt = cam_run(params, lay, times)
        sys.stderr.write(
            f"カム曲線から作った仮の値: {times[0]:.3f} 〜 {times[-1]:.3f} s を "
            f"{fps:g} コマ/s で {len(states)} コマ\n"
            "  注意: 物理コアの書き直しが済むまでの間に合わせで、"
            "値そのものに物理的な意味は無い\n")

    sys.stderr.write(f"  {args.size[0]}x{args.size[1]} / 並列 {args.jobs} / "
                     f"視点 {', '.join(names) if names else 'turntable'}\n")

    # ---- 出力先 ----------------------------------------------------------
    n_out = len(cams3d) + len(cams2d) + (1 if args.turntable else 0)
    split = args.subdirs == "always" or (args.subdirs == "auto" and n_out > 1)
    if args.subdirs == "never" and n_out > 1:
        raise SystemExit("--subdirs never では視点を 1 つしか焼けない（上書きになる）")

    def dir_for(name):
        return Path(args.out_dir) / name if split else Path(args.out_dir)

    dirs = {c: dir_for(c) for c in cams3d + cams2d}
    turn_dir = dir_for(f"turntable_{args.turntable_base}") if args.turntable else None
    if not args.keep:
        for d in list(dirs.values()) + ([turn_dir] if turn_dir else []):
            d.mkdir(parents=True, exist_ok=True)
            n_old = _clear(d, args.prefix)
            if n_old:
                sys.stderr.write(f"  {d}: 古い PNG を {n_old} 枚消した\n")

    made = {}
    t_all = time.time()

    # ---- 3D ---------------------------------------------------------------
    if cams3d:
        t0 = time.time()
        made.update(render_sequence(
            params, lay, states, Path(args.out_dir), size=tuple(args.size),
            jobs=args.jobs, with_cover=args.cover, with_floor=not args.no_floor,
            prefix=args.prefix, aa=args.aa, cams=cams3d, dirs=dirs))
        el = time.time() - t0
        n_shot = len(states) * len(cams3d)
        sys.stderr.write(
            f"  3D {len(cams3d)} 視点 x {len(states)} コマ = {n_shot} 枚 / "
            f"{el:.1f} s / {el / len(states):.2f} s/コマ / {el / n_shot:.2f} s/枚\n")

    # ---- ターンテーブル ---------------------------------------------------
    if args.turntable:
        n_turn = int(args.turntable)
        # 視点はここで 1 周ぶんまとめて決める。画角は機械の実測外形から出るので、
        # 先に測っておく（描画器は子プロセスの中なので、その中で測っても
        # ここには返ってこない。測っていないとテーブル径の控えで寄りすぎる）。
        scene.ensure_extent(lay, params=params)
        # 枠に機械を当てはめる視点は画面の横縦比で決め方が変わる。描画器は
        # 子プロセスの中なので、こちらで焼く大きさから横縦比を渡す。
        turn_cams = cameras.turntable(lay, n_turn, base=args.turntable_base,
                                      params=params,
                                      frame_aspect=args.size[0] / args.size[1])
        if args.turntable_mode == "hold":
            # 機械は止めて視点だけ回す。止める時刻のコマを取る
            if args.turntable_at is None:
                k = 0
            else:
                k = int(np.argmin(np.abs(times - float(args.turntable_at))))
            turn_states = [states[k]] * n_turn
            sys.stderr.write(f"  ターンテーブル(hold): t = {times[k]:.3f} s で止めて "
                             f"{n_turn} コマで 1 周\n")
        else:
            # 機械を動かしながら回す。選んだ区間を n_turn コマに割り振る
            pick = np.round(np.linspace(0, len(states) - 1, n_turn)).astype(int)
            turn_states = [states[i] for i in pick]
            sys.stderr.write(f"  ターンテーブル(spin): {times[0]:.3f} 〜 {times[-1]:.3f} s を "
                             f"{n_turn} コマに割り振って 1 周\n")
        t0 = time.time()
        got = render_sequence(
            params, lay, turn_states, turn_dir, size=tuple(args.size),
            jobs=args.jobs, with_cover=args.cover, with_floor=not args.no_floor,
            prefix=args.prefix, aa=args.aa, cams=[f"turntable_{args.turntable_base}"],
            dirs={f"turntable_{args.turntable_base}": turn_dir},
            per_frame_cams=turn_cams)
        made.update(got)
        el = time.time() - t0
        sys.stderr.write(f"  ターンテーブル {n_turn} 枚 / {el:.1f} s / "
                         f"{el / n_turn:.2f} s/枚\n")

    # ---- 2D パネル --------------------------------------------------------
    if cams2d:
        station = args.panel_station
        if station is None:
            station = scene.station_under_nozzle(lay["stations"],
                                                 states[0].table_angle_rad,
                                                 lay.get("fill_deg"))
        if args.source == "dump":
            _t, dense_t, ch, level, tilt = dump_panel_feed(dump, indices, params,
                                                           lay, station)
            title = "core dump"
            # ダンプのタクトと params.json のタクトが違えば、それは別の機械の記録。
            # 物理コアの書き直しが済むまでは食い違ったままになる。
            tact_dump = float(getattr(dump.header, "tact_s", 0.0))
            tact_now = float(params["cycle"]["tact_s"])
            if abs(tact_dump - tact_now) > 1e-6:
                sys.stderr.write(
                    f"注意: ダンプのタクトは {tact_dump:.3f} s で、params.json の "
                    f"{tact_now:.3f} s と違う（別の機械の記録）。"
                    "波形と 3D は同じダンプなので時刻は揃うが、"
                    "params.json から作る cam パネルとは対応しない\n")
        else:
            title = "placeholder (cam kinematics, not from the physics core)"
        # 軸は一度だけ決めて全コマに渡す。コマごとに決め直すとちらつく
        ylim = panels.fixed_ylim(ch)
        sys.stderr.write(f"  パネルの追うステーション: {station} / "
                         f"波形 {len(dense_t)} 点 / 軸は固定\n")
        for name in cams2d:
            paths = render_panels(params, lay, name, times, dense_t, ch, level, tilt,
                                  dirs[name], size=args.panel_size,
                                  prefix=args.prefix, ylim=ylim, title=title)
            made[name] = paths

    total_n = sum(len(v) for v in made.values())
    total_mb = sum(p.stat().st_size for v in made.values() for p in v) / 1e6
    sys.stderr.write(
        f"合計 {total_n} 枚 / {time.time() - t_all:.1f} s / {total_mb:.1f} MB\n")
    for name, paths in made.items():
        sys.stderr.write(f"  {name:22s} {len(paths):4d} 枚 -> {paths[0].parent}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
