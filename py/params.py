"""params.json を読むだけの薄いローダ。

単位変換はここに全部集める。この先のコード（ref.py / テスト）は SI 単位しか見ない。
json の生の値は `raw` から参照できるが、計算に使うのは SI の属性のほうにする。
数値をこのファイルに直書きしないこと。ここにあってよいのは単位の係数だけ。

params.json rev.3（カム式インデックスユニット＋誘導ギヤモータ）に合わせてある。
rev.2 から消えたもの: `motor`（サーボ諸元一式）、`cycle.profile` / `cycle.accel_fraction`、
`control` のループ周波数。新設: `machine` / `stations` / `indexer` / `drive` / `viz`。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# json のキーに付いている単位を SI に直すための係数
MM_TO_M = 1.0e-3
ML_TO_M3 = 1.0e-6           # 1 mL = 1 cm^3 = 1e-6 m^3
RPM_TO_RAD_S = 2.0 * math.pi / 60.0
DEG_TO_RAD = math.pi / 180.0

# params.json はラボ直下にある（このファイルは py/ の下）
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.json"


class _Group:
    """json の 1 セクションを属性で引けるようにするだけの入れ物。

    生の値（単位付きのキー名のまま）を見たいときに使う。
    """

    def __init__(self, name: str, data: dict):
        self._name = name
        self._data = dict(data)

    def __getattr__(self, key: str):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"{self._name} に {key!r} は無い") from None

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def __repr__(self) -> str:
        return f"<{self._name}: {', '.join(sorted(self._data))}>"


class Params:
    """params.json を SI 単位で持つ。

    属性名には単位を付けない（全部 SI なので付ける意味がない）。
    元の値がどのキーから来たかは、下の代入を読めば分かるようにしてある。
    """

    def __init__(self, data: dict, path: Path | None = None):
        self.raw = data
        self.path = path

        mac = data["machine"]
        b = data["bottle"]
        liq = data["liquid"]
        tbl = data["table"]
        sta = data["stations"]
        idx = data["indexer"]
        cyc = data["cycle"]
        drv = data["drive"]
        ctl = data["control"]
        fil = data["fill"]
        sim = data["sim"]

        # 生セクション（source 欄などを読みたいとき用）
        self.machine = _Group("machine", mac)
        self.bottle = _Group("bottle", b)
        self.liquid = _Group("liquid", liq)
        self.table = _Group("table", tbl)
        self.stations = _Group("stations", sta)
        self.indexer = _Group("indexer", idx)
        self.cycle = _Group("cycle", cyc)
        self.drive = _Group("drive", drv)
        self.control = _Group("control", ctl)
        self.fill = _Group("fill", fil)
        self.sim = _Group("sim", sim)
        self.sensors = _Group("sensors", data.get("sensors", {}))
        self.viz = _Group("viz", data.get("viz", {}))
        # 故障は名前引きの辞書のまま持つ（有効・無効の判定は ref.py 側）
        self.faults = {k: _Group(f"faults.{k}", v)
                       for k, v in data.get("faults", {}).items()
                       if isinstance(v, dict)}

        # --- 機械 ---
        self.throughput_bpm = mac["throughput_bpm"]

        # --- ボトル [m, kg] ---
        self.bottle_inner_diameter = b["inner_diameter_mm"] * MM_TO_M
        self.R = self.bottle_inner_diameter / 2.0          # 内半径。スロッシングの R
        self.body_height = b["body_height_mm"] * MM_TO_M   # 円筒部の高さ = こぼれ判定の上限
        self.shoulder_height = b["shoulder_height_mm"] * MM_TO_M
        self.neck_diameter = b["neck_diameter_mm"] * MM_TO_M
        self.neck_height = b["neck_height_mm"] * MM_TO_M
        self.wall_thickness = b["wall_thickness_mm"] * MM_TO_M
        self.bottle_empty_mass = b["empty_mass_kg"]
        # fill_height は下の「充填」で target_volume から導く。
        # 液深と充填量は同じ量の言い換えなので、params.json には片方しか置いていない。

        # --- 液 ---
        self.rho = liq["density_kg_m3"]
        self.nu = liq["kinematic_viscosity_m2_s"]
        self.zeta = liq["slosh_damping_ratio"]
        self.liquid_name = liq["name"]

        # --- テーブル ---
        self.pitch_circle_diameter = tbl["pitch_circle_diameter_mm"] * MM_TO_M
        self.Rp = self.pitch_circle_diameter / 2.0         # ボトル中心の回転半径
        self.n_stations = int(tbl["stations"])
        self.index_angle = tbl["index_angle_deg"] * DEG_TO_RAD
        self.plate_diameter = tbl["plate_diameter_mm"] * MM_TO_M
        self.plate_thickness = tbl["plate_thickness_mm"] * MM_TO_M
        self.plate_density = tbl["plate_density_kg_m3"]
        self.plate_mass = tbl["plate_mass_kg"]
        self.bare_inertia = tbl["bare_inertia_kg_m2"]      # 極慣性 (1/2) m r^2

        # --- ステーション（世界角 [rad]）---
        self.infeed_angle = sta["infeed_deg"] * DEG_TO_RAD
        self.fill_angle = sta["fill_deg"] * DEG_TO_RAD
        self.discharge_angle = sta["discharge_deg"] * DEG_TO_RAD

        # --- カム索引ユニット ---
        self.cam_curve = idx["cam_curve"]
        self.index_angle_input = idx["index_angle_input_deg"] * DEG_TO_RAD
        self.dwell_angle_input = idx["dwell_angle_input_deg"] * DEG_TO_RAD
        # 曲線定数は json の値をそのまま持つ（参照用）。ref.py は解析式から作り直した
        # 値を使い、両者が一致することをテストで見る。
        self.curve_Ca_ref = idx["curve_Ca"]
        self.curve_Cv_ref = idx["curve_Cv"]
        # 摩擦の 3 つは出所が無い（json の _friction_grade: "assumed"）。仮置き。
        self.cam_efficiency = idx["efficiency"]
        self.input_drag_torque = idx["input_drag_torque_Nm"]
        self.table_viscous = idx["table_viscous_Nms_rad"]

        # --- サイクル [s] ---
        self.tact = cyc["tact_s"]
        self.index_time = cyc["index_time_s"]
        # 停留時間は導出値。params.json に重複した値を置くと、片方を直し忘れて食い違う。
        self.dwell_time = self.tact - self.index_time
        # 下の 3 つは json 側が持っている確定値。ref.py はカム曲線から作り直すので、
        # ここは突き合わせのための参照値として持つだけ。
        self.table_omega_max_ref = cyc["table_omega_max_rad_s"]
        self.table_alpha_max_ref = cyc["table_alpha_max_rad_s2"]
        self.tangential_accel_max_ref = cyc["tangential_accel_max_m_s2"]

        # --- 駆動系（誘導ギヤモータ＋インバータ）---
        self.gear_ratio = drv["gear_ratio"]
        self.output_speed = drv["output_rpm"] * RPM_TO_RAD_S       # 入力軸（カム軸）[rad/s]
        self.motor_speed = drv["motor_rpm_at_operating_point"] * RPM_TO_RAD_S
        self.motor_rated_speed_50hz = drv["motor_rated_rpm_50hz"] * RPM_TO_RAD_S
        self.inverter_hz = drv["inverter_hz_at_operating_point"]
        self.motor_rated_output = drv["motor_rated_output_W"]
        self.motor_poles = int(drv["poles"])

        # --- 制御（開ループのタイミングだけ）---
        self.control_architecture = ctl["architecture"]
        self.plc_scan = ctl["plc_scan_ms"] * 1.0e-3

        # --- 充填 ---
        self.target_volume = fil["target_volume_mL"] * ML_TO_M3
        self.flow_rate = fil["flow_rate_mL_s"] * ML_TO_M3
        self.nozzle_diameter = fil["nozzle_diameter_mm"] * MM_TO_M
        self.nozzle_velocity_ref = fil["nozzle_velocity_m_s"]   # 参照値。実際は流量/断面積
        self.valve_open_delay = fil["valve_open_delay_s"]
        self.valve_close_delay = fil["valve_close_delay_s"]
        self.start_delay = fil["start_delay_s"]
        # 満量の液深。target_volume を円筒部の断面積で割っただけ。
        self.fill_height = self.target_volume / (math.pi * self.R ** 2)

        # --- 積分・出力 ---
        self.dt = sim["dt_s"]
        self.log_rate = sim["log_rate_hz"]
        self.frame_rate = sim["frame_rate_hz"]
        self.g = sim["gravity_m_s2"]

    # ---- そのまま使える派生量（単位換算と同じ場所に置いておく） ----

    @property
    def cross_section(self) -> float:
        """円筒部の断面積 [m^2]。h = V / cross_section で使う。"""
        return math.pi * self.R ** 2

    @property
    def cylinder_volume(self) -> float:
        """円筒部を body_height まで満たしたときの体積 [m^3]。"""
        return self.cross_section * self.body_height

    @property
    def station_pitch(self) -> float:
        """ステーション間の角度 [rad]。"""
        return 2.0 * math.pi / self.n_stations

    @property
    def fill_duration(self) -> float:
        """流量一定で target_volume を入れるのにかかる時間 [s]。"""
        return self.target_volume / self.flow_rate

    @property
    def nozzle_area(self) -> float:
        """ノズルの断面積 [m^2]。"""
        return math.pi * (self.nozzle_diameter / 2.0) ** 2

    @property
    def nozzle_velocity(self) -> float:
        """ノズル出口の流速 [m/s] = 流量 / 断面積。"""
        return self.flow_rate / self.nozzle_area

    @property
    def input_shaft_speed(self) -> float:
        """カム入力軸の角速度 [rad/s]。1 タクトで 1 回転する。"""
        return 2.0 * math.pi / self.tact

    @property
    def plate_mass_from_geometry(self) -> float:
        """板の寸法と密度から出した質量 [kg]。json の plate_mass_kg の検算用。"""
        r = self.plate_diameter / 2.0
        return math.pi * r * r * self.plate_thickness * self.plate_density

    @property
    def polar_inertia_from_geometry(self) -> float:
        """円板の極慣性 (1/2) m r^2 [kg m^2]。json の bare_inertia_kg_m2 の検算用。

        直径まわりの (1/4) m r^2 ではない。前版はここを取り違えていた。
        """
        r = self.plate_diameter / 2.0
        return 0.5 * self.plate_mass_from_geometry * r * r

    def liquid_mass(self, volume: float) -> float:
        """体積 [m^3] から液の質量 [kg]。"""
        return self.rho * volume

    def height_from_volume(self, volume: float) -> float:
        """円筒部にある前提での液面高さ [m]。"""
        return volume / self.cross_section

    def station_world_angle(self, i: int, th_t: float) -> float:
        """ホルダ i の世界角 [rad]。テーブル角 th_t のときの位置。"""
        return th_t + i * self.station_pitch

    def holder_at(self, world_angle: float, th_t: float) -> int:
        """世界角 world_angle にいるホルダ番号。割り切れない位置なら最寄りを返す。"""
        k = (world_angle - th_t) / self.station_pitch
        return int(round(k)) % self.n_stations

    def fault(self, name: str) -> _Group | None:
        """故障ブロックを名前で引く。無ければ None。"""
        return self.faults.get(name)

    def fault_enabled(self, name: str) -> bool:
        f = self.faults.get(name)
        return bool(f) and bool(f.get("enabled", False))

    def __repr__(self) -> str:
        return (f"<Params R={self.R * 1e3:.1f}mm Rp={self.Rp * 1e3:.0f}mm "
                f"stations={self.n_stations} tact={self.tact}s>")


def load(path: str | Path | None = None) -> Params:
    """params.json を読んで Params を返す。"""
    p = Path(path) if path is not None else DEFAULT_PARAMS_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Params(data, path=p)
