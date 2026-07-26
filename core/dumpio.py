"""`RFILL004` / `RFEVT002` を読む（core/FORMAT.md 準拠）。

`sensors/read_dump.py` は 001 / 002 しか知らず、rev.3 の書き直しの対象外だった。
`core/` のチェックスクリプトが共通で使う読み込みをここに置く。
**並びの正典は core/FORMAT.md** で、このファイルはそれを写しただけ。

読めない版・長さの合わないファイルは黙って読み替えず、素直に落とす。
静かに間違ったデータを下流へ流すほうが厄介なので、迷ったら落とす側に倒してある。

------------------------------------------------------------------------
外（sensors / analysis / viz）から使うとき
------------------------------------------------------------------------
numpy 以外に依存しない。パスを通して import するだけで使える。

    import sys, os
    sys.path.insert(0, os.path.join(ROOT, "core"))
    from dumpio import read_dump, read_events

    d = read_dump("core/out.bin")
    d.t, d.th_t, d.omega, d.V, d.phi_t, d.present   # 属性で引ける
    d.header.tact_s, d.header.max_tilt_rad          # ヘッダはデータクラス

`sensors/read_dump.py` は `RFILL001` / `002` 専用で、004 は読めない。
どちらを使うかは magic で分けること（`peek_magic()`）。
旧版の列名からの移行は `LEGACY_COLUMNS` に対応表がある（`describe_migration()` で表示）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

DUMP_MAGIC = b"RFILL004"
EV_MAGIC = b"RFEVT002"
DUMP_HEADER_BYTES = 160
EV_HEADER_BYTES = 64
EV_RECORD_BYTES = 56
EV_DATA = 8

# レコードの並び（FORMAT.md 2.1 / 2.2）。py/ref.py の Record と 1 対 1。
SCALARS = ("t", "psi", "th_t", "omega", "alpha", "th_m", "j_load",
           "torque_table", "torque_input", "torque_slosh",
           "m_bend", "m_bend_x", "m_bend_y", "f_tab_x", "f_tab_y")
STATION = ("V", "h", "phi_t", "phi_r", "spill")
# レコードの末尾に付く u8 の並び（1 ステーションに 1 個）
FLAGS = ("present",)

EV_KINDS = {
    0: "bearing_impulse",
    1: "cam_impact",
    2: "infeed",
    3: "infeed_missed",
    4: "discharge",
    5: "valve_drip",
}


# 旧版（RFILL002・サーボ機）の列名から、現行版への対応。
# 機構が変わって消えた列は None。移行するコードはこの表を機械的に当てればよい。
LEGACY_COLUMNS = {
    "th_t":          ("th_t", "同じ。ただし旧版は指令に追従した値、現行はカム形状そのもの"),
    "omega":         ("omega", "同じ"),
    "alpha":         ("alpha", "同じ"),
    "J_load":        ("j_load", "名前だけ小文字に。中身はどちらも液の m0 のみ"),
    "T_slosh":       ("torque_slosh", "**符号が逆**。現行はテーブルが受ける側"),
    "F_tab_t":       ("f_tab_y", "**世界座標 y。符号も逆**（旧版は世界 y だが向きが反対）"),
    "F_tab_r":       ("f_tab_x", "**世界座標 x。符号も逆**"),
    "motor_torque":  (None, "モータ軸トルクは列に無い。torque_input / gear_ratio で作る"),
    "motor_current": (None, "電流は出していない。トルク電流係数が params.json に無い"),
    "a_bear":        (None, "軸受衝撃は連続量に乗せない。イベント列（kind=0）を使う"),
    "bl_slip":       (None, "バックラッシュはこの機構に無い（カムは停留部で予圧される）"),
    "T_bl":          (None, "同上。カムフォロワ摩耗のイベント（kind=1）が代わり"),
    "F_liq_t":       (None, "ステーション毎の液の力は列から落とした。f_tab_* の合力だけ"),
    "F_liq_r":       (None, "同上"),
    "dz_over_R":     (None, "hypot(phi_t, phi_r) の tan で作る。ヘッダに最大値がある"),
    "range_flag":    (None, "ヘッダの range_exceeded / max_dz_over_R を見る"),
}

# 現行版で増えた列（旧版に対応が無いもの）
NEW_COLUMNS = ("psi", "th_m", "torque_table", "torque_input",
               "m_bend", "m_bend_x", "m_bend_y", "present")


def peek_magic(path) -> bytes:
    """先頭 8 バイトを読むだけ。どの読み込みへ回すかの振り分けに使う。"""
    with open(path, "rb") as fp:
        return fp.read(8)


def describe_migration() -> str:
    """旧版の列名からの移行表を文字列で返す（移行作業の手引き）。"""
    out = ["旧版（RFILL002）の列 -> 現行（RFILL004）"]
    for old, (new, note) in LEGACY_COLUMNS.items():
        out.append(f"  {old:<14} -> {new if new else '（無し）':<14}  {note}")
    out.append("  新しく増えた列: " + ", ".join(NEW_COLUMNS))
    return "\n".join(out)


class DumpFormatError(Exception):
    """core/FORMAT.md の想定と合わないバイナリを読もうとした。"""


@dataclass
class DumpHeader:
    magic: bytes
    header_bytes: int
    record_bytes: int
    n_stations: int
    n_scalars: int
    n_per_station: int
    elem_bytes: int
    n_records: int
    fault_flags: int
    dt_s: float
    log_dt_s: float
    pitch_radius_m: float
    bottle_radius_m: float
    body_height_m: float
    max_tilt_rad: float
    max_dz_over_R: float
    range_exceeded: int
    range_limit: float
    tact_s: float
    index_time_s: float
    dwell_s: float
    index_angle_rad: float
    input_shaft_speed_rad_s: float
    motor_speed_rad_s: float
    gear_ratio: float
    target_volume_m3: float
    flow_rate_m3_s: float
    w1_full_rad_s: float
    infeed_angle_rad: float
    fill_angle_rad: float
    discharge_angle_rad: float
    cam_efficiency: float
    input_drag_torque_Nm: float
    table_viscous_Nms_rad: float
    bearing_defect_freq_hz: float
    missing_station: int
    flag_bytes: int

    @property
    def fault_missing(self) -> bool:
        return bool(self.fault_flags & 1)

    @property
    def fault_valve(self) -> bool:
        return bool(self.fault_flags & 2)

    @property
    def fault_cam(self) -> bool:
        return bool(self.fault_flags & 4)

    @property
    def fault_bearing(self) -> bool:
        return bool(self.fault_flags & 8)


@dataclass
class Dump:
    """1 回ぶんの連続ログ。配列はすべて float64 に上げてある。"""

    header: DumpHeader
    scal: dict
    st: dict
    warnings: list = field(default_factory=list)

    def __getattr__(self, name):
        # t / psi / th_t ... と V / h / phi_t ... を属性で引けるようにする
        s = self.__dict__.get("scal") or {}
        if name in s:
            return s[name]
        q = self.__dict__.get("st") or {}
        if name in q:
            return q[name]
        raise AttributeError(name)

    @property
    def n_records(self) -> int:
        return len(self.scal["t"])

    @property
    def n_stations(self) -> int:
        return self.header.n_stations

    @property
    def log_dt_s(self) -> float:
        """実際のログ間隔 [s]。

        コアは `t >= next_log` で書くので、`log_dt` が `dt` の整数倍でないと
        刻みが揺れる。ヘッダの `dt_s` と `log_dt_s`（どちらも f64）から
        1 レコードあたりの積分ステップ数を整数で復元して作り直す。
        """
        dt = float(self.header.dt_s)
        nominal = float(self.header.log_dt_s)
        if dt > 0.0 and nominal > 0.0:
            steps = round(nominal / dt)
            if steps >= 1:
                return steps * dt
        t = self.scal["t"]
        return float(np.median(np.diff(t))) if t.size > 1 else nominal

    @property
    def log_rate_hz(self) -> float:
        """実際のログ周波数 [Hz]（公称ではなく :attr:`log_dt_s` から）。"""
        return 1.0 / self.log_dt_s

    def summary(self) -> str:
        t = self.scal["t"]
        return (f"{self.n_records} レコード / {self.n_stations} ステーション / "
                f"{1.0 / self.log_dt_s:.1f} Hz / {t[-1] - t[0]:.3f} s / "
                f"{'f64' if self.header.elem_bytes == 8 else 'f32'}")


def _u32(raw, off, n=1):
    return [int(v) for v in np.frombuffer(raw, "<u4", count=n, offset=off)]


def _i32(raw, off, n=1):
    return [int(v) for v in np.frombuffer(raw, "<i4", count=n, offset=off)]


def _f32(raw, off, n=1):
    return [float(v) for v in np.frombuffer(raw, "<f4", count=n, offset=off)]


def _f64(raw, off, n=1):
    return [float(v) for v in np.frombuffer(raw, "<f8", count=n, offset=off)]


def parse_header(raw: bytes) -> DumpHeader:
    if len(raw) < DUMP_HEADER_BYTES:
        raise DumpFormatError(f"ヘッダに届かない（{len(raw)} バイトしかない）")
    magic = bytes(raw[:8])
    if magic != DUMP_MAGIC:
        raise DumpFormatError(
            f"magic が {magic!r}。この読み込みは {DUMP_MAGIC!r} 専用。"
            " 旧版（RFILL001 / RFILL002）は sensors/read_dump.py、"
            " RFILL003 は 2026-07-23 の途中版で残っていない")
    hb, rb, nst, nsc, nps, eb, nrec, flags = _u32(raw, 8, 8)
    if hb != DUMP_HEADER_BYTES:
        raise DumpFormatError(f"header_bytes = {hb} が {DUMP_HEADER_BYTES} でない")
    if eb not in (4, 8):
        raise DumpFormatError(f"elem_bytes = {eb} が 4 でも 8 でもない")
    if nsc != len(SCALARS) or nps != len(STATION):
        raise DumpFormatError(
            f"項目数が {nsc} / {nps}。この版は {len(SCALARS)} / {len(STATION)} のはず。"
            " core/FORMAT.md と実装のどちらかが古い")
    fb, = _u32(raw, 156)
    if rb != eb * (nsc + nps * nst) + fb:
        raise DumpFormatError(
            f"record_bytes = {rb} がヘッダの項目数と合わない"
            f"（{eb}*({nsc} + {nps}*{nst}) + 在荷 {fb} のはず）")
    if fb < nst:
        raise DumpFormatError(f"在荷フラグの領域 {fb} バイトがステーション数 {nst} に足りない")
    dt, log_dt = _f64(raw, 40, 2)
    if not (dt > 0.0) or not (log_dt > 0.0):
        raise DumpFormatError(f"dt_s = {dt} / log_dt_s = {log_dt} が不正")
    Rp, R, bh, mt, mdzr = _f32(raw, 56, 5)
    rex, = _u32(raw, 76)
    (rlim, tact, itime, dwell, iang, iss, ms, gr, tv, fr, w1f,
     ia, fa, da, eff, drag, visc, bdf) = _f32(raw, 80, 18)
    miss, = _i32(raw, 152)
    return DumpHeader(
        magic=magic, header_bytes=hb, record_bytes=rb, n_stations=nst,
        n_scalars=nsc, n_per_station=nps, elem_bytes=eb, n_records=nrec,
        fault_flags=flags, dt_s=dt, log_dt_s=log_dt,
        pitch_radius_m=Rp, bottle_radius_m=R, body_height_m=bh,
        max_tilt_rad=mt, max_dz_over_R=mdzr, range_exceeded=rex, range_limit=rlim,
        tact_s=tact, index_time_s=itime, dwell_s=dwell, index_angle_rad=iang,
        input_shaft_speed_rad_s=iss, motor_speed_rad_s=ms, gear_ratio=gr,
        target_volume_m3=tv, flow_rate_m3_s=fr, w1_full_rad_s=w1f,
        infeed_angle_rad=ia, fill_angle_rad=fa, discharge_angle_rad=da,
        cam_efficiency=eff, input_drag_torque_Nm=drag,
        table_viscous_Nms_rad=visc, bearing_defect_freq_hz=bdf,
        missing_station=miss, flag_bytes=fb)


def read_dump(path: str | os.PathLike) -> Dump:
    """連続ログを読んで :class:`Dump` を返す。"""
    with open(path, "rb") as fp:
        raw = fp.read()

    head = parse_header(raw)
    warnings: list[str] = []

    body = len(raw) - head.header_bytes
    n_from_size, rest = divmod(body, head.record_bytes)
    if rest:
        raise DumpFormatError(
            f"レコード長 {head.record_bytes} でファイルが割り切れない（端数 {rest} バイト）")
    n_records = head.n_records
    if n_records == 0:
        n_records = n_from_size          # 標準出力へ吐いたときは書き戻せない
        warnings.append(f"ヘッダの n_records が 0。ファイル長から {n_records} とみなした")
    elif n_records != n_from_size:
        raise DumpFormatError(
            f"ヘッダの n_records = {n_records} とファイル長から数えた {n_from_size} が食い違う")
    if n_records < 2:
        raise DumpFormatError(f"レコードが {n_records} 個しかない")

    # レコード = 浮動小数の並び + 末尾の在荷フラグ（u8）。まずバイトで切る。
    body = np.frombuffer(raw, dtype=np.uint8, count=n_records * head.record_bytes,
                         offset=head.header_bytes).reshape(n_records, head.record_bytes)
    n_float_bytes = head.record_bytes - head.flag_bytes
    dtype = "<f4" if head.elem_bytes == 4 else "<f8"
    n_cols = n_float_bytes // head.elem_bytes
    arr = np.frombuffer(body[:, :n_float_bytes].tobytes(), dtype=dtype)
    arr = arr.reshape(n_records, n_cols).astype(np.float64)
    scal = {nm: arr[:, i].copy() for i, nm in enumerate(SCALARS)}
    st = arr[:, head.n_scalars:].reshape(n_records, head.n_stations, head.n_per_station)
    stn = {nm: st[:, :, i].copy() for i, nm in enumerate(STATION)}
    # 在荷フラグ（0/1）。詰め物は捨てる
    stn["present"] = body[:, n_float_bytes:n_float_bytes + head.n_stations].astype(np.int64)

    if head.range_exceeded:
        warnings.append(
            f"適用範囲の逸脱あり（dz/R 最大 {head.max_dz_over_R:.3f} >"
            f" {head.range_limit:.2f}）。1 次モード近似の外なので値をそのまま信じない")
    return Dump(header=head, scal=scal, st=stn, warnings=warnings)


@dataclass
class Events:
    header: dict
    t: np.ndarray
    kind: np.ndarray
    station: np.ndarray
    th_t: np.ndarray
    th_m: np.ndarray
    d: np.ndarray            # [件, 8]

    def of(self, kind: int) -> "Events":
        m = self.kind == kind
        return Events(self.header, self.t[m], self.kind[m], self.station[m],
                      self.th_t[m], self.th_m[m], self.d[m])

    def __len__(self) -> int:
        return len(self.t)


def read_events(path: str | os.PathLike) -> Events:
    """イベント列を読む（core/FORMAT.md 5 節）。"""
    raw = np.fromfile(path, dtype=np.uint8).tobytes()
    if len(raw) < EV_HEADER_BYTES or bytes(raw[:8]) != EV_MAGIC:
        raise DumpFormatError(f"{path}: magic が {bytes(raw[:8])!r}。イベント列ではない")
    hb, rb, n_ev, flags = _u32(raw, 8, 4)
    ring_hz, ring_zeta, accel, defect_hz, cam_tq, cam_clr = _f32(raw, 24, 6)
    dt, = _f64(raw, 48)
    dur, = _f32(raw, 56)
    if hb != EV_HEADER_BYTES or rb != EV_RECORD_BYTES:
        raise DumpFormatError(f"{path}: ヘッダ長 {hb} / レコード長 {rb} が想定と違う")
    head = dict(n_events=n_ev, fault_flags=flags,
                bearing_ring_freq_hz=ring_hz, bearing_ring_damping=ring_zeta,
                bearing_accel_m_s2=accel, bearing_defect_freq_hz=defect_hz,
                cam_impact_torque_Nm=cam_tq, cam_clearance_rad=cam_clr,
                dt_s=dt, duration_s=dur)
    body = raw[hb:]
    if len(body) % rb:
        raise DumpFormatError(f"{path}: レコード長 {rb} で割り切れない")
    n = len(body) // rb
    if n != n_ev:
        raise DumpFormatError(f"{path}: ヘッダの件数 {n_ev} とファイル長の {n} が違う")
    b = np.frombuffer(body, dtype=np.uint8).reshape(n, rb)
    t = np.frombuffer(b[:, 0:8].tobytes(), "<f8").copy()
    kind = np.frombuffer(b[:, 8:12].tobytes(), "<u4").astype(np.int64)
    station = np.frombuffer(b[:, 12:16].tobytes(), "<i4").astype(np.int64)
    th_t = np.frombuffer(b[:, 16:20].tobytes(), "<f4").astype(np.float64)
    th_m = np.frombuffer(b[:, 20:24].tobytes(), "<f4").astype(np.float64)
    d = np.frombuffer(b[:, 24:24 + 4 * EV_DATA].tobytes(), "<f4")
    d = d.reshape(n, EV_DATA).astype(np.float64)
    return Events(head, t, kind, station, th_t, th_m, d)


if __name__ == "__main__":   # 手で確かめる用
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--migration":
        print(describe_migration())
        raise SystemExit(0)
    target = sys.argv[1] if len(sys.argv) > 1 else "out.bin"
    d = read_dump(target)
    hd = d.header
    print(f"{target}: {d.summary()}")
    print(f"  dt {hd.dt_s} s / log {hd.log_dt_s} s / tact {hd.tact_s} / 割出し {hd.index_time_s}")
    print(f"  Rp {hd.pitch_radius_m:.4f} m / R {hd.bottle_radius_m:.4f} m / "
          f"胴高 {hd.body_height_m:.3f} m / 減速比 {hd.gear_ratio:g}")
    print(f"  最大傾き {hd.max_tilt_rad * 1e3:.3f} mrad / dz/R {hd.max_dz_over_R:.4f}"
          f" (限界 {hd.range_limit:.2f}, 逸脱 {hd.range_exceeded})")
    print(f"  在荷（最終） {np.array2string(d.present[-1])}")
    print(f"  故障 flags {hd.fault_flags:#06b} (欠品 {hd.fault_missing} / 弁 {hd.fault_valve}"
          f" / カム {hd.fault_cam} / 軸受 {hd.fault_bearing})")
    print(f"  omega max {np.abs(d.omega).max():.5f} rad/s / alpha max {np.abs(d.alpha).max():.5f}")
    print(f"  T_table max {np.abs(d.torque_table).max():.5f} Nm / "
          f"T_in max {d.torque_input.max():.5f} Nm / T_slosh max {np.abs(d.torque_slosh).max():.5f}")
    print(f"  M_bend {d.m_bend.min():.4f} 〜 {d.m_bend.max():.4f} Nm / "
          f"水平合力 max {np.max(np.hypot(d.f_tab_x, d.f_tab_y)):.4f} N")
    print(f"  液量 [mL] 最終 {np.array2string(d.V[-1] * 1e6, precision=2)}")
    for w in d.warnings:
        print(f"  注意: {w}")
    ev_path = str(target) + ".events"
    if os.path.exists(ev_path):
        ev = read_events(ev_path)
        from collections import Counter
        c = Counter(EV_KINDS.get(int(k), str(k)) for k in ev.kind)
        print(f"  イベント {len(ev)} 件: {dict(c)}")
