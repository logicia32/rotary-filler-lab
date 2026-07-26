"""物理コアが吐くバイナリの読み込み（core/FORMAT.md 準拠）。

対応している版は FORMAT.md 0 節の表のとおり。

| magic      | ヘッダ長 | スカラ数 | ステーション毎 |
|------------|---------|---------|--------------|
| `RFILL001` | 64      | 6       | 5            |
| `RFILL002` | 128     | 13      | 9            |

版が上がると並びもヘッダ長も変わるので、**magic で分岐する**。読めない版は素直に落とす。

* magic が違う           → 別物として弾く
* 版が未対応             → その旨を言って落ちる
* スカラ数・要素数が違う → 期待している並びを添えて落ちる
* ファイル長が合わない   → 端数バイト数を言って落ちる

黙って読み替えると、後段のセンサ信号が静かに間違う。そちらのほうが厄介なので、
迷ったら落とす側に倒してある。

想定より項目が増えているだけなら strict=False で先頭の既知ぶんを取れる。
未対応の版でも allow_unknown_version=True で読めるが、そのときは
FORMAT.md が保証している不変量（`h == V/(pi R^2)`、`spill` 単調、時刻の刻み）で
検算して、合わなければ例外にする。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

# RFILL004（rev.3 の現行形式）は core/dumpio.py の canonical な読み込みに委譲する。
# このファイルが直に知っているのは旧 001 / 002 だけ。004 は magic で振り分ける
# （read_dump() が入口。004 は dumpio、001/002 は read_legacy_dump）。
_CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
import dumpio as _dumpio                    # noqa: E402  core/dumpio.py
from dumpio import read_events, Events      # noqa: E402  イベント列（RFEVT002）

# 呼び出し側が canonical な型も名前で引けるように再エクスポートする。
# 004 の Dump/DumpHeader は dumpio 側（列は小文字・符号は FORMAT.md 正典）。
CanonicalDump = _dumpio.Dump
CanonicalDumpHeader = _dumpio.DumpHeader

MAGIC_PREFIX = b"RFILL"

# 版ごとの並び。増える方向にしか変わらない前提は置かず、版ごとに全部書く。
LAYOUTS = {
    "001": {
        "header_bytes": 64,
        "n_scalars": 6,
        "n_per_station": 5,
        "scalars": ("t", "th_t", "omega", "alpha", "motor_torque", "motor_current"),
        "station": ("V", "h", "phi_t", "phi_r", "spill"),
    },
    "002": {
        "header_bytes": 128,
        "n_scalars": 13,
        "n_per_station": 9,
        "scalars": ("t", "th_t", "omega", "alpha", "motor_torque", "motor_current",
                    "F_tab_t", "F_tab_r", "T_slosh", "a_bear", "bl_slip", "T_bl",
                    "J_load"),
        "station": ("V", "h", "phi_t", "phi_r", "spill",
                    "F_liq_t", "F_liq_r", "dz_over_R", "range_flag"),
    },
}
SUPPORTED_VERSIONS = tuple(LAYOUTS)

# 001 でも 002 でも同じ位置にある最低限の項目
BASE_SCALARS = LAYOUTS["001"]["scalars"]
BASE_STATION = LAYOUTS["001"]["station"]


class DumpFormatError(Exception):
    """core/FORMAT.md の想定と合わないバイナリを読もうとした。"""


@dataclass
class DumpHeader:
    magic: bytes
    version: str
    header_bytes: int
    record_bytes: int
    n_stations: int
    n_scalars: int
    n_per_station: int
    n_records: int
    dt_s: float
    log_dt_s: float
    pitch_radius_m: float
    bottle_radius_m: float
    body_height_m: float
    max_tilt_deg: float
    # 002 で増えたぶん。001 では既定値のまま。
    max_dz_over_R: float = 0.0
    range_exceeded: int = 0
    range_limit: float = 0.20
    fault_flags: int = 0
    tact_s: float = 0.0
    index_time_s: float = 0.0
    dwell_s: float = 0.0
    gear_ratio: float = 0.0
    bearing_ring_freq_hz: float = 0.0
    bearing_defect_ratio: float = 0.0
    target_volume_m3: float = 0.0
    flow_rate_m3_s: float = 0.0
    imbalance_station: int = 0
    imbalance_ratio: float = 1.0
    w1_full_rad_s: float = 0.0

    @property
    def nominal_log_rate_hz(self) -> float:
        """ヘッダに書いてある公称のログ周波数。

        実際の刻みは dt の整数倍に丸められていて、これとは一致しないことがある
        （core/FORMAT.md 6 節）。周波数軸を作るときは :attr:`Dump.log_rate_hz`
        （時刻列から測った実際の値）を使うこと。
        """
        return 1.0 / self.log_dt_s

    @property
    def fault_fill(self) -> bool:
        return bool(self.fault_flags & 1)

    @property
    def fault_bearing(self) -> bool:
        return bool(self.fault_flags & 2)

    @property
    def fault_backlash(self) -> bool:
        return bool(self.fault_flags & 4)


@dataclass
class Dump:
    """1 回ぶんのログ。配列はすべて float64 に上げてある（元は f32）。

    002 で増えた項目は、001 のダンプでは None になる。使う前に有無を確かめること。
    """

    header: DumpHeader
    t: np.ndarray
    th_t: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    motor_torque: np.ndarray     # モータ軸
    motor_current: np.ndarray
    V: np.ndarray                # [時刻, ステーション] m^3
    h: np.ndarray
    phi_t: np.ndarray
    phi_r: np.ndarray
    spill: np.ndarray
    # --- 002 ---
    F_tab_t: np.ndarray | None = None    # 世界固定のセンサ方位（世界角 0）での接線成分
    F_tab_r: np.ndarray | None = None    # 同 半径成分
    T_slosh: np.ndarray | None = None    # テーブル軸。m1 の分だけ
    a_bear: np.ndarray | None = None     # 軸受衝撃（ログ周波数では折り返している）
    bl_slip: np.ndarray | None = None
    T_bl: np.ndarray | None = None       # モータ軸
    J_load: np.ndarray | None = None
    F_liq_t: np.ndarray | None = None    # [時刻, ステーション] 局所座標
    F_liq_r: np.ndarray | None = None
    dz_over_R: np.ndarray | None = None
    range_flag: np.ndarray | None = None
    warnings: list = field(default_factory=list)

    @property
    def n_records(self) -> int:
        return len(self.t)

    @property
    def n_stations(self) -> int:
        return self.V.shape[1]

    @property
    def log_dt_s(self) -> float:
        """実際のログ間隔 [s]。

        コアは `round(1/(log_rate_hz*dt))` ステップごとに書くので、公称値どおりには
        ならないことがある（刻み 2.0e-5 のときは 4000 Hz の指定に対して実際は 3846.15 Hz。
        刻み 2.5e-5 ならちょうど 4000 Hz）。周波数軸をここで間違えると、
        全部のスペクトルが数 % ずれる。

        **時刻列から測ってはいけない。** 時刻は f32 なので、記録が長くなると
        分解能が落ちる。480 s の記録では 1 LSB が 6.1e-5 s あり、刻み 2.5e-4 s が
        2.44e-4 s（4096 Hz）に丸まって 2.3 % ずれる。
        そこでヘッダの `dt_s` と `log_dt_s`（どちらも f64）から
        1 レコードあたりの積分ステップ数を整数で復元し、そこから作り直す。
        """
        head = self.header
        dt = float(head.dt_s)
        nominal = float(head.log_dt_s)
        if dt > 0.0 and nominal > 0.0:
            steps = round(nominal / dt)
            if steps >= 1:
                exact = steps * dt
                # 復元した値が、時刻列から測った値と桁違いでないことだけ確かめる
                measured = float(np.median(np.diff(self.t))) if self.t.size > 1 else exact
                if measured > 0.0 and abs(exact - measured) / measured < 0.05:
                    return exact
        return float(np.median(np.diff(self.t)))

    @property
    def log_rate_hz(self) -> float:
        """実際のログ周波数 [Hz]（公称値ではない）。"""
        return 1.0 / self.log_dt_s

    @property
    def has_reaction(self) -> bool:
        """コアが反力を出しているか（002 以降）。"""
        return self.F_tab_t is not None

    def summary(self) -> str:
        return (
            f"版 {self.header.version} / {self.n_records} レコード / "
            f"{self.n_stations} ステーション / {self.log_rate_hz:.1f} Hz / "
            f"{self.t[-1] - self.t[0]:.3f} s"
        )


def parse_header(raw: bytes, allow_unknown_version: bool = False) -> DumpHeader:
    """ヘッダを解釈する。版ごとに読む長さが違う。"""
    if len(raw) < 64:
        raise DumpFormatError(f"ヘッダに届かない（{len(raw)} バイトしかない）")
    magic = bytes(raw[:8])
    if not magic.startswith(MAGIC_PREFIX):
        raise DumpFormatError(
            f"magic が {magic!r}。物理コアの出力ではない（先頭は {MAGIC_PREFIX!r} のはず）"
        )
    version = magic[5:8].decode("ascii", errors="replace")
    known = version in LAYOUTS
    if not known and not allow_unknown_version:
        raise DumpFormatError(
            f"フォーマット版 {version} は未対応（対応: {', '.join(SUPPORTED_VERSIONS)}）。"
            " core/FORMAT.md 0 節の表を見て、この読み込み側に版を足すこと。"
            " 先頭の並びが変わっていないほうに賭けるなら allow_unknown_version=True"
        )

    u32 = np.frombuffer(raw, dtype="<u4", count=6, offset=8)
    hb, rb, nst, nsc, nps, nrec = (int(v) for v in u32)
    f64 = np.frombuffer(raw, dtype="<f8", count=2, offset=32)
    f32 = np.frombuffer(raw, dtype="<f4", count=4, offset=48)

    if hb < 64:
        raise DumpFormatError(f"header_bytes = {hb} が 64 未満")
    if len(raw) < hb:
        raise DumpFormatError(f"ヘッダ {hb} バイトに満たないファイル（{len(raw)} バイト）")
    if nst < 1:
        raise DumpFormatError(f"n_stations = {nst} が不正")
    expect_rb = 4 * (nsc + nps * nst)
    if rb != expect_rb:
        raise DumpFormatError(
            f"record_bytes = {rb} がヘッダの項目数と合わない"
            f"（4*({nsc} + {nps}*{nst}) = {expect_rb} のはず）"
        )
    if not (f64[0] > 0.0) or not (f64[1] > 0.0):
        raise DumpFormatError(f"dt_s = {f64[0]} / log_dt_s = {f64[1]} が不正")
    if known and hb != LAYOUTS[version]["header_bytes"]:
        raise DumpFormatError(
            f"版 {version} のヘッダ長は {LAYOUTS[version]['header_bytes']} のはずが {hb}"
        )

    head = DumpHeader(
        magic=magic, version=version, header_bytes=hb, record_bytes=rb,
        n_stations=nst, n_scalars=nsc, n_per_station=nps, n_records=nrec,
        dt_s=float(f64[0]), log_dt_s=float(f64[1]),
        pitch_radius_m=float(f32[0]), bottle_radius_m=float(f32[1]),
        body_height_m=float(f32[2]), max_tilt_deg=float(f32[3]),
    )
    if hb >= 128:   # 002 の拡張部
        head.max_dz_over_R = float(np.frombuffer(raw, "<f4", count=1, offset=64)[0])
        head.range_exceeded = int(np.frombuffer(raw, "<u4", count=1, offset=68)[0])
        head.range_limit = float(np.frombuffer(raw, "<f4", count=1, offset=72)[0])
        head.fault_flags = int(np.frombuffer(raw, "<u4", count=1, offset=76)[0])
        (head.tact_s, head.index_time_s, head.dwell_s, head.gear_ratio,
         head.bearing_ring_freq_hz, head.bearing_defect_ratio,
         head.target_volume_m3, head.flow_rate_m3_s) = (
            float(v) for v in np.frombuffer(raw, "<f4", count=8, offset=80))
        head.imbalance_station = int(np.frombuffer(raw, "<u4", count=1, offset=112)[0])
        head.imbalance_ratio = float(np.frombuffer(raw, "<f4", count=1, offset=116)[0])
        head.w1_full_rad_s = float(np.frombuffer(raw, "<f4", count=1, offset=120)[0])
    return head


def _validate_layout(dump: Dump) -> None:
    """FORMAT.md が保証している不変量で、想定した並びが正しいかを確かめる。

    未対応の版を allow_unknown_version で読んだときの安全網。
    """
    R = dump.header.bottle_radius_m
    if not (R > 0):
        raise DumpFormatError(f"ヘッダの bottle_radius_m = {R} が不正")
    live = dump.V > 1e-9
    if np.any(live):
        pred = dump.V[live] / (np.pi * R ** 2)
        err = float(np.max(np.abs(dump.h[live] - pred) / pred))
        if err > 1e-3:
            raise DumpFormatError(
                f"h と V/(pi*R^2) が最大 {err:.3g} 食い違う。"
                " レコードの並びが FORMAT.md 2 節と違う"
            )
    dspill = np.diff(dump.spill, axis=0)
    if dspill.size and float(np.min(dspill)) < -1e-12:
        raise DumpFormatError("spill が単調非減少になっていない。並びが想定と違う")
    for nm, arr in (("phi_t", dump.phi_t), ("phi_r", dump.phi_r)):
        if arr.size and float(np.max(np.abs(arr))) > np.pi:
            raise DumpFormatError(f"{nm} が |pi| を超えている。並びが想定と違う")


def read_legacy_dump(path: str | os.PathLike, strict: bool = True,
                     allow_unknown_version: bool = False) -> Dump:
    """旧 RFILL001 / 002 を読んで :class:`Dump` を返す。

    rev.3 の現行形式 RFILL004 はここではなく :func:`read_dump` 経由で
    core/dumpio へ回る。この関数は古い保存ダンプ（すべて rev.3 前）専用。
    """
    with open(path, "rb") as fp:
        raw = fp.read()

    head = parse_header(raw, allow_unknown_version=allow_unknown_version)
    warnings: list[str] = []
    unknown = head.version not in LAYOUTS
    layout = LAYOUTS.get(head.version, LAYOUTS["001"])
    if unknown:
        warnings.append(
            f"フォーマット版 {head.version} は core/FORMAT.md に記述が無い。"
            " 先頭 6 スカラとステーション先頭 5 要素の並びが 001 のままだと仮定して読んだ"
            "（不変量では検算済み。FORMAT.md を見て版を足すこと）"
        )
        strict = False

    want_sc, want_st = layout["n_scalars"], layout["n_per_station"]
    if head.n_scalars < len(BASE_SCALARS) or head.n_per_station < len(BASE_STATION):
        raise DumpFormatError(
            f"項目が足りない（n_scalars={head.n_scalars},"
            f" n_per_station={head.n_per_station}）。"
            f" 最低でも スカラ {BASE_SCALARS}、ステーション {BASE_STATION} が要る"
        )
    if not unknown and (head.n_scalars != want_sc or head.n_per_station != want_st):
        msg = (f"版 {head.version} の項目数は {want_sc} / {want_st} のはずが"
               f" {head.n_scalars} / {head.n_per_station}。"
               " core/FORMAT.md と実装のどちらかが古い")
        if strict:
            raise DumpFormatError(msg + "。先頭の既知ぶんだけで読み進めるなら strict=False")
        warnings.append(msg + "。先頭の既知ぶんだけを取り出した")

    body = len(raw) - head.header_bytes
    n_from_size, rest = divmod(body, head.record_bytes)
    if rest:
        raise DumpFormatError(
            f"レコード長 {head.record_bytes} でファイルが割り切れない（端数 {rest} バイト）。"
            " 書き込み途中か、フォーマットが変わっている"
        )
    n_records = head.n_records
    if n_records == 0:
        n_records = n_from_size      # 標準出力へ吐いたときは書き戻せない（FORMAT.md 1 節）
        warnings.append(
            f"ヘッダの n_records が 0。ファイル長から {n_records} レコードとみなした")
    elif n_records != n_from_size:
        raise DumpFormatError(
            f"ヘッダの n_records = {n_records} とファイル長から数えた {n_from_size} が食い違う")
    if n_records < 2:
        raise DumpFormatError(f"レコードが {n_records} 個しかない。信号として扱えない")

    n_cols = head.record_bytes // 4
    arr = np.frombuffer(raw, dtype="<f4", count=n_records * n_cols,
                        offset=head.header_bytes).reshape(n_records, n_cols).astype(np.float64)
    scal = arr[:, :head.n_scalars]
    st = arr[:, head.n_scalars:].reshape(n_records, head.n_stations, head.n_per_station)

    def sc(i):
        return scal[:, i].copy()

    def stn(i):
        return st[:, :, i].copy()

    dump = Dump(
        header=head,
        t=sc(0), th_t=sc(1), omega=sc(2), alpha=sc(3),
        motor_torque=sc(4), motor_current=sc(5),
        V=stn(0), h=stn(1), phi_t=stn(2), phi_r=stn(3), spill=stn(4),
        warnings=warnings,
    )
    if not unknown and head.version == "002":
        dump.F_tab_t, dump.F_tab_r = sc(6), sc(7)
        dump.T_slosh, dump.a_bear = sc(8), sc(9)
        dump.bl_slip, dump.T_bl, dump.J_load = sc(10), sc(11), sc(12)
        dump.F_liq_t, dump.F_liq_r = stn(5), stn(6)
        dump.dz_over_R, dump.range_flag = stn(7), stn(8)

    if unknown:
        _validate_layout(dump)

    dt_med = dump.log_dt_s
    if dt_med <= 0:
        raise DumpFormatError("時刻が単調増加していない")
    rel = abs(dt_med - head.log_dt_s) / head.log_dt_s
    if rel > 1e-3:
        dump.warnings.append(
            f"実際のログ周波数は {1/dt_med:.2f} Hz（公称 {head.nominal_log_rate_hz:.1f} Hz、"
            f"差 {rel*100:.1f}%）。コアは dt の整数倍で間引くので公称どおりにはならない"
            "（FORMAT.md 6 節）。周波数軸には実測値を使う")
    if head.range_exceeded:
        dump.warnings.append(
            f"適用範囲の逸脱あり（dz/R 最大 {head.max_dz_over_R:.3f} >"
            f" {head.range_limit:.2f}、傾き {head.max_tilt_deg:.1f} deg）。"
            " 1 次モード近似の外なので、この区間の値はそのまま信じない")
    elif head.max_tilt_deg > 11.3:
        dump.warnings.append(
            f"最大傾き {head.max_tilt_deg:.1f} deg。線形スロッシングの適用範囲の外")
    if head.fault_bearing and head.bearing_ring_freq_hz > 0.5 * head.nominal_log_rate_hz:
        dump.warnings.append(
            f"軸受のリンギング {head.bearing_ring_freq_hz:.0f} Hz がログのナイキスト"
            f" {0.5 * head.nominal_log_rate_hz:.0f} Hz を超える。"
            " ログの a_bear は折り返しているので波形としては使えない"
            "（センサ層はイベントとして作り直す）")
    return dump


def peek_magic(path: str | os.PathLike) -> bytes:
    """先頭 8 バイトだけ読む。どの読み込みへ回すかの振り分けに使う。"""
    with open(path, "rb") as fp:
        return fp.read(8)


def read_dump(path: str | os.PathLike, strict: bool = True,
              allow_unknown_version: bool = False):
    """物理コアのダンプを読む。**形式は magic で振り分ける。**

    - ``RFILL004``（rev.3 現行）→ core/dumpio へ委譲し、canonical な
      :class:`dumpio.Dump` を返す（列は小文字・符号は FORMAT.md 正典。
      ``torque_input`` / ``torque_slosh`` / ``f_tab_x`` / ``f_tab_y`` /
      ``present`` など。旧名 ``motor_current`` / ``T_slosh`` / ``F_tab_t``
      は**持たない**。読み側でそのまま canonical 名に合わせること）。
    - ``RFILL001`` / ``002`` → :func:`read_legacy_dump`（古い保存ダンプ専用）。

    返す :class:`Dump` の型が版で変わる点に注意。rev.3 のコアが吐くのは 004 だけで、
    001/002 は撤去済みの機械の遺物なので、実運用では常に 004（canonical）が返る。
    ``strict`` / ``allow_unknown_version`` は 001/002 経路にだけ効く。
    """
    magic = peek_magic(path)
    if magic[:8] == _dumpio.DUMP_MAGIC:          # RFILL004
        return _dumpio.read_dump(path)
    return read_legacy_dump(path, strict=strict,
                            allow_unknown_version=allow_unknown_version)


def load_run(path: str | os.PathLike):
    """連続ログとサイドカーのイベント列をまとめて読む。

    RFILL004 は衝撃（軸受・カム当たり）を連続ログに載せず ``<out>.events``
    （RFEVT002）へ出す。センサ層はこの両方を要る。戻り値は
    ``(dump, events)``。イベント列が無ければ ``events`` は ``None``。
    """
    dump = read_dump(path)
    ev_path = str(path) + ".events"
    events = read_events(ev_path) if os.path.exists(ev_path) else None
    return dump, events


if __name__ == "__main__":  # 手で確かめる用
    target = sys.argv[1] if len(sys.argv) > 1 else "core/out.bin"
    loose = len(sys.argv) > 2
    d = read_dump(target, strict=not loose, allow_unknown_version=loose)
    hd = d.header

    if isinstance(d, CanonicalDump):        # RFILL004（現行）
        print(f"{target}: {d.summary()}")
        print(f"  tact {hd.tact_s} / 割出し {hd.index_time_s} / 減速比 {hd.gear_ratio:g}")
        print(f"  最大傾き {hd.max_tilt_rad*1e3:.3f} mrad / dz/R {hd.max_dz_over_R:.4f}"
              f" (限界 {hd.range_limit:.2f}, 逸脱 {hd.range_exceeded})")
        print(f"  故障 flags {hd.fault_flags:#06b} (欠品 {hd.fault_missing} / 弁 {hd.fault_valve}"
              f" / カム {hd.fault_cam} / 軸受 {hd.fault_bearing})")
        print(f"  T_in 最大 {d.torque_input.max():.5f} Nm / T_slosh 最大"
              f" {np.abs(d.torque_slosh).max():.5f} Nm / 水平合力 最大"
              f" {np.max(np.hypot(d.f_tab_x, d.f_tab_y)):.4g} N")
        print(f"  在荷（最終） {np.array2string(d.present[-1])}")
        ev_path = str(target) + ".events"
        if os.path.exists(ev_path):
            from collections import Counter
            from dumpio import EV_KINDS
            ev = read_events(ev_path)
            c = Counter(EV_KINDS.get(int(k), str(k)) for k in ev.kind)
            print(f"  イベント {len(ev)} 件: {dict(c)}")
    else:                                    # 旧 RFILL001 / 002
        print(f"{target}: {d.summary()}")
        print(f"  dt {hd.dt_s} s / log {hd.log_dt_s} s / tact {hd.tact_s} / index {hd.index_time_s}")
        print(f"  最大傾き {hd.max_tilt_deg:.3f} deg / dz/R {hd.max_dz_over_R:.4f}"
              f" (限界 {hd.range_limit:.2f}, 逸脱 {hd.range_exceeded})")
        print(f"  電流 {d.motor_current.min():.3f} 〜 {d.motor_current.max():.3f} A")
        if d.has_reaction:
            print(f"  水平合力 最大 {np.max(np.hypot(d.F_tab_t, d.F_tab_r)):.4g} N /"
                  f" T_slosh 最大 {np.abs(d.T_slosh).max():.4g} Nm")
    for w in d.warnings:
        print(f"  注意: {w}")
