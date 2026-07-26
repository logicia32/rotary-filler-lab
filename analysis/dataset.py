"""物理コアを回して、仮想センサの信号を作るところまで。

`analysis/` は物理コアにもセンサ層にも手を入れない。ここはその 2 つを呼ぶだけの層。

* 物理コアの実行     `core/filler`（別の担当が改修中なので、退避したコピーを実行する）
* ダンプの読み込み   `sensors.read_dump.load_run`（連続ログ RFILL004 ＋イベント列 RFEVT002）
* センサ信号の合成   `sensors.virtual.build_truth` / `synthesize`

ダンプは大きい（480 s で 623 MB）ので、リポジトリには入れず作業領域に置く。
使ったバイナリの md5 を控えて `RESULTS.md` に残す（コアの改修前後を後から判別するため）。
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sensors import virtual
# 004（現行）の canonical Dump は core/dumpio。read_dump.load_run が連続ログと
# サイドカーのイベント列（RFEVT002）をまとめて返す。旧 read_dump 直呼びはしない。
from sensors.read_dump import CanonicalDump as Dump, Events, load_run

from . import config

# 故障条件。`core/filler --help`（RFILL004・カム式）のフラグをそのまま並べる。
#
# rev.3 のカム式は単ノズルで、工程配置（供給315 / 充填0 / 排出225）が非対称なので、
# 故障が無くても水平合力は 0 にならない。旧サーボ機のように「8 本が揃うと打ち消えて 0」
# にはならないため、旧版で入れていた 1 % の充填ばらつき（基準を雑音床から離すための細工）
# は不要になった。正常運転（--no-faults）そのものを基準にする。
# 故障ビットの割り当ては bit0 欠品 / bit1 弁 / bit2 カム / bit3 軸受（core/FORMAT.md 1 節）。
CONDITIONS = {
    "normal": {
        "label": "正常（故障なし）",
        "args": ["--no-faults"],
        "note": "工程配置から作った定常状態（満量5本＋空瓶1本＋空ホルダ2つ）。"
                "非対称なので水平合力は 0 にならず、これを基準にする",
    },
    "missing": {
        "label": "欠品（1 ホルダに瓶が載らない）",
        "args": ["--fault-missing"],
        "note": "供給スターホイールの取りこぼしで特定ホルダが空のまま回る。満量 400 g が"
                "丸ごと欠ける（params.json faults.missing_bottle）。旧版の充填アンバランスの置換",
    },
    "valve": {
        "label": "弁閉じ遅れ",
        "args": ["--fault-valve"],
        "note": "充填弁の閉じが遅れて全瓶が等しく過充填になり、割出し中に液垂れが落ちる"
                "（params.json faults.valve_close_delay）",
    },
    "cam": {
        "label": "カムフォロワ摩耗",
        "args": ["--fault-cam"],
        "note": "予圧が抜けて割出しの入口・出口でカムリブに当たる。旧版のバックラッシュの置換"
                "（イベント列 kind=1・params.json faults.cam_follower_wear）",
    },
    "bearing": {
        "label": "軸受外輪傷",
        "args": ["--fault-bearing"],
        "note": "モータ軸の外輪傷。欠陥通過 89.5 Hz・リンギング 3 kHz。連続ログに乗せず"
                "イベント列 kind=0 で渡る（params.json faults.bearing_outer_race）",
    },
}

CHANNEL_LABELS = {
    "accel_lf_tangential": "accel_lf 接線",
    "accel_lf_radial": "accel_lf 半径",
    "accel_hf_radial": "accel_hf 半径",
    "strain": "strain",
    "current": "current",
}


def md5_of(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_core(name: str, workdir, filler: str, duration: float,
             params_path=None, reuse: bool = True, prefix: str = "",
             tact_s: float | None = None) -> Path:
    """物理コアを 1 条件ぶん回して、ダンプのパスを返す。

    すでに同じ名前のファイルがあれば作り直さない（reuse=False で強制）。
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tag = f"{prefix}{name}" + (f"_tact{tact_s:g}" if tact_s is not None else "")
    out = workdir / f"{tag}.bin"
    if reuse and out.exists() and out.stat().st_size > 0:
        return out
    cond = CONDITIONS[name]
    cmd = [str(filler), "--params", str(params_path or config.PARAMS_PATH),
           "--duration", f"{duration:g}", "--out", str(out)] + cond["args"]
    if tact_s is not None:
        cmd += ["--tact", f"{tact_s:g}"]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"filler が失敗した（{name}）:\n{res.stderr}")
    (workdir / f"{tag}.log").write_text(res.stderr, encoding="utf-8")
    print(f"  [core] {tag}: {time.time() - t0:.1f} s -> {out.name}")
    return out


# ---------------------------------------------------------------------------
# 読み込みとセンサ合成
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """1 条件ぶんの、解析に必要なものだけ。ダンプ本体は保持しない。"""

    name: str
    label: str
    dump_path: str
    dump_md5: str
    t_log: np.ndarray
    theta_log: np.ndarray
    log_rate_hz: float
    tact_s: float
    n_stations: int
    revolution_s: float
    peak_rev_per_s: float
    ring_freq_hz: float       # 軸受リンギング [Hz]。004 はイベント列ヘッダから取る
    defect_freq_hz: float     # 軸受欠陥通過 [Hz]。004 はダンプ／イベント列ヘッダから
    w1_full_rad_s: float
    max_dz_over_R: float
    range_exceeded: int
    fault_flags: int
    channels: dict = field(default_factory=dict)
    truth: object = None          # センサ連鎖に入る前の量。種を変えて作り直すときに使い回す
    bearing_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cam_times: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 旧 backlash_times
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def slosh_freq_hz(self) -> float:
        return self.w1_full_rad_s / (2.0 * np.pi)


def true_log_dt(dump: Dump) -> float:
    """ログの実際の刻み [s]。

    時刻列は f32 なので、100 s を超えたあたりでは 1 LSB が刻みの 3 % になり、
    差分の中央値から測ると 0.05 % ずれる。コアは dt の整数倍で間引くので、
    ヘッダの dt と公称ログ間隔から整数比を復元するほうが正確
    （core/FORMAT.md 6 節）。測った値と 0.1 % 以内で合うことを確かめてから使う。
    """
    h = dump.header
    step = int(round(h.log_dt_s / h.dt_s))
    exact = step * h.dt_s
    measured = float((dump.t[-1] - dump.t[0]) / (len(dump.t) - 1))
    if abs(measured - exact) / exact > 1e-3:
        raise ValueError(f"ログ刻みが合わない: 整数比から {exact}、時刻列から {measured}")
    return exact


def structure_variant(params: dict, model: str, freq_hz: float | None = None,
                      section: str = "accel_lf") -> dict:
    """params をメモリ上だけ差し替えて、構造モデルを変えた辞書を返す。

    params.json は書き換えない。加速度センサが力を加速度に変える伝達は
    架台をどうモデル化するかで決まり、そこが結果を支配する（SENSORS.md 3 節）。
    """
    import copy
    p = copy.deepcopy(params)
    sec = p["sensors"][section]
    sec["structure_model"] = model
    if freq_hz is not None:
        sec["structure_freq_hz"] = float(freq_hz)
    return p


def build_case(name: str, path, params: dict, seed: int,
               channels: tuple | None = None, keep_dump: bool = False):
    """ダンプとサイドカーのイベント列を読んで、センサ信号まで作る。

    返り値は (Case, dump)。keep_dump=False なら dump は None（大きいので捨てる）。
    """
    dump, events = load_run(path)
    case = case_from_dump(name, path, dump, events, params, seed, channels=channels)
    return (case, dump) if keep_dump else (case, None)


def case_from_dump(name: str, path, dump: Dump, events: Events | None,
                   params: dict, seed: int, channels: tuple | None = None) -> Case:
    """読み込み済みのダンプ（RFILL004 canonical）とイベント列からセンサ信号を作る。

    衝撃（軸受 kind=0 / カム当たり kind=1）は連続ログに無く、`events`（RFEVT002）に
    ある。`events` は :func:`sensors.read_dump.load_run` から渡す。
    """
    h = dump.header
    log_dt = true_log_dt(dump)
    # 時刻列を作り直してから真値を合成する。
    # ログの時刻は f32 精度なので、長い記録では 1 LSB が 6e-5 s あり、
    # 差分の中央値から刻みを測ると 4096 Hz に丸まってしまう。
    # そのままだとセンサ層のフィルタが 4096 Hz を前提に設計され、
    # 帯域制限のカットオフも周波数軸もずれる。整数比から復元した刻みで作り直す。
    dump.t = float(dump.t[0]) + np.arange(len(dump.t)) * log_dt
    t_log = dump.t

    truth = virtual.build_truth(dump, params)

    # 衝撃はイベント列から。bearing_events / cam_events は 004 で (events, params) 受け。
    bev = virtual.bearing_events(events, params)
    cam = virtual.cam_events(events, params)
    chans, truth = virtual.synthesize(dump, params, seed=seed, truth=truth,
                                      impacts=bev if len(bev) else None,
                                      channels=channels)
    cond = CONDITIONS.get(name, {"label": name})
    # 割出し中のテーブルのピーク角速度 [回転/s]。カム（変形正弦）の最大角速度は
    # params.json cycle.table_omega_max_rad_s（= Cv * 割出し角 / 割出し時間）で一意に決まる。
    # 旧サーボ機の台形速度（平均 index_angle/index_time を 1/(1-accel_fraction) 倍する）
    # は無くなり、accel_fraction キーも params.json に無い。
    peak_omega = float(params["cycle"]["table_omega_max_rad_s"])
    peak_rev = peak_omega / (2.0 * np.pi)

    warnings = list(dump.warnings)
    # 軸受リンギングは 004 ではダンプヘッダから消え、イベント列ヘッダへ移った。
    if events is None:
        ring_freq_hz = float("nan")
        warnings.append(
            "イベント列（<out>.events / RFEVT002）が無い。軸受リンギング周波数と"
            "衝撃時刻が取れない。004 の正規の出力にはサイドカーが要る（黙って 0 にはしない）")
    else:
        ring_freq_hz = float(events.header["bearing_ring_freq_hz"])
    return Case(
        name=name, label=cond.get("label", name), dump_path=str(path),
        dump_md5="", t_log=t_log, theta_log=np.asarray(dump.th_t, dtype=float),
        log_rate_hz=1.0 / log_dt, tact_s=float(h.tact_s),
        n_stations=int(h.n_stations),
        revolution_s=float(h.tact_s) * int(h.n_stations),
        peak_rev_per_s=peak_rev,
        ring_freq_hz=ring_freq_hz,
        defect_freq_hz=float(h.bearing_defect_freq_hz),
        w1_full_rad_s=float(h.w1_full_rad_s),
        max_dz_over_R=float(h.max_dz_over_R),
        range_exceeded=int(h.range_exceeded),
        fault_flags=int(h.fault_flags),
        channels=chans, truth=truth,
        bearing_times=bev.times.copy(),
        cam_times=cam.times.copy(),
        notes=list(truth.notes), warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 窓の切り方（角度で切る）
# ---------------------------------------------------------------------------

def reseed(case: Case, dump: Dump, params: dict, seed: int,
           channels: tuple | None = None, impacts=None) -> dict:
    """同じ真値のまま、センサ雑音の種だけ変えて信号を作り直す。

    物理コアは決定論的なので、同じ条件を何回回しても波形は 1 通りしかない。
    正常データのばらつきを作るには、雑音の種を振るしかない。
    真値の合成（重い）はやり直さないので速い。
    """
    chans, _ = virtual.synthesize(dump, params, seed=seed, truth=case.truth,
                                  impacts=impacts, channels=channels)
    return chans


@dataclass
class Segment:
    """解析の窓。整数回転ぶん。"""

    index: int
    theta0: float
    n_rev: int
    t0: float
    t1: float


def segments(case: Case, n_rev_skip: float, n_rev_per_seg: int,
             max_segments: int | None = None) -> list[Segment]:
    """立ち上げを捨てて、整数回転ずつの窓に切る。

    窓は時間ではなく角度で切る。テーブルは停止を挟むので、
    時間で切ると窓ごとに含まれる回転量が変わってしまう。
    """
    from .features import monotone_angle, plateau_collapse
    th = monotone_angle(case.theta_log)
    idx = plateau_collapse(th)
    thk, tk = th[idx], case.t_log[idx]
    two_pi = 2.0 * np.pi
    theta_start = float(thk[0]) + n_rev_skip * two_pi
    span = float(thk[-1]) - theta_start
    n = int(np.floor(span / (n_rev_per_seg * two_pi)))
    if max_segments is not None:
        n = min(n, max_segments)
    out = []
    for j in range(n):
        a = theta_start + j * n_rev_per_seg * two_pi
        b = a + n_rev_per_seg * two_pi
        t0 = float(np.interp(a, thk, tk))
        t1 = float(np.interp(b, thk, tk))
        out.append(Segment(index=j, theta0=a, n_rev=n_rev_per_seg, t0=t0, t1=t1))
    return out


def slice_channel(ch, t0: float, t1: float):
    """チャネルを時間で切り出して (t, y) を返す。"""
    sel = (ch.t >= t0) & (ch.t < t1)
    return ch.t[sel], ch.y[sel]
