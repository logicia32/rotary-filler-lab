"""C 物理コア（core/twin.c）と Python 参照実装（py/ref.py）の突き合わせ。

このラボの検証の型は「同じ式を 2 つの実装で書いて、数字で突き合わせる」。
`filler --f64` が吐いたダンプと、**同じ params.json・同じ初期状態**で回した
`py/ref.py` の `simulate()` の結果を、1 レコードずつ比べる。

`py/` は読むだけ。合わない箇所があっても参照実装は直さない。差の出どころを
特定して報告するだけにする（答えに合わせて参照を動かしたら検証にならない）。

------------------------------------------------------------------------
rev.2 との違い: 丸ごと比べられるようになった
------------------------------------------------------------------------
前版は「割出し軸が作った加速度だけを共通入力にして、液まわりだけ比べる」形だった。
軸に制御ループがあり、その構成（電流ループを何で閉じるか、ゲインをどこから作るか）が
C と py/ref.py で違ったので、軸そのものは突き合わせられなかった。

rev.3 では位置・速度・電流のループが存在しない。テーブル角は
カム入力軸角 psi の関数で一意に決まる。だから **軸・慣性・トルク・曲げ・
スロッシング・充填・受け渡しまで全部、同じ土俵で比べられる。**

比べるのは FORMAT.md 2 節のレコード全 15 スカラ + ステーション毎 5 量、
それにイベント列（種別ごとの件数・時刻・振幅）。

------------------------------------------------------------------------
初期状態
------------------------------------------------------------------------
両側とも工程配置から作る（`ref.steady_holders()` と `TWIN_PRIME_GEOM` が同じ式）。
`k = ((i*pitch - infeed)/pitch) mod N`、滞留 `n_res = ((discharge - infeed)/pitch) mod N` で
`k >= n_res` は在荷なし、`k == 0` は空瓶、それ以外は満量。
2026-07-23 の一時期だけ必要だった互換モード（`--prime-ref`）は撤去した。
下の「初期状態の照合」で毎回突き合わせている。

------------------------------------------------------------------------
しきい値と、その根拠
------------------------------------------------------------------------
両実装とも倍精度・同じ半陰的オイラー・同じ刻み・同じ演算順序で書いてある。
差の出どころは、式を書き下す順番のわずかな違い（C は `w1sq` を保持せず
`w1*w1` を作り直す、`hypot` と `math.hypot` の実装差、など）だけになる。
1 演算あたりの丸めは倍精度のイプシロン 2.2e-16。

- 波形（角度・角速度・力・トルク・曲げ）: `1e-11`
  1 ステップの丸めが 1e-16 程度、それが振動子の中を伝わる。減衰があるので
  誤差は発散せず、ステップ数 N のランダムウォークとして sqrt(N)*eps 程度に
  収まる。N = 2.4e5 なら 1e-13。実測値は下の表に出るので、そこを見ること。
  判定は**最大値で正規化した差**（点ごとの相対誤差は零交差で発散するので使わない）。
- 液量・液深・こぼれ: `1e-12`（相対）
  同じ増分を同じ順序で足すだけなので、本来は完全一致する。
  clamp の書き方の差だけを見るために、丸め 2、3 個ぶんの余裕を置いた。
- 時刻・角度（t / psi / th_t / th_m）: `1e-13`
  積分ではなくステップ番号から作る量。ここが合わないのは丸めではなく式の違い。
- イベントの時刻: `1e-12` s、振幅: f32 の分解能 `1e-6`（相対）
  イベント列だけは f32 で書いてあるので、そこは f32 の丸め（6e-8）が下限。

**しきい値を割ったら、まず疑うのは実装。**`py/ref.py` は動かさない。

------------------------------------------------------------------------
使い方
------------------------------------------------------------------------
    make -C core compare

手で回すなら:

    cd core
    ./filler --cycles 2 --f64 --out .cmp.bin --quiet
    ../.venv/bin/python compare_ref.py .cmp.bin --cycles 2
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "py"))
sys.path.insert(0, HERE)

import params as params_mod   # noqa: E402  py/params.py
import ref                    # noqa: E402  py/ref.py
from dumpio import read_dump, read_events, EV_KINDS, SCALARS, STATION  # noqa: E402

# --- しきい値（根拠は先頭の説明） ---
TOL_WAVE = 1.0e-11     # 波形。最大値で正規化した差
TOL_VOL = 1.0e-12      # 液量・液深・こぼれ（相対）
TOL_EXACT = 1.0e-13    # 積分しない量（時刻・角度）
TOL_EV_T = 1.0e-12     # イベント時刻 [s]
TOL_EV_AMP = 1.0e-6    # イベントの振幅（f32 で書いてあるので下限は 6e-8）

# ref.Record の属性名。ダンプの列名（dumpio.SCALARS）と 1 対 1。
REC_ATTR = dict(zip(SCALARS, SCALARS))
# ステーション量の Record 側の属性名（present は u8 なので別扱い）
ST_ATTR = {"V": "volume", "h": "height", "phi_t": "phi_t",
           "phi_r": "phi_r", "spill": "spill"}

# 積分しない量（時刻・角度）は別のしきい値で見る
EXACT_COLS = ("t", "psi", "th_t", "omega", "alpha", "th_m")
VOL_COLS = ("V", "h", "spill")

EV_NAME_TO_KIND = {v: k for k, v in EV_KINDS.items()}


class Report:
    def __init__(self):
        self.rows = []
        self.notes = []
        self.failed = 0

    def check(self, name, err, tol, unit=""):
        ok = bool(err <= tol) and math.isfinite(err)
        if not ok:
            self.failed += 1
        self.rows.append((name, err, tol, unit, ok))

    def note(self, text):
        self.notes.append(text)

    def show(self, title):
        print(f"\n--- {title} ---")
        w = max((len(r[0]) for r in self.rows), default=10)
        for name, err, tol, unit, ok in self.rows:
            print(f"  {'OK ' if ok else 'NG '} {name:<{w}}  差 {err:9.3e} {unit:<8}"
                  f" (しきい値 {tol:.0e})")
        for n in self.notes:
            print(f"  * {n}")
        self.rows = []
        self.notes = []


def norm_err(c: np.ndarray, r: np.ndarray) -> float:
    """最大値で正規化した差。点ごとの相対誤差は零交差で発散するので使わない。"""
    scale = float(np.max(np.abs(r)))
    if scale <= 0.0:
        return float(np.max(np.abs(c - r)))
    return float(np.max(np.abs(c - r)) / scale)


def rel_err(c: np.ndarray, r: np.ndarray) -> float:
    """値が 0 を跨がない量（液量・液深）の相対誤差。"""
    live = np.abs(r) > 0.0
    if not np.any(live):
        return float(np.max(np.abs(c - r)))
    return float(np.max(np.abs(c[live] - r[live]) / np.abs(r[live])))


def ref_columns(result, n_stations: int):
    """ref.Result のレコードを、ダンプと同じ形の配列に並べ直す。"""
    recs = result.records
    scal = {nm: np.asarray([getattr(x, REC_ATTR[nm]) for x in recs], dtype=float)
            for nm in SCALARS}
    st = {nm: np.asarray([getattr(x, ST_ATTR[nm]) for x in recs], dtype=float)
          for nm in STATION}
    st["present"] = np.asarray([[1 if q else 0 for q in x.present] for x in recs],
                               dtype=np.int64)
    for nm in STATION:
        if st[nm].shape[1] != n_stations:
            raise SystemExit(f"参照側のステーション数 {st[nm].shape[1]} が {n_stations} と違う")
    return scal, st


def compare_records(d, scal_r, st_r, rep: Report) -> None:
    if d.n_records != len(scal_r["t"]):
        raise SystemExit(
            f"レコード数が合わない: C {d.n_records} / 参照 {len(scal_r['t'])}。"
            " --cycles と params.json の tact / dt / log_rate を揃えること")

    for nm in SCALARS:
        c = d.scal[nm]
        r = scal_r[nm]
        tol = TOL_EXACT if nm in EXACT_COLS else TOL_WAVE
        rep.check(f"スカラ {nm}", norm_err(c, r), tol, "[-]")
    for nm in STATION:
        c = d.st[nm]
        r = st_r[nm]
        if nm in VOL_COLS:
            rep.check(f"ステーション {nm}", rel_err(c, r), TOL_VOL, "[-]")
        else:
            rep.check(f"ステーション {nm}", norm_err(c, r), TOL_WAVE, "[-]")
    # 在荷フラグ（u8）。1 個でも違えば落とす
    rep.check("ステーション present（在荷）",
              float(np.max(np.abs(d.present - st_r["present"]))), 0.5, "[本]")

    # 振幅の実測値（しきい値の妥当性を数字で残す）
    rep.note(f"振幅: th_t 最大 {np.max(scal_r['th_t']):.4f} rad / "
             f"T_table 最大 {np.max(np.abs(scal_r['torque_table'])):.4f} Nm / "
             f"T_in 最大 {np.max(scal_r['torque_input']):.4f} Nm / "
             f"M_bend 最大 {np.max(scal_r['m_bend']):.4f} Nm")
    rep.note(f"     phi_t 最大 {np.max(np.abs(st_r['phi_t'])):.5f} rad / "
             f"T_slosh 最大 {np.max(np.abs(scal_r['torque_slosh'])):.5f} Nm / "
             f"J_load {np.min(scal_r['j_load']):.5f}〜{np.max(scal_r['j_load']):.5f} kg m^2")
    tilt = np.hypot(st_r["phi_t"], st_r["phi_r"])
    rep.note(f"最大傾き {np.max(tilt) * 1e3:.3f} mrad（dz/R = {math.tan(np.max(tilt)):.4f}、"
             f"適用範囲 {ref.linearity_flag(float(np.max(tilt)))}）")
    if float(np.max(st_r["spill"])) > 0.0:
        rep.note("こぼれが出ている。両実装とも柱で落としているので同じはずだが、"
                 " 量そのものは 1〜2 桁の過大評価（py/NOTES.md 6 節）")
    else:
        rep.note("こぼれは両側とも 0（液面が縁に届いていない）")


def compare_events(ev, events_r, rep: Report) -> None:
    """イベント列を種別ごとに突き合わせる。

    py/ref.py は種別ごとに data の中身が違うので、比べるのは
    「件数」「時刻」「その種別で意味のある値」の 3 つ。
    """
    kinds_c = sorted(set(int(k) for k in ev.kind))
    kinds_r = sorted(set(EV_NAME_TO_KIND[e.kind] for e in events_r
                         if e.kind in EV_NAME_TO_KIND))
    unknown = sorted(set(e.kind for e in events_r if e.kind not in EV_NAME_TO_KIND))
    if unknown:
        rep.note(f"参照側に番号を割り当てていない種別がある: {unknown}")

    rep.check("イベント総数の差", float(abs(len(ev) - len(events_r))), 0.5, "[件]")
    rep.check("種別の集合の差", float(len(set(kinds_c) ^ set(kinds_r))), 0.5, "[種]")

    for kind in sorted(set(kinds_c) | set(kinds_r)):
        name = EV_KINDS.get(kind, str(kind))
        sub_c = ev.of(kind)
        sub_r = [e for e in events_r if EV_NAME_TO_KIND.get(e.kind) == kind]
        rep.check(f"{name}: 件数の差", float(abs(len(sub_c) - len(sub_r))), 0.5, "[件]")
        if len(sub_c) != len(sub_r) or not len(sub_c):
            continue
        t_c = np.asarray(sub_c.t, dtype=float)
        t_r = np.asarray([e.t for e in sub_r], dtype=float)
        rep.check(f"{name}: 時刻の差", float(np.max(np.abs(t_c - t_r))), TOL_EV_T, "[s]")
        st_c = np.asarray(sub_c.station, dtype=float)
        st_r = np.asarray([e.station for e in sub_r], dtype=float)
        rep.check(f"{name}: ホルダ番号の差", float(np.max(np.abs(st_c - st_r))), 0.5, "[番]")

        # 種別ごとに意味のある値（FORMAT.md 5.3 の d0..d7 の割り当て）
        want = []
        if kind == EV_NAME_TO_KIND["bearing_impulse"]:
            want = [("accel_m_s2", 0), ("ring_freq_hz", 1), ("ring_damping", 2)]
            # モータ角も比べる（軸受はモータ軸に置いてある）
            m_c = np.asarray(sub_c.th_m, dtype=float)
            m_r = np.asarray([e.data["motor_angle_rad"] for e in sub_r], dtype=float)
            rep.check(f"{name}: モータ軸角の差", norm_err(m_c, m_r), TOL_EV_AMP, "[-]")
        elif kind == EV_NAME_TO_KIND["cam_impact"]:
            want = [("torque_Nm", 0), ("clearance_rad", 1)]
        elif kind == EV_NAME_TO_KIND["infeed"]:
            want = [("mass_kg", 0)]
        elif kind == EV_NAME_TO_KIND["discharge"]:
            want = [("volume_m3", 0), ("tilt_rad", 1), ("phi_t_rad", 2),
                    ("phi_r_rad", 3), ("dphi_t_rad_s", 4), ("dphi_r_rad_s", 5),
                    ("mass_kg", 6)]
        elif kind == EV_NAME_TO_KIND["valve_drip"]:
            want = [("volume_m3", 0)]
        for key, idx in want:
            a = np.asarray(sub_c.d[:, idx], dtype=float)
            b = np.asarray([float(e.data[key]) for e in sub_r], dtype=float)
            rep.check(f"{name}: {key}", norm_err(a, b), TOL_EV_AMP, "[-]")


def load_params(enable_faults: bool):
    """params.json を読む。`--faults` なら faults を全部有効にして読ませる。

    `py/` は読むだけなので、有効化は**読み込んだ辞書の上**で行う。
    C 側は `--fault-*` を CLI で立てるので、同じ状態を参照側にも作らないと
    イベント列が比べられない（params.json の既定はすべて無効）。
    """
    import json

    path = params_mod.DEFAULT_PARAMS_PATH
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if enable_faults:
        for v in data.get("faults", {}).values():
            if isinstance(v, dict):
                v["enabled"] = True
    return params_mod.Params(data, path=path)


def geometric_holders(p):
    """工程配置から作った定常状態の (液量, 在荷)。core/twin.c の TWIN_PRIME_GEOM と同じ規則。

    供給からの割出し回数 k、滞留 n_res に対し
      k == 0        : 空のボトル
      0 < k < n_res : 満量
      k >= n_res    : ボトル無し
    """
    n = p.n_stations
    pitch = p.station_pitch
    n_res = int(round((p.discharge_angle - p.infeed_angle) / pitch)) % n
    vols, pres = [], []
    for i in range(n):
        k = int(round((i * pitch - p.infeed_angle) / pitch)) % n
        if k >= n_res:
            pres.append(False); vols.append(0.0)
        else:
            pres.append(True); vols.append(0.0 if k == 0 else p.target_volume)
    return vols, pres, n_res


def check_prime(p, d) -> None:
    """py/ref.py の steady_holders と、工程配置から作った並びを見比べる。

    判定はしない。**食い違っていたら、それは物理の差ではなく
    params.json の反映漏れ**なので、そう分かるように両方の数字を出す。
    """
    vols_g, pres_g, n_res = geometric_holders(p)
    holders = ref.steady_holders(p)
    vols_r = [h.volume for h in holders]
    pres_r = [h.has_bottle for h in holders]

    def fmt(vols, pres):
        return " ".join("--- " if not q else f"{v * 1e6:4.0f}" for v, q in zip(vols, pres))

    j_g = ref.rigid_load_inertia(vols_g, p, pres_g)
    j_r = ref.rigid_load_inertia(vols_r, p, pres_r)
    print(f"  工程配置   : 供給 {math.degrees(p.infeed_angle):.0f} / "
          f"充填 {math.degrees(p.fill_angle):.0f} / 排出 {math.degrees(p.discharge_angle):.0f} deg  "
          f"滞留 {n_res} 割出し = {n_res * p.tact:.1f} s")
    print(f"  初期状態（幾何から, mL）    : {fmt(vols_g, pres_g)}  "
          f"J_load(m0) = {j_g:.6f}")
    print(f"  初期状態（ref.steady_holders）: {fmt(vols_r, pres_r)}  "
          f"J_load(m0) = {j_r:.6f}")
    same = (vols_g == vols_r) and (pres_g == pres_r)
    if same:
        print("  * 一致（両側とも工程配置から作っている）")
    else:
        print("  * **食い違っている。** どちらかが params.json の工程角に追いついていない。"
              " これは物理の差ではないので、先にここを合わせること")


def main() -> int:
    args = sys.argv[1:]
    path = os.path.join(HERE, ".cmp.bin")
    n_cycles = 2
    enable_faults = False
    i = 0
    while i < len(args):
        if args[i] == "--cycles" and i + 1 < len(args):
            n_cycles = int(args[i + 1]); i += 2
        elif args[i] == "--faults":
            enable_faults = True; i += 1
        else:
            path = args[i]; i += 1

    d = read_dump(path)
    hd = d.header
    p = load_params(enable_faults)

    print("C 物理コア（core/twin.c）と Python 参照実装（py/ref.py）の突き合わせ")
    print(f"  params.json: R={p.R * 1e3:.2f} mm  Rp={p.Rp * 1e3:.1f} mm  "
          f"{p.n_stations} ステーション  zeta={p.zeta}  dt={p.dt:g} s")
    print(f"  ダンプ     : {os.path.basename(path)}  {d.summary()}")
    for w in d.warnings:
        print(f"  注意: {w}")

    if hd.elem_bytes != 8:
        print("  * ダンプが f32。f32 の丸め（相対 6e-8）が差の下限になるので、"
              " しきい値を割る。--f64 で吐き直すこと")
    # ダンプが params.json と同じ条件で作られたことを確かめる（違えば比較が無意味）
    for name, got, want in (("dt", hd.dt_s, p.dt),
                            ("log_dt", hd.log_dt_s, 1.0 / p.log_rate),
                            ("tact", hd.tact_s, p.tact),
                            ("index_time", hd.index_time_s, p.index_time)):
        if abs(got - want) > 1e-9 * max(1.0, abs(want)):
            raise SystemExit(f"ダンプの {name} = {got} が params.json の {want} と違う。"
                             " 同じ条件で回さないと突き合わせにならない")
    if bool(hd.fault_flags) != enable_faults:
        raise SystemExit(
            f"ダンプの故障フラグ {hd.fault_flags:#06b} と --faults の指定が食い違う。"
            " C 側の --fault-* と参照側の有効化を揃えないと比べられない")
    if enable_faults:
        print(f"  故障      : フラグ {hd.fault_flags:#06b}（欠品 st{hd.missing_station} / "
              f"軸受 {hd.bearing_defect_freq_hz:.1f} Hz）")

    # 初期状態の照合（params.json の工程角が動いたときにここが効く）
    print()
    check_prime(p, d)

    print(f"\npy/ref.py の simulate() を {n_cycles} サイクル回す（少し時間がかかる）…",
          flush=True)
    result = ref.simulate(p, n_cycles=n_cycles)
    scal_r, st_r = ref_columns(result, p.n_stations)

    rep = Report()
    compare_records(d, scal_r, st_r, rep)
    rep.show(f"連続量（{d.n_records} レコード x {len(SCALARS)} + "
             f"{len(STATION)}x{p.n_stations}）")

    ev_path = str(path) + ".events"
    if os.path.exists(ev_path):
        ev = read_events(ev_path)
        compare_events(ev, result.events, rep)
        rep.show(f"イベント列（C {len(ev)} 件 / 参照 {len(result.events)} 件）")
    else:
        print(f"\n（イベント列 {os.path.basename(ev_path)} が無いので、そちらは比べていない）")

    for w in result.warnings:
        print(f"参照側の警告: {w}")

    print()
    if rep.failed:
        print(f"NG: {rep.failed} 件がしきい値を超えた。"
              " 差の出どころを特定すること（py/ は直さない）")
        return 1
    print("OK: すべてしきい値の内側")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
