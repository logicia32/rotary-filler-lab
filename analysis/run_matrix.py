"""総当たり。故障 × センサ × 特徴量を全部回して表にする。

    .venv/bin/python -m analysis.run_matrix --workdir <作業ディレクトリ> --filler <退避したコアのパス>

やること

1. 物理コアを条件ごとに 1 回ずつ回す（既にダンプがあれば作り直さない）
2. センサ層で信号を作る（基準用と試験用で乱数の種を変える）
3. 窓（整数回転）ごとに特徴量を出す
4. 正常データから基準を作り、しきい値で判定する
5. 表を標準出力へ、数値を workdir/results.json へ、図を figs/analysis_*.png へ

見えなかったものは見えなかったと書く。検出できた欄だけ並べた表は嘘になるので、
測れなかった欄には理由（帯域外・雑音床の下・交流結合で落ちた）を残す。
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from . import config, dataset, detect, features
from .dataset import CHANNEL_LABELS, CONDITIONS

CHANNELS = tuple(CHANNEL_LABELS)
ACCEL_LF = ("accel_lf_tangential", "accel_lf_radial")

# 特徴量の名前と、表に出す見出し
FEATURE_LABELS = {
    "rms": "実効値",
    "crest": "波高率",
    "order1": "次数 1 の振幅",
    "order_sb": "次数 1±8 の振幅",
    "slosh_amp": "スロッシングの線の振幅",
    "slosh_rms": "スロッシング帯の実効値",
    "slosh_peak_hz": "スロッシング帯のピーク周波数",
    "env_bpfo": "包絡線の BPFO 線の振幅",
    "impacts_per_s": "衝撃の検出回数 / 秒",
}
# 主表に出す特徴量（判定に使うもの）
JUDGED = ("rms", "crest", "order1", "order_sb", "slosh_amp", "slosh_rms",
          "env_bpfo", "impacts_per_s")


def _nan_dict(keys):
    return {k: float("nan") for k in keys}


def base_window_features(case, ch, seg, ana, params, spr):
    """1 窓ぶんの、連続量から出る特徴量。"""
    t, y = dataset.slice_channel(ch, seg.t0, seg.t1)
    out = _nan_dict(("rms", "crest", "order1", "order_sb",
                     "slosh_amp", "slosh_rms", "slosh_peak_hz"))
    if len(y) < 16:
        return out
    out["rms"] = float(np.sqrt(np.mean(y ** 2)))
    out["crest"] = float(np.max(np.abs(y)) / out["rms"]) if out["rms"] > 0 else float("nan")

    # 角度リサンプルの前に、角度領域のナイキストに合わせて低域通過を掛ける
    ylp, _ = features.order_antialias(y, ch.fs_hz, spr, case.peak_rev_per_s)
    th = features.angle_at(t, case.t_log, case.theta_log)
    try:
        sig = features.angle_resample(th, ylp, spr, seg.n_rev, theta0=seg.theta0)
    except ValueError:
        return out
    orders, amp = features.order_spectrum(sig, window="rect")
    k1 = seg.n_rev                       # 次数 1 のビン
    out["order1"] = float(amp[k1])
    sb = [amp[seg.n_rev * k] for k in (7, 9) if seg.n_rev * k < len(amp)]
    out["order_sb"] = float(max(sb)) if sb else float("nan")

    f1 = case.slosh_freq_hz
    lo, hi = (r * f1 for r in ana["spectrum"]["slosh_band_ratio"])
    peak_f, peak_a = features.line_amplitude(y, ch.fs_hz, lo, hi)
    out["slosh_amp"] = peak_a
    out["slosh_peak_hz"] = peak_f
    rms, _, _ = features.band_rms(y, ch.fs_hz, lo, hi,
                                  ana["spectrum"]["psd_resolution_hz"])
    out["slosh_rms"] = rms
    return out


def envelope_window_features(case, ch, seg, ana, params):
    """1 窓ぶんの、包絡線から出る特徴量。帯域外なら理由を残して nan。"""
    out = _nan_dict(("env_bpfo", "impacts_per_s"))
    note = ""
    fault = params["faults"]["bearing_outer_race"]
    ring = float(fault["ring_freq_hz"])
    # BPFO は機械の定数（欠陥次数 × モータ回転数）で、故障の有無に依らず定義される。
    # ダンプヘッダの bearing_defect_freq_hz は故障無効だと 0 になり、正常の基準
    # （健全なときの BPFO 帯の値）が作れない。params から出して正常・故障で同じ帯域を見る。
    motor_rev_s = float(params["drive"]["motor_rpm_at_operating_point"]) / 60.0
    bpfo = float(fault["defect_freq_ratio"]) * motor_rev_s      # BPFO [Hz]
    lo, hi = (r * ring for r in ana["envelope"]["band_ratio"])
    t, y = dataset.slice_channel(ch, seg.t0, seg.t1)
    if len(y) < 64:
        return out, "窓に入るサンプルが足りない"
    try:
        env = features.envelope(
            y, ch.fs_hz, (lo, hi), ring * ana["envelope"]["lowpass_ratio"],
            order=ana["envelope"]["filter_order"], method="abs",
            decimate_to_hz=ana["envelope"]["decimate_target_hz"], t0=float(t[0]))
    except features.BandOutOfRange as e:
        return out, str(e)

    # 軸受はモータ軸（定速 1500rpm）に付いているので、欠陥通過 BPFO は
    # **時間的に一定**（テーブル回転の次数ではない。テーブルは停留で止まるので
    # 角度リサンプルは成り立たない）。包絡線を**時間領域のまま**スペクトルにして
    # BPFO のピークを読む。旧版はテーブル角へ移して「次数 3.58」を見ていたが、
    # それはテーブル軸受の前提で、rev.3 の機構には合わない。
    b_lo, b_hi = (r * bpfo for r in ana["spectrum"]["defect_order_band_ratio"])
    if b_hi >= 0.5 * env.fs_hz:
        return out, (f"BPFO 帯 {b_lo:.0f}〜{b_hi:.0f} Hz が包絡線のナイキスト"
                     f" {0.5 * env.fs_hz:.0f} Hz を超える")
    _, peak_amp = features.line_amplitude(env.y, env.fs_hz, b_lo, b_hi)
    out["env_bpfo"] = peak_amp

    # 衝撃の最短間隔も時間基準（1/BPFO の一部）。旧版は rev_s/次数で 6.7 s と
    # なり、実際の 1/89.5 = 11 ms に対し 600 倍で、ほとんどの衝撃を潰していた。
    min_sep = ana["envelope"]["impact_min_separation_ratio"] / bpfo
    times, thr = features.impact_times(
        env, ana["envelope"]["impact_threshold_sigma"], min_separation_s=min_sep)
    window_s = float(seg.t1 - seg.t0)
    out["impacts_per_s"] = len(times) / window_s if window_s > 0.0 else float("nan")
    return out, note


def collect(case, ana, params, max_base=None):
    """1 条件ぶんの特徴量を全部集める。"""
    spr = ana["angle_resample"]["samples_per_rev"]
    skip = ana["run"]["startup_revolutions"]
    base_segs = dataset.segments(case, skip, ana["segment"]["base_revolutions"], max_base)
    env_segs = dataset.segments(case, skip, ana["segment"]["envelope_revolutions"])

    res = {"base": {}, "env": {}, "notes": {}, "meta": {
        "n_base_segments": len(base_segs),
        "n_env_segments": len(env_segs),
        "base_revolutions": ana["segment"]["base_revolutions"],
        "envelope_revolutions": ana["segment"]["envelope_revolutions"],
        "window_s": base_segs[0].t1 - base_segs[0].t0 if base_segs else float("nan"),
        "t_start_s": base_segs[0].t0 if base_segs else float("nan"),
        "t_end_s": base_segs[-1].t1 if base_segs else float("nan"),
    }}
    for name in CHANNELS:
        ch = case.channels.get(name)
        if ch is None:
            continue
        rows = [base_window_features(case, ch, s, ana, params, spr) for s in base_segs]
        res["base"][name] = {k: [r[k] for r in rows] for k in rows[0]} if rows else {}
        erows, enote = [], ""
        for s in env_segs:
            r, note = envelope_window_features(case, ch, s, ana, params)
            erows.append(r)
            enote = enote or note
        res["env"][name] = {k: [r[k] for r in erows] for k in erows[0]} if erows else {}
        if enote:
            res["notes"][name] = enote
        res.setdefault("unit", {})[name] = ch.unit
    return res


# ---------------------------------------------------------------------------
# 判定と表
# ---------------------------------------------------------------------------

def make_baselines(base_res: dict, unit: dict) -> dict:
    """正常データの特徴量から基準を作る。"""
    out = {}
    for kind in ("base", "env"):
        for chan, feats in base_res[kind].items():
            for feat, vals in feats.items():
                if feat not in JUDGED:
                    continue
                out[(chan, feat)] = detect.build_baseline(
                    vals, name=f"{chan}/{feat}", unit=unit.get(chan, ""))
    return out


def judge_case(res: dict, baselines: dict, k_sigma: float) -> dict:
    """1 条件ぶんの判定。窓ごとの値の中央値を代表値にする。"""
    out = {}
    for kind in ("base", "env"):
        for chan, feats in res[kind].items():
            for feat, vals in feats.items():
                if feat not in JUDGED:
                    continue
                bl = baselines.get((chan, feat))
                if bl is None:
                    continue
                finite = [v for v in vals if np.isfinite(v)]
                value = float(np.median(finite)) if finite else float("nan")
                note = res["notes"].get(chan, "") if not finite else ""
                out[(chan, feat)] = detect.judge(value, bl, k_sigma,
                                                 feature=feat, channel=chan, note=note)
    return out


def fmt(x, digits=3):
    if x is None or not np.isfinite(x):
        return "—"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def matrix_table(verdicts: dict, conditions: list, k_sigma: float) -> str:
    """故障 × センサ × 特徴量の表（markdown）。"""
    lines = []
    head = "| センサ | 特徴量 | 単位 | 正常の基準 (平均 ± σ) | " + \
           " | ".join(CONDITIONS[c]["label"] for c in conditions) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (4 + len(conditions)))
    for chan in CHANNELS:
        for feat in JUDGED:
            key = (chan, feat)
            any_row = any(key in verdicts[c] for c in conditions)
            if not any_row:
                continue
            ref = next(verdicts[c][key] for c in conditions if key in verdicts[c])
            cells = []
            for c in conditions:
                v = verdicts[c].get(key)
                if v is None or not np.isfinite(v.value):
                    cells.append(f"測れない（{v.note}）" if v is not None and v.note else "—")
                    continue
                mark = "検出" if v.detected else "見えない"
                cells.append(f"{fmt(v.value)} / {fmt(v.z, 3)} σ / {mark}")
            base = f"{fmt(ref.baseline_mean)} ± {fmt(ref.baseline_std)}"
            lines.append(f"| {CHANNEL_LABELS[chan]} | {FEATURE_LABELS[feat]} | "
                         f"{ref.unit} | {base} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary_table(verdicts: dict, conditions: list) -> str:
    """センサ × 故障の要約。いちばん強く出た特徴量だけ書く。"""
    lines = ["| センサ | " + " | ".join(CONDITIONS[c]["label"] for c in conditions) + " |",
             "|" + "---|" * (1 + len(conditions))]
    for chan in CHANNELS:
        cells = []
        for c in conditions:
            best, bz = None, -np.inf
            for (ch2, feat), v in verdicts[c].items():
                if ch2 != chan or not np.isfinite(v.z):
                    continue
                if v.detected and v.z > bz:
                    best, bz = (feat, v), v.z
            if best is None:
                cells.append("見えない")
            else:
                feat, v = best
                cells.append(f"{FEATURE_LABELS[feat]}（{fmt(v.z, 3)} σ）")
        lines.append(f"| {CHANNEL_LABELS[chan]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主処理
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="故障 × センサ × 特徴量の総当たり")
    ap.add_argument("--workdir", required=True, help="ダンプと中間結果を置く場所（リポジトリの外）")
    ap.add_argument("--filler", required=True, help="物理コアの実行ファイル（退避したコピー）")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--rerun-core", action="store_true")
    ap.add_argument("--skip-startup", action="store_true")
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ana = config.analysis_params()
    duration = args.duration or ana["run"]["duration_s"]
    params, added = config.load_machine_params()
    names = [n for n in args.conditions.split(",") if n]

    core_md5 = dataset.md5_of(args.filler)
    print(f"物理コア {args.filler} md5={core_md5}")
    print(f"params.json で既定値を補ったキー: {added or 'なし'}")

    out = {"core_md5": core_md5, "duration_s": duration, "analysis": ana,
           "conditions": {}, "structure": {}, "slosh_snr": {}, "events": {}}
    figdata = {}

    # 1. コアを回す
    paths = {n: dataset.run_core(n, workdir, args.filler, duration,
                                 reuse=not args.rerun_core) for n in names}

    # 2. 条件ごとに特徴量を出す
    for name in names:
        t0 = time.time()
        from sensors.read_dump import load_run
        dump, events = load_run(paths[name])
        roles = {"test": ana["seed"]["test"]}
        if name == "normal":
            roles["baseline"] = ana["seed"]["baseline"]
        for role, seed in roles.items():
            case = dataset.case_from_dump(name, paths[name], dump, events, params, seed)
            res = collect(case, ana, params)
            res["meta"]["seed"] = seed
            res["meta"]["role"] = role
            res["meta"]["max_dz_over_R"] = case.max_dz_over_R
            res["meta"]["range_exceeded"] = case.range_exceeded
            res["meta"]["n_bearing_events"] = int(len(case.bearing_times))
            res["meta"]["n_cam_events"] = int(len(case.cam_times))
            out["conditions"].setdefault(name, {})[role] = res
            if role == "test":
                _stash_figdata(figdata, name, case, ana, params)
                out["events"][name] = _event_check(case, ana, params)
        del dump, events
        gc.collect()
        print(f"  [feat] {name}: {time.time() - t0:.1f} s")

    # 3. 基準と判定
    k = ana["detect"]["sigma_fallback"]
    base_res = out["conditions"]["normal"]["baseline"]
    baselines = make_baselines(base_res, base_res.get("unit", {}))
    n_base = max((b.n for b in baselines.values()), default=0)
    per_day = 86400.0 / (base_res["meta"]["window_s"])
    k_calc, p_target = detect.sigma_for_false_alarm(n_base, per_day,
                                                    ana["detect"]["target_false_alarm_per_month"])
    print(f"\n基準の窓 {n_base} 個 / 1 窓 {base_res['meta']['window_s']:.1f} s "
          f"= 1 日 {per_day:.0f} 回の判定")
    print(f"月 1 回の誤警報を許すなら片側確率 {p_target:.3g}。自由度 {n_base-1} の t 分布で {k_calc:.2f} σ。"
          f"採用 {k:.1f} σ（1 回あたりの誤警報 {detect.false_alarm_rate(k, n_base):.3g}）")

    verdicts = {c: judge_case(out["conditions"][c]["test"], baselines, k) for c in names}
    out["threshold"] = {"k_sigma": k, "k_from_target": k_calc, "p_target": p_target,
                        "n_baseline": n_base,
                        "false_alarm_per_decision": detect.false_alarm_rate(k, n_base),
                        "decisions_per_day": per_day}
    out["baselines"] = {f"{c}/{f}": {"mean": b.mean, "std": b.std, "n": b.n,
                                     "unit": b.unit, "skew": b.skew(),
                                     "normality": detect.normality_note(b)}
                        for (c, f), b in baselines.items()}
    out["verdicts"] = {c: {f"{ch}/{ft}": vars(v) for (ch, ft), v in vs.items()}
                       for c, vs in verdicts.items()}

    # 全条件を列に出す。基準の normal も列に残すと、基準が自分自身に誤警報しない
    # （z がほぼ 0・「見えない」）ことのサニティになる。
    faults = list(names)
    print("\n## A. 定常窓: センサ × 故障の要約\n")
    print(summary_table(verdicts, faults))
    print("\n## A. 定常窓: 故障 × センサ × 特徴量\n")
    print(matrix_table(verdicts, faults, k))
    print("\n## A. 衝撃イベントの突き合わせ\n")
    print(event_table(out["events"]))

    out["long_average"] = long_average(workdir, args.filler, params, ana,
                                       figdata=figdata)
    print("\n## A. 次数 1 は雑音のどれだけ下にいるか（ひずみゲージ・定常窓）\n")
    print(long_average_table(out["long_average"]))

    # 4. 立ち上げの窓（ホルダの中身が揃っていない 1 回転）
    if not args.skip_startup:
        print("\n(立ち上げ窓の解析)")
        sm = startup_matrix(workdir, args.filler, params, ana, figdata)
        out["startup"] = sm
        print(f"\n## B. 立ち上げ窓（{sm['conditions']['normal']['window_s']:.1f} s = 1 回転・"
              f"雑音の種 {len(sm['base_seeds'])} 通りで基準）\n")
        print(startup_tables(sm, ana))
        print("\n## B. 構造モデルの感度（accel_lf 接線・立ち上げ窓）\n")
        print(structure_table(sm["structure"], sm["noise_floor"]))
        print("\n## B. スロッシングの線は、どのセンサで雑音床から出るか\n")
        print(slosh_visibility_table(sm["slosh_visibility"]))
        print("\n## B. タクトを変えたときのスロッシングの線\n")
        print(tact_table(sm["tact"]))

    (workdir / "results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    np.savez_compressed(workdir / "figdata.npz", **figdata)
    print(f"\n数値 -> {workdir / 'results.json'} / 図の材料 -> {workdir / 'figdata.npz'}")

    if not args.no_figures:
        from . import figures
        figures.make_all(workdir, out)
    return out


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _event_check(case, ana, params) -> dict:
    """包絡線で拾った衝撃の時刻を、真のイベント列と突き合わせる。"""
    fault = params["faults"]["bearing_outer_race"]
    ring = float(fault["ring_freq_hz"])
    motor_rev_s = float(params["drive"]["motor_rpm_at_operating_point"]) / 60.0
    bpfo = float(fault["defect_freq_ratio"]) * motor_rev_s      # BPFO [Hz]
    ch = case.channels.get("accel_hf_radial")
    if ch is None or not (bpfo > 0.0):
        return {}
    skip = ana["run"]["startup_revolutions"]
    segs = dataset.segments(case, skip, ana["segment"]["envelope_revolutions"])
    if not segs:
        return {}
    seg = segs[0]
    t, y = dataset.slice_channel(ch, seg.t0, seg.t1)
    lo, hi = (r * ring for r in ana["envelope"]["band_ratio"])
    env = features.envelope(y, ch.fs_hz, (lo, hi), ring * ana["envelope"]["lowpass_ratio"],
                            order=ana["envelope"]["filter_order"], method="abs",
                            decimate_to_hz=ana["envelope"]["decimate_target_hz"],
                            t0=float(t[0]))
    # 衝撃の最短間隔は時間基準（1/BPFO の一部）。旧版のテーブル回転次数基準ではない。
    min_sep = ana["envelope"]["impact_min_separation_ratio"] / bpfo
    found, thr = features.impact_times(env, ana["envelope"]["impact_threshold_sigma"], min_sep)
    truth = case.bearing_times[(case.bearing_times >= seg.t0) & (case.bearing_times < seg.t1)]
    hits, n_truth, extra, rms = features.match_events(found, truth, tolerance_s=0.01)
    return {"window_s": seg.t1 - seg.t0, "threshold": thr, "n_found": int(len(found)),
            "n_truth": int(n_truth), "hits": int(hits), "extra": int(extra),
            "timing_rms_ms": rms * 1e3 if np.isfinite(rms) else float("nan")}


def long_average(workdir, filler, params, ana, conds=("normal", "missing"),
                 figdata: dict | None = None) -> dict:
    """次数 1 が雑音のどれだけ下にいるのか、そして何回転まとめれば出るのかを出す。

    表の「見えない」は、そのままでは「どのくらい足りないのか」が分からない。
    そこで
      * センサ連鎖に入る前の真値の次数 1 の振幅（雑音ゼロの答え）
      * 1 回転の窓での雑音床（センサ出力の次数 1 のビン）
    を並べて、必要な同期平均の回転数を見積もる。

    角度同期平均は、窓を N 回転に伸ばせば次数 1 のビンの雑音が sqrt(N) 倍下がる
    （信号は回転に同期しているので下がらない）。6 σ を確保するのに要る N は

        N = (6 * 雑音床 / 信号)^2
    """
    from sensors.read_dump import load_run
    spr = ana["angle_resample"]["samples_per_rev"]
    skip = ana["run"]["startup_revolutions"]
    out = {}
    for name in conds:
        path = workdir / f"{name}.bin"
        if not path.exists():
            continue
        dump, events = load_run(path)
        case = dataset.case_from_dump(name, path, dump, events, params, ana["seed"]["test"],
                                      channels=("strain",))
        ch = case.channels["strain"]
        segs = dataset.segments(case, skip, ana["segment"]["base_revolutions"])
        if not segs:
            out[name] = {"error": "区間が作れない（--duration が立ち上げ捨て＋1 回転より短い）"}
            continue
        n_rev = len(segs)
        seg1 = segs[0]
        fs_truth = 1.0 / float(np.median(np.diff(ch.truth_t)))

        def order1_of(t, y, fs, s, nrev):
            ylp, _ = features.order_antialias(y, fs, spr, case.peak_rev_per_s)
            th = features.angle_at(t, case.t_log, case.theta_log)
            sig = features.angle_resample(th, ylp, spr, nrev, theta0=s.theta0)
            return features.order_amplitude(sig, 1.0)

        sel1 = (ch.truth_t >= seg1.t0) & (ch.truth_t < seg1.t1)
        truth1 = order1_of(ch.truth_t[sel1], ch.truth[sel1], fs_truth, seg1, 1)
        t1, y1 = dataset.slice_channel(ch, seg1.t0, seg1.t1)
        outp1 = order1_of(t1, y1, ch.fs_hz, seg1, 1)

        # 全区間を 1 つの窓（n_rev 回転）にまとめる
        long_seg = dataset.Segment(0, seg1.theta0, n_rev, seg1.t0, segs[-1].t1)
        selL = (ch.truth_t >= long_seg.t0) & (ch.truth_t < long_seg.t1)
        truthL = order1_of(ch.truth_t[selL], ch.truth[selL], fs_truth, long_seg, n_rev)
        tL, yL = dataset.slice_channel(ch, long_seg.t0, long_seg.t1)
        outL = order1_of(tL, yL, ch.fs_hz, long_seg, n_rev)

        if figdata is not None:
            # 図の材料: 真値とセンサ出力の次数比スペクトル（同じ 1 回転の窓）
            for tag, (tt, yy, fs) in (
                    ("truth", (ch.truth_t[sel1], ch.truth[sel1], fs_truth)),
                    ("out", (t1, y1, ch.fs_hz))):
                ylp, _ = features.order_antialias(yy, fs, spr, case.peak_rev_per_s)
                th = features.angle_at(tt, case.t_log, case.theta_log)
                sg = features.angle_resample(th, ylp, spr, 1, theta0=seg1.theta0)
                o, a = features.order_spectrum(sg, window="rect")
                figdata[f"ord1_{name}_{tag}_x"] = o
                figdata[f"ord1_{name}_{tag}_y"] = a

        need = (6.0 * outp1 / truth1) ** 2 if truth1 > 0 else float("inf")
        out[name] = {"truth_order1_1rev": truth1, "out_order1_1rev": outp1,
                     "snr_db_1rev": 20 * np.log10(truth1 / outp1) if outp1 > 0 else float("nan"),
                     "n_rev_long": n_rev, "truth_order1_long": truthL,
                     "out_order1_long": outL,
                     "need_revolutions_for_6sigma": need}
        del dump, events, case
        gc.collect()
    return out


def long_average_table(la: dict) -> str:
    lines = ["| 条件 | 真値の次数 1 [ustrain] | 1 回転窓の出力 [ustrain] | S/N [dB] |"
             f" まとめた窓の出力 [ustrain] | 6 σ に要る回転数 |",
             "|---|---|---|---|---|---|"]
    for name, v in la.items():
        lines.append(
            f"| {CONDITIONS[name]['label']} | {fmt(v['truth_order1_1rev'])} | "
            f"{fmt(v['out_order1_1rev'])} | {v['snr_db_1rev']:+.1f} | "
            f"{fmt(v['out_order1_long'])}（{v['n_rev_long']} 回転） | "
            f"{v['need_revolutions_for_6sigma']:.0f} |")
    return "\n".join(lines)


def tact_table(tact: dict) -> str:
    lines = ["| タクト [s] | タクト周波数 [Hz] | スロッシング f1 [Hz] | f1 / タクト周波数 |"
             " 帯域のピーク [Hz] | ピークの振幅 [ustrain] |",
             "|---|---|---|---|---|---|"]
    for key, v in tact.items():
        lines.append(f"| {v['tact_s']:g} | {v['tact_hz']:.4f} | {v['slosh_hz']:.4f} | "
                     f"{v['slosh_over_tact']:.4f} | {v['peak_hz']:.4f} | "
                     f"{fmt(v['peak_amp'])} |")
    return "\n".join(lines)


def event_table(events: dict) -> str:
    lines = ["| 条件 | 窓 [s] | 真のイベント | 検出 | 一致 | 余分 | 時刻の差 [ms] |",
             "|---|---|---|---|---|---|---|"]
    for name, e in events.items():
        if not e:
            continue
        lines.append(f"| {CONDITIONS[name]['label']} | {e['window_s']:.1f} | {e['n_truth']} | "
                     f"{e['n_found']} | {e['hits']} | {e['extra']} | {fmt(e['timing_rms_ms'])} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 立ち上げの窓（ホルダの中身が揃っていない 1 回転）
# ---------------------------------------------------------------------------

STARTUP_CHANNELS = ("accel_lf_tangential", "accel_lf_radial", "strain", "current")
STARTUP_CONDITIONS = ("normal", "missing")


def startup_matrix(workdir, filler, params, ana, figdata) -> dict:
    """立ち上げの 1 回転を、雑音の種を振って解析する。

    NOTE(004 移行): この立ち上げ窓解析は旧サーボ機（8 本対称で、揃うと水平合力が
    打ち消えて 0 になる機械）向けに組んだもの。カム式（単ノズル・工程配置が非対称）
    では定常運転でも合力は 0 にならず、「立ち上げだけ中身が揃わない」という前提が
    そのままでは成り立たない。加えて 004 の既定は工程配置から作った定常状態で始まる
    （空テーブルからの充填過渡は `--no-prime` を付けたときだけ）。ここは呼び出しと
    条件名を 004 に合わせて動くようにしただけで、物理の解釈は要見直し（報告参照）。
    """
    from sensors.read_dump import load_run
    st = ana["startup"]
    base_seeds = [ana["seed"]["baseline"] + i for i in range(st["n_baseline_seeds"])]
    test_seeds = [ana["seed"]["test"] + 1000 * (i + 1) for i in range(st["n_test_seeds"])]
    out = {"conditions": {}, "structure": {}, "tact": {},
           "base_seeds": base_seeds, "test_seeds": test_seeds}
    noise_floor = {}

    for name in STARTUP_CONDITIONS:
        path = dataset.run_core(name, workdir, filler, st["duration_s"], prefix="startup_")
        dump, events = load_run(path)
        case = dataset.case_from_dump(name, path, dump, events, params, base_seeds[0],
                                      channels=STARTUP_CHANNELS)
        segs = dataset.segments(case, 0.0, st["revolutions"])
        if not segs:
            raise RuntimeError(f"立ち上げの窓が取れない（{name}）")
        seg = segs[0]
        spr = ana["angle_resample"]["samples_per_rev"]
        rec = {"baseline": {}, "test": {}, "window_s": seg.t1 - seg.t0}
        for role, seeds in (("baseline", base_seeds), ("test", test_seeds)):
            rows = {chan: [] for chan in STARTUP_CHANNELS}
            for sd in seeds:
                case.channels = dataset.reseed(case, dump, params, sd,
                                               channels=STARTUP_CHANNELS)
                for chan in STARTUP_CHANNELS:
                    rows[chan].append(base_window_features(
                        case, case.channels[chan], seg, ana, params, spr))
            rec[role] = {chan: {k: [r[k] for r in rs] for k in rs[0]}
                         for chan, rs in rows.items()}
        rec["unit"] = {c: case.channels[c].unit for c in STARTUP_CHANNELS}
        out["conditions"][name] = rec
        if name == "normal":
            # 雑音床は「入力を 0 にしたときの出力」で測る。真値を丸ごと 0 にした入力を
            # 同じ連鎖（量子化・飽和・雑音）に通して出てきたものが雑音床。条件には依らない
            # （どの条件の dump でも、真値を 0 にすれば同じ床が出る）ので normal で 1 回測る。
            noise_floor.update(_zero_input_floor(case, dump, params, ana, seg,
                                                 base_seeds[:8], spr))
        _stash_startup_figdata(figdata, name, case, seg, ana, params)
        if name in ("normal", "missing"):
            out["structure"][name] = structure_sweep(name, path, dump, events, params, ana,
                                                     noise_floor, figdata, seg)
        del dump, events, case
        gc.collect()

    # タクトを変えて、揺れの線だけ動かないことを見る
    out["tact"] = tact_sweep(workdir, filler, params, ana, figdata)
    out["noise_floor"] = noise_floor
    out["slosh_visibility"] = slosh_visibility(workdir, params, ana, noise_floor)
    return out


def slosh_visibility(workdir, params, ana, noise_floor) -> dict:
    """スロッシングの線が、各センサの雑音床からどれだけ出ているか。

    揺れは故障ではないので「異常として検出できるか」を聞いても意味がない
    （正常の基準にも同じ揺れが入っている）。聞くべきは
    そのセンサでそもそも観測できるのかで、それは
    「連鎖に入る前の真値の線」と「完全に揃った機械（力が打ち消えて 0）の出力」
    の比で決まる。
    """
    from sensors.read_dump import load_run
    path = Path(workdir) / "startup_missing.bin"
    dump, events = load_run(path)
    case = dataset.case_from_dump("missing", path, dump, events, params, ana["seed"]["test"],
                                  channels=STARTUP_CHANNELS)
    seg = dataset.segments(case, 0.0, ana["startup"]["revolutions"])[0]
    f1 = case.slosh_freq_hz
    lo, hi = (r * f1 for r in ana["spectrum"]["slosh_band_ratio"])
    out = {}
    for chan in STARTUP_CHANNELS:
        ch = case.channels[chan]
        sel = (ch.truth_t >= seg.t0) & (ch.truth_t < seg.t1)
        fs_truth = 1.0 / float(np.median(np.diff(ch.truth_t)))
        pf, pa = features.line_amplitude(ch.truth[sel], fs_truth, lo, hi)
        floor = noise_floor.get(chan, float("nan"))
        out[chan] = {"truth_line": pa, "truth_line_hz": pf, "floor": floor,
                     "snr_db": 20 * np.log10(pa / floor)
                     if floor > 0 and pa > 0 else float("nan"),
                     "unit": ch.unit}
    del dump, events, case
    gc.collect()
    return out


def slosh_visibility_table(sv: dict) -> str:
    lines = ["| センサ | 真値の 3.75 Hz の線 | 雑音床（同じ窓） | S/N [dB] | 観測できるか |",
             "|---|---|---|---|---|"]
    for chan, v in sv.items():
        vis = "見える" if v["snr_db"] > 0 else "埋もれる"
        lines.append(f"| {CHANNEL_LABELS[chan]} | {fmt(v['truth_line'])} {v['unit']} | "
                     f"{fmt(v['floor'])} {v['unit']} | {v['snr_db']:+.1f} | {vis} |")
    return "\n".join(lines)


def _zero_input_floor(case, dump, params, ana, seg, seeds, spr) -> dict:
    """真値を 0 にした入力を同じ連鎖に通して、雑音床を測る。

    量子化と飽和は非線形なので、単に「雑音の実効値」から計算するのではなく、
    実際に連鎖へ 0 を通して出てきたものを測る。
    """
    import copy as _copy

    from sensors import virtual
    zero_truth = _copy.copy(case.truth)
    for attr in ("force_t", "force_r", "accel_lf_tangential", "accel_lf_radial",
                 "accel_hf_radial", "strain_ustrain", "current_A", "torque_slosh_Nm"):
        zero_truth.__dict__[attr] = np.zeros_like(getattr(case.truth, attr))
    zero_truth.notes = []
    f1 = case.slosh_freq_hz
    lo, hi = (r * f1 for r in ana["spectrum"]["slosh_band_ratio"])
    acc = {chan: [] for chan in STARTUP_CHANNELS}
    for sd in seeds:
        chans, _ = virtual.synthesize(dump, params, seed=sd, truth=zero_truth,
                                      channels=STARTUP_CHANNELS)
        for chan in STARTUP_CHANNELS:
            ch = chans[chan]
            _, y = dataset.slice_channel(ch, seg.t0, seg.t1)
            _, a = features.line_amplitude(y, ch.fs_hz, lo, hi)
            acc[chan].append(a)
    return {chan: float(np.median(v)) for chan, v in acc.items()}


def tact_sweep(workdir, filler, params, ana, figdata) -> dict:
    """タクトを変えて、スロッシングの線と割出しの高調波を見分ける。

    定常のスペクトルはタクト周波数（1/tact）の高調波の線スペクトルになる。
    スロッシング 3.751 Hz は液深と容器径だけで決まりタクトには依らないので、
    タクトを変えると高調波は動き、揺れの線は動かない。これが「回転に同期しない」
    ことの示し方。基準タクト 3.0 s ではタクト周波数 0.333 Hz で、揺れは 11.25 倍と
    整数次からは外れる（旧サーボ機のタクト 2.4 s では 9 倍ちょうどで高調波と重なった）。
    """
    from sensors.read_dump import load_run
    st = ana["startup"]
    base_tact = float(params["cycle"]["tact_s"])   # 起動時間は tact に比例させて同サイクル数にする
    out = {}
    for tact in st["tact_variants_s"]:
        path = dataset.run_core("missing", workdir, filler,
                                st["duration_s"] * tact / base_tact,
                                prefix="startup_", tact_s=tact)
        dump, events = load_run(path)
        case = dataset.case_from_dump("missing", path, dump, events, params,
                                      ana["seed"]["test"], channels=("strain",))
        segs = dataset.segments(case, 0.0, st["revolutions"])
        seg = segs[0]
        ch = case.channels["strain"]
        t, y = dataset.slice_channel(ch, seg.t0, seg.t1)
        freq, amp = features.amplitude_spectrum(y, ch.fs_hz, window="hann")
        m = freq <= 12.0
        figdata[f"tact_{tact:g}_f"] = freq[m]
        figdata[f"tact_{tact:g}_a"] = amp[m]
        sel = (ch.truth_t >= seg.t0) & (ch.truth_t < seg.t1)
        fs_truth = 1.0 / float(np.median(np.diff(ch.truth_t)))
        frt, at = features.amplitude_spectrum(ch.truth[sel], fs_truth, window="hann")
        mt = frt <= 12.0
        figdata[f"tacttruth_{tact:g}_f"] = frt[mt]
        figdata[f"tacttruth_{tact:g}_a"] = at[mt]
        f1 = case.slosh_freq_hz
        lo, hi = (r * f1 for r in ana["spectrum"]["slosh_band_ratio"])
        pk_f, pk_a = features.band_peak(freq, amp, lo, hi)
        out[f"{tact:g}"] = {
            "tact_s": tact, "tact_hz": 1.0 / tact, "slosh_hz": f1,
            "slosh_over_tact": f1 * tact,
            "peak_hz": pk_f, "peak_amp": pk_a,
            "max_dz_over_R": case.max_dz_over_R,
        }
        del dump, events, case
        gc.collect()
    return out


def _stash_startup_figdata(figdata, name, case, seg, ana, params):
    spr = ana["angle_resample"]["samples_per_rev"]
    f1 = case.slosh_freq_hz
    lo, hi = (r * f1 for r in ana["spectrum"]["slosh_band_ratio"])
    for chan in ("strain", "accel_lf_tangential", "current"):
        ch = case.channels.get(chan)
        if ch is None:
            continue
        t, y = dataset.slice_channel(ch, seg.t0, seg.t1)
        ylp, _ = features.order_antialias(y, ch.fs_hz, spr, case.peak_rev_per_s)
        th = features.angle_at(t, case.t_log, case.theta_log)
        sig = features.angle_resample(th, ylp, spr, seg.n_rev, theta0=seg.theta0)
        orders, amp = features.order_spectrum(sig, window="rect")
        figdata[f"sorder_{name}_{chan}_x"] = orders
        figdata[f"sorder_{name}_{chan}_y"] = amp
        fr, ao = features.amplitude_spectrum(y, ch.fs_hz)
        m = fr <= 12.0
        figdata[f"samp_{name}_{chan}_f"] = fr[m]
        figdata[f"samp_{name}_{chan}_a"] = ao[m]
        sel = (ch.truth_t >= seg.t0) & (ch.truth_t < seg.t1)
        fs_truth = 1.0 / float(np.median(np.diff(ch.truth_t)))
        frt, at = features.amplitude_spectrum(ch.truth[sel], fs_truth)
        mt = frt <= 12.0
        figdata[f"strut_{name}_{chan}_f"] = frt[mt]
        figdata[f"strut_{name}_{chan}_a"] = at[mt]
    ch = case.channels.get("strain")
    if ch is not None:
        # 停止区間の扱いを見せる抜粋。センサ雑音を通す前の真値を使う
        # （出力は雑音のほうがずっと大きくて、波形として何も見えないため）。
        t0 = seg.t0
        t1 = t0 + 2.0 * case.tact_s
        sel = (ch.truth_t >= t0) & (ch.truth_t < t1)
        t = ch.truth_t[sel]
        y = ch.truth[sel]
        th = features.angle_at(t, case.t_log, case.theta_log)
        figdata[f"excerpt_{name}_t"] = t - t[0]
        figdata[f"excerpt_{name}_y"] = y
        figdata[f"excerpt_{name}_th"] = np.degrees(th - th[0])
        figdata[f"excerpt_{name}_keep"] = features.plateau_collapse(th).astype(np.int64)
        ts, ys = dataset.slice_channel(ch, t0, t1)
        figdata[f"excerpt_{name}_out_t"] = ts - ts[0]
        figdata[f"excerpt_{name}_out_y"] = ys


def startup_tables(sm, ana) -> str:
    """立ち上げ窓の判定表。"""
    k = ana["detect"]["sigma_fallback"]
    base = sm["conditions"]["normal"]["baseline"]
    unit = sm["conditions"]["normal"]["unit"]
    lines = []
    feats = ("rms", "order1", "order_sb", "slosh_amp", "slosh_rms")
    baselines = {}
    for chan, fv in base.items():
        for feat in feats:
            baselines[(chan, feat)] = detect.build_baseline(
                fv[feat], name=f"{chan}/{feat}", unit=unit.get(chan, ""))
    conds = [c for c in STARTUP_CONDITIONS if c != "ideal"]
    lines.append("| センサ | 特徴量 | 単位 | 正常の基準 (平均 ± σ) | "
                 + " | ".join(CONDITIONS[c]["label"] for c in conds) + " |")
    lines.append("|" + "---|" * (4 + len(conds)))
    for chan in STARTUP_CHANNELS:
        for feat in feats:
            bl = baselines[(chan, feat)]
            cells = []
            for c in conds:
                vals = [v for v in sm["conditions"][c]["test"][chan][feat] if np.isfinite(v)]
                val = float(np.median(vals)) if vals else float("nan")
                v = detect.judge(val, bl, k, feature=feat, channel=chan)
                cells.append(f"{fmt(v.value)} / {fmt(v.z, 3)} σ / "
                             f"{'検出' if v.detected else '見えない'}")
            lines.append(f"| {CHANNEL_LABELS[chan]} | {FEATURE_LABELS[feat]} | {bl.unit} | "
                         f"{fmt(bl.mean)} ± {fmt(bl.std)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 構造モデルの感度
# ---------------------------------------------------------------------------

STRUCTURE_VARIANTS = (
    ("grounded", 250.0, "接地・250 Hz（params.json の既定）"),
    ("grounded", 78.0, "接地・78 Hz（柔らかい架台）"),
    ("free", 250.0, "自由な剛体 a = F / m"),
)


def structure_sweep(name, path, dump, events, params, ana, noise_floor, figdata, seg) -> dict:
    """params.json は書き換えず、読み込んだ辞書だけ差し替えて 3 通り回す。

    加速度センサが力を加速度に変える伝達は、架台をどうモデル化するかで決まる。
    接地した 2 次系（アクセレランス）なら共振より下は f^2 で落ちるし、
    自由な剛体なら a = F/m で低周波もそのまま出る。実機を測らずには決まらない。
    """
    f1 = None
    out = {}
    for model, freq, label in STRUCTURE_VARIANTS:
        p = dataset.structure_variant(params, model, freq)
        case = dataset.case_from_dump(name, path, dump, events, p, ana["seed"]["test"],
                                      channels=ACCEL_LF)
        f1 = case.slosh_freq_hz
        lo, hi = (r * f1 for r in ana["spectrum"]["slosh_band_ratio"])
        key = f"{model}_{freq:g}"
        rec = {"model": model, "freq_hz": freq, "label": label, "channels": {}}
        for chan in ACCEL_LF:
            ch = case.channels[chan]
            _, y = dataset.slice_channel(ch, seg.t0, seg.t1)
            _, out_amp = features.line_amplitude(y, ch.fs_hz, lo, hi)
            sel = (ch.truth_t >= seg.t0) & (ch.truth_t < seg.t1)
            fs_truth = 1.0 / float(np.median(np.diff(ch.truth_t)))
            _, truth_amp = features.line_amplitude(ch.truth[sel], fs_truth, lo, hi)
            floor = noise_floor.get(chan, float("nan"))
            rec["channels"][chan] = {
                "out_amp": out_amp, "truth_amp": truth_amp, "floor_amp": floor,
                "snr_db": 20.0 * np.log10(truth_amp / floor)
                if np.isfinite(floor) and floor > 0 and truth_amp > 0 else float("nan"),
            }
            if chan == "accel_lf_tangential":
                frt, at = features.amplitude_spectrum(ch.truth[sel], fs_truth)
                m = frt <= 12.0
                figdata[f"struct_{name}_{key}_ftruth"] = frt[m]
                figdata[f"struct_{name}_{key}_atruth"] = at[m]
                fr, ao = features.amplitude_spectrum(y, ch.fs_hz)
                m2 = fr <= 12.0
                figdata[f"struct_{name}_{key}_f"] = fr[m2]
                figdata[f"struct_{name}_{key}_a"] = ao[m2]
        out[key] = rec
        del case
        gc.collect()
    out["_slosh_hz"] = f1
    return out


def structure_table(structure: dict, noise_floor: dict) -> str:
    lines = ["| 条件 | 構造モデル | 真値のスロッシングの線 [m/s^2] | 雑音床 [m/s^2] |"
             " S/N [dB] | 出力に出るか |",
             "|---|---|---|---|---|---|"]
    for name, rec in structure.items():
        for key, v in rec.items():
            if key.startswith("_"):
                continue
            c = v["channels"]["accel_lf_tangential"]
            visible = "出る" if c["snr_db"] > 0 else "埋もれる"
            lines.append(
                f"| {CONDITIONS[name]['label']} | {v['label']} | {fmt(c['truth_amp'])} | "
                f"{fmt(c['floor_amp'])} | {c['snr_db']:+.1f} | {visible} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 図の材料
# ---------------------------------------------------------------------------

def _stash_figdata(figdata, name, case, ana, params):
    """図に使う配列だけ抜いておく（ダンプを捨てても図が描けるように）。"""
    skip = ana["run"]["startup_revolutions"]
    spr = ana["angle_resample"]["samples_per_rev"]
    n_rev_long = ana["segment"]["envelope_revolutions"]
    segs = dataset.segments(case, skip, n_rev_long)
    if not segs:
        return
    seg = segs[0]
    f1 = case.slosh_freq_hz
    lo, hi = (r * f1 for r in ana["spectrum"]["slosh_band_ratio"])

    # 次数比スペクトル（strain）
    for chan in ("strain", "accel_lf_tangential", "current"):
        ch = case.channels.get(chan)
        if ch is None:
            continue
        t, y = dataset.slice_channel(ch, seg.t0, seg.t1)
        ylp, _ = features.order_antialias(y, ch.fs_hz, spr, case.peak_rev_per_s)
        th = features.angle_at(t, case.t_log, case.theta_log)
        sig = features.angle_resample(th, ylp, spr, seg.n_rev, theta0=seg.theta0)
        orders, amp = features.order_spectrum(sig, window="rect")
        figdata[f"order_{name}_{chan}_x"] = orders
        figdata[f"order_{name}_{chan}_y"] = amp
        # 時間軸のスペクトル（スロッシング帯を見るため）
        _, _, (fr, psd) = features.band_rms(y, ch.fs_hz, lo, hi,
                                            ana["spectrum"]["psd_resolution_hz"])
        figdata[f"psd_{name}_{chan}_f"] = fr
        figdata[f"psd_{name}_{chan}_p"] = psd

    # 包絡線の次数スペクトル（accel_hf）
    ch = case.channels.get("accel_hf_radial")
    if ch is not None:
        fault = params["faults"]["bearing_outer_race"]
        ring = float(fault["ring_freq_hz"])
        blo, bhi = (r * ring for r in ana["envelope"]["band_ratio"])
        t, y = dataset.slice_channel(ch, seg.t0, seg.t1)
        try:
            env = features.envelope(y, ch.fs_hz, (blo, bhi),
                                    ring * ana["envelope"]["lowpass_ratio"],
                                    order=ana["envelope"]["filter_order"], method="abs",
                                    decimate_to_hz=ana["envelope"]["decimate_target_hz"],
                                    t0=float(t[0]))
        except features.BandOutOfRange:
            return
        # 包絡線の時間波形（先頭 1 回転ぶん）。軸受はモータ軸で定速なので欠陥通過
        # BPFO は時間的に一定。図はこの波形をそのまま時間領域で FFT して横軸
        # 周波数[Hz]の包絡線スペクトルにする（テーブル角へのリサンプルはしない）。
        m = env.t < env.t[0] + case.revolution_s
        figdata[f"envwave_{name}_t"] = env.t[m] - env.t[0]
        figdata[f"envwave_{name}_y"] = env.y[m]
        bt = case.bearing_times
        figdata[f"envwave_{name}_events"] = bt[(bt >= env.t[0]) &
                                               (bt < env.t[0] + case.revolution_s)] - env.t[0]


if __name__ == "__main__":
    main()
