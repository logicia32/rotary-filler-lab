"""供給と排出を入れたときに、機械が定常状態で回り続けるかを見る。

工程は 供給 -> 充填 -> （通過）-> 排出。1 本のボトルは供給で入り、次の停留で充填され、
そこから運ばれて排出される。滞留は工程角から一意に決まる
（`(排出角 - 供給角) / ピッチ` を法 N で。既定の 315 / 0 / 225 deg なら **6 割出し = 18 秒**）。

この文書は、それが数字で成り立っているかを確かめる。

  - 満量のホルダ数が「滞留 - 1」本で一定か（残りは 空瓶 1 + 空ホルダ N-滞留）
  - ホルダごとの液量が n_stations サイクル周期で繰り返すか
  - 負荷慣性 J_load と曲げモーメントが 1 サイクル周期で戻るか
    （どのホルダが満量かは毎サイクル変わるが、本数は変わらないので和は 1 サイクル周期）
  - 1 サイクルに入る量と出る量が釣り合っているか
  - 供給から排出までが滞留のとおりか（イベント列から数える）

初期状態は工程配置から作った定常状態（FORMAT.md 4 節）なので、**ホルダの中身の並びは
最初のサイクルから定常**。ただし液の揺れはそうならない。t=0 のボトルは揺れが 0 で、
本当の定常なら「前の割出しの残り」を持っているはずだからで、そこに落ち着くには
減衰の時定数 1/(zeta*w1) = 8.5 s の数倍が要る。**判定は n_stations サイクル
（24 秒）ぶん回してから**にしてある。

    make -C core steady
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dumpio import read_dump, read_events, EV_KINDS   # noqa: E402

EV_INFEED, EV_DISCHARGE = 2, 4

# f64 で書いていれば厳密に一致するが、f32 のダンプでも通るようにしておく
TOL_REL = 1.0e-6


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".steady.bin")
    d = read_dump(path)
    hd = d.header
    n = d.n_stations
    tact = hd.tact_s
    if tact <= 0:
        raise SystemExit("ヘッダに tact が入っていない")
    n_cycle = int(round((d.t[-1] + hd.log_dt_s) / tact))
    bad = 0

    print(f"{os.path.basename(path)}: {d.summary()}  tact={tact:g} s  {n_cycle} サイクル")
    if n_cycle < 3 * n:
        print(f"  NG: 判定には {3 * n} サイクル以上要る（いまは {n_cycle}）")
        return 1
    warm = n          # 揺れが定常に落ち着くまでの助走（減衰時定数 8.5 s の 3 倍弱）
    pitch = 2.0 * math.pi / n
    n_res = int(round((hd.discharge_angle_rad - hd.infeed_angle_rad) / pitch)) % n
    n_full_want = n_res - 1          # 満量（供給直後の空瓶を除く）
    print(f"  工程 供給 {math.degrees(hd.infeed_angle_rad):.0f} / "
          f"充填 {math.degrees(hd.fill_angle_rad):.0f} / "
          f"排出 {math.degrees(hd.discharge_angle_rad):.0f} deg  ->  "
          f"滞留 {n_res} 割出し = {n_res * tact:.1f} s、"
          f"定常は 満量 {n_full_want} + 空瓶 1 + 空ホルダ {n - n_res}")
    # 充填の止め方は dt に量子化されるので、1 本あたり flow*dt だけばらつく
    quant_mL = hd.flow_rate_m3_s * hd.dt_s * 1e6

    # 各サイクルの停留の終わり（次の割出しが始まる直前）で切る
    idx = [int(np.searchsorted(d.t, (k + 1) * tact - 1e-9)) - 1 for k in range(n_cycle)]
    idx = [i for i in idx if 0 <= i < len(d.t)]
    V = d.V[idx] * 1e6      # [サイクル, ホルダ] mL

    print("\n--- 満量のホルダ数（サイクル末） ---")
    full = (V > 1.0).sum(axis=1)
    print("  " + " ".join(str(v) for v in full))
    if np.all(full[warm:] == n_full_want):
        print(f"  OK: いつも {n_full_want} 本が満量、残りが 空瓶 1 + 空ホルダ {n - n_res}")
    else:
        print(f"  NG: 定常なら常に {n_full_want} 本のはずが "
              f"{sorted(set(full[warm:].tolist()))}")
        bad += 1

    print(f"\n--- 液量の {n} サイクル周期（助走 {warm} サイクルの後）---")
    a, b = V[warm:warm + n], V[warm + n:warm + 2 * n]
    err = float(np.max(np.abs(a - b)))
    tol = 3.0 * quant_mL
    print(f"  サイクル {warm}..{warm + n - 1} と {warm + n}..{warm + 2 * n - 1} の差: "
          f"{err:.3e} mL（許容 {tol:.3e} = 充填の量子化 flow*dt の 3 倍）")
    if err > tol:
        print("  NG: 周期になっていない")
        bad += 1
    else:
        print("  OK: 同じ並びが繰り返している")
    for k in range(warm, warm + n):
        print(f"    cycle {k:2d}  " + " ".join(f"{v:6.1f}" for v in V[k]))
    print(f"  * 充填量は 1 本ごとに最大 {quant_mL:.4f} mL ばらつく"
          "（弁を閉じる判定が積分刻みに量子化されるため。実機の話ではない）")

    print("\n--- J_load と曲げモーメントの 1 サイクル周期 ---")
    step = int(round(tact / d.log_dt_s))
    for name, arr, tol in (("J_load", d.j_load, 3.0 * quant_mL * 1e-6 * hd.pitch_radius_m ** 2 * 1e3),
                           ("M_bend", d.m_bend, 3.0 * quant_mL * 1e-6 * 1e3 * 9.80665 * hd.pitch_radius_m),
                           ("T_table", d.torque_table, 1.0e-4)):
        seg = [arr[(warm + k) * step:(warm + k + 1) * step] for k in range(4)]
        m = min(len(x) for x in seg)
        if m < 2:
            continue
        dif = [np.abs(seg[0][:m] - seg[j][:m]) for j in range(1, len(seg))]
        e = max(float(np.max(x)) for x in dif)
        n_over = max(int(np.count_nonzero(x > tol)) for x in dif)
        # 弁が開く時刻（停留開始 + 0.08 s）は積分刻みの格子とちょうど重なるので、
        # サイクルによって 1 ステップ前後する。そこにログ点が乗ると、その 1 点だけ
        # 噴流（0.159 Nm）の有無が入れ替わる。物理ではなく境目の丸め。
        okk = (e <= tol) or (n_over <= 2)
        print(f"  {'OK ' if okk else 'NG '} {name}: サイクル {warm} と {warm + 1}〜{warm + 3} の差 "
              f"{e:.3e}（振れ幅 {float(np.max(arr) - np.min(arr)):.6f}、許容 {tol:.1e}、"
              f"超えた点 {n_over}/{m}）")
        if not okk:
            bad += 1
        elif n_over:
            print(f"      * {n_over} 点だけ外れる。弁が開く時刻が積分刻みの格子と重なるため、"
                  " サイクルによって開弁が 1 ステップ前後し、その点だけ噴流の有無が入れ替わる。"
                  " MODEL.md 7.1 の PLC 5 ms 格子を入れれば消えるが、py/ref.py が"
                  " 実装していないのでこちらも入れていない")
    print(f"  J_load は {float(np.min(d.j_load)):.6f}〜{float(np.max(d.j_load)):.6f} kg m^2 "
          f"（充填で増える分 + 受け渡しで落ちる分）")
    # **単位の違う量を比べないこと。** 下は 2 つ別の比で、混ぜると桁を間違える。
    mb = float(np.max(d.m_bend))
    ts = float(np.max(np.abs(d.torque_slosh)))
    fh = float(np.max(np.hypot(d.f_tab_x, d.f_tab_y)))
    arm = 0.150      # sensors.strain.arm_length_mm。**センサ側の値で、機械側の確認は無い**
    print(f"  M_bend は {float(np.min(d.m_bend)):.4f}〜{mb:.4f} Nm、"
          f" T_slosh は最大 {ts:.4f} Nm、水平合力は最大 {fh:.4f} N")
    print(f"  (a) 曲げ / スロッシング反力トルク = {mb:.4f} / {ts:.4f} = {mb / max(1e-12, ts):.1f} 倍"
          "   ← どちらも Nm だが、片方は z 軸まわり、片方は水平軸まわり")
    print(f"  (b) 曲げ / 水平力の曲げ寄与 = {mb:.4f} / ({fh:.4f} x {arm:.3f}) = "
          f"{mb / max(1e-12, fh * arm):.1f} 倍"
          "   ← 同じ「支持部の曲げ」どうしの比。腕の長さは仮置き")
    print("  M_bend には水平力の寄与を入れていない（取り付け位置と腕の長さが未決）")

    print("\n--- 1 サイクルの出入り（イベント列から）---")
    ev_path = path + ".events"
    if os.path.exists(ev_path):
        ev = read_events(ev_path)
        ins, outs = ev.of(EV_INFEED), ev.of(EV_DISCHARGE)
        print(f"  供給 {len(ins)} 本 / 排出 {len(outs)} 本 / {n_cycle} サイクル")
        if len(ins) == n_cycle and len(outs) == n_cycle:
            print("  OK: 1 サイクルに 1 本ずつ入って 1 本ずつ出る")
        else:
            print("  NG: 出入りの本数がサイクル数と合わない")
            bad += 1
        if len(outs):
            vol = outs.d[:, 0] * 1e6
            print(f"  排出した液量: {vol.min():.3f}〜{vol.max():.3f} mL "
                  f"（目標 {hd.target_volume_m3 * 1e6:.1f} mL）")
            okv = float(np.max(np.abs(vol - hd.target_volume_m3 * 1e6))) < 0.05
            print(f"  {'OK ' if okv else 'NG '} 満量のまま抜けている")
            if not okv:
                bad += 1
        # 供給から排出までの割出し回数
        lag = []
        for st, t0 in zip(ins.station, ins.t):
            later = outs.t[(outs.station == st) & (outs.t > t0)]
            if len(later):
                lag.append(round((float(later[0]) - float(t0)) / tact))
        if lag:
            uniq = sorted(set(lag))
            okl = uniq == [n_res]
            print(f"  {'OK ' if okl else 'NG '} 供給から排出まで {uniq} 割出し"
                  f"（{n_res} 割出し = {n_res * tact:.1f} s のはず）")
            if not okl:
                bad += 1
        else:
            print("  ! 供給と排出の対応が取れるサイクルがない（もっと長く回すこと）")
    else:
        print("  ! イベント列が無いので出入りは数えられない（--no-events で回した？）")

    spill = float(np.max(d.spill)) * 1e6
    print(f"\n  こぼれ積算の最大: {spill:.4g} mL")
    print(f"  最大傾き {hd.max_tilt_rad * 1e3:.3f} mrad / dz/R {hd.max_dz_over_R:.4f}"
          f"（適用範囲 {hd.range_limit:.2f}、逸脱 {hd.range_exceeded}）")

    print()
    if bad:
        print(f"NG: {bad} 件。定常状態になっていない")
        return 1
    print("OK: 供給と排出を入れると定常状態で回り続ける")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
