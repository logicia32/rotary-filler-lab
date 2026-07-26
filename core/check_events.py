"""イベント列（サイドカー `RFEVT002`）が取り決めどおりかを確かめる。

イベントを別ファイルにした理由は SENSORS.md 5.5 節と core/FORMAT.md の冒頭にある。
軸受のリンギング 3 kHz を連続ログ 4 kHz に乗せるとナイキストを割って折り返すので、
衝撃は波形にせず「いつ・どれだけ」の列として渡し、波形への合成はセンサ層が
各センサの刻みで行う。

ここで見るのは 7 つ。

  1. ヘッダの実効値が、その回に実際に使った値になっているか
  2. 時刻が昇順で、種別の割り当てが FORMAT.md 5.3 のとおりか
  3. **軸受は時間軸で等間隔**（モータ軸に置いてあるので停留中も打つ）
  4. 軸受のモータ軸角が `motor_speed * t` と合うか
  5. カムフォロワは割出しの入口と出口だけ、1 サイクル 2 件
  6. 受け渡しは停留の先頭、1 サイクル 1 件ずつ。排出のホルダ番号が幾何と合うか
  7. **衝撃が連続量に漏れていないこと**
     （カムフォロワ＋軸受だけを有効にしたダンプが、正常時と 1 ビットも違わない）

    ../.venv/bin/python check_events.py <dump> [<events>]
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dumpio import read_dump, read_events, EV_KINDS   # noqa: E402

EV_BEARING, EV_CAM, EV_INFEED, EV_MISSED, EV_DISCHARGE, EV_DRIP = 0, 1, 2, 3, 4, 5


class Checks:
    def __init__(self):
        self.bad = 0

    def ok(self, cond, text):
        cond = bool(cond)
        if not cond:
            self.bad += 1
        print(f"  {'OK ' if cond else 'NG '} {text}")
        return cond

    def note(self, text):
        print(f"  *  {text}")


def holder_at(world, th_t, n):
    """世界角 world にいるホルダ番号（FORMAT.md 4 節）。"""
    pitch = 2.0 * math.pi / n
    return int(np.round((world - th_t) / pitch)) % n


def main() -> int:
    dump_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".fault.bin")
    ev_path = sys.argv[2] if len(sys.argv) > 2 else dump_path + ".events"

    d = read_dump(dump_path)
    ev = read_events(ev_path)
    hd = d.header
    h = ev.header
    c = Checks()
    n = d.n_stations
    tact = hd.tact_s
    n_cycle = int(round((d.t[-1] + hd.log_dt_s) / tact))

    from collections import Counter
    cnt = Counter(EV_KINDS.get(int(k), str(k)) for k in ev.kind)
    print(f"{os.path.basename(ev_path)}: {len(ev)} 件 / {n_cycle} サイクル")
    print(f"  内訳: {dict(cnt)}")
    print(f"  軸受 {h['bearing_defect_freq_hz']:.2f} Hz / 振幅 {h['bearing_accel_m_s2']:.4g} m/s^2"
          f" / リンギング {h['bearing_ring_freq_hz']:.0f} Hz (zeta {h['bearing_ring_damping']:.4g})")
    print(f"  カムフォロワ 衝撃 {h['cam_impact_torque_Nm']:.4g} Nm / "
          f"すきま {h['cam_clearance_rad'] * 180 / math.pi:.4g} deg")

    print("\n--- 1. ヘッダの実効値 ---")
    c.ok(abs(h["dt_s"] - hd.dt_s) < 1e-15, f"連続ログと同じ dt ({h['dt_s']:g} s)")
    c.ok(h["fault_flags"] == hd.fault_flags,
         f"故障フラグが一致 ({h['fault_flags']:#06b})")
    if hd.fault_bearing:
        c.ok(abs(h["bearing_defect_freq_hz"] - hd.bearing_defect_freq_hz) < 1e-4,
             "軸受の欠陥通過周波数が連続ログのヘッダと一致")
        c.ok(h["bearing_ring_freq_hz"] > 0.0 and h["bearing_ring_damping"] > 0.0,
             "センサ層が波形を合成するのに要る定数（リンギング周波数・減衰比）が入っている")

    print("\n--- 2. 並びと種別 ---")
    c.ok(np.all(np.diff(ev.t) >= -1e-15), "時刻の昇順で並んでいる")
    c.ok(set(int(k) for k in ev.kind) <= set(EV_KINDS),
         f"種別が FORMAT.md 5.3 の範囲に収まっている: {sorted(set(int(k) for k in ev.kind))}")
    c.ok(np.all(ev.t >= -1e-15) and np.all(ev.t <= n_cycle * tact + 1e-9),
         "時刻が計算区間の中にある")

    print("\n--- 3. 軸受（モータ軸。時間軸で等間隔・停留中も打つ）---")
    be = ev.of(EV_BEARING)
    if len(be) >= 3:
        period = 1.0 / h["bearing_defect_freq_hz"]
        dt_ev = np.diff(be.t)
        err = float(np.max(np.abs(dt_ev - period)))
        c.ok(err < 1e-12,
             f"間隔 {period * 1e3:.4f} ms の等間隔（最大ずれ {err * 1e12:.2f} ps）")
        c.ok(abs(len(be) / n_cycle - h["bearing_defect_freq_hz"] * tact) < 1.5,
             f"1 タクトあたり {len(be) / n_cycle:.1f} 件"
             f"（{h['bearing_defect_freq_hz'] * tact:.1f} 件のはず）")
        # 停留中（psi >= 180 deg）にも出ているか。前版はここが 0 だった。
        in_dwell = (be.t % tact) >= hd.index_time_s - 1e-12
        c.ok(in_dwell.sum() > 0.4 * len(be),
             f"停留中にも {int(in_dwell.sum())} 件（テーブルが止まっていても打ち続ける）")
        # モータ軸角が motor_speed*t と合うか（f32 なので相対 1e-6 まで）
        want = hd.motor_speed_rad_s * be.t
        rel = float(np.max(np.abs(be.th_m - want) / np.maximum(1.0, np.abs(want))))
        c.ok(rel < 1e-6, f"モータ軸角が motor_speed*t と一致（相対差 {rel:.2e}）")
        # テーブル角は連続ログから補間した値と合うか
        t_in = np.interp(be.t, d.t, d.th_t)
        derr = float(np.max(np.abs(t_in - be.th_t)))
        c.ok(derr < 1e-3,
             f"テーブル角が連続ログと整合（最大差 {derr * 1e3:.3f} mrad、"
             f"ログの刻み {hd.log_dt_s * 1e3:.2f} ms ぶんの補間誤差を含む）")
        c.note(f"1 サイクル {len(be) / n_cycle:.1f} 件。旧版（テーブル軸に置いていた）は"
               " 0.44 件だった。包絡線解析が実用域に入るのはこの違い")
    elif hd.fault_bearing:
        c.ok(False, "軸受故障が有効なのにイベントが 3 件未満")
    else:
        print("  -  軸受故障が無効。この節は飛ばす")

    print("\n--- 4. カムフォロワ（割出しの入口と出口）---")
    cm = ev.of(EV_CAM)
    if len(cm):
        c.ok(len(cm) == 2 * n_cycle, f"1 サイクル 2 件（{len(cm)} 件 / {n_cycle} サイクル）")
        phase = cm.t % tact
        at_in = np.abs(phase) < 1e-12
        at_out = np.abs(phase - hd.index_time_s) < 1e-12
        c.ok(np.all(at_in | at_out), "すべて割出しの入口（psi=0）か出口（psi=180deg）にある")
        c.ok(int(at_in.sum()) == n_cycle and int(at_out.sum()) == n_cycle,
             f"入口 {int(at_in.sum())} 件 / 出口 {int(at_out.sum())} 件")
        c.ok(np.all(np.abs(cm.d[:, 0] - h["cam_impact_torque_Nm"]) < 1e-6),
             "衝撃トルクがヘッダの実効値と一致")
        c.note("割出しの途中（加速度が符号を変える tau=0.5）でも当たりは起きるはずだが、"
               " params.json が入口と出口しか書いていないのでそこに合わせてある。未確認")
    elif hd.fault_cam:
        c.ok(False, "カムフォロワ故障が有効なのにイベントが無い")
    else:
        print("  -  カムフォロワ故障が無効。この節は飛ばす")

    print("\n--- 5. 受け渡し（停留の先頭）---")
    ins = ev.of(EV_INFEED)
    outs = ev.of(EV_DISCHARGE)
    miss = ev.of(EV_MISSED)
    c.ok(len(ins) + len(miss) == n_cycle,
         f"供給（または欠品）は 1 サイクル 1 件（{len(ins)} + {len(miss)} / {n_cycle}）")
    c.ok(len(outs) <= n_cycle, f"排出は 1 サイクル 1 件以下（{len(outs)}）")
    if len(outs):
        phase = outs.t % tact
        c.ok(float(np.max(np.abs(phase - hd.index_time_s))) < 1e-9,
             "排出はすべて停留の先頭で起きている")
        # 排出のホルダ番号が幾何と合うか
        want = [holder_at(hd.discharge_angle_rad, th, n) for th in outs.th_t]
        c.ok(list(outs.station) == want,
             f"排出のホルダ番号が holder_at(排出角, th_t) と一致: {list(outs.station)}")
        c.ok(np.all(outs.d[:, 0] > 0.0), "排出したボトルには中身が入っている")
        c.note(f"排出時の傾き {outs.d[:, 1].min() * 1e3:.2f}〜{outs.d[:, 1].max() * 1e3:.2f} mrad。"
               " 揺れたまま外へ持ち出されるので、連続量には残らない")
    if len(ins):
        want = [holder_at(hd.infeed_angle_rad, th, n) for th in ins.th_t]
        c.ok(list(ins.station) == want, "供給のホルダ番号が holder_at(供給角, th_t) と一致")
    if len(miss):
        c.ok(np.all(miss.station == hd.missing_station),
             f"欠品はいつも同じホルダ（st{hd.missing_station}）")

    print("\n--- 6. 弁の液垂れ ---")
    dr = ev.of(EV_DRIP)
    if len(dr):
        c.ok(len(dr) <= n_cycle, f"1 サイクル 1 件以下（{len(dr)} 件）")
        c.ok(np.all(dr.d[:, 0] > 0.0), "液垂れの体積が正")
        c.note(f"液垂れ {dr.d[0, 0] * 1e6:.3f} mL/回。"
               " これは割出し中のテーブル上に落ちるものとして、連続量には入れていない")
    else:
        print("  -  液垂れ無し")

    print("\n--- 7. 衝撃が連続量に漏れていないこと ---")
    if hd.fault_bearing or hd.fault_cam:
        base = os.path.join(HERE, ".noimp.bin")
        exe = os.path.join(HERE, "filler")
        args = [exe, "--cycles", str(n_cycle), "--log-rate", str(1.0 / hd.log_dt_s),
                "--out", base, "--no-events", "--quiet", "--no-faults"]
        if hd.fault_valve or hd.fault_missing:
            print("  -  弁・欠品も有効なダンプなので、この比較はできない"
                  "（この 2 つは連続量を変える）")
        else:
            r = subprocess.run(args, capture_output=True)
            if r.returncode != 0:
                c.ok(False, f"比較用の正常データを作れなかった: {r.stderr.decode()[:200]}")
            else:
                d0 = read_dump(base)
                same = True
                for nm in ("th_t", "omega", "alpha", "torque_table", "torque_input",
                           "torque_slosh", "m_bend", "f_tab_x", "f_tab_y", "j_load"):
                    a, b = d.scal[nm], d0.scal[nm]
                    if a.shape != b.shape or not np.array_equal(a, b):
                        same = False
                        print(f"     {nm} が違う（最大差 "
                              f"{float(np.max(np.abs(a - b))) if a.shape == b.shape else float('nan'):.3e}）")
                c.ok(same, "カムフォロワ＋軸受を有効にしても連続量が 1 ビットも変わらない"
                           "（衝撃はイベント列にしか無い）")
                os.remove(base)
        c.note("カムフォロワ摩耗のテーブル角オフセット（MODEL.md 8.4 の ±clearance/2）は"
               " py/ref.py が実装していないので、こちらも入れていない。既知の欠落")
    else:
        print("  -  衝撃系の故障が無効。この節は飛ばす")

    print()
    if c.bad:
        print(f"NG: {c.bad} 件")
        return 1
    print("OK: イベント列は取り決めどおり")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
