"""解析側の取り決め（params.json に無い数値をここに集める）。

params.json は「外から決まる機械の仕様」を置く場所で、書き換えない。
一方、窓の長さ・帯域の幅・何シグマにするか、といった解析の選び方は
機械の仕様ではないので params.json には無い。かといってコードの奥に
直書きすると根拠が追えなくなるので、この 1 ファイルに集めて全部に理由を付ける。

同じ書式を params.json に足すなら `analysis` セクションになる。
提案の形は `analysis/PARAMS_ADDED.md` に書いた。
"""

from __future__ import annotations

import copy
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = LAB_ROOT / "params.json"

# ---------------------------------------------------------------------------
# 解析の取り決め
# ---------------------------------------------------------------------------

ANALYSIS = {
    "run": {
        "duration_s": 600.0,
        "source": "タクト 3.0 s × 8 ステーション = 1 回転 24.0 s。600 s = 200 サイクル ="
                  " ちょうど 25 回転。次数比の窓を回転で割り切るために整数回転に合わせた"
                  "（旧サーボ機はタクト 2.4 s = 1 回転 19.2 s で 480 s = 25 回転だった）",
        "startup_revolutions": 3.0,
        "startup_source": "004 の既定は工程配置から作った定常状態で始まる（空テーブルからの"
                          "充填過渡は --no-prime のときだけ）。それでも初期条件（スロッシング"
                          "phi・速度）は厳密には周期的でないので、過渡が落ちるまで頭の 3 回転を"
                          "捨てる。25 回転から 3 を引いて基準窓は 22 個（自由度 21）",
    },

    "startup": {
        "duration_s": 27.0,
        "duration_source": "9 サイクル（1 回転 = 8 サイクル ＋ 1 サイクルの余裕）× タクト 3.0 s",
        "revolutions": 1,
        "n_baseline_seeds": 24,
        "n_test_seeds": 8,
        "seeds_source": "物理コアは決定論的なので、同じ条件を何回回しても同じ波形が出る。"
                        "正常データのばらつきはセンサ雑音しかない。そこで窓を分けるかわりに"
                        "雑音の種を変えて基準を作る。基準 24 通り、試験 8 通り",
        "tact_variants_s": [3.0, 3.4],
        "tact_variants_source": "スロッシング 3.751 Hz は液深と容器径だけで決まりタクトに依らない。"
                                "タクトを変えると割出しの高調波（1/tact の倍数）は動くが揺れは動かない。"
                                "これで「回転に同期しない成分」だと示せる。基準 3.0 s と 3.4 s の 2 点。"
                                "カム式は停留時間 = tact/2（割付 180deg / 停留 180deg）なので、"
                                "充填所要 1.322 s（弁遅れ込み・params.json fill）を停留が上回るには"
                                "tact ≧ 2.65 s が要る。旧サーボ機で使えた 2.6 s は停留 1.3 s となり"
                                "充填が 1 サイクルに収まらないので選べない",
    },

    "segment": {
        "base_revolutions": 1,
        "base_source": "次数比は 1 回転を整数個並べた窓でしか整数次に落ちない。"
                       "1 回転ちょうどにすると次数 1 が DFT のビン中心に乗り、"
                       "矩形窓のまま他の整数次と漏れなく分離できる",
        "envelope_revolutions": 4,
        "envelope_source": "包絡線スペクトルに要る窓長。4 回転 = 96 s（タクト 3.0 s）で"
                           "SENSORS.md の「60 s 以上」を満たす。"
                           "NOTE(004): 旧版の軸受はテーブル軸（欠陥 0.447 Hz・1 回転 3.58 回）で"
                           "この 4 回転は「発数を稼ぐ」ためだったが、rev.3 で軸受はモータ軸へ移り"
                           "欠陥通過 89.5 Hz・時間等間隔（1 タクト 268 件）になった。窓長としては"
                           "問題ないが、包絡線を「テーブル回転の次数」で見る run_matrix の枠組みは"
                           "モータ同期の欠陥には合わない（報告参照）",
    },

    "spectrum": {
        "psd_resolution_hz": 0.1,
        "psd_resolution_source": "定常状態の応答はタクト周波数 1/3.0 = 0.3333 Hz の高調波の線スペクトルになる。"
                                 "隣の線と混ざらないよう、分解能をその 1/3（0.111 Hz）以下に取り 0.1 Hz とした"
                                 "（旧タクト 2.4 s = 0.4167 Hz のときは 0.125 Hz）",
        "slosh_band_ratio": [0.90, 1.10],
        "slosh_band_source": "スロッシングの帯域。減衰比 0.005 なので共振自体の半値幅は 0.04 Hz しかないが、"
                             "液深が浅い側では f1 が下がる（120 mm で 3.751 Hz、50 mm で 3.738 Hz、"
                             "20 mm で 3.381 Hz）。満量まわりの変動を拾える幅として ±10 % を取った",
        "defect_order_band_ratio": [0.90, 1.10],
        "defect_order_band_source": "包絡線の次数スペクトルで欠陥次数を探す幅。"
                                    "非整数次（3.58）はビン中心に乗らないので、窓の主ローブぶんの余裕が要る",
    },

    "envelope": {
        "band_ratio": [0.70, 1.40],
        "band_source": "帯域通過の幅。リンギング周波数を中心に、減衰 5 % の共振（半値幅 10 %）を"
                       "十分に包む幅として ±30〜40 % を取った。広く取りすぎると雑音を、"
                       "狭く取りすぎると衝撃の立ち上がりを落とす",
        "lowpass_ratio": 0.20,
        "lowpass_source": "包絡線の低域通過。リンギングの 20 %（3 kHz なら 600 Hz）。"
                          "見たい欠陥通過周波数（回転 0.052 Hz の 3.58 倍 = 0.19 Hz）よりはるかに高く、"
                          "搬送波の 2 倍（6 kHz）よりは十分低い",
        "filter_order": 4,
        "filter_order_source": "解析側の帯域通過。センサ層の 2 次と違い、ここは後処理なので"
                               "零位相（filtfilt）で掛ける。実効 8 次相当",
        "decimate_target_hz": 1600.0,
        "decimate_source": "包絡線は低域通過済みなので間引いてよい。"
                           "角度リサンプルの手前で軽くして計算量を落とす",
        "impact_threshold_sigma": 5.0,
        "impact_threshold_source": "衝撃のイベント検出。包絡線の中央値+5×(中央絶対偏差から出した σ)。"
                                   "平均と標準偏差だと衝撃自身が基準を持ち上げるので、"
                                   "外れ値に強い中央値ベースにした",
        "impact_min_separation_ratio": 0.25,
        "impact_min_separation_source": "続けて数える最小間隔を、欠陥通過間隔の 25 % に取る。"
                                        "1 発の減衰振動を 2 回数えないため",
    },

    "angle_resample": {
        "samples_per_rev": 1024,
        "samples_per_rev_source": "角度の刻み 2pi/1024 = 6.1e-3 rad。カムの最大角速度"
                                  "0.921 rad/s（params.json cycle.table_omega_max_rad_s）のとき"
                                  "1000 Hz サンプリングでの角度前進は 9.2e-4 rad なので、"
                                  "いちばん速い瞬間でも 1 刻みに 6.7 点ある（補間が効く条件）",
        "samples_per_rev_envelope": 4096,
        "samples_per_rev_envelope_source": "包絡線は 1600 Hz まで間引いてあるので、より細かく取れる。"
                                           "次数 2048 まで見える",
        "dwell_rule": "last",
        "dwell_rule_source": "停止区間は角度が進まないので、角度領域には存在しない。"
                             "プラトーを 1 点に潰すとき、その最後の点（次の割出しが引き継ぐ状態）を残す",
    },

    "detect": {
        "target_false_alarm_per_month": 1.0,
        "target_source": "特徴量は 1 回転（24.0 s）に 1 個出る = 1 日 3600 個。"
                         "誤警報を月 1 回未満に抑えるなら片側確率 9.3e-6 が要る",
        "sigma_fallback": 6.0,
        "sigma_fallback_source": "上の条件を自由度 21（基準 22 窓）の t 分布で解くと 5.6 σ 前後。"
                                 "基準サンプル数が変わっても揺れないよう、丸めた 6 σ を既定にする。"
                                 "実際の値は run_matrix が detect.sigma_for_false_alarm で毎回計算して併記する",
    },

    "seed": {
        "baseline": 20260722,
        "test": 20260723,
        "source": "センサ雑音の種。基準を作る側と試験する側で必ず変える。"
                  "同じ種で作った基準に自分を当てるのは試験になっていない",
    },
}


def analysis_params(overrides: dict | None = None) -> dict:
    """解析の取り決めを返す。overrides で一部だけ差し替えられる。"""
    out = copy.deepcopy(ANALYSIS)
    if overrides:
        for sec, vals in overrides.items():
            out.setdefault(sec, {}).update(vals)
    return out


def load_machine_params():
    """params.json を読む。

    読み込みは `sensors.virtual.load_params` に任せる（センサ層が既定値を補うのと
    同じ辞書でないと、こちらの解析とセンサ層の見ている値がずれる）。
    返り値は (params, 既定値で補ったキーの一覧)。ファイルは読むだけ。

    `py/params.py` ではなくこちらを使うのは、センサ層と同じ辞書を見るため
    （あちらは SI に直した属性を返すので、`sensors` セクションの生の値が取れない）。
    """
    from sensors.virtual import load_params
    return load_params(PARAMS_PATH)
