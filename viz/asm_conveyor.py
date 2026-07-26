"""搬送コンベア。供給と排出の 2 本。ボトルも乗せる。

置き方
------
どちらもスターホイールのピッチ円に接する線の上に乗る。接点はテーブル中心から
450（テーブルのピッチ円 225 とスターホイールのピッチ円 112.5 の和に、さらに 112.5）。
スターホイールの寸法は asm_transfer が正典なので、ここでは持たずに lay から読む。
ポケットのピッチをテーブルのステーション間隔に合わせた（4 ポケット・ピッチ円
半径 112.5）ので、接点は以前より 65 外へ出た。向きは変わらない。

向きは運動学から出る。テーブルは CCW（供給 315deg -> 充填 0deg が +45deg）で、
受け渡し点で速度を合わせるとスターホイールは 2 台とも CW になる。接点での
ボトルの進む向きは、どちらの台も「そのステーションの世界角 - 90deg」。
滑車と同じで、星車は入口と出口でボトルの進む向きが反転して見えるので、
コンベアを伸ばす向きは供給だけがボトルの進む向きの逆になる。

    供給 315deg  接点 ( 318.20, -318.20)  ボトル (-0.707, -0.707) 機械へ向かう
                                           伸ばす向き (+0.707, +0.707)
    排出 225deg  接点 (-318.20, -318.20)  ボトル (-0.707, +0.707) 機外へ出る
                                           伸ばす向き (-0.707, +0.707)

排出は 270deg から 225deg へ移した。星車が大きくなり、供給と排出を 1 ステーション
（45deg）しか離さないと 2 台の星車が抱えるボトルどうしが当たるため、2 ステーション
離すのが最小になった。工程角は params から読むので、この式のままで追従する。

以前はこの 2 本とも「世界角 + 90deg」で伸ばしていた。供給は合っていたが排出が
逆を向き、排出側の三日月ガイドが排出レーンを塞ぐ・星車を出たボトルが壁に当たる・
サイドガイドが接点の近くに立たない、という 3 つの症状になって出ていた。
向きを直すと機械の幅は広がるが、レーンは通る。

2 本は接点から見て左上と右上へ V 字に開く。y 軸を挟んで鏡の関係にあり、
接点どうしが 636.4 離れて先へ行くほど広がるので、2 本の隙間は制約にならない。

高さと逃げ
----------
搬送面はテーブル上面と同じ 892。ここがずれるとボトルが段差を越えることになる。
機械寄りの端はフレームを架台天板（上面 868）の上に収まる厚みまで薄くしてある。
接点が外へ出たので枠の帯は天板の縁（半径 347.2）を 13 かわすところまで下がり、
真上に重なってはいない。それでも縁との余りは 13 しかなく、天板の上には星車の
軸受台も立つので、機械寄りの端は薄いままにして深い側枠と脚を先で始める。

枠と受け皿はさらに、天板の上に載っている受け渡し部の出っ張りを避けたところから
始める。星車軸まわりには歯車（歯先円 116.5・搬送面のすぐ下）と門型の軸受台の柱
（芯まで 136.5）があり、軸はレーンの真横 112.5 なので、実機並みの枠幅では必ず
掛かる。柱の向きは受け渡し側が決めるので、向きに依らない一つの円で見て、その外
から枠を始める。チェーンは歯車の頭より上を通るので接点まで届かせ、枠だけを下げる。

**この寸法は受け渡し側から読む。控えに落ちたら黙って進まない。** 以前
trf_bearing_foot_d が消えたとき、控えが 0 になって枠が接点まで伸び、上がってきた
歯車に食い込んだ。キーが欠けていたら警告を出す。

サイドガイドと支柱の頭は、上を通るスターホイール板の下面より低いところで
頭打ちにする。ガイド板だけでなく、板より高く突き出す支柱の頭で見る。

サイドガイドと支柱は接点まで伸ばさない。空ける相手は 3 つで、三日月ガイドの板
（弧の先に継いだ接線方向の直線レール込み）、上を通るスターホイール板の真下、
搬送面の下の歯車。三日月は受け渡し側でコンベア寄りを切り詰めてあるので、円
まるごとではなく実物の板で見る。帯の半幅は見当で置かず、parts の作りから測る。

効いているのは直線レールの端で、サイドガイドはその先を継ぐ形になる。レールの
内面とガイドの内面は 1.4 しかずれていないので、ボトルは段差を拾わない。
チェーンは接点まで届き、枠だけが下がる。

駆動ヘッド
----------
機外側の端に駆動ドラムとギヤモータを置く。ドラムはチェーンの端に接して外へ
はみ出し、側板で軸を受ける。機械寄りの端は架台天板の上で場所が無いので、
ここには置かない。ギヤモータは機械側の面に付ける。反対側へ出すと、そのぶん
機械の外形が広がる。

ボトル
------
供給側は待ち行列（空）、排出側は充填済み（満量）を数本。どちらも搬送面の上に
置き、三日月ガイドに触らない位置から並べる。ガラスと液は透過なので、群は
不透明の部品より後ろに置いてある。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parts  # noqa: E402


# 絵にするためだけの比率。params.json の値に対する倍率で書く。
PROPORTION = {
    # --- コンベア本体 ---
    # 接点から機外側の先端までの長さ / テーブル外径。機外へ十分に抜ける長さ。
    "length_x_plate_d": 1.70,
    # チェーン幅 / ボトル外径。
    "belt_w_x_bottle_d": 1.47,
    # 側枠 1 枚の幅 / ボトル外径。フレーム幅 = チェーン幅 + これの 2 枚ぶん。
    # 2 本が V 字に開いて枠どうしは 524.7 空くので、実機並みの太さで置ける。
    # 幅を決めているのは 2 本の隙間ではなく、星車の軸受台との逃げのほう。
    "rail_w_x_bottle_d": 0.59,
    # チェーンの厚み / テーブル板厚。
    "belt_t_x_plate_t": 1.0,
    # 機械寄りの端のフレームを、架台天板の上にどれだけ浮かせるか / テーブル板厚。
    "pan_gap_x_plate_t": 0.25,
    # サイドガイド内幅 / ボトル外径。ボトルが通る逃げを見た幅。
    "guide_inner_w_x_bottle_d": 1.10,
    # サイドガイド板厚 / ボトル外径。
    "guide_t_x_bottle_d": 0.075,
    # サイドガイド高さ / ボトル全高。板より高く出る支柱の頭で、上を通る
    # スターホイール板の下面に当たらないところまで。
    "guide_h_x_bottle_h": 0.28,
    # ガイド支柱の丸棒 径 / ボトル外径。
    "post_d_x_bottle_d": 0.16,
    "posts_per_line": 4,
    # サイドガイドと三日月ガイドの間に見る逃げ / ボトル半径。
    # これを足した外半径の内側にはガイドも支柱もボトルも入れない。
    "guide_start_clear_x_bottle_r": 0.15,
    # 深い側枠を始める位置 / コンベア長さ。架台天板の外まで下げる。
    "beam_start_x_length": 0.20,
    # 深い側枠の高さ / ボトル全高。
    "beam_h_x_bottle_h": 0.46,
    # 深い側枠の板厚 / 側枠幅。側枠の中に納めて面が重ならないようにする。
    "beam_w_x_rail_w": 0.85,
    # 脚の角パイプ一辺 / 架台の角パイプ一辺。
    "leg_x_frame_pipe": 0.62,
    # 脚を立てる位置 / コンベア長さ。
    "leg_at_x_length": (0.42, 0.88),
    # 脚の下の丸座 半径・高さ / 脚の一辺。
    "foot_r_x_leg": 0.55,
    "foot_h_x_leg": 0.70,
    # 脚どうしを繋ぐ桁の位置 / 脚の高さ。
    "tie_at_x_leg_h": 0.22,

    # --- 駆動ヘッド ---
    # ドラム半径 / ボトル半径。チェーンの端に接して機外へはみ出す。
    "drum_r_x_bottle_r": 0.90,
    # ドラム幅 / チェーン幅。
    "drum_w_x_belt_w": 1.0,
    # ドラム軸の半径 / ドラム半径。
    "drum_shaft_r_x_drum_r": 0.22,
    # ドラムを受ける側板の板厚 / テーブル板厚と、側板をチェーンの外へ出す量。
    "drum_side_t_x_plate_t": 1.0,
    # ギヤモータ各部 / ボトル外径。
    "motor_d_x_bottle_d": 0.56,
    "motor_len_x_bottle_d": 1.05,
    "gearhead_size_x_bottle_d": 0.72,
    "gearhead_len_x_bottle_d": 0.50,
    "motor_terminal_x_bottle_d": (0.40, 0.28, 0.28),
    "motor_shaft_d_x_plate_t": 1.4,
    "motor_shaft_len_x_bottle_r": 0.30,
    "motor_fins": 5,

    # --- 乗せるボトル ---
    "infeed_queue": 5,          # 待ち行列の本数（空）
    "discharge_bottles": 2,     # 排出側の本数（満量）
    # 待ち行列の間隔 / ボトル外径。詰めて並べる。
    "queue_pitch_x_bottle_d": 1.06,
    # 排出側の間隔 / ボトル外径。星車が 1 ポケットずつ送り出した間隔。
    "discharge_pitch_x_bottle_d": 1.90,
    # 搬送面にどれだけ沈めるか / チェーン厚み。面どうしが同じ高さだと縞になる。
    "bottle_sink_x_belt_t": 0.06,
    # 稜線の輪の太さ / ボトルの肉厚。
    "edge_thickness_x_wall": 0.85,
    # 液面の縁の太さ / 液の半径。
    "rim_thickness_x_liquid_r": 0.045,

    # --- asm_transfer が居ないときの控え。ここだけに置く ---
    # スターホイールのピッチ円半径 / テーブルピッチ円半径 -> 112.5。
    # ポケットのピッチをテーブルのステーション間隔 176.7 に合わせた値。
    "star_pitch_r_x_pitch_r": 0.5,
    # スターホイール外半径 / そのピッチ円半径 -> 117.5（外径 phi235）。
    "star_outer_r_x_star_pitch_r": 1.0444444,
    # スターホイール板厚 / テーブル板厚 -> 15。
    "star_t_x_plate_t": 1.875,
    # スターホイール板の下面を、テーブル上面からどれだけ上げるか / ボトル胴高さ。
    "star_lift_x_body_h": 0.466667,
    # 三日月ガイド外半径 = スターホイールのピッチ円 + ボトル半径のこの倍。
    "crescent_ro_x_bottle_r": 1.35,
    # 歯先円半径 / スターホイールのピッチ円半径 -> 116.5。歯車はピッチ円が
    # 星車と同じで、歯先はそこからモジュール 1 つぶん外へ出る。
    "gear_tip_r_x_star_pitch_r": 1.0356,
    # 歯車の上面をテーブル上面からどれだけ下げるか / テーブル板厚 -> 10。
    "gear_top_gap_x_plate_t": 1.25,
    # 門型の軸受台の柱までの半径 / スターホイールのピッチ円半径 -> 148.5。
    "ped_r_x_star_pitch_r": 1.32,
}

# 材質。接頭辞は cnv_。ボトルと液だけは scene 側の材質をそのまま使う。
MATERIALS = {
    # チェーンは暗いつや消し。搬送面がボトルの下で沈んで見えるようにする。
    "cnv_belt":  dict(color="#33383e", pbr=True, metallic=0.25, roughness=0.88),
    # フレームは架台と同系。少しだけ明度をずらして境目を残す。
    "cnv_frame": dict(color="#767e87", pbr=True, metallic=0.55, roughness=0.64),
    # サイドガイドは白い樹脂。透明にしない。
    "cnv_guide": dict(color="#eceff2", pbr=True, metallic=0.0, roughness=0.45),
    # 支柱とクランプ、駆動ドラムは磨いた金属。
    "cnv_post":  dict(color="#c6ced5", pbr=True, metallic=0.72, roughness=0.32),
    # ギヤモータは塗装。金属より暗く落として、フレームと分けて見せる。
    "cnv_motor": dict(color="#5b636c", pbr=True, metallic=0.30, roughness=0.70),
}

# 描く順。MATERIALS に足したものはすべてここに載せる。載せ忘れると絵に出ない。
# 後ろの 4 つは搬送中のボトル。透過なので不透明の部品より後ろへ置く。
DRAW_ORDER = (
    ("cnv_frame", "cnv_frame"),
    ("cnv_belt", "cnv_belt"),
    ("cnv_guide", "cnv_guide"),
    ("cnv_post", "cnv_post"),
    ("cnv_motor", "cnv_motor"),
    ("cnv_rim", "rim"),
    ("cnv_liquid", "liquid"),
    ("cnv_glass", "glass"),
    ("cnv_bottle", "bottle"),
)

ROTATING = False


# --------------------------------------------------------------------------
# 受け渡しの相手（asm_transfer）から読む寸法
# --------------------------------------------------------------------------
def _star(lay: dict) -> dict:
    """スターホイールと三日月ガイドの寸法。asm_transfer が置いた値が正典。

    まだ居ないときだけ PROPORTION の控えで見当をつける。控えはここだけに置く。
    """
    p = PROPORTION
    pcd_r = lay.get("trf_star_pcd_r", lay["pitch_r"] * p["star_pitch_r_x_pitch_r"])
    return {
        "pcd_r": float(pcd_r),
        "center_r": float(lay.get("trf_center_r", lay["pitch_r"] + pcd_r)),
        "outer_r": float(lay.get("trf_star_outer_r",
                                 pcd_r * p["star_outer_r_x_star_pitch_r"])),
        "t": float(lay.get("trf_star_t", lay["plate_t"] * p["star_t_x_plate_t"])),
        "base_z": float(lay.get("trf_star_base_z",
                                lay["table_top"]
                                + lay["body_h"] * p["star_lift_x_body_h"])),
        "crescent_ro": float(lay.get("trf_crescent_ro",
                                     pcd_r + lay["bottle_r"]
                                     * p["crescent_ro_x_bottle_r"])),
        # 歯先円。歯車は天板の上に載っていて、搬送面のすぐ下を占める。
        "gear_r": float(lay["trf_gear_tip_d"]) / 2.0 if "trf_gear_tip_d" in lay
        else pcd_r * p["gear_tip_r_x_star_pitch_r"],
        "gear_z": (float(lay.get("trf_gear_base_z", lay["deck_top"])),
                   float(lay.get("trf_gear_top_z",
                                 lay["table_top"] - lay["plate_t"]
                                 * p["gear_top_gap_x_plate_t"]))),
        # 門型の軸受台。柱の向きは受け渡し側が決めるので、向きに依らない
        # 円で見る。柱の芯までの半径に丸棒の太さを足した値。
        "ped_r": (float(lay["trf_ped_at_r"]) + float(lay["trf_ped_post_r"])
                  if "trf_ped_at_r" in lay and "trf_ped_post_r" in lay
                  else pcd_r * p["ped_r_x_star_pitch_r"]),
    }


def _star_centers(lay: dict) -> tuple:
    """スターホイール 2 台の中心。"""
    center_r = _star(lay)["center_r"]
    out = []
    for deg in (lay["infeed_deg"], lay["discharge_deg"]):
        th = np.radians(float(deg))
        out.append((float(center_r * np.cos(th)), float(center_r * np.sin(th))))
    return tuple(out)


def _crescent_centers(lay: dict) -> tuple:
    """三日月ガイドの中心。実物の弧があればそこから拾う。"""
    arcs = lay.get("trf_crescent_arcs")
    if arcs:
        centers = {(round(float(c[0]), 3), round(float(c[1]), 3)) for c, _, _ in arcs}
        return tuple(sorted(centers))
    return _star_centers(lay)


def _crescent_plate(lay: dict):
    """三日月ガイドの板そのものを点で拾う。asm_transfer が居ないときは None。

    弧は受け渡し側でコンベア寄りを切り詰めてあるので、円まるごとで見るより
    ずっと手前まで空く。逃げを取る相手は円ではなく、この板。弧の先に継いだ
    接線方向の直線レールも同じ板の続きなので、ここに混ぜる。
    """
    arcs = lay.get("trf_crescent_arcs")
    ri = lay.get("trf_crescent_ri")
    ro = lay.get("trf_crescent_ro")
    if not arcs or ri is None or ro is None:
        return None
    out = []
    for (cx, cy), a0, a1 in arcs:
        t = np.radians(np.linspace(float(a0), float(a1), 181))
        for rr in (float(ri), float(ro)):
            out.append(np.column_stack([cx + rr * np.cos(t), cy + rr * np.sin(t)]))

    # 直線レール。芯の並びと、板厚ぶん左右に振った 2 面を拾う。
    half_t = (float(ro) - float(ri)) / 2.0
    for lead in lay.get("trf_crescent_leads", ()):
        cx, cy = (float(v) for v in lead["xy"])
        th = np.radians(float(lead["head_deg"]))
        d = np.array([np.cos(th), np.sin(th)])
        n = np.array([-d[1], d[0]])
        s = np.linspace(-0.5, 0.5, 61)[:, None] * float(lead["length"])
        base = np.array([cx, cy]) + s * d
        for off in (-half_t, 0.0, half_t):
            out.append(base + off * n)
    return np.vstack(out)


def _clear_along(line: dict, centers, reach: float, half_w: float,
                 round_object=False) -> float:
    """接点から数えて、半径 reach の丸い邪魔物に触らずに済む最小の距離。

    半幅 half_w のものを線の上に置いたときの値。round_object=True なら
    半径 half_w の丸物として見る（ボトル）。False なら幅 half_w の帯として
    見る（ガイド・支柱・枠）。
    """
    d = np.asarray(line["direction"], dtype=float)
    n = np.array([-d[1], d[0]])                        # 進む向きの左手
    c0 = np.asarray(line["contact"], dtype=float)

    start = 0.0
    for cx, cy in centers:
        rel = np.array([cx, cy]) - c0
        along = float(rel @ d)                         # 接点から前後の位置
        lat = abs(float(rel @ n))                      # 横のすれ違い量
        if round_object:
            r = reach + half_w                         # 中心どうしの距離で見る
        else:
            r, lat = reach, lat - half_w               # 帯の角で見る
        if lat >= r:                                   # 横だけで離れている
            continue
        start = max(start, along + np.sqrt(r ** 2 - max(lat, 0.0) ** 2))
    return max(start, 0.0)


def _band_gap(line: dict, pts, x0: float, x1: float, half_w: float):
    """線の上の帯（接点から x0〜x1、半幅 half_w）と点との水平距離。

    帯の外なら正、中に入っていれば食い込み量を負で返す。相手の部品との
    最短距離を数で押さえるために使う。
    """
    d = np.asarray(line["direction"], dtype=float)
    n = np.array([-d[1], d[0]])
    rel = (np.atleast_2d(np.asarray(pts, dtype=float))
           - np.asarray(line["contact"], dtype=float))
    a = rel @ d
    b = rel @ n
    dx = np.maximum(np.maximum(x0 - a, a - x1), 0.0)
    dy = np.maximum(np.abs(b) - half_w, 0.0)
    bite = -np.minimum(np.minimum(a - x0, x1 - a), half_w - np.abs(b))
    return np.where((dx > 0.0) | (dy > 0.0), np.hypot(dx, dy), bite)


# --------------------------------------------------------------------------
# 2 本の線
# --------------------------------------------------------------------------
def _lines(lay: dict, length: float) -> tuple:
    """2 本ぶんの「接点・向き・原点」を作る。

    parts.conveyor は長さの中央を原点にして x 方向へ伸びる形で出るので、
    接点から伸ばす向きへ長さの半分だけ進んだところが原点になる。
    """
    star = _star(lay)
    contact_r = star["center_r"] + star["pcd_r"]

    out = []
    for name, station_deg, outbound in (("infeed", lay["infeed_deg"], False),
                                        ("discharge", lay["discharge_deg"], True)):
        th = np.radians(float(station_deg))
        contact = (float(contact_r * np.cos(th)), float(contact_r * np.sin(th)))
        # 星車は 2 台とも CW。接点でのボトルの進む向きは半径方向から -90deg。
        travel_deg = (float(station_deg) - 90.0) % 360.0
        # 供給はボトルが機械へ向かうので、コンベアはその逆へ伸ばす。
        heading_deg = travel_deg if outbound else (travel_deg + 180.0) % 360.0
        ph = np.radians(heading_deg)
        direction = (float(np.cos(ph)), float(np.sin(ph)))
        tv = np.radians(travel_deg)
        origin = (contact[0] + direction[0] * length / 2.0,
                  contact[1] + direction[1] * length / 2.0)
        # ギヤモータを付ける面。テーブル中心を向いている側に付ける。
        normal = (-direction[1], direction[0])
        motor_side = 1.0 if (-contact[0] * normal[0]
                             - contact[1] * normal[1]) > 0.0 else -1.0
        out.append(dict(name=name, station_deg=float(station_deg),
                        contact=contact, outbound=outbound,
                        travel_deg=travel_deg,
                        travel=(float(np.cos(tv)), float(np.sin(tv))),
                        heading_deg=heading_deg,
                        direction=direction, origin=origin,
                        motor_side=motor_side))
    return tuple(out)


def _queue(line: dict, count: int, pitch: float, start: float) -> tuple:
    """線の上に並べるボトルの中心。接点から機外側へ数える。"""
    d = line["direction"]
    c0 = line["contact"]
    out = []
    for i in range(max(int(count), 0)):
        a = start + pitch * i
        out.append((float(c0[0] + d[0] * a), float(c0[1] + d[1] * a), float(a)))
    return tuple(out)


def layout(params: dict, lay: dict) -> dict:
    """コンベア 2 本の絶対座標と寸法。キーは cnv_ 接頭辞つき。"""
    p = PROPORTION
    bottle_r = lay["bottle_r"]
    bottle_d = 2.0 * bottle_r

    belt_t = lay["plate_t"] * p["belt_t_x_plate_t"]
    belt_w = bottle_d * p["belt_w_x_bottle_d"]
    rail_w = bottle_d * p["rail_w_x_bottle_d"]
    frame_w = belt_w + 2.0 * rail_w
    # parts.conveyor の側枠は幅 8 を下回らない。深い側枠と脚をその真下に置くので、
    # 向こうが使う値をここでも同じ式で出しておく。
    rail_w = max((frame_w - belt_w) / 2.0, 8.0)

    belt_top = lay["table_top"]            # 段差を作らない。テーブル上面と同じ
    frame_top = belt_top - belt_t
    # 機械寄りの端は架台天板の上を通る。天板の上面との間に少し隙間を残す。
    pan_h = frame_top - lay["deck_top"] - lay["plate_t"] * p["pan_gap_x_plate_t"]

    # 接点から機外側の先端まで。先端の駆動ドラムのぶんだけチェーンと枠を短くし、
    # ドラムをその外に接して置く。枠の中にドラムを入れると下の繋ぎ板と食い合う。
    span = lay["plate_d"] * p["length_x_plate_d"]
    drum_r = bottle_r * p["drum_r_x_bottle_r"]
    length = span - 2.0 * drum_r

    beam_h = lay["bottle_h"] * p["beam_h_x_bottle_h"]
    beam_w = rail_w * p["beam_w_x_rail_w"]
    rail_at = (belt_w + rail_w) / 2.0
    leg = lay["frame_pipe"] * p["leg_x_frame_pipe"]
    foot_h = leg * p["foot_h_x_leg"]
    guide_t = bottle_d * p["guide_t_x_bottle_d"]
    post_d = bottle_d * p["post_d_x_bottle_d"]
    guide_inner_w = bottle_d * p["guide_inner_w_x_bottle_d"]
    guide_h = lay["bottle_h"] * p["guide_h_x_bottle_h"]
    posts = int(p["posts_per_line"])

    # ガイドまわりの実際の半幅。支柱もクランプもガイド板の外へ出るので、
    # 見当で置かずに parts の作りから測る（長さは幅に効かない）。
    probe = parts.conveyor(length=1.0, belt_width=belt_w, frame_width=frame_w,
                           top_z=belt_top, belt_thickness=belt_t,
                           frame_height=pan_h, guide_inner_w=guide_inner_w,
                           guide_h=guide_h, guide_t=guide_t, post_d=post_d,
                           posts=posts)
    w_guide = max(max(abs(probe[k].bounds[2]), abs(probe[k].bounds[3]))
                  for k in ("guides", "guide_posts"))

    # サイドガイド・支柱・ボトルを、接点からどれだけ先で始めるか。避ける相手は
    # 三日月ガイドの板（直線レール込み）と、上を通るスターホイール板の真下、
    # それに搬送面の下の歯車。
    clear = bottle_r * p["guide_start_clear_x_bottle_r"]
    star = _star(lay)
    # スターホイール板の縁と支柱の頭は 3 しか離れていない。平面でも重ねない。
    star_reach = star["outer_r"] + clear
    # 三日月は実物の板で見る。まだ居なければ円まるごとで保守側に見当をつける。
    plate = _crescent_plate(lay)
    cre_reach = star["crescent_ro"] + clear
    cre_at = _crescent_centers(lay)
    # 枠と受け皿は搬送面の下にあるので、天板の上に載っている受け渡し部の
    # 出っ張り（歯車と門型の軸受台の柱）を丸ごと避ける。柱の向きは向こうが
    # 決めるので、向きに依らない一つの円で見る。チェーンは歯車の頭より上を
    # 通るので接点まで届かせ、枠だけを下げる。
    hub_reach = max(star["gear_r"], star["ped_r"]) + clear
    # 受け渡し側のキーが消えていたら黙って控えに落ちない。以前 trf_bearing_foot_d
    # が無くなったとき、控えが 0 になって枠が接点まで伸び、歯車に食い込んだ。
    missing = [k for k in ("trf_gear_tip_d", "trf_ped_at_r", "trf_ped_post_r",
                           "trf_star_outer_r", "trf_center_r")
               if k not in lay]
    if missing:
        print(f"[cnv] 受け渡し側の寸法が無いので控えで置く: {', '.join(missing)}",
              file=sys.stderr)
    star_at = _star_centers(lay)
    lines = []
    for line in _lines(lay, length):
        if plate is None:
            guide_start = _clear_along(line, cre_at, cre_reach, w_guide)
            lead = _clear_along(line, cre_at, cre_reach, bottle_r,
                                round_object=True)
        else:
            guide_start = _clear_along(line, plate, clear, w_guide)
            lead = _clear_along(line, plate, clear, bottle_r, round_object=True)
        # 三日月を切り詰めても、スターホイール板と歯車の真下までは詰められない
        guide_start = max(guide_start,
                          _clear_along(line, star_at, star_reach, w_guide),
                          _clear_along(line, star_at, star["gear_r"] + clear,
                                       w_guide))
        lead = max(lead, _clear_along(line, star_at, star_reach, bottle_r,
                                      round_object=True))
        frame_start = _clear_along(line, star_at, hub_reach, frame_w / 2.0)
        if line["outbound"]:
            count, pitch = (int(p["discharge_bottles"]),
                            bottle_d * p["discharge_pitch_x_bottle_d"])
        else:
            count, pitch = (int(p["infeed_queue"]),
                            bottle_d * p["queue_pitch_x_bottle_d"])
        lines.append(dict(line, guide_start=guide_start,
                          frame_start=frame_start, bottle_lead=lead,
                          bottles=_queue(line, count, pitch, lead)))

    # 排出側は満量、供給側は空。液深は params の充填量から出す（h = V / (pi R^2)）。
    liquid_r = lay["liquid_r"]
    full_level = float(np.clip(
        float(params["fill"]["target_volume_mL"]) * 1000.0
        / (np.pi * liquid_r * liquid_r), 0.0, lay["body_h"]))

    return {
        "cnv_lines": tuple(lines),
        "cnv_span": span,
        "cnv_length": length,
        "cnv_belt_top": belt_top,
        "cnv_belt_t": belt_t,
        "cnv_belt_w": belt_w,
        "cnv_frame_w": frame_w,
        "cnv_frame_top": frame_top,
        "cnv_rail_w": rail_w,
        "cnv_rail_at": rail_at,                        # 側枠の中心の横位置
        "cnv_pan_h": pan_h,
        "cnv_guide_inner_w": guide_inner_w,
        "cnv_guide_t": guide_t,
        "cnv_guide_h": guide_h,
        "cnv_guide_half_w": w_guide,                   # ガイドまわりの実際の半幅
        "cnv_post_d": post_d,
        "cnv_posts": posts,
        "cnv_hub_reach": hub_reach,                    # 枠が避ける丸い占有域
        "cnv_beam_start": length * p["beam_start_x_length"],
        "cnv_beam_h": beam_h,
        "cnv_beam_w": beam_w,
        "cnv_beam_base": frame_top - beam_h,
        "cnv_leg": leg,
        "cnv_leg_at": tuple(float(v) for v in p["leg_at_x_length"]),
        "cnv_foot_r": leg * p["foot_r_x_leg"],
        "cnv_foot_h": foot_h,
        "cnv_tie_z": foot_h + (frame_top - beam_h - foot_h) * p["tie_at_x_leg_h"],
        # 駆動ヘッド
        "cnv_drum_r": drum_r,
        "cnv_drum_w": belt_w * p["drum_w_x_belt_w"],
        "cnv_drum_z": belt_top - drum_r,               # ドラム軸の高さ
        "cnv_drum_shaft_r": drum_r * p["drum_shaft_r_x_drum_r"],
        "cnv_drum_side_t": lay["plate_t"] * p["drum_side_t_x_plate_t"],
        "cnv_motor_at": rail_at + beam_w / 2.0,        # ギヤモータの取付面
        "cnv_motor_shaft_len": bottle_r * p["motor_shaft_len_x_bottle_r"],
        # 乗せるボトル
        "cnv_bottle_base": belt_top - belt_t * p["bottle_sink_x_belt_t"],
        "cnv_full_level": full_level,
    }


# --------------------------------------------------------------------------
# 組み立て
# --------------------------------------------------------------------------
def _stand(lay: dict, matrix) -> list:
    """深い側枠と、床まで届く脚。コンベアの局所座標（x が流れる向き）で組む。

    parts.conveyor のフレームは架台天板を避けるために薄くしてあるので、
    そのままだと宙に浮いた板に見える。天板の外へ出たところから深い側枠を回し、
    そこから床まで脚を下ろす。
    """
    ln = lay["cnv_length"]
    x0 = -ln / 2.0 + lay["cnv_beam_start"]             # 深い側枠を始める位置
    x1 = ln / 2.0
    beam_len = x1 - x0
    y_rail = lay["cnv_rail_at"]
    beam_base = lay["cnv_beam_base"]
    leg = lay["cnv_leg"]
    leg_base = lay["cnv_foot_h"]

    out = []
    for sy in (-1, 1):
        out.append(parts.box((beam_len, lay["cnv_beam_w"], lay["cnv_beam_h"]),
                             center=((x0 + x1) / 2.0, sy * y_rail,
                                     beam_base + lay["cnv_beam_h"] / 2.0)))

    for frac in lay["cnv_leg_at"]:
        x = -ln / 2.0 + ln * float(frac)
        for sy in (-1, 1):
            out.append(parts.box((leg, leg, beam_base - leg_base),
                                 center=(x, sy * y_rail,
                                         (beam_base + leg_base) / 2.0)))
            out.append(parts.cylinder(lay["cnv_foot_r"], leg_base, base_z=0.0,
                                      resolution=parts.RES_COARSE,
                                      matrix=parts.transform_matrix(
                                          translate=(x, sy * y_rail, 0.0))))
        # 脚どうしを繋ぐ桁。1 本脚に見えないようにする
        out.append(parts.box((leg * 0.7, 2.0 * y_rail + leg, leg * 0.7),
                             center=(x, 0.0, lay["cnv_tie_z"])))

    return [parts.place(m, matrix) for m in out]


def _drive_head(lay: dict, line: dict, matrix) -> dict:
    """機外側の端の駆動ヘッド。ドラム・軸・側板・ギヤモータ。

    局所座標（x が流れる向き、原点はチェーンの中央）で組む。ドラムはチェーンの
    端に接して外へはみ出し、その外周の頭が搬送面と揃う。
    """
    p = PROPORTION
    bottle_d = 2.0 * lay["bottle_r"]
    r = lay["cnv_drum_r"]
    x_end = lay["cnv_length"] / 2.0                    # チェーンと枠の端
    x_drum = x_end + r
    z = lay["cnv_drum_z"]
    side = float(line["motor_side"])
    mount_y = lay["cnv_motor_at"]
    side_t = lay["cnv_drum_side_t"]
    y_side = lay["cnv_drum_w"] / 2.0 + side_t / 2.0

    out = {"cnv_post": [], "cnv_frame": [], "cnv_motor": []}

    # ドラムと軸
    out["cnv_post"].append(
        parts.horizontal_cylinder(r, lay["cnv_drum_w"], axis="y",
                                  center=(x_drum, 0.0, z)))
    shaft_len = 2.0 * (mount_y - lay["cnv_motor_shaft_len"])
    out["cnv_post"].append(
        parts.horizontal_cylinder(lay["cnv_drum_shaft_r"], shaft_len, axis="y",
                                  center=(x_drum, 0.0, z),
                                  resolution=parts.RES_COARSE))

    # 軸を受ける側板。枠の端に掛けて、ドラムを抱く
    for sy in (-1.0, 1.0):
        out["cnv_frame"].append(
            parts.box((2.0 * r + side_t, side_t, 2.0 * r),
                      center=(x_end + r - side_t / 2.0, sy * y_side, z)))
    # モータ側は、側板から取付面までを塞ぐブラケット
    out["cnv_frame"].append(
        parts.box((2.0 * r, mount_y - (y_side + side_t / 2.0), 1.8 * r),
                  center=(x_drum, side * (mount_y + y_side + side_t / 2.0) / 2.0, z)))

    # ギヤモータ。出力軸が取付面からドラム軸へ向き、胴は機外側へ出る。
    # parts.gearmotor は軸の正方向へ出力軸が出るので、+y の面に付けるときは
    # 原点を反対側に取って z まわりに半回転させる（鏡にすると面の裏表が返る）。
    if side > 0.0:
        motor_origin = (-x_drum, -mount_y, z)
        extra = parts.transform_matrix(rot_z_deg=180.0)
    else:
        motor_origin = (x_drum, -mount_y, z)
        extra = None
    motor = parts.gearmotor(
        motor_d=bottle_d * p["motor_d_x_bottle_d"],
        motor_len=bottle_d * p["motor_len_x_bottle_d"],
        gearhead_size=bottle_d * p["gearhead_size_x_bottle_d"],
        gearhead_len=bottle_d * p["gearhead_len_x_bottle_d"],
        shaft_d=lay["plate_t"] * p["motor_shaft_d_x_plate_t"],
        shaft_len=lay["cnv_motor_shaft_len"],
        fins=int(p["motor_fins"]),
        terminal_size=tuple(bottle_d * v for v in p["motor_terminal_x_bottle_d"]),
        axis="y", origin=motor_origin, matrix=extra)
    # 取付ベース板は床まで届かないところに浮くので使わない
    for key in ("motor", "fins", "gearhead", "shaft", "terminal_box"):
        out["cnv_motor"].append(motor[key])

    return {k: [parts.place(m, matrix) for m in v] for k, v in out.items()}


def _bottles(params: dict, lay: dict, line: dict) -> dict:
    """線の上のボトル。供給側は空、排出側は満量。"""
    p = PROPORTION
    b = params["bottle"]
    base_z = lay["cnv_bottle_base"]
    level = lay["cnv_full_level"] if line["outbound"] else 0.0
    liquid_base = base_z + b["wall_thickness_mm"]

    out = {"cnv_bottle": [], "cnv_glass": [], "cnv_liquid": [], "cnv_rim": []}
    for x, y, _ in line["bottles"]:
        mat = parts.transform_matrix(translate=(x, y, 0.0))
        out["cnv_bottle"].append(
            parts.bottle(inner_diameter=b["inner_diameter_mm"],
                         body_height=b["body_height_mm"],
                         shoulder_height=b["shoulder_height_mm"],
                         neck_diameter=b["neck_diameter_mm"],
                         neck_height=b["neck_height_mm"],
                         wall_thickness=b["wall_thickness_mm"],
                         base_z=base_z, matrix=mat))
        out["cnv_glass"].append(
            parts.bottle_edges(inner_diameter=b["inner_diameter_mm"],
                               body_height=b["body_height_mm"],
                               shoulder_height=b["shoulder_height_mm"],
                               neck_diameter=b["neck_diameter_mm"],
                               neck_height=b["neck_height_mm"],
                               wall_thickness=b["wall_thickness_mm"],
                               base_z=base_z,
                               thickness=b["wall_thickness_mm"]
                               * p["edge_thickness_x_wall"],
                               matrix=mat))
        if level <= 0.0:
            continue
        out["cnv_liquid"].append(
            parts.liquid(lay["liquid_r"], level, base_z=liquid_base, matrix=mat))
        out["cnv_rim"].append(
            parts.liquid_rim(lay["liquid_r"], level, base_z=liquid_base,
                             thickness=lay["liquid_r"]
                             * p["rim_thickness_x_liquid_r"],
                             matrix=mat))
    return out


def _conveyor(lay: dict, length: float, matrix):
    """parts.conveyor を lay の寸法で呼ぶ。長さだけ差し替えられるようにしておく。"""
    return parts.conveyor(length=length,
                          belt_width=lay["cnv_belt_w"],
                          frame_width=lay["cnv_frame_w"],
                          top_z=lay["cnv_belt_top"],
                          belt_thickness=lay["cnv_belt_t"],
                          frame_height=lay["cnv_pan_h"],
                          guide_inner_w=lay["cnv_guide_inner_w"],
                          guide_h=lay["cnv_guide_h"],
                          guide_t=lay["cnv_guide_t"],
                          post_d=lay["cnv_post_d"],
                          posts=lay["cnv_posts"],
                          matrix=matrix)


def build(params: dict, lay: dict) -> dict:
    """メッシュ群の名前 -> PolyData。寸法は lay と PROPORTION からしか読まない。"""
    groups: dict = {key: [] for key, _ in DRAW_ORDER}
    ln = lay["cnv_length"]

    for line in lay["cnv_lines"]:
        dx, dy = line["direction"]
        matrix = parts.transform_matrix(
            translate=(line["origin"][0], line["origin"][1], 0.0),
            rot_z_deg=line["heading_deg"])

        def shifted(start, line=line, dx=dx, dy=dy):
            """接点寄りを start だけ空けた置き方。短くした本体を先へずらす。"""
            return parts.transform_matrix(
                translate=(line["origin"][0] + dx * start / 2.0,
                           line["origin"][1] + dy * start / 2.0, 0.0),
                rot_z_deg=line["heading_deg"])

        # チェーンだけは接点まで届かせる。枠・サイドガイド・支柱は、それぞれ
        # 避けたいものが違うので、空ける長さも別々に取る。
        full = _conveyor(lay, ln, matrix)
        framed = _conveyor(lay, ln - line["frame_start"],
                           shifted(line["frame_start"]))
        guided = _conveyor(lay, ln - line["guide_start"],
                           shifted(line["guide_start"]))

        groups["cnv_belt"].append(full["belt"])
        groups["cnv_frame"].append(framed["frame"])
        groups["cnv_guide"].append(guided["guides"])
        groups["cnv_post"].append(guided["guide_posts"])
        groups["cnv_frame"] += _stand(lay, matrix)
        for key, meshes in _drive_head(lay, line, matrix).items():
            groups[key] += meshes
        for key, meshes in _bottles(params, lay, line).items():
            groups[key] += meshes

    out = {k: parts.merge(v) for k, v in groups.items() if v}

    # ガイドまわりの頭がスターホイール板の下に収まっているか。ガイド板より
    # 支柱の頭のほうが高いので、板だけを見ていると見落とす。
    star_base = _star(lay)["base_z"]
    heads = [float(out[k].bounds[5]) for k in ("cnv_guide", "cnv_post") if k in out]
    top = max(heads) if heads else 0.0
    if top >= star_base:
        print(f"[cnv] ガイドまわりの頭 {top:.1f} がスターホイール板の下面 "
              f"{star_base:.1f} に当たる", file=sys.stderr)
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    import pyvista as pv

    import cameras
    import scene

    ap = argparse.ArgumentParser(description="コンベアの据わりを目で確かめる")
    ap.add_argument("--out", default="/tmp/cnv", help="PNG の置き場")
    ap.add_argument("--size", type=int, nargs=2, default=(800, 600),
                    metavar=("W", "H"))
    args = ap.parse_args()

    params = scene.load_params()
    lay = scene.derive_layout(params)

    # 静止側と回る側はどちらも "steel" を返す。辞書の update でまとめると
    # 片方が丸ごと消える（固定ノズル・支柱とアーム・旋回軸受リング・中心柱が
    # 確認画像から落ちていたのはこれ）。scene の合成ヘルパを通す。
    meshes = scene.merge_groups(scene.build_static(params, lay, with_floor=True),
                                scene.build_carousel(params, lay))

    # 受け渡しの相手は別担当。まだ居ないときだけ、位置関係を見るための
    # 仮の円板を置く。寸法は _star()（asm_transfer が居れば向こうの値）から取る。
    if "trf_star_base_z" not in lay:
        star = _star(lay)
        fake = []
        for deg in (lay["infeed_deg"], lay["discharge_deg"]):
            th = np.radians(deg)
            fake.append(parts.plate(2.0 * star["outer_r"], star["t"],
                                    base_z=star["base_z"],
                                    matrix=parts.transform_matrix(
                                        translate=(star["center_r"] * np.cos(th),
                                                   star["center_r"] * np.sin(th),
                                                   0.0))))
        meshes = scene.merge_groups(meshes, {"steel": parts.merge(fake)})
        print("スターホイールは仮の円板で置いた（asm_transfer 待ち）")

    mine = build(params, lay)
    meshes = scene.merge_groups(meshes, mine)
    scene.ensure_extent(lay, meshes, params)

    # --- 数値の確認 -------------------------------------------------------
    star = _star(lay)
    print("接点と向き（契約の値と突き合わせる）")
    for line in lay["cnv_lines"]:
        print(f"  {line['name']:9s} 接点 ({line['contact'][0]:8.2f},"
              f" {line['contact'][1]:8.2f})"
              f"  ボトル ({line['travel'][0]:+.3f}, {line['travel'][1]:+.3f})"
              f"  伸ばす ({line['direction'][0]:+.3f}, {line['direction'][1]:+.3f})"
              f"  原点 ({line['origin'][0]:8.2f}, {line['origin'][1]:8.2f})"
              f"  回転 {line['heading_deg']:6.1f}deg")
        print(f"            枠開始 {line['frame_start']:6.1f}"
              f" / ガイド開始 {line['guide_start']:6.1f}"
              f" / ボトル先頭 {line['bottle_lead']:6.1f}"
              f" / {len(line['bottles'])} 本"
              f" / ギヤモータ面 {line['motor_side']:+.0f}")
    print(f"接点半径 {star['center_r'] + star['pcd_r']:.2f}"
          f"（星車 中心半径 {star['center_r']:.2f} + ピッチ円 {star['pcd_r']:.2f}）")
    print(f"長さ 接点から先端 {lay['cnv_span']:.1f}"
          f"（チェーン {lay['cnv_length']:.1f} + ドラム {2 * lay['cnv_drum_r']:.1f}）"
          f" / チェーン幅 {lay['cnv_belt_w']:.1f}"
          f" / フレーム幅 {lay['cnv_frame_w']:.1f}")
    print(f"搬送面 {lay['cnv_belt_top']:.1f}（テーブル上面 {lay['table_top']:.1f}）"
          f" / 端のフレーム下面 {lay['cnv_frame_top'] - lay['cnv_pan_h']:.1f}"
          f"（架台天板上面 {lay['deck_top']:.1f}）")
    print(f"ガイド上端 {mine['cnv_guide'].bounds[5]:.1f}"
          f" / 支柱の頭 {mine['cnv_post'].bounds[5]:.1f}"
          f"（スターホイール板 下面 {star['base_z']:.1f}）")
    print(f"ドラム 半径 {lay['cnv_drum_r']:.1f} / 軸の高さ {lay['cnv_drum_z']:.1f}"
          f" / 下端 {lay['cnv_drum_z'] - lay['cnv_drum_r']:.1f}")
    print(f"ボトル 底 {lay['cnv_bottle_base']:.2f}"
          f"（搬送面 {lay['cnv_belt_top']:.1f}）"
          f" / 満量の液深 {lay['cnv_full_level']:.1f}")
    print(f"深い側枠 下面 {lay['cnv_beam_base']:.1f} / 脚は床 0 まで")

    for name, mesh in sorted(mine.items()):
        b = mesh.bounds
        print(f"  {name:10s} x[{b[0]:8.1f},{b[1]:8.1f}]"
              f" y[{b[2]:8.1f},{b[3]:8.1f}] z[{b[4]:7.1f},{b[5]:7.1f}]")

    # レーンのボトルと三日月ガイドの隙間。負なら食い込んでいる。
    pts = _crescent_plate(lay)
    if pts is not None:

        def _near(x, y):
            """三日月ガイドの板までの水平距離。"""
            return float(np.hypot(pts[:, 0] - x, pts[:, 1] - y).min())

        print("三日月ガイドとボトルの隙間（水平距離。正なら空いている）")
        for line in lay["cnv_lines"]:
            gaps = [_near(x, y) - lay["bottle_r"] for x, y, _ in line["bottles"]]
            # 置いた本数だけでなく、レーンを流れていくボトルの通り道ぜんぶで見る
            d = np.asarray(line["direction"])
            c0 = np.asarray(line["contact"])
            a = np.linspace(0.0, lay["cnv_span"], 400)
            lane = [(_near(*(c0 + t * d)) - lay["bottle_r"], t) for t in a]
            worst = min(lane)
            print(f"  {line['name']:9s} 置いたボトル 先頭 {gaps[0]:+7.1f}"
                  f" / 最小 {min(gaps):+7.1f}"
                  f"   通り道の最小 {worst[0]:+7.1f}"
                  f"（接点から {worst[1]:.0f}）")

    # --- 受け渡し側の部品との最短距離（水平。負なら食い込み）--------------
    # 帯は置いたときと同じ幅で見る。ガイドまわりの半幅は実測値。
    ln_full = lay["cnv_length"]
    w_frame = lay["cnv_frame_w"] / 2.0
    w_guide = lay["cnv_guide_half_w"]
    star_at = _star_centers(lay)
    print("受け渡し側との水平の隙間（正なら空いている）")
    for line in lay["cnv_lines"]:
        bands = {"枠": (line["frame_start"], w_frame),
                 "ガイド": (line["guide_start"], w_guide),
                 "チェーン": (0.0, lay["cnv_belt_w"] / 2.0)}
        rows = []
        # 星車板は高さが分けてあるので水平は参考値。歯車は搬送面のすぐ下なので
        # 枠と受け皿にとっては実害のある値。門型の柱は向きに依らない円で
        # 見ているため、実物より太い。枠を置くのに使った値なので枠だけ出す。
        for label, radius, keys in (
                ("星車板", star["outer_r"], tuple(bands)),
                ("歯車", star["gear_r"], tuple(bands)),
                ("門型の円", star["ped_r"], ("枠",))):
            for key in keys:
                x0, half = bands[key]
                g = float(_band_gap(line, star_at, x0, ln_full, half).min())
                rows.append((f"{label}-{key}", g - radius))
        if pts is not None:
            for key, (x0, half) in bands.items():
                g = float(_band_gap(line, pts, x0, ln_full, half).min())
                rows.append((f"三日月-{key}", g))
        floors = [s["xy"] for s in lay.get("trf_crescent_stays", ())
                  if s.get("kind") == "floor"]
        if floors:
            r_stand = float(lay.get("trf_crescent_stand_r", 0.0))
            g = float(_band_gap(line, floors, line["frame_start"],
                                ln_full, w_frame).min())
            rows.append(("三日月の床柱-枠", g - r_stand))
        print(f"  {line['name']:9s} "
              + " / ".join(f"{k} {v:+.1f}" for k, v in rows))
    gz0, gz1 = star["gear_z"]
    print(f"  （星車板の下面 {star['base_z']:.1f} と支柱の頭 "
          f"{mine['cnv_post'].bounds[5]:.1f} で高さは分けてある。"
          f"歯車 z {gz0:.1f}〜{gz1:.1f} に対しチェーン下面 "
          f"{lay['cnv_belt_top'] - lay['cnv_belt_t']:.1f}）")

    # --- 焼く -------------------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def shot(cam, path):
        plotter = pv.Plotter(off_screen=True, window_size=list(args.size))
        plotter.set_background(scene.BACKGROUND)
        plotter.set_environment_texture(scene.studio_cubemap(), is_srgb=True)
        try:
            plotter.renderer.GetEnvMapPrefiltered().SetPrefilterMaxSamples(64)
            plotter.renderer.GetEnvMapIrradiance().SetIrradianceSize(32)
        except AttributeError:
            pass
        for name, material in scene.DRAW_ORDER:
            mesh = meshes.get(name)
            if mesh is None or mesh.n_points == 0:
                continue
            plotter.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True,
                             feature_angle=35.0, **scene.MATERIAL[material])
        cameras.apply_resolved(plotter, cam)
        plotter.show(screenshot=str(path))
        plotter.close()
        print("焼いた:", path)

    # 2 本が V 字に開いて機械の外形が広がった。枠は実測の外形（extent_r）に
    # 対する倍率で決めるので、部品を外へ足しても枠から外れない。
    span = {"view_span_x_extent_d": 1.06}
    shot(cameras.resolve("top", lay, params, **span), out_dir / "cnv_top.png")
    # 斜め上。脚が床に着いているところまで入れたいので平行投影で引く。
    shot(cameras.resolve("iso", lay, params, elevation_deg=30.0,
                         azimuth_deg=325.0, parallel=True,
                         view_span_x_extent_d=1.15),
         out_dir / "cnv_iso.png")
    # 真横。搬送面がテーブル上面と揃っているかを見る。それぞれのレーンに
    # 直角な向きから見るので、方位角は伸ばす向きから 90deg 振って出す。
    for line in lay["cnv_lines"]:
        shot(cameras.resolve("top", lay, params, elevation_deg=0.0,
                             azimuth_deg=line["heading_deg"] - 90.0,
                             parallel=True, **span),
             out_dir / f"cnv_side_{line['name']}.png")
