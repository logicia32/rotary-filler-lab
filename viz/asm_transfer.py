"""受け渡し系。供給・排出のスターホイールと、それを囲む固定ガイド。

1 ポケットの送り弧長をテーブルのステーション間隔に合わせてある。ポケットの
ピッチが揃っているので、テーブルと星車を**同時に**動かせて、受け渡し点での
相対速度は 0 になる。それでも円盤どうしは噛み合わない（ポケットが同じ位置に
来るのは受け渡し点だけで、他の歯は互いを避けない）ので、平面図では 15 mm
食い込む。高さを分けるのが唯一の道で、スターホイール板はテーブル上面より
持ち上げてボトル胴の中ほどだけを掴む。足元は下のポケット板と渡し板が受けた
ままなので、上のポケットが抜けることで受け渡しが起きる。

このモジュールが置くのは 6 つ。

    スターホイール 2 台   供給 315deg / 排出 225deg。中心はピッチ円半径と
                          星車ピッチ円半径の和の位置。自分の軸で自転する
                          （テーブル軸まわりではない）
    駆動                  テーブル出力から 2:1 の平歯車で取る。歯車は架台天板の
                          上面とテーブル板の下面のあいだに置き、軸は天板を
                          貫かずに上面で止める。軸受は歯車の上に来るので、
                          歯先円の外に立てた柱から腕を渡した門型で受ける
    渡し板                テーブル板とコンベアのあいだ、ボトルの底が浮く区間を
                          埋める固定板。回るテーブル・ポケット板とは当たらない。
                          コンベア側は世界半径の円ではなく、搬送面の手前の端
                          （接点を通る、搬送の向きに直交する面）で切る。円で
                          切ると板が接点の 50deg 手前で退いて足元が空く
    三日月ガイド          ポケットからはみ出したボトルを外から押さえる固定板。
                          コンベア側は接点で切らず、接線方向の直線レールを継いで
                          コンベアのサイドガイドが立つ位置まで受けを伸ばす。
                          天板の外へ出た区間は渡し板の耳から棒を立てて受ける
                          （渡し板は天板の短柱に載っているので、受けは天板から
                          出ていることになる）。床から立てる柱は使わない
    テーブル外周ガイド    ボトル外周のすぐ外に立てる円弧板。受け渡しの位置だけ
                          切り欠く。供給と排出が 2 ステーション離れたので、
                          切り欠きは融合せず 2 つに分かれる
    ポケットのボトル      星車に抱えられて一緒に回るボトル。供給側は空・排出側は満量

供給と排出は 2 ステーション（90deg）離す。1 ステーションだと 2 台の星車が
抱えるボトルどうしが当たる（ピッチ円の隙間 33.3 に対しボトル外径 68）。
離したことで、以前は 2 台のガイドが食い合って削っていた三日月が丸ごと残る。

スターホイールはテーブルと**同時に**動く。割出し中（入力軸角 psi 0〜180deg）に
テーブルが +45deg 進むあいだ、星車は -90deg（1 ポケット）回る。停留中
（psi 180〜360deg）はどちらも止まる。速比はちょうど 2.0 で、向きは両方とも CW。
テーブル角と同じ psi の関数を使い、係数だけ -2.0 倍にして揃える。

駆動はベルトではなく歯車で取る。テーブルが CCW で星車が CW なので、平行掛けの
ベルトでは向きが合わない。歯車なら噛み合いがそのまま反転になる。中心距離 337.5
と速比 2 からピッチ半径は 225 と 112.5 に一意に決まり、これはテーブルと星車の
ボトルピッチ円そのもの。つまり「ポケットのピッチを合わせる」条件と歯車の
噛み合い条件が同じ式になっている。

歯は刻む。歯先円の和は 229 + 116.5 = 345.5 で中心距離 337.5 を 8 mm 上回るので、
片方でも平らな円板だとその 8 mm がそのまま食い込みになる。歯どうしが互いの
歯溝に入って初めて成立する（この寸法で歯どうしの最短距離は 0.77〜0.95）。位相は
「相手の歯の中心が、こちらの歯溝の中心とピッチ点で出会う」条件から解く。
歯車は星車板と同じ群に入れると、真上から見たとき星車の縁に歯が生えているように
見える（歯先円 phi233 と星車の外径 phi235 がほぼ同じで、ポケットの切り欠きから
下の歯が覗く）。別の群に分けて、鋳物の色を当てて見分けが付くようにする。

以前は「交互に動かす」としていた。あれはポケットのピッチがテーブルと違って
いたころ、送り量の食い違いを避けるための回避策で、ピッチを合わせた今は不要。
しかも交互では受け渡し点のボトルが板材に食い込んで成立しない。

寸法の根拠は viz/ASSEMBLY_CONTRACT.md と params.json の諸元。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parts  # noqa: E402


# --------------------------------------------------------------------------
# 絵にするためだけの比率。長さはすべて params.json 由来の値の倍率で書く。
# 角度だけは倍率にならないので度で持つ。
# --------------------------------------------------------------------------
PROPORTION = {
    # --- スターホイール本体 -------------------------------------------------
    # ピッチ円半径はテーブルのピッチ円半径のちょうど半分。ポケット 4 個なので
    # 1 ポケットの送り弧長 = 2pi*112.5/4 = 176.7 で、テーブルのステーション
    # 間隔（2pi*225/8 = 176.7）と一致する。ここが揃っていないと同時に回せない
    "star_pcd_r_x_pitch_r": 0.5,           # ピッチ円半径 / テーブルピッチ円半径 -> 112.5
    "star_outer_r_x_pitch_r": 0.5222222,   # 外半径 -> 117.5（外径 phi235）
    "star_pocket_r_x_bottle_r": 1.1764706,  # ポケット半径 / ボトル外半径 -> 40
    "star_t_x_plate_t": 1.875,             # 板厚 / テーブル板厚 -> 15
    "star_lift_x_body_h": 0.4666667,       # 板下面をテーブル上面から持ち上げる量 -> 70
    "star_hub_d_x_pocket_r": 1.90,         # 中心ボス径 -> 76
    "star_hub_h_x_plate_t": 3.10,          # 同 高さ -> 24.8
    "star_shaft_d_x_plate_t": 3.75,        # 軸径 -> 30
    "star_shaft_stub_x_plate_t": 0.75,     # ボスの上に出る軸の長さ -> 6
    "star_pockets": 4,                     # ポケット数

    # --- 軸受台と歯車（どちらも天板の上）-----------------------------------
    # 軸は天板を貫かない。天板の上面で止め、歯車は天板上面とテーブル板下面の
    # あいだ（16 mm）に置く。ここには脚が来ない（脚の上端は天板の下面）。
    # 軸受は歯車より上に来るので、歯車の外を跨ぐ門型の軸受台で受ける
    "bearing_d_x_shaft_d": 1.50,           # 軸受箱の外径 / 軸径 -> 45
    "bearing_bore_x_shaft_d": 1.06,        # 軸が通る穴の径 -> 31.8
    "bearing_gap_x_plate_t": 1.0,          # 軸受箱の上端と円盤下面の隙間 -> 8
    # 門型の柱。歯車の歯先円の外に立て、天板の上で受ける
    "ped_post_r_x_shaft_d": 0.40,          # 柱の半径 -> 12
    "ped_post_gap_x_plate_t": 1.0,         # 柱と歯先円の隙間 -> 8
    "ped_foot_d_x_post_r": 2.60,           # 柱の据付座の径 -> 62.4
    "ped_foot_t_x_plate_t": 0.60,          # 同 板厚 -> 4.8
    "ped_arm_t_x_plate_t": 1.50,           # 歯車を跨ぐ腕の厚み -> 12
    "ped_arm_w_x_post_r": 2.00,            # 同 幅 -> 48
    # 腕の下面が逃げる相手は歯車の上面ではなく、その上に出ている締めボスの頭。
    # 上面 882 で見ていたので、ボス頭 886 に固定側が 2 mm 埋まっていた
    "ped_arm_gap_x_plate_t": 0.25,         # 腕の下面と締めボス頭の隙間 -> 2
    "ped_at_deg": 110.0,                   # 柱を立てる局所角（ボトルが通らない側）
    "ped_deck_margin_x_post_r": 0.50,      # 柱を天板の縁から戻す量 / 柱半径
    "ped_plate_gap_x_plate_t": 1.0,        # 回るテーブル板の縁からの逃げ -> 8
    # 星車軸の平歯車。ピッチ円は星車のボトルピッチ円と同じ phi225（半径 112.5）
    # で、中心距離 337.5 と速比 2 から一意に決まる。テーブル出力側はこの速比倍
    # （phi450 = テーブルのボトルピッチ円）。駆動を組む側がこの値を読む。
    #
    # 歯は刻む。歯先円の和 229 + 116.5 = 345.5 に対し中心距離は 337.5 なので、
    # 片方でも平らな円板だと必ず 8 mm 食い込む。歯どうしが互いの歯溝に入って
    # 初めて成立する。諸元と位相の決め方はテーブル側と揃える（同じモジュール・
    # 同じ歯先歯底・同じ逃げ・歯 1 枚の中心をピッチ点で歯溝の中心に合わせる）
    "gear_teeth": 45,                      # 星車側の歯数。モジュール = 225/45 = 5
    "gear_addendum_x_module": 0.80,        # 歯先の出 / モジュール -> 4
    "gear_dedendum_x_module": 1.10,        # 歯底の落ち / モジュール -> 5.5
    "gear_backlash": 0.12,                 # 歯厚を細める割合（噛み合いの逃げ）
    "gear_pressure_deg": 20.0,             # 歯すじの傾き（歯を台形で描く用）
    "gear_tooth_bite_x_module": 0.30,      # 歯の付け根を胴へ食い込ませる量 -> 1.5
    "gear_table_phase_deg": 0.0,           # テーブル側の歯 1 枚の中心が向く世界角
    "gear_deck_gap_x_plate_t": 0.20,       # 歯車下面と天板上面の隙間 -> 1.6
    "gear_table_gap_x_plate_t": 0.25,      # 歯車上面とテーブル板下面の隙間 -> 2
    "gear_boss_d_x_shaft_d": 1.70,         # 歯車の締めボス径 -> 51
    "gear_boss_h_x_plate_t": 0.50,         # 同 高さ -> 4

    # --- 渡し板（デッドプレート）-------------------------------------------
    # ボトルの底が浮く区間を埋める固定板。ボトル中心は星車のピッチ円上を通る
    # ので、板も星車の中心まわりの環状扇形にする
    "dead_clear_x_bottle_r": 0.07,         # ボトル底の外側に足す幅 -> 2.38
    # 板厚は歯車との兼ね合いで決まる。歯車が天板の上に来たので、渡し板の
    # 下面は歯車の上面より上でなければならない
    "dead_t_x_plate_t": 0.90,              # 板厚 -> 7.2
    "dead_top_drop_x_plate_t": 0.10,       # 上面をテーブル上面から下げる量 -> 0.8
    "dead_table_gap_x_plate_t": 0.50,      # 回るテーブル板の縁からの逃げ -> 4
    # コンベア側の切り方。搬送面は接点で終わり、そこから機外へ伸びる。搬送の
    # 向きに直交する面（接点を通る）より手前なら搬送面とは当たらないので、
    # 板はその面のすぐ手前まで出せる。世界半径の円で切ると接点の 50deg も
    # 手前で板が退き、走行 100 mm ぶんボトルの足元が空く
    "dead_conv_gap_x_plate_t": 0.25,       # 搬送面の端から手前へ戻す量 -> 2
    "dead_stand_r_x_stay_r": 0.75,         # 板を支える短柱の半径 -> 4.8
    "dead_stand_deck_margin_x_plate_t": 2.0,  # 短柱を天板の縁から戻す量 -> 16
    "dead_stand_at": (0.20, 0.50, 0.80),   # 短柱を立てる位置（板の何割か）
    "dead_min_span_deg": 12.0,             # これより短い板は捨てる
    # テーブル板の縁に沿う固定エプロン。星車まわりの扇形では届かない帯を埋める
    "dead_apron_margin_deg": 3.0,          # 実測した範囲の左右に足す逃げ角
    "dead_apron_out_x_plate_t": 0.75,      # 実測した外端にさらに足す幅 -> 6
    # 三日月ガイドの棒を立てる耳。板の外縁から棒の座まで張り出させる。
    # 板へ食い込ませる量と棒の外へ出す量に同じ値を使う
    "dead_lug_bite_x_stay_r": 1.60,        # 板への食い込み / 棒の外への出しろ -> 10.2
    "dead_lug_w_x_stay_r": 3.60,           # 耳の幅（棒の座のところで）-> 23

    # --- 三日月ガイド -------------------------------------------------------
    "crescent_clear_x_bottle_r": 0.06,     # 内面とボトルの隙間 -> 2.04
    "crescent_t_x_plate_t": 1.25,          # 板厚（半径方向）-> 10
    "crescent_base_x_star_lift": 0.55,     # 下端をテーブル上面から上げる量 -> 38.5
    "crescent_h_x_star_lift": 0.90,        # 高さ -> 63。板を挟んで上下に出す
    "crescent_lane_clear_x_plate_t": 0.125,  # コンベア側のボトル通り道に対する逃げ -> 1
    "crescent_stay_r_x_plate_t": 0.80,     # ガイドを支える丸棒の半径 -> 6.4
    "crescent_stay_bite_x_stay_r": 0.92,   # 丸棒の芯を板の外面から外へ出す量 / 棒の半径
    "crescent_stay_deck_margin_x_stay_r": 1.5,  # 丸棒を天板の縁から戻す量
    "crescent_stay_at": (0.25, 0.75),      # 丸棒を立てる位置（円弧の何割か）
    "crescent_stay_window": 0.18,          # 目標から動かしてよい幅（円弧の何割か）
    # 三日月のテーブル側の端は世界半径 274.6 まで来るので、そこの丸棒は天板に
    # 載る。コンベア側の端は世界半径 496 で、天板（378）の外なので載らない。
    # そこは渡し板の耳に立てる。渡し板自体が天板の短柱に載っているので、
    # 受けは天板から出ていることになる。床から棒を立てることはしない
    # （アンカーも筋交いも無い 950 mm の棒が丸座 1 枚で立つ形になり、
    # しかも搬送の足元と操作盤の扉の前を塞ぐ）
    "crescent_stay_plate_margin_deg": 3.0,  # 耳を渡し板の端から戻す角
    "crescent_ring_gap_x_plate_t": 0.50,   # テーブル外周ガイド外面からの逃げ -> 4
    # コンベア側は接点で切らずに接線方向の直線レールを継ぐ。三日月の円弧は
    # 接点から先で必ずボトルの通り道へ食い込むので、円弧はそこで止め、あとは
    # 接線に沿った真っすぐな受けにする。長さはコンベアのサイドガイドが立てる
    # 位置（星車中心から外半径ぶん離れるところ）まで
    "crescent_lead_gap_x_plate_t": 0.25,   # 直線レールの端を手前で止める量 -> 2
    "crescent_pair_margin_deg": 2.0,       # 2 台のガイドどうしの逃げ角
    "crescent_min_span_deg": 20.0,         # これより短い断片は捨てる

    # --- テーブル外周のボトル押さえガイド -----------------------------------
    "ring_clear_x_bottle_r": 0.0882353,    # 内面とボトルの隙間 -> 3（半径 262）
    "ring_t_x_plate_t": 1.0,               # 板厚（半径方向）-> 8
    "ring_base_x_star_lift": 0.36,         # 下端をテーブル上面から上げる量 -> 25.2
    "ring_h_x_star_lift": 0.55,            # 高さ -> 38.5
    "ring_notch_margin_deg": 4.0,          # 切り欠きの左右に足す逃げ角
    "ring_max_span_deg": 150.0,            # 1 本の円弧の上限。超えたら割る
    "ring_joint_deg": 2.5,                 # 割ったときの継ぎ目の隙間

    # --- 外周ガイドを支える支柱 ---------------------------------------------
    # 支柱は回るテーブル板（280）の外、天板の縁（378.0）の内側に立てる。
    # 縁寄りは液受け溝とコンベアの枠が来るので、板のすぐ外に寄せる
    "ring_post_at_x_plate_r": 1.055,       # 支柱を立てる半径 -> 295.4
    "ring_post_r_x_stay_r": 1.0,           # 支柱の半径。三日月のステーと同じ棒 -> 6.4
    "ring_post_foot_d_x_post_r": 4.0,      # 据付座の径 -> 25.6
    "ring_post_foot_t_x_plate_t": 0.45,    # 同 板厚 -> 3.6
    "ring_arm_t_x_plate_t": 0.90,          # リングへ出す腕の厚み -> 7.2
    "ring_arm_w_x_post_r": 2.20,           # 同 幅 -> 14.1
    "ring_arm_z_x_ring_h": 0.50,           # 腕の高さ（ガイドの高さの何割か）
    "ring_post_gap_x_plate_t": 1.0,        # 他の部品との隙間 -> 8
    "ring_post_at": (0.25, 0.75),          # 支柱を立てる位置（円弧の何割か）
    "ring_post_single_span_deg": 60.0,     # これより短い円弧は支柱 1 本

    # --- 星車のポケットに乗せるボトル ---------------------------------------
    # 受け渡し点から数えて何ポケット目に置くか。実機で埋まっているポケット
    # をそのまま並べる。
    #   供給   コンベア接点（局所 315deg）で受けて 2 割出しで受け渡し点へ。
    #          受け渡し点のポケットは渡した直後なので空。1・2 ポケット目が実
    #   排出   受け渡し点（局所 90deg）で受けて 2 割出しでコンベアへ抜ける。
    #          受けた直後のポケットが実なので 0・1 ポケット目。0 ポケット目は
    #          その瞬間だけテーブル側のボトルと重なるが、位置が完全に一致する
    #          ので絵には出ない
    "star_bottle_steps_infeed": (1, 2),
    "star_bottle_steps_discharge": (0, 1),
}

# 材質。ガイド類は白い樹脂で、透かさない。板はステンレス。支持は架台と同じ鋼。
# ボトルと液は scene 側の材質をそのまま借りる（テーブル上のボトルと同じ見え方に
# するため。透過物なので DRAW_ORDER の並びには注意する）。
MATERIALS = {
    "trf_steel": dict(color="#d8dee4", pbr=True, metallic=0.84, roughness=0.28),
    "trf_mount": dict(color="#8d959c", pbr=True, metallic=0.55, roughness=0.62),
    "trf_guide": dict(color="#eceff1", pbr=True, metallic=0.0, roughness=0.42),
    # 歯車だけは黒染めの機械部品として暗く落とす。星車の外径 phi235 と歯先円
    # phi233 がほぼ同じで、真上から見るとポケットの切り欠き越しに下の歯が
    # 覗く。同じ明るいステンレスにすると、星車の縁に歯が生えて見える
    "trf_gear": dict(color="#4c525a", pbr=True, metallic=0.72, roughness=0.46),
    # 渡し板とエプロン。回る星車板と同じ明るいステンレスにすると、真上から
    # 見たとき固定の板と回る板がひと続きの白い塊になる。ヘアライン仕上げの
    # ぶんだけ暗く、艶を落とす
    "trf_plate": dict(color="#a7b0b8", pbr=True, metallic=0.30, roughness=0.68),
}

# 自転する群の名前。1 つの群に 1 つの中心しか持てないので、2 台は必ず分ける。
STAR_ROLE_SUFFIX = {"infeed": "in", "discharge": "out"}
# 群の枝と、当てる材質。枝ごとに 2 台ぶんの群ができる。
STAR_GROUP_PARTS = (("", "trf_steel"), ("_gear", "trf_gear"),
                    ("_liquid", "liquid"), ("_glass", "glass"),
                    ("_bottle", "bottle"))


def _star_group(role: str, part: str = "") -> str:
    return f"trf_star_{STAR_ROLE_SUFFIX[role]}{part}"


# 描く順。不透明を先に置く。ボトル・ガラス・液は透過物なので、モジュールの
# 断片の中では最後に回す（scene 側の並びと同じ 液 -> ガラス -> ボトル）。
DRAW_ORDER = (
    ("trf_mount", "trf_mount"),
    ("trf_plate", "trf_plate"),
    ("trf_steel", "trf_steel"),
    ("trf_star_in", "trf_steel"),
    ("trf_star_out", "trf_steel"),
    ("trf_star_in_gear", "trf_gear"),
    ("trf_star_out_gear", "trf_gear"),
    ("trf_guide", "trf_guide"),
    ("trf_star_in_liquid", "liquid"),
    ("trf_star_out_liquid", "liquid"),
    ("trf_star_in_glass", "glass"),
    ("trf_star_out_glass", "glass"),
    ("trf_star_in_bottle", "bottle"),
    ("trf_star_out_bottle", "bottle"),
)

# テーブルとは一緒に回らない。固定側（スターホイールは自分の軸で自転する）。
ROTATING = False


# --------------------------------------------------------------------------
# 自転
# --------------------------------------------------------------------------
_PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.json"


def _unit(deg):
    """世界角 [deg] の単位ベクトル。角度まわりはすべて度で扱う。"""
    a = np.radians(deg)
    return np.array([np.cos(a), np.sin(a)])


def _star_centers(params: dict) -> dict:
    """スターホイール 2 台の中心 (x, y)。群名 -> 中心。

    2 つのピッチ円が受け渡し点で接する、という条件だけで決まる。
    """
    pitch_r = float(params["table"]["pitch_circle_diameter_mm"]) / 2.0
    center_r = pitch_r * (1.0 + PROPORTION["star_pcd_r_x_pitch_r"])
    st = params["stations"]
    out = {}
    for role, key in (("infeed", "infeed_deg"), ("discharge", "discharge_deg")):
        c = center_r * _unit(float(st[key]))
        for suffix, _ in STAR_GROUP_PARTS:
            out[_star_group(role, suffix)] = (float(c[0]), float(c[1]))
    return out


def _boot_centers() -> dict:
    """読み込み時点の自転中心。layout() が呼ばれれば同じ値で上書きされる。

    寸法を直書きしないために params.json から出す。読めなければ空で返し、
    layout() が入れるまで待つ。
    """
    try:
        with open(_PARAMS_PATH, encoding="utf-8") as fp:
            return _star_centers(json.load(fp))
    except Exception:                                  # noqa: BLE001 読めないなら後で
        return {}


# 自転する群 -> 自転の中心 (x, y)。中身は layout() が同じ辞書のまま入れ替える
# ので、読み込み時に控えを取った側からも最新の値が見える。
SPIN_CENTERS: dict = {}


def _speed_ratio(params: dict, lay: dict) -> float:
    """テーブル 1 割出しに対する星車の回転比。ポケット数から出る。

    1 割出しで星車はちょうど 1 ポケット（360/ポケット数）回る。テーブルは
    その割出し角だけ回る。ピッチが揃っているので、この比は受け渡し点での
    速度が合う比そのものになる。4 ポケット・割出し 45deg でちょうど 2.0。
    """
    pockets = int(lay.get("trf_star_pockets", PROPORTION["star_pockets"]))
    index_deg = float(params["table"]["index_angle_deg"])
    return (360.0 / pockets) / index_deg


def _table_angle_rad(params: dict, psi_rad: float) -> float:
    """入力軸角 psi [rad] -> テーブル角 [rad]。基盤側と同じ関数を通す。

    星車の自転角はこれの -速比 倍。同じ関数を通すことだけが、受け渡し点の
    相対速度を 0 に保つ条件になる。ここで別の曲線を書くとカム曲線の違いが
    そのまま滑りになる。
    """
    import scene                                      # noqa: PLC0415 循環参照を避ける

    return float(scene.table_angle_from_cam(params, float(psi_rad)))


def spin_angles(params: dict, lay: dict, state) -> dict:
    """群名 -> 自転角 [rad]。state.cam_angle_rad（入力軸角 psi）から出す。

    テーブルと**同時に**動く。psi 0〜180deg（割出し）のあいだにテーブルが
    +45deg 進み、星車は CW へ 90deg = 1 ポケット回る。psi 180〜360deg
    （停留）はどちらも止まる。
    """
    psi = float(getattr(state, "cam_angle_rad", 0.0) or 0.0)
    ang = -_speed_ratio(params, lay) * _table_angle_rad(params, psi)
    return {name: float(ang) for name in (SPIN_CENTERS or _boot_centers())}


SPIN_CENTERS.update(_boot_centers())


# --------------------------------------------------------------------------
# 角度まわりの小道具。すべて度で扱う
# --------------------------------------------------------------------------
def _deck_r(lay: dict) -> float:
    """架台天板の外半径。天板の直径は基盤側が lay["deck_d"] に出している。

    以前はここでカバー半径から作っていたが、天板とカバーは別々の値になった
    ので、カバー側を見ていると 18 mm 大きい天板を当てにすることになる。
    """
    return float(lay["deck_d"]) / 2.0


def _plate_overlap(center, pcd_r, outer_r, pocket_r, pockets, phase_deg,
                   station_deg, index_deg, pitch_r, bottle_r, ratio,
                   table_steps=91, ring=360):
    """受け渡し点のボトルが星車の板材に食い込む点の数。

    テーブル角を 0 から 1 割出しぶんまで刻み、そのときのボトル円（半径
    bottle_r）の周を点で刻む。各点が「星車の外半径の内側」かつ「どのポケット
    円の外側」なら、そこは板材で、ボトルと重なっている。

    星車の自転角はテーブル角の -ratio 倍。ピッチが揃っていれば重なりは 0 に
    なる。揃っていないと、割出しの序盤でボトルがポケットの縁に乗り上げる。
    戻りは (重なった点の数, 数えた点の総数, いちばん悪かったテーブル角)。
    """
    c = np.asarray(center, dtype=float)
    ring_th = np.linspace(0.0, 360.0, ring, endpoint=False)
    ring_pts = bottle_r * np.column_stack([np.cos(np.radians(ring_th)),
                                           np.sin(np.radians(ring_th))])
    hit = 0
    total = 0
    worst_deg = 0.0
    worst_hit = -1
    for th in np.linspace(0.0, index_deg, table_steps):
        b = pitch_r * _unit(station_deg + th) - c        # 星車中心を原点にする
        q = b + ring_pts
        rho = np.hypot(q[:, 0], q[:, 1])
        inside_disc = rho < outer_r
        in_pocket = np.zeros(len(q), dtype=bool)
        for j in range(int(pockets)):
            pc = pcd_r * _unit(phase_deg - ratio * th + 360.0 * j / pockets)
            in_pocket |= np.hypot(q[:, 0] - pc[0], q[:, 1] - pc[1]) <= pocket_r
        n = int(np.count_nonzero(inside_disc & ~in_pocket))
        if n > worst_hit:
            worst_hit, worst_deg = n, float(th)
        hit += n
        total += len(q)
    return hit, total, worst_deg


def _leg_phase_deg(params: dict, lay: dict) -> float:
    """架台の脚を対角（45deg）から振ってある角 [deg]。基盤側の表が正典。

    脚の芯は正方形配置なので、素のままなら世界角 45/135/225/315deg で、
    供給星車の中心（半径 337.5・世界角 315deg）とほぼ同心になる。星車の軸は
    脚とほぼ同じ半径に来るため、この角が食い違うと当たり判定が丸ごと嘘になる。

    以前はキーが無いと黙って -22.5deg の控えに落ちていた。ほかのモジュールは
    同じ場面で 0.0 を使うので、脚の位置についてだけ受け渡しが 22.5deg ずれた
    まま誰にも気付かれない。控えは 0.0 に揃え、落ちたことは stderr に出す。
    """
    if "frame_leg_phase_deg" in lay:
        return float(lay["frame_leg_phase_deg"])
    for key in ("frame_leg_deg", "frame_leg_deg_all"):
        if key in lay:
            v = lay[key]
            got = float(v[0] if isinstance(v, (tuple, list)) else v) - 45.0
            print(f"[trf] frame_leg_phase_deg が無いので {key} から出した:"
                  f" {got:+.1f}deg", file=sys.stderr)
            return got
    print("[trf] 架台の脚の振りが lay に無いので控えの 0.0deg で当たりを見る"
          "（frame_leg_phase_deg / frame_leg_deg / frame_leg_deg_all のどれも"
          "無い）。脚を振ってあるなら、星車の軸と歯車の当たり判定は当てにならない",
          file=sys.stderr)
    return 0.0


def _leg_clash(lay: dict, center, r, z0, z1, phase_deg=0.0) -> float:
    """架台の脚（角パイプ）と、半径 r・高さ z0..z1 の丸物の食い込み [mm]。

    脚は芯々 frame_span の正方形配置で、断面は一辺 frame_pipe の角。上端は
    天板の下面。架台を phase_deg 振ってあるので、丸物の中心を脚の座標系へ
    戻してから当てる。食い込んでいなければ負の値（隙間）を返す。
    """
    half = float(lay["frame_span"]) / 2.0
    pipe = float(lay["frame_pipe"]) / 2.0
    leg_top = float(lay["deck_top"]) - float(lay["deck_t"])
    if z0 >= leg_top:
        return -float("inf")
    a = np.radians(-float(phase_deg))
    cx = float(center[0]) * np.cos(a) - float(center[1]) * np.sin(a)
    cy = float(center[0]) * np.sin(a) + float(center[1]) * np.cos(a)
    worst = -float("inf")
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            dx = abs(cx - sx * half) - pipe
            dy = abs(cy - sy * half) - pipe
            # 角の外は距離、内は負。丸物の半径を足すと食い込み量になる
            out = np.hypot(max(dx, 0.0), max(dy, 0.0)) + min(max(dx, dy), 0.0)
            worst = max(worst, float(r) - out)
    return worst


def _subtract_arc(a0, a1, b0, b1):
    """円弧 [a0, a1] から [b0, b1] を引いた残りを返す。戻りは [(s, e), ...]。

    b は 360 ずらした枝も見る。a1 > a0、b1 > b0 で、どちらも 360 未満の幅を前提。
    """
    span_b = b1 - b0
    k = np.floor((a0 - b0) / 360.0)
    out = []
    cur = a0
    for shift in (k, k + 1.0):
        s = max(b0 + 360.0 * shift, a0)
        e = min(b0 + 360.0 * shift + span_b, a1)
        if e <= s:
            continue
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if a1 > cur:
        out.append((cur, a1))
    return out


def _world_r(center_r, r, delta_deg):
    """中心が半径 center_r にある円の、中心角から delta ずれた点の世界半径。"""
    return np.sqrt(center_r ** 2 + r ** 2
                   + 2.0 * center_r * r * np.cos(np.radians(delta_deg)))


def _max_delta_outside(center_r, ri, ro, r_clear, step=0.25):
    """中心が半径 center_r にある円弧板が、世界半径 r_clear の外に留まる振り角。

    中心角からの振り Delta を 0 から広げていくと、板は機械の中心へ寄っていく。
    r_clear を割る手前で止めた角を返す。テーブル上のボトルへ突っ込ませないため。
    """
    rr = np.linspace(ri, ro, 7)
    d = 0.0
    while d <= 180.0:
        if _world_r(center_r, rr, d).min() < r_clear:
            return max(d - step, 0.0)
        d += step
    return 180.0


def _min_delta_behind_lane(ri, gap):
    """搬送面の手前の端より、板が丸ごと手前に退く、いちばん小さい振り角 [deg]。

    搬送面はスターホイールのピッチ円との接点から機外へ伸びる。接点を通り
    搬送の向きに直交する面より手前（機械側）なら、搬送面とは高さが重なって
    いても当たらない。星車の中心から半径 r・接点方向から Delta ずれた点の、
    搬送の向きの座標は -r sin(Delta) なので、いちばん内側の半径 ri が
    gap だけ手前に退けば板は丸ごと退く。

    以前はここを「外縁が世界半径 438.1 の円に収まるまで退く」で切っていた。
    ボトル中心の通り道が世界 450 に届くのは接点だけなので、板が 50deg
    （走行 約 100 mm）手前で退き、そのあいだボトルの足元が空いていた。
    円は搬送面の輪郭ではない。
    """
    ri = float(ri)
    if ri <= 0.0:
        return 90.0
    return float(np.degrees(np.arcsin(np.clip(float(gap) / ri, -1.0, 1.0))))


def _clash_window(c_a, ri_a, ro_a, c_b, ro_b, margin_deg):
    """円弧板 a のうち、相手 b の外径に食い込む中心角の範囲を返す。

    供給と排出を 2 ステーション離した今は 2 台の中心間が 477.3 あって、
    ガイドどうしは食い合わない（この関数は None を返す）。1 ステーションに
    戻すと中心間は 258.3 まで詰まって必ずぶつかるので、判定は残しておく。
    """
    v = np.asarray(c_b, dtype=float) - np.asarray(c_a, dtype=float)
    d = float(np.hypot(*v))
    th = float(np.degrees(np.arctan2(v[1], v[0])))
    dmax = 0.0
    for r in np.linspace(ri_a, ro_a, 9):
        cos_d = (r * r + d * d - ro_b * ro_b) / (2.0 * r * d)
        if cos_d < 1.0:
            dmax = max(dmax, float(np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))))
    if dmax <= 0.0:
        return None
    return (th - dmax - margin_deg, th + dmax + margin_deg)


def _transfer_window(center, pcd_r, swept, bottle_r, ri, ro, ref_deg):
    """スターホイールに抱えられたボトルが、外周ガイドの帯を横切る世界角の幅。

    ボトル中心をスターホイールのピッチ円に沿って走らせ、ボトルの丸が帯
    [ri, ro] に掛かる世界角を全部拾う。ここを切り欠かないとボトルが出入り
    できない。戻りは ref_deg を基準にした (lo, hi)。
    """
    lo, hi = None, None
    for t in np.linspace(swept[0], swept[1], 241):
        q = np.asarray(center, dtype=float) + pcd_r * _unit(t)
        rho = float(np.hypot(*q))
        phi = float(np.degrees(np.arctan2(q[1], q[0])))
        phi = ref_deg + (phi - ref_deg + 180.0) % 360.0 - 180.0
        alpha = 0.0
        touched = False
        for rr in np.linspace(ri, ro, 9):
            if abs(rr - rho) > bottle_r:
                continue
            touched = True
            cos_a = (rr * rr + rho * rho - bottle_r ** 2) / (2.0 * rr * rho)
            alpha = max(alpha, float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))))
        if not touched:
            continue
        lo = phi - alpha if lo is None else min(lo, phi - alpha)
        hi = phi + alpha if hi is None else max(hi, phi + alpha)
    return None if lo is None else (lo, hi)


def _free_runs(windows, step=0.5):
    """0..360 から windows を抜いた残りの円弧を返す。跨ぎも畳む。

    切り欠きが 2 つあっても、近ければ 1 つに繋がる。その判定を素直に書くと
    場合分けが増えるので、細かい枡を塗って空いている連なりを拾う。
    """
    n = int(round(360.0 / step))
    occ = np.zeros(n, dtype=bool)
    for lo, hi in windows:
        i0 = int(np.floor(lo / step))
        i1 = int(np.ceil(hi / step))
        for i in range(i0, i1):
            occ[i % n] = True
    if occ.all():
        return []
    if not occ.any():
        return [(0.0, 360.0)]
    # 空いている連なりを跨ぎのところで切らないよう、塞がった枡の次から数える
    start = next(i for i in range(n) if occ[i - 1] and not occ[i])
    runs = []
    i = 0
    cur = None
    while i < n:
        k = (start + i) % n
        if not occ[k]:
            if cur is None:
                cur = start + i
        elif cur is not None:
            runs.append((cur * step, (start + i) * step))
            cur = None
        i += 1
    if cur is not None:
        runs.append((cur * step, (start + n) * step))
    return runs


def _split_run(a0, a1, max_span, joint):
    """長い円弧を、継ぎ目を空けながら等分する。実機のガイドも継いである。"""
    span = a1 - a0
    if span <= max_span:
        return [(a0, a1)]
    n = int(np.ceil(span / max_span))
    piece = (span - joint * (n - 1)) / n
    return [(a0 + i * (piece + joint), a0 + i * (piece + joint) + piece)
            for i in range(n)]


def _pick_along(a0, a1, targets, ok, samples=241):
    """円弧 [a0, a1] の目標割合に近い、条件 ok(t) を満たす角を選ぶ。

    そのままの位置が使えないときだけ、円弧に沿って近い方へ寄せる。
    見つからなければ None を並べる。ok は 1 本決めるたびに条件が変わる
    （立てたものどうしの逃げ）ので、目標ごとに数え直す。
    """
    grid = np.linspace(a0, a1, samples)
    out = []
    for frac in targets:
        want = a0 + (a1 - a0) * float(frac)
        good = np.array([bool(ok(float(t))) for t in grid])
        if not good.any():
            out.append(None)
            continue
        cand = grid[good]
        out.append(float(cand[int(np.argmin(np.abs(cand - want)))]))
    return out


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
def layout(params: dict, lay: dict) -> dict:
    """スターホイール・駆動・渡し板・ガイドの絶対座標を決める。

    ここで幾何の自己検査を通す。口の幅がボトル外径を下回るか、スターホイール板
    とポケット板の z が重なったら、絵を焼く前に止める。
    """
    p = PROPORTION
    pitch_r = lay["pitch_r"]
    bottle_r = lay["bottle_r"]
    bottle_d = 2.0 * bottle_r
    plate_t = lay["plate_t"]
    plate_r = lay["plate_r"]
    table_top = lay["table_top"]
    deck_top = lay["deck_top"]
    deck_r = _deck_r(lay)

    st = params["stations"]
    units_in = ((float(st["infeed_deg"]), "infeed", -1.0),
                (float(st["discharge_deg"]), "discharge", +1.0))

    # --- スターホイール本体 -----------------------------------------------
    star_pcd_r = pitch_r * p["star_pcd_r_x_pitch_r"]
    star_outer_r = pitch_r * p["star_outer_r_x_pitch_r"]
    pocket_r = bottle_r * p["star_pocket_r_x_bottle_r"]
    star_t = plate_t * p["star_t_x_plate_t"]
    center_r = pitch_r + star_pcd_r          # 2 つのピッチ円が受け渡し点で接する
    lift = lay["body_h"] * p["star_lift_x_body_h"]
    star_base_z = table_top + lift
    star_top_z = star_base_z + star_t
    tangent_r = center_r + star_pcd_r        # コンベアとの接点の世界半径

    # 口の幅。parts._notched_disc_outline() は外半径がポケットを飲み込む寸法を
    # 渡されると頭を抑えるので、抑えられた後の外半径で測る。
    r_eff = min(star_outer_r, star_pcd_r + pocket_r * 0.55)
    r_eff = max(r_eff, star_pcd_r - pocket_r * 0.50)
    star_mouth = _mouth_width(r_eff, star_pcd_r, pocket_r)
    if star_mouth < bottle_d:
        raise ValueError(
            "スターホイールの口が狭くてボトルが入らない: "
            f"口 {star_mouth:.2f} < ボトル外径 {bottle_d:.2f}"
            f"（外半径 {r_eff:.2f} / ピッチ円半径 {star_pcd_r:.2f} /"
            f" ポケット半径 {pocket_r:.2f}）")

    # ポケット板の口も同じ式で確かめる。板を組むのは scene 側だが、
    # 受け渡しは両方の口が開いていて初めて成立する。寸法は lay から読む。
    pp_base_z = float(lay["pocket_base"])
    pp_top_z = float(lay["pocket_top"])
    pp_r = float(lay["pocket_plate_r"])
    pp_pocket_r = float(lay["pocket_r"])
    pp_r_eff = min(pp_r, pitch_r + pp_pocket_r * 0.55)
    pp_r_eff = max(pp_r_eff, pitch_r - pp_pocket_r * 0.50)
    pp_mouth = _mouth_width(pp_r_eff, pitch_r, pp_pocket_r)
    if pp_mouth < bottle_d:
        raise ValueError(
            "ポケット板の口が狭くてボトルが出入りできない: "
            f"口 {pp_mouth:.2f} < ボトル外径 {bottle_d:.2f}"
            f"（外半径 {pp_r_eff:.2f}）")

    # 高さの突き合わせ。ここが重なったら平面図で 25 mm 食い込む配置がそのまま
    # 3D の干渉になる。
    if star_base_z < pp_top_z and pp_base_z < star_top_z:
        raise ValueError(
            "スターホイール板とポケット板の z が重なる: "
            f"スター {star_base_z:.1f}〜{star_top_z:.1f} / "
            f"ポケット板 {pp_base_z:.1f}〜{pp_top_z:.1f}")

    hub_d = pocket_r * p["star_hub_d_x_pocket_r"]
    hub_h = plate_t * p["star_hub_h_x_plate_t"]
    shaft_d = plate_t * p["star_shaft_d_x_plate_t"]
    shaft_top_z = star_top_z + hub_h + plate_t * p["star_shaft_stub_x_plate_t"]

    # --- 歯車（天板の上・テーブル板の下）-----------------------------------
    # 平歯車。ピッチ円は星車のボトルピッチ円と同じで、テーブル側はその速比倍。
    # 中心距離 337.5 = 225 + 112.5 がそのまま噛み合いの条件になる。
    #
    # 高さは天板の上面とテーブル板の下面のあいだ（16 mm）に置く。天板の下に
    # 置くと歯先円 phi233 が架台の脚に食い込む（脚の芯は半径 340.5、星車の
    # 軸まわりに空くのは 98.3 mm しかない）。ここには脚も桁も来ないうえ、
    # 軸が天板を貫かなくなるので天板の穴も要らなくなる。
    deck_bottom = deck_top - lay["deck_t"]
    gear_teeth = int(p["gear_teeth"])
    gear_pcd_r = star_pcd_r                       # 星車のピッチ円そのもの
    gear_module = 2.0 * gear_pcd_r / gear_teeth
    gear_tip_r = gear_pcd_r + gear_module * p["gear_addendum_x_module"]
    gear_root_r = gear_pcd_r - gear_module * p["gear_dedendum_x_module"]
    gear_base_z = deck_top + plate_t * p["gear_deck_gap_x_plate_t"]
    gear_top_z = lay["table_base"] - plate_t * p["gear_table_gap_x_plate_t"]
    gear_face = gear_top_z - gear_base_z
    gear_boss_d = shaft_d * p["gear_boss_d_x_shaft_d"]
    gear_boss_h = plate_t * p["gear_boss_h_x_plate_t"]
    # 締めボスは歯車の上面よりさらに上に出る。回る側でいちばん高いのはここで、
    # 固定側（軸受台の腕と軸受箱）はこの頭を跨がなければならない
    gear_boss_top_z = gear_top_z + gear_boss_h
    # 軸は天板を貫かない。天板の上面で止める
    shaft_base_z = deck_top
    if gear_face <= 0.0:
        raise ValueError(
            f"歯車が天板とテーブル板のあいだに入らない: 歯幅 {gear_face:.1f} "
            f"（天板上面 {deck_top:.1f} / テーブル板下面 {lay['table_base']:.1f}）")
    if gear_base_z < deck_top or gear_top_z > lay["table_base"]:
        raise ValueError("歯車が天板かテーブル板に食い込む")
    # テーブル出力側の歯車。速比 2 なので歯数も径も 2 倍で、ピッチ円は
    # テーブルのボトルピッチ円に一致する。ここが一致していないと
    # 「ポケットのピッチを合わせる」条件と噛み合いの条件が別物になる
    ratio = _speed_ratio(params, lay)
    drive_gear_pcd_r = gear_pcd_r * ratio
    if abs(drive_gear_pcd_r - pitch_r) > 1e-9:
        raise ValueError(
            f"歯車のピッチ円がボトルのピッチ円と合わない: "
            f"テーブル側 {drive_gear_pcd_r:.3f} != {pitch_r:.3f}")
    if abs((gear_pcd_r + drive_gear_pcd_r) - center_r) > 1e-9:
        raise ValueError(
            f"歯車のピッチ円が接しない: {gear_pcd_r:.3f} + "
            f"{drive_gear_pcd_r:.3f} != 中心距離 {center_r:.3f}")

    # --- 軸受台（門型。歯車を跨いで天板の上で受ける）-----------------------
    # 片持ちアームで上から吊る形は保守で外せないので採らない。軸は歯車の上で
    # 受ける。歯車が軸まわり半径 116.5 を占めるので、柱はその外へ逃がし、
    # 歯車を跨ぐ腕で軸受箱を持つ。柱を立てられるのはボトルが通らない側だけ
    bearing_d = shaft_d * p["bearing_d_x_shaft_d"]
    bearing_bore = shaft_d * p["bearing_bore_x_shaft_d"]
    bearing_top_z = star_base_z - plate_t * p["bearing_gap_x_plate_t"]
    ped_post_r = shaft_d * p["ped_post_r_x_shaft_d"]
    ped_at_r = gear_tip_r + ped_post_r + plate_t * p["ped_post_gap_x_plate_t"]
    ped_foot_d = ped_post_r * p["ped_foot_d_x_post_r"]
    ped_foot_t = plate_t * p["ped_foot_t_x_plate_t"]
    ped_arm_t = plate_t * p["ped_arm_t_x_plate_t"]
    ped_arm_w = ped_post_r * p["ped_arm_w_x_post_r"]
    # 腕と軸受箱は歯車の締めボスの頭を跨ぐ。上面 882 で見ると、その上に出る
    # ボス（頭 886・回る）に固定側が 2 mm 埋まる
    ped_arm_base_z = gear_boss_top_z + plate_t * p["ped_arm_gap_x_plate_t"]
    ped_arm_top_z = ped_arm_base_z + ped_arm_t
    if ped_arm_top_z >= bearing_top_z:
        raise ValueError("軸受台の腕が円盤下面まで届いてしまう")
    if ped_arm_base_z <= gear_boss_top_z:
        raise ValueError(
            f"軸受台の腕が歯車の締めボスに埋まる: 腕の下面 {ped_arm_base_z:.1f}"
            f" <= ボスの頭 {gear_boss_top_z:.1f}")
    # テーブル出力側の歯車。速比 2 なので歯数も径も 2 倍で、ピッチ円は
    # テーブルのボトルピッチ円に一致する。ここが一致していないと
    # 「ポケットのピッチを合わせる」条件と噛み合いの条件が別物になる
    ratio = _speed_ratio(params, lay)
    drive_gear_pcd_r = gear_pcd_r * ratio
    if abs(drive_gear_pcd_r - pitch_r) > 1e-9:
        raise ValueError(
            f"歯車のピッチ円がボトルのピッチ円と合わない: "
            f"テーブル側 {drive_gear_pcd_r:.3f} != {pitch_r:.3f}")
    if abs((gear_pcd_r + drive_gear_pcd_r) - center_r) > 1e-9:
        raise ValueError(
            f"歯車のピッチ円が接しない: {gear_pcd_r:.3f} + "
            f"{drive_gear_pcd_r:.3f} != 中心距離 {center_r:.3f}")

    # --- ガイドの半径 -------------------------------------------------------
    cre_ri = star_pcd_r + bottle_r + bottle_r * p["crescent_clear_x_bottle_r"]
    cre_ro = cre_ri + plate_t * p["crescent_t_x_plate_t"]
    cre_base_z = table_top + lift * p["crescent_base_x_star_lift"]
    cre_h = lift * p["crescent_h_x_star_lift"]

    ring_ri = pitch_r + bottle_r + bottle_r * p["ring_clear_x_bottle_r"]
    ring_ro = ring_ri + plate_t * p["ring_t_x_plate_t"]
    ring_base_z = table_top + lift * p["ring_base_x_star_lift"]
    ring_h = lift * p["ring_h_x_star_lift"]
    if ring_base_z + ring_h >= star_base_z:
        raise ValueError(
            "テーブル外周ガイドの上端がスターホイール板に当たる: "
            f"{ring_base_z + ring_h:.1f} >= {star_base_z:.1f}")

    # --- 渡し板 -------------------------------------------------------------
    dead_ri = star_pcd_r - bottle_r - bottle_r * p["dead_clear_x_bottle_r"]
    dead_ro = star_pcd_r + bottle_r + bottle_r * p["dead_clear_x_bottle_r"]
    dead_t = plate_t * p["dead_t_x_plate_t"]
    dead_top_z = table_top - plate_t * p["dead_top_drop_x_plate_t"]
    dead_base_z = dead_top_z - dead_t
    # 回るテーブル板の縁から逃げる。ポケット板（外半径 235）はもっと内側
    dead_table_clear = plate_r + plate_t * p["dead_table_gap_x_plate_t"]

    # --- 台ごとの座標 -------------------------------------------------------
    units = []
    for ang, role, swept_sign in units_in:
        c = center_r * _unit(ang)
        units.append(dict(
            role=role,
            angle_deg=ang,
            center=(float(c[0]), float(c[1])),
            # ポケットの 1 つをテーブル側（受け渡し点）へ正対させる
            phase_deg=(ang + 180.0) % 360.0,
            swept_sign=swept_sign,
            # ボトルが載っている側の半周。ガイドはここを覆う
            swept=(ang, ang + 180.0) if swept_sign > 0 else (ang - 180.0, ang),
        ))

    # --- 軸受台の柱を立てる位置 ---------------------------------------------
    # ボトルが通らない側（swept の反対側）へ振る。天板の上に足が載り、回る
    # テーブル板の外でなければならないので、目標角から近い順に当てて選ぶ。
    ped_w_min = plate_r + plate_t * p["ped_plate_gap_x_plate_t"] + ped_post_r
    ped_w_max = deck_r - ped_post_r * (1.0 + p["ped_deck_margin_x_post_r"])
    for u in units:
        want = -u["swept_sign"] * p["ped_at_deg"]
        picked = None
        for k in range(0, 33):
            for sign in ((1.0,) if k == 0 else (1.0, -1.0)):
                d = want + sign * k * 2.5 * (-u["swept_sign"])
                q = np.asarray(u["center"], dtype=float) + ped_at_r * _unit(
                    u["angle_deg"] + d)
                w = float(np.hypot(*q))
                if ped_w_min <= w <= ped_w_max:
                    picked = (float(d), (float(q[0]), float(q[1])), w)
                    break
            if picked is not None:
                break
        if picked is None:
            raise ValueError(
                f"{u['role']} の軸受台の柱が天板に載らない: 世界半径 "
                f"{ped_w_min:.1f}〜{ped_w_max:.1f} に入る角が無い")
        u["ped_delta_deg"], u["ped_xy"], u["ped_world_r"] = picked

    # --- 受け渡し点のボトルと星車の板材 ------------------------------------
    # ここが今回の要。ピッチが揃っていれば、テーブルが 1 割出しぶん動くあいだ
    # ずっとボトルは板材に触れない。揃っていないと割出しの序盤でボトルが
    # ポケットの縁に乗り上げる（前の 6 ポケット・ピッチ半径 80 がそれ）。
    index_deg = float(params["table"]["index_angle_deg"])
    overlap = {}
    for u in units:
        hit, total, worst = _plate_overlap(
            u["center"], star_pcd_r, star_outer_r, pocket_r,
            p["star_pockets"], u["phase_deg"], u["angle_deg"], index_deg,
            pitch_r, bottle_r, ratio)
        overlap[u["role"]] = (hit, total, worst)
        u["plate_overlap"] = (hit, total, worst)
        if hit:
            raise ValueError(
                f"{u['role']} の受け渡し点でボトルが星車の板材と重なる: "
                f"{hit}/{total} 点（テーブル角 {worst:.1f}deg がいちばん悪い）。"
                f"ポケット数 {p['star_pockets']} / ピッチ円半径 {star_pcd_r:.1f} /"
                f" 速比 {ratio:.3f} の組み合わせが合っていない")

    # 三日月ガイドの角度範囲。
    #
    # コンベア側はコンベアとの接点で切る。接点から先へ円弧を回すと、板は接線の
    # 内側へ入っていくのでボトルの通り道を塞ぐ。塞がずに回せる限界は
    # arccos((ピッチ円半径 + ボトル半径 + 逃げ) / 内半径) で 7deg も無く、
    # 伸ばす意味が無い。接点から先は接線方向の直線レールで受ける。
    #
    # テーブル側はテーブル外周ガイドの外面に触れる手前まで。供給と排出が
    # 2 ステーション離れたので、以前 2 台のガイドが食い合っていた窓は消える。
    # 窓を出す計算は残す（配置が戻ったときに黙って重ならないため）。
    lane_clear = plate_t * p["crescent_lane_clear_x_plate_t"]
    d_conv_max = float(np.degrees(np.arccos(
        np.clip((star_pcd_r + bottle_r + lane_clear) / cre_ri, -1.0, 1.0))))
    d_conv = 0.0                                   # 接点そのもので切る
    if d_conv > d_conv_max:
        raise ValueError(
            f"三日月ガイドがコンベアのボトル通り道へ入る: "
            f"{d_conv:.1f} > {d_conv_max:.1f}deg")
    d_table = _max_delta_outside(
        center_r, cre_ri, cre_ro,
        ring_ro + plate_t * p["crescent_ring_gap_x_plate_t"])

    cre_arcs = []
    for i, u in enumerate(units):
        other = units[1 - i]
        s = u["swept_sign"]
        a = u["angle_deg"]
        lo, hi = (a + d_conv, a + d_table) if s > 0 else (a - d_table, a - d_conv)
        pieces = [(lo, hi)]
        win = _clash_window(u["center"], cre_ri, cre_ro, other["center"], cre_ro,
                            p["crescent_pair_margin_deg"])
        if win is not None:
            out = []
            for s0, s1 in pieces:
                out += _subtract_arc(s0, s1, win[0], win[1])
            pieces = out
        pieces = [q for q in pieces if q[1] - q[0] >= p["crescent_min_span_deg"]]
        if not pieces:
            raise ValueError(f"{u['role']} の三日月ガイドが 1 本も残らない")
        u["guide_arcs"] = tuple((float(q[0]), float(q[1])) for q in pieces)
        cre_arcs += [(u["center"], float(q[0]), float(q[1])) for q in pieces]

    # コンベア側に継ぐ直線レール。接点から接線方向へ伸ばす。
    # 搬送側のサイドガイドは星車まわりの丸物（三日月の外半径）を避けてから
    # でないと立てられないので、接点からしばらく横の受けが無い区間ができる。
    # その区間は接線に沿った真っすぐな板で埋める。長さは、外半径 cre_ro の
    # 丸物から接線が抜け出る距離 sqrt(cre_ro^2 - ピッチ円半径^2) まで。
    lead_len = float(np.sqrt(max(cre_ro ** 2 - star_pcd_r ** 2, 0.0))
                     - plate_t * p["crescent_lead_gap_x_plate_t"])
    lead_off = 0.5 * (cre_ri + cre_ro) - star_pcd_r   # 通り道の芯から板の芯まで
    leads = []
    for u in units:
        a = u["angle_deg"]
        s = u["swept_sign"]
        contact = tangent_r * _unit(a)
        # 接点でのボトルの進む向きは半径方向から -90deg（星車は 2 台とも CW）。
        # 供給はボトルが機械へ入ってくるので、レールは進む向きの逆へ伸ばす
        head_deg = a - 90.0 * s
        head = _unit(head_deg)
        q = contact + head * (lead_len / 2.0) + _unit(a) * lead_off
        u["lead"] = dict(xy=(float(q[0]), float(q[1])),
                         head_deg=float(head_deg), length=float(lead_len),
                         contact=(float(contact[0]), float(contact[1])))
        leads.append(u["lead"])

    # 渡し板の角度範囲。テーブル側は回る板から逃げ、コンベア側は搬送面の
    # 手前の端まで出す。ボトルの底が浮くのはこのあいだだけなので、ここを
    # 削ると足元がそのまま空く。
    dead_arcs = []
    for u in units:
        s = u["swept_sign"]
        a = u["angle_deg"]
        d_out = _max_delta_outside(center_r, dead_ri, dead_ro, dead_table_clear)
        d_in = _min_delta_behind_lane(dead_ri,
                                      plate_t * p["dead_conv_gap_x_plate_t"])
        if d_out - d_in < p["dead_min_span_deg"]:
            raise ValueError(
                f"{u['role']} の渡し板が残らない: {d_in:.1f}〜{d_out:.1f}deg")
        lo, hi = (a + d_in, a + d_out) if s > 0 else (a - d_out, a - d_in)
        u["dead_arc"] = (float(lo), float(hi))
        u["dead_delta_deg"] = (float(d_in), float(d_out))
        dead_arcs.append((u["center"], float(lo), float(hi)))

    # テーブル板の縁に沿う固定エプロン。
    #
    # 星車の中心まわりの扇形だけでは、受け渡し点の手前でボトルの外側が浮く。
    # そのあたりでは「星車の中心から遠い側」が機械の中心へ向かうので、扇形を
    # そのまま伸ばすと回るテーブル板に食い込む。伸ばせないのはそちらの都合
    # なので、足りない帯は機械の中心まわりの円弧板で受ける。実機の渡し板が
    # 単純な扇形ではなく腎臓形をしているのはこのため。
    #
    # 受ける範囲は測って決める。ボトルの底を刻んで、テーブル板の外に出て
    # いて、かつ星車まわりの扇形にも載っていない点を全部拾う。見るのは
    # 受け渡し点から扇形のテーブル側の端までのあいだだけ。コンベア側で
    # 扇形から外れる点は搬送面が受けるので、ここでは数えない。
    apron = []
    for u in units:
        a0, a1 = u["dead_arc"]
        c = np.asarray(u["center"], dtype=float)
        ref = u["angle_deg"]
        lo = hi = None
        need_ro = dead_table_clear
        # 受け渡し点（テーブルのピッチ円に接する側）から、扇形のテーブル側の端まで
        table_end = a0 if u["swept_sign"] < 0 else a1
        hand_deg = u["angle_deg"] - 180.0 * u["swept_sign"]
        hand_deg = table_end + (hand_deg - table_end + 180.0) % 360.0 - 180.0
        for t in np.linspace(hand_deg, table_end, 121):
            q = c + star_pcd_r * _unit(t)
            for frac in np.linspace(0.0, 1.0, 6):
                for k in np.linspace(0.0, 360.0, 37)[:-1]:
                    pt = q + bottle_r * frac * _unit(k)
                    w = float(np.hypot(*pt))
                    if w <= dead_table_clear:
                        continue               # 回るテーブル板が受けている
                    rel = pt - c
                    rr = float(np.hypot(*rel))
                    th = float(np.degrees(np.arctan2(rel[1], rel[0])))
                    th = a0 + (th - a0) % 360.0
                    if dead_ri <= rr <= dead_ro and a0 <= th <= a1:
                        continue               # 星車まわりの扇形が受けている
                    phi = float(np.degrees(np.arctan2(pt[1], pt[0])))
                    phi = ref + (phi - ref + 180.0) % 360.0 - 180.0
                    lo = phi if lo is None else min(lo, phi)
                    hi = phi if hi is None else max(hi, phi)
                    need_ro = max(need_ro, w)
        if lo is None:
            continue
        m = p["dead_apron_margin_deg"]
        apron.append((float(lo - m), float(hi + m),
                      float(need_ro + plate_t * p["dead_apron_out_x_plate_t"])))
    apron_ro = max([q[2] for q in apron], default=dead_table_clear)
    apron = [(q[0], q[1]) for q in apron]

    # 三日月ガイドを支える棒。円弧の 1/4・3/4 に立て、板の外面（ro 側）に
    # 当てて留める。帯の真ん中を通すと板を貫いて内面へ出てしまう。
    #
    # 星車の中心（半径 337.5）は架台天板の縁（378.0）の内側。三日月の
    # テーブル寄りの端は世界半径 274.6 まで戻ってくるので天板に載るが、
    # コンベア寄りは世界半径 496 まで出て天板から外れる。そこは渡し板の
    # 耳に立てる。渡し板は天板の短柱に載っているので、受けは天板から出て
    # いることになる。床から棒を立てることはしない（アンカーも筋交いも無い
    # 950 mm の棒が丸座 1 枚で立ち、搬送の足元と操作盤の扉の前を塞ぐ）。
    stay_r = plate_t * p["crescent_stay_r_x_plate_t"]
    stay_at_r = cre_ro + stay_r * p["crescent_stay_bite_x_stay_r"]
    stay_lim = deck_r - stay_r * p["crescent_stay_deck_margin_x_stay_r"]
    stay_clear = max(ring_ro, plate_r) + plate_t * p["ring_post_gap_x_plate_t"]
    # 渡し板の耳。板の外縁から棒の座まで張り出す環状の帯
    dead_lug_bite = stay_r * p["dead_lug_bite_x_stay_r"]
    dead_lug_ri = dead_ro - dead_lug_bite
    dead_lug_ro = stay_at_r + stay_r + dead_lug_bite
    dead_lug_deg = float(np.degrees(np.arctan2(
        0.5 * stay_r * p["dead_lug_w_x_stay_r"], stay_at_r)))
    stays = []

    def _stay_kind(u, t, w):
        """局所角 t（世界半径 w）に棒を立てるとき、天板の上か渡し板の上か。

        どちらにも載らなければ None。天板を先に取る（渡し板は片持ちなので、
        載せるものは少ないほうがよい）。
        """
        if w + stay_r < stay_lim:
            return "deck"
        a0, a1 = u["dead_arc"]
        m = p["crescent_stay_plate_margin_deg"] + dead_lug_deg
        if (a0 + m) <= t <= (a1 - m):
            return "plate"
        return None

    for i, u in enumerate(units):
        # 相手のガイドの実際の板を点で拾う。円周まるごとで見ると、
        # 相手が居ない角まで巻き添えで落ちる
        other = units[1 - i]
        oc = np.asarray(other["center"], dtype=float)
        opts = []
        for b0, b1 in other["guide_arcs"]:
            for t in np.linspace(b0, b1, 181):
                for rr in (cre_ri, cre_ro):
                    opts.append(oc + rr * _unit(t))
        opts = np.asarray(opts)

        def _ok(t, u=u, opts=opts, want=None):
            q = np.asarray(u["center"], dtype=float) + stay_at_r * _unit(t)
            w = float(np.hypot(*q))
            kind = _stay_kind(u, t, w)
            if kind is None or (want is not None and kind != want):
                return False
            if kind == "deck" and w - stay_r < stay_clear:
                return False
            if float(np.hypot(*(opts - q).T).min()) < stay_r + 2.0:
                return False
            for s in stays:
                if (float(np.hypot(q[0] - s["xy"][0], q[1] - s["xy"][1]))
                        < 2.0 * stay_r + 4.0):
                    return False
            return True

        def _ok_deck(t, u=u, opts=opts):
            return _ok(t, u=u, opts=opts, want="deck")

        for a0, a1 in u["guide_arcs"]:
            # 1 本ずつ決める。天板に載る位置があるならそちらを先に取るが、
            # 目標から遠くへは動かさない。動かし放題にすると 2 本とも天板に
            # 載る側へ寄って、コンベア寄りの端が丸ごと宙に浮く
            got = []
            win = (a1 - a0) * p["crescent_stay_window"]
            for frac in p["crescent_stay_at"]:
                want = a0 + (a1 - a0) * frac
                near = (max(a0, want - win), min(a1, want + win))
                t = _pick_along(near[0], near[1], (0.5,), _ok_deck)[0]
                if t is None:
                    t = _pick_along(near[0], near[1], (0.5,), _ok)[0]
                if t is None:
                    t = _pick_along(a0, a1, (frac,), _ok)[0]
                if t is None:
                    continue
                got.append(t)
                q = np.asarray(u["center"], dtype=float) + stay_at_r * _unit(t)
                stays.append(dict(xy=(float(q[0]), float(q[1])),
                                  kind=_stay_kind(u, t, float(np.hypot(*q))),
                                  world_r=float(np.hypot(*q)),
                                  local_deg=float(t),
                                  center=tuple(u["center"]),
                                  role=u["role"]))
            if not got:
                u.setdefault("guide_arcs_unsupported", []).append((a0, a1))

    # 渡し板を支える短柱。天板の上に収まる位置だけを使う。
    # 柱は天板から渡し板の下面まで立つので、歯車（軸まわり半径 116.5・
    # z 869.6〜882）の中を通せない。ボトルの通り道（ピッチ円）ではなく、
    # 歯先円の外の半径に立てる。
    # 板をコンベアの手前まで伸ばしたぶん天板の外へ出る区間が長くなるので、
    # 立てられる場所を 3 つ狙って、取れたものだけを使う
    stand_r = stay_r * p["dead_stand_r_x_stay_r"]
    stand_lim = deck_r - plate_t * p["dead_stand_deck_margin_x_plate_t"]
    stand_at_r = max(star_pcd_r,
                     gear_tip_r + stand_r + plate_t * p["ped_post_gap_x_plate_t"])
    if stand_at_r + stand_r > dead_ro:
        raise ValueError(
            f"渡し板の短柱が板の外へ出る: {stand_at_r + stand_r:.1f} > "
            f"{dead_ro:.1f}（歯先円 {gear_tip_r:.1f} を避けた結果）")
    stands = []
    for u in units:
        a0, a1 = u["dead_arc"]

        def _ok(t, u=u):
            q = np.asarray(u["center"], dtype=float) + stand_at_r * _unit(t)
            if float(np.hypot(*q)) + stand_r >= stand_lim:
                return False
            # 天板に載る範囲が短いと 2 本の目標が同じところへ寄る。重ねない
            for sx, sy in stands:
                if float(np.hypot(q[0] - sx, q[1] - sy)) < 4.0 * stand_r:
                    return False
            return True

        for t in _pick_along(a0, a1, p["dead_stand_at"], _ok):
            if t is None:
                continue
            q = np.asarray(u["center"], dtype=float) + stand_at_r * _unit(t)
            stands.append((float(q[0]), float(q[1])))

    # テーブル外周ガイドの切り欠き。ボトルが帯を横切る幅を実際に測って開ける。
    notches = []
    for u in units:
        win = _transfer_window(u["center"], star_pcd_r, u["swept"], bottle_r,
                               ring_ri, ring_ro, u["angle_deg"])
        if win is None:
            continue
        m = p["ring_notch_margin_deg"]
        lo = min(win[0] - m, u["angle_deg"] - m)
        hi = max(win[1] + m, u["angle_deg"] + m)
        notches.append((lo % 360.0, (lo % 360.0) + (hi - lo)))
        u["ring_notch_deg"] = (float(lo), float(hi))

    ring_arcs = []
    for a0, a1 in _free_runs(notches):
        ring_arcs += _split_run(a0, a1, p["ring_max_span_deg"], p["ring_joint_deg"])
    ring_arcs = [(float(a0), float(a1)) for a0, a1 in ring_arcs]

    # 外周ガイドを支える支柱。天板から立てて、リングの外面へ腕を出す。
    # 回る板より外に立て、ノズルの門柱・三日月のステー・渡し板を避ける。
    ring_post_at_r = plate_r * p["ring_post_at_x_plate_r"]
    ring_post_r = stay_r * p["ring_post_r_x_stay_r"]
    ring_post_gap = plate_t * p["ring_post_gap_x_plate_t"]
    if ring_post_at_r + ring_post_r > deck_r:
        raise ValueError(
            f"外周ガイドの支柱が架台天板からはみ出す: "
            f"{ring_post_at_r + ring_post_r:.1f} > {deck_r:.1f}")
    if ring_post_at_r - ring_post_r < plate_r:
        raise ValueError(
            f"外周ガイドの支柱が回るテーブル板に当たる: "
            f"{ring_post_at_r - ring_post_r:.1f} < {plate_r:.1f}")
    gantry = lay["post_at_r"] * _unit(lay["fill_deg"])
    dead_pts = []
    for c, a0, a1 in dead_arcs:
        for t in np.linspace(a0, a1, 121):
            for rr in (dead_ri, dead_ro):
                dead_pts.append(np.asarray(c, dtype=float) + rr * _unit(t))
    for a0, a1 in apron:
        for t in np.linspace(a0, a1, 61):
            for rr in (dead_table_clear, apron_ro):
                dead_pts.append(rr * _unit(t))
    dead_pts = np.asarray(dead_pts)
    # 三日月ガイドのテーブル寄りの端は外周ガイドの外面まで戻ってくるので、
    # 支柱を立てる半径（295.4）を横切る。板そのものを点で拾って避ける
    cre_pts = []
    for c, a0, a1 in cre_arcs:
        for t in np.linspace(a0, a1, 181):
            for rr in (cre_ri, cre_ro):
                cre_pts.append(np.asarray(c, dtype=float) + rr * _unit(t))
    cre_pts = np.asarray(cre_pts)
    ring_posts = []

    def _post_ok(t):
        q = ring_post_at_r * _unit(t)
        if float(np.hypot(*(q - gantry))) < ring_post_r + lay["post_r"] + ring_post_gap:
            return False
        if float(np.hypot(*(dead_pts - q).T).min()) < ring_post_r + ring_post_gap:
            return False
        if float(np.hypot(*(cre_pts - q).T).min()) < ring_post_r + ring_post_gap:
            return False
        for s in stays:
            if (float(np.hypot(q[0] - s["xy"][0], q[1] - s["xy"][1]))
                    < ring_post_r + stay_r + ring_post_gap):
                return False
        for s in ring_posts:
            if float(np.hypot(q[0] - s[0], q[1] - s[1])) < 2.0 * ring_post_r + ring_post_gap:
                return False
        return True

    unsupported_ring = []
    for a0, a1 in ring_arcs:
        want = (p["ring_post_at"] if a1 - a0 > p["ring_post_single_span_deg"]
                else (0.5,))
        picked = [t for t in _pick_along(a0, a1, want, _post_ok) if t is not None]
        if not picked:
            unsupported_ring.append((a0, a1))
        for t in picked:
            q = ring_post_at_r * _unit(t)
            ring_posts.append((float(q[0]), float(q[1])))

    ring_arm_z = ring_base_z + ring_h * p["ring_arm_z_x_ring_h"]

    # 自転の中心を、いま決めた座標で入れ直す。同じ辞書のまま入れ替えるので、
    # 読み込み時に SPIN_CENTERS の控えを取った側にも届く。
    SPIN_CENTERS.clear()
    for u in units:
        for suffix, _ in STAR_GROUP_PARTS:
            SPIN_CENTERS[_star_group(u["role"], suffix)] = tuple(u["center"])

    # 星車のポケットに乗せるボトル。割出し中の 90deg 送り（1 ポケット）の
    # あいだ受け渡し点を通らないポケットだけを使う（受け渡し点のポケットは
    # テーブル側のボトルと場所が重なるため）。
    star_bottles = []
    for u in units:
        steps = p[f"star_bottle_steps_{u['role']}"]
        u["bottle_steps"] = tuple(int(k) for k in steps)
        for k in u["bottle_steps"]:
            t = u["phase_deg"] - u["swept_sign"] * 360.0 * k / p["star_pockets"]
            q = np.asarray(u["center"], dtype=float) + star_pcd_r * _unit(t)
            star_bottles.append((u["role"], float(q[0]), float(q[1])))

    # 星車軸の歯車・軸と架台の脚の当たり。天板の下だけが相手（脚の上端は
    # 天板の下面）。歯車はプーリより一回り大きいので、当たるかどうかは
    # 脚を回した角も込みで数えて出す
    leg_phase = _leg_phase_deg(params, lay)
    legs = {}
    for u in units:
        legs[u["role"]] = dict(
            shaft=_leg_clash(lay, u["center"], shaft_d / 2.0,
                             shaft_base_z, deck_bottom, leg_phase),
            gear=_leg_clash(lay, u["center"], gear_tip_r,
                            gear_base_z, gear_top_z, leg_phase))

    return {
        "trf_center_r": center_r,
        "trf_tangent_r": tangent_r,
        "trf_speed_ratio": ratio,
        "trf_index_deg": index_deg,
        "trf_deck_r": deck_r,
        "trf_plate_overlap": overlap,
        "trf_leg_clash": legs,
        "trf_star_pcd_r": star_pcd_r,
        "trf_star_outer_r": star_outer_r,
        "trf_star_outer_r_eff": r_eff,
        "trf_star_pocket_r": pocket_r,
        "trf_star_pockets": int(p["star_pockets"]),
        "trf_star_t": star_t,
        "trf_star_base_z": star_base_z,
        "trf_star_top_z": star_top_z,
        "trf_star_mouth": star_mouth,
        "trf_pocket_plate_mouth": pp_mouth,
        "trf_pocket_plate_z": (pp_base_z, pp_top_z),
        "trf_hub_d": hub_d,
        "trf_hub_h": hub_h,
        "trf_shaft_d": shaft_d,
        "trf_shaft_top_z": shaft_top_z,
        "trf_shaft_base_z": shaft_base_z,
        "trf_bearing_d": bearing_d,
        "trf_bearing_bore": bearing_bore,
        "trf_bearing_top_z": bearing_top_z,
        "trf_ped_post_r": ped_post_r,
        "trf_ped_at_r": ped_at_r,
        "trf_ped_foot_d": ped_foot_d,
        "trf_ped_foot_t": ped_foot_t,
        "trf_ped_arm_t": ped_arm_t,
        "trf_ped_arm_w": ped_arm_w,
        "trf_ped_arm_base_z": ped_arm_base_z,
        "trf_ped_arm_top_z": ped_arm_top_z,
        # --- 平歯車。駆動を組む側はここを読む -------------------------------
        # 星車側 phi225（= 星車のボトルピッチ円）/ テーブル側 phi450（=
        # テーブルのボトルピッチ円）。中心距離 337.5 で 2 つのピッチ円が接する
        "trf_gear_pcd": 2.0 * gear_pcd_r,
        "trf_gear_pcd_r": gear_pcd_r,
        "trf_gear_teeth": gear_teeth,
        "trf_gear_module": gear_module,
        "trf_gear_tip_d": 2.0 * gear_tip_r,
        "trf_gear_root_d": 2.0 * gear_root_r,
        "trf_gear_face": gear_face,
        "trf_gear_base_z": gear_base_z,
        "trf_gear_top_z": gear_top_z,
        "trf_gear_boss_d": gear_boss_d,
        "trf_gear_boss_h": gear_boss_h,
        "trf_gear_boss_top_z": gear_boss_top_z,
        "trf_gear_center_dist": center_r,
        "trf_drive_gear_pcd": 2.0 * drive_gear_pcd_r,
        "trf_drive_gear_teeth": int(round(gear_teeth * ratio)),
        "trf_drive_gear_tip_d": 2.0 * (drive_gear_pcd_r
                                       + gear_module * p["gear_addendum_x_module"]),
        "trf_drive_gear_root_d": 2.0 * (drive_gear_pcd_r
                                        - gear_module * p["gear_dedendum_x_module"]),
        "trf_leg_phase_deg": leg_phase,
        "trf_deck_bottom_z": deck_bottom,
        "trf_dead_ri": dead_ri,
        "trf_dead_ro": dead_ro,
        "trf_dead_base_z": dead_base_z,
        "trf_dead_t": dead_t,
        "trf_dead_arcs": tuple(dead_arcs),
        "trf_dead_apron_arcs": tuple(apron),
        "trf_dead_apron_ri": dead_table_clear,
        "trf_dead_apron_ro": apron_ro,
        "trf_dead_stands": tuple(stands),
        "trf_dead_stand_r": stand_r,
        "trf_dead_stand_at_r": stand_at_r,
        "trf_dead_lug_ri": dead_lug_ri,
        "trf_dead_lug_ro": dead_lug_ro,
        "trf_dead_lug_deg": dead_lug_deg,
        "trf_dead_lug_w": stay_r * p["dead_lug_w_x_stay_r"],
        "trf_crescent_ri": cre_ri,
        "trf_crescent_ro": cre_ro,
        "trf_crescent_base_z": cre_base_z,
        "trf_crescent_h": cre_h,
        "trf_crescent_arcs": tuple(cre_arcs),
        "trf_crescent_trim_deg": (d_conv, d_table),
        "trf_crescent_conv_limit_deg": d_conv_max,
        "trf_crescent_leads": tuple(leads),
        "trf_crescent_lead_len": lead_len,
        "trf_crescent_lead_off": lead_off,
        "trf_crescent_stays": tuple(stays),
        "trf_crescent_stay_r": stay_r,
        "trf_crescent_stay_at_r": stay_at_r,
        # 床から立てる柱は無くなった。相手側が「床柱の太さ」で逃げを見ている
        # ところがあるので、キーは同じ棒の半径で残す（kind="floor" の棒は
        # 1 本も出さないので、その判定はどのみち空回りする）
        "trf_crescent_stand_r": stay_r,
        "trf_ring_ri": ring_ri,
        "trf_ring_ro": ring_ro,
        "trf_ring_base_z": ring_base_z,
        "trf_ring_h": ring_h,
        "trf_ring_arcs": tuple(ring_arcs),
        "trf_ring_notches": tuple(u.get("ring_notch_deg") for u in units),
        "trf_ring_posts": tuple(ring_posts),
        "trf_ring_post_r": ring_post_r,
        "trf_ring_post_at_r": ring_post_at_r,
        "trf_ring_post_foot_d": ring_post_r * p["ring_post_foot_d_x_post_r"],
        "trf_ring_post_foot_t": plate_t * p["ring_post_foot_t_x_plate_t"],
        "trf_ring_arm_z": ring_arm_z,
        "trf_ring_arm_t": plate_t * p["ring_arm_t_x_plate_t"],
        "trf_ring_arm_w": ring_post_r * p["ring_arm_w_x_post_r"],
        "trf_ring_unsupported": tuple(unsupported_ring),
        "trf_star_bottles": tuple(star_bottles),
        "trf_units": tuple(units),
    }


def _mouth_width(outer_r, pcd_r, pocket_r):
    """外周円とポケット円の交点から出る、切り欠きの開口幅。

        x = (R^2 - p^2 + c^2) / (2c)      口の幅 = 2 sqrt(R^2 - x^2)
    """
    x = (outer_r ** 2 - pocket_r ** 2 + pcd_r ** 2) / (2.0 * pcd_r)
    return 2.0 * float(np.sqrt(max(outer_r ** 2 - x * x, 0.0)))


# --------------------------------------------------------------------------
# 組み立て
# --------------------------------------------------------------------------
def _gear_profile(lay: dict):
    """星車軸の平歯車の胴の外形（r, z）。歯底円までの円板と締めボス。

    軸から軸まで 1 本でたどるので、回すと 1 つの閉じた塊になる。歯はこの外に
    1 枚ずつ角柱で足す（`_gear_tooth_meshes`）。
    """
    root_r = lay["trf_gear_root_d"] / 2.0
    z0 = lay["trf_gear_base_z"]
    z1 = lay["trf_gear_top_z"]
    boss_r = lay["trf_gear_boss_d"] / 2.0
    return [(0.0, z0), (root_r, z0), (root_r, z1), (boss_r, z1),
            (boss_r, z1 + lay["trf_gear_boss_h"]), (0.0, z1 + lay["trf_gear_boss_h"])]


def _tooth_half_angle(pitch_r, teeth, backlash, pressure_rad, radius):
    """歯の半角 [rad]。ピッチ円で歯厚がピッチの半分、外へ出るほど細くなる。

    歯すじを圧力角ぶん傾けた台形で近似する。インボリュートは刻まない。
    この縮尺では歯 1 枚が 1 画素前後なので、噛み合いが成立する幅と逃げが
    出ていれば足りる。テーブル側と同じ式でなければ噛み合わないので、
    諸元（モジュール・歯先歯底・逃げ・圧力角）は必ず揃える。
    """
    half_at_pitch = (np.pi / (2.0 * teeth)) * (1.0 - backlash)
    return half_at_pitch - (radius - pitch_r) * np.tan(pressure_rad) / pitch_r


def _gear_spec(lay: dict) -> dict:
    """噛み合う 2 枚の諸元。星車側は自分の値、テーブル側は駆動側が出した値。"""
    p = PROPORTION
    module = float(lay["trf_gear_module"])
    return dict(
        module=module,
        addendum=module * p["gear_addendum_x_module"],
        dedendum=module * p["gear_dedendum_x_module"],
        backlash=float(p["gear_backlash"]),
        pressure_rad=float(np.radians(p["gear_pressure_deg"])),
        pitch_r_star=float(lay["trf_gear_pcd_r"]),
        teeth_star=int(lay["trf_gear_teeth"]),
        pitch_r_table=float(lay["trf_drive_gear_pcd"]) / 2.0,
        teeth_table=int(lay["trf_drive_gear_teeth"]),
        ratio=float(lay["trf_speed_ratio"]),
        table_phase=float(np.radians(p["gear_table_phase_deg"])),
    )


def _gear_phase_rad(lay: dict, u: dict) -> float:
    """星車側の歯 1 枚の中心が向く角 [rad]。自転角 0 の姿勢で測る。

    噛み合いの条件は「相手の歯の中心が、こちらの歯溝の中心とピッチ点で
    出会う」こと。2 つのピッチ円は滑らずに転がるので、ピッチ点からの弧長で
    合わせればよい。向きは反転するので符号が入れ替わる。

        テーブル側の歯の中心がピッチ点から弧長 +a のところにある
            -> 星車側の歯溝の中心は弧長 -a のところに来る

    工程角がピッチ角の整数倍とは限らないので、半ピッチずらすだけでは合わない。
    ずれを実際に測ってから移す。テーブル側は歯 1 枚の中心を世界角 0 に置いて
    あるものとして解く（駆動側の決め）。
    """
    gs = _gear_spec(lay)
    axis = np.radians(float(u["angle_deg"]))          # テーブル中心から見た向き
    p1 = 2.0 * np.pi / gs["teeth_table"]
    p2 = 2.0 * np.pi / gs["teeth_star"]
    k = round((axis - gs["table_phase"]) / p1)
    delta = (gs["table_phase"] + k * p1) - axis
    space = axis + np.pi - gs["ratio"] * delta        # 歯溝の中心が向く角
    return float(space + p2 / 2.0)                    # 歯溝の中心 -> 歯の中心


_WARNED: set = set()


def _warn_once(key: str, text: str) -> None:
    """同じ警告を何度も出さない。相手のキーが欠けたことは 1 回だけ言う。"""
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"[trf] {text}", file=sys.stderr)


def _table_tooth_bite(lay: dict) -> float:
    """テーブル側の歯が、付け根を胴へ食い込ませている量 [mm]。

    歯面は「歯底円 - この量」から歯先へ引いた弦なので、すき間を測るには
    この値が要る。星車側の歯先の角がここに近づくので、効き量は小さくない
    （0.4 で 0.11 mm 変わる）。駆動側がまだ lay に出していないので、無ければ
    0 で（＝歯底円から引いた弦として）測る。そのぶんすき間は狭めに出る。
    絵そのものを切って測る `_mesh_gap_measured()` のほうが正典。
    """
    if "drv_gear_tooth_bite" in lay:
        return float(lay["drv_gear_tooth_bite"])
    _warn_once("drv_gear_tooth_bite",
               "drv_gear_tooth_bite が無いので、テーブル側の歯の付け根は"
               "歯底円ちょうどとして噛み合いのすき間を測る（狭めに出る）。"
               "駆動側に出してもらうこと")
    return 0.0


def _point_segment_dist(pts, seg_a, seg_b):
    """点の並び pts と、線分の並び (seg_a -> seg_b) の距離の行列。

    戻りは形 (len(pts), len(seg_a))。総当たりで出す。歯車 1 枚の輪郭は
    数百点なので、これで足りる。
    """
    ab = seg_b - seg_a                                   # (m, 2)
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom > 0.0, denom, 1.0)
    ap = pts[:, None, :] - seg_a[None, :, :]             # (n, m, 2)
    t = np.clip(np.einsum("nmk,mk->nm", ap, ab) / denom, 0.0, 1.0)
    foot = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = pts[:, None, :] - foot
    return np.hypot(d[:, :, 0], d[:, :, 1])


def _segments_cross(a0, a1, b0, b1) -> bool:
    """線分の並び a と b が 1 組でも交わるか。総当たりで見る。"""
    def side(p, q, r):
        return ((q[:, None, 0] - p[:, None, 0]) * (r[None, :, 1] - p[:, None, 1])
                - (q[:, None, 1] - p[:, None, 1]) * (r[None, :, 0] - p[:, None, 0]))

    d1 = side(a0, a1, b0)
    d2 = side(a0, a1, b1)
    d3 = side(b0, b1, a0).T
    d4 = side(b0, b1, a1).T
    return bool((((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))).any())


def _edges_gap(a0, a1, b0, b1) -> float:
    """線分の並び a と b のあいだの、いちばん短い距離 [mm]。

    2 つの図形が離れているとき、最短距離は必ず「片方の端点と、もう片方の
    線分」の組で実現する（平行な辺どうしで並ぶ場合も、その値は端点と辺の
    距離に等しい）。だから両向きの点対線分だけを見れば厳密に出る。
    """
    return float(min(_point_segment_dist(a0, b0, b1).min(),
                     _point_segment_dist(a1, b0, b1).min(),
                     _point_segment_dist(b0, a0, a1).min(),
                     _point_segment_dist(b1, a0, a1).min()))


def _drawn_gear_edges(pitch_r, teeth, add, ded, backlash, pressure_rad,
                      phase_rad, bite, center=(0.0, 0.0), root_steps=4):
    """**実際に描いてある**歯車の断面の輪郭を、線分の並びで返す。

    描いているのは「歯底円までの胴（丸）」と「1 枚ずつの角柱の歯」の重ね合わせ
    で、歯の付け根は胴に食い込ませてある。歯面は (歯底円 - bite, 歯底半角) と
    (歯先円, 歯先半角) を結ぶ弦で、歯底円から引いた弦とは傾きがわずかに違う。
    すき間を測る相手は絵に出ているほうでなければならないので、ここは押し出しに
    渡している多角形（`_gear_tooth_meshes` と同じ式）で組む。
    """
    r_tip = pitch_r + add
    r_root = pitch_r - ded
    psi_tip = _tooth_half_angle(pitch_r, teeth, backlash, pressure_rad, r_tip)
    psi_root = _tooth_half_angle(pitch_r, teeth, backlash, pressure_rad, r_root)
    pitch_angle = 2.0 * np.pi / int(teeth)
    if psi_tip <= 0.0 or psi_root >= pitch_angle / 2.0:
        raise ValueError(
            f"歯が成立しない: 歯数 {teeth} / 歯先半角 {np.degrees(psi_tip):.3f}deg /"
            f" 歯底半角 {np.degrees(psi_root):.3f}deg / ピッチ半角 "
            f"{np.degrees(pitch_angle / 2.0):.3f}deg")
    r_base = r_root - float(bite)
    c = np.asarray(center, dtype=float)
    a, b = [], []
    for i in range(int(teeth)):
        th = phase_rad + pitch_angle * i
        poly = np.array([(r * np.cos(ang), r * np.sin(ang)) for r, ang in
                         ((r_base, th - psi_root), (r_tip, th - psi_tip),
                          (r_tip, th + psi_tip), (r_base, th + psi_root))]) + c
        a.append(poly)
        b.append(np.roll(poly, -1, axis=0))
    # 胴の丸。押し出しの分割に合わせて多角形で持つ
    n = max(int(teeth) * int(root_steps), 64)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.column_stack([r_root * np.cos(th), r_root * np.sin(th)]) + c
    a.append(ring)
    b.append(np.roll(ring, -1, axis=0))
    return np.vstack(a), np.vstack(b)


def _mesh_clearance(lay: dict, spin_rad=0.0) -> dict:
    """噛み合っている 2 枚の歯のあいだの最小すき間 [mm]。役目 -> すき間。

    負なら歯が交わっている（その場合の値の大きさは当てにならない。0 に近い）。

    以前はここで、片方の外形の点をもう片方の中心から見た極角に直し、その角で
    相手の外形半径を内挿して**半径方向の差**を測っていた。歯面は半径方向から
    寝ているので、半径方向の差は面と面の最短距離より必ず大きく出る。この寸法
    では約 1.9 倍の過大評価で、+1.527 と出ていた。さらに、測っていた輪郭は
    歯底円から歯先へ引いた弦で、実際に押し出している歯（歯底円より bite だけ
    内から引いた弦）とは歯厚がわずかに違う。いまは描いてある断面どうしの
    厳密な最短距離を測る。
    """
    gs = _gear_spec(lay)
    tbl = _drawn_gear_edges(
        gs["pitch_r_table"], gs["teeth_table"], gs["addendum"], gs["dedendum"],
        gs["backlash"], gs["pressure_rad"],
        gs["table_phase"] - float(spin_rad) / lay["trf_speed_ratio"],
        _table_tooth_bite(lay))
    out = {}
    for u in lay["trf_units"]:
        star = _drawn_gear_edges(
            gs["pitch_r_star"], gs["teeth_star"], gs["addendum"], gs["dedendum"],
            gs["backlash"], gs["pressure_rad"],
            _gear_phase_rad(lay, u) + float(spin_rad),
            gs["module"] * PROPORTION["gear_tooth_bite_x_module"], u["center"])
        gap = _edges_gap(star[0], star[1], tbl[0], tbl[1])
        if _segments_cross(star[0], star[1], tbl[0], tbl[1]):
            gap = -gap
        out[u["role"]] = float(gap)
    return out


def _gear_tooth_meshes(lay: dict, u: dict) -> list:
    """歯を 1 枚ずつ角柱にして並べる。戻りは PolyData の並び。

    外形をまるごと押し出すと中実の円板になるので、胴とは別に足す。付け根は
    歯底円より少し内へ入れて胴と重ねる（胴の外周は円、歯の付け根は弦なので、
    突き合わせると隙間が出る）。ブール演算は使わない。
    """
    gs = _gear_spec(lay)
    r_tip = gs["pitch_r_star"] + gs["addendum"]
    r_root = gs["pitch_r_star"] - gs["dedendum"]
    psi_tip = _tooth_half_angle(gs["pitch_r_star"], gs["teeth_star"],
                                gs["backlash"], gs["pressure_rad"], r_tip)
    psi_root = _tooth_half_angle(gs["pitch_r_star"], gs["teeth_star"],
                                 gs["backlash"], gs["pressure_rad"], r_root)
    r_base = r_root - gs["module"] * PROPORTION["gear_tooth_bite_x_module"]
    phase = _gear_phase_rad(lay, u)
    pitch_angle = 2.0 * np.pi / gs["teeth_star"]
    c = np.asarray(u["center"], dtype=float)
    out = []
    for i in range(int(gs["teeth_star"])):
        th = phase + pitch_angle * i
        poly = [(r_base, th - psi_root), (r_tip, th - psi_tip),
                (r_tip, th + psi_tip), (r_base, th + psi_root)]
        pts = np.array([(r * np.cos(a), r * np.sin(a)) for r, a in poly]) + c
        out.append(parts.extrude_polygon(pts, lay["trf_gear_face"],
                                         base_z=lay["trf_gear_base_z"]))
    return out


def _fill_level(params: dict, lay: dict) -> float:
    """満量のときの液深 [mm]。MODEL.md の h = V / (pi R^2)。胴で頭打ちにする。"""
    r = lay["liquid_r"]
    h = float(params["fill"]["target_volume_mL"]) * 1000.0 / (np.pi * r * r)
    return float(min(h, lay["body_h"]))


def build(params: dict, lay: dict) -> dict:
    """スターホイール 2 台・駆動・渡し板・ガイド・ポケットのボトルを組む。

    自転する群は自転角 0 の姿勢で組む（テーブルと同じ規約。あとから
    SPIN_CENTERS まわりの回転を当てる）。
    """
    b = params["bottle"]
    groups = {name: [] for name, _ in DRAW_ORDER}
    level = _fill_level(params, lay)

    for u in lay["trf_units"]:
        cx, cy = u["center"]
        role = u["role"]
        at = parts.transform_matrix(translate=(cx, cy, 0.0))
        mat = parts.transform_matrix(translate=(cx, cy, 0.0),
                                     rot_z_deg=u["phase_deg"])
        spin = groups[_star_group(role)]

        # --- 円盤とハブ ---------------------------------------------------
        sw = parts.star_wheel(outer_d=2.0 * lay["trf_star_outer_r"],
                              pockets=lay["trf_star_pockets"],
                              pocket_r=lay["trf_star_pocket_r"],
                              pcd=2.0 * lay["trf_star_pcd_r"],
                              thickness=lay["trf_star_t"],
                              base_z=lay["trf_star_base_z"],
                              hub_d=lay["trf_hub_d"], hub_h=lay["trf_hub_h"],
                              shaft_d=lay["trf_shaft_d"], shaft_h=1.0,
                              matrix=mat)
        spin += [sw["disc"], sw["hub"]]

        # --- 軸。ハブの上から天板の上面まで一本で通す。天板は貫かない ------
        spin.append(
            parts.cylinder(lay["trf_shaft_d"] / 2.0,
                           lay["trf_shaft_top_z"] - lay["trf_shaft_base_z"],
                           base_z=lay["trf_shaft_base_z"], matrix=at))

        # --- 天板の上の平歯車。テーブル出力側と噛み合わせる ----------------
        # 胴（歯底円までの円板と締めボス）を 1 本の輪郭で回し、その外へ歯を
        # 1 枚ずつ足す。歯先円の和が中心距離を 8 mm 上回るので、片方でも
        # 平らな円板だと必ず食い込む。歯どうしが互いの歯溝に入って成立する。
        # 星車板とは別の群に入れる。歯先円 phi233 と星車の外径 phi235 が
        # ほぼ同じなので、同じ材質だと真上から星車の縁に歯が生えて見える
        gear = groups[_star_group(role, "_gear")]
        gear.append(parts.revolve(_gear_profile(lay), matrix=at))
        gear += _gear_tooth_meshes(lay, u)

        # --- 門型の軸受台。歯車を跨いで天板の上で受ける --------------------
        # 柱はボトルが通らない側に立て、歯車の歯先円の外に置く。腕で歯車を
        # 跨いで軸の上まで渡し、その上に軸受箱を載せる
        px, py = u["ped_xy"]
        pm = parts.transform_matrix(translate=(px, py, 0.0))
        groups["trf_mount"].append(
            parts.cylinder(lay["trf_ped_foot_d"] / 2.0, lay["trf_ped_foot_t"],
                           base_z=lay["deck_top"], matrix=pm))
        groups["trf_mount"].append(
            parts.cylinder(lay["trf_ped_post_r"],
                           lay["trf_ped_arm_top_z"] - lay["deck_top"],
                           base_z=lay["deck_top"], matrix=pm))
        # 腕。柱の芯から軸受箱の胴まで 1 本の角材で渡す。軸の芯まで通すと
        # 軸そのものを串刺しにするので、箱の外面で止める
        r_in = lay["trf_bearing_d"] / 2.0 - lay["plate_t"] * 0.25
        r_out = lay["trf_ped_at_r"] + lay["trf_ped_post_r"]
        groups["trf_mount"].append(
            parts.box((r_out - r_in, lay["trf_ped_arm_w"], lay["trf_ped_arm_t"]),
                      center=(0.5 * (r_in + r_out), 0.0,
                              0.5 * (lay["trf_ped_arm_base_z"]
                                     + lay["trf_ped_arm_top_z"])),
                      matrix=parts.transform_matrix(
                          translate=(cx, cy, 0.0),
                          rot_z_deg=u["angle_deg"] + u["ped_delta_deg"])))
        # 軸受箱。軸の通る穴を空けた筒。中実にすると軸が丸ごと埋まる
        groups["trf_mount"].append(
            parts.tube(lay["trf_bearing_d"] / 2.0, lay["trf_bearing_bore"] / 2.0,
                       lay["trf_bearing_top_z"] - lay["trf_ped_arm_base_z"],
                       base_z=lay["trf_ped_arm_base_z"], matrix=at))

    # --- 渡し板とその短柱 ---------------------------------------------------
    for (cx, cy), a0, a1 in lay["trf_dead_arcs"]:
        groups["trf_plate"].append(
            parts.crescent_guide(lay["trf_dead_ri"], lay["trf_dead_ro"],
                                 a0, a1, lay["trf_dead_t"],
                                 base_z=lay["trf_dead_base_z"],
                                 matrix=parts.transform_matrix(
                                     translate=(cx, cy, 0.0))))
    # テーブル板の縁に沿うエプロン。こちらは機械の中心まわりの円弧板
    for a0, a1 in lay["trf_dead_apron_arcs"]:
        groups["trf_plate"].append(
            parts.crescent_guide(lay["trf_dead_apron_ri"],
                                 lay["trf_dead_apron_ro"], a0, a1,
                                 lay["trf_dead_t"],
                                 base_z=lay["trf_dead_base_z"]))
    for sx, sy in lay["trf_dead_stands"]:
        groups["trf_mount"].append(
            parts.cylinder(lay["trf_dead_stand_r"],
                           lay["trf_dead_base_z"] - lay["deck_top"],
                           base_z=lay["deck_top"],
                           matrix=parts.transform_matrix(translate=(sx, sy, 0.0))))

    # --- 三日月ガイド -------------------------------------------------------
    for (cx, cy), a0, a1 in lay["trf_crescent_arcs"]:
        groups["trf_guide"].append(
            parts.crescent_guide(lay["trf_crescent_ri"], lay["trf_crescent_ro"],
                                 a0, a1, lay["trf_crescent_h"],
                                 base_z=lay["trf_crescent_base_z"],
                                 matrix=parts.transform_matrix(
                                     translate=(cx, cy, 0.0))))

    # コンベア側に継ぐ直線レール。円弧の接点から接線方向へ伸ばして、
    # 搬送側のサイドガイドが立つ位置まで横の受けを繋ぐ。
    for ld in lay["trf_crescent_leads"]:
        groups["trf_guide"].append(
            parts.box((ld["length"], lay["trf_crescent_ro"] - lay["trf_crescent_ri"],
                       lay["trf_crescent_h"]),
                      center=(0.0, 0.0,
                              lay["trf_crescent_base_z"] + lay["trf_crescent_h"] / 2.0),
                      matrix=parts.transform_matrix(
                          translate=(ld["xy"][0], ld["xy"][1], 0.0),
                          rot_z_deg=ld["head_deg"])))

    # ガイドを留める棒。板の外面に当てた丸棒でガイドの上端まで通す。足元は
    # 天板の上か、天板から外れる位置では渡し板の耳。床からは立てない。
    cre_top_z = lay["trf_crescent_base_z"] + lay["trf_crescent_h"]
    dead_top_z = lay["trf_dead_base_z"] + lay["trf_dead_t"]
    for s in lay["trf_crescent_stays"]:
        m = parts.transform_matrix(translate=(s["xy"][0], s["xy"][1], 0.0))
        if s["kind"] == "plate":
            # 渡し板の外縁に耳を足して、その上に棒を立てる。耳は板と同じ厚みで
            # 同じ高さ。棒の芯は板の外縁より外なので、耳が無いと棒が宙に浮く
            groups["trf_plate"].append(
                parts.crescent_guide(
                    lay["trf_dead_lug_ri"], lay["trf_dead_lug_ro"],
                    s["local_deg"] - lay["trf_dead_lug_deg"],
                    s["local_deg"] + lay["trf_dead_lug_deg"],
                    lay["trf_dead_t"], base_z=lay["trf_dead_base_z"],
                    matrix=parts.transform_matrix(
                        translate=(s["center"][0], s["center"][1], 0.0))))
            rod_base_z = dead_top_z
        else:
            rod_base_z = lay["deck_top"]
        groups["trf_mount"].append(
            parts.cylinder(lay["trf_crescent_stay_r"], cre_top_z - rod_base_z,
                           base_z=rod_base_z, matrix=m))

    # --- テーブル外周のボトル押さえと、その支柱 ----------------------------
    for a0, a1 in lay["trf_ring_arcs"]:
        groups["trf_guide"].append(
            parts.crescent_guide(lay["trf_ring_ri"], lay["trf_ring_ro"],
                                 a0, a1, lay["trf_ring_h"],
                                 base_z=lay["trf_ring_base_z"]))

    for px, py in lay["trf_ring_posts"]:
        m = parts.transform_matrix(translate=(px, py, 0.0))
        groups["trf_mount"].append(
            parts.cylinder(lay["trf_ring_post_foot_d"] / 2.0,
                           lay["trf_ring_post_foot_t"],
                           base_z=lay["deck_top"], matrix=m))
        groups["trf_mount"].append(
            parts.cylinder(lay["trf_ring_post_r"],
                           lay["trf_ring_arm_z"] + lay["trf_ring_arm_t"] / 2.0
                           - lay["deck_top"],
                           base_z=lay["deck_top"], matrix=m))
        # リングの外面へ出す腕。支柱の芯からリング外面までを 1 本の角材で渡す
        th = float(np.degrees(np.arctan2(py, px)))
        arm_len = lay["trf_ring_post_at_r"] - lay["trf_ring_ro"]
        groups["trf_mount"].append(
            parts.box((arm_len, lay["trf_ring_arm_w"], lay["trf_ring_arm_t"]),
                      center=(lay["trf_ring_ro"] + arm_len / 2.0, 0.0,
                              lay["trf_ring_arm_z"]),
                      matrix=parts.transform_matrix(rot_z_deg=th)))

    # --- 星車のポケットに乗るボトル ----------------------------------------
    # 自転する群に入れる（星車と一緒に回る）。供給側は空、排出側は満量。
    for role, bx, by in lay["trf_star_bottles"]:
        m = parts.transform_matrix(translate=(bx, by, 0.0))
        groups[_star_group(role, "_bottle")].append(
            parts.bottle(inner_diameter=b["inner_diameter_mm"],
                         body_height=b["body_height_mm"],
                         shoulder_height=b["shoulder_height_mm"],
                         neck_diameter=b["neck_diameter_mm"],
                         neck_height=b["neck_height_mm"],
                         wall_thickness=b["wall_thickness_mm"],
                         base_z=lay["table_top"], matrix=m))
        groups[_star_group(role, "_glass")].append(
            parts.bottle_edges(inner_diameter=b["inner_diameter_mm"],
                               body_height=b["body_height_mm"],
                               shoulder_height=b["shoulder_height_mm"],
                               neck_diameter=b["neck_diameter_mm"],
                               neck_height=b["neck_height_mm"],
                               wall_thickness=b["wall_thickness_mm"],
                               base_z=lay["table_top"], matrix=m))
        if role == "discharge":
            groups[_star_group(role, "_liquid")].append(
                parts.liquid(lay["liquid_r"], level,
                             base_z=lay["table_top"] + b["wall_thickness_mm"],
                             matrix=m))

    return {k: parts.merge(v) for k, v in groups.items() if v}


# --------------------------------------------------------------------------
# 確認用。scene 側の機械に自分のメッシュを足して焼く
# --------------------------------------------------------------------------
def _report(lay: dict) -> None:
    print(f"スターホイール 中心半径 {lay['trf_center_r']:.2f} / "
          f"外半径 {lay['trf_star_outer_r']:.2f}"
          f"（有効 {lay['trf_star_outer_r_eff']:.2f}） / "
          f"ポケット R{lay['trf_star_pocket_r']:.2f} / "
          f"PCD {2 * lay['trf_star_pcd_r']:.1f}")
    print(f"口の幅  スターホイール {lay['trf_star_mouth']:.2f} / "
          f"ポケット板 {lay['trf_pocket_plate_mouth']:.2f}")
    print(f"z 範囲  スターホイール板 {lay['trf_star_base_z']:.1f}〜"
          f"{lay['trf_star_top_z']:.1f} / "
          f"ポケット板 {lay['trf_pocket_plate_z'][0]:.1f}〜"
          f"{lay['trf_pocket_plate_z'][1]:.1f}")
    print(f"軸 {lay['trf_shaft_base_z']:.1f}〜{lay['trf_shaft_top_z']:.1f}"
          f"（天板上面 {lay['deck_top']:.1f} で止める。天板は貫かない） / "
          f"歯車 z {lay['trf_gear_base_z']:.1f}〜{lay['trf_gear_top_z']:.1f}"
          f"（天板上面 {lay['deck_top']:.1f}〜テーブル板下面 "
          f"{lay['table_base']:.1f} のあいだ） / "
          f"軸受箱 phi{lay['trf_bearing_d']:.1f} "
          f"{lay['trf_ped_arm_base_z']:.1f}〜{lay['trf_bearing_top_z']:.1f}")
    for u in lay["trf_units"]:
        print(f"  {u['role']:9s} 門型の柱 局所 {u['ped_delta_deg']:+.1f}deg・"
              f"軸から {lay['trf_ped_at_r']:.1f}（歯先円 "
              f"{lay['trf_gear_tip_d'] / 2.0:.1f} の外へ "
              f"{lay['trf_ped_at_r'] - lay['trf_gear_tip_d'] / 2.0 - lay['trf_ped_post_r']:.1f}） "
              f"世界 ({u['ped_xy'][0]:7.2f}, {u['ped_xy'][1]:7.2f}) "
              f"半径 {u['ped_world_r']:.1f}")
    print(f"速比 {lay['trf_speed_ratio']:.3f}"
          f"（テーブル {lay['trf_index_deg']:.1f}deg に対し星車 "
          f"{-lay['trf_speed_ratio'] * lay['trf_index_deg']:.1f}deg = 1 ポケット）")
    print(f"平歯車 星車側 ピッチ円 phi{lay['trf_gear_pcd']:.1f}"
          f"（歯数 {lay['trf_gear_teeth']} / モジュール {lay['trf_gear_module']:.2f}"
          f" / 歯先 phi{lay['trf_gear_tip_d']:.1f}"
          f" / 歯底 phi{lay['trf_gear_root_d']:.1f}"
          f" / 歯幅 {lay['trf_gear_face']:.1f}）")
    print(f"       テーブル側 ピッチ円 phi{lay['trf_drive_gear_pcd']:.1f}"
          f"（歯数 {lay['trf_drive_gear_teeth']} / "
          f"歯先 phi{lay['trf_drive_gear_tip_d']:.1f}） -> "
          f"ピッチ円半径の和 "
          f"{(lay['trf_gear_pcd'] + lay['trf_drive_gear_pcd']) / 2.0:.1f} = "
          f"中心距離 {lay['trf_gear_center_dist']:.1f}")
    for role, (hit, total, worst) in lay["trf_plate_overlap"].items():
        print(f"  {role:9s} 受け渡し点のボトルと板材の重なり {hit}/{total} 点"
              f"（テーブル角 0〜{lay['trf_index_deg']:.0f}deg を刻んで最悪 "
              f"{worst:.1f}deg）")
    print(f"渡し板 内半径 {lay['trf_dead_ri']:.2f} / 外半径 {lay['trf_dead_ro']:.2f} / "
          f"z {lay['trf_dead_base_z']:.1f}〜"
          f"{lay['trf_dead_base_z'] + lay['trf_dead_t']:.1f} / "
          f"短柱 {len(lay['trf_dead_stands'])} 本")
    for u in lay["trf_units"]:
        a0, a1 = u["dead_arc"]
        w0 = _world_r(lay["trf_center_r"], lay["trf_star_pcd_r"],
                      u["dead_delta_deg"][0])
        w1 = _world_r(lay["trf_center_r"], lay["trf_star_pcd_r"],
                      u["dead_delta_deg"][1])
        print(f"  {u['role']:9s} 局所 {a0:.1f}〜{a1:.1f}deg "
              f"（ボトル中心 世界半径 {min(w0, w1):.1f}〜{max(w0, w1):.1f}）")
    print(f"  テーブル板の縁に沿うエプロン 半径 {lay['trf_dead_apron_ri']:.1f}〜"
          f"{lay['trf_dead_apron_ro']:.1f} / "
          + " / ".join(f"世界角 {a0:.1f}〜{a1:.1f}deg"
                       for a0, a1 in lay["trf_dead_apron_arcs"]))
    print(f"三日月ガイド 内半径 {lay['trf_crescent_ri']:.2f} / "
          f"外半径 {lay['trf_crescent_ro']:.2f} / "
          f"z {lay['trf_crescent_base_z']:.1f}〜"
          f"{lay['trf_crescent_base_z'] + lay['trf_crescent_h']:.1f} / "
          f"コンベア側は接点で切る（回せる限界 "
          f"{lay['trf_crescent_conv_limit_deg']:.1f}deg）・テーブル側 "
          f"{lay['trf_crescent_trim_deg'][1]:.1f}deg")
    for u in lay["trf_units"]:
        print(f"  {u['role']:9s} {u['angle_deg']:5.1f}deg "
              f"中心 ({u['center'][0]:7.2f}, {u['center'][1]:7.2f}) "
              f"位相 {u['phase_deg']:5.1f}deg "
              f"ボトル {u['bottle_steps']} ポケット目 "
              f"ガイド " + " / ".join(
                  f"{a0:.1f}〜{a1:.1f}（{a1 - a0:.1f}deg）"
                  for a0, a1 in u["guide_arcs"]))
    print(f"三日月に継ぐ直線レール 長さ {lay['trf_crescent_lead_len']:.1f} / "
          f"通り道の芯からの偏り {lay['trf_crescent_lead_off']:.2f}"
          f"（ボトル外周まで {lay['trf_crescent_lead_off'] - lay['bottle_r'] - (lay['trf_crescent_ro'] - lay['trf_crescent_ri']) / 2.0:.2f}）")
    for ld in lay["trf_crescent_leads"]:
        print(f"  接点 ({ld['contact'][0]:7.2f}, {ld['contact'][1]:7.2f}) から "
              f"{ld['head_deg'] % 360.0:5.1f}deg の向きへ")
    print(f"外周ガイド 内半径 {lay['trf_ring_ri']:.2f} / "
          f"外半径 {lay['trf_ring_ro']:.2f} / "
          f"z {lay['trf_ring_base_z']:.1f}〜"
          f"{lay['trf_ring_base_z'] + lay['trf_ring_h']:.1f} / "
          f"支柱 {len(lay['trf_ring_posts'])} 本（半径 "
          f"{lay['trf_ring_post_at_r']:.1f}・腕 z {lay['trf_ring_arm_z']:.1f}）")
    notches = [w for w in lay["trf_ring_notches"] if w is not None]
    for w in notches:
        print(f"  切り欠き {w[0]:.1f}〜{w[1]:.1f}deg（{w[1] - w[0]:.1f}deg）")
    if len(notches) == 2:
        # 供給と排出が 2 ステーション離れているので、2 つの切り欠きは融合しない。
        # 融合していれば残る円弧は 2 本になるが、分かれていれば 3 本以上になる
        a, b = sorted(w[1] % 360.0 for w in notches), sorted(w[0] % 360.0
                                                             for w in notches)
        print(f"  切り欠きは 2 つ。あいだに残る帯 "
              f"{min(a):.1f}〜{max(b):.1f}deg（{max(b) - min(a):.1f}deg）")
    for a0, a1 in lay["trf_ring_arcs"]:
        print(f"  円弧 {a0:.1f}〜{a1:.1f}deg（{a1 - a0:.1f}deg）")
    for px, py in lay["trf_ring_posts"]:
        print(f"  支柱 世界角 {np.degrees(np.arctan2(py, px)) % 360.0:.1f}deg")


def _carried_local_deg(lay: dict, u: dict, s: float):
    """割出しの進み s（0..1）のとき、その星車が抱えているボトルの局所角。

    ポケットは 1 割出しで 1 ポケットぶん CW へ回る。埋まっている番号は
    bottle_steps がそのまま持っている。
    """
    n = int(lay["trf_star_pockets"])
    return [u["phase_deg"] - u["swept_sign"] * 360.0 * k / n - 360.0 * s / n
            for k in u["bottle_steps"]]


def _cross_bottle_gap(lay: dict, steps=241):
    """供給と排出の星車が抱えるボトルどうしの、いちばん近い中心間距離。

    戻りは (距離, そのときの割出しの進み, 供給の局所角, 排出の局所角)。
    2 台のピッチ円は離れているが、抱えたボトルは半径ぶん外へ出る。
    """
    if len(lay["trf_units"]) < 2:
        return None
    ua, ub = lay["trf_units"][0], lay["trf_units"][1]
    ca = np.asarray(ua["center"], dtype=float)
    cb = np.asarray(ub["center"], dtype=float)
    r = lay["trf_star_pcd_r"]
    best = None
    for s in np.linspace(0.0, 1.0, steps):
        for a in _carried_local_deg(lay, ua, s):
            pa = ca + r * _unit(a)
            for b in _carried_local_deg(lay, ub, s):
                d = float(np.hypot(*(pa - cb - r * _unit(b))))
                if best is None or d < best[0]:
                    best = (d, float(s), float(a % 360.0), float(b % 360.0))
    return best


def _cross_plate_gap(lay: dict, steps=241):
    """星車が抱えるボトルと、相手の星車の円盤（外周円）の隙間。"""
    if len(lay["trf_units"]) < 2:
        return None
    r = lay["trf_star_pcd_r"]
    ro = lay["trf_star_outer_r"]
    br = lay["bottle_r"]
    worst = None
    for i in (0, 1):
        u = lay["trf_units"][i]
        other = np.asarray(lay["trf_units"][1 - i]["center"], dtype=float)
        c = np.asarray(u["center"], dtype=float)
        for s in np.linspace(0.0, 1.0, steps):
            for a in _carried_local_deg(lay, u, s):
                g = float(np.hypot(*(c + r * _unit(a) - other))) - ro - br
                worst = g if worst is None else min(worst, g)
    return worst


def _leg_phase_clash(lay: dict, phase_deg: float) -> float:
    """脚を phase_deg 振ったときの、歯車と脚のいちばん悪い食い込み [mm]。"""
    worst = -float("inf")
    for u in lay["trf_units"]:
        worst = max(worst, _leg_clash(lay, u["center"], lay["trf_gear_tip_d"] / 2.0,
                                      lay["trf_gear_base_z"], lay["trf_gear_top_z"],
                                      phase_deg))
    return float(worst)


def _leg_phase_needed(lay: dict, step=0.1) -> float:
    """歯車が脚に当たらなくなる、いちばん浅い振り角 [deg]。

    脚は 90deg ごとなので、振りは 0 から -45deg までを見れば足りる。
    """
    ph = 0.0
    while ph >= -45.0:
        if _leg_phase_clash(lay, ph) <= 0.0:
            return float(ph)
        ph -= step
    return -45.0


def _clearance_check(lay: dict) -> None:
    """絵を焼く前に、干渉しやすい組を数値で当たっておく。"""
    ok = True
    br = lay["bottle_r"]
    deck_r = _deck_r(lay)

    # 三日月ガイドとテーブル上のボトル
    r_bottle_out = lay["pitch_r"] + br
    worst = None
    for (cx, cy), a0, a1 in lay["trf_crescent_arcs"]:
        for t in np.linspace(a0, a1, 121):
            for rr in (lay["trf_crescent_ri"], lay["trf_crescent_ro"]):
                q = np.array([cx, cy]) + rr * _unit(t)
                w = float(np.hypot(*q)) - r_bottle_out
                worst = w if worst is None else min(worst, w)
    print(f"三日月ガイド とテーブル上ボトル外周の隙間 最小 {worst:.1f} mm")
    ok &= worst > 0.0

    # 三日月ガイドと、そこに触れる外周ガイドの外面
    worst = None
    for (cx, cy), a0, a1 in lay["trf_crescent_arcs"]:
        for t in np.linspace(a0, a1, 121):
            q = np.array([cx, cy]) + lay["trf_crescent_ri"] * _unit(t)
            w = float(np.hypot(*q)) - lay["trf_ring_ro"]
            worst = w if worst is None else min(worst, w)
    print(f"三日月ガイド と外周ガイド外面の隙間 最小 {worst:.1f} mm")
    ok &= worst > 0.0

    # コンベア側に継いだ直線レール。通り道を塞いでいないか、星車の板に
    # 当たっていないか、円弧の端と繋がっているか
    half_t = 0.5 * (lay["trf_crescent_ro"] - lay["trf_crescent_ri"])
    lane_gap = lay["trf_crescent_lead_off"] - half_t - br
    print(f"直線レール 長さ {lay['trf_crescent_lead_len']:.1f} / "
          f"内面とボトル外周の隙間 {lane_gap:.2f} mm")
    ok &= lane_gap > 0.0
    mid_r = 0.5 * (lay["trf_crescent_ri"] + lay["trf_crescent_ro"])
    for ld in lay["trf_crescent_leads"]:
        head = _unit(ld["head_deg"])
        side = np.array([-head[1], head[0]])
        near = np.asarray(ld["xy"], dtype=float) - head * ld["length"] / 2.0
        worst_disc = None
        for s in np.linspace(0.0, ld["length"], 121):
            for sg in (-1.0, 1.0):
                q = near + head * s + side * sg * half_t
                for u in lay["trf_units"]:
                    d = (float(np.hypot(*(q - np.asarray(u["center"]))))
                         - lay["trf_star_outer_r"])
                    worst_disc = d if worst_disc is None else min(worst_disc, d)
        # 円弧のコンベア側の端（帯の芯）と、レールの起点が重なっているか
        u = next(q for q in lay["trf_units"]
                 if abs((q["angle_deg"] - 90.0 * q["swept_sign"] - ld["head_deg"])
                        % 360.0) < 1e-6)
        end = np.asarray(u["center"], dtype=float) + mid_r * _unit(u["angle_deg"])
        join = float(np.hypot(*(near - end)))
        print(f"  {ld['head_deg'] % 360.0:5.1f}deg へ / 星車板までの隙間 "
              f"最小 {worst_disc:.1f} mm / 円弧との継ぎ目のずれ {join:.3f} mm")
        ok &= worst_disc > 0.0 and join < 1e-6

    # 三日月ガイドどうし
    pts = []
    for (cx, cy), a0, a1 in lay["trf_crescent_arcs"]:
        t = np.linspace(a0, a1, 121)
        for rr in np.linspace(lay["trf_crescent_ri"], lay["trf_crescent_ro"], 3):
            p = np.column_stack([cx + rr * np.cos(np.radians(t)),
                                 cy + rr * np.sin(np.radians(t))])
            pts.append((cx, cy, p))
    gap = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if (pts[i][0], pts[i][1]) == (pts[j][0], pts[j][1]):
                continue
            d = np.hypot(pts[i][2][:, None, 0] - pts[j][2][None, :, 0],
                         pts[i][2][:, None, 1] - pts[j][2][None, :, 1]).min()
            gap = d if gap is None else min(gap, d)
    if gap is not None:
        print(f"三日月ガイドどうしの隙間 最小 {gap:.1f} mm")
        ok &= gap > 0.0

    # ガイドを留める丸棒。板の外面に当たっているか（貫通していないか）、
    # 天板の上に収まっているか、棒どうしが当たらないか
    sr = lay["trf_crescent_stay_r"]
    bite = lay["trf_crescent_stay_at_r"] - lay["trf_crescent_ro"]
    print(f"ガイドの丸棒 芯を板の外面から {bite:.1f} mm 外へ / 半径 {sr:.1f}"
          f" -> 内面まで残り {lay['trf_crescent_ro'] - lay['trf_crescent_ri'] - (sr - bite):.1f} mm")
    ok &= sr - bite < lay["trf_crescent_ro"] - lay["trf_crescent_ri"]
    stays = lay["trf_crescent_stays"]
    pr = sr
    print(f"ガイドの棒 {len(stays)} 本（床から立てるものは無い）")
    for s in stays:
        w = s["world_r"]
        if s["kind"] == "floor":
            print(f"  {s['role']:9s} 床から立つ棒が残っている 世界半径 {w:.1f}")
            ok = False
        elif s["kind"] == "plate":
            # 渡し板の耳の上。板の弧の中に収まっているか
            a0, a1 = next(q["dead_arc"] for q in lay["trf_units"]
                          if q["role"] == s["role"])
            t = s["local_deg"]
            print(f"  {s['role']:9s} 渡し板の耳から 世界半径 {w:.1f}"
                  f"（局所 {t:.1f}deg・板の弧 {a0:.1f}〜{a1:.1f}deg の内へ "
                  f"{min(t - a0, a1 - t):.1f}deg / 棒の下端 "
                  f"{lay['trf_dead_base_z'] + lay['trf_dead_t']:.1f}）")
            ok &= a0 < t < a1
        else:
            print(f"  {s['role']:9s} 天板から 世界半径 {w:.1f}"
                  f"（天板の縁まで {deck_r - w - sr:.1f} / "
                  f"回る板の縁まで {w - sr - lay['plate_r']:.1f}）")
            ok &= w + sr < deck_r and w - sr > lay["plate_r"]
    for i in range(len(stays)):
        for j in range(i + 1, len(stays)):
            g = (float(np.hypot(stays[i]["xy"][0] - stays[j]["xy"][0],
                                stays[i]["xy"][1] - stays[j]["xy"][1]))
                 - 2.0 * pr)
            ok &= g > 0.0
    # 耳。棒の座が耳に載っていること、耳が板と重なって繋がっていること
    print(f"渡し板の耳 半径 {lay['trf_dead_lug_ri']:.1f}〜"
          f"{lay['trf_dead_lug_ro']:.1f}（幅 {lay['trf_dead_lug_w']:.1f}・"
          f"板と同じ z {lay['trf_dead_base_z']:.1f}〜"
          f"{lay['trf_dead_base_z'] + lay['trf_dead_t']:.1f}） / "
          f"棒の芯 {lay['trf_crescent_stay_at_r']:.1f} は耳の縁の内へ "
          f"{lay['trf_dead_lug_ro'] - lay['trf_crescent_stay_at_r'] - sr:.1f} / "
          f"板と重なる幅 {lay['trf_dead_ro'] - lay['trf_dead_lug_ri']:.1f}")
    ok &= lay["trf_crescent_stay_at_r"] + sr <= lay["trf_dead_lug_ro"]
    ok &= lay["trf_dead_lug_ri"] < lay["trf_dead_ro"]
    for u in lay["trf_units"]:
        if u.get("guide_arcs_unsupported"):
            print(f"{u['role']}: 支えの付かないガイドが残る "
                  f"{u['guide_arcs_unsupported']}")
            ok = False

    # 渡し板。回る板と重ならないか、ボトルの底の下にあるか
    dead_top = lay["trf_dead_base_z"] + lay["trf_dead_t"]
    w_min = None
    for (cx, cy), a0, a1 in lay["trf_dead_arcs"]:
        for t in np.linspace(a0, a1, 181):
            for rr in (lay["trf_dead_ri"], lay["trf_dead_ro"]):
                q = np.array([cx, cy]) + rr * _unit(t)
                w = float(np.hypot(*q))
                w_min = w if w_min is None else min(w_min, w)
    print(f"渡し板 いちばん内側の世界半径 {w_min:.1f}（回るテーブル板 "
          f"{lay['plate_r']:.1f} / ポケット板 {lay['pocket_plate_r']:.1f}）")
    ok &= w_min > lay["plate_r"]
    print(f"渡し板のエプロン 半径 {lay['trf_dead_apron_ri']:.1f}〜"
          f"{lay['trf_dead_apron_ro']:.1f}（回るテーブル板の縁から "
          f"{lay['trf_dead_apron_ri'] - lay['plate_r']:.1f}） "
          + " / ".join(f"世界角 {a0:.1f}〜{a1:.1f}deg"
                       for a0, a1 in lay["trf_dead_apron_arcs"]))
    ok &= lay["trf_dead_apron_ri"] > lay["plate_r"]
    # エプロンと外周ガイドの支柱。支柱は天板から渡し板の高さを突き抜ける
    for px, py in lay["trf_ring_posts"]:
        w = float(np.hypot(px, py))
        if not (lay["trf_dead_apron_ri"] - lay["trf_ring_post_r"] < w
                < lay["trf_dead_apron_ro"] + lay["trf_ring_post_r"]):
            continue
        phi = float(np.degrees(np.arctan2(py, px)))
        for a0, a1 in lay["trf_dead_apron_arcs"]:
            if (phi - a0) % 360.0 <= (a1 - a0):
                print(f"外周ガイドの支柱（世界角 {phi % 360.0:.1f}deg・"
                      f"半径 {w:.1f}）がエプロンに刺さる")
                ok = False
    print(f"渡し板 上面 {dead_top:.1f} / テーブル上面 {lay['table_top']:.1f}")
    ok &= dead_top <= lay["table_top"]

    # 渡し板と搬送面。板は搬送面の手前の端（接点を通り、搬送の向きに直交する
    # 面）より手前に収まっていること。ここを世界半径の円で切っていたせいで
    # 板が接点の 50deg 手前で退き、ボトルの足元が空いていた
    lines = lay.get("cnv_lines")
    if not lines:
        _warn_once("cnv_lines",
                   "cnv_lines が無いので、搬送面の端は接点と接線から出す")
    for u in lay["trf_units"]:
        a = u["angle_deg"]
        contact = lay["trf_tangent_r"] * _unit(a)
        head = _unit(a - 90.0 * u["swept_sign"])       # 機外へ伸びる向き
        if lines:
            got = min(lines, key=lambda q, c=contact: float(
                np.hypot(*(np.asarray(q["contact"], dtype=float) - c))))
            contact = np.asarray(got["contact"], dtype=float)
            head = np.asarray(got["direction"], dtype=float)
        a0, a1 = u["dead_arc"]
        worst = None
        for t in np.linspace(a0, a1, 181):
            for rr in (lay["trf_dead_ri"], lay["trf_dead_ro"]):
                q = np.asarray(u["center"], dtype=float) + rr * _unit(t)
                s = float((q - contact) @ head)        # 搬送の向きの座標
                worst = s if worst is None else max(worst, s)
        print(f"  {u['role']:9s} 渡し板の端から搬送面の端まで {-worst:.1f} mm"
              f"（板 局所 {a0:.1f}〜{a1:.1f}deg / ボトル中心の通り道は"
              f" 接点まであと "
              f"{lay['trf_star_pcd_r'] * np.radians(min(abs(a1 - a), abs(a0 - a))):.1f} mm）")
        ok &= worst < 0.0

    # 外周ガイドの支柱
    print(f"外周ガイドの支柱 {len(lay['trf_ring_posts'])} 本 / "
          f"半径 {lay['trf_ring_post_at_r']:.1f}"
          f"（回る板の縁まで {lay['trf_ring_post_at_r'] - lay['trf_ring_post_r'] - lay['plate_r']:.1f} / "
          f"天板の縁まで {deck_r - lay['trf_ring_post_at_r'] - lay['trf_ring_post_r']:.1f}）")
    ok &= lay["trf_ring_post_at_r"] - lay["trf_ring_post_r"] > lay["plate_r"]
    ok &= lay["trf_ring_post_at_r"] + lay["trf_ring_post_r"] < deck_r
    if lay["trf_ring_unsupported"]:
        print(f"支えの付かない外周ガイドが残る: {lay['trf_ring_unsupported']}")
        ok = False

    # スターホイールの軸受台（門型）と、その足元
    print(f"軸受箱 外半径 {lay['trf_bearing_d'] / 2:.1f}（回るテーブル板まで "
          f"{lay['trf_center_r'] - lay['trf_bearing_d'] / 2 - lay['plate_r']:.1f}） / "
          f"z {lay['trf_ped_arm_base_z']:.1f}〜{lay['trf_bearing_top_z']:.1f}"
          f"（円盤下面 {lay['trf_star_base_z']:.1f} まで "
          f"{lay['trf_star_base_z'] - lay['trf_bearing_top_z']:.1f}）")
    ok &= lay["trf_center_r"] - lay["trf_bearing_d"] / 2.0 > lay["plate_r"]
    ok &= lay["trf_bearing_top_z"] < lay["trf_star_base_z"]
    pr = lay["trf_ped_post_r"]
    # 跨ぐ相手は歯車の上面ではなく、その上に出る締めボスの頭。回る側で
    # いちばん高いところを見ないと、固定側の腕と軸受箱がボスに埋まる
    boss_top = lay["trf_gear_boss_top_z"]
    print(f"門型の柱 半径 {pr:.1f} / 据付座 phi{lay['trf_ped_foot_d']:.1f} / "
          f"z {lay['deck_top']:.1f}〜{lay['trf_ped_arm_top_z']:.1f} / "
          f"腕 {lay['trf_ped_arm_base_z']:.1f}〜{lay['trf_ped_arm_top_z']:.1f}"
          f"（歯車上面 {lay['trf_gear_top_z']:.1f}・締めボス頭 {boss_top:.1f}"
          f"（phi{lay['trf_gear_boss_d']:.1f}・回る）を "
          f"{lay['trf_ped_arm_base_z'] - boss_top:.1f} 跨ぐ）")
    ok &= lay["trf_ped_arm_base_z"] > boss_top
    # 軸受箱も同じボスの上に載る。腕と同じ高さから始まるので同じ判定になるが、
    # 別々に書いておく（どちらかの高さを触ったときに片方だけ通るのを防ぐ）
    print(f"軸受箱の下端 {lay['trf_ped_arm_base_z']:.1f} / 締めボス頭 "
          f"{boss_top:.1f} -> 隙間 "
          f"{lay['trf_ped_arm_base_z'] - boss_top:.1f} mm")
    ok &= lay["trf_ped_arm_base_z"] > boss_top
    for u in lay["trf_units"]:
        w = u["ped_world_r"]
        print(f"  {u['role']:9s} 世界半径 {w:.1f}"
              f"（天板の縁まで {deck_r - w - lay['trf_ped_foot_d'] / 2.0:.1f} / "
              f"回る板の縁まで {w - pr - lay['plate_r']:.1f} / "
              f"歯先円まで {lay['trf_ped_at_r'] - pr - lay['trf_gear_tip_d'] / 2.0:.1f}）")
        ok &= w + lay["trf_ped_foot_d"] / 2.0 < deck_r
        ok &= w - pr > lay["plate_r"]
        ok &= lay["trf_ped_at_r"] - pr > lay["trf_gear_tip_d"] / 2.0
        # 柱と腕はボトルが通らない側に立てる。抱えたボトルと当たらないこと
        worst = None
        for s in np.linspace(0.0, 1.0, 121):
            for a in _carried_local_deg(lay, u, s):
                q = (np.asarray(u["center"], dtype=float)
                     + lay["trf_star_pcd_r"] * _unit(a))
                d = float(np.hypot(*(q - np.asarray(u["ped_xy"])))) - br - pr
                worst = d if worst is None else min(worst, d)
        print(f"            柱と抱えたボトルの隙間 最小 {worst:.1f} mm")
        ok &= worst > 0.0

    # 天板・テーブル板との高さの取り合い。軸が天板を貫いていないこと
    print(f"軸の下端 {lay['trf_shaft_base_z']:.1f} / 天板上面 {lay['deck_top']:.1f}"
          f" -> 貫通 {'なし' if lay['trf_shaft_base_z'] >= lay['deck_top'] else 'あり'}")
    ok &= lay["trf_shaft_base_z"] >= lay["deck_top"]
    print(f"歯車 z {lay['trf_gear_base_z']:.1f}〜{lay['trf_gear_top_z']:.1f}"
          f"（歯幅 {lay['trf_gear_face']:.1f}） 天板上面まで "
          f"{lay['trf_gear_base_z'] - lay['deck_top']:.1f} / テーブル板下面まで "
          f"{lay['table_base'] - lay['trf_gear_top_z']:.1f}")
    ok &= lay["trf_gear_base_z"] > lay["deck_top"]
    ok &= lay["trf_gear_top_z"] < lay["table_base"]
    # 歯車と渡し板・その短柱。歯車は軸まわり半径 116.5 を占める
    print(f"歯車上面 {lay['trf_gear_top_z']:.1f} / 渡し板下面 "
          f"{lay['trf_dead_base_z']:.1f} -> 隙間 "
          f"{lay['trf_dead_base_z'] - lay['trf_gear_top_z']:.1f} mm")
    ok &= lay["trf_dead_base_z"] > lay["trf_gear_top_z"]
    print(f"渡し板の短柱 軸から {lay['trf_dead_stand_at_r']:.1f}"
          f"（歯先円 {lay['trf_gear_tip_d'] / 2.0:.1f} の外へ "
          f"{lay['trf_dead_stand_at_r'] - lay['trf_dead_stand_r'] - lay['trf_gear_tip_d'] / 2.0:.1f} / "
          f"渡し板の外縁 {lay['trf_dead_ro']:.1f} の内へ "
          f"{lay['trf_dead_ro'] - lay['trf_dead_stand_at_r'] - lay['trf_dead_stand_r']:.1f}）")
    ok &= (lay["trf_dead_stand_at_r"] - lay["trf_dead_stand_r"]
           > lay["trf_gear_tip_d"] / 2.0)
    # 歯車と、天板の上に立っているほかの柱
    posts = ([(x, y, lay["trf_ring_post_r"]) for x, y in lay["trf_ring_posts"]]
             + [(s["xy"][0], s["xy"][1], lay["trf_crescent_stay_r"])
                for s in lay["trf_crescent_stays"]]
             + [(x, y, lay["trf_dead_stand_r"]) for x, y in lay["trf_dead_stands"]])
    worst = None
    for u in lay["trf_units"]:
        c = np.asarray(u["center"], dtype=float)
        for x, y, r in posts:
            d = float(np.hypot(x - c[0], y - c[1])) - r - lay["trf_gear_tip_d"] / 2.0
            worst = d if worst is None else min(worst, d)
    print(f"歯車の歯先円と天板の上の柱（外周ガイド {len(lay['trf_ring_posts'])} 本"
          f"・三日月 {len(lay['trf_crescent_stays'])} 本"
          f"・渡し板 {len(lay['trf_dead_stands'])} 本）の隙間 最小 {worst:.1f} mm")
    ok &= worst > 0.0

    # 2 台が抱えているボトルどうし。ピッチ円は 33.3 mm 離れているが、ボトルは
    # 半径 34 あるので、中心間が 68 を割ると胴がぶつかる
    got = _cross_bottle_gap(lay)
    if got is not None:
        d, s, a, bb = got
        print(f"供給と排出のボトルの最小中心間距離 {d:.1f}（ボトル外径 "
              f"{2 * br:.1f} / 隙間 {d - 2 * br:.1f} mm）"
              f" 割出しの進み {s:.2f}・供給局所 {a:.1f}deg・排出局所 {bb:.1f}deg")
        ok &= d - 2.0 * br > 0.0

    # 星車が抱えるボトルと、相手の星車の板
    gap = _cross_plate_gap(lay)
    if gap is not None:
        print(f"星車のボトルと相手の星車板の隙間 最小 {gap:.1f} mm")
        ok &= gap > 0.0

    # スターホイール 2 台の隙間。中心間の距離と外径の和で見る
    c0 = np.asarray(lay["trf_units"][0]["center"], dtype=float)
    c1 = np.asarray(lay["trf_units"][1]["center"], dtype=float)
    span = float(np.hypot(*(c1 - c0)))
    print(f"星車 2 台 中心間 {span:.1f} / 外径の和 "
          f"{2 * lay['trf_star_outer_r']:.1f} -> 隙間 "
          f"{span - 2 * lay['trf_star_outer_r']:.1f} mm")
    ok &= span - 2.0 * lay["trf_star_outer_r"] > 0.0

    # 歯車のピッチ円が接すること。中心距離がピッチ円半径の和と一致するかを見る
    gap = (lay["trf_gear_center_dist"]
           - 0.5 * (lay["trf_gear_pcd"] + lay["trf_drive_gear_pcd"]))
    print(f"歯車 ピッチ円半径の和 "
          f"{0.5 * (lay['trf_gear_pcd'] + lay['trf_drive_gear_pcd']):.3f} / "
          f"中心距離 {lay['trf_gear_center_dist']:.3f} -> ずれ {gap:.2e} mm / "
          f"歯先円の和が中心距離を超える量 "
          f"{0.5 * (lay['trf_gear_tip_d'] + lay['trf_drive_gear_tip_d']) - lay['trf_gear_center_dist']:.2f} mm"
          f"（この量ぶん歯が互いの歯溝に入る。平らな円板だと同じ量だけ食い込む）")
    ok &= abs(gap) < 1e-9
    # 歯どうしのすき間。割出しのあいだ回しても保たれるかまで見る
    for name, spin in (("停留", 0.0),
                       ("割出し 1/4", -np.radians(lay["trf_index_deg"]) * 0.5),
                       ("割出し 1/2", -np.radians(lay["trf_index_deg"]) * 1.0),
                       ("割出し 3/4", -np.radians(lay["trf_index_deg"]) * 1.5)):
        got = _mesh_clearance(lay, spin)
        line = " / ".join(f"{role} {v:+.3f}" for role, v in got.items())
        print(f"  歯どうしのすき間（{name}） {line} mm")
        ok &= min(got.values()) > 0.0

    # 星車軸・歯車と架台の脚。歯車はプーリより一回り大きいので当たり方が変わる。
    # 脚は架台側の持ち物なので、当たっていてもこちらでは直せない。数だけ出して
    # 受け渡しの判定とは分けて報告する
    first = 45.0 + float(lay["trf_leg_phase_deg"])
    at = " / ".join(f"{(first + 90.0 * k) % 360.0:.1f}" for k in range(4))
    print(f"架台の脚は世界角 {at}deg に立っている前提"
          f"（対角から {lay['trf_leg_phase_deg']:+.1f}deg）")
    handoff = []
    for role, d in lay["trf_leg_clash"].items():
        for what, v in d.items():
            if v > 0.0:
                print(f"  {role:9s} {what} が架台の脚に {v:.1f} mm 食い込む")
                handoff.append(f"{role} の {what} が脚に {v:.1f} mm 食い込む")
            elif not np.isfinite(v):
                print(f"  {role:9s} {what} は脚の上端 "
                      f"{lay['trf_deck_bottom_z']:.1f} より上。当たらない")
            else:
                print(f"  {role:9s} {what} と架台の脚の隙間 {-v:.1f} mm")
    if handoff:
        need = _leg_phase_needed(lay)
        print("  -> 架台側へ回す案件。歯車のピッチ円半径 112.5 は中心距離と"
              "速比から動かせないので、脚の振りで逃がすほかない。")
        print(f"     いまの振り {lay['trf_leg_phase_deg']:+.1f}deg / "
              f"当たらなくなるのは {need:+.1f}deg より深く振ったとき / "
              f"-45.0deg なら脚は世界角 0/90/180/270 に来て "
              f"{-_leg_phase_clash(lay, -45.0):.1f} mm 空く")
    # ボトルは軸受台の外を通る
    print(f"軸受台とポケットのボトルの隙間 "
          f"{lay['trf_star_pcd_r'] - br - lay['trf_bearing_d'] / 2:.1f} mm")
    ok &= lay["trf_star_pcd_r"] - br > lay["trf_bearing_d"] / 2.0

    # 外周ガイドとスターホイール板の高さ
    print(f"外周ガイド上端 {lay['trf_ring_base_z'] + lay['trf_ring_h']:.1f} / "
          f"スターホイール板下面 {lay['trf_star_base_z']:.1f}")
    ok &= lay["trf_ring_base_z"] + lay["trf_ring_h"] < lay["trf_star_base_z"]

    print("干渉の当たり（受け渡し部の受け持ち）: "
          + ("問題なし" if ok else "要修正"))
    for row in handoff:
        print("架台側へ回す: " + row)


def _spin_state(params: dict, psi_deg: float):
    """確認用の状態。入力軸角 psi だけを持たせる。

    基盤側の MachineState にまだ cam_angle_rad が無くても動くよう、
    ここでは属性を 1 つ持つだけの入れ物を使う。
    """
    class _S:
        cam_angle_rad = np.radians(float(psi_deg))
        table_angle_rad = 0.0
    return _S()


def _table_angle_deg(params: dict, psi_deg: float) -> float:
    """確認用のテーブル角 [deg]。基盤側の psi -> テーブル角をそのまま使う。

    星車の自転もこれの -速比 倍で出しているので、ここを別の式で近似すると
    確かめにならない。
    """
    return float(np.degrees(_table_angle_rad(params, np.radians(float(psi_deg)))))


def _apply_spin(meshes: dict, params: dict, lay: dict, psi_deg: float) -> dict:
    """自転する群に、自分の中心まわりの回転を当てる。

    基盤側の受け口ができるまでの確認用。SPIN_CENTERS と spin_angles() が
    そのまま使えるかを、ここで数値と絵の両方で当たっておく。
    """
    state = _spin_state(params, psi_deg)
    turns = [(SPIN_CENTERS, spin_angles(params, lay, state))]
    # ほかのモジュールの自転も当てる。噛み合う相手（テーブル側の歯車）が
    # 止まったままだと、絵の上でだけ歯が食い違って見える
    try:
        import scene                                  # noqa: PLC0415 循環参照を避ける

        for asm in getattr(scene, "ASSEMBLIES", ()):
            if asm.name == Path(__file__).stem:
                continue
            centers = dict(getattr(asm, "spin_centers", {}) or {})
            if centers:
                turns.append((centers, asm.spin_angles(params, lay, state)))
    except Exception:                                  # noqa: BLE001 書きかけなら諦める
        pass

    out = dict(meshes)
    for centers, ang in turns:
        for name, center in centers.items():
            mesh = out.get(name)
            if mesh is None or center is None or name not in ang:
                continue
            cx, cy = center
            mat = (parts.transform_matrix(translate=(cx, cy, 0.0))
                   @ parts.transform_matrix(rot_z_deg=np.degrees(ang[name]))
                   @ parts.transform_matrix(translate=(-cx, -cy, 0.0)))
            out[name] = parts.place(mesh, mat)
    return out


def _combine(*pieces) -> dict:
    """群ごとに合流させる。scene 側に合成ヘルパが出ていればそれを使う。"""
    import scene

    fn = getattr(scene, "merge_groups", None)
    if callable(fn):
        try:
            return fn(*pieces)
        except TypeError:
            pass
    groups: dict = {}
    for piece in pieces:
        for name, mesh in piece.items():
            groups.setdefault(name, []).append(mesh)
    return {k: parts.merge(v) for k, v in groups.items() if v}


def _relative_speed(params: dict, lay: dict, psi_deg: float, role: str,
                    d_psi=0.05):
    """テーブル側と星車側が 1 本のボトルを運ぶ速度 [mm/deg]。

    テーブル角と星車の自転角を psi で数値微分して、受け渡しに関わる 1 本の
    ボトルの速度ベクトルを両側から出す。戻りは (テーブル側の速さ, 星車側の
    速さ, ベクトルの差の大きさ)。

    ピッチが揃っているので**速さは常に一致する**。ベクトルまで一致するのは
    2 つのピッチ円が接している受け渡しの瞬間（psi = 0 と 180deg）だけで、
    そこから離れると両者は別々の円を進むので向きがずれていく。ボトルは
    その瞬間に持ち替わるので、これで滑りは出ない。
    """
    u = next(q for q in lay["trf_units"] if q["role"] == role)
    c = np.asarray(u["center"], dtype=float)

    def _pair(psi):
        th = _table_angle_deg(params, psi)
        spin = np.degrees(
            spin_angles(params, lay, _spin_state(params, psi))[_star_group(role)])
        return (lay["pitch_r"] * _unit(u["angle_deg"] + th),
                c + lay["trf_star_pcd_r"] * _unit(u["phase_deg"] + spin))

    a0, b0 = _pair(psi_deg - d_psi)
    a1, b1 = _pair(psi_deg + d_psi)
    va = (a1 - a0) / (2.0 * d_psi)
    vb = (b1 - b0) / (2.0 * d_psi)
    return float(np.hypot(*va)), float(np.hypot(*vb)), float(np.hypot(*(va - vb)))


def _spin_report(params: dict, lay: dict) -> None:
    """自転角を数値で確かめる。テーブルと同時に動いているか。

    速さも出す。ピッチが揃っていれば、割出しのどの瞬間でもテーブル側と
    星車側のボトルの速さは一致する。ベクトルまで一致するのは 2 つのピッチ円が
    接している受け渡しの瞬間（psi = 0 と 180deg）だけ。
    """
    print("自転の確かめ（psi = 入力軸角。割出し 0〜180deg / 停留 180〜360deg）")
    for psi in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0):
        ang = spin_angles(params, lay, _spin_state(params, psi))
        deg = np.degrees(ang[_star_group("infeed")])
        va, vb, dv = _relative_speed(params, lay, psi, "infeed")
        print(f"  psi {psi:5.1f}deg -> テーブル {_table_angle_deg(params, psi):5.1f}deg / "
              f"星車 {deg:7.1f}deg  速さ テーブル側 {va:6.3f} / "
              f"星車側 {vb:6.3f} mm/deg（差 {abs(va - vb):.2e}） "
              f"ベクトル差 {dv:6.3f}")
    print("  自転の中心: " + " / ".join(
        f"{k} ({v[0]:.2f}, {v[1]:.2f})" for k, v in SPIN_CENTERS.items()
        if not k.endswith(("_bottle", "_glass", "_liquid"))))


def _bottle_probe(centers, lay: dict, z_levels=5, step=3.0):
    """ボトル胴の中身を粗い格子で埋めた点群。干渉の数え上げに使う。

    ボトルの面の点だけを見ても、板の高さに面の点が無ければ食い込みを
    見落とす。胴を中身ごと埋めて、板の中に入った点を数える。
    """
    import pyvista as pv

    br = lay["bottle_r"]
    z0 = lay["trf_star_base_z"]
    z1 = lay["trf_star_top_z"]
    n = max(int(2.0 * br / step), 3)
    g = np.linspace(-br, br, n)
    gx, gy = np.meshgrid(g, g)
    keep = (gx ** 2 + gy ** 2) <= br * br
    gx, gy = gx[keep], gy[keep]
    pts = []
    for z in np.linspace(z0 + 0.5, z1 - 0.5, z_levels):
        for cx, cy in centers:
            pts.append(np.column_stack([gx + cx, gy + cy, np.full(gx.size, z)]))
    if not pts:
        return None
    return pv.PolyData(np.vstack(pts))


def _inside_count(probe, solid) -> int:
    """probe の点のうち solid の中に入っている数。干渉の目安に使う。"""
    if probe is None or solid is None or probe.n_points == 0 or solid.n_points == 0:
        return 0
    sel = probe.select_enclosed_points(solid.extract_surface().triangulate(),
                                       tolerance=0.0, check_surface=False)
    return int(np.count_nonzero(sel["SelectedPoints"]))


def _spun_star_bottles(params: dict, lay: dict, psi_deg: float, role: str):
    """自転角 psi のときの、ポケットのボトル中心。"""
    ang = spin_angles(params, lay, _spin_state(params, psi_deg))[_star_group(role)]
    cx, cy = SPIN_CENTERS[_star_group(role)]
    out = []
    for r, bx, by in lay["trf_star_bottles"]:
        if r != role:
            continue
        v = np.array([bx - cx, by - cy])
        c, s = np.cos(ang), np.sin(ang)
        out.append((cx + c * v[0] - s * v[1], cy + s * v[0] + c * v[1]))
    return out


def _table_bottles(lay: dict, table_angle_deg: float, params=None):
    """テーブルのホルダのうち、ボトルが載っているものの中心。

    実機のテーブルは全部のホルダが埋まってはいない。基盤側の
    scene.station_present() が「供給から排出の手前まで」を出すので、
    それに従う。空のホルダまで数えると、板が通って当たり前の場所を
    干渉として数えてしまう。
    """
    n = int(lay["stations"])
    present = [True] * n
    if params is not None:
        import scene                                  # noqa: PLC0415 循環参照を避ける

        fn = getattr(scene, "station_present", None)
        if callable(fn):
            present = list(fn(params, lay, np.radians(float(table_angle_deg))))
    return [tuple(lay["pitch_r"] * _unit(table_angle_deg + 360.0 * k / n))
            for k in range(n) if present[k]]


def _spun_star_disc(params: dict, lay: dict, role: str, psi_deg: float):
    """自転角 psi のときのスターホイール円盤だけを組む。干渉の数え上げ用。"""
    u = next(q for q in lay["trf_units"] if q["role"] == role)
    ang = np.degrees(spin_angles(params, lay,
                                 _spin_state(params, psi_deg))[_star_group(role)])
    mat = parts.transform_matrix(translate=(u["center"][0], u["center"][1], 0.0),
                                 rot_z_deg=u["phase_deg"] + ang)
    return parts.star_wheel(outer_d=2.0 * lay["trf_star_outer_r"],
                            pockets=lay["trf_star_pockets"],
                            pocket_r=lay["trf_star_pocket_r"],
                            pcd=2.0 * lay["trf_star_pcd_r"],
                            thickness=lay["trf_star_t"],
                            base_z=lay["trf_star_base_z"],
                            hub_d=lay["trf_hub_d"], hub_h=lay["trf_hub_h"],
                            shaft_d=lay["trf_shaft_d"], shaft_h=1.0,
                            matrix=mat)["disc"]


def _deck_intrusion(lay: dict, meshes: dict) -> None:
    """天板の板（半径 deck_r・z は板厚ぶん）の中に入っている自分の点を数える。

    軸を天板の上で止め、柱も天板の上で受ける形にしたので、ここは 0 でなければ
    ならない。0 でなければ天板に穴を開けてもらう話が戻ってくる。
    """
    z0 = float(lay["trf_deck_bottom_z"])
    z1 = float(lay["deck_top"])
    r = _deck_r(lay)
    print(f"天板（半径 {r:.1f} / z {z0:.1f}〜{z1:.1f}）に食い込む自分の点")
    total = 0
    for name, mesh in sorted(meshes.items()):
        if not name.startswith("trf_") or mesh is None or mesh.n_points == 0:
            continue
        q = np.asarray(mesh.points)
        n = int(np.count_nonzero((q[:, 2] > z0 + 1e-6) & (q[:, 2] < z1 - 1e-6)
                                 & (np.hypot(q[:, 0], q[:, 1]) < r - 1e-6)))
        if n:
            print(f"  {name} {n} 点")
        total += n
    print(f"  合計 {total} 点" + ("" if total else "（貫通なし）"))


def _gear_intrusion(lay: dict, meshes: dict) -> None:
    """星車の歯車の円筒に入っている、ほかの群の点を数える。

    歯車は天板の上・テーブル板の下に移したので、そこに何が居るかは組んで
    みないと分からない。噛み合う相手（テーブル側の歯車）は入って当たり前。
    """
    # 相手の点が帯の縁ちょうどに乗っていることがある（噛み合う歯車の上面と
    # こちらの上面は同じ高さ）。少しだけ広げて拾う
    tol = float(lay["plate_t"]) * 0.1
    z0 = float(lay["trf_gear_base_z"]) - tol
    z1 = float(lay["trf_gear_top_z"]) + tol
    rt = float(lay["trf_gear_tip_d"]) / 2.0
    centers = [np.asarray(u["center"], dtype=float) for u in lay["trf_units"]]
    print(f"星車の歯車の円筒（軸まわり {rt:.1f} / z {z0:.1f}〜{z1:.1f}・"
          f"縁は {tol:.1f} 甘く見る）に入っている他の群の点")
    hit = False
    for name, mesh in sorted(meshes.items()):
        if name.startswith("trf_") or mesh is None or mesh.n_points == 0:
            continue
        q = np.asarray(mesh.points)
        keep = (q[:, 2] > z0) & (q[:, 2] < z1)
        if not keep.any():
            continue
        qq = q[keep]
        n = 0
        deep = 0.0
        for c in centers:
            d = rt - np.hypot(qq[:, 0] - c[0], qq[:, 1] - c[1])
            n += int(np.count_nonzero(d > 1e-6))
            if (d > 1e-6).any():
                deep = max(deep, float(d.max()))
        if n:
            hit = True
            note = "（噛み合う相手。入って当たり前）" if name.startswith("drv_gear") else ""
            print(f"  {name} {n} 点 / いちばん深いところで {deep:.1f} mm{note}")
    if not hit:
        print("  なし")


def _slice_edges(mesh, z, near, reach):
    """メッシュを高さ z で切った断面の線分。near から reach 以内だけ拾う。

    戻りは (始点, 終点)。切り口が 1 本も無ければ None。
    """
    if mesh is None or mesh.n_points == 0:
        return None
    sl = mesh.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, float(z)))
    if sl is None or sl.n_points == 0 or sl.n_cells == 0:
        return None
    pts = np.asarray(sl.points)[:, :2]
    arr = np.asarray(sl.lines).ravel()
    idx, i = [], 0
    while i < len(arr):
        n = int(arr[i])
        row = arr[i + 1:i + 1 + n]
        idx += [(int(row[k]), int(row[k + 1])) for k in range(n - 1)]
        i += 1 + n
    if not idx:
        return None
    idx = np.asarray(idx)
    a, b = pts[idx[:, 0]], pts[idx[:, 1]]
    near = np.asarray(near, dtype=float)
    keep = ((np.hypot(*(a - near).T) < reach) | (np.hypot(*(b - near).T) < reach))
    if not keep.any():
        return None
    return a[keep], b[keep]


def _mesh_gap_measured(lay: dict, meshes: dict) -> None:
    """**絵に出ている**歯車を高さで切って、噛み合いのすき間を実測する。

    `_mesh_clearance()` は諸元から組み直した断面で測るので、相手（テーブル側）
    の歯の付け根の食い込みが lay に出ていないと少し狭めに出る。ここは
    描かれたメッシュそのものを切るので、その心配が無い。
    """
    z = 0.5 * (float(lay["trf_gear_base_z"]) + float(lay["trf_gear_top_z"]))
    tbl_name = next((k for k in meshes if k.startswith("drv_gear_table")), None)
    if tbl_name is None:
        _warn_once("drv_gear_table",
                   "テーブル側の歯車のメッシュが無いので、噛み合いの実測は飛ばす")
        return
    reach = float(lay["trf_gear_module"]) * 6.0
    print(f"噛み合いのすき間（絵を高さ {z:.1f} で切って実測・"
          f"ピッチ点まわり {reach:.0f} mm）")
    for u in lay["trf_units"]:
        pitch_pt = lay["pitch_r"] * _unit(u["angle_deg"])
        star = _slice_edges(meshes.get(_star_group(u["role"], "_gear")), z,
                            pitch_pt, reach)
        tbl = _slice_edges(meshes[tbl_name], z, pitch_pt, reach)
        if star is None or tbl is None:
            print(f"  {u['role']:9s} 切り口が取れない")
            continue
        gap = _edges_gap(star[0], star[1], tbl[0], tbl[1])
        crossed = _segments_cross(star[0], star[1], tbl[0], tbl[1])
        print(f"  {u['role']:9s} {'交差あり ' if crossed else ''}"
              f"{gap:+.3f} mm（線分 星車側 {len(star[0])} 本 / "
              f"テーブル側 {len(tbl[0])} 本）")


def _bottom_probe(bottle_r, rings=((0.0, 1), (0.55, 8), (0.92, 8))):
    """ボトルの底に光線を落とす当て所。中心 1 + 内輪 8 + 外輪 8 = 17 点。"""
    pts = []
    for frac, n in rings:
        for k in range(int(n)):
            a = 2.0 * np.pi * k / float(n)
            pts.append((bottle_r * frac * np.cos(a),
                        bottle_r * frac * np.sin(a)))
    return np.asarray(pts, dtype=float)


def _drop_tree(meshes: dict, z0, z1):
    """光線を当てる相手。z0〜z1 に掛かる不透明な群だけを 1 つに束ねる。

    透過物（ボトル・ガラス・液）は外す。高さで絞ると木がずっと小さくなり、
    光線 1 本あたりが速くなる。
    """
    import vtk                                          # noqa: PLC0415

    keep = []
    for k, m in meshes.items():
        if m is None or not m.n_points:
            continue
        if any(w in k for w in ("bottle", "glass", "liquid")):
            continue
        b = m.bounds
        if b[5] < z0 or b[4] > z1:
            continue
        keep.append(m)
    if not keep:
        return None
    surf = parts.merge(keep).extract_surface().triangulate()
    tree = vtk.vtkOBBTree()
    tree.SetDataSet(surf)
    tree.SetTolerance(1e-7)
    tree.BuildLocator()
    return tree


def _drop_hit_z(tree, hits, x, y, z0, z1):
    """(x, y) から z0 -> z1 へ光線を落として、いちばん上の当たりの高さ。

    当たらなければ None。hits は使い回す入れ物。
    """
    if not tree.IntersectWithLine((float(x), float(y), float(z0)),
                                  (float(x), float(y), float(z1)), hits, None):
        return None
    n = hits.GetNumberOfPoints()
    if n == 0:
        return None
    return max(float(hits.GetPoint(i)[2]) for i in range(n))


def _run_length(flags, step):
    """並びの中で、False が続いたいちばん長い区間の長さ [mm]。"""
    best = run = 0
    for f in flags:
        run = 0 if f else run + 1
        best = max(best, run)
    return best * float(step)


def _foot_support_check(params: dict, lay: dict, meshes: dict,
                        steps=41, step_mm=1.5) -> bool:
    """星車が運ぶボトルの足元に、受けがあるかを光線で数える。

    ボトル中心を星車のピッチ円に沿ってコンベア接点から受け渡し点まで走らせ、
    底へ真下向きの光線を落とす。当たった面がボトル底より板厚ぶん以上下なら、
    そこは受けが無い（宙に浮いている）と数える。

    数えるのは 2 つ。

        17 点のうち受けのある数   レビューが出した数と突き合わせるため
        受けの切れ目の幅 [mm]     走行の向きに 1.5 mm 刻みで並べた光線が、
                                  連続して外し続けた長さ。回る板と固定板の
                                  合わせ目のような数 mm の隙は、ボトル外径
                                  68 に対しては受けが切れたことにならない。
                                  板が丸ごと退いていれば、ここが 100 mm 規模
                                  になって出る

    円で切っていたころは、この区間の 50deg（走行 約 100 mm）で見つかる面が
    歯車の上面（ボトル底より 10 mm 下）だけになり、最悪 17 点中 6 点しか
    受けが無かった。
    """
    import vtk                                          # noqa: PLC0415

    br = float(lay["bottle_r"])
    top = float(lay["table_top"])
    floor_z = float(lay["deck_top"]) - float(lay["plate_t"])
    limit = top - float(lay["plate_t"])                 # ここより下は受けでない
    tree = _drop_tree(meshes, floor_z, top)
    if tree is None:
        print("足元の受け: 当てる相手が無い")
        return False
    hits = vtk.vtkPoints()
    probe = _bottom_probe(br)
    # 走行の向きに沿った 3 本の走査線。真ん中と、左右へ寄せた 2 本
    lanes = (-0.6, 0.0, 0.6)
    print(f"足元の受け（ボトル底 {top:.1f} の下 {top - limit:.1f} mm 以内に面が"
          f"あるか。{len(probe)} 点の数え上げと、走行の向きの走査 "
          f"{step_mm} mm 刻み）")
    ok = True
    for u in lay["trf_units"]:
        c = np.asarray(u["center"], dtype=float)
        worst_n = (len(probe), None)
        worst_run = (0.0, None)
        for t in np.linspace(u["swept"][0], u["swept"][1], steps):
            q = c + lay["trf_star_pcd_r"] * _unit(t)
            got = 0
            for dx, dy in probe:
                z = _drop_hit_z(tree, hits, q[0] + dx, q[1] + dy,
                                top - 0.1, floor_z)
                got += int(z is not None and z >= limit)
            if got < worst_n[0]:
                worst_n = (got, float(t))
            head = _unit(t + 90.0)                      # 通り道の接線
            side = np.array([-head[1], head[0]])
            for lane in lanes:
                half = float(np.sqrt(max(br * br - (lane * br) ** 2, 0.0)))
                n = max(int(2.0 * half / step_mm), 2)
                flags = []
                for s in np.linspace(-half, half, n):
                    pt = q + head * s + side * (lane * br)
                    z = _drop_hit_z(tree, hits, pt[0], pt[1], top - 0.1, floor_z)
                    flags.append(bool(z is not None and z >= limit))
                run = _run_length(flags, 2.0 * half / (n - 1))
                if run > worst_run[0]:
                    worst_run = (run, float(t))
        print(f"  {u['role']:9s} いちばん少ないところで {worst_n[0]}/{len(probe)} 点"
              f"（局所 {worst_n[1]:.1f}deg） / 受けの切れ目 いちばん広くて "
              f"{worst_run[0]:.1f} mm（局所 "
              f"{worst_run[1] if worst_run[1] is not None else float('nan'):.1f}deg・"
              f"ボトル外径 {2 * br:.0f} に対し "
              f"{100.0 * worst_run[0] / (2 * br):.0f}%）")
        ok &= worst_run[0] <= float(lay["plate_t"])
    return ok


def _pocket_plate_depth(lay: dict, pts, table_angle_deg: float):
    """点が回るポケット板の材の中へどれだけ入っているか [mm]。外なら負。"""
    n = int(lay["stations"])
    rho = np.hypot(pts[:, 0], pts[:, 1])
    d = float(lay["pocket_plate_r"]) - rho
    for k in range(n):
        c = lay["pitch_r"] * _unit(table_angle_deg + 360.0 * k / n)
        d = np.minimum(d, np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
                       - float(lay["pocket_r"]))
    return d


def _ring_guide_depth(lay: dict, pts):
    """点がテーブル外周ガイドの材の中へどれだけ入っているか [mm]。外なら負。"""
    rho = np.hypot(pts[:, 0], pts[:, 1])
    phi = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    best = np.full(len(pts), -np.inf)
    for a0, a1 in lay["trf_ring_arcs"]:
        span = float(a1) - float(a0)
        off = (phi - float(a0)) % 360.0
        d = np.minimum(float(lay["trf_ring_ro"]) - rho,
                       rho - float(lay["trf_ring_ri"]))
        d = np.minimum(d, np.minimum(off, span - off) * np.pi / 180.0 * rho)
        best = np.maximum(best, d)
    return best


def _disc_grid(radius, step):
    """半径 radius の円を step 刻みで埋めた点。中心から輪を重ねる。

    正方格子より点が均される。食い込みの深さは輪郭の近くで決まるので、
    外周まわりが粗くならないようにする。
    """
    out = [np.zeros((1, 2))]
    n_ring = max(int(round(float(radius) / float(step))), 1)
    # 外周ちょうどを必ず入れる。食い込みがいちばん深いのは縁なので、
    # 刻みの都合で最後の輪が内側に落ちると、そのぶん浅く出る
    for r in np.linspace(float(radius) / n_ring, float(radius), n_ring):
        n = max(int(2.0 * np.pi * r / float(step)), 6)
        a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        out.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(out)


def _reverse_bite_check(params: dict, lay: dict, step=0.6) -> bool:
    """**星車が抱えるボトル**が、テーブル側のものへ食い込んでいないか。

    これまで数えていたのは「テーブルのボトル -> 星車板」の一方向だけで、
    逆向き（星車のボトル -> ポケット板・外周ガイド）は誰も見ていなかった。
    ポケット板への +1.27 mm の食い込みがそれで残っていた。

    相手はどれも回転体か円弧板なので、メッシュを組まずに式で当たる。ボトル
    胴を格子で埋め、材の中へ入った深さのいちばん大きいものを返す。
    """
    br = lay["bottle_r"]
    grid = _disc_grid(br, step)
    # 高さの重なり。ボトル胴は table_top から body_h まで
    b0, b1 = float(lay["table_top"]), float(lay["table_top"]) + float(lay["bottle_h"])
    rows = [("ポケット板", float(lay["pocket_base"]), float(lay["pocket_top"])),
            ("外周ガイド", float(lay["trf_ring_base_z"]),
             float(lay["trf_ring_base_z"]) + float(lay["trf_ring_h"]))]
    print(f"逆向きの当たり（星車のボトル {len(grid)} 点 x 全 psi -> テーブル側）")
    ok = True
    for name, z0, z1 in rows:
        if z1 <= b0 or z0 >= b1:
            print(f"  {name}: z {z0:.1f}〜{z1:.1f} はボトル胴 {b0:.1f}〜{b1:.1f} と"
                  "重ならない。当たりようがない")
            continue

        def _bite(psi, name=name):
            table_deg = _table_angle_deg(params, float(psi))
            worst = None
            for role in ("infeed", "discharge"):
                for cx, cy in _spun_star_bottles(params, lay, float(psi), role):
                    pts = grid + np.array([cx, cy])
                    d = (_pocket_plate_depth(lay, pts, table_deg)
                         if name == "ポケット板" else _ring_guide_depth(lay, pts))
                    v = float(d.max())
                    if worst is None or v > worst[0]:
                        worst = (v, role, float(table_deg))
            return worst

        # 粗く当たりを付けてから、山の高いほうから順に細かく見る。深さは psi
        # の関数として尖っていて山も 1 つではないので、いちばん高い粗点の
        # まわりだけを詰めると別の山を見落とす（実際に 1.30 を 1.07 と
        # 読み違えた）
        coarse = [(_bite(psi), float(psi)) for psi in np.linspace(0.0, 360.0, 73)]
        coarse.sort(key=lambda q: -q[0][0])
        best = coarse[0]
        for _, psi0 in coarse[:6]:
            for psi in np.linspace(psi0 - 5.0, psi0 + 5.0, 41):
                got = _bite(psi)
                if got[0] > best[0][0]:
                    best = (got, float(psi))
        (v, role, table_deg), psi = best
        if v > 0.0:
            who = ("寸法は回る側の担当（ポケット板の外半径か溝半径）"
                   if name == "ポケット板" else "こちらの受け持ち")
            print(f"  {name}: {v:+.2f} mm 食い込む"
                  f"（psi {psi:.1f}deg・テーブル {table_deg:.1f}deg・{role}）"
                  f" -> {who}")
            ok = False
        else:
            print(f"  {name}: いちばん近づいて隙間 {-v:.2f} mm"
                  f"（psi {psi:.1f}deg・{role}）")
    return ok


def _interference_check(params: dict, lay: dict, meshes: dict) -> None:
    """自転を振って、星車板とボトルの食い込みを数える。

    数えるのは「ボトル胴を埋めた点のうち、板の中に入った数」。0 なら
    食い込み無し。板は自転角 0 で組んであるので、psi ごとに回してから見る。
    """
    _deck_intrusion(lay, meshes)
    _gear_intrusion(lay, meshes)
    _mesh_gap_measured(lay, meshes)
    ok = _foot_support_check(params, lay, meshes)
    ok &= _reverse_bite_check(params, lay)
    print("足元の受けと逆向きの当たり: " + ("問題なし" if ok else "要修正"))
    print("干渉の数え上げ（ボトル胴を 3mm 格子で埋めた点のうち板の中に入った数）")
    for psi in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0,
                240.0, 300.0, 359.9):
        table_deg = _table_angle_deg(params, psi)
        rows = []
        for role in ("infeed", "discharge"):
            # 当てるのは円盤だけ。軸や歯車まで混ぜた群を渡すと、塊どうしが
            # 交わったところで内外の判定が壊れて、離れている点まで拾う
            disc = _spun_star_disc(params, lay, role, psi)
            if disc is None:
                continue
            rows.append(_inside_count(
                _bottle_probe(_table_bottles(lay, table_deg, params), lay), disc))
            rows.append(_inside_count(
                _bottle_probe(_spun_star_bottles(params, lay, psi, role), lay), disc))
        print(f"  psi {psi:5.1f}deg テーブル {table_deg:5.1f}deg  "
              f"供給 板xテーブルのボトル {rows[0]:4d} / 板x自分のボトル {rows[1]:4d}  "
              f"排出 板xテーブルのボトル {rows[2]:4d} / 板x自分のボトル {rows[3]:4d}")


def _render(meshes, cam, out_path, size, draw_order, material):
    import pyvista as pv
    import scene

    # 視点の表に "_only" があれば、その接頭辞の群だけを描く。天板の下は
    # 搬送の枠と小物が手前に来て歯車が見えないので、そこだけ間引く
    cam = dict(cam)
    only = cam.pop("_only", None)
    drop = cam.pop("_drop", None)
    if only:
        meshes = {k: v for k, v in meshes.items() if k.startswith(tuple(only))}
    if drop:
        meshes = {k: v for k, v in meshes.items() if not k.startswith(tuple(drop))}
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    plotter.set_background(scene.BACKGROUND)
    plotter.set_environment_texture(scene.studio_cubemap(), is_srgb=True)
    try:
        plotter.renderer.GetEnvMapPrefiltered().SetPrefilterMaxSamples(64)
        plotter.renderer.GetEnvMapIrradiance().SetIrradianceSize(32)
    except AttributeError:
        pass
    for name, mat in draw_order:
        mesh = meshes.get(name)
        if mesh is None or mesh.n_points == 0:
            continue
        plotter.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True,
                         feature_angle=35.0, **material[mat])
    import cameras
    cameras.apply_resolved(plotter, cam)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(out_path))
    plotter.close()
    return out_path


def _look(focal, azimuth_deg, elevation_deg, view_h, view_angle=22.0):
    """注視点と方位・仰角から、透視の視点諸元を組む。

    cameras.py の表は機械まるごとの構図を持っているが、受け渡し部の確認では
    寄り引きを自分で決めたいので、ここは自前で組む。
    """
    focal = np.asarray(focal, dtype=float)
    dist = view_h / (2.0 * np.tan(np.radians(view_angle) / 2.0))
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    eye = focal + dist * np.array([np.cos(el) * np.cos(az),
                                   np.cos(el) * np.sin(az), np.sin(el)])
    return dict(position=tuple(float(v) for v in eye),
                focal_point=tuple(float(v) for v in focal),
                view_up=(0.0, 0.0, 1.0), view_angle=view_angle,
                parallel=False, parallel_scale=None)


def _views(lay: dict, params: dict) -> dict:
    # 機械まるごとが入る大きさ。受け渡し部が端で切れないよう接点まで見る
    span = lay["trf_tangent_r"] + lay["bottle_h"]
    mid_z = lay["table_top"] + lay["bottle_h"] * 0.5
    views = {
        "top": dict(position=(0.0, 0.0, lay["total_height"] + 2000.0),
                    focal_point=(0.0, 0.0, lay["table_top"]),
                    view_up=(0.0, 1.0, 0.0), view_angle=20.0,
                    parallel=True, parallel_scale=span),
        "iso": _look((0.0, -lay["pitch_r"] * 0.6, mid_z), 300.0, 32.0, span * 2.1),
        "side": _look((0.0, -lay["pitch_r"] * 0.6, mid_z), 292.0, 4.0, span * 2.1),
    }
    # 受け渡し部の真上寄り。噛み合わせはこれで見る
    mid = 0.5 * (np.array(lay["trf_units"][0]["center"])
                 + np.array(lay["trf_units"][1]["center"]))
    views["top_zoom"] = dict(
        position=(float(mid[0]), float(mid[1]), lay["trf_star_top_z"] + 2000.0),
        focal_point=(float(mid[0]), float(mid[1]), lay["trf_star_base_z"]),
        view_up=(0.0, 1.0, 0.0), view_angle=20.0,
        parallel=True, parallel_scale=lay["trf_center_r"] * 1.15)
    # スターホイールを真横から。板の高さ・軸受台・渡し板の高さを見る
    c0 = lay["trf_units"][0]["center"]
    views["star_side"] = _look(
        (c0[0], c0[1], 0.5 * (lay["deck_top"] + lay["trf_star_top_z"])),
        lay["trf_units"][0]["angle_deg"] - 25.0, 8.0, lay["trf_center_r"] * 1.1)
    # 渡し板。ボトルの底の下に板が入っているかを低い視点で見る。以前は
    # 板が退いていて足元が空いていた区間（接点から 50deg 手前まで）の
    # 真ん中に注視点を置き、機外側から搬送面すれすれの高さで覗く
    u = lay["trf_units"][0]
    q = (np.asarray(u["center"], dtype=float)
         + lay["trf_star_pcd_r"] * _unit(u["angle_deg"] - u["swept_sign"] * -30.0))
    views["dead"] = _look(
        (q[0], q[1], lay["table_top"] + lay["bottle_h"] * 0.10),
        float(np.degrees(np.arctan2(q[1], q[0]))) - 34.0, 6.0,
        lay["bottle_h"] * 1.5)
    # 渡し板と搬送面の継ぎ目。ここがいちばん足元の空きやすいところなので、
    # 搬送面すれすれの高さで機外から寄る
    q = (np.asarray(u["center"], dtype=float)
         + lay["trf_star_pcd_r"] * _unit(u["angle_deg"] + 25.0 * u["swept_sign"]))
    views["foot"] = _look(
        (q[0], q[1], lay["table_top"] + lay["bottle_h"] * 0.12),
        float(np.degrees(np.arctan2(q[1], q[0]))) + 40.0, 9.0,
        lay["bottle_h"] * 2.0)
    # 歯車の高さ（天板の上・テーブル板の下）を真横から。軸が天板の上で
    # 止まっていることと、門型の軸受台の足元がここで見える
    gear_mid_z = 0.5 * (lay["trf_gear_base_z"] + lay["trf_gear_top_z"])
    for name, u in (("drive", lay["trf_units"][1]), ("drive_in", lay["trf_units"][0])):
        c1 = u["center"]
        views[name] = _look((c1[0], c1[1], gear_mid_z),
                            u["angle_deg"], 9.0, lay["trf_center_r"] * 0.75)
    # 噛み合い点の寄り。歯が互いの歯溝に入っているかは真上からでないと
    # 分からない。テーブル板が上に被さっているので、歯車だけを描く
    u = lay["trf_units"][1]
    pitch_pt = lay["pitch_r"] * _unit(u["angle_deg"])
    views["mesh"] = dict(
        position=(float(pitch_pt[0]), float(pitch_pt[1]),
                  lay["trf_gear_top_z"] + lay["bottle_h"]),
        focal_point=(float(pitch_pt[0]), float(pitch_pt[1]), gear_mid_z),
        view_up=(0.0, 1.0, 0.0), view_angle=20.0,
        parallel=True, parallel_scale=lay["trf_gear_module"] * 8.0,
        _only=("trf_star_in", "trf_star_out", "drv_gear"),
        # 受け渡し点のボトルがちょうどここに立つので、透過物だけを外す。
        # 接頭辞まるごとで落とすと歯車の群まで消える
        _drop=tuple(_star_group(role, suffix)
                    for role in STAR_ROLE_SUFFIX
                    for suffix, mat in STAR_GROUP_PARTS
                    if mat in ("bottle", "glass", "liquid")))
    # 天板とテーブル板のあいだの隙間を、機械の外から覗き込む。テーブル側の
    # 歯車と星車側の歯車が並んで噛み合っているかはこの向きでないと分からない。
    # 手前に来る搬送の枠と小物は外す
    u = lay["trf_units"][1]
    mid = 0.5 * np.asarray(u["center"], dtype=float)
    views["under"] = dict(
        _look((float(mid[0]), float(mid[1]), gear_mid_z),
              u["angle_deg"] + 20.0, 4.0, lay["trf_center_r"] * 1.15),
        _only=("frame", "trf_", "drv_", "steel"))
    # 供給・排出それぞれの寄り。注視点を星車の中心へ置いて斜め上から覗く
    for name, u, az in (("infeed", lay["trf_units"][0], 300.0),
                        ("discharge", lay["trf_units"][1], 250.0)):
        c = u["center"]
        views[name] = _look((c[0], c[1], lay["trf_star_base_z"] - lay["bottle_h"] * 0.2),
                            az, 30.0, lay["trf_center_r"] * 1.8)
    return views


def _main(out_dir, size=(800, 600), solo=False, spin=True, checks=True):
    import scene

    params = scene.load_params()
    lay = scene.derive_layout(params)
    if "trf_center_r" not in lay:                 # scene が拾っていなければ自分で
        lay.update(layout(params, lay))
    _report(lay)
    _clearance_check(lay)
    _spin_report(params, lay)

    mine = build(params, lay)
    # scene.build() と同じ合流。静止側と回る側はどちらも "steel" を返すので、
    # update で受けると片方が丸ごと消える（ノズルと門柱が絵から落ちる）。
    base = _combine(scene.build_static(params, lay),
                    scene.build_carousel(params, lay))
    meshes = _combine(base, mine)
    if solo:
        # 他のサブシステムを外して、自分の部品だけを見る
        meshes = {k: v for k, v in meshes.items()
                  if not k.startswith(("drv_", "cnv_", "det_"))}

    if checks:
        _interference_check(params, lay, meshes)

    material = dict(scene.MATERIAL)
    material.update(MATERIALS)
    order = tuple(scene.DRAW_ORDER)
    for row in DRAW_ORDER:                     # scene 側が拾っていなければ足す
        if tuple(row) not in order:
            order += (tuple(row),)

    out_dir = Path(out_dir)
    views = _views(lay, params)
    for name, cam in views.items():
        path = _render(_apply_spin(meshes, params, lay, 0.0), cam,
                       out_dir / f"transfer_{name}.png", size, order, material)
        print("焼いた:", path)

    if spin:
        # 自転を振る。テーブル角も psi から出して、同時に動く様子を見る。
        # 0〜180deg が割出しなので、その中を 4 枚に割る
        cam = views["top_zoom"]
        for psi in (0.0, 45.0, 90.0, 135.0):
            table_ang = np.radians(_table_angle_deg(params, psi))
            frame = _combine(scene.build_static(params, lay),
                             scene.build_carousel(params, lay, table_ang),
                             mine)
            path = _render(_apply_spin(frame, params, lay, psi), cam,
                           out_dir / f"transfer_psi{int(psi):03d}.png",
                           size, order, material)
            print("焼いた:", path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="受け渡し系の確認")
    ap.add_argument("--out", type=Path, required=True, help="PNG の出力先")
    ap.add_argument("--size", type=int, nargs=2, default=(800, 600))
    ap.add_argument("--solo", action="store_true",
                    help="他のサブシステムを外して自分の部品だけ描く")
    ap.add_argument("--no-spin", action="store_true", help="自転を振った絵を焼かない")
    ap.add_argument("--no-checks", action="store_true", help="干渉の数え上げを飛ばす")
    a = ap.parse_args()
    _main(a.out, tuple(a.size), solo=a.solo, spin=not a.no_spin,
          checks=not a.no_checks)
