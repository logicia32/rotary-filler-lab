"""ロータリー充填機を 1 台組み立てて、固定視点で 1 枚描く。

主役は液面。ボトルの中の液がどれだけ入っていて、どちらへどれだけ傾いて
いるかが読み取れることを最優先にしてある。

ノズルは 1 本で、世界座標に固定してある。テーブルが回ってもノズルは動かず、
停止中にその下へ来たステーション 1 本だけを充填する。物理側と同じ作り。

使い方
------
    .venv/bin/python viz/scene.py                      # figs/scene.png
    .venv/bin/python viz/scene.py --cover              # 安全カバー付き
    .venv/bin/python viz/scene.py --tilt-deg 10 --out figs/tilt_10.png
    .venv/bin/python viz/scene.py --camera top          # 視点の表は cameras.py
    .venv/bin/python viz/scene.py --size 2400 1800

組み立て
--------
機械はサブシステムごとに viz/asm_*.py へ割ってある。ここはそれを読み込んで
合成するだけ。モジュールが守る形は viz/ASSEMBLY_CONTRACT.md にある。
まだ書かれていないモジュールがあっても、標準エラーに 1 行出して飛ばす。

外から呼ぶとき
--------------
時系列を流し込んでコマ送りにする場合は `render_state()` を使う。

    import scene
    params = scene.load_params()
    lay = scene.derive_layout(params)
    st = scene.MachineState(table_angle_rad=..., volumes_mL=[...],
                            tilt_t=[...], tilt_r=[...])
    scene.render_state(params, lay, st, Path("frames/0001.png"))

寸法の出どころ
--------------
機械の寸法は params.json だけを読む。ここに数値を直書きしない。
params.json に載っていないのは「絵にするためだけの比率」（架台の脚の太さ、
カバーの半径など、解析に効かない見た目の値）で、それは PROPORTION に
まとめて置いてある。すべて params.json の値に対する倍率で書いてある。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cameras  # noqa: E402
import parts  # noqa: E402

LAB_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = LAB_ROOT / "params.json"
DEFAULT_OUT = LAB_ROOT / "figs" / "scene.png"


# --------------------------------------------------------------------------
# 絵にするためだけの比率。解析には出てこない。すべて params.json の値の倍率。
# 機械の見え方を変えたいときはここを触る。
# --------------------------------------------------------------------------
PROPORTION = {
    "work_height_x_plate_d": 1.55,     # 床から架台天板の上面まで / テーブル外径
    "deck_thickness_x_plate_t": 1.40,  # 架台天板の板厚 / テーブル板厚
    "frame_pipe_x_plate_d": 0.086,     # 架台の角パイプ一辺 / テーブル外径
    "frame_span_x_plate_d": 0.86,      # 脚の芯々距離 / テーブル外径
    # 脚を立てる向き。parts.frame() は対角（45/135/225/315 度）に立てるので、
    # そこからの振り [deg]。**-22.5 で 22.5/112.5/202.5/292.5 度になる。**
    # 対角のままだと、供給スターホイールの中心（半径 337.5・世界角 315 度）と
    # 脚（半径 340.5・世界角 315 度）がほぼ同心で、星車の軸とプーリが脚を
    # 貫通する（軸が 36.9 mm、プーリが 69.8 mm 食い込んでいた）。22.5 度回すと
    # 芯々で 132.3 mm 空き、要る量（プーリつば 47.9 + 脚の半対角 34.1 = 82）に
    # 対して余裕がある。天板は丸いので見た目は脚と桁の向きしか変わらない。
    "frame_leg_phase_deg": -22.5,
    # 架台天板の下穴。テーブルを回す出力軸が通る。軸径にこれだけ逃げを足す
    # （半径で）。中実の円板のままだと軸が板に食い込む。
    "deck_bore_gap_x_plate_t": 0.5,
    "bearing_outer_x_plate_r": 0.35,   # 旋回軸受リング外半径 / テーブル半径
    "bearing_inner_x_plate_r": 0.26,   # 同 内半径
    "bearing_height_x_plate_t": 2.0,   # 同 高さ / テーブル板厚
    "hub_dia_x_plate_d": 0.20,         # テーブル中心ハブ径 / テーブル外径
    "hub_height_x_plate_t": 3.5,       # 同 高さ / テーブル板厚
    # 中心柱は回転軸を見せるためだけの短い出っ張り。上には何も載せない。
    # 太く高くすると、斜め上から見たとき奥のボトルを隠してしまう（下の注記）。
    "column_r_x_plate_r": 0.045,       # 中心柱の半径 / テーブル半径
    "column_h_x_hub_h": 1.1,           # 中心柱の高さ / ハブ高さ
    # 中心柱の頭に載せる面取りの円錐台。以前は build_static() に直書きしてあった
    "column_cap_r_x_column_r": 1.25,   # 面取りの下の半径 / 中心柱の半径
    "column_cap_top_x_column_r": 0.9,  # 同 上の半径 / 中心柱の半径
    "column_cap_h_x_plate_t": 0.8,     # 同 高さ / テーブル板厚
    "nozzle_bore_x_neck_d": 0.75,      # ノズル外径 / ボトル首径
    "nozzle_len_x_body_h": 0.62,       # ノズル長さ / ボトル胴高さ
    "nozzle_gap_x_neck_h": 0.60,       # ノズル先端とボトル口の隙間 / 首高さ
    "arm_thickness_x_plate_t": 1.2,    # ノズル支持アームの厚み / テーブル板厚
    "arm_width_x_nozzle_bore": 1.9,    # 同 幅 / ノズル外径
    "post_r_x_plate_r": 0.055,         # ノズル支柱の半径 / テーブル半径
    "post_at_x_plate_r": 1.10,         # 支柱を立てる半径 / テーブル半径（外に立てる）
    # 架台天板の直径。カバーの支柱と枠（半径 371.7 まで）と脚の外角（374.6）を
    # 天板の上に載せきる大きさが要る。1.24（半径 347.2）だと支柱が天板の縁から
    # 20 mm はみ出して宙に立っていた。1.35 = 半径 378。
    "deck_d_x_plate_d": 1.35,          # 架台天板の直径 / テーブル外径
    # 安全カバー。ノズル支柱（半径 308 に太さ 15.4 で立つ）の外を通す。
    # 1.15（322）だと支柱をかすめ、しかも受け渡し部を塞いだ。
    # **この値は板の内面（内法）。** 板厚・枠・支柱はここから外へ積むので、
    # 外へ張り出す量は parts.cover() を見ること（枠の外面は 371.7）。
    "cover_r_x_plate_r": 1.22,         # 安全カバー半径（内法）/ テーブル半径
    "cover_top_margin_x_plate_t": 2.5, # カバー上端がアームより上に出る量
    "cover_wall_x_cover_r": 0.012,     # カバー板の厚み / カバー半径
    # 受け渡し部の開口。スターホイールが張り出す角度に、左右の余裕を足した幅で
    # 開ける。スターホイールの寸法が lay に無いときの控えが cover_open_half_deg。
    "cover_open_half_deg": 18.0,       # 開口の半幅 [deg]（控え）
    "cover_open_margin_deg": 8.0,      # 開口の左右に見る余裕 [deg]
    "cover_post_x_frame_pipe": 0.45,   # カバーの支柱の一辺 / 架台の角パイプ一辺
    "stream_r_x_neck_d": 0.16,         # 充填中の液柱の半径 / ボトル首径
    "rim_thickness_x_liquid_r": 0.075, # 液面の縁の線の太さ / 液の半径
    "edge_thickness_x_wall": 0.85,     # ボトル稜線の輪の太さ / ボトル肉厚
    "bottle_sink_x_plate_t": 0.06,     # ボトル底をテーブル面へ沈める量 / テーブル板厚
    # 接地部の暗がり。ボトルの際から外へ 3 段で薄くする。すべてボトル外径比。
    # 最外はテーブル板の縁を越えないこと。越えると板の無いところに暗がりだけが
    # 浮く（以前 1.90 = 64.6 で、ピッチ円 225 と足して 289.6 > 板半径 280 だった）。
    # 越えた場合は build_carousel() が板の縁で頭を抑える。
    "shade_radii_x_bottle_r": (0.96, 1.12, 1.30, 1.52),
    # ポケット板。テーブル板の上に載せる、ボトルを受ける U 溝の板。
    # 溝の口の幅は外周円とポケット円の交点から決まる。
    #   x = (R^2 - p^2 + c^2) / (2c)      口の幅 = 2 * sqrt(R^2 - x^2)
    # ここでは R = ピッチ円 + 出しろ、c = ピッチ円、p = ボトル半径 + 逃げ。
    # 出しろを詰めると口が狭まってボトルが入らなくなる。
    #
    # **出しろは 10 -> 6 に詰めてある（R235 -> R231）。** スターホイールが
    # 抱えるボトルは半径 112.5 の円弧で入ってくるので、受け渡し点を外れると
    # ポケット板の外周へ寄る。出しろ 10 だと、テーブル角 15.95deg（星車が
    # 受け渡し点から 31.9deg 回ったところ）でボトルが板の材へ 2.22 mm 食い込む
    # （食い込み = ボトル円を板の材から離すのに要る最短の移動量）。
    # 出しろ 6 で -0.37 mm、つまり隙間が 0.37 mm 残って接触しなくなる。
    # 口の幅は 73.75 で、ボトル外径 68 に対して +5.75。詰めても口は広がる側
    # （外周が下がると交点が溝の奥に寄る）ので、出入りは楽になる。
    "pocket_lip_x_plate_t": 0.75,      # 板の外半径をピッチ円から外へ出す量 / テーブル板厚
    "pocket_clear_x_plate_t": 0.375,   # 溝半径のボトル半径からの逃げ / テーブル板厚
    "pocket_t_x_plate_t": 1.25,        # ポケット板の板厚 / テーブル板厚
    "floor_r_x_plate_d": 20.0,         # 床面の半径 / テーブル外径
    "demo_fill_progress": 0.35,        # 見本の絵で、充填中のボトルをどこまで入れるか
    "demo_tilt_deg": 11.0,             # 見本の絵で、いちばん揺れているボトルの傾き
}

# 工程配置の世界角 [deg]。正典は params.json の stations で、ここにあるのは
# params を渡されなかったときの控え。**控えはここ 1 つだけ。** 以前は
# ノズル角・供給角・排出角の控えがモジュールごとに散らばっていて、しかも
# derive_layout() がモジュール変数を書き換えて回っていた（呼ぶ順で答えが変わる）。
# 角度が要る関数には引数で渡す。
DEFAULT_STATION_DEG = {"fill": 0.0, "infeed": 315.0, "discharge": 225.0}
# 液柱を描く許容ずれ [deg]。ステーションがこれ以上ノズルからずれていたら
# 注いでいる絵にはしない（割出しの途中で液柱が宙に浮くのを防ぐ）。
STREAM_ALIGN_TOL_DEG = 6.0

# 材質。PBR で描く。
#
# **振り分けの指針（asm_*.py も同じ規律で揃えること）**
# 実機の写真で機構がすぐ読めるのは、部材ごとに仕上げが違うからで、明度と
# 光沢の差がそのまま部品の切れ目になっている。全部を同じ明るいステンレスに
# すると、真上から見たときに架台も回転部もガイドも 1 つの白い塊になる。
# 下の 5 群に振り分ける。**新しい材質を足すときは、どれかの群に寄せる。**
#
#   1 艶消しの白い樹脂（ガイド類）  metallic 0.00 / roughness 0.40〜0.50
#     三日月ガイド・外周ガイド・コンベアのサイドガイド。金属光沢は載せない。
#     色は #e8ecef 前後。**白いが光らない**ので、光る回転部と分かれて見える。
#   2 ヘアライン金属（架台・天板・据付もの）  metallic 0.40〜0.55 / roughness 0.68〜0.80
#     角パイプ・天板・軸受台・門型の柱。#68〜#7f の中明度に落とす。
#   3 明るい金属（回るもの・配管・タンク）  metallic 0.30〜0.40 / roughness 0.50〜0.62
#     テーブル板・ポケット板・星車板。**metallic を上げすぎない。**
#     0.85 付近だと真上から見たとき環境マップの天井（ほぼ白）をそのまま映して
#     白飛びし、板の輪郭も溝の切り欠きも消える。上げるのは metallic ではなく
#     色の明度で、映り込みは roughness で散らす。
#   4 黒（脚のゴム座・チェーン・センサ）  metallic 0.00 / roughness 0.85〜0.95
#     床と機械の境目をここで作る。
#   5 塗装（制御箱・板金カバーの枠・ギヤモータ）  metallic 0.10〜0.35 / roughness 0.60〜0.75
#     つや消しで、わずかに色味を入れる。金属より暗い。
#
# ボトル = 薄いガラス（中の液面は透けるが、肩と首の輪郭は残る濃さ）。
# 液 = 水。ごく淡い青の半透明 / カバーの板 = ごく薄い。
# 液の色は params.json の liquid.name（water）に合わせてある。以前は琥珀色で、
# 記事の「水で計算した」という前提と絵が食い違っていた。無色に振り切ると
# ガラスと見分けが付かなくなるので、水色を薄く残して液面の高さを読ませる。
# **液面は記事の主役なので、機械の側を暗く落としても液の色は動かさない。**
# 縁（rim）は液面の線。液より濃くしないと、真横から見たとき面が消える。
# shade_* は接地部の暗がり。影が使えないので、面を暗く汚して代わりにする。
# 陰影を付けると向きで明るさが変わってしまうので lighting は切ってある。
MATERIAL = {
    # 2 ヘアライン金属。架台の角パイプと桁。天板より暗くして、縦の材と
    # 水平の板を分ける
    "frame":     dict(color="#6a727a", pbr=True, metallic=0.46, roughness=0.78),
    # 2 同上。架台天板。真上図でいちばん面積が広いので艶を落とす
    "deck":      dict(color="#7b838b", pbr=True, metallic=0.44, roughness=0.72),
    # 4 黒。アジャスタの丸柱＝床に着く脚。機械の底をここで切る
    "foot":      dict(color="#23262a", pbr=True, metallic=0.00, roughness=0.92),
    # 3 明るい金属だが回らない側（旋回軸受リング・中心柱・ノズルとその支持）。
    # 細い丸物ばかりで面積が小さいので、回る板より光らせてよい
    "steel":     dict(color="#c8cfd6", pbr=True, metallic=0.66, roughness=0.38),
    # 3 回るもの（テーブル板・ハブ）。**ここが白飛びの主因だった。**
    # 以前は steel と同じ metallic 0.88 / roughness 0.24 で、真上から見ると
    # 板が真っ白に飛んでポケットの切り欠きすら読めなかった
    "steel_rot": dict(color="#adb6be", pbr=True, metallic=0.34, roughness=0.58),
    # 3 同じ「回る明るい金属」だが、ポケット板だけ一段明るくする。テーブル板の
    # 上に載る別の板で、U 溝の切り欠きがボトルを抱える所そのものなので、
    # 同じ色にすると真上図で板の縁が消えて溝が読めない
    "steel_pocket": dict(color="#c3cbd2", pbr=True, metallic=0.36, roughness=0.52),
    # 5 塗装。安全カバーの枠と支柱。板は透明でも枠は塗ってある
    "cover_frame": dict(color="#59636e", pbr=True, metallic=0.14, roughness=0.72),
    "shade_0":   dict(color="#242b33", lighting=False, opacity=0.26),
    "shade_1":   dict(color="#2b333b", lighting=False, opacity=0.15),
    "shade_2":   dict(color="#343c45", lighting=False, opacity=0.07),
    "bottle":    dict(color="#e4eef6", pbr=True, metallic=0.0, roughness=0.06, opacity=0.30),
    "glass":     dict(color="#f4fbff", pbr=True, metallic=0.30, roughness=0.04, opacity=0.72),
    # 液は水（params.json の liquid.name = water）。無色透明が実物だが、それだと
    # 液面の高さも傾きも絵から読めない。記事の主役は液面なので、技術図の慣例どおり
    # 淡い青を付けて濃さで読ませる。琥珀色にすると水ではなくジュースに見える。
    "liquid":    dict(color="#6fb8d6", pbr=True, metallic=0.0, roughness=0.10, opacity=0.80),
    "rim":       dict(color="#1d5877", pbr=True, metallic=0.0, roughness=0.55, opacity=1.0),
    "cover":     dict(color="#c2ced6", pbr=True, metallic=0.0, roughness=0.05, opacity=0.07),
    # 床。機械より暗く、彩度を落とす。ここを機械と同じ明るさにすると、
    # 真上図で機械の輪郭が背景に溶ける
    "floor":     dict(color="#a3abb2", pbr=True, metallic=0.0, roughness=0.95),
}

# 寒色寄りの無地。機械でいちばん明るいガイド類（白い樹脂）より明るくしない。
# 明るくすると、真上図で機械の外形が背景に飲まれる
BACKGROUND = "#cfd6dc"

# 描く順。(メッシュ群の名前, 当てる材質) の並びで、これが唯一の表。
# 深度ピーリングがこの環境（ソフトウェア OpenGL）では使えず、半透明の面が
# まるごと落ちる。使わずに描く順だけで前後を合わせているので、この並びが崩れると
# 絵が壊れる。不透明を先に、透過は奥から順に重ねる。接地部の暗がりはテーブル面の
# すぐ上に貼るので金属の直後に置く。
#
# 静止画（render）もコマ送り（animate.FrameRenderer）も必ずこれを読むこと。
# 部品を足したらここに 1 行足す。足し忘れるとその部品は絵に出ない。
#
# ここに書いてあるのは scene.py が組む部品だけ。asm_*.py の断片は読み込み時に
# ("frame", "frame") の直後へ差し込まれる（下の compose_draw_order）。
DRAW_ORDER = (
    ("floor", "floor"),
    ("frame", "frame"),          # 架台の角パイプ（脚と桁）
    ("deck", "deck"),            # 架台天板
    ("foot", "foot"),            # アジャスタの丸柱（床に着く）
    ("steel", "steel"),          # 回らない金属（旋回軸受リング・ノズル・中心柱）
    ("steel_rot", "steel_rot"),  # 回る金属（テーブル板・ハブ）
    ("steel_pocket", "steel_pocket"),  # 回る金属（ポケット板）
    ("cover_frame", "cover_frame"),   # 安全カバーの枠と支柱（不透明）
    ("shade_2", "shade_2"),
    ("shade_1", "shade_1"),
    ("shade_0", "shade_0"),
    ("rim", "rim"),              # 液面の縁（毎コマ作り直す）
    ("liquid", "liquid"),        # 液（毎コマ作り直す）
    ("stream", "liquid"),        # 充填中の液柱（毎コマ作り直す。回らない）
    ("glass", "glass"),
    ("bottle", "bottle"),
    ("cover", "cover"),
)

# テーブルと一緒に回る群。コマ送りではメッシュを組み直さず、アクタの変換行列
# だけ差し替えて回す。**この表も DRAW_ORDER と同じでここが唯一。** 以前は
# animate.py 側にも同じ並びを写していて、片方だけ古くなると「絵には出るが
# 回らない」部品ができた。ROTATING=True のモジュールの群は
# rotating_groups() がここへ足す。
ROTATING_GROUPS = ("steel_rot", "steel_pocket", "shade_2", "shade_1", "shade_0",
                   "rim", "liquid", "glass", "bottle")

# build_carousel() が返す群のうち、コマ送りで別のアクタに分けるもの。
# **いまは空。** build_carousel() が最初から回る側だけの群名（"steel_rot" /
# "steel_pocket"）を返すようにしたので
# 読み替える先が無い。以前は回る板も回らない金属も "steel" で返っていて、
# (a) コマ送りで 1 つのアクタに同居して回す変換が架台まで巻き込む、
# (b) 静止画では merge_groups() が両方を "steel" に混ぜるので、回る板だけ
# 材質を変えることができない、の 2 つが起きていた。読み替えの仕掛けは
# 残してあるので、群名がまた衝突したらここに 1 行足せば分けられる。
CAROUSEL_ALIAS: dict = {}

# 構図は cameras.py が唯一の表。ここには持たない（以前あった CAMERA は
# cameras.CAMERAS["iso"] へ移した。画角の決め方はその後、機械の外接円柱を
# 枠に当てはめるやり方に変えてある）。


# --------------------------------------------------------------------------
# 組み立てモジュールの読み込みと合成
# --------------------------------------------------------------------------
# 機械の組み立てはサブシステムごとに viz/asm_*.py へ割ってある。各モジュールが
# 公開する形は viz/ASSEMBLY_CONTRACT.md が正典。ここはそれを読み込んで
# MATERIAL / DRAW_ORDER / derive_layout / build_* に合成するだけ。
#
# まだ書かれていないモジュールがあっても落ちない。無いものは標準エラーに
# 1 行出して飛ばす。書きかけで壊れているものも同じ扱いにする（4 人が同時に
# 書いているあいだ、絵を焼く手が止まらないようにするため）。
ASM_MODULES = ("asm_drive", "asm_transfer", "asm_conveyor", "asm_details")

# モジュールの断片を DRAW_ORDER のどこへ差し込むか。モジュールの部品は
# すべて不透明なので、透過物（暗がり・液・ガラス・ボトル）より先に描く。
DRAW_ORDER_ANCHOR = ("frame", "frame")


@dataclass
class Assembly:
    """読み込めた組み立てモジュール 1 つ。

    spin_centers は**モジュールの `SPIN_CENTERS` そのもの**を持つ。値を写して
    はいけない。写すと `layout()` が中身を入れ替えても基盤側に届かず、
    自転の中心だけが読み込み時の値に取り残される（絵は出るので気付けない）。
    モジュール側が中身を入れ替える辞書でも、読まれた時に組む Mapping でも、
    同じように最新の値が見える。
    """

    name: str
    module: object
    materials: dict
    draw_order: tuple
    rotating: bool
    proportion: dict
    spin_centers: object

    def layout(self, params: dict, lay: dict) -> dict:
        fn = getattr(self.module, "layout", None)
        return dict(fn(params, dict(lay))) if fn is not None else {}

    def build(self, params: dict, lay: dict) -> dict:
        fn = getattr(self.module, "build", None)
        return dict(fn(params, lay)) if fn is not None else {}

    def spin_angles(self, params: dict, lay: dict, state) -> dict:
        """群名 -> 自転角 [rad]。公開していなければ空。"""
        fn = getattr(self.module, "spin_angles", None)
        if fn is None:
            return {}
        return {k: float(v) for k, v in dict(fn(params, lay, state)).items()}


def _warn(message: str) -> None:
    """絵から部品が消えたときに、原因がここだと分かるように出す。"""
    print(f"[scene] {message}", file=sys.stderr)


def load_assemblies(names=ASM_MODULES) -> list:
    """asm_*.py を読み込む。無いもの・壊れているものは飛ばす。

    契約に反しているもの（材質名や群の名前の衝突）は飛ばさずに止める。
    黙って上書きすると、どちらの部品が消えたのか絵からは分からない。
    """
    found = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:                      # 中で別のものを import し損ねた
                _warn(f"{name}: 読み込めない（{exc}）。飛ばす")
            else:
                _warn(f"{name}: まだ無い。飛ばす")
            continue
        except Exception as exc:                      # noqa: BLE001 書きかけで壊れている
            _warn(f"{name}: 読み込めない（{type(exc).__name__}: {exc}）。飛ばす")
            continue

        draw_order = tuple(tuple(row) for row in getattr(module, "DRAW_ORDER", ()))
        # **参照のまま持つ。`dict(...)` で写さない。**（Assembly の注記）
        spin = getattr(module, "SPIN_CENTERS", None) or {}
        if spin and getattr(module, "spin_angles", None) is None:
            _warn(f"{name}: SPIN_CENTERS はあるが spin_angles() が無い。自転しない")
            spin = {}
        found.append(Assembly(name=name, module=module,
                              materials=dict(getattr(module, "MATERIALS", {})),
                              draw_order=draw_order,
                              rotating=bool(getattr(module, "ROTATING", False)),
                              proportion=dict(getattr(module, "PROPORTION", {})),
                              spin_centers=spin))
        if not draw_order:
            _warn(f"{name}: DRAW_ORDER が空。このモジュールの部品は絵に出ない")
        if getattr(module, "build", None) is None:
            _warn(f"{name}: build() が無い。材質と寸法だけ受け取る")
    return found


def merge_materials(material: dict, assemblies) -> dict:
    """モジュールの材質を取り込む。名前が衝突したら止める。"""
    owner = {k: "scene" for k in material}
    for asm in assemblies:
        for key, value in asm.materials.items():
            if key in owner:
                raise RuntimeError(
                    f"材質名の衝突: {key!r} は {owner[key]} と {asm.name} の両方にある。"
                    "モジュール接頭辞を付けて分けること")
            owner[key] = asm.name
            material[key] = value
    return material


def compose_draw_order(base, assemblies) -> tuple:
    """モジュールの断片を DRAW_ORDER_ANCHOR の直後へ差し込む。

    群の名前が衝突したら止める。同じ名前だと片方の部品が絵から消えるが、
    どちらが消えたのかは絵を見ても分からない。
    """
    base = tuple(base)
    owner = {name: "scene" for name, _ in base}
    fragment = []
    for asm in assemblies:
        for row in asm.draw_order:
            if len(row) != 2:
                raise RuntimeError(f"{asm.name}: DRAW_ORDER の行は (群, 材質) の 2 つ組: {row!r}")
            group, material = row
            if group in owner:
                raise RuntimeError(
                    f"メッシュ群の名前の衝突: {group!r} は {owner[group]} と {asm.name} の"
                    "両方にある。モジュール接頭辞を付けて分けること")
            if material not in MATERIAL:
                raise RuntimeError(
                    f"{asm.name}: 群 {group!r} が知らない材質 {material!r} を指している"
                    f"（MATERIALS に入れ忘れ）")
            owner[group] = asm.name
            fragment.append((group, material))

    if not fragment:
        return base
    try:
        at = base.index(DRAW_ORDER_ANCHOR) + 1
    except ValueError:                                # 差し込み先が消えたら末尾へ
        _warn(f"DRAW_ORDER に {DRAW_ORDER_ANCHOR} が無い。モジュールの断片を末尾に置く")
        at = len(base)
    return base[:at] + tuple(fragment) + base[at:]


def compose_spin_centers(assemblies) -> dict:
    """モジュールの SPIN_CENTERS を 1 つの表にまとめる。

    返すのは 群名 -> (モジュール名, (x, y))。同じ群を 2 つのモジュールが
    自転させようとしたら止める。**呼ばれた時のモジュールの中身で組む。**
    """
    out: dict = {}
    for asm in assemblies:
        for group, center in dict(asm.spin_centers).items():
            if group in out:
                raise RuntimeError(
                    f"自転する群の重複: {group!r} を {out[group][0]} と {asm.name} の"
                    "両方が回そうとしている")
            out[group] = (asm.name, (float(center[0]), float(center[1])))
    return out


class _SpinCenterTable(Mapping):
    """群名 -> (モジュール名, (x, y))。**読まれた瞬間にモジュールから引き直す。**

    以前はここが読み込み時に組んだ普通の辞書で、モジュールが `layout()` で
    `SPIN_CENTERS` を入れ替えても基盤側には届かなかった。いまたまたま正しく
    回っていたのは、モジュールが読み込み時に params.json を直に読んで同じ値を
    出していたからで、契約が定めた経路（`layout()` が入れ替える）は死んでいた。
    自転の中心を lay から決めるようになった瞬間、星車が別の点まわりに回る。
    **絵は出るので気付けない。**

    中身を写さないので、`dict(scene.SPIN_CENTERS)` で控えを取る側（animate）も
    その時点の最新の値を受け取る。
    """

    def _resolve(self) -> dict:
        return compose_spin_centers(ASSEMBLIES)

    def __getitem__(self, key):
        return self._resolve()[key]

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self):
        return len(self._resolve())

    def __repr__(self):
        return repr(self._resolve())


ASSEMBLIES = load_assemblies()
MATERIAL = merge_materials(MATERIAL, ASSEMBLIES)
DRAW_ORDER = compose_draw_order(DRAW_ORDER, ASSEMBLIES)
# 読み込み時にいちど組んで、重複と形を確かめておく（黙って壊れているより、
# 読み込みで止まったほうが原因が分かる）。使うのは下の生きた表のほう。
compose_spin_centers(ASSEMBLIES)
SPIN_CENTERS = _SpinCenterTable()


def rotating_groups() -> tuple:
    """テーブルと一緒に回る群の名前。**animate はここを読むだけ。**

    scene が組む分（ROTATING_GROUPS）に、ROTATING=True のモジュールが
    DRAW_ORDER に載せた群を足して返す。自転する群はテーブル軸まわりには
    回らないので入らない（自転は spin_matrices() が別に当てる）。
    """
    out = list(ROTATING_GROUPS)
    for asm in ASSEMBLIES:
        if not asm.rotating:
            continue
        out += [g for g, _ in asm.draw_order if g not in out]
    spinning = dict(SPIN_CENTERS)                     # 引き直しは 1 回で足りる
    return tuple(g for g in out if g not in spinning)


# --------------------------------------------------------------------------
# 機械の状態
# --------------------------------------------------------------------------
@dataclass
class MachineState:
    """1 コマぶんの機械の状態。解析側の出力をそのまま入れられる形にしてある。

    table_angle_rad : テーブルの回転角
    cam_angle_rad   : カム入力軸の角 psi。1 タクトで 0 -> 2pi。回りっぱなしで
                      積み上がってよい。テーブル角も自転角もここから出る。
                      **テーブル角だけでは停留中のどこにいるかが分からない**
                      ので、自転する部品（スターホイール）はこちらを見る。
                      既定は 0（渡さなければ割出しの始まり）。
    volumes_mL      : ステーションごとの液量
    tilt_t, tilt_r  : ステーションごとの液面の傾き [rad]。
                      t は接線方向（回転が進む向き）、r は半径方向（外向き）。
                      正のとき、その向きの側で液面が上がる。
    filling_index   : いま注いでいるステーション。None なら液柱を描かない。
    bottle_present  : ステーションごとに、ホルダにボトルが載っているか。
                      **空の列を渡すと station_present() の定常状態になる。**
                      全部埋まった絵にしたいときは [True] * n を明示する。
                      実機のテーブルは供給から排出までの区間しか埋まらない。
    """

    table_angle_rad: float = 0.0
    volumes_mL: list = field(default_factory=list)
    tilt_t: list = field(default_factory=list)
    tilt_r: list = field(default_factory=list)
    filling_index: int | None = None
    cam_angle_rad: float = 0.0
    bottle_present: list = field(default_factory=list)


def load_params(path=PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


# --------------------------------------------------------------------------
# カム入力軸角 psi から機械の姿勢を出す
#
# 入力軸は一定回転で、1 タクトで 1 回転する。割付 180 度でテーブルが 45 度
# 進み、残り 180 度は停留。**テーブル角も、スターホイールの自転角も、
# ここから出す。** 割出しと停留のどちらにいるかはテーブル角からは読めない
# （停留中はテーブル角が動かないので、停留のどこにいるか分からない）ので、
# 交互に動くものを描くには psi が要る。
# --------------------------------------------------------------------------
def cam_angle_from_time(params: dict, t):
    """時刻 [s] -> カム入力軸角 psi [rad]。積み上がった角を返す。"""
    tact = float(params["cycle"]["tact_s"])
    return 2.0 * np.pi * np.asarray(t, dtype=float) / tact


def cam_index_fraction(params: dict) -> float:
    """入力軸 1 回転のうち、割付が占める割合。残りは停留。"""
    idx = params.get("indexer", {})
    a_idx = float(idx.get("index_angle_input_deg", 180.0))
    a_dwl = float(idx.get("dwell_angle_input_deg", 180.0))
    total = a_idx + a_dwl
    return a_idx / total if total > 0.0 else 0.5


def cam_phase(params: dict, cam_angle_rad):
    """psi -> (割出しの進み具合 x, 何回目の割出しか)。

    x は割付区間を 0..1 に正規化した値で、停留中は 1 に張り付く。
    """
    psi = np.asarray(cam_angle_rad, dtype=float) / (2.0 * np.pi)
    turn = np.floor(psi)
    x = np.clip((psi - turn) / cam_index_fraction(params), 0.0, 1.0)
    return x, turn


def _modified_sine(x):
    """変形正弦の無次元 変位・速度・加速度。式は panels.py が持っている。

    panels は線図を描くために matplotlib を読む。絵を組むだけのときに
    そこまで抱えたくないので、呼ぶときに読み込む。
    """
    import panels                                   # noqa: PLC0415
    return panels.modified_sine(np.asarray(x, dtype=float))


def table_angle_from_cam(params: dict, cam_angle_rad):
    """カム入力軸角 psi [rad] -> テーブル角 [rad]。回りっぱなしで積み上がる。"""
    x, turn = cam_phase(params, cam_angle_rad)
    d_th = np.radians(float(params["table"]["index_angle_deg"]))
    s, _v, _a = _modified_sine(x)
    out = d_th * (turn + s)
    return float(out) if np.ndim(cam_angle_rad) == 0 else out


def table_alpha_from_cam(params: dict, cam_angle_rad):
    """カム入力軸角 psi [rad] -> テーブルの角加速度 [rad/s2]。

    入力軸が一定回転なので、psi と時刻は比例する。割付の実時間で割って戻す。
    """
    x, _turn = cam_phase(params, cam_angle_rad)
    d_th = np.radians(float(params["table"]["index_angle_deg"]))
    t_idx = float(params["cycle"]["tact_s"]) * cam_index_fraction(params)
    _s, _v, a = _modified_sine(x)
    alpha = np.asarray(a, dtype=float) * d_th / t_idx ** 2
    # 停留中は動かない。x が 1 に張り付いている区間は加速度も 0
    alpha = np.where(np.asarray(x) >= 1.0, 0.0, alpha)
    return float(alpha) if np.ndim(cam_angle_rad) == 0 else alpha


def station_deg(params, which="fill") -> float:
    """工程配置の世界角 [deg]。params.json の stations が正典。

    params を渡さない（渡せない）ときだけ DEFAULT_STATION_DEG の控えを使う。
    控えはこの 1 箇所だけで、他のモジュールもここを通す。
    """
    if params is not None:
        value = params.get("stations", {}).get(f"{which}_deg")
        if value is not None:
            return float(value)
    return float(DEFAULT_STATION_DEG[which])


def _mouth_width(outer_r: float, pocket_r: float, pocket_at_r: float) -> float:
    """U 溝の口の開き。外周円とポケット円の交点から出る。

    ボトル外径を下回ると、そもそもボトルが溝に入らない。
    """
    x = (outer_r ** 2 - pocket_r ** 2 + pocket_at_r ** 2) / (2.0 * pocket_at_r)
    if abs(x) >= outer_r:
        return 0.0
    return 2.0 * float(np.sqrt(outer_r ** 2 - x ** 2))


def derive_layout(params: dict) -> dict:
    """params.json と PROPORTION から、機械各部の絶対寸法と高さを決める。

    ここが唯一の「どこに何があるか」の表。組み立てはこれを読むだけ。
    asm_*.py の layout() が返した寸法もここに合流する（キーが衝突したら止める）。
    """
    b = params["bottle"]
    t = params["table"]
    p = PROPORTION

    plate_d = t["plate_diameter_mm"]
    plate_t = t["plate_thickness_mm"]
    plate_r = plate_d / 2.0
    pitch_r = t["pitch_circle_diameter_mm"] / 2.0

    bottle_r = b["inner_diameter_mm"] / 2.0 + b["wall_thickness_mm"]
    bottle_h = b["body_height_mm"] + b["shoulder_height_mm"] + b["neck_height_mm"]

    # 下から順に積む
    deck_top = plate_d * p["work_height_x_plate_d"]
    deck_t = plate_t * p["deck_thickness_x_plate_t"]
    bearing_h = plate_t * p["bearing_height_x_plate_t"]
    table_base = deck_top + bearing_h
    table_top = table_base + plate_t
    hub_h = plate_t * p["hub_height_x_plate_t"]
    bottle_top = table_top + bottle_h

    nozzle_tip = bottle_top + b["neck_height_mm"] * p["nozzle_gap_x_neck_h"]
    nozzle_len = b["body_height_mm"] * p["nozzle_len_x_body_h"]
    nozzle_bore = b["neck_diameter_mm"] * p["nozzle_bore_x_neck_d"]
    arm_base = nozzle_tip + nozzle_len
    arm_t = plate_t * p["arm_thickness_x_plate_t"]
    arm_top = arm_base + arm_t

    cover_r = plate_r * p["cover_r_x_plate_r"]
    cover_top = arm_top + plate_t * p["cover_top_margin_x_plate_t"]

    # ポケット板。テーブル板の上に直に載る。溝の中心はピッチ円上。
    pocket_r = bottle_r + plate_t * p["pocket_clear_x_plate_t"]
    pocket_plate_r = pitch_r + plate_t * p["pocket_lip_x_plate_t"]
    pocket_t = plate_t * p["pocket_t_x_plate_t"]

    # 工程配置。世界角 [deg]。ノズルは fill_deg に固定してある。
    # 供給と排出はスターホイールとコンベアの担当が読む。
    fill_deg = station_deg(params, "fill")
    infeed_deg = station_deg(params, "infeed")
    discharge_deg = station_deg(params, "discharge")

    # 架台の脚の向き。parts.frame() は対角に立てるので、そこから振る。
    # **脚の世界角はここが唯一の表。** アジャスタ座・配線ダクト・操作盤・
    # 腰板は脚に付くので、各モジュールはこの frame_leg_deg / frame_leg_xy を
    # 読むこと（45/135/225/315 度を直書きしない）。
    frame_span = plate_d * p["frame_span_x_plate_d"]
    leg_phase = float(p["frame_leg_phase_deg"])
    leg_r = frame_span / 2.0 * np.sqrt(2.0)
    leg_deg = tuple(float((45.0 + leg_phase + 90.0 * k) % 360.0) for k in range(4))

    lay = {
        "stations": int(t["stations"]),
        "pitch_r": pitch_r,
        "plate_d": plate_d,
        "plate_r": plate_r,
        "plate_t": plate_t,
        "bottle_r": bottle_r,
        "bottle_h": bottle_h,
        "liquid_r": b["inner_diameter_mm"] / 2.0,
        "body_h": b["body_height_mm"],
        "deck_top": deck_top,
        "deck_t": deck_t,
        # 架台天板の直径。以前はカバー半径から作っていたので、カバーを広げると
        # 架台まで太った。別々の値にして切り離してある。
        "deck_d": plate_d * p["deck_d_x_plate_d"],
        "frame_pipe": plate_d * p["frame_pipe_x_plate_d"],
        "frame_span": frame_span,
        # 脚の向きと位置。**frame_leg_deg は脚 1 本目の世界角（1 つの数）。**
        # 残りの 3 本はここから 90 度ずつ。4 本ぶんの並びが要るなら
        # frame_leg_deg_all、位置が要るなら frame_leg_xy（同じ順の (x, y)）。
        # frame_leg_r は芯の半径、frame_leg_phase_deg は parts.frame() の既定
        # （対角 45/135/225/315）からの振り。架台に付く部品はここを読む。
        "frame_leg_phase_deg": leg_phase,
        "frame_leg_deg": leg_deg[0],
        "frame_leg_deg_all": leg_deg,
        "frame_leg_r": leg_r,
        "frame_leg_xy": tuple((float(leg_r * np.cos(np.radians(a))),
                               float(leg_r * np.sin(np.radians(a))))
                              for a in leg_deg),
        "bearing_ro": plate_r * p["bearing_outer_x_plate_r"],
        "bearing_ri": plate_r * p["bearing_inner_x_plate_r"],
        "bearing_h": bearing_h,
        "table_base": table_base,
        "table_top": table_top,
        "hub_d": plate_d * p["hub_dia_x_plate_d"],
        "hub_h": hub_h,
        "column_r": plate_r * p["column_r_x_plate_r"],
        "column_base": table_top + hub_h,
        "column_top": table_top + hub_h * (1.0 + p["column_h_x_hub_h"]),
        "bottle_top": bottle_top,
        "nozzle_tip": nozzle_tip,
        "nozzle_len": nozzle_len,
        "nozzle_bore": nozzle_bore,
        # 固定ノズルの支持。テーブルとは縁が切れていて一緒には回らない
        "arm_base": arm_base,
        "arm_t": arm_t,
        "arm_top": arm_top,
        "arm_w": nozzle_bore * p["arm_width_x_nozzle_bore"],
        "post_r": plate_r * p["post_r_x_plate_r"],
        "post_at_r": plate_r * p["post_at_x_plate_r"],
        "cover_r": cover_r,
        "cover_base": deck_top,
        "cover_h": cover_top - deck_top,
        "total_height": cover_top,
        "floor_r": plate_d * p["floor_r_x_plate_d"],
        # ポケット板（テーブルと一緒に回る）
        "pocket_plate_r": pocket_plate_r,
        "pocket_r": pocket_r,
        "pocket_t": pocket_t,
        "pocket_base": table_top,
        "pocket_top": table_top + pocket_t,
        "pocket_mouth_w": _mouth_width(pocket_plate_r, pocket_r, pitch_r),
        # 工程配置 [deg]
        "fill_deg": fill_deg,
        "infeed_deg": infeed_deg,
        "discharge_deg": discharge_deg,
    }

    if lay["pocket_mouth_w"] <= 2.0 * bottle_r:
        _warn(f"ポケット板の口の幅 {lay['pocket_mouth_w']:.2f} が"
              f"ボトル外径 {2.0 * bottle_r:.2f} 以下。ボトルが溝に入らない")

    # asm_*.py の寸法を合流させる。キーが衝突したら止める（黙って上書きすると
    # どちらの寸法が効いているのか分からなくなる）。
    for asm in ASSEMBLIES:
        extra = asm.layout(params, lay)
        clash = sorted(set(extra) & set(lay))
        if clash:
            raise RuntimeError(
                f"{asm.name}.layout() のキーが既にある: {', '.join(clash)}。"
                "モジュール接頭辞を付けて分けること")
        lay.update(extra)

    # 安全カバーの開口。スターホイールの寸法（asm_transfer が出す）を読むので、
    # モジュールの寸法を合流させたあとで決める。
    lay["cover_wall"] = lay["cover_r"] * p["cover_wall_x_cover_r"]
    lay["cover_arcs"] = cover_arcs(lay)

    # 架台天板の下穴。テーブルを回す出力軸（asm_drive が出す）が通る。
    # 駆動が読み込まれていなければ穴は開けない（中実の円板のまま）。
    shaft_d = lay.get("drv_out_shaft_d")
    if shaft_d:
        lay["deck_bore_d"] = float(shaft_d) + 2.0 * plate_t * p["deck_bore_gap_x_plate_t"]
        if lay["deck_bore_d"] >= 2.0 * lay["bearing_ri"]:
            _warn(f"天板の下穴 {lay['deck_bore_d']:.1f} が旋回軸受リングの内径"
                  f" {2.0 * lay['bearing_ri']:.1f} 以上。リングが天板に載らない")
    else:
        lay["deck_bore_d"] = 0.0
    return lay


def _free_arcs(openings, step=0.5) -> list:
    """円周から openings を抜いた残りの区間を並べる。単位 deg。

    区間は (始まり, 終わり) で、始まりから反時計回りに終わりまで。
    重なった開口はここでまとまる。
    """
    n = int(round(360.0 / step))
    blocked = np.zeros(n, dtype=bool)
    for a0, a1 in openings:
        span = (float(a1) - float(a0)) % 360.0
        i0 = int(np.floor((float(a0) % 360.0) / step))
        k = int(np.ceil(span / step))
        blocked[(i0 + np.arange(k)) % n] = True
    if blocked.all():
        return []
    if not blocked.any():
        return [(0.0, 360.0)]

    # 円周なので、塞がれた bin から数え始めて切れ目をまたがないようにする
    start = int(np.argmax(blocked))
    runs = []
    run0 = None
    for j in range(n + 1):
        free = (not blocked[(start + j) % n]) if j < n else False
        if free and run0 is None:
            run0 = j
        elif not free and run0 is not None:
            runs.append(((start + run0) * step, (start + j) * step))
            run0 = None
    return [(a0 % 360.0, a0 % 360.0 + (a1 - a0)) for a0, a1 in runs]


def cover_arcs(lay: dict) -> tuple:
    """安全カバーの板を張る角度範囲 [(始まり, 終わり), ...]。単位 deg。

    受け渡し部は開ける。塞ぐとスターホイールを串刺しにし、コンベアの
    出入り口も無くなる。開口の半幅はスターホイールの張り出しから出す
    （中心半径 c に外半径 R の円盤が載っていれば、テーブル中心から見て
    asin(R/c) だけ広がる）。asm_transfer が寸法を lay に出していればそれを
    読み、無ければ PROPORTION の控えを使う。
    """
    p = PROPORTION
    center_r = float(lay.get("trf_center_r", 0.0))
    star_r = float(lay.get("trf_star_outer_r", 0.0))
    if center_r > star_r > 0.0:
        half = float(np.degrees(np.arcsin(star_r / center_r)))
    else:
        half = float(p["cover_open_half_deg"])
    half += float(p["cover_open_margin_deg"])
    openings = [(lay["infeed_deg"] - half, lay["infeed_deg"] + half),
                (lay["discharge_deg"] - half, lay["discharge_deg"] + half)]
    return tuple(_free_arcs(openings))


# --------------------------------------------------------------------------
# 液量と傾きの扱い
# --------------------------------------------------------------------------
def volume_to_level(params: dict, volume_mL: float) -> float:
    """液量 [mL] を液深 [mm] に直す。MODEL.md の h = V / (pi R^2)。

    液は胴の中だけを考える約束なので、胴高さで頭打ちにする。
    """
    r = params["bottle"]["inner_diameter_mm"] / 2.0
    level = float(volume_mL) * 1000.0 / (np.pi * r * r)
    return float(np.clip(level, 0.0, params["bottle"]["body_height_mm"]))


def tilt_to_world(tilt_t: float, tilt_r: float, station_angle_rad: float):
    """接線・半径方向の傾きを、傾きの大きさと「液面が上がる向き」に直す。

    接線方向の単位ベクトルは (-sin, cos)、半径方向は (cos, sin)。
    """
    vx = tilt_r * np.cos(station_angle_rad) - tilt_t * np.sin(station_angle_rad)
    vy = tilt_r * np.sin(station_angle_rad) + tilt_t * np.cos(station_angle_rad)
    return float(np.hypot(vx, vy)), float(np.arctan2(vy, vx))


def station_angles(n: int, table_angle_rad=0.0) -> np.ndarray:
    """各ステーションの方位角 [rad]。"""
    return float(table_angle_rad) + np.arange(n) * 2.0 * np.pi / n


def station_present(params: dict, lay: dict, table_angle_rad=0.0) -> list:
    """ホルダにボトルが載っているか。ステーション番号の順に並べて返す。

    実機のテーブルは全部のホルダが埋まってはいない。供給から排出までの
    区間だけがボトルを抱えて回り、排出から供給までの戻りは空で回る。

    判定は世界角で行う。テーブルが CCW に回るので、供給の世界角から
    反時計回りに排出の世界角の**手前まで**が「ボトル在り」。排出の位置は
    スターホイールが持ち去ったあとなので空。世界角はいちばん近い停留の
    ものを使う（割出しの途中でボトルが消えたり湧いたりしないように、
    割出し 1 回のあいだは載っているものが変わらない）。

    params.json の stations が 供給 315 / 充填 0 / 排出 225 のとき、
    テーブル角 0 では 0/45/90/135/180 と 315 に在り、225 と 270 が空になる。
    定常状態の内訳（満量 5 本 + 空瓶 1 本 + 空ホルダ 2 つ）と一致する。
    """
    n = int(lay["stations"])
    step = 360.0 / n
    infeed = float(lay.get("infeed_deg", station_deg(params, "infeed")))
    discharge = float(lay.get("discharge_deg", station_deg(params, "discharge")))
    span = (discharge - infeed) % 360.0
    if span <= 0.0:
        return [True] * n
    dwell = np.floor(np.degrees(float(table_angle_rad)) / step) * step
    return [bool(((dwell + i * step - infeed) % 360.0) < span - 1e-9)
            for i in range(n)]


def station_under_nozzle(n: int, table_angle_rad=0.0, nozzle_deg=None) -> int:
    """固定ノズルの真下にいちばん近いステーション番号。

    ノズルは世界座標に固定なので、テーブル角が変わると下に来る番号が変わる。
    ノズルの世界角は引数で渡す（`lay["fill_deg"]`）。省くと控えの 0 度。
    """
    if nozzle_deg is None:
        nozzle_deg = DEFAULT_STATION_DEG["fill"]
    diff = station_angles(n, table_angle_rad) - np.radians(float(nozzle_deg))
    return int(np.argmin(np.abs(np.arctan2(np.sin(diff), np.cos(diff)))))


def _angle_from_nozzle(angle_rad: float, nozzle_deg=None) -> float:
    """ノズルの世界角からのずれ [rad]。-pi..pi に畳む。"""
    if nozzle_deg is None:
        nozzle_deg = DEFAULT_STATION_DEG["fill"]
    d = float(angle_rad) - np.radians(float(nozzle_deg))
    return float(np.arctan2(np.sin(d), np.cos(d)))


def demo_state(params: dict, lay: dict, filling_index="auto",
               table_angle_rad=0.0, tilt_deg=None, cam_angle_rad=0.0) -> MachineState:
    """静止画用の見本の状態。

    filling_index は "auto" で、固定ノズルの真下に来ているステーションを
    充填中にする。テーブル角を振っても、注いでいるのは必ずノズルの下の
    1 本になる。None を渡すと充填中のボトルを作らない。
    tilt_deg を渡すと全ステーションを同じ傾きにする（傾きの見え方の確認用）。
    渡さないときは、ステーションごとに傾きを変えて並べる。割出し直後ほど
    揺れが残っている、という見立て。実際の値は解析側が出す。
    cam_angle_rad はカム入力軸の角。自転する部品（スターホイール）の姿勢は
    これで決まる。テーブル角と一緒に振りたいときは table_angle_from_cam()。
    """
    n = lay["stations"]
    if isinstance(filling_index, str):
        filling_index = station_under_nozzle(n, table_angle_rad, lay.get("fill_deg"))
    target = params["fill"]["target_volume_mL"]
    # 定常状態の内訳（満量 5 本 + 空瓶 1 本 + 空ホルダ 2 つ）に合わせる。
    # ボトルの無いホルダは空。供給されたばかりの 1 本はまだ入っていない。
    present = station_present(params, lay, table_angle_rad)
    # 供給されたばかりの 1 本。有無の判定と同じく、いちばん近い停留の角で引く
    # （割出しの途中で「空瓶」の役が隣のホルダへ飛び移らないように）。
    step = 360.0 / n
    dwell = np.radians(np.floor(np.degrees(table_angle_rad) / step) * step)
    infeed_index = station_under_nozzle(n, dwell, lay.get("infeed_deg"))
    volumes = [target if present[i] and i != infeed_index else 0.0
               for i in range(n)]
    if filling_index is not None and present[filling_index]:
        volumes[filling_index] = target * PROPORTION["demo_fill_progress"]
    elif filling_index is not None:
        filling_index = None            # 空ホルダの下では注がない

    if tilt_deg is None:
        peak = np.radians(PROPORTION["demo_tilt_deg"])
        tilts = [peak * float(np.exp(-0.42 * i)) for i in range(n)]
    else:
        tilts = [np.radians(float(tilt_deg))] * n

    return MachineState(table_angle_rad=table_angle_rad,
                        volumes_mL=volumes,
                        tilt_t=tilts,
                        tilt_r=[0.0] * n,
                        filling_index=filling_index,
                        cam_angle_rad=cam_angle_rad,
                        bottle_present=present)


# --------------------------------------------------------------------------
# 組み立て
# --------------------------------------------------------------------------
def _at(seq, index, default):
    """短い列が来ても落ちないように取り出す。"""
    try:
        return seq[index]
    except (IndexError, TypeError):
        return default


def _fill_angle_rad(lay: dict) -> float:
    """充填ステーションの世界角 [rad]。params.json の stations.fill_deg が正典。"""
    return np.radians(float(lay.get("fill_deg", DEFAULT_STATION_DEG["fill"])))


def _nozzle_matrix(lay: dict) -> np.ndarray:
    """固定ノズルの位置。充填ステーションの世界角のピッチ円上。"""
    th = _fill_angle_rad(lay)
    return parts.transform_matrix(
        translate=(lay["pitch_r"] * np.cos(th), lay["pitch_r"] * np.sin(th), 0.0))


def _asm_build(params: dict, lay: dict, rotating: bool) -> dict:
    """asm_*.py の build() を集める。回る側 / 回らない側で分ける。

    DRAW_ORDER に載っていない群が返ってきたら警告する。載っていない群は
    描かれないので、部品が絵から消えた原因はたいていこれ。
    """
    known = {name for name, _ in DRAW_ORDER}
    out: dict = {}
    for asm in ASSEMBLIES:
        if bool(asm.rotating) != bool(rotating):
            continue
        for name, mesh in asm.build(params, lay).items():
            if name not in known:
                _warn(f"{asm.name}: 群 {name!r} が DRAW_ORDER に無い。この部品は絵に出ない")
                continue
            if mesh is None or mesh.n_points == 0:
                continue
            if name in out:
                raise RuntimeError(f"メッシュ群 {name!r} を複数のモジュールが返した")
            out[name] = mesh
    return out


def _liquid_base_z(params: dict, lay: dict) -> float:
    """ボトル内の液の底の高さ。"""
    return lay["table_top"] + params["bottle"]["wall_thickness_mm"]


def build_static(params: dict, lay: dict, with_cover=False,
                 with_floor=True) -> dict:
    """テーブルと一緒には回らない部分。

    床・架台・旋回軸受リング・中心柱・固定ノズルとその支持・安全カバーと、
    ROTATING=False の asm_*.py（駆動・受け渡し・搬送・小物）。
    テーブル角にも液量にも依存しないので、コマ送りでは 1 回作れば足りる。
    """
    n = lay["stations"]
    groups = {k: [] for k in MATERIAL}

    # --- 床 ---------------------------------------------------------------
    if with_floor:
        groups["floor"].append(parts.plate(lay["floor_r"] * 2.0,
                                           lay["plate_t"] * 0.5,
                                           base_z=-lay["plate_t"] * 0.5))

    # --- 架台 -------------------------------------------------------------
    # 脚は対角ではなく 22.5 度振った向きに立てる。対角のままだと供給
    # スターホイールの軸とプーリが脚を貫通する（PROPORTION の注記）。
    # 天板は丸いので、回しても天板の見た目は変わらない。
    # 天板には出力軸の下穴を開ける。中実だと軸が板に食い込む。
    fr = parts.frame(span=lay["frame_span"],
                     deck_top_z=lay["deck_top"],
                     pipe=lay["frame_pipe"],
                     deck_diameter=lay["deck_d"],
                     deck_thickness=lay["deck_t"],
                     deck_bore_d=lay.get("deck_bore_d"),
                     matrix=parts.transform_matrix(
                         rot_z_deg=lay.get("frame_leg_phase_deg", 0.0)))
    # 架台は 3 つに分ける。角パイプ（縦の材）・天板（水平の面）・床に着く丸柱。
    # 全部同じ材質にすると、真上から見たとき天板の円と脚の区別が付かない。
    groups["frame"] += [fr["legs"], fr["rails"]]
    groups["deck"].append(fr["deck"])
    groups["foot"].append(fr["feet"])

    # --- 旋回軸受リング（外輪は回らない側） -------------------------------
    groups["steel"].append(
        parts.tube(lay["bearing_ro"], lay["bearing_ri"], lay["bearing_h"],
                   base_z=lay["deck_top"]))

    # --- 中心柱 -----------------------------------------------------------
    # 回転軸の頭を見せるだけの短い出っ張り。上には何も載せない
    # （ノズルは回らないので、ここから吊る必要が無い）。高くすると
    # 斜め上から見たときに奥のボトルを隠す。
    groups["steel"].append(
        parts.cylinder(lay["column_r"],
                       lay["column_top"] - lay["column_base"],
                       base_z=lay["column_base"]))
    groups["steel"].append(
        parts.cone_frustum(lay["column_r"] * PROPORTION["column_cap_r_x_column_r"],
                           lay["column_r"] * PROPORTION["column_cap_top_x_column_r"],
                           lay["plate_t"] * PROPORTION["column_cap_h_x_plate_t"],
                           base_z=lay["column_top"]))

    # --- 固定ノズル 1 本とその支持 ---------------------------------------
    # ノズルは世界座標に固定。テーブル角には一切依存しない。
    # 停止中にこの下へ来た 1 本だけを充填する、という物理側の作りに合わせる。
    nozzle_th = _fill_angle_rad(lay)
    gantry = parts.nozzle_gantry(nozzle_r=lay["pitch_r"],
                                 post_r=lay["post_at_r"],
                                 post_radius=lay["post_r"],
                                 post_base_z=lay["deck_top"],
                                 arm_base_z=lay["arm_base"],
                                 arm_thickness=lay["arm_t"],
                                 arm_width=lay["arm_w"],
                                 angle_rad=nozzle_th)
    groups["steel"] += [gantry["post"], gantry["arm"]]
    groups["steel"].append(
        parts.nozzle(lay["nozzle_bore"], lay["nozzle_len"],
                     base_z=lay["nozzle_tip"], matrix=_nozzle_matrix(lay)))

    # --- 安全カバー -------------------------------------------------------
    # 受け渡し部は開けておく。全周を塞ぐとスターホイールを串刺しにし、
    # コンベアの出入り口も無くなる（開ける角度は cover_arcs()）。
    if with_cover:
        cv = parts.cover(radius=lay["cover_r"], height=lay["cover_h"],
                         base_z=lay["cover_base"], panels=n,
                         post=lay["frame_pipe"] * PROPORTION["cover_post_x_frame_pipe"],
                         wall=lay["cover_wall"],
                         arcs=lay["cover_arcs"])
        groups["cover"].append(cv["panels"])
        # 枠と支柱は塗装。機械の金属と同じにすると、透明な板の中で枠だけが
        # 光って、カバーの外か中かが分からなくなる
        groups["cover_frame"].append(cv["posts"])

    out = {k: parts.merge(v) for k, v in groups.items() if v}
    out.update(_asm_build(params, lay, rotating=False))
    return out


def _present_list(params: dict, lay: dict, present, table_angle_rad) -> list:
    """ボトルの有無の列を整える。空か None なら定常状態から出す。"""
    n = int(lay["stations"])
    if present is None or len(present) == 0:
        return station_present(params, lay, table_angle_rad)
    return [bool(_at(present, i, True)) for i in range(n)]


def build_carousel(params: dict, lay: dict, table_angle_rad=0.0,
                   present=None) -> dict:
    """テーブルと一緒に回る剛体。テーブル板・ポケット板・ハブ・ボトル・暗がり。

    形はテーブル角によらないので、コマ送りでは `table_angle_rad=0` で 1 回作り、
    あとは z 軸まわりに回すだけでよい（`rotation_matrix()`）。

    present はステーションごとのボトルの有無。省くと `station_present()` の
    定常状態（供給から排出までの区間だけ埋まる）になる。ボトルの無いホルダは
    ボトルも稜線も接地の暗がりも作らない。ボトルはホルダに乗って一緒に回るので、
    この並びはテーブル座標系のまま（割出しのあいだ入れ替わらない）。
    """
    b = params["bottle"]
    n = lay["stations"]
    groups = {k: [] for k in MATERIAL}

    # 回る金属は "steel_rot"。回らない金属（"steel"）と名前を分けてあるので、
    # 静止画で merge_groups() に混ぜても、コマ送りでアクタを分けても、
    # 回る側だけ材質を変えられる。
    table = parts.rotary_table(lay["plate_d"], lay["plate_t"],
                               base_z=lay["table_base"],
                               hub_diameter=lay["hub_d"], hub_height=lay["hub_h"])
    groups["steel_rot"] += [table["disc"], table["rim"], table["hub"]]

    # ポケット板。テーブル板の上に直に載せる。溝がボトルの下部を抱えて位置を
    # 決めるので、テーブルと一緒に回る。寸法は既定値を使わず lay から渡す
    # （既定の板半径だと口がボトル外径より狭く、溝にボトルが入らない）。
    groups["steel_pocket"].append(
        parts.pocket_plate(plate_r=lay["pocket_plate_r"],
                           pcd=2.0 * lay["pitch_r"],
                           stations=n,
                           pocket_r=lay["pocket_r"],
                           thickness=lay["pocket_t"],
                           base_z=lay["pocket_base"],
                           matrix=rotation_matrix(table_angle_rad)))

    bottle_r = lay["bottle_r"]
    # 接地部の暗がり。影が使えないので、面を直接暗く汚して代わりにする。
    # 内側を濃く、外へ向かって薄くした 3 枚の円環で、じわりと薄れる感じを出す。
    # 貼る先はポケット板の上面。テーブル面に貼るとポケット板に埋まって、
    # 溝の切り欠きからだけ覗く汚れになってしまう。
    shade_z = lay["pocket_top"] + lay["plate_t"] * 0.02
    # 暗がりはテーブル板の縁で頭を抑える。板の外へ出ると、下に何も無いところに
    # 暗がりだけが浮いて、ボトルが宙に立っているように見える。
    shade_max = lay["plate_r"] - lay["pitch_r"]
    shade_r = [min(bottle_r * v, shade_max)
               for v in PROPORTION["shade_radii_x_bottle_r"]]
    # ボトルの底の面をテーブル面とぴったり同じ高さに置くと、どちらも中心から
    # 切った三角形の集まりなので、放射状の縞になって奥行きが取り合いになる。
    # わずかに沈めて、底の面をテーブルの陰に隠す。
    bottle_base_z = lay["table_top"] - lay["plate_t"] * PROPORTION["bottle_sink_x_plate_t"]

    on_table = _present_list(params, lay, present, table_angle_rad)

    for i, th in enumerate(station_angles(n, table_angle_rad)):
        if not on_table[i]:
            continue                    # 空ホルダ。ボトルも暗がりも作らない
        mat = parts.transform_matrix(
            translate=(lay["pitch_r"] * np.cos(th), lay["pitch_r"] * np.sin(th), 0.0))

        for k in range(len(shade_r) - 1):
            groups[f"shade_{k}"].append(
                parts.flat_ring(shade_r[k], shade_r[k + 1], z=shade_z, matrix=mat))

        groups["bottle"].append(
            parts.bottle(inner_diameter=b["inner_diameter_mm"],
                         body_height=b["body_height_mm"],
                         shoulder_height=b["shoulder_height_mm"],
                         neck_diameter=b["neck_diameter_mm"],
                         neck_height=b["neck_height_mm"],
                         wall_thickness=b["wall_thickness_mm"],
                         base_z=bottle_base_z, matrix=mat))
        groups["glass"].append(
            parts.bottle_edges(inner_diameter=b["inner_diameter_mm"],
                               body_height=b["body_height_mm"],
                               shoulder_height=b["shoulder_height_mm"],
                               neck_diameter=b["neck_diameter_mm"],
                               neck_height=b["neck_height_mm"],
                               wall_thickness=b["wall_thickness_mm"],
                               base_z=bottle_base_z,
                               thickness=b["wall_thickness_mm"]
                               * PROPORTION["edge_thickness_x_wall"],
                               matrix=mat))

    out = {k: parts.merge(v) for k, v in groups.items() if v}
    out.update(_asm_build(params, lay, rotating=True))
    return out


def build_carousel_meshes(params: dict, lay: dict, table_angle_rad=0.0,
                          present=None) -> dict:
    """コマ送り用。build_carousel() の群を CAROUSEL_ALIAS で読み替えて返す。

    いまは読み替える先が無い（build_carousel() が最初から回る側だけの群名を
    返す）ので素通しになる。回らない側と名前がぶつかる群ができたときに、
    ここで別の名前へ移してアクタを分けられるようにしてある。
    **呼ぶ側で辞書から抜き取らないこと。** 以前は `carousel.pop("steel")` と
    書いてあって、テーブル板の群名が変わった瞬間に KeyError で落ちた。
    """
    out: dict = {}
    for name, mesh in build_carousel(params, lay, table_angle_rad, present).items():
        out[CAROUSEL_ALIAS.get(name, name)] = mesh
    return out


def build_liquid(params: dict, lay: dict, state: MachineState,
                 table_angle_rad=None) -> dict:
    """毎コマ形が変わる部分。ボトル内の液と液面の縁。

    `table_angle_rad` を省くと `state.table_angle_rad` を使う。0 を渡すと
    テーブル座標系（回す前）の形になる。傾きは接線・半径の局所量で受け取る
    ので、0 で作って z 軸まわりに回した結果は、最初から回して作ったものと
    一致する。コマ送りではこれを使って回転ぶんをアクタ側の変換に任せる。
    """
    if table_angle_rad is None:
        table_angle_rad = state.table_angle_rad
    liquid_r = lay["liquid_r"]
    liquid_base = _liquid_base_z(params, lay)
    rim_thick = liquid_r * PROPORTION["rim_thickness_x_liquid_r"]
    groups = {"liquid": [], "rim": []}
    # ボトルの無いホルダには液も作らない。作ると液だけが宙に浮く。
    # 有無を引くテーブル角は、**渡された table_angle_rad のほう**を使う。
    # ボトル本体を組む build_carousel() も同じ角で引くので、コマ送りのように
    # 0（回す前の形）で作るときも、両者が同じホルダを選ぶ。
    on_table = _present_list(params, lay, state.bottle_present, table_angle_rad)

    for i, th in enumerate(station_angles(lay["stations"], table_angle_rad)):
        if not on_table[i]:
            continue
        level = volume_to_level(params, _at(state.volumes_mL, i, 0.0))
        if level <= 0.0:
            continue
        mat = parts.transform_matrix(
            translate=(lay["pitch_r"] * np.cos(th), lay["pitch_r"] * np.sin(th), 0.0))
        tilt, tilt_dir = tilt_to_world(_at(state.tilt_t, i, 0.0),
                                       _at(state.tilt_r, i, 0.0), th)
        groups["liquid"].append(
            parts.liquid(liquid_r, level, base_z=liquid_base,
                         tilt_rad=tilt, tilt_dir_rad=tilt_dir, matrix=mat))
        groups["rim"].append(
            parts.liquid_rim(liquid_r, level, base_z=liquid_base,
                             tilt_rad=tilt, tilt_dir_rad=tilt_dir,
                             thickness=rim_thick, matrix=mat))

    return {k: parts.merge(v) for k, v in groups.items() if v}


def build_stream(params: dict, lay: dict, state: MachineState):
    """充填中の液柱。固定ノズルの先端から、その下のボトルの液面まで。

    ノズルは回らないので、これはテーブルと一緒には回らない。ステーションが
    ノズルから `STREAM_ALIGN_TOL_DEG` より離れていたら（割出しの途中なら）
    None を返す。液柱が宙に浮くのを防ぐ。
    """
    i = state.filling_index
    if i is None:
        return None
    if not _present_list(params, lay, state.bottle_present,
                         state.table_angle_rad)[i]:
        return None                     # 空ホルダ。注ぐ相手がいない
    th = station_angles(lay["stations"], state.table_angle_rad)[i]
    if abs(np.degrees(_angle_from_nozzle(th, lay.get("fill_deg")))) \
            > STREAM_ALIGN_TOL_DEG:
        return None
    level = volume_to_level(params, _at(state.volumes_mL, i, 0.0))
    surface_z = _liquid_base_z(params, lay) + level
    if lay["nozzle_tip"] <= surface_z:
        return None
    return parts.cylinder(
        params["bottle"]["neck_diameter_mm"] * PROPORTION["stream_r_x_neck_d"],
        lay["nozzle_tip"] - surface_z,
        base_z=surface_z, resolution=parts.RES_COARSE,
        matrix=_nozzle_matrix(lay))


def rotation_matrix(table_angle_rad: float) -> np.ndarray:
    """テーブル角ぶんの回転（z 軸まわり）。アクタの変換に渡す用。"""
    return parts.transform_matrix(rot_z_deg=np.degrees(float(table_angle_rad)))


def spin_matrix(center, angle_rad: float) -> np.ndarray:
    """自転ぶんの変換。中心へ寄せて z 軸まわりに回して戻す。

    スターホイールはテーブル軸ではなく自分の軸で回るので、テーブルと同じ
    `rotation_matrix()` では動かせない。
    """
    cx, cy = float(center[0]), float(center[1])
    to_origin = parts.transform_matrix(translate=(-cx, -cy, 0.0))
    turn = parts.transform_matrix(rot_z_deg=np.degrees(float(angle_rad)))
    back = parts.transform_matrix(translate=(cx, cy, 0.0))
    return back @ turn @ to_origin


def spin_angles(params: dict, lay: dict, state: MachineState) -> dict:
    """群名 -> 自転角 [rad]。モジュールの spin_angles() を集める。

    SPIN_CENTERS に載っていない群を返したモジュールがあれば、その群は
    自転の中心が分からないので飛ばす（絵には出るが回らない）。
    """
    out: dict = {}
    for asm in ASSEMBLIES:
        if not asm.spin_centers:
            continue
        for group, angle in asm.spin_angles(params, lay, state).items():
            if group not in asm.spin_centers:
                _warn(f"{asm.name}: 群 {group!r} の自転角を返したが "
                      "SPIN_CENTERS に中心が無い。回さない")
                continue
            out[group] = float(angle)
    return out


def spin_matrices(params: dict, lay: dict, state: MachineState) -> dict:
    """群名 -> 当てる 4x4。自転する群が無ければ空。

    静止画（build）もコマ送り（animate.FrameRenderer）もここを読む。
    **中心は毎回モジュールから引き直す。** `layout()` が入れ替えた自転中心が
    ここに届かないと、星車が別の点まわりに回った絵がそのまま出る。
    """
    angles = spin_angles(params, lay, state)
    return {group: spin_matrix(center, angles.get(group, 0.0))
            for group, (_owner, center) in dict(SPIN_CENTERS).items()}


def merge_groups(*group_dicts) -> dict:
    """{群名: メッシュ} をいくつも受け取って 1 つにまとめる。

    **同じ群名は捨てずに結合する。** build_static() と build_carousel() は
    どちらも "steel" を返すので、`dict.update()` でまとめると静止側の
    金属（旋回軸受リング・中心柱・固定ノズル・その支柱とアーム）が丸ごと
    消える。確認用に機械を組むときは必ずこれを通すこと。
    """
    groups: dict = {}
    for piece in group_dicts:
        if not piece:
            continue
        for name, mesh in piece.items():
            if mesh is None or getattr(mesh, "n_points", 0) == 0:
                continue
            groups.setdefault(name, []).append(mesh)
    return {k: parts.merge(v) for k, v in groups.items() if v}


def build(params: dict, lay: dict, state: MachineState,
          with_cover=False, with_floor=True) -> dict:
    """機械 1 台を組み立てて、材質ごとにまとめたメッシュの辞書を返す。

    静止画用。回る部分も回らない部分も一度に組む。コマ送りでは
    `build_static` / `build_carousel` / `build_liquid` / `build_stream` を
    別々に呼んで、回らない部分を作り直さないようにする。
    自転する群（スターホイール）には、ここで `state.cam_angle_rad` から
    出した自転ぶんの変換を焼き込む。
    """
    stream = build_stream(params, lay, state)
    groups = merge_groups(
        build_static(params, lay, with_cover=with_cover, with_floor=with_floor),
        build_carousel(params, lay, state.table_angle_rad, state.bottle_present),
        build_liquid(params, lay, state),
        {"stream": stream} if stream is not None else None)

    # 自転は群ごとに丸ごと当たるので、合流させてから焼き込んでよい
    for group, mat in spin_matrices(params, lay, state).items():
        mesh = groups.get(group)
        if mesh is not None:
            groups[group] = parts.place(mesh, mat)
    return groups


# --------------------------------------------------------------------------
# 描画
# --------------------------------------------------------------------------
def studio_cubemap(size=64) -> pv.Texture:
    """周囲の映り込み用の環境マップを手で作る。

    金属を PBR で描くと、映り込む先が無いと真っ黒になる。外部の画像は
    使わずに、上が明るく下が暗い無地の箱を 6 面ぶん組んで代わりにする。
    """
    top = np.array([236.0, 239.0, 242.0])
    bottom = np.array([104.0, 109.0, 116.0])
    faces = []
    for i in range(6):
        if i == 2:      # 天井
            rgb = np.ones((size, size, 3)) * np.array([248.0, 249.0, 250.0])
        elif i == 3:    # 床
            rgb = np.ones((size, size, 3)) * np.array([92.0, 96.0, 102.0])
        else:           # 側面は上下のグラデーション
            v = np.linspace(0.0, 1.0, size)[:, None, None]
            rgb = (top * (1.0 - v) + bottom * v) * np.ones((1, size, 1))
        img = pv.ImageData(dimensions=(size, size, 1))
        img["data"] = rgb.astype(np.uint8).reshape(-1, 3)
        faces.append(img)
    return pv.Texture(faces)


def new_plotter(size=(1600, 1200)) -> pv.Plotter:
    """描画器を 1 つ作って、背景と環境マップまで済ませて返す。

    **描画器の下ごしらえはここが唯一。** 以前は同じ 5 行が 7 箇所に写して
    あって、1 箇所だけ映り込みの前計算の精度が違っていた（同じ機械を撮った
    はずの絵で金属の明るさが変わる）。
    """
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    plotter.set_background(BACKGROUND)
    plotter.set_environment_texture(studio_cubemap(), is_srgb=True)
    # 環境マップは無地なので、映り込みの前計算は既定より粗くて足りる。
    # 既定のままにすると 1 枚描くのに分単位かかる。
    try:
        plotter.renderer.GetEnvMapPrefiltered().SetPrefilterMaxSamples(64)
        plotter.renderer.GetEnvMapIrradiance().SetIrradianceSize(32)
    except AttributeError:
        pass
    return plotter


# --------------------------------------------------------------------------
# 機械の外形（カメラの画角の基準）
#
# 画角をテーブル径の倍率で決めていたので、外へ部品を足すたびに枠から外れた。
# 組み上がったメッシュから実測して lay に入れ、cameras.py はそれを読む。
# 測るのはプロセスに 1 回。組み直さないよう、測った値は覚えておく。
# --------------------------------------------------------------------------
EXTENT_KEYS = ("extent_r", "extent_top", "extent_bottom", "extent_x", "extent_y")
# 外形を測るときに外す群。床は機械ではないし、半径 11200 で桁が違う。
EXTENT_SKIP = ("floor",)


def extent_from_meshes(meshes: dict) -> dict:
    """組み上がったメッシュから機械の外形を測る。

    extent_r は水平の最大半径（テーブル軸から測る）。カメラは注視点を軸上に
    置くので、枠に収めるのに要るのは幅ではなくこの半径。
    """
    r2 = 0.0
    z0, z1 = np.inf, -np.inf
    x0, x1, y0, y1 = np.inf, -np.inf, np.inf, -np.inf
    for name, mesh in meshes.items():
        if name in EXTENT_SKIP or mesh is None or mesh.n_points == 0:
            continue
        pts = np.asarray(mesh.points, dtype=float)
        r2 = max(r2, float(np.max(pts[:, 0] ** 2 + pts[:, 1] ** 2)))
        x0, x1 = min(x0, float(pts[:, 0].min())), max(x1, float(pts[:, 0].max()))
        y0, y1 = min(y0, float(pts[:, 1].min())), max(y1, float(pts[:, 1].max()))
        z0, z1 = min(z0, float(pts[:, 2].min())), max(z1, float(pts[:, 2].max()))
    if not np.isfinite(z0):
        return {}
    return {"extent_r": float(np.sqrt(r2)), "extent_top": z1, "extent_bottom": z0,
            "extent_x": (x0, x1), "extent_y": (y0, y1)}


_EXTENT_CACHE: dict = {}


def ensure_extent(lay: dict, meshes=None, params=None) -> dict:
    """lay に機械の外形の実測値を入れる。既に入っていれば何もしない。

    meshes を渡せばそれを測る（コマ送りは組んだものをそのまま渡せばよい）。
    渡さないときは、覚えてある値を使う。どちらも無ければ組んで測る
    （params が無ければ諦めて何も入れない。cameras 側が控えに落ちる）。
    """
    if all(k in lay for k in EXTENT_KEYS):
        return lay
    got = extent_from_meshes(meshes) if meshes else dict(_EXTENT_CACHE)
    if not got and params is not None:
        got = extent_from_meshes(merge_groups(
            build_static(params, lay, with_floor=False),
            build_carousel(params, lay, 0.0)))
    if got:
        _EXTENT_CACHE.update(got)
        lay.update(got)
    return lay


def set_camera(plotter: pv.Plotter, lay: dict, name="iso", params=None,
               meshes=None, **overrides) -> dict:
    """視点を当てる。構図の表は cameras.py が持っている。

    名前を省くと既定の "iso"（以前ここにあった CAMERA と同じ構図）。
    params を渡すと工程配置（stations.fill_deg）に追従する視点が正しく向く。
    画角は機械の実測外形に合わせるので、meshes を渡せるときは渡す。
    """
    ensure_extent(lay, meshes, params)
    return cameras.apply(plotter, lay, name, params, **overrides)


def render(meshes: dict, lay: dict, out_path: Path, size=(1600, 1200),
           shadows=False, camera="iso", params=None) -> Path:
    """組み上げたメッシュを 1 枚に焼く。"""
    plotter = new_plotter(size)

    # 描く順は DRAW_ORDER 唯一つ。理由はそちらのコメントにある。
    for name, material in DRAW_ORDER:
        mesh = meshes.get(name)
        if mesh is None or mesh.n_points == 0:
            continue
        plotter.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True,
                         feature_angle=35.0, **MATERIAL[material])

    # 影を落とすとこの環境では半透明の面が一部消える（手前のボトルが抜ける）。
    # 既定では切ってあり、不透明だけの絵を作るときだけ立てる。
    if shadows:
        try:
            plotter.enable_shadows()
        except Exception:                                 # noqa: BLE001
            pass

    set_camera(plotter, lay, camera, params, meshes=meshes)
    plotter.enable_anti_aliasing("ssaa")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(out_path))
    plotter.close()
    return out_path


def render_state(params: dict, lay: dict, state: MachineState, out_path,
                 size=(1600, 1200), with_cover=False, with_floor=True,
                 shadows=False, camera="iso") -> Path:
    """状態を 1 つ受け取って 1 枚描く。コマ送りはこれを繰り返し呼ぶ。"""
    meshes = build(params, lay, state, with_cover=with_cover, with_floor=with_floor)
    return render(meshes, lay, Path(out_path), size=size, shadows=shadows,
                  camera=camera, params=params)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ロータリー充填機を 1 枚描く")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="出力 PNG")
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 1200),
                    metavar=("W", "H"), help="画素数")
    ap.add_argument("--filling", type=int, default=None,
                    help="充填中に見せるステーション番号。"
                         "既定はノズルの真下のステーション。-1 で無し")
    ap.add_argument("--table-angle", type=float, default=None,
                    help="テーブルの回転角 [deg]。既定 0")
    ap.add_argument("--cam-angle", type=float, default=None,
                    help="カム入力軸の角 psi [deg]。1 タクトで 0〜360。"
                         "0〜180 が割出し（テーブルが 45 度進む）、"
                         "180〜360 が停留（スターホイールが 60 度送る）。"
                         "渡すとテーブル角もここから出る（--table-angle は使わない）")
    ap.add_argument("--tilt-deg", type=float, default=None,
                    help="全ステーションの液面の傾きを揃える [deg]")
    ap.add_argument("--cover", action="store_true",
                    help="安全カバーを描く（既定は描かない）")
    ap.add_argument("--no-floor", action="store_true", help="床面を描かない")
    ap.add_argument("--shadows", action="store_true",
                    help="影を落とす（半透明が欠けるので既定は切ってある）")
    ap.add_argument("--camera", default="iso", choices=cameras.names("3d"),
                    help="視点。表は cameras.py")
    args = ap.parse_args(argv)

    params = load_params()
    lay = derive_layout(params)
    if args.filling is None:
        filling = "auto"
    elif args.filling < 0:
        filling = None
    else:
        filling = args.filling % lay["stations"]

    # カム入力軸角を渡されたら、テーブル角も自転角もそこから出す。
    # 渡されなければテーブル角だけを直に振る（自転する部品は割出しの
    # 始まりの姿勢で止まる）。
    if args.cam_angle is not None:
        if args.table_angle is not None:
            _warn("--cam-angle があるので --table-angle は使わない")
        cam_rad = np.radians(args.cam_angle)
        table_rad = table_angle_from_cam(params, cam_rad)
    else:
        cam_rad = 0.0
        table_rad = np.radians(args.table_angle or 0.0)
    state = demo_state(params, lay, filling_index=filling,
                       table_angle_rad=table_rad,
                       tilt_deg=args.tilt_deg, cam_angle_rad=cam_rad)

    level = volume_to_level(params, params["fill"]["target_volume_mL"])
    print(f"満量の液深: {level:.1f} mm / 胴高さ: {lay['body_h']:.0f} mm")
    print(f"機械全高: {lay['total_height']:.0f} mm / "
          f"テーブル上面: {lay['table_top']:.0f} mm")
    print(f"ノズル: 1 本 / 世界角 {lay['fill_deg']:.0f} deg / "
          f"充填中のステーション: {state.filling_index}")
    print(f"工程配置: 供給 {lay['infeed_deg']:.0f} / 充填 {lay['fill_deg']:.0f} / "
          f"排出 {lay['discharge_deg']:.0f} deg")
    print(f"ポケット板: R{lay['pocket_plate_r']:.0f} / 溝 R{lay['pocket_r']:.0f} / "
          f"口の幅 {lay['pocket_mouth_w']:.2f}（ボトル外径 {2 * lay['bottle_r']:.0f}）/ "
          f"z {lay['pocket_base']:.0f}〜{lay['pocket_top']:.0f}")
    print(f"カム入力軸: {np.degrees(cam_rad):.1f} deg / "
          f"テーブル角: {np.degrees(table_rad):.2f} deg")
    spins = spin_angles(params, lay, state)
    if spins:
        # 同じ中心・同じ角のものはまとめて 1 行にする
        centers = dict(SPIN_CENTERS)
        by_axis: dict = {}
        for g, a in sorted(spins.items()):
            by_axis.setdefault((round(np.degrees(a), 2),
                                centers[g][1]), []).append(g)
        for (ang, center), groups in by_axis.items():
            print(f"自転: 中心 ({center[0]:7.2f}, {center[1]:7.2f}) "
                  f"{ang:6.1f} deg / 群 {', '.join(groups)}")
    else:
        print("自転する群: 無し（モジュールが SPIN_CENTERS を出していない）")
    loaded = ", ".join(a.name for a in ASSEMBLIES) or "無し"
    print(f"組み立てモジュール: {loaded} / 視点: {args.camera}")

    out = render_state(params, lay, state, args.out, size=tuple(args.size),
                       with_cover=args.cover, with_floor=not args.no_floor,
                       shadows=args.shadows, camera=args.camera)
    print(f"外形の実測: 最大半径 {lay.get('extent_r', float('nan')):.0f} / "
          f"高さ {lay.get('extent_bottom', float('nan')):.0f}〜"
          f"{lay.get('extent_top', float('nan')):.0f} mm")
    print(f"書き出し: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
