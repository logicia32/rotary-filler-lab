"""駆動系。カム式インデックスユニットと、それを回す誘導ギヤモータ。

テーブルを 45deg ずつ間欠で送っているのはこの一式で、機械の下半分の主役になる。
組み方は実機の据え付けをそのままなぞる。

    架台天板の下に、テーブルと同軸でインデックスユニットを吊る
      -> 上面の段付き出力フランジと出力軸ボスが天板の下面まで立ち上がる
      -> そこから出力軸が天板を貫いて上へ抜け、旋回軸受の内側を通って
         テーブル板の下面のハブへ締結される（ここで動力がテーブルへ渡る）
      -> 入力軸だけ横へ水平に逃がす（世界角は `drv_angle_deg`。架台の 1 面の
         芯を向く向き DRIVE_FACE_DEG に、架台の振りを足したもの）
      -> ボスの端面から継手までを露出した軸にして、その途中から直角に
         原点検出の取り出しを立てる（かさ歯車 1:1 -> 縦軸 -> ドグ板 ->
         近接センサ。入力軸 1 回転で 1 パルス）
      -> その延長にカップリングを挟んでギヤモータを直結する
      -> 回る継手は板金の角箱で包む。**両側面に点検窓を開けて、継手と
         かさ歯車が外から見えるようにする**
      -> 外へ張り出すモータは板金カバーで囲う（端面に点検窓）
      -> モータの取付ベースは架台の脚に渡した梁と棚板が受ける
      -> 天板の上、テーブル板の下面に大歯車（環）を締める。環の胴は
         テーブル板の下面まで立ち上げて突き当て、12 本のボルトで締める。
         星車 2 台の小歯車がこれに噛み合い、外歯どうしなので向きが
         反転して 2:1 で回る

動力の流れは 1 本につながっている。**絵でも 1 本につながっていること。**

    ギヤモータ -> 継手 -> 入力軸（20 rpm 連続）-> カム -> 出力軸
              -> テーブル板の下面のハブ -> テーブル板 -> 大歯車の環
              -> 星車の小歯車

入力軸は**停留中も止まらない**。カム式インデックスの見どころはそこなので、
入力軸から取り出したドグ板を絵の中で回す（`SPIN_CENTERS` / `spin_angles`）。

大歯車を天板の上へ置く理由は 1 つだけ。天板の下は架台の脚が 4 本立っていて、
星車の軸まわりに空くのは 98.26 mm しかない。ピッチ円半径 112.5 の歯車は
入らない。脚の上端は天板の下面なので、天板の上には脚が無い。

高さの決め方はひとつだけ覚えておけばよい。天板の下面から下へ、出力フランジ
2 段と出力軸ボスの合計ぶんだけ下げた所が本体箱の上面。そこから箱の高さぶん
下がった所が箱の底で、入力軸の高さは箱の底から箱の高さの
`parts.INDEX_UNIT_INPUT_AXIS_X_BOX_H` 倍（`parts.index_unit` が入力ボスを
置く高さそのもの）。ギヤモータの軸高さはこの値をそのまま使うので、両者は
必ず一直線に並ぶ。

部品は入力軸を -x へ出した局所座標で組み、最後にまとめて回して世界へ置く。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parts  # noqa: E402


# --------------------------------------------------------------------------
# 絵にするためだけの比率。すべて params.json の値の倍率で、絶対寸法は書かない。
# 元にした目安は params.json の諸元と MODEL.md。
# --------------------------------------------------------------------------
PROPORTION = {
    # --- カム式インデックスユニット本体（すべてテーブル外径 560 の倍率） ---
    "unit_box_w_x_plate_d": 0.500,        # 本体箱 幅（テーブル径の 0.45〜0.55）
    "unit_box_d_x_plate_d": 0.440,        # 本体箱 奥行
    "unit_box_h_x_plate_d": 0.375,        # 本体箱 高さ
    "unit_flange_outer_x_plate_d": 0.375, # 出力フランジ 下段の外径
    "unit_flange_inner_x_plate_d": 0.268, # 出力フランジ 上段の外径
    "unit_flange_t_outer_x_plate_t": 2.25,  # 下段の厚み / テーブル板厚
    "unit_flange_t_inner_x_plate_t": 1.50,  # 上段の厚み / テーブル板厚
    "unit_boss_d_x_plate_d": 0.1875,      # 出力軸ボス 径
    "unit_boss_h_x_plate_d": 0.0893,      # 出力軸ボス 高さ
    "unit_input_boss_d_x_plate_d": 0.125, # 入力軸ボス 径
    "unit_input_boss_len_x_plate_d": 0.0536,  # 入力軸ボス 突き出し長さ
    "unit_bolt_d_x_plate_t": 2.50,        # 取付ボルト頭 径 / テーブル板厚
    "unit_bolt_h_x_plate_t": 0.75,        # 同 高さ
    "unit_sight_d_x_plate_d": 0.0536,     # 油面窓 径
    "unit_oil_port_d_x_plate_d": 0.0464,  # 給油口 径
    # 入力ボスの高さと据付段の大きさは `parts.index_unit` が内部で使う値と
    # 同じでなければならない。写すと片方だけ動いてずれるので、
    # parts.INDEX_UNIT_INPUT_AXIS_X_BOX_H / INDEX_UNIT_SKIRT_SCALE を直接読む

    # --- 出力軸（天板を貫いてテーブルへ動力を渡す） ---
    # ボスの頭で止めると天板の下で動力が切れる。軸は必ず天板を貫いて、
    # 旋回軸受の内側を通り、テーブル板の下面のハブへ締結する
    "out_shaft_d_x_boss_d": 0.55,         # 出力軸 径 / 出力軸ボス径
    "out_shaft_into_boss": 0.60,          # 軸の下端をボスのどこまで潜らせるか
    "out_seat_ro_x_shaft_d": 0.72,        # 天板上の軸受座 外半径 / 出力軸径
    "out_seat_h_x_plate_t": 0.75,         # 同 高さ
    "out_hub_r_x_bearing_ri": 0.92,       # テーブル側ハブ 半径 / 旋回軸受の内半径
    "out_hub_gap_x_plate_t": 0.25,        # 軸受座の頭とハブ下面の逃げ（回る側と
                                          # 止まっている側の隙間）

    # --- スターホイールへの同期取り出し（テーブル出力から 2:1 の平歯車）---
    # 星車のポケットピッチはテーブルのステーション間隔に揃っているので、
    # テーブルが 45deg 送るあいだに星車は 90deg 送る。速比はちょうど 2.0。
    #
    # 伝えるのはベルトではなく歯車にする。テーブルは CCW、星車は CW なので
    # 向きが反転しなければならず、外歯どうしの噛み合いはそれを自然に満たす。
    # しかも中心距離と速比からピッチ円半径が一意に決まる。
    #
    #     r1 + r2 = 中心距離 337.5   かつ   r1 / r2 = 2
    #     -> r1 = 225.0  r2 = 112.5
    #
    # これはテーブルのボトルピッチ円と星車のボトルピッチ円そのもの。偶然では
    # なく、「ポケットのピッチを合わせて転がす」条件が歯車の噛み合い条件
    # （2 つのピッチ円が接して転がる）と同じものだから。ボトルの受け渡し点と
    # 歯車の噛み合い点は平面図で同じ位置に来る。
    #
    # したがってピッチ円半径は比率で持たない。lay のピッチ円をそのまま使う。
    # ここで持つのは歯の大きさと据わりだけ。
    #
    # 置く高さは**架台天板の上面とテーブル板の下面のあいだ**。天板の下は
    # 架台の脚が 4 本立っていて、星車の軸まわりに空くのは芯々 132.31 −
    # 脚の半対角 34.05 = 98.26 しかない。ピッチ円半径 112.5 の歯車は入らない。
    # 天板の上には脚が無いので（脚の上端は天板の下面）、そこへ上げる
    "gear_ratio": 2.0,                    # 星車の回転 / テーブルの回転
    "gear_teeth_star": 45,                # 星車側の歯数（テーブル側はこの 2 倍）
    "gear_addendum_x_module": 0.80,       # 歯先の出 / モジュール
    "gear_dedendum_x_module": 1.10,       # 歯底の落ち / モジュール
    "gear_pressure_deg": 20.0,            # 歯すじの傾き（歯を台形で描く用）
    "gear_backlash": 0.12,                # 歯厚を細める割合（噛み合いの逃げ）
    "gear_deck_gap_x_plate_t": 0.25,      # 歯車の下面と架台天板の上面の隙間
    # 大歯車の内穴。旋回軸受リングの外側へ逃がす。リングは天板の上面から
    # テーブル板の下面まで丸ごと詰まっているので、内側を通る道は無い。
    # 歯車はテーブル板の下面に締める環になる
    "gear_bore_gap_x_plate_t": 0.50,      # 内穴と旋回軸受リング外径の隙間
    "gear_tooth_bite_x_plate_t": 0.05,    # 歯の付け根を環へ食い込ませる量
    "gear_bolts": 12,                     # テーブル板へ締めるボルト 本数
    "gear_bolt_d_x_plate_t": 1.25,        # 同 頭 径
    "gear_bolt_h_x_gap": 0.60,            # 同 頭 高さ / 天板との隙間
    "gear_bolt_at": 0.18,                 # 穴と歯底のあいだのどこに並べるか
    # 歯の幅は小歯車と揃えるが、**環の胴はテーブル板の下面まで立ち上げる。**
    # 歯だけの厚みで止めると環がどこにも触れず、宙に浮いた輪になる（絵では
    # 回っていても、機構としては入力が無い）。胴の頭を板の下面に突き当てて
    # ボルトで締めれば、動力は 出力軸 -> ハブ -> テーブル板 -> 環 と流れる

    # --- 入力軸の露出部と原点検出の取り出し -------------------------------
    # カム式インデックスの見どころは「入力は回り続けるのに出力は止まる」。
    # 入力軸は板金の中で、しかも水平軸なので、そのままでは絵で回せない
    # （自転の仕掛けは z 軸まわりしか扱えない）。**直角に 1 回取り出して
    # 縦軸に載せ替える。** 載せ替えた先のドグ板は z 軸まわりに回るので、
    # 停留中も回り続けるものが画面に残る。1:1 なので入力軸 1 回転で 1 回転、
    # ドグは近接センサの前を 1 回通る（params.json の control.cam_home_sensor）
    "input_shaft_d_x_plate_d": 0.0393,    # 入力軸 径
    "input_shaft_free_x_plate_d": 0.088,  # ボス端面から継手までの露出長さ
    # 縦軸を露出軸のどこに立てるか（ボス端面からの割合）。**架台の上桁が
    # 効く。** 上桁は脚の芯々の線に乗っていて、局所 x で
    # frame_span/2 ± frame_pipe/2 の帯を、ドグ板と同じ高さで横切る。
    # ドグ板の外縁がその帯へ入ると桁を突き抜けるので、縦軸はボス寄りに置く
    "takeoff_at_free": 0.22,              # 縦軸の位置 / 露出長さ
    "takeoff_gear_r_x_plate_d": 0.0357,   # かさ歯車 ピッチ半径（1:1 の対）
    "takeoff_gear_root_frac": 0.55,       # 歯の小端 / ピッチ半径
    "takeoff_shaft_d_x_gear_r": 0.60,     # 縦軸 径 / かさ歯車ピッチ半径
    "takeoff_housing_ro_x_shaft_d": 1.15, # 軸受箱 外半径 / 縦軸 径
    "takeoff_housing_h_x_plate_d": 0.048, # 軸受箱 丈
    # ドグ板は**半月**にする。丸い板に小さな突起を付ける形だと、天板の下を
    # 横から覗く画角（この機械で駆動系が見える唯一の向き）では突起が板の陰に
    # 入り、回っているのかどうか絵から読めない。半月なら輪郭そのものが向きを
    # 持つので、どの向きから見ても回転が分かる。近接センサの前を金属が通るのは
    # 1 回転につき 1 回なので、1 回転 1 パルスは変わらない
    "dog_disc_r_x_plate_d": 0.0536,       # ドグ板 半径
    "dog_disc_t_x_plate_t": 1.25,         # ドグ板 厚
    "dog_disc_gap_x_plate_t": 1.50,       # 軸受箱の頭とドグ板の下面の逃げ
    "dog_chord_x_disc_r": 0.30,           # 弦の位置 / 半径（軸の後ろに残す肉）
    # 近接センサは上から下向きに構える。横向きにすると、検出面まで届かせる
    # ぶんステーが外へ伸びて架台の上桁に刺さる。上から構えれば、ステーも
    # センサもドグ板の半径の内側で収まる
    "sensor_d_x_plate_t": 2.25,           # 近接センサ 径
    "sensor_len_x_plate_t": 3.00,         # 同 長さ
    "sensor_gap_x_plate_t": 0.50,         # 検出面とドグ板の上面のすき間
    "sensor_at_disc_r": 0.70,             # センサを板のどの半径に構えるか
    "sensor_stay_x_plate_t": 1.50,        # 取付ステー 一辺
    "sensor_stay_at_disc_r": 1.35,        # ステーを立てる位置 / ドグ板半径

    # --- 誘導ギヤモータ ---
    "motor_d_x_plate_d": 0.157,           # 胴 径
    "motor_len_x_plate_d": 0.259,         # 胴 長さ
    "gearhead_size_x_plate_d": 0.179,     # ギヤヘッド 一辺
    "gearhead_len_x_plate_d": 0.134,      # ギヤヘッド 長さ
    "motor_shaft_d_x_plate_d": 0.0357,    # 出力軸 径
    "motor_shaft_len_x_plate_d": 0.0804,  # 出力軸 突き出し長さ
    "motor_fins": 7,                      # 冷却フィンの枚数（寸法ではなく本数）
    "terminal_x_plate_d": (0.0893, 0.0625, 0.0625),   # 端子箱 W/D/H
    "motor_base_x_plate_d": (0.393, 0.214, 0.0179),   # 取付ベース W/D/t
    # ベース上面と胴の下端の逃げは `parts.gearmotor` が内部で足している値
    # そのもの。棚板の高さをここから逆算するので、parts.GEARMOTOR_BASE_GAP_MM
    # を直接読む（写すと片方だけ動いてモータが宙に浮く）

    # --- カップリングと安全カバー ---
    "coupling_d_x_plate_d": 0.075,        # 継手 径
    "coupling_len_x_plate_d": 0.104,      # 継手 長さ
    "coupling_engage": 0.50,              # モータ軸端を継手のどこまで差し込むか
    "guard_side_x_coupling_d": 2.15,      # カバー 横幅 / 継手径
    "guard_head_x_coupling_d": 0.45,      # 継手上端からカバー天井までの余裕 / 継手径
    "guard_panel_t_x_plate_t": 0.25,      # カバー 板厚
    # 継手のカバーは中実の箱にしない。中実にすると継手もかさ歯車も丸ごと
    # 飲まれて、駆動列が絵から消える。両側面に点検窓を開けて桟を渡す
    "guard_window_frac": (0.56, 0.46),    # 点検窓 幅・高さ / 側面の寸法
    "guard_bars": 2,                      # 同 桟の本数
    "guard_bar_w_x_plate_t": 0.62,        # 桟 幅
    # 上蓋が本体より一回り大きくなる比と板厚は、部品側の継手カバーが内部で
    # 使う値そのもの。カバーはギヤヘッド前面とユニット据付段の間に丁度収める
    # ので、ここがずれると上蓋の角がギヤヘッドに食い込む。
    # parts.COUPLING_COVER_LID_SCALE / COUPLING_COVER_LID_T_X_H を直接読む

    # --- モータ取付台（架台の脚に渡す梁＋棚板＋吊り柱） ---
    "mount_shelf_t_x_plate_t": 1.25,      # 棚板 厚み / テーブル板厚
    "mount_shelf_half_w_x_plate_d": 0.179,  # 棚板 半幅
    "mount_shelf_out_x_plate_d": 0.036,   # 棚板がモータ後端より外へ出る量
    "mount_shelf_gap_x_plate_t": 0.25,    # 棚板の内端とユニット据付段の隙間
    "mount_post_x_plate_d": 0.050,        # 吊り柱 一辺
    "mount_post_at_x_deck_r": 0.87,       # 吊り柱を立てる半径 / 天板半径
    "mount_post_gap_x_plate_t": 0.375,    # 吊り柱とモータベースの隙間

    # --- モータの板金カバー ---
    # 天板の外径からモータが張り出す。腿の高さで回る物がむき出しになるので、
    # 天板の下面から棚板まで届く角箱で囲う。手前（機械の中心側）は開けたまま
    # にして、継手とカップリングカバーは見えるように残す
    "cover_panel_t_x_plate_t": 0.25,      # 板金 板厚
    "cover_clear_x_plate_t": 0.25,        # 内側の物との隙間
    # 点検窓。前は同じ寸法の板を端面に貼っただけの「蓋」で、中は見えなかった。
    # 実際に開口して桟を渡す
    "cover_lid_w_frac": 0.62,             # 点検窓の幅 / 端面の幅
    "cover_lid_h_frac": 0.50,             # 同 高さ / 端面の高さ
    "cover_bars": 3,                      # 同 桟の本数
    "cover_knob_d_x_plate_t": 1.00,       # 窓枠の摘み 径
    "cover_knob_len_x_plate_t": 1.50,     # 同 長さ
    "cover_knob_at_frac": 0.30,           # 摘みを窓の幅のどこに置くか
}

# 材質。鋳物の本体は架台（#7c848c）より暗いつや消し、モータ胴はやや青みのある灰、
# 油面窓は暗い琥珀。継手と軸だけは削り出しの明るい金属にして、駆動列を目で追える
# ようにしてある。板金カバーは架台に近い明るさで、鋳物と見分けがつくようにした。
# すべて不透明。
MATERIALS = {
    "drv_cast":  dict(color="#565d64", pbr=True, metallic=0.22, roughness=0.86),
    "drv_motor": dict(color="#6d7987", pbr=True, metallic=0.42, roughness=0.52),
    "drv_shaft": dict(color="#c6ced5", pbr=True, metallic=0.85, roughness=0.28),
    "drv_sight": dict(color="#4a2a08", pbr=True, metallic=0.10, roughness=0.22),
    "drv_guard": dict(color="#98a1a9", pbr=True, metallic=0.40, roughness=0.55),
    "drv_mount": dict(color="#6f777f", pbr=True, metallic=0.50, roughness=0.66),
    "drv_panel": dict(color="#8b939b", pbr=True, metallic=0.30, roughness=0.62),
    "drv_gear":  dict(color="#7e878f", pbr=True, metallic=0.78, roughness=0.34),
    # 近接センサだけは金属から外す。暗い樹脂にしておくと、灰色の機構の中で
    # 「回っているものを見張っている物」がどこにあるか一目で分かる
    "drv_sensor": dict(color="#2f343a", pbr=True, metallic=0.05, roughness=0.45),
}

# 描く順。すべて不透明なので前後は深度で決まるが、材質の載せ忘れを防ぐために
# MATERIALS と 1 対 1 で並べる。ここに無い群は絵に出ない。
DRAW_ORDER = (
    ("drv_mount", "drv_mount"),
    ("drv_cast", "drv_cast"),
    ("drv_motor", "drv_motor"),
    ("drv_shaft", "drv_shaft"),
    ("drv_sight", "drv_sight"),
    ("drv_guard", "drv_guard"),
    ("drv_panel", "drv_panel"),
    ("drv_home", "drv_shaft"),
    ("drv_sensor", "drv_sensor"),
    ("drv_gear_table", "drv_gear"),
)

# 駆動系は架台側に固定されていて、テーブルと一緒には回らない。
# ただし大歯車と原点取り出しの縦軸だけは別で、下の SPIN_CENTERS で自分の
# 軸に回す。
ROTATING = False

# 歯車の群名。**このモジュールが組むのはテーブル側の大歯車 1 枚だけ。**
# 星車に付く小歯車 2 枚は星車の軸に乗る部品なので受け渡し系が組む
# （`trf_gear_*`）。同じ物を二重に置かない。ここに星車側の名前が残って
# いるのは、噛み合いを数で確かめるときに相手のピッチ円を作るため。
GEAR_TABLE = "drv_gear_table"
GEAR_STARS = {"infeed": "_star_in", "discharge": "_star_out"}

# 原点検出の取り出し。縦軸・ドグ板・ドグがこの群に入る。**停留中も回る
# 唯一の群。** かさ歯車で入力軸から 1:1 で取っているので、1 タクトで 1 回転。
HOME_SHAFT = "drv_home"


# 入力軸を横へ逃がす向き。ユニット本体はテーブルと同軸なので、この角度は
# 「モータをどの面へ張り出すか」だけを決める。
#
# **これは世界角ではなく、架台の局所座標での角。** 架台は 4 隅に脚が立つ
# 角パイプ枠で、脚は架台の局所 45/135/225/315deg（＝正方形の 4 隅）に立つ。
# 局所 90deg は隣り合う脚 2 本のちょうど真ん中、つまり枠の 1 面の芯を向く。
# 面の芯へ出せば脚は左右に等しく分かれ、モータが脚を串刺しにしない。
#
# 架台そのものは `frame_leg_phase_deg` ぶん振ってある（脚が供給スターホイール
# の軸と同心になるのを避けるため）。駆動系は架台に載る一式なので、**振りを
# そのまま足して世界へ置く。** 足し忘れると梁と板金カバーだけが元の向きに
# 残り、腰板と上桁を突き抜ける。世界角は layout() が `drv_angle_deg` に出す。
#
# 面を 4 つのうちどれにするかは、他の面が埋まっているので選べない。
# **架台は振ってあるので、世界角と局所角は `frame_leg_phase_deg`（-22.5deg）
# ぶんずれる。** 局所 = 世界 - 振り。
#   世界 315deg（局所 337.5deg）  供給コンベアが張り出す側
#   世界 135deg（局所 157.5deg）  配線ダクトが這う側
#   世界 225deg（局所 247.5deg）  排出コンベアと排出スターホイールがいる
# 残るのは局所 90deg で、そこだけが空いている。世界では 67.5deg。
DRIVE_FACE_DEG = 90.0


def _drive_yaw_deg(leg_phase_deg: float) -> float:
    """局所座標を世界へ回す角。局所 -x が架台の 1 面の芯を向く。"""
    return DRIVE_FACE_DEG + float(leg_phase_deg) - 180.0


def _takeoff_local_x(plate_d: float) -> float:
    """原点取り出しの縦軸の局所 x。露出した入力軸の上、ボス寄り。

    `layout()` と、読み込み時に `SPIN_CENTERS` を組む所の**両方から呼ぶ。**
    どちらか一方に式を写すと、自転の中心だけが元の位置に取り残される。
    """
    p = PROPORTION
    box_w = float(plate_d) * p["unit_box_w_x_plate_d"]
    boss_len = float(plate_d) * p["unit_input_boss_len_x_plate_d"]
    free_len = float(plate_d) * p["input_shaft_free_x_plate_d"]
    return -(box_w / 2.0 + boss_len) - free_len * p["takeoff_at_free"]


def _takeoff_center_world(plate_d: float, leg_phase_deg: float) -> tuple:
    """原点取り出しの縦軸の世界 (x, y)。自転の中心そのもの。"""
    th = np.radians(_drive_yaw_deg(leg_phase_deg))
    tx = _takeoff_local_x(plate_d)
    return (float(tx * np.cos(th)), float(tx * np.sin(th)))


def _spin_center_inputs() -> tuple:
    """自転の中心を出すための (テーブル外径, 架台の振り)。読めなければ None。

    テーブル外径は params.json、架台の振りは scene の PROPORTION が正典。
    **scene は import しない。** モジュールの本体で import すると、
    「asm_drive を先に読んでから scene を読む」順のときに、scene の
    load_assemblies が書きかけの asm_drive を掴んで駆動系がまるごと絵から
    消える（実際に build() が無いと言われて飛ばされる）。読まれた時点で
    sys.modules にいれば拾う、いなければ控えで組んで stderr に出す。
    """
    plate_d = None
    try:
        path = Path(__file__).resolve().parent.parent / "params.json"
        with open(path, encoding="utf-8") as fp:
            plate_d = float(json.load(fp)["table"]["plate_diameter_mm"])
    except Exception as exc:                          # noqa: BLE001 読めなければ控え
        print("[asm_drive] params.json が読めない（%s）。自転の中心を控えで組む"
              % exc, file=sys.stderr)

    leg_phase = None
    mod = sys.modules.get("scene")
    if mod is not None:
        try:
            leg_phase = float(mod.PROPORTION["frame_leg_phase_deg"])
        except Exception as exc:                      # noqa: BLE001 同上
            print("[asm_drive] 架台の振り（frame_leg_phase_deg）が読めない"
                  "（%s）。自転の中心を控えで組む" % exc, file=sys.stderr)
    return (plate_d, leg_phase)


class _SpinCenters(Mapping):
    """自転する群 -> 自転の中心 (x, y)。**中身は読まれた瞬間に決まる。**

    scene は読み込みのときに `SPIN_CENTERS` を値ごと写すので、`layout()` から
    入れ替える道が無い。一方で原点取り出しの中心は架台の振りに乗るので、
    振りが分かってからでないと出せない。**読まれた時に組む**ことで、
    どちらの読み込み順でも正しい中心が写る。両方揃ったときだけ覚える。

    テーブル側の大歯車は機械の軸の上なので中心は原点で固定。
    """

    def __init__(self):
        self._table = None

    def _resolve(self) -> dict:
        if self._table is not None:
            return self._table
        plate_d, leg_phase = _spin_center_inputs()
        table = {
            GEAR_TABLE: (0.0, 0.0),
            HOME_SHAFT: _takeoff_center_world(
                plate_d if plate_d else 560.0,
                leg_phase if leg_phase is not None else 0.0),
        }
        if plate_d is not None and leg_phase is not None:
            self._table = table
        return table

    def __getitem__(self, key):
        return self._resolve()[key]

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self):
        return len(self._resolve())

    def __repr__(self):
        return repr(self._resolve())


SPIN_CENTERS = _SpinCenters()


def spin_angles(params: dict, lay: dict, state) -> dict:
    """群名 -> 自転角 [rad]。state.cam_angle_rad（入力軸角 psi）から出す。

    大歯車は出力ボスに締まっているのでテーブルと同じ角で回る。噛み合って
    いる星車側はその逆向きに速比倍で回るが、あちらを回すのは受け渡し系。
    速比が 2.0 ちょうど・歯数が 90 と 45 なので、どの角でも噛み合いは保たれる。

    原点取り出しの縦軸は**入力軸そのものの角**で回す。入力軸は 20 rpm の
    連続回転で、1 タクトで 1 回転。**割出しでも停留でも同じ速さで回り続ける**
    ので、テーブルが止まっている 1.5 秒のあいだも絵が動く。かさ歯車は 1:1、
    直角の取り出しなので向きは反転する（符号を反転させる）。
    """
    import scene                                       # noqa: PLC0415 循環参照を避ける

    psi = float(getattr(state, "cam_angle_rad", 0.0) or 0.0)
    return {GEAR_TABLE: float(scene.table_angle_from_cam(params, psi)),
            HOME_SHAFT: -psi}


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
def _first(lay: dict, names, fallback: float, what=None) -> float:
    """names のうち lay にある最初のものを返す。無ければ fallback。

    受け渡し系のキー名が変わっている途中でも組めるようにするための逃げ。
    **1 つも無ければ stderr に出す。** 黙って控えを使うと、相手がキー名を
    変えただけで歯車の平面が 2 mm ずれ、そのまま気付かずに焼き上がる。
    """
    for name in names:
        if name in lay:
            return float(lay[name])
    if what is not None:
        print("[asm_drive] %s のキー（%s）が lay に無い。控え %.4f で組む"
              % (what, " / ".join(names), float(fallback)), file=sys.stderr)
    return float(fallback)


def _need(lay: dict, key: str, fallback, what: str):
    """lay からキーを 1 つ読む。**無ければ stderr に出してから控えを返す。**

    `lay.get(key, 控え)` は書きやすいが、相手がキー名を変えても消しても
    静かに動く。架台の振り（`frame_leg_phase_deg`）がまさにそれで、受け渡し
    系は -22.5、こちらは控えの 0.0 のまま、警告も出ずに食い違っていた。
    """
    if key in lay:
        return lay[key]
    print("[asm_drive] %s のキー %r が lay に無い。控え %r で組む"
          % (what, key, fallback), file=sys.stderr)
    return fallback


def _publish(lay: dict, key: str, value) -> None:
    """`build()` で確定した値を lay へ出す。

    高さのように「相手が合流したあとでないと決まらない」ものは layout() で
    出せない。ここで出せば、あとから組む他のモジュールは実際に組んだ値を
    読める。同じキーに違う値を 2 度入れたら stderr に出す（黙って上書き
    すると、どちらが効いているのか絵からは分からない）。
    """
    old = lay.get(key)
    if old is not None and np.any(np.asarray(old, dtype=float)
                                  != np.asarray(value, dtype=float)):
        print("[asm_drive] %r を上書きしようとした: %r -> %r"
              % (key, old, value), file=sys.stderr)
    lay[key] = value


def _plate_panels(normal: str, center, span_u: float, span_v: float,
                  thickness: float, window=None, window_at=(0.0, 0.0),
                  bars=0, bar_w=0.0, matrix=None) -> list:
    """板金 1 枚。真ん中に四角い窓を空けられる。戻りは PolyData の並び。

    normal は板の法線（'x' / 'y' / 'z'）。板の面内の 2 軸 (u, v) は
    x -> (y, z) / y -> (x, z) / z -> (x, y) の順。window に (幅, 高さ) を
    渡すと、その周りの 4 本の帯に分ける。**この環境ではブール演算で穴を
    開けないので、開口は帯に割って作る。** bars は開口に渡す桟の本数で、
    v 方向に立てる。window_at は開口の中心を板の中心からずらす量。
    """
    idx = {"x": (1, 2, 0), "y": (0, 2, 1), "z": (0, 1, 2)}[str(normal).lower()]
    iu, iv, ino = idx
    cen0 = [float(v) for v in center]

    def _piece(cu, cv, su, sv):
        if su <= 0.0 or sv <= 0.0:
            return None
        size = [0.0, 0.0, 0.0]
        cen = list(cen0)
        size[iu], size[iv], size[ino] = su, sv, float(thickness)
        cen[iu] += cu
        cen[iv] += cv
        return parts.box(tuple(size), center=tuple(cen), matrix=matrix)

    if not window:
        return [m for m in (_piece(0.0, 0.0, span_u, span_v),) if m is not None]

    wu, wv = (float(v) for v in window)
    ou, ov = (float(v) for v in window_at)
    u0, u1 = ou - wu / 2.0, ou + wu / 2.0
    v0, v1 = ov - wv / 2.0, ov + wv / 2.0
    out = [
        # 開口の左右
        _piece((-span_u / 2.0 + u0) / 2.0, 0.0, u0 + span_u / 2.0, span_v),
        _piece((span_u / 2.0 + u1) / 2.0, 0.0, span_u / 2.0 - u1, span_v),
        # 開口の上下（左右の帯のあいだだけ）
        _piece((u0 + u1) / 2.0, (-span_v / 2.0 + v0) / 2.0, wu,
               v0 + span_v / 2.0),
        _piece((u0 + u1) / 2.0, (span_v / 2.0 + v1) / 2.0, wu,
               span_v / 2.0 - v1),
    ]
    for k in range(1, int(bars) + 1):
        out.append(_piece(u0 + wu * k / (int(bars) + 1.0), (v0 + v1) / 2.0,
                          float(bar_w), wv))
    return [m for m in out if m is not None]


def _tooth_half_angle(pitch_r, teeth, backlash, pressure_rad, radius):
    """歯の半角 [rad]。ピッチ円で歯厚がピッチの半分、外へ出るほど細くなる。

    歯すじを圧力角ぶん傾けた台形で近似する。インボリュートは刻まない。
    この縮尺では歯 1 枚が 1 画素前後なので、噛み合いが成立する幅と逃げが
    出ていれば足りる。
    """
    half_at_pitch = (np.pi / (2.0 * teeth)) * (1.0 - backlash)
    return half_at_pitch - (radius - pitch_r) * np.tan(pressure_rad) / pitch_r


def _gear_outline(pitch_r, teeth, addendum, dedendum, backlash, pressure_rad,
                  phase_rad, arc_steps=3) -> np.ndarray:
    """平歯車の外形（反時計回りの (x, y) の列）。歯先と歯底を台形で結ぶ。

    phase_rad は「歯 1 枚の中心」が向く角。噛み合う相手とはここで位相を
    合わせる。片方の歯の中心が中心線上に来たとき、相手は歯溝の中心が
    中心線上に来ていなければならない。
    """
    r_tip = pitch_r + addendum
    r_root = pitch_r - dedendum
    psi_tip = _tooth_half_angle(pitch_r, teeth, backlash, pressure_rad, r_tip)
    psi_root = _tooth_half_angle(pitch_r, teeth, backlash, pressure_rad, r_root)
    pitch_angle = 2.0 * np.pi / teeth
    if psi_tip <= 0.0 or psi_root >= pitch_angle / 2.0:
        raise ValueError(
            f"歯が成立しない: 歯数 {teeth} / 歯先半角 {np.degrees(psi_tip):.3f}deg /"
            f" 歯底半角 {np.degrees(psi_root):.3f}deg / ピッチ半角 "
            f"{np.degrees(pitch_angle / 2.0):.3f}deg")

    pts = []
    for i in range(int(teeth)):
        th = phase_rad + pitch_angle * i
        # 歯 1 枚（歯底 -> 歯先 -> 歯先 -> 歯底）
        for r, a in ((r_root, th - psi_root), (r_tip, th - psi_tip),
                     (r_tip, th + psi_tip), (r_root, th + psi_root)):
            pts.append((r * np.cos(a), r * np.sin(a)))
        # 次の歯までの歯溝。歯底の円弧を刻んで、直線で切り落とさないようにする
        a0 = th + psi_root
        a1 = th + pitch_angle - psi_root
        for k in range(1, int(arc_steps)):
            a = a0 + (a1 - a0) * k / float(arc_steps)
            pts.append((r_root * np.cos(a), r_root * np.sin(a)))
    return np.asarray(pts, dtype=float)


def _gear_teeth(pitch_r, teeth, addendum, dedendum, backlash, pressure_rad,
                phase_rad, face, base_z, bite, center=(0.0, 0.0)) -> list:
    """歯を 1 枚ずつ角柱にして並べる。戻りは PolyData の並び。

    外形をまるごと押し出すと中実の円板になるので、環にしたい歯車ではこちらを
    使う。付け根は歯底円より `bite` だけ内へ入れて、胴の筒と重ねる（環の外周
    は円、歯の付け根は弦なので、突き合わせると隙間が出る）。
    """
    r_tip = pitch_r + addendum
    r_root = pitch_r - dedendum
    psi_tip = _tooth_half_angle(pitch_r, teeth, backlash, pressure_rad, r_tip)
    psi_root = _tooth_half_angle(pitch_r, teeth, backlash, pressure_rad, r_root)
    pitch_angle = 2.0 * np.pi / teeth
    if psi_tip <= 0.0 or psi_root >= pitch_angle / 2.0:
        raise ValueError(
            f"歯が成立しない: 歯数 {teeth} / 歯先半角 {np.degrees(psi_tip):.3f}deg /"
            f" 歯底半角 {np.degrees(psi_root):.3f}deg")

    r_base = r_root - float(bite)
    out = []
    for i in range(int(teeth)):
        th = phase_rad + pitch_angle * i
        poly = [(r_base, th - psi_root), (r_tip, th - psi_tip),
                (r_tip, th + psi_tip), (r_base, th + psi_root)]
        pts = np.array([(r * np.cos(a), r * np.sin(a)) for r, a in poly])
        out.append(parts.extrude_polygon(pts + np.asarray(center, dtype=float),
                                         face, base_z=base_z))
    return out


def _gear_train(lay: dict, strict=False) -> dict:
    """テーブル出力軸と星車 2 台を繋ぐ平歯車 3 枚の諸元。

    strict を立てると、受け渡し系のキーが無いときに stderr へ出す。
    **`layout()` からは立てない。** モジュールの合流順の都合で、こちらの
    `layout()` の時点では `trf_*` がまだ 1 つも lay に無い（実測で 27 個
    すべて不在）。そこで鳴らしても「まだ来ていない」だけなので意味が無い。
    `build()` のときは合流済みなので、無ければ本当に消えている。

    ピッチ円半径は決め打ちにしない。中心距離と速比だけで決まる。

        r1 + r2 = 中心距離   かつ   r1 / r2 = 速比
        -> r1 = テーブルのピッチ円半径   r2 = 星車のピッチ円半径

    つまり lay にある 2 つのピッチ円をそのまま使えばよい。歯車の噛み合いは
    「2 つのピッチ円が接して転がる」ことなので、受け渡しの条件と同じものが
    そのまま歯車の条件になっている。

    モジュールは星車側の歯数から出す（m = 2 * r2 / z2）。テーブル側の歯数は
    その速比倍で、割り切れなければ噛み合わないので stderr に出す。

    高さは架台天板の上面とテーブル板の下面のあいだ。受け渡し系がその平面を
    lay へ出していればそれに合わせる。
    """
    p = PROPORTION
    t = lay["plate_t"]
    ratio = float(p["gear_ratio"])

    # --- ピッチ円と中心距離 -----------------------------------------------
    r_table = float(lay["pitch_r"])
    r_star = float(lay.get("trf_star_pcd_r", r_table / ratio))
    center_r = float(lay.get("trf_center_r", r_table + r_star))
    if abs((r_table + r_star) - center_r) > 1e-6:
        print("[asm_drive] ピッチ円が中心距離で接していない: %.4f + %.4f != %.4f"
              % (r_table, r_star, center_r), file=sys.stderr)
    if abs(r_table / r_star - ratio) > 1e-6:
        print("[asm_drive] ピッチ円の比が速比と合わない: %.6f != %.6f"
              % (r_table / r_star, ratio), file=sys.stderr)

    # --- 歯数とモジュール ---------------------------------------------------
    z_star = int(p["gear_teeth_star"])
    z_table = int(round(ratio * z_star))
    module = 2.0 * r_star / z_star
    if abs(module * z_table / 2.0 - r_table) > 1e-6:
        print("[asm_drive] 歯数とピッチ円が合わない: m %.4f * z %d / 2 = %.4f "
              "!= %.4f" % (module, z_table, module * z_table / 2.0, r_table),
              file=sys.stderr)

    addendum = module * p["gear_addendum_x_module"]
    dedendum = module * p["gear_dedendum_x_module"]

    # --- 高さ ---------------------------------------------------------------
    # 架台天板の上面とテーブル板の下面のあいだに 1 平面。歯の幅は小歯車と
    # 揃える（受け渡し系が出している平面をそのまま使う）。下は止まっている
    # 天板が相手なので必ず空ける。
    #
    # **環の胴だけはテーブル板の下面まで立ち上げる。** 歯の幅で止めると環が
    # どこにも触れず、宙に浮いた輪になる。胴の頭を板の下面へ突き当てて締める
    deck_gap = t * p["gear_deck_gap_x_plate_t"]
    base_z = _first(lay, ("trf_gear_base_z",), lay["deck_top"] + deck_gap,
                    what="歯車の据わり" if strict else None)
    face = _first(lay, ("trf_gear_face", "trf_gear_h"),
                  lay["table_base"] - base_z, what="歯幅" if strict else None)
    body_top = float(lay["table_base"])               # 環の胴の頭 = 板の下面
    room = lay["table_base"] - lay["deck_top"]
    if base_z < lay["deck_top"] or base_z + face > lay["table_base"]:
        print("[asm_drive] 歯車が天板とテーブル板のあいだに収まらない: "
              "z %.2f..%.2f / 使える帯 %.2f..%.2f（%.2f mm）"
              % (base_z, base_z + face, lay["deck_top"], lay["table_base"],
                 room), file=sys.stderr)

    # --- 3 枚の中心 ---------------------------------------------------------
    # 歯の位相は中心から決まるので `_gear_phase` に任せる
    centers = _gear_centers_from_lay(lay, center_r)

    # --- 大歯車の内穴 -------------------------------------------------------
    # 旋回軸受リングは天板の上面からテーブル板の下面まで丸ごと詰まっている
    # ので、その外側へ逃がす。歯車は軸ではなくテーブル板の下面に締まる環になる
    bore_r = lay["bearing_ro"] + t * p["gear_bore_gap_x_plate_t"]
    root_r_table = r_table - dedendum
    bolt_r = bore_r + (root_r_table - bore_r) * p["gear_bolt_at"]
    return {
        "ratio": ratio,
        "module": module,
        "teeth_table": z_table,
        "teeth_star": z_star,
        "pitch_r_table": r_table,
        "pitch_r_star": r_star,
        "center_r": center_r,
        "addendum": addendum,
        "dedendum": dedendum,
        "backlash": float(p["gear_backlash"]),
        "pressure_rad": float(np.radians(p["gear_pressure_deg"])),
        "face": face,
        "base_z": base_z,
        # 環の胴。歯より背が高く、頭がテーブル板の下面に突き当たる
        "body_top": body_top,
        "body_h": body_top - base_z,
        "centers": centers,
        "bore_r": bore_r,
        "root_r_table": root_r_table,
        "tooth_bite": t * p["gear_tooth_bite_x_plate_t"],
        # テーブル板の下面へ締めるボルト。頭は歯車の下面から出るので、
        # 天板とのすき間より低く抑える
        "bolts": int(p["gear_bolts"]),
        "bolt_r": bolt_r,
        "bolt_d": t * p["gear_bolt_d_x_plate_t"],
        # 頭の丈は「実際に空いた隙間」から取る。控えの deck_gap で決めると、
        # 受け渡し系がもっと低い平面を出したときに頭が天板を叩く
        "bolt_h": (base_z - lay["deck_top"]) * p["gear_bolt_h_x_gap"],
        "deck_gap": base_z - lay["deck_top"],
    }


def _check_gear_against_transfer(gt: dict, lay: dict) -> None:
    """小歯車を組む受け渡し系と、こちらの諸元が合っているかを照らす。

    **必ず `build()` から呼ぶ。** `layout()` の時点では `trf_*` が 1 つも
    lay に無いので、そこで照らしても全部「相手が居ない」で終わる（以前は
    `layout()` から呼んでいて、`lay.get(key)` が常に None を返し、この照合が
    一度も鳴らなかった）。

    受け渡し系は歯車の寸法を直径で出すものと半径で出すものが混ざっている
    ので、こちらで直径に揃えてから比べる。
    """
    checks = (
        ("trf_gear_pcd_r", gt["pitch_r_star"], "星車のピッチ円半径"),
        ("trf_gear_pcd", 2.0 * gt["pitch_r_star"], "星車のピッチ円直径"),
        ("trf_gear_teeth", gt["teeth_star"], "星車の歯数"),
        ("trf_gear_module", gt["module"], "モジュール"),
        ("trf_gear_tip_d", 2.0 * (gt["pitch_r_star"] + gt["addendum"]),
         "星車の歯先円直径"),
        ("trf_gear_root_d", 2.0 * (gt["pitch_r_star"] - gt["dedendum"]),
         "星車の歯底円直径"),
        ("trf_gear_center_dist", gt["center_r"], "中心距離"),
        ("trf_center_r", gt["center_r"], "星車の中心半径"),
        ("trf_gear_base_z", gt["base_z"], "歯車の下面"),
        ("trf_gear_top_z", gt["base_z"] + gt["face"], "歯の上面"),
        ("trf_gear_face", gt["face"], "歯幅"),
    )
    missing = []
    for key, here, what in checks:
        if key not in lay:
            missing.append(key)
            continue
        there = float(lay[key])
        if abs(there - float(here)) > 1e-6:
            print("[asm_drive] %s が受け渡し系と合わない: こちら %.4f / 向こう "
                  "%.4f (%s)" % (what, float(here), there, key), file=sys.stderr)
    # 相手が出していないキーは、名前が変わったのか、そもそも出していないのか
    # 分からない。**片方だけ静かに動くのがいちばん困る**ので、まとめて出す
    if missing:
        print("[asm_drive] 受け渡し系が出していない歯車のキー: %s"
              % ", ".join(missing), file=sys.stderr)


def _gear_phase(gt: dict, name: str) -> float:
    """歯車 1 枚の位相 [rad]。「歯 1 枚の中心」が向く角。

    テーブル側は好きに決めてよい（0deg に歯の中心を置く）。星車側はそれに
    合わせて解く。噛み合いの条件は「相手の歯の中心が、こちらの歯溝の中心と
    ピッチ点で出会う」こと。2 つのピッチ円は滑らずに転がるので、ピッチ点
    からの弧長で合わせればよい。向きは反転するので符号が入れ替わる。

        テーブル側の歯の中心がピッチ点から弧長 +a のところにある
            -> 星車側の歯溝の中心は弧長 -a のところに来る

    工程角がピッチ角の整数倍とは限らないので、単に半ピッチずらすのでは
    合わない。ずれ a を実際に測ってから移す。
    """
    if name == GEAR_TABLE:
        return 0.0

    cx, cy = gt["centers"][name]
    axis = float(np.arctan2(cy, cx))                 # テーブル中心から見た向き
    p1 = 2.0 * np.pi / gt["teeth_table"]
    p2 = 2.0 * np.pi / gt["teeth_star"]
    # ピッチ点にいちばん近いテーブル側の歯の中心。そこまでの角のずれ
    k = round((axis - _gear_phase(gt, GEAR_TABLE)) / p1)
    delta = (_gear_phase(gt, GEAR_TABLE) + k * p1) - axis
    # 弧長を星車側の角へ直す（= 速比倍）。転がりで向きが反転するので引く
    space = axis + np.pi - gt["ratio"] * delta
    return space + p2 / 2.0                          # 歯溝の中心 -> 歯の中心


def _mesh_clearance(gt: dict, samples=4) -> dict:
    """噛み合っている 2 枚の歯のあいだの最小すき間 [mm]。群名 -> すき間。

    負なら歯が食い込んでいる。歯車はどちらも中心から見て星形（半径が角度の
    1 価関数）なので、片方の外形の点をもう片方の中心から見た (半径, 角) に
    直し、その角での相手の外形半径と比べれば内外が決まる。辺の途中で
    いちばん近づくので、外形は刻み直してから測る。
    """
    def outline(name):
        is_table = (name == GEAR_TABLE)
        pts = _gear_outline(
            gt["pitch_r_table"] if is_table else gt["pitch_r_star"],
            gt["teeth_table"] if is_table else gt["teeth_star"],
            gt["addendum"], gt["dedendum"], gt["backlash"], gt["pressure_rad"],
            _gear_phase(gt, name))
        # 辺を刻む。頂点だけだと台形の斜面の途中を見落とす
        nxt = np.roll(pts, -1, axis=0)
        dense = [pts + (nxt - pts) * (k / float(samples)) for k in range(samples)]
        cx, cy = gt["centers"][name]
        return np.concatenate(dense, axis=1).reshape(-1, 2) + np.array([cx, cy])

    def radius_at(name, pts):
        """name の外形が、その中心から見て角 th のときに持つ半径。"""
        cx, cy = gt["centers"][name]
        own = outline(name) - np.array([cx, cy])
        th = np.arctan2(own[:, 1], own[:, 0])
        r = np.hypot(own[:, 0], own[:, 1])
        q = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        # 周期を渡すと numpy 側で並べ替えと折り返しをやってくれる。
        # 頂点のあいだを直線で結ぶぶん半径をわずかに多めに見るので、
        # すき間は控えめに出る（食い込みを見落とす側には転ばない）
        return np.interp(q, th, r, period=2.0 * np.pi)

    out = {}
    tbl = outline(GEAR_TABLE)
    for name in GEAR_STARS.values():
        star = outline(name)
        cx, cy = gt["centers"][name]
        # 星車の外形の点が、テーブル歯車の外形の内側に入っていないか
        d_star = np.hypot(star[:, 0], star[:, 1]) - radius_at(GEAR_TABLE, star)
        # テーブルの外形の点が、星車歯車の外形の内側に入っていないか
        d_tbl = (np.hypot(tbl[:, 0] - cx, tbl[:, 1] - cy)
                 - radius_at(name, tbl))
        out[name] = float(min(d_star.min(), d_tbl.min()))
    return out


def _leg_gap(lay: dict, center, radius: float, z0=None) -> tuple:
    """架台の脚と、半径 radius の丸物とのすき間 [mm]。負なら食い込み。

    脚の芯は基盤側が `frame_leg_xy` に、世界角を `frame_leg_deg` に出して
    いる。**45/135/225/315deg を直書きしない**（脚は振ってある）。角パイプの
    断面もその角ぶん傾いているので、測る前に脚の局所へ戻す。

    z0 に丸物の下端を渡すと、脚の上端（天板の下面）より上にあるかを見る。
    上にあれば脚は平面に存在しないので、当たりようがない。
    戻りは (すき間, いちばん近い脚の座標)。すき間が無限大なら高さで外れている。
    """
    leg_top = float(lay["deck_top"]) - float(lay["deck_t"])
    if z0 is not None and float(z0) >= leg_top:
        return (float("inf"), (0.0, 0.0))
    pipe_half = float(lay["frame_pipe"]) / 2.0
    # 1 本目の世界角から振りを出す。parts.frame() は対角（45deg）に立てる。
    # スカラでも 4 本ぶんの並びでも受ける
    leg_deg = lay["frame_leg_deg"]
    if isinstance(leg_deg, (list, tuple)):
        leg_deg = leg_deg[0]
    phase = np.radians(float(leg_deg) - 45.0)
    cos_p, sin_p = np.cos(-phase), np.sin(-phase)
    best = None
    for leg in lay["frame_leg_xy"]:
        dx = float(center[0]) - float(leg[0])
        dy = float(center[1]) - float(leg[1])
        # 脚の局所へ戻してから角の断面で測る
        qx = cos_p * dx - sin_p * dy
        qy = sin_p * dx + cos_p * dy
        ex = max(abs(qx) - pipe_half, 0.0)
        ey = max(abs(qy) - pipe_half, 0.0)
        inside = min(max(abs(qx) - pipe_half, abs(qy) - pipe_half), 0.0)
        gap = float(np.hypot(ex, ey) + inside) - float(radius)
        if best is None or gap < best[0]:
            best = (gap, (float(leg[0]), float(leg[1])))
    return best


def _gear_centers_from_lay(lay: dict, center_r: float) -> dict:
    """歯車 3 枚の中心 (x, y)。lay の工程角から作る。

    テーブル側は機械の軸の上。星車側は 2 つのピッチ円が接する半径の上に
    乗る。星車側の 2 つはこのモジュールでは組まないが、大歯車の歯の位相と
    噛み合いの確認に要るので、位置だけは同じ式で出しておく。
    """
    out = {GEAR_TABLE: (0.0, 0.0)}
    for role, key in (("infeed", "infeed_deg"), ("discharge", "discharge_deg")):
        th = np.radians(float(lay[key]))
        out[GEAR_STARS[role]] = (center_r * float(np.cos(th)),
                                 center_r * float(np.sin(th)))
    return out


def layout(params: dict, lay: dict) -> dict:
    """駆動系の座標と寸法。`lay` は読むだけ。

    高さの鎖はひとつながりになっている。

        天板下面 -> 本体箱の上面 -> 本体箱の底 -> 入力軸の高さ
                 -> ギヤモータの軸高さ -> ベース下面 -> 棚板 -> 梁

    上へは別の鎖が伸びる。

        ボスの頭（天板下面）-> 出力軸 -> 天板を貫通 -> 旋回軸受の内側
                           -> テーブル板の下面のハブ

    横へも 1 本伸びる。ボスの外周に大歯車を締め、天板の下で星車 2 台の
    小歯車と噛み合わせる。ピッチ円半径は中心距離と速比だけで決まるので、
    ここでは決め打ちにしない（`_gear_train` を見ること）。

    どこか 1 つ動かせば下流が全部ついてくるので、軸の高さがずれることはない。

    z は世界そのまま。x/y は入力軸を -x へ出した状態の局所座標で返す。
    世界へは z 軸まわりに `drv_yaw_deg` 回した位置に来る（本体はテーブルと
    同軸なので、回っても軸の高さと半径は変わらない）。

    **この局所座標は架台の局所座標に固定してある。** 局所 -x が架台の 1 面の
    芯を向き、脚 2 本が左右に等しく分かれる。梁の長さも板金カバーの張り出しも
    `frame_span` から出しているので、架台が振れたらこの一式もそのまま振れる。
    """
    p = PROPORTION
    d = lay["plate_d"]
    t = lay["plate_t"]

    # 架台の振り。**駆動系は架台に載る一式なので、この振りを丸ごと引き継ぐ。**
    # 引き継がないと、frame_span から長さを出している梁と板金カバーだけが
    # 元の向きに残って腰板・上桁を突き抜ける。
    leg_phase = float(_need(lay, "frame_leg_phase_deg", 0.0, "架台の振り"))
    drive_deg = DRIVE_FACE_DEG + leg_phase
    # 架台の外側の面までの距離。脚の芯々の半分に角パイプの半分を足したところ。
    # 板金カバーはこれより外へ出す（内側へ入れると上桁を飲み込む）
    frame_face = lay["frame_span"] / 2.0 + lay["frame_pipe"] / 2.0

    # --- 本体箱と出力フランジ ---------------------------------------------
    box_w = d * p["unit_box_w_x_plate_d"]
    box_d = d * p["unit_box_d_x_plate_d"]
    box_h = d * p["unit_box_h_x_plate_d"]
    skirt_half = box_w * parts.INDEX_UNIT_SKIRT_SCALE / 2.0   # 据付段の端面

    flange_t_outer = t * p["unit_flange_t_outer_x_plate_t"]
    flange_t_inner = t * p["unit_flange_t_inner_x_plate_t"]
    boss_h = d * p["unit_boss_h_x_plate_d"]
    boss_d = d * p["unit_boss_d_x_plate_d"]

    # 天板の下面。ここにボスの頭が突き当たり、ここから上へ出力軸が抜ける
    deck_bottom = lay["deck_top"] - lay["deck_t"]
    stack = flange_t_outer + flange_t_inner + boss_h
    box_top = deck_bottom - stack
    base_z = box_top - box_h
    boss_base = deck_bottom - boss_h

    # --- 出力軸（天板を貫いてテーブルへ） ---------------------------------
    out_shaft_d = boss_d * p["out_shaft_d_x_boss_d"]
    out_shaft_z0 = boss_base + boss_h * (1.0 - p["out_shaft_into_boss"])
    out_shaft_z1 = lay["table_base"]                    # テーブル板の下面まで
    out_seat_h = t * p["out_seat_h_x_plate_t"]
    out_hub_r = lay["bearing_ri"] * p["out_hub_r_x_bearing_ri"]
    # ハブは旋回軸受の内側に収まり、軸受座の頭とテーブル板の下面のあいだを埋める。
    # 座は天板側（止まっている）、ハブはテーブル側（回る）なので必ず隙間を空ける
    out_hub_z0 = lay["deck_top"] + out_seat_h + t * p["out_hub_gap_x_plate_t"]
    out_hub_t = out_shaft_z1 - out_hub_z0

    # --- 同期取り出し（出力ボスに締める 2:1 の平歯車）---------------------
    # 星車側の実寸は受け渡し系が lay へ出す。ただし合流はこのモジュールの
    # あとなので、ここでは控えで解いておき、build() で解き直す
    gt = _gear_train(dict(lay, drv_boss_d=boss_d))

    # --- 入力軸 -----------------------------------------------------------
    # ボスの端面から継手までを露出させる。ここが**この機械で唯一、停留中も
    # 回っている所**で、原点検出の取り出しもここから立てる。中実のカバーで
    # 包むと絵から消えるので、カバーには点検窓を開ける
    input_z = base_z + box_h * parts.INDEX_UNIT_INPUT_AXIS_X_BOX_H
    input_boss_len = d * p["unit_input_boss_len_x_plate_d"]
    input_face_x = -(box_w / 2.0 + input_boss_len)      # 入力ボスの端面
    input_shaft_d = d * p["input_shaft_d_x_plate_d"]
    input_free_len = d * p["input_shaft_free_x_plate_d"]
    shaft_end_x = input_face_x - input_free_len         # 露出した軸の端

    # --- 原点検出の取り出し（かさ歯車 1:1 -> 縦軸 -> ドグ板 -> 近接センサ）
    # 水平の入力軸は自転の仕掛けでは回せない（z 軸まわりしか扱えない）。
    # 直角に 1 回取り出して縦軸へ載せ替えると、そこから先は z 軸まわりに
    # なるので、停留中も回り続けるものを絵に残せる。
    # 縦軸の局所 x は `_takeoff_local_x`（読み込み時に SPIN_CENTERS を組む
    # 所と同じ式）。ここで別に計算すると自転の中心だけが取り残される
    takeoff_x = _takeoff_local_x(d)
    takeoff_gear_r = d * p["takeoff_gear_r_x_plate_d"]
    takeoff_gear_r0 = takeoff_gear_r * p["takeoff_gear_root_frac"]
    takeoff_shaft_d = takeoff_gear_r * p["takeoff_shaft_d_x_gear_r"]

    # --- カップリング -----------------------------------------------------
    coupling_len = d * p["coupling_len_x_plate_d"]
    coupling_d = d * p["coupling_d_x_plate_d"]
    coupling_x = shaft_end_x - coupling_len / 2.0       # 露出軸の端に突き当てる

    # --- ギヤモータ -------------------------------------------------------
    # origin は出力軸の付け根。軸端を継手の engage の位置まで差し込む
    shaft_len = d * p["motor_shaft_len_x_plate_d"]
    shaft_tip_x = shaft_end_x - coupling_len * (1.0 - p["coupling_engage"])
    motor_origin = (shaft_tip_x - shaft_len, 0.0, input_z)

    motor_d = d * p["motor_d_x_plate_d"]
    motor_len = d * p["motor_len_x_plate_d"]
    gearhead_size = d * p["gearhead_size_x_plate_d"]
    gearhead_len = d * p["gearhead_len_x_plate_d"]
    base_w, base_d, base_t = (d * v for v in p["motor_base_x_plate_d"])

    # ベース上面の高さ。parts.gearmotor が胴とギヤヘッドの大きいほうに
    # 逃げを足して置いているので、同じ計算をここでも通す
    half = max(motor_d / 2.0, gearhead_size / 2.0)
    base_top = input_z - half - parts.GEARMOTOR_BASE_GAP_MM
    base_bottom = base_top - base_t
    # ベースの中心は胴の後端の半分の位置（parts.gearmotor と同じ置き方）
    base_cx = motor_origin[0] - (gearhead_len + motor_len) / 2.0
    base_x0 = base_cx - base_w / 2.0
    base_x1 = base_cx + base_w / 2.0

    # --- 安全カバー -------------------------------------------------------
    # ギヤヘッド前面とユニットの据付段のあいだに、上蓋の外形がちょうど収まる
    # 長さにする。ここを継手長さから決めると上蓋の角がギヤヘッドに食い込む
    skirt_face_x = -skirt_half
    guard_len = (skirt_face_x - motor_origin[0]) / parts.COUPLING_COVER_LID_SCALE
    guard_side = coupling_d * p["guard_side_x_coupling_d"]
    guard_bottom = base_bottom                        # 棚板の上面に載せる
    guard_top = input_z + coupling_d / 2.0 * (1.0 + 2.0 * p["guard_head_x_coupling_d"])
    guard_h = guard_top - guard_bottom
    guard_cx = (skirt_face_x + motor_origin[0]) / 2.0
    guard_panel_t = t * p["guard_panel_t_x_plate_t"]
    guard_lid_t = guard_h * parts.COUPLING_COVER_LID_T_X_H
    guard_lid_top = guard_top + guard_lid_t / 2.0     # 上蓋の頭。ここに軸受箱が座る

    # --- 原点検出（つづき。高さは継手カバーの頭から積む）-------------------
    takeoff_housing_h = d * p["takeoff_housing_h_x_plate_d"]
    takeoff_housing_top = guard_lid_top + takeoff_housing_h
    dog_disc_r = d * p["dog_disc_r_x_plate_d"]
    dog_disc_t = t * p["dog_disc_t_x_plate_t"]
    dog_disc_z = takeoff_housing_top + t * p["dog_disc_gap_x_plate_t"]
    dog_chord = -dog_disc_r * p["dog_chord_x_disc_r"]  # 弦の位置（局所）
    dog_top_z = dog_disc_z + dog_disc_t
    # 近接センサは上から下向きに構える。検出面をドグ板の上面へ向け、
    # ステーはカバーの上蓋に立てる。**横向きにするとステーが外へ伸びて
    # 架台の上桁に刺さる**（上桁はドグ板と同じ高さを横切っている）
    sensor_len = t * p["sensor_len_x_plate_t"]
    sensor_z0 = dog_top_z + t * p["sensor_gap_x_plate_t"]
    sensor_at = dog_disc_r * p["sensor_at_disc_r"]     # 局所 +y 側に構える
    stay = t * p["sensor_stay_x_plate_t"]
    stay_at = dog_disc_r * p["sensor_stay_at_disc_r"]
    stay_top = sensor_z0 + sensor_len + stay

    # --- 取付台 -----------------------------------------------------------
    shelf_t = t * p["mount_shelf_t_x_plate_t"]
    shelf_top = base_bottom                           # ベースをそのまま受ける
    shelf_half_w = d * p["mount_shelf_half_w_x_plate_d"]
    # 内端は据付段の手前で止める。ユニットに食い込ませない
    shelf_x1 = -(skirt_half + t * p["mount_shelf_gap_x_plate_t"])
    shelf_x0 = base_x0 - d * p["mount_shelf_out_x_plate_d"]

    # 梁は架台の 2 本の脚（モータを張り出す側）に渡す。棚板の内寄りをこれが受ける。
    #
    # 局所 -x は架台の 1 面の芯なので、その面の脚 2 本は局所 (-span/2, ±span/2)
    # にいる。梁はその 2 本を結ぶ線に乗るから、芯の位置は -span/2、長さは芯々の
    # span。**ただし芯々のまま伸ばすと端が脚の中へ 半パイプぶん埋まる。**
    # 角パイプ 1 本ぶん詰めて、脚の内側の面に突き当てる（実機の突き合わせ溶接と
    # 同じ）。脚の外へ出すと、外側を這う配線ダクトに当たる
    beam_pipe = lay["frame_pipe"]
    beam_x = -lay["frame_span"] / 2.0
    beam_span = lay["frame_span"] - beam_pipe
    beam_top = shelf_top - shelf_t

    # 吊り柱は棚板の外寄りから天板の下面まで。モータ胴の脇を通す
    deck_r = lay["cover_r"] + beam_pipe / 2.0
    post_size = d * p["mount_post_x_plate_d"]
    post_x = -deck_r * p["mount_post_at_x_deck_r"]
    post_y = base_d / 2.0 + post_size / 2.0 + t * p["mount_post_gap_x_plate_t"]

    # --- 板金カバー -------------------------------------------------------
    # 外端は棚板の端に揃える。継手まで包むと駆動列が絵から消えるので、内側は
    # 開けておく。半幅は吊り柱と棚板のどちらも中に収まる側で決める
    panel_t = t * p["cover_panel_t_x_plate_t"]
    panel_clear = t * p["cover_clear_x_plate_t"]
    panel_x0 = shelf_x0
    # 内端はギヤヘッドの前面か架台の外側の面か、外にあるほうで止める。
    # **架台の中まで引き込むと上桁を飲み込む。** 上桁は脚の芯々の線に乗って
    # いて、天板の下面から少し下がった高さ（カバーの丈のちょうど中ほど）を
    # 横切る。内端が架台の面より内側にあると側板が桁を突き抜けるので、
    # 囲うのは架台から外へ張り出したぶんだけにする
    panel_x1 = min(motor_origin[0], -(frame_face + panel_clear))
    panel_half_w = max(post_y + post_size / 2.0, shelf_half_w) + panel_clear
    panel_z0 = shelf_top                              # 棚板の上面に立てる
    panel_z1 = deck_bottom                            # 天井は天板の下面で塞ぐ

    return {
        # 本体
        "drv_box": (box_w, box_d, box_h),
        "drv_base_z": base_z,
        "drv_box_top": box_top,
        "drv_flange_outer_d": d * p["unit_flange_outer_x_plate_d"],
        "drv_flange_inner_d": d * p["unit_flange_inner_x_plate_d"],
        "drv_flange_t_outer": flange_t_outer,
        "drv_flange_t_inner": flange_t_inner,
        "drv_boss_d": boss_d,
        "drv_boss_h": boss_h,
        "drv_boss_base": boss_base,
        "drv_bolt_d": t * p["unit_bolt_d_x_plate_t"],
        "drv_bolt_h": t * p["unit_bolt_h_x_plate_t"],
        "drv_input_boss_d": d * p["unit_input_boss_d_x_plate_d"],
        "drv_input_boss_len": input_boss_len,
        "drv_sight_d": d * p["unit_sight_d_x_plate_d"],
        "drv_oil_port_d": d * p["unit_oil_port_d_x_plate_d"],
        "drv_flange_top": deck_bottom,
        "drv_skirt_half": skirt_half,
        # 出力軸（天板を貫く側）
        "drv_out_shaft_d": out_shaft_d,
        "drv_out_shaft_z": (out_shaft_z0, out_shaft_z1),
        "drv_out_seat_ro": out_shaft_d * p["out_seat_ro_x_shaft_d"],
        "drv_out_seat_h": out_seat_h,
        "drv_out_seat_z": lay["deck_top"],
        "drv_out_hub_r": out_hub_r,
        "drv_out_hub_t": out_hub_t,
        "drv_out_hub_z": out_hub_z0,
        # 同期取り出し。速比・モジュール・歯数と、2 つのピッチ円をここから
        # 出す。受け渡し系が歯車の高さを lay へ出していれば build() が
        # そちらに合わせて解き直すので、高さは控え
        "drv_gear_ratio": gt["ratio"],
        "drv_gear_module": gt["module"],
        "drv_gear_teeth": (gt["teeth_table"], gt["teeth_star"]),
        "drv_gear_pitch_r": (gt["pitch_r_table"], gt["pitch_r_star"]),
        "drv_gear_center_r": gt["center_r"],
        "drv_gear_tip_r": (gt["pitch_r_table"] + gt["addendum"],
                           gt["pitch_r_star"] + gt["addendum"]),
        "drv_gear_root_r": (gt["pitch_r_table"] - gt["dedendum"],
                            gt["pitch_r_star"] - gt["dedendum"]),
        # **高さはここから出さない。** 受け渡し系の layout() はこのモジュール
        # より後に合流するので、ここでは向こうの平面をまだ読めない（実測で
        # `trf_*` が 27 個すべて不在）。以前はここに控えの (870.0, 884.0) を
        # 出していて、実際に組む値（869.6 と歯幅 12.4）と 0.4 / 1.6 mm 食い
        # 違っていた。しかも控えはテーブル板の下面に接する値なので、他の
        # モジュールがそれを根拠に使うと歯の逃げが消える。
        # 確定した高さは build() が `drv_gear_z` / `drv_gear_face` /
        # `drv_gear_body_z` に出す（相手が合流したあとなので実際の値になる）
        "drv_gear_bore_r": gt["bore_r"],
        "drv_gear_bolts": gt["bolts"],
        "drv_gear_bolt_r": gt["bolt_r"],
        # 入力軸まわり
        "drv_input_z": input_z,
        "drv_input_face_x": input_face_x,
        "drv_input_shaft_d": input_shaft_d,
        "drv_input_shaft_x": (shaft_end_x, input_face_x),
        "drv_coupling_d": coupling_d,
        "drv_coupling_len": coupling_len,
        "drv_coupling_x": coupling_x,
        # 原点検出の取り出し（かさ歯車 1:1 -> 縦軸 -> ドグ板 -> 近接センサ）
        "drv_takeoff_x": takeoff_x,
        "drv_takeoff_gear_r": (takeoff_gear_r0, takeoff_gear_r),
        "drv_takeoff_shaft_d": takeoff_shaft_d,
        "drv_takeoff_housing": (takeoff_shaft_d * p["takeoff_housing_ro_x_shaft_d"],
                                guard_lid_top, takeoff_housing_top),
        # ドグ板（半径, 厚み, 下面, 弦の位置）
        "drv_dog_disc": (dog_disc_r, dog_disc_t, dog_disc_z, dog_chord),
        # 近接センサ（径, 長さ, 検出面の高さ, 構える半径。局所 +y 側）
        "drv_sensor": (t * p["sensor_d_x_plate_t"], sensor_len, sensor_z0,
                       sensor_at),
        # 取付ステー（一辺, 立てる位置 +y, 頭の高さ）
        "drv_sensor_stay": (stay, stay_at, stay_top),
        # ギヤモータ
        "drv_motor_origin": motor_origin,
        "drv_motor_d": motor_d,
        "drv_motor_len": motor_len,
        "drv_gearhead_size": gearhead_size,
        "drv_gearhead_len": gearhead_len,
        "drv_shaft_d": d * p["motor_shaft_d_x_plate_d"],
        "drv_shaft_len": shaft_len,
        "drv_fins": int(p["motor_fins"]),
        "drv_terminal_size": tuple(d * v for v in p["terminal_x_plate_d"]),
        "drv_motor_base_size": (base_w, base_d, base_t),
        "drv_motor_base_x": (base_x0, base_x1),
        "drv_motor_base_top": base_top,
        "drv_motor_base_bottom": base_bottom,
        # 継手の安全カバー。中実の箱ではなく板金の組み合わせで、両側面に
        # 点検窓を開ける。上蓋には縦軸の逃げ穴を開ける
        "drv_guard_size": (guard_len, guard_side, guard_h),
        "drv_guard_center": (guard_cx, 0.0, guard_bottom + guard_h / 2.0),
        "drv_guard_panel_t": guard_panel_t,
        "drv_guard_lid": (guard_len * parts.COUPLING_COVER_LID_SCALE,
                          guard_side * parts.COUPLING_COVER_LID_SCALE,
                          guard_lid_t, guard_top),
        "drv_guard_window": (p["guard_window_frac"][0] * guard_len,
                             p["guard_window_frac"][1] * guard_h,
                             int(p["guard_bars"]),
                             t * p["guard_bar_w_x_plate_t"]),
        # 取付台
        "drv_shelf_x": (shelf_x0, shelf_x1),
        "drv_shelf_half_w": shelf_half_w,
        "drv_shelf_t": shelf_t,
        "drv_shelf_top": shelf_top,
        "drv_beam_x": beam_x,
        "drv_beam_pipe": beam_pipe,
        "drv_beam_span": beam_span,
        "drv_beam_top": beam_top,
        "drv_post_size": post_size,
        "drv_post_x": post_x,
        "drv_post_y": post_y,
        "drv_post_top": deck_bottom,
        # 板金カバー
        "drv_panel_t": panel_t,
        "drv_panel_x": (panel_x0, panel_x1),
        "drv_panel_half_w": panel_half_w,
        "drv_panel_z": (panel_z0, panel_z1),
        "drv_panel_lid": (p["cover_lid_w_frac"], p["cover_lid_h_frac"]),
        "drv_panel_bars": (int(p["cover_bars"]), t * p["guard_bar_w_x_plate_t"]),
        "drv_panel_knob": (t * p["cover_knob_d_x_plate_t"],
                           t * p["cover_knob_len_x_plate_t"],
                           p["cover_knob_at_frac"]),
        # 入力軸を逃がす世界角。架台の局所での向き（DRIVE_FACE_DEG）に架台の
        # 振りを足したもの。カメラの方位もここから取る
        "drv_angle_deg": drive_deg,
        # 局所座標を世界へ回す角。build() はこれを 1 枚の変換にして全部品に
        # 掛ける。局所 -x（＝局所 180deg）が世界の drive_deg を向く
        "drv_yaw_deg": drive_deg - 180.0,
        # 架台の外側の面まで（板金カバーの内端を決めた根拠。確認用に出す）
        "drv_frame_face": frame_face,
    }


def build(params: dict, lay: dict) -> dict:
    """メッシュ群の名前 -> PolyData。寸法は lay からしか読まない。

    部品は入力軸を -x へ出した局所座標で組み、最後に 1 枚の回転を全部に
    掛けて世界へ持っていく。回すのは駆動系ぜんぶで同じ角なので、組み方の
    中で角度を気にする所は無い。

    確定した高さは lay へ書き戻す（`_publish`）。`layout()` の時点では受け渡し
    系がまだ合流していないので、歯車の平面はここでしか確定しない。
    """
    groups = {k: [] for k in MATERIALS}
    yaw = parts.transform_matrix(rot_z_deg=lay["drv_yaw_deg"])

    # --- カム式インデックスユニット ---------------------------------------
    unit = parts.index_unit(
        matrix=yaw,
        box_size=lay["drv_box"],
        flange_outer_d=lay["drv_flange_outer_d"],
        flange_inner_d=lay["drv_flange_inner_d"],
        flange_t_outer=lay["drv_flange_t_outer"],
        flange_t_inner=lay["drv_flange_t_inner"],
        boss_d=lay["drv_boss_d"], boss_h=lay["drv_boss_h"],
        bolt_d=lay["drv_bolt_d"], bolt_h=lay["drv_bolt_h"],
        input_boss_d=lay["drv_input_boss_d"],
        input_boss_len=lay["drv_input_boss_len"],
        sight_d=lay["drv_sight_d"], oil_port_d=lay["drv_oil_port_d"],
        base_z=lay["drv_base_z"])
    groups["drv_cast"] += [unit["housing"], unit["flange"], unit["boss"],
                           unit["input_boss"], unit["oil_port"]]
    groups["drv_shaft"].append(unit["bolts"])
    groups["drv_sight"].append(unit["sight_glass"])

    # --- 出力軸。天板を貫いてテーブル板の下面まで --------------------------
    # ここが切れていると、絵の上ではテーブルが何にも繋がっていない。
    # 軸はボスの中から立ち上がり、天板の穴を通り、旋回軸受の内側を抜けて
    # テーブル側のハブへ入る。天板の上には軸受座を入れて、板を突き破った
    # だけの穴に見えないようにする。ボスの頭は天板の下面へそのまま突き当たる
    # ので、押さえ環は要らない
    z0, z1 = lay["drv_out_shaft_z"]
    shaft_r = lay["drv_out_shaft_d"] / 2.0
    groups["drv_shaft"].append(
        parts.cylinder(shaft_r, z1 - z0, base_z=z0, matrix=yaw))

    # 天板の上面。旋回軸受の内側に収まる軸受座
    groups["drv_cast"].append(
        parts.tube(lay["drv_out_seat_ro"], shaft_r, lay["drv_out_seat_h"],
                   base_z=lay["drv_out_seat_z"], matrix=yaw))

    # テーブル板の下面のハブ。旋回軸受の内側に収まり、ここで動力がテーブルへ
    # 渡る。軸はハブの中を通ってテーブル板の下面まで届いている
    groups["drv_shaft"].append(
        parts.cylinder(lay["drv_out_hub_r"], lay["drv_out_hub_t"],
                       base_z=lay["drv_out_hub_z"], matrix=yaw))

    # --- スターホイールへの同期取り出し -----------------------------------
    # 天板の下に平歯車を 3 枚。出力ボスに締めた大歯車（ピッチ円 = テーブルの
    # ボトルピッチ円）に、星車 2 台の小歯車（同 星車のボトルピッチ円）が
    # 噛み合う。外歯どうしなので向きは自然に反転し、テーブル CCW に対して
    # 星車は両方とも CW になる。速比はピッチ円の比そのもので 2.0。
    #
    # 歯は台形で刻む。段を付けただけの円板にすると、歯先円どうしが
    # 中心距離を越えて食い込んだ絵になる（歯先円の和は中心距離より大きい。
    # 実物はそこで歯が互い違いに入るので当たらない）
    gt = _gear_train(lay, strict=True)
    _check_gear_against_transfer(gt, lay)

    # 組むのは大歯車 1 枚だけ。星車に付く小歯車は星車の軸に乗る部品なので
    # 受け渡し系が組む。
    #
    # 環の胴と歯を分けて作る。歯を含めた外形をそのまま押し出すと中実の円板に
    # なり、旋回軸受リングを丸ごと飲み込む。この環境ではブール演算で穴を
    # 開けないので、胴は筒で作り、歯だけを 1 枚ずつ角柱で足す。
    #
    # **胴の頭はテーブル板の下面（`table_base`）まで立ち上げる。** 歯の幅
    # （小歯車と揃える）で止めると、環は上のテーブル板とも下の天板とも
    # どことも触れない。絵では回っていても、機構としては入力が無い輪になる。
    # 板の下面へ突き当てて 12 本で締めれば、動力は
    #     出力軸 -> ハブ -> テーブル板 -> 環 -> 星車の小歯車
    # と 1 本につながる（旋回軸受リングは止まっている側なので、環の内穴は
    # その外へ逃がしたまま）
    groups[GEAR_TABLE] = [
        parts.tube(gt["root_r_table"], gt["bore_r"], gt["body_h"],
                   base_z=gt["base_z"]),
    ]
    groups[GEAR_TABLE] += _gear_teeth(
        gt["pitch_r_table"], gt["teeth_table"], gt["addendum"], gt["dedendum"],
        gt["backlash"], gt["pressure_rad"], _gear_phase(gt, GEAR_TABLE),
        gt["face"], gt["base_z"], gt["tooth_bite"])

    # テーブル板の下面へ締めるボルト。頭は環の下面から出る。下から見上げた
    # ときに「板へ締まっている環」だと読めるのはこの列だけなので、歯車と
    # 同じ群に入れて一緒に回す
    for i in range(gt["bolts"]):
        th = 2.0 * np.pi * i / gt["bolts"]
        groups[GEAR_TABLE].append(
            parts.cylinder(gt["bolt_d"] / 2.0, gt["bolt_h"],
                           base_z=gt["base_z"] - gt["bolt_h"],
                           resolution=parts.RES_COARSE,
                           matrix=parts.transform_matrix(
                               translate=(gt["bolt_r"] * np.cos(th),
                                          gt["bolt_r"] * np.sin(th), 0.0))))

    # 確定した高さをここで出す。**layout() では出さない。** 受け渡し系が
    # 合流する前は控えしか作れず、実際に組む値と食い違う
    _publish(lay, "drv_gear_z", (gt["base_z"], gt["base_z"] + gt["face"]))
    _publish(lay, "drv_gear_face", gt["face"])
    _publish(lay, "drv_gear_body_z", (gt["base_z"], gt["body_top"]))

    # --- 誘導ギヤモータ ---------------------------------------------------
    gm = parts.gearmotor(
        motor_d=lay["drv_motor_d"], motor_len=lay["drv_motor_len"],
        gearhead_size=lay["drv_gearhead_size"], gearhead_len=lay["drv_gearhead_len"],
        shaft_d=lay["drv_shaft_d"], shaft_len=lay["drv_shaft_len"],
        fins=lay["drv_fins"], terminal_size=lay["drv_terminal_size"],
        base_size=lay["drv_motor_base_size"],
        axis="x", origin=lay["drv_motor_origin"], matrix=yaw)
    groups["drv_motor"] += [gm["motor"], gm["fins"], gm["gearhead"],
                            gm["terminal_box"], gm["base"]]
    groups["drv_shaft"].append(gm["shaft"])

    # --- 露出した入力軸と、原点検出の取り出し ------------------------------
    # 入力ボスの端面から継手までを軸のまま見せる。**この機械で停留中も
    # 回っているのはここだけ。** その途中からかさ歯車 1 対で直角に取り出し、
    # 縦軸へ載せ替える（1:1）。載せ替えた先のドグ板は z 軸まわりに回るので、
    # 自転の仕掛けに乗る（水平軸のままでは回せない）
    sx0, sx1 = lay["drv_input_shaft_x"]
    input_z = lay["drv_input_z"]
    groups["drv_shaft"].append(
        parts.horizontal_cylinder(lay["drv_input_shaft_d"] / 2.0, sx1 - sx0,
                                  axis="x", matrix=yaw,
                                  center=((sx0 + sx1) / 2.0, 0.0, input_z)))

    tx = lay["drv_takeoff_x"]
    g_r0, g_r1 = lay["drv_takeoff_gear_r"]
    # 自転の中心は読み込み時に組んである（scene が値ごと写すので、あとから
    # 入れ替える道が無い）。**実際に置いた所と食い違っていないかをここで見る。**
    # 食い違うと、ドグ板だけが別の場所を中心にして宙を飛ぶ
    want = SPIN_CENTERS.get(HOME_SHAFT)
    have = _to_world(lay, tx, 0.0)
    if want is None or max(abs(a - b) for a, b in zip(want, have)) > 1e-6:
        print("[asm_drive] 原点取り出しの自転の中心が置いた所と合わない: "
              "SPIN_CENTERS %r / 実際 (%.4f, %.4f)。ドグ板が別の軸で回る"
              % (want, have[0], have[1]), file=sys.stderr)
    # 架台の上桁はドグ板と同じ高さを横切る。板の外縁が桁の内側の面を越えたら
    # 突き抜けるので、ここで止めずに出す（一度これで刺さっている）
    rail_in = float(lay["frame_span"]) / 2.0 - float(lay["frame_pipe"]) / 2.0
    edge = -(tx - lay["drv_dog_disc"][0])
    if edge > rail_in:
        print("[asm_drive] ドグ板が架台の上桁に入る: 外縁 %.2f / 桁の内側の面 "
              "%.2f（%.2f mm 食い込み）" % (edge, rail_in, edge - rail_in),
              file=sys.stderr)
    # かさ歯車は 2 つとも頂点が軸の交点に来る。45deg の円錐なので、頂点から
    # の距離がそのまま半径になる。水平側は -x（モータ側）へ、縦側は +z へ
    # 開く。この 2 つの円錐は (-1, 0, 1) の向きで接する
    groups["drv_shaft"].append(
        parts.cone_frustum(g_r0, g_r1, g_r1 - g_r0, base_z=g_r0,
                           resolution=parts.RES_COARSE,
                           matrix=yaw @ parts.transform_matrix(
                               translate=(tx, 0.0, input_z), rot_y_deg=-90.0)))

    ts_r = lay["drv_takeoff_shaft_d"] / 2.0
    h_ro, h_z0, h_z1 = lay["drv_takeoff_housing"]
    dog_r, dog_t, dog_z, dog_chord = lay["drv_dog_disc"]
    on_axis = parts.transform_matrix(translate=(tx, 0.0, 0.0))
    # 回る側（縦のかさ歯車・縦軸・ドグ板）。**この群だけが停留中も動く。**
    # ドグ板は半月。弦を軸の後ろへ寄せて、軸まわりに肉を残す
    th0 = float(np.arccos(np.clip(dog_chord / dog_r, -1.0, 1.0)))
    th = np.linspace(-th0, th0, 28)                   # 反時計回り（弧 -> 弦）
    dog_face = np.column_stack([dog_r * np.cos(th) + tx, dog_r * np.sin(th)])
    groups[HOME_SHAFT] = [
        parts.cone_frustum(g_r0, g_r1, g_r1 - g_r0, base_z=input_z + g_r0,
                           resolution=parts.RES_COARSE, matrix=yaw @ on_axis),
        parts.cylinder(ts_r, dog_z - (input_z + g_r1),
                       base_z=input_z + g_r1, matrix=yaw @ on_axis),
        parts.extrude_polygon(dog_face, dog_t, base_z=dog_z, matrix=yaw),
    ]
    # 止まっている側（軸受箱・ステー・近接センサ）。センサは上から下向きに
    # 構え、ドグ板の上面を見る。半月なので 1 回転で 1 回だけ金属が前を通る
    groups["drv_cast"].append(
        parts.tube(h_ro, ts_r, h_z1 - h_z0, base_z=h_z0, matrix=yaw @ on_axis))
    sen_d, sen_len, sen_z0, sen_at = lay["drv_sensor"]
    stay, stay_at, stay_top = lay["drv_sensor_stay"]
    groups["drv_mount"] += [
        # 上蓋に立てる柱
        parts.box((stay, stay, stay_top - h_z0), matrix=yaw,
                  center=(tx, stay_at, (h_z0 + stay_top) / 2.0)),
        # 板の上へ差し渡す腕。先端にセンサをぶら下げる
        parts.box((stay, stay_at - sen_at + stay, stay), matrix=yaw,
                  center=(tx, (stay_at + sen_at) / 2.0, stay_top - stay / 2.0)),
    ]
    groups["drv_sensor"] = [
        parts.cylinder(sen_d / 2.0, sen_len, base_z=sen_z0,
                       resolution=parts.RES_COARSE,
                       matrix=yaw @ parts.transform_matrix(
                           translate=(tx, sen_at, 0.0))),
    ]

    # --- カップリングと安全カバー -----------------------------------------
    groups["drv_shaft"].append(
        parts.coupling(diameter=lay["drv_coupling_d"], length=lay["drv_coupling_len"],
                       axis="x", matrix=yaw,
                       center=(lay["drv_coupling_x"], 0.0, lay["drv_input_z"])))

    # 継手のカバーは中実の箱にしない。中実だと継手もかさ歯車も丸ごと飲まれて、
    # 駆動列が絵から消える（覆いが完全に隠している、というのはこれ）。
    # 両側面に点検窓を開けて桟を渡し、上蓋には縦軸の逃げ穴を開ける。
    # x の両端は開けたまま（ギヤヘッドの前面とユニットの据付段が塞ぐ）
    g_len, g_side, g_h = lay["drv_guard_size"]
    g_cx, _g_cy, g_cz = lay["drv_guard_center"]
    g_t = lay["drv_guard_panel_t"]
    win_w, win_h, win_bars, win_bar_w = lay["drv_guard_window"]
    for sy in (-1.0, 1.0):
        groups["drv_guard"] += _plate_panels(
            "y", (g_cx, sy * (g_side / 2.0 - g_t / 2.0), g_cz), g_len, g_h, g_t,
            window=(win_w, win_h), window_at=(0.0, (input_z - g_cz)),
            bars=win_bars, bar_w=win_bar_w, matrix=yaw)
    lid_len, lid_side, lid_t, lid_z = lay["drv_guard_lid"]
    groups["drv_guard"] += _plate_panels(
        "z", (g_cx, 0.0, lid_z), lid_len, lid_side, lid_t,
        window=(h_ro * 1.6, h_ro * 1.6), window_at=(tx - g_cx, 0.0), matrix=yaw)

    # --- モータ取付台 -----------------------------------------------------
    # 梁（脚に渡す）＋棚板（梁の上）＋吊り柱（棚板から天板の下面まで）。
    # この 3 つでモータのベースを受ける。どれか欠けるとモータが宙に浮く
    beam_pipe = lay["drv_beam_pipe"]
    groups["drv_mount"].append(
        parts.box((beam_pipe, lay["drv_beam_span"], beam_pipe), matrix=yaw,
                  center=(lay["drv_beam_x"], 0.0, lay["drv_beam_top"] - beam_pipe / 2.0)))

    sx0, sx1 = lay["drv_shelf_x"]
    shelf_t = lay["drv_shelf_t"]
    groups["drv_mount"].append(
        parts.box((sx1 - sx0, lay["drv_shelf_half_w"] * 2.0, shelf_t), matrix=yaw,
                  center=((sx0 + sx1) / 2.0, 0.0, lay["drv_shelf_top"] - shelf_t / 2.0)))

    post = lay["drv_post_size"]
    post_h = lay["drv_post_top"] - lay["drv_shelf_top"]
    for sy in (-1.0, 1.0):
        groups["drv_mount"].append(
            parts.box((post, post, post_h), matrix=yaw,
                      center=(lay["drv_post_x"], sy * lay["drv_post_y"],
                              lay["drv_shelf_top"] + post_h / 2.0)))

    # --- モータの板金カバー -----------------------------------------------
    # 天井・両側面・外端の 4 枚。底は棚板が塞ぎ、機械の中心側は開けたまま
    # にして継手を残す。外端には点検蓋と摘みを付けて、開く所が分かるようにする
    px0, px1 = lay["drv_panel_x"]
    pz0, pz1 = lay["drv_panel_z"]
    pt = lay["drv_panel_t"]
    phw = lay["drv_panel_half_w"]
    p_len = px1 - px0
    p_hgt = pz1 - pz0

    win_w, win_h = lay["drv_panel_lid"]
    n_bar, bar_w = lay["drv_panel_bars"]
    lid_z = lay["drv_input_z"]

    # 天井
    groups["drv_panel"].append(
        parts.box((p_len, phw * 2.0, pt), matrix=yaw,
                  center=((px0 + px1) / 2.0, 0.0, pz1 - pt / 2.0)))
    # 側面。ここにも窓を開ける。**中がモータだけでも、閉じた箱と開いた箱では
    # 機械の見え方が違う。** 桟を渡して手が入らないようにする
    for sy in (-1.0, 1.0):
        groups["drv_panel"] += _plate_panels(
            "y", ((px0 + px1) / 2.0, sy * (phw - pt / 2.0), (pz0 + pz1) / 2.0),
            p_len, p_hgt, pt,
            window=(p_len * win_w, p_hgt * win_h),
            window_at=(0.0, lid_z - (pz0 + pz1) / 2.0),
            bars=int(n_bar), bar_w=bar_w, matrix=yaw)

    # 外端。**ここは実際に開口して点検窓にする。** 前は同じ大きさの板を外側へ
    # 貼っただけの「蓋」で、板金は塞がったままだった。開口の中心は軸の高さ、
    # 桟を渡して手が入らないようにする
    groups["drv_panel"] += _plate_panels(
        "x", (px0 + pt / 2.0, 0.0, (pz0 + pz1) / 2.0), phw * 2.0, p_hgt, pt,
        window=(phw * 2.0 * win_w, p_hgt * win_h),
        window_at=(0.0, lid_z - (pz0 + pz1) / 2.0),
        bars=int(n_bar), bar_w=bar_w, matrix=yaw)

    # 窓枠の摘み。開口の左右の帯に付く
    knob_d, knob_len, knob_at = lay["drv_panel_knob"]
    for sy in (-1.0, 1.0):
        groups["drv_mount"].append(
            parts.horizontal_cylinder(
                knob_d / 2.0, knob_len, axis="x", matrix=yaw,
                center=(px0 - knob_len / 2.0,
                        sy * phw * win_w * (1.0 + knob_at), lid_z),
                resolution=parts.RES_COARSE))

    return {k: parts.merge(v) for k, v in groups.items() if v}


def _to_world(lay: dict, x: float, y: float) -> tuple:
    """駆動系の局所 (x, y) を世界へ。build() が掛けるのと同じ回転。"""
    th = np.radians(float(lay["drv_yaw_deg"]))
    c, s = np.cos(th), np.sin(th)
    return (float(c * x - s * y), float(s * x + c * y))


def _print_frame_follow(lay: dict) -> None:
    """架台に追従できているかを数で出す。

    梁も板金カバーも長さを `frame_span` から出しているので、架台が振れたら
    同じだけ振れていなければならない。追従が漏れると、脚から外れた梁が
    腰板を突き抜ける（振る前の向きに取り残される）。ここはその見張り。
    """
    phase = float(lay.get("frame_leg_phase_deg", 0.0))
    print("架台の振り       %+.1f deg -> 駆動の世界角 %.1f deg（局所 %.1f）"
          % (phase, float(lay["drv_angle_deg"]), DRIVE_FACE_DEG))
    print("  脚の世界角     %s"
          % " / ".join("%.1f" % v for v in lay["frame_leg_deg_all"]))

    # 梁の端は脚 2 本の内側の面に突き当たっているか。芯までの距離が
    # 角パイプの半分ちょうどなら、面と面が合っている
    half = float(lay["drv_beam_span"]) / 2.0
    pipe_half = float(lay["frame_pipe"]) / 2.0
    for sy in (-1.0, 1.0):
        ex, ey = _to_world(lay, lay["drv_beam_x"], sy * half)
        d = [(float(np.hypot(ex - lx, ey - ly)), (lx, ly))
             for lx, ly in lay["frame_leg_xy"]]
        gap, leg = min(d)
        flag = "" if abs(gap - pipe_half) < 1e-6 else "  <-- 脚から外れている"
        print("  梁の端         (%8.2f, %8.2f) -> 脚 (%8.2f, %8.2f) まで %.2f "
              "（角パイプの半分 %.2f）%s" % (ex, ey, leg[0], leg[1], gap,
                                            pipe_half, flag))

    # 板金カバーの内端。架台の外側の面より内側に入っていると上桁を飲み込む
    face = float(lay["drv_frame_face"])
    inner = -float(lay["drv_panel_x"][1])
    flag = "" if inner >= face else "  <-- 架台の中へ入っている"
    print("  カバーの内端   %.2f / 架台の外側の面 %.2f -> 外へ %+.2f%s"
          % (inner, face, inner - face, flag))
    # 上桁は脚の芯々の線に乗る。カバーの内端がその外側の面より外なら当たらない
    rail_out = float(lay["frame_span"]) / 2.0 + float(lay["frame_pipe"]) / 2.0
    print("  上桁の外側の面 %.2f -> カバーとのすき間 %+.2f" % (rail_out, inner - rail_out))


# --------------------------------------------------------------------------
# 確認用。既存の機械に自分のメッシュを足して何方向かから焼く
# --------------------------------------------------------------------------
def _check(out_dir, size=(800, 600)) -> None:
    import pyvista as pv
    import cameras
    import scene

    params = scene.load_params()
    lay = scene.derive_layout(params)
    # scene 側がこのモジュールを読み込んでいれば寸法もメッシュも既に合流済み。
    # まだ読み込んでいない（書きかけで飛ばされた）ときだけ自分で足す
    if "drv_input_z" not in lay:
        lay.update(layout(params, lay))

    # 機械 1 台は scene 側の合成ヘルパに任せる。build_static と build_carousel を
    # 自分で dict.update すると、どちらも返す "steel" 群が上書きで消えて、
    # 固定ノズル・ガントリ・旋回軸受リング・中心柱が絵から落ちる
    mine = build(params, lay)
    meshes = scene.merge_groups(scene.build_static(params, lay, with_floor=True),
                                scene.build_carousel(params, lay))
    # scene がこのモジュールを読み込めていれば drv_* は既に入っている。
    # 飛ばされていたときだけ自分のぶんを足す
    for name, mesh in mine.items():
        meshes.setdefault(name, mesh)
    # 画角は機械の実測外形に合わせる。組んだものをそのまま測らせる
    scene.ensure_extent(lay, meshes, params)

    # 描く順と材質も同じ扱い。scene 側に既に載っていれば足さない
    order = list(scene.DRAW_ORDER)
    known = {tuple(row) for row in order}
    order += [row for row in DRAW_ORDER if tuple(row) not in known]
    material = dict(scene.MATERIAL)
    material.update(MATERIALS)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 高さの鎖が通っているかを数値でも出す。絵と突き合わせる用
    print("deck             %.1f .. %.1f" % (lay["deck_top"] - lay["deck_t"],
                                             lay["deck_top"]))
    print("boss             %.1f .. %.1f" % (lay["drv_boss_base"],
                                             lay["drv_flange_top"]))
    print("out shaft        %.1f .. %.1f  (d %.1f)"
          % (lay["drv_out_shaft_z"][0], lay["drv_out_shaft_z"][1],
             lay["drv_out_shaft_d"]))
    print("  bearing ring   %.1f .. %.1f  (ri %.1f)"
          % (lay["deck_top"], lay["table_base"], lay["bearing_ri"]))
    print("  hub            %.1f .. %.1f  (r %.1f)"
          % (lay["drv_out_hub_z"], lay["table_base"], lay["drv_out_hub_r"]))
    gt = _gear_train(lay, strict=True)
    print("歯車 m %.3f  歯数 %d / %d  歯幅 %.1f  z %.1f .. %.1f"
          % (gt["module"], gt["teeth_table"], gt["teeth_star"], gt["face"],
             gt["base_z"], gt["base_z"] + gt["face"]))
    print("  環の胴          z %.1f .. %.1f （テーブル板の下面 %.1f まで / 差 %+.3f）"
          % (gt["base_z"], gt["body_top"], lay["table_base"],
             gt["body_top"] - lay["table_base"]))
    for name, (cx, cy) in gt["centers"].items():
        is_table = (name == GEAR_TABLE)
        pr = gt["pitch_r_table"] if is_table else gt["pitch_r_star"]
        print("  %-14s 中心 (%8.2f, %8.2f)  ピッチ %.2f / 歯先 %.2f / 歯底 %.2f"
              % (name, cx, cy, pr, pr + gt["addendum"], pr - gt["dedendum"]))
    print("  ピッチ円の接し   %.4f + %.4f = %.4f  (中心距離 %.4f / 差 %.2e)"
          % (gt["pitch_r_table"], gt["pitch_r_star"],
             gt["pitch_r_table"] + gt["pitch_r_star"], gt["center_r"],
             abs(gt["pitch_r_table"] + gt["pitch_r_star"] - gt["center_r"])))
    print("  速比             %.6f （ピッチ円の比）"
          % (gt["pitch_r_table"] / gt["pitch_r_star"]))
    for name, gap in _mesh_clearance(gt).items():
        print("  歯の噛み合い     %-14s 最小すき間 %+.3f mm（両方に歯を刻んだ場合）"
              % (name, gap))
    # 小歯車を組むのは受け渡し系。向こうが歯を刻んでいないと、こちらの歯先が
    # 相手の歯先円へそのまま入る。入る量を出しておく
    there_tip = lay.get("trf_gear_tip_d")
    if there_tip is not None:
        bite = ((gt["pitch_r_table"] + gt["addendum"]) + float(there_tip) / 2.0
                - gt["center_r"])
        print("  歯先円の重なり   %+.2f mm（歯が互い違いに入る量。相手が"
              "平らな円板なら、そのまま食い込みになる）" % bite)
    print("  据わり           天板上面 %.1f +%.2f -> 歯 %.1f..%.1f / 胴 "
          "%.1f..%.1f -> テーブル板下面 %.1f (胴と板の差 %+.3f)"
          % (lay["deck_top"], gt["base_z"] - lay["deck_top"], gt["base_z"],
             gt["base_z"] + gt["face"], gt["base_z"], gt["body_top"],
             lay["table_base"], gt["body_top"] - lay["table_base"]))
    print("  動力の流れ       出力軸 %.1f..%.1f -> ハブ %.1f..%.1f (r %.2f) -> "
          "テーブル板 %.1f -> 環の胴 -> 歯 -> 小歯車"
          % (lay["drv_out_shaft_z"][0], lay["drv_out_shaft_z"][1],
             lay["drv_out_hub_z"], lay["table_base"], lay["drv_out_hub_r"],
             lay["table_base"]))
    print("  内穴 r %.2f / 旋回軸受リング外半径 %.2f -> すき間 %+.2f"
          % (gt["bore_r"], lay["bearing_ro"], gt["bore_r"] - lay["bearing_ro"]))
    print("  歯先 r %.2f / テーブル板 外半径 %.2f -> 板の内側に %+.2f"
          % (gt["pitch_r_table"] + gt["addendum"], lay["plate_r"],
             lay["plate_r"] - gt["pitch_r_table"] - gt["addendum"]))
    for name, (cx, cy) in gt["centers"].items():
        is_table = (name == GEAR_TABLE)
        tip = ((gt["pitch_r_table"] if is_table else gt["pitch_r_star"])
               + gt["addendum"])
        gap, leg = _leg_gap(lay, (cx, cy), tip, z0=gt["base_z"])
        who = "" if is_table else "（受け渡し系が組む）"
        if not np.isfinite(gap):
            print("  架台の脚まで     %-10s 脚の上端 %.1f より上なので当たらない%s"
                  % (name, lay["deck_top"] - lay["deck_t"], who))
        else:
            flag = "" if gap >= 0.0 else "  <-- 食い込み"
            print("  架台の脚まで     %-10s %+8.2f mm  (脚 %.1f, %.1f)%s%s"
                  % (name, gap, leg[0], leg[1], flag, who))
    print("  締めボルト       %d 本 phi%.1f  r %.2f  頭 %.2f mm（天板とのすき間 %.2f）"
          % (gt["bolts"], gt["bolt_d"], gt["bolt_r"], gt["bolt_h"],
             gt["deck_gap"]))
    print("box top / base   %.1f / %.1f" % (lay["drv_box_top"], lay["drv_base_z"]))
    print("input axis z     %.1f  (face x %.1f)"
          % (lay["drv_input_z"], lay["drv_input_face_x"]))
    print("motor origin     (%.1f, %.1f, %.1f)  軸の高さの差 %+.3f"
          % (lay["drv_motor_origin"] + (lay["drv_motor_origin"][2]
                                        - lay["drv_input_z"],)))
    print("coupling         x %.1f  長さ %.1f  径 %.1f  "
          "モータ軸の差し込み %.2f mm（長さの %.2f）"
          % (lay["drv_coupling_x"], lay["drv_coupling_len"], lay["drv_coupling_d"],
             (lay["drv_coupling_x"] + lay["drv_coupling_len"] / 2.0)
             - (lay["drv_motor_origin"][0] + lay["drv_shaft_len"]),
             PROPORTION["coupling_engage"]))
    # 原点検出の取り出し。**停留中に画面で動くのはここだけ**なので、
    # 置き場所と逃げは数でも押さえておく
    tx = lay["drv_takeoff_x"]
    sx0, sx1 = lay["drv_input_shaft_x"]
    g_r0, g_r1 = lay["drv_takeoff_gear_r"]
    dog_r, dog_t, dog_z, dog_chord = lay["drv_dog_disc"]
    sen_d, sen_len, sen_z0, sen_at = lay["drv_sensor"]
    stay, stay_at, stay_top = lay["drv_sensor_stay"]
    _h_ro, h_z0, h_z1 = lay["drv_takeoff_housing"]
    print("露出した入力軸   x %.1f .. %.1f （長さ %.1f / 径 %.1f）"
          % (sx0, sx1, sx1 - sx0, lay["drv_input_shaft_d"]))
    print("  かさ歯車       頂点 x %.2f  ピッチ半径 %.1f（小端 %.1f）-> "
          "軸に載る範囲 x %.1f .. %.1f（-x へ開く）"
          % (tx, g_r1, g_r0, tx - g_r1, tx - g_r0))
    print("  縦軸           x %.2f (世界 %.2f, %.2f)  径 %.1f  "
          "軸受箱 z %.1f..%.1f"
          % (tx, _to_world(lay, tx, 0.0)[0], _to_world(lay, tx, 0.0)[1],
             lay["drv_takeoff_shaft_d"], h_z0, h_z1))
    print("  ドグ板         半月 r %.1f  t %.1f  z %.1f..%.1f  弦 %.1f "
          "（軸の後ろに残す肉）"
          % (dog_r, dog_t, dog_z, dog_z + dog_t, abs(dog_chord)))
    print("  近接センサ     phi%.1f 長さ %.1f  検出面 z %.1f (半径 %.1f) -> "
          "ドグ板の上面とのすき間 %+.2f"
          % (sen_d, sen_len, sen_z0, sen_at, sen_z0 - (dog_z + dog_t)))
    # 上桁は脚の芯々の線に乗り、ドグ板と同じ高さを横切る。**ここは実際に
    # 一度刺さった。** 板の外縁が桁の内側の面より外へ出ていないかを見る
    rail_in = lay["frame_span"] / 2.0 - lay["frame_pipe"] / 2.0
    edge = -(tx - dog_r)
    flag = "" if edge <= rail_in else "  <-- 上桁を突き抜ける"
    print("  逃げ           ドグ板の外縁 %.1f / 上桁の内側の面 %.1f -> %+.2f%s"
          % (edge, rail_in, rail_in - edge, flag))
    print("                 ドグ板の内縁 x %.1f / 本体箱の壁 %.1f -> %+.2f  ／ "
          "ステーの頭 %.1f / 天板の下面 %.1f -> %+.2f"
          % (tx + dog_r, -lay["drv_box"][0] / 2.0,
             -lay["drv_box"][0] / 2.0 - (tx + dog_r),
             stay_top, lay["deck_top"] - lay["deck_t"],
             (lay["deck_top"] - lay["deck_t"]) - stay_top))
    print("                 ステー +y %.1f / 上蓋の端 %.1f -> %+.2f"
          % (stay_at + stay / 2.0, lay["drv_guard_lid"][1] / 2.0,
             lay["drv_guard_lid"][1] / 2.0 - (stay_at + stay / 2.0)))
    print("  自転           %s を psi で回す（1:1 / 1 タクト 1 回転）。中心 "
          "(%.2f, %.2f)" % (HOME_SHAFT, SPIN_CENTERS[HOME_SHAFT][0],
                            SPIN_CENTERS[HOME_SHAFT][1]))
    print("motor base z     %.1f .. %.1f  x %.1f .. %.1f"
          % (lay["drv_motor_base_bottom"], lay["drv_motor_base_top"],
             lay["drv_motor_base_x"][0], lay["drv_motor_base_x"][1]))
    print("shelf top        %.1f  x %.1f .. %.1f"
          % (lay["drv_shelf_top"], lay["drv_shelf_x"][0], lay["drv_shelf_x"][1]))
    print("panel            x %.1f .. %.1f  y +-%.1f  z %.1f .. %.1f"
          % (lay["drv_panel_x"][0], lay["drv_panel_x"][1], lay["drv_panel_half_w"],
             lay["drv_panel_z"][0], lay["drv_panel_z"][1]))
    _print_frame_follow(lay)

    # 真横は駆動軸に直交する向きから見る。軸に沿って見ると軸の高さが読めない。
    # 逆側（軸 -90 度）はコンベアが手前に来て駆動系が隠れる
    drive_deg = float(lay["drv_angle_deg"])
    side_az = drive_deg + 90.0
    # 画面に入れる大きさは機械の外接円の直径（実測）の倍率で指定する。
    # 基準を 2 つ渡すと cameras 側が止まるので、既定と同じキーだけを使う
    cams = {
        "iso": cameras.resolve("iso", lay, params, view_span_x_extent_d=0.78,
                               focus_bottle_ratio=-1.0),
        # 既定の斜め上は駆動系が機械の裏へ回る。張り出す側からも 1 枚焼く
        "iso_drive": cameras.resolve("iso", lay, params,
                                     azimuth_deg=drive_deg + 38.0,
                                     elevation_deg=26.0,
                                     view_span_x_extent_d=0.86,
                                     focus_bottle_ratio=-1.0),
        "side": cameras.resolve("iso", lay, params, azimuth_deg=side_az,
                                elevation_deg=0.0, view_span_x_extent_d=0.78,
                                focus_bottle_ratio=-1.0),
        "top": cameras.resolve("top", lay, params),
    }
    # 真横は駆動軸に直交する向きになるが、その向きには架台の腰板が貼ってあって
    # 駆動系が丸ごと隠れる。腰板を外した同じ画角も 1 枚焼いて、軸の高さと
    # 取付台の渡り方を読めるようにする
    cams["side_frame"] = dict(cams["side"], _only=("drv_", "frame"))
    # 駆動列の寄り。軸の高さが合っているかはこの 1 枚がいちばん効く
    close = cameras.resolve("iso", lay, params, azimuth_deg=side_az, elevation_deg=6.0,
                            view_span_x_extent_d=0.34, focus_bottle_ratio=-1.0)
    # 注視点も局所座標なので、部品と同じ角だけ回してから渡す
    span = (lay["drv_input_face_x"] + lay["drv_motor_origin"][0]) / 2.0
    yaw = np.radians(lay["drv_yaw_deg"])
    focal = (span * np.cos(yaw), span * np.sin(yaw), lay["drv_input_z"])
    cams["drive"] = _refocus(close, focal)
    # 架台の脚が手前に来て継手を隠すので、駆動系だけの絵も 1 枚焼く
    cams["solo"] = dict(cams["drive"], _solo=True)

    # 見上げ。駆動系は天板の下にあるので、これを焼かないと何も確かめられない。
    # 床板より下へ回り込むと床に遮られるので、仰角はほどほどに留める。
    #
    # 方位は駆動の向きから振る。**振る側は液の供給系と反対にする。** タンクは
    # 供給側の角に立っていて、そちらへ振ると架台の中がタンクの胴で丸ごと
    # 隠れる。振り幅は脚の間隔の半分より小さく取り、脚を正面に置かない
    under_az = drive_deg + 35.0
    under = cameras.resolve("iso", lay, params, azimuth_deg=under_az,
                            elevation_deg=-26.0, view_span_x_extent_d=0.62,
                            focus_bottle_ratio=-1.0)
    under = _refocus(under, (0.0, 0.0, lay["drv_box_top"]))
    cams["under"] = _above_floor(under, lay["plate_t"] * 8.0)
    cams["under_solo"] = dict(cams["under"], _solo=True)

    # 取付台の寄り。梁が脚に渡っているか、板金カバーが腰板と上桁を外して
    # いるかはここで見る。架台の外から、モータの張り出す面をほぼ正面に置く
    mount = cameras.resolve("iso", lay, params, azimuth_deg=drive_deg + 22.0,
                            elevation_deg=-14.0, view_span_x_extent_d=0.40,
                            focus_bottle_ratio=-1.0)
    mount_r = lay["drv_frame_face"]           # 注視点は架台の外側の面の上
    mount = _refocus(mount, (mount_r * np.cos(np.radians(drive_deg)),
                             mount_r * np.sin(np.radians(drive_deg)),
                             lay["drv_shelf_top"]))
    cams["mount"] = _above_floor(mount, lay["plate_t"] * 8.0)
    # 液の供給系とコンベアが手前に立って取付台を隠すので、架台と駆動だけの
    # 絵も 1 枚。梁の端がどの脚に渡っているかはこちらのほうが読める
    cams["mount_frame"] = dict(cams["mount"], _only=("drv_", "frame"))
    # 真下から。梁・棚板・吊り柱の渡り方は見上げないと分からない
    below = cameras.resolve("iso", lay, params, azimuth_deg=drive_deg + 8.0,
                            elevation_deg=-52.0, view_span_x_extent_d=0.52,
                            focus_bottle_ratio=-1.0)
    below = _refocus(below, (mount_r * np.cos(np.radians(drive_deg)) * 0.6,
                             mount_r * np.sin(np.radians(drive_deg)) * 0.6,
                             lay["drv_beam_top"]))
    cams["mount_below"] = dict(_above_floor(below, lay["plate_t"] * 2.0),
                               _only=("drv_", "frame"))

    # 天板の下の歯車。3 枚が同じ高さに並び、ピッチ円が接しているかは
    # この 1 枚で決まる。
    #
    # 横から撮れる角度は狭い。仰角を下げて見上げると、架台の上の桁（天板の
    # 下面から角パイプ 1.2 本ぶん下）が視線を切る。真下からはインデックス
    # ユニットの本体箱に隠れる。残るのは歯車の高さからほぼ水平に覗く
    # 1 通りだけで、仰角は -2deg 前後しか使えない。方位は星車 2 台の
    # あいだ（290deg）。ここならモータの板金カバーもコンベアも外れる
    gear_z = gt["base_z"] + gt["face"] / 2.0
    side_gear = cameras.resolve("iso", lay, params, azimuth_deg=290.0,
                                elevation_deg=-2.0, view_span_x_extent_d=0.24,
                                focus_bottle_ratio=-1.0)
    side_gear = _refocus(side_gear, (0.0, -lay["pitch_r"] * 0.75, gear_z))
    # ほぼ水平なので、手前に立つものは何でも視線を塞ぐ。動力の通り道と
    # 架台だけに絞る（受け渡し系の床置きの支柱まで入れると歯車が消える）
    cams["gear_side"] = dict(side_gear, _only=("drv_", "trf_star_", "frame"))
    cams["gear_side_all"] = side_gear

    # 噛み合いは歯車の面に正対して見るのがいちばん分かる。**見上げる側から
    # 撮る。** 見下ろすとスターホイールの円盤とボトルが真上に乗っていて、
    # 噛み合っている所がまるごと隠れる。下から見れば歯車がいちばん手前に
    # 来るので何も遮らない。
    #
    # 小歯車は星車の軸に乗る部品で、受け渡し系が自分の群（trf_star_*）に
    # 入れている。solo にすると噛み合う相手が消えるので、群で絞る
    gear_only = ("drv_gear_", "trf_star_in", "trf_star_out")
    top_gear = cameras.resolve("iso", lay, params, azimuth_deg=270.0,
                               elevation_deg=-75.0, view_span_x_extent_d=0.44,
                               focus_bottle_ratio=-1.0)
    cams["gear_top"] = dict(
        _refocus(top_gear, (0.0, -gt["center_r"] * 0.45, gear_z)),
        _only=gear_only)

    # 噛み合い点の寄り。ピッチ点は星車の中心をテーブル側のピッチ円まで
    # 縮めた所（2 つのピッチ円が接する点そのもの）。排出側で見る
    out_c = np.asarray(gt["centers"][GEAR_STARS["discharge"]], dtype=float)
    pitch_pt = out_c * gt["pitch_r_table"] / gt["center_r"]
    zoom = cameras.resolve("iso", lay, params,
                           azimuth_deg=float(lay["discharge_deg"]),
                           elevation_deg=-72.0, view_span_x_extent_d=0.10,
                           focus_bottle_ratio=-1.0)
    cams["gear_mesh"] = dict(
        _refocus(zoom, (float(pitch_pt[0]), float(pitch_pt[1]), gear_z)),
        _only=gear_only)

    # 出力軸とテーブルの繋がりの寄り。天板・旋回軸受・テーブル板の 3 段を
    # 横から覗く高さに注視点を置く
    joint_z = (lay["deck_top"] - lay["deck_t"] + lay["table_top"]) / 2.0
    joint = cameras.resolve("iso", lay, params, azimuth_deg=drive_deg - 60.0,
                            elevation_deg=-9.0, view_span_x_extent_d=0.30,
                            focus_bottle_ratio=-1.0)
    cams["joint"] = _above_floor(_refocus(joint, (0.0, 0.0, joint_z)),
                                 lay["plate_t"] * 8.0)
    cams["joint_solo"] = dict(cams["joint"], _solo=True)

    # 大歯車とハブの繋がりの寄り。**下から見上げる。** 上からだとテーブル板
    # に丸ごと隠れる。環の胴がテーブル板の下面に突き当たり、締めボルトの頭が
    # 環の下面に並んでいるのがこの 1 枚で読める
    ring_z = (gt["base_z"] + lay["table_base"]) / 2.0
    ring = cameras.resolve("iso", lay, params, azimuth_deg=drive_deg - 40.0,
                           elevation_deg=-34.0, view_span_x_extent_d=0.22,
                           focus_bottle_ratio=-1.0)
    ring = _refocus(ring, (0.0, 0.0, ring_z))
    cams["ring_hub"] = dict(_above_floor(ring, lay["plate_t"] * 8.0),
                            _only=("drv_", "steel", "table"))

    # --- 駆動列を見せる道 -------------------------------------------------
    # 板金カバーは中を隠すためのものなので、**外す絵と窓から覗く絵の 2 通り**を
    # 用意する。外す側は板金だけを落とし、機構はそのまま残す
    open_cam = cameras.resolve("iso", lay, params, azimuth_deg=drive_deg + 50.0,
                               elevation_deg=-8.0, view_span_x_extent_d=0.24,
                               focus_bottle_ratio=-1.0)
    open_focal = _to_world(lay, (lay["drv_input_face_x"]
                                 + lay["drv_motor_origin"][0]) / 2.0, 0.0)
    open_cam = _refocus(open_cam, (open_focal[0], open_focal[1],
                                   lay["drv_input_z"]))
    cams["train_open"] = dict(_above_floor(open_cam, lay["plate_t"] * 4.0),
                              _only=("drv_", "frame"),
                              _skip=("drv_panel", "drv_guard"))
    cams["train_window"] = dict(_above_floor(open_cam, lay["plate_t"] * 4.0),
                                _only=("drv_", "frame"))

    # 原点検出の寄り。ドグ板・ドグ・近接センサ・かさ歯車をまとめて入れる。
    # **psi を振ったときに動くのはこの 1 枚の中だけ**なので、コマ送りの
    # 確認もこの画角で焼く。
    #
    # 方位は駆動の向きから 55deg 振る。真正面（駆動の向きそのもの）は自分の
    # 板金カバーが立ちはだかり、真横は架台の腰板が塞ぐ。カバーの角をかすめて
    # 腰板の切れ目から覗けるのはこの帯だけで、**機械を丸ごと組んだままでも
    # ドグ板が見える向きはここしか無い。**
    home_focal = _to_world(lay, lay["drv_takeoff_x"], 0.0)
    home_z = (lay["drv_input_z"] + lay["drv_sensor_stay"][2]) / 2.0
    home = cameras.resolve("iso", lay, params, azimuth_deg=drive_deg + 55.0,
                           elevation_deg=-6.0, view_span_x_extent_d=0.16,
                           focus_bottle_ratio=-1.0)
    home = _refocus(home, (home_focal[0], home_focal[1], home_z))
    cams["home"] = _above_floor(home, lay["plate_t"] * 4.0)
    # 手前の板金と腰板を落とした同じ画角。取り出しの形はこちらが読める
    cams["home_open"] = dict(cams["home"], _only=("drv_", "frame"),
                             _skip=("drv_panel",))

    _render(scene, pv, cams, size, out_dir, order, material, meshes, mine)

    # --- psi を振る -------------------------------------------------------
    # カム入力軸の角 psi を 0/90/180/270deg で焼く。**180〜360deg は停留**で、
    # テーブルも星車も止まっている。そのあいだも原点取り出しは回り続ける
    _psi_sweep(scene, pv, params, lay, size, out_dir, order, material,
               cams["home"])


def _render(scene, pv, cams, size, out_dir, order, material, meshes, mine,
            suffix="") -> None:
    """カメラの表を 1 枚ずつ焼く。`_only` で群を絞り、`_skip` で落とす。"""
    import cameras                                     # noqa: PLC0415

    for tag, cam in cams.items():
        pl = _new_plotter(scene, pv, size)
        rows = DRAW_ORDER if cam.get("_solo") else order
        only = cam.get("_only")
        skip = cam.get("_skip")
        for name, mat in rows:
            if only is not None and not name.startswith(only):
                continue
            if skip is not None and name.startswith(skip):
                continue
            mesh = (mine if cam.get("_solo") else meshes).get(name)
            if mesh is not None and mesh.n_points:
                pl.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True,
                            feature_angle=35.0, **material[mat])

        cameras.apply_resolved(pl, cam)
        path = out_dir / ("asm_drive_%s%s.png" % (tag, suffix))
        pl.show(screenshot=str(path))
        pl.close()
        print("焼いた: %s" % path)


def _psi_sweep(scene, pv, params, lay, size, out_dir, order, material,
               home_cam, angles=(0.0, 90.0, 180.0, 270.0)) -> None:
    """カム入力軸角 psi を振って焼く。**割出しと停留の違いを絵で見る用。**

    テーブルの姿勢（回る側）と自転する群は psi ごとに組み直す。止まっている
    側は 1 度組めば使い回せる。**停留（psi 180〜360deg）ではテーブルも星車も
    止まり、原点取り出しだけが回り続ける。** そこが読めなければ意味が無い
    ので、ドグの世界角も一緒に出す。
    """
    static = scene.build_static(params, lay, with_floor=True)
    for psi_deg in angles:
        psi = float(np.radians(psi_deg))
        table_rad = float(scene.table_angle_from_cam(params, psi))
        state = scene.demo_state(params, lay, table_angle_rad=table_rad,
                                 cam_angle_rad=psi)
        meshes = scene.merge_groups(static,
                                    scene.build_carousel(params, lay, table_rad,
                                                         state.bottle_present))
        for group, mat in scene.spin_matrices(params, lay, state).items():
            mesh = meshes.get(group)
            if mesh is not None:
                meshes[group] = parts.place(mesh, mat)
        dog = float(np.degrees(
            scene.spin_angles(params, lay, state).get(HOME_SHAFT, 0.0)))
        print("psi %5.1f deg -> テーブル %6.2f deg / ドグ %7.2f deg  (%s)"
              % (psi_deg, np.degrees(table_rad), dog % 360.0,
                 "割出し" if psi_deg < 180.0 else "停留"))
        _render(scene, pv, {"home": home_cam}, size, out_dir, order, material,
                meshes, meshes, suffix="_psi%03d" % int(psi_deg))


def _refocus(cam: dict, focal) -> dict:
    """視線の向きと距離はそのままに、注視点だけ差し替える。"""
    offset = [p0 - f0 for p0, f0 in zip(cam["position"], cam["focal_point"])]
    out = dict(cam)
    out["focal_point"] = tuple(float(v) for v in focal)
    out["position"] = tuple(float(f + o) for f, o in zip(focal, offset))
    return out


def _above_floor(cam: dict, floor_z: float) -> dict:
    """床板より下へ回り込んだ視点を、床の上へ起こす。

    見上げの絵は仰角を負にして作るが、そのまま下げると視点が床板（半径が
    機械の何倍もある円板）の裏へ潜り、画面が背景一色になる。注視点までの
    距離と方位はそのままに、高さだけ床の上へ持ち上げる。
    """
    px, py, pz = cam["position"]
    fx, fy, fz = cam["focal_point"]
    if pz >= floor_z:
        return cam
    dist = float(np.linalg.norm([px - fx, py - fy, pz - fz]))
    az = float(np.arctan2(py - fy, px - fx))
    dz = float(floor_z) - fz
    dh = float(np.sqrt(max(dist ** 2 - dz ** 2, 0.0)))
    out = dict(cam)
    out["position"] = (fx + dh * np.cos(az), fy + dh * np.sin(az), float(floor_z))
    return out


def _new_plotter(scene, pv, size):
    """描画器の下ごしらえ。scene 側にヘルパがあればそれに任せる。

    無いときは scene.render() と同じ設定（環境マップの前計算 64/32）で組む。
    ここだけ粗い値にすると、他のモジュールの確認画像と明るさが揃わない。
    """
    helper = getattr(scene, "new_plotter", None)
    if helper is not None:
        return helper(size)

    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background(scene.BACKGROUND)
    pl.set_environment_texture(scene.studio_cubemap(), is_srgb=True)
    try:
        pl.renderer.GetEnvMapPrefiltered().SetPrefilterMaxSamples(64)
        pl.renderer.GetEnvMapIrradiance().SetIrradianceSize(32)
    except AttributeError:
        pass
    return pl


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="駆動系の確認")
    ap.add_argument("--out", type=Path, default=Path("."), help="PNG の出力先")
    ap.add_argument("--size", type=int, nargs=2, default=(800, 600),
                    help="画の大きさ（SSAA なし）")
    a = ap.parse_args()
    _check(a.out, tuple(a.size))
