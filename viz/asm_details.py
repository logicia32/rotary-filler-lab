"""機械まわりの小物と液の供給系。

タンクとポンプ、そこからノズルまでの配管、液受け溝、制御箱、操作盤と
シグナルタワー、架台の板金、配線ダクト、銘板。1 つ 1 つは小さいが、
数と置き場所で実機らしさが決まる部分をまとめて持つ。
寸法は params.json の諸元を基準に、viz/ASSEMBLY_CONTRACT.md の座標に合わせる。

置き場所の約束（viz/ASSEMBLY_CONTRACT.md）
  * 割出しユニットとギヤモータの場所は lay["drv_angle_deg"]（実測 67.5 度）。
    天板の下はそこから半径 533 まで塞がっていて、露出した入力軸まで含めると
    世界角でおよそ 48〜87 度が埋まる。ここは配管も通せない。
  * 供給（lay["infeed_deg"]）と排出（lay["discharge_deg"]）はスターホイールと
    コンベアの場所。どちらのコンベアも接点（半径 450）から機外へ伸びるので、
    接点より手前——つまり 2 本のレーンに挟まれた側——は空いている。
  * ここで使うのは 2 本のレーンの中央の外（操作盤。作業者が供給と排出の両方を
    見渡せる位置）と、充填ステーションから +45 度の外（タンクとポンプ）、
    そこから充填までの天板の下（液配管の回り込み）と、架台の脚まわり。
  **世界角を直書きしない。** 操作盤は工程角（供給と排出）の中央から、タンクは
  充填ステーションからの振りで出す。排出を動かすと両方が付いてくる。

架台に付く小物は脚の世界角に追従する
  脚の向きは scene.derive_layout() が出す lay["frame_leg_deg_all"]（4 本ぶんの
  世界角）/ lay["frame_leg_xy"] / lay["frame_leg_phase_deg"] が唯一の表。アジャスタ座・
  腰板・底板・制御箱・配線ダクト・銘板はこれを読んで組む。45/135/225/315 度を
  直書きしない（脚は供給スターホイールの軸を避けるために振ってある）。
  架台の作り（脚の丈・桁の高さ）は parts.FRAME_* を import して読む。写すと
  parts.frame() を直したときに小物だけ静かにずれる。

液の供給系はノズルに追従する
  タンクとポンプはノズルの世界角（lay["fill_deg"]）から +45 度の所に置き、
  配管も同じ角を読んで組む。天板の下を半径一定でぐるりと回り込み、充填
  ステーションの手前で支柱へ寄せる。fill_deg を変えると回り込みの角度だけが
  変わり、ノズルとの繋がりは保たれる。

欠けたキーは黙って埋めない
  lay に無いキーを控えの数で埋めると、モジュールごとに違う機械が組み上がる
  （実際に frame_leg_phase_deg の控えが asm_transfer と食い違っていた）。
  控えを使うときは必ず _need() を通し、stderr に出す。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parts  # noqa: E402


# 一度出した警告は繰り返さない（コマ送りで 240 回出ると読めなくなる）
_WARNED: set = set()


def _warn(msg: str) -> None:
    """stderr に 1 回だけ出す。"""
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    print(f"[asm_details] {msg}", file=sys.stderr)


def _need(lay: dict, key: str, fallback, why: str = ""):
    """lay[key] を読む。無ければ控えを返し、**必ず stderr に出す。**

    控えの数はここにしか無いので、基盤側がキーを消すとこのモジュールだけが
    別の機械を組む。黙って続けないための入口。
    """
    if key in lay and lay[key] is not None:
        return lay[key]
    _warn(f"lay に {key} が無い。控え {fallback} で組む{('。' + why) if why else ''}")
    return fallback


# --------------------------------------------------------------------------
# 絵にするためだけの比率。既存の寸法に対する倍率で書く。絶対値は直書きしない。
# --------------------------------------------------------------------------
PROPORTION = {
    # 架台の作り（脚の丈・桁の高さ）は parts.FRAME_FOOT_H_X_PIPE /
    # FRAME_RAIL_LOW_X_LEG_H / FRAME_RAIL_HIGH_X_PIPE を直に読む。ここに
    # 写しを置かない。アジャスタ座・腰板・底板・銘板の高さが全部そこから
    # 出るので、写すと parts.frame() を直したときに小物だけ静かにずれる。

    # --- 架台の板金（腰板・底板）----------------------------------------
    # 脚 4 本と桁だけだと正面から向こう側が見通せて机に見える。外側の面へ
    # 板金を貼り、下は底板で塞ぐ。塞ぐのは +x と -x の 2 面だけにする。
    #   +y = ギヤモータが張り出す（asm_drive）ので貼れない
    #   -y = 中の制御箱を見せる面。ここを塞ぐと箱が消える
    "skirt_t_x_plate_t": 0.5,          # 腰板の板厚 / テーブル板厚（4 相当）
    "skirt_top_gap_x_plate_t": 0.0,    # 腰板の上端を天板下面から下げる量
    "pan_t_x_plate_t": 0.75,           # 底板の板厚（6 相当）

    # --- 制御箱（架台の中。インバータと PLC が入る）---------------------
    "cab_w_x_plate_d": 0.607,          # 幅（x 方向。340 相当）
    "cab_d_x_plate_d": 0.357,          # 奥行き（y 方向。200 相当）
    "cab_h_x_plate_d": 0.536,          # 高さ（300 相当）
    "cab_gap_x_plate_t": 0.5,          # 割出しユニットの下面との空き

    # --- 操作盤（扉 250 x 180、奥行き 120 相当）-------------------------
    "panel_w_x_plate_d": 0.446,        # 扉の幅 / テーブル外径
    "panel_h_x_plate_d": 0.321,        # 扉の高さ / テーブル外径
    "panel_d_x_plate_d": 0.214,        # 奥行き / テーブル外径
    # 扉を向ける世界角は**直書きしない**。供給と排出の 2 本のレーンに挟まれた
    # 側の中央（工程角から出す。実測では 270 度）に置く。どちらのコンベアも
    # 接点（半径 450）から機外へ伸びるので、その中央は帯に入らず、供給と排出の
    # 両方を見渡せる。以前は 185 度の直書きで、排出を 270 度から 225 度へ
    # 動かしたときに操作盤だけ帯の中に取り残された。
    "panel_clear_deg": 30.0,           # 操作盤が工程角と駆動系から空ける角度
    # 半径は架台の脚のアジャスタ座（半径 340.5）とスタンドの据付板が当たら
    # ない所まで外す。520 なら座の縁から 31 空く。
    "panel_at_r_x_plate_r": 1.86,      # 箱の中心半径 / テーブル半径（521 相当）
    "panel_off_x_plate_d": 0.0,        # 横へのずらしは無し（2 本のレーンの真ん中）
    # 非常停止の中心高さ / 架台天板の高さ。以前は 775 で天板 868 より低く、
    # 操作者がかがむ位置だった。970 まで上げる。
    "estop_z_x_deck_top": 1.118,
    "panel_mount_x_frame_pipe": 0.05,  # 扉面から操作器具の座を出す量

    # --- 操作盤の脚（床置きのスタンド）----------------------------------
    # 箱の下端が天板より上に来るので、脚の角には抱かせられない。床から
    # 角柱で立てる。天板（半径 346）の外なので何にも当たらない。
    "stand_post_x_frame_pipe": 1.87,   # 角柱の一辺 / 架台の角パイプ（90 相当）
    "stand_base_x_plate_d": 0.5,       # 据付板の一辺（280 相当）
    "stand_base_t_x_plate_t": 1.75,    # 同 板厚（14 相当）

    # --- 積層シグナルタワー（操作盤の上）--------------------------------
    "tower_stem_d_x_plate_d": 0.054,   # 支柱の径（30 相当）
    "tower_stem_h_x_plate_d": 0.107,   # 同 丈（60 相当）
    "tower_lamp_d_x_plate_d": 0.104,   # 灯の径（58 相当）
    "tower_lamp_h_x_plate_d": 0.0786,  # 灯 1 段の丈（44 相当）
    "tower_cap_h_x_plate_d": 0.018,    # 頭のふた（10 相当）

    # --- 非常停止と操作器具（扉の上での置き方は扉寸法の比で持つ）--------
    "estop_base_x_plate_d": 0.107,     # 黄ベース径 / テーブル外径（60 相当）
    "estop_button_x_estop_base": 0.75, # 赤きのこ径 / 黄ベース径（45 相当）
    "estop_at_x_panel_w": 0.22,        # 非常停止の横振り / 扉の幅
    "estop_at_x_panel_h": 0.25,        # 同 縦振り / 扉の高さ
    "lamp_d_x_estop_base": 0.30,       # 表示灯の径 / 黄ベース径
    "lamp_out_x_estop_base": 0.22,     # 表示灯の出っ張り
    "lamp_at_x_panel_h": 0.31,         # 表示灯の縦振り
    "knob_d_x_estop_base": 0.50,       # 速度つまみの径
    "knob_out_x_estop_base": 0.33,     # 同 出っ張り
    "stop_d_x_estop_base": 0.38,       # 運転停止押しボタンの径
    "stop_out_x_estop_base": 0.20,     # 同 出っ張り
    "op_left_x_panel_w": 0.08,         # 押しボタン列の横振り（内側）
    "op_right_x_panel_w": 0.20,        # 同（外側）
    "op_low_x_panel_h": 0.06,          # 押しボタン列の縦振り
    "label_x_estop_base": 0.67,        # 警告ラベル 1 辺 / 黄ベース径（40 相当）
    "label_at_x_panel_h": 0.25,        # 警告ラベルの縦振り（扉の下側）

    # --- アジャスタ脚 ---------------------------------------------------
    # parts.frame() が脚の下に丸柱を作っている。ここでねじ軸をもう 1 本
    # 立てると二重になるので、丸柱をねじ軸に見立てて丸座とロックナットだけ
    # 足す。
    "foot_pad_d_x_frame_pipe": 1.45,   # 丸座の径 / 角パイプ一辺（70 相当）
    "foot_pad_top_x_frame_pipe": 1.16, # 丸座の上面の径（56 相当）
    "foot_pad_h_x_frame_pipe": 0.29,   # 丸座の厚み（14 相当）
    "foot_nut_x_frame_pipe": 1.20,     # ロックナットの径（58 相当）
    "foot_nut_h_x_frame_pipe": 0.25,   # 同 厚み（12 相当）

    # --- 配線ダクト（架台の外側の面に沿わせる）--------------------------
    # 架台の局所 -y 面（脚が振れているので世界角 247.5 度）の下桁に沿って
    # 走り、操作盤の側へ抜けてスタンドの柱を立ち上がる。-y 面を選ぶのは、
    # そこが操作盤にいちばん近く、腰板を貼っていない面だから。
    "duct_x_frame_pipe": 0.83,         # ダクト一辺 / 角パイプ一辺（40 相当）
    "duct_embed_x_duct": 0.10,         # 腰板の面へ食い込ませる量 / ダクト一辺
    "duct_wall_x_duct": 0.075,         # 樋の肉厚 / ダクト一辺

    # --- 液受け溝（テーブル外周の環状の溝）------------------------------
    # 以前は 160x100 の浅い皿を世界角 26 度に 1 枚置いていた。ノズル軸から
    # 155 mm ずれていて滴を受けられず、回転テーブルとの隙間も 11 mm しか
    # 無かった。天板の上に環状の溝を作り直す。
    # 内半径は外周ガイドの支柱の据付座（半径 308 まで）から 7 空ける。
    # 回転テーブル板（半径 280）からは 35 空く。外半径は架台天板
    # （半径 346）の内に 8 残す。
    "gutter_ri_x_plate_r": 1.125,      # 溝の内半径 / テーブル半径（315 相当）
    "gutter_ro_x_plate_r": 1.207,      # 同 外半径（338 相当）
    "gutter_floor_x_plate_t": 0.5,     # 底板の板厚（4 相当）
    "gutter_h_x_plate_t": 3.2,         # 立ち上がりの高さ（26 相当）
    "gutter_wall_x_plate_t": 0.4,      # 立ち上がりの板厚（3.2 相当）
    # 角度範囲は充填ステーションからの相対角で、**始まりは支柱から出す**。
    # 以前は 55 度からで、ノズル（0 度）から垂れた滴の行き先が無かった。
    # テーブルは供給 315 -> 充填 0 -> 45 -> ... と反時計回りに送るので、
    # ノズルの下でテーブルに落ちた滴は 0 度から反時計回りへ運ばれて外周から
    # 落ちる。溝の始まりを支柱のすぐ先まで戻し、そこから数十度は内側へ口を
    # 広げて（gutter_mouth_*）テーブルの縁の真下で受ける。
    # 終わりは 180 度。実測で排出側の星車の据付が世界角 186.6〜204.0 度・
    # 半径 303〜333 まで来て溝の帯（315〜338）に入るので、その手前で切る。
    "gutter_post_dodge_x_post_r": 2.6,  # 支柱の芯から逃げる量 / 支柱半径。
                                        # 根元の据付フランジが支柱半径の 2 倍まで
                                        # 張り出す（parts.nozzle_gantry）のでその外
    "gutter_to_deg": 180.0,
    "gutter_star_gap_x_plate_t": 1.5,  # 星車の軸から空ける量 / テーブル板厚（12 相当）
    "gutter_end_deg": 1.4,             # 端を塞ぐ板の角度幅
    # 受け口。溝の内壁をテーブルの縁の近くまで寄せる区間。回るテーブル板
    # （半径 280）から 10 空け、外周ガイドの支柱の据付座（半径 295.4 ± 12.8）に
    # 当たる手前で閉じる（座の角は lay["trf_ring_posts"] から出す）。
    "gutter_mouth_ri_x_plate_r": 1.036,   # 受け口の内半径（290 相当）
    "gutter_mouth_gap_deg": 3.0,          # ガイド支柱の座から空ける角度
    "gutter_drain_deg": 160.0,         # 排液口の相対角
    "gutter_drain_d_x_plate_t": 1.9,   # 排液口の径（15 相当）
    "gutter_drain_h_x_plate_d": 0.09,  # 天板下面から下げる量（50 相当）
    "gutter_valve_d_x_plate_t": 3.6,   # 排液弁の径（29 相当）
    "gutter_valve_h_x_plate_t": 4.5,   # 同 丈（36 相当）
    # 天板を貫くものの座。板厚の中で管を切ると穴の無い板を管が突き抜けて
    # 見えるので、板をまたぐカラーを 1 つ被せて貫通部の座に見せる。
    "bulkhead_d_x_pipe_d": 2.2,        # 座の径 / 管の径
    "bulkhead_up_x_pipe_d": 0.75,      # 天板上面から出す丈
    "bulkhead_down_x_pipe_d": 0.50,    # 天板下面から出す丈

    # --- センタカバー（テーブル中央）------------------------------------
    # 真上から見るとテーブル中央が直径 380 の空白になる。中心柱に締める
    # 段付きのハブキャップで埋める。大径の段・中径の段・締め座の 3 段で、
    # 大径の段の上にボルト円、中径の段の上に持ち上げ用のつまみを載せる。
    #
    # **回るハブ（半径 56・上面が中心柱の根元）を絶対に飲まないこと。**
    # 以前は中実の円板で組んでいて、固定のカバーが回るハブを 12 mm 飲んで
    # いた。カバーが不透明で大径なので絵では気付けない。どの段も中空の輪で
    # 作り、内半径をハブの外へ出す（center_*_ri_x_hub_r）。
    #
    # 丈は斜め上から見て奥のボトルの足元を隠さない所で止める。仰角 26 度の
    # 見通し線はテーブル面から (225 - 半径) * tan(26 度) なので、半径 150 で
    # 36.6・半径 82 で 69.5・半径 55 で 82.9 まで。段が内へ寄るほど高くできる。
    "center_d_x_plate_d": 0.536,       # 大径の段の外径（300 相当）
    "center_gap_x_plate_t": 2.0,       # 回るテーブル板との空き（16 相当）
    "center_step1_h_x_plate_t": 2.0,   # 大径の段の丈（16 相当）
    "center_step1_ri_x_hub_r": 1.473,  # 同 内半径 / ハブ半径（82.5 相当）
    "center_step2_h_x_plate_t": 2.75,  # 中径の段の丈（22 相当）
    "center_top_x_column_r": 1.6,      # 段の内穴の半径 / 中心柱の半径
    "center_collar_x_column_r": 2.48,  # 締め座の外半径 / 中心柱の半径
    "center_collar_h_x_plate_t": 1.0,  # 同 丈
    "center_bolts": 8,                 # ボルト円の本数
    "center_bolt_d_x_plate_t": 1.25,   # ボルト頭の径 / テーブル板厚（10 相当）
    "center_bolt_h_x_plate_t": 0.75,   # 同 頭の高さ
    "center_knobs": 2,                 # つまみの数（180 度対向）
    "center_knob_at_x_step2": 0.67,    # つまみの半径 / 中径の段の外半径
    "center_knob_d_x_plate_t": 1.5,    # つまみの軸の径
    "center_knob_h_x_plate_t": 1.4,    # 同 軸の丈
    "center_knob_cap_x_knob_d": 2.2,   # 頭の径 / 軸の径

    # --- 銘板 -----------------------------------------------------------
    "plate_w_x_plate_d": 0.107,        # 銘板の幅（60 相当）
    "plate_h_x_plate_d": 0.054,        # 同 高さ（30 相当）
    "plate_t_x_plate_t": 0.19,         # 同 厚み（1.5 相当）
    # 貼る位置の横振り / 脚の芯々距離。作業者側の面のうち、操作盤のスタンド
    # （世界角 270 度）の陰にならない側へ振る。ダクトは下段なので掛からない
    "plate_at_x_frame_span": -0.21,

    # --- 製品タンク -----------------------------------------------------
    # 円筒 φ320 x 高さ 360 で 29 L 相当。毎分 20 本 x 500 mL = 10 L/min に
    # 対して 3 分弱ぶんの中間タンクで、実機でもこのくらいの大きさに収まる。
    #
    # 置き場所は**充填ステーションから +45 度**（世界角の直書きをやめた）。
    # ポンプから充填部への回り込みは天板の下を半径 425 で走るが、その半径では
    # 駆動系が世界角 54〜81 度を塞いでいる（lay["drv_angle_deg"] 67.5 度・
    # 覆いの半幅 102）。+45 度なら回り込みは 45 度ぶんで済み、途中に架台の脚が
    # 1 本あるだけになる。**60〜70 度へは寄せられない**（駆動系の帯に入り、
    # ポンプ棚の脚が割出しユニットの梁に当たる）。
    #
    # 大きさは φ370 x 420 から一回り落とした。以前は胴の頭が 1193 まで来て、
    # 斜め上からの絵でテーブルの奥半分（世界角 45〜90 度）を丸ごと隠していた。
    # 頭を 1089（ボトルの頭 1087 とほぼ同じ）まで下げると、隠す面積が減る。
    "tank_from_fill_deg": 45.0,        # 充填ステーションからの振り
    "tank_at_r_x_plate_r": 2.75,       # タンク中心の半径（770 相当）
    "tank_d_x_plate_d": 0.571,         # 胴の径（320 相当）
    "tank_h_x_plate_d": 0.643,         # 胴の丈（360 相当）
    "tank_dome_h_x_tank_d": 0.16,      # 上の鏡板の丈
    "tank_neck_d_x_tank_d": 0.30,      # 投入口の径
    "tank_neck_h_x_tank_d": 0.09,      # 同 丈
    "tank_cone_h_x_tank_d": 0.173,     # 下の絞りの丈
    "tank_cone_d_x_tank_d": 0.12,      # 同 下端の径
    "tank_lift_x_tank_d": 0.40,        # 架台の天板から胴の下面までの丈（128 相当）
    "tank_leg_x_frame_pipe": 0.62,     # 胴を持ち上げる短い角柱の一辺
    "tank_leg_at_x_tank_d": 0.30,      # 同 芯々の半分
    # 投入口のふたとベント（呼吸口）。密閉すると液が落ちない
    "tank_lid_h_x_tank_d": 0.028,      # 投入口のふたの厚み
    "tank_lid_d_x_neck_d": 1.18,       # 同 径 / 投入口の径
    "tank_vent_d_x_pipe_d": 0.75,      # ベントの管の径
    "tank_vent_h_x_tank_d": 0.19,      # 同 丈
    "tank_vent_at_x_tank_d": 0.30,     # 同 立てる半径
    "tank_vent_cap_x_vent_d": 1.9,     # 頭の傘の径 / ベントの径
    # 出口弁（絞りの下。ここを閉めるとポンプを外せる）
    "tank_valve_d_x_pipe_d": 2.3,      # 弁箱の径 / 液配管の径
    "tank_valve_h_x_pipe_d": 1.5,      # 同 丈
    "tank_valve_lever_x_pipe_d": (2.6, 0.5, 0.35),   # レバーの (長さ, 幅, 厚み)
    # タンク架台（parts.frame で組む）
    "stand_top_x_deck_top": 0.60,      # 架台天板の上面 / 機械の天板高さ（521 相当）
    "stand_span_x_tank_d": 0.75,       # 脚の芯々
    "stand_pipe_x_frame_pipe": 0.70,   # 角パイプ一辺
    "stand_deck_d_x_tank_d": 1.20,     # 丸天板の径
    "stand_deck_t_x_plate_t": 1.75,    # 同 板厚
    # 液面計（胴の側面に沿う細い管）
    "gauge_d_x_pipe_d": 0.90,          # 管の径
    "gauge_z0_x_tank_h": 0.10,         # 下端 / 胴の丈
    "gauge_z1_x_tank_h": 0.80,         # 上端 / 胴の丈

    # --- ポンプ ---------------------------------------------------------
    # タンクの下、機械寄りの棚に載せる。胴（モータ）とヘッドを半径方向に
    # 並べ、吸込みと吐出しをヘッドの上に立てる。
    "pump_shelf_top_x_deck_top": 0.4378,   # 棚の上面（380 相当）
    "pump_shelf_t_x_plate_t": 1.75,        # 棚の板厚（14 相当）
    "pump_shelf_ri_x_plate_r": 1.43,       # 棚の内半径（400 相当）
    "pump_shelf_ro_x_plate_r": 2.36,       # 同 外半径（661 相当）
    "pump_shelf_w_x_tank_d": 0.92,         # 棚の幅（340 相当）
    "pump_leg_at_r_x_plate_r": 1.54,       # 棚の脚を立てる半径
    "pump_leg_x_frame_pipe": 0.70,         # 同 一辺
    "pump_base_t_x_plate_t": 1.75,         # ポンプ座の板厚
    "pump_base_len_x_tank_d": 0.51,        # 同 半径方向の長さ
    "pump_base_w_x_tank_d": 0.49,          # 同 幅
    "pump_body_d_x_tank_d": 0.35,          # 胴の径（129 相当）
    "pump_body_len_x_tank_d": 0.46,        # 同 長さ（170 相当）
    "pump_body_at_r_x_plate_r": 2.054,     # 胴の中心半径（575 相当）
    "pump_head_d_x_tank_d": 0.42,          # ヘッドの径（155 相当）
    "pump_head_len_x_tank_d": 0.21,        # 同 長さ（78 相当）
    "pump_head_at_r_x_plate_r": 1.611,     # ヘッドの中心半径（451 相当）
    "pump_in_at_r_x_plate_r": 1.679,       # 吸込み口の半径位置（470 相当）
    "pump_out_at_r_x_plate_r": 1.518,      # 吐出し口の半径位置（425 相当）
    "pump_stub_d_x_pipe_d": 1.50,          # 口の座の径
    "pump_stub_h_x_pipe_d": 0.94,          # 同 丈
    "pump_gauge_d_x_tank_d": 0.157,        # 圧力計の径（58 相当）
    "pump_gauge_t_x_plate_t": 2.0,         # 同 厚み
    "pump_gauge_h_x_pipe_d": 0.90,         # 圧力計を吐出し口から持ち上げる丈

    # --- 配管 -----------------------------------------------------------
    # parts.nozzle() の作りに合わせた取付カラーの位置。ここへ液配管を差す。
    "nozzle_collar_z_x_len": 0.665,    # カラーの高さ / ノズル長さ
    "nozzle_collar_r_x_bore": 0.75,    # カラーの半径 / ノズル外径
    "pipe_d_x_nozzle_bore": 0.76,      # 液配管の径 / ノズル外径（16 相当）
    "tube_d_x_nozzle_bore": 0.38,      # エアチューブの径（8 相当）
    "pipe_side_x_post_r": 1.69,        # 液配管を支柱の芯から振る量 / 支柱半径
    "tube_side_x_post_r": 1.43,        # エアチューブを振る量（反対側）
    "pipe_enter_x_deck_t": 0.8,        # 天板へ差し込む深さ / 天板の板厚
    "clamp_x_post_r": 1.7,             # 支柱と配管を挟むクランプの一辺 / 支柱半径
    "clamp_z_x_post": (0.42, 0.78),    # クランプの高さ / 支柱の丈（下から）
    # 天板の下を回り込む区間。半径は天板（346）とタンク架台の内側の間、
    # 高さは割出しユニットとギヤモータの頭より上、天板の下面より下。
    "run_r_x_plate_r": 1.518,          # 回り込みの半径（425 相当）
    "run_z_x_deck_top": 0.9332,        # 同 高さ（810 相当）
    "run_seg_deg": 12.0,               # 回り込みを折る刻み
    "tank_out_z_x_deck_top": 0.6797,   # タンクからポンプへ渡る高さ（590 相当）
    # 回り込みを受ける腕木。天板の下面から吊って配管を載せる。
    # 受けの相対角。架台の脚を外した角に立てる。脚は振れることがあるので、
    # lay["frame_leg_deg_all"] から hanger_leg_gap_deg 以上離れるよう寄せ直す。
    "hanger_at_deg": (10.0, 36.0),
    "hanger_leg_gap_deg": 12.0,        # 脚の芯から空ける角度（半径 346 で 72 相当）
    "hanger_ri_x_plate_r": 1.21,       # 受けの内半径
    "hanger_w_x_frame_pipe": 0.83,     # 受けの幅
    "hanger_t_x_plate_t": 1.75,        # 受けの板厚
    # 空気弁の箱。以前はアームの上（z=1233）に置いてあって機械の全高より
    # 高かった。支柱の外側へ下ろす。
    "valve_x_arm_w": (1.5, 1.0, 0.8),  # 弁の箱 / アーム幅
    "valve_z_x_post": 0.62,            # 弁の高さ / 支柱の丈（下から）
    "valve_air_z_x_post": 0.79,        # 弁から出たエアを渡す高さ / 支柱の丈

    # --- ノズルの滴受け -------------------------------------------------
    # ノズルの真下に固定の受けは置けない。そこは回るテーブルの通り道で、
    # ボトルの頭（1087）とノズル先端（1099）の間は 12 しか無い。実機と同じに、
    # (1) ノズルの胴を伝って落ちる液は胴に締めた受け皿で止め、
    # (2) 先端から垂れてテーブルに落ちた滴は、テーブルが反時計回りに送る先で
    #     外周から液受け溝の受け口へ落とす、の 2 段で受ける。
    # 受け皿はノズルのテーパが終わって胴が立つ高さに締める（そこより下は
    # 径が細っていて皿が締まらない）。
    "cup_z_x_nozzle_len": 0.235,       # 皿の底 / ノズル長さ（先端から）
    "cup_ro_x_nozzle_bore": 1.52,      # 皿の外半径 / ノズル外径（32 相当）
    "cup_ri_x_nozzle_bore": 0.475,     # 同 内半径。ノズル胴へ少し食い込ませて
                                       # 締めた輪に見せる（面が重なると散らつく）
    "cup_floor_x_plate_t": 0.5,        # 皿の底の板厚
    "cup_h_x_nozzle_bore": 0.9,        # 皿の立ち上がりの丈
    "cup_wall_x_plate_t": 0.3,         # 同 板厚
    # 落とし管の通り道。支柱（半径 308・φ30.8）の外、安全カバーの板
    # （内面 341.6）の内という狭い帯を通す。降りる所は空気弁の箱を避けて
    # 充填ステーションから cup_drain_side_deg だけ振る。
    "cup_drain_r_x_gutter_ro": 0.985,  # 降りる半径 / 液受け溝の外半径（333 相当）
    "cup_drain_side_deg": 8.0,         # 降りる位置の振り（支柱と空気弁の箱を避ける）
    "cup_drain_seg_deg": 4.0,          # 帯に沿って回る区間の刻み
    "cup_drain_over_x_plate_t": 1.5,   # 溝の立ち上がりの上を渡す高さ
    "cup_drain_in_x_plate_t": 1.0,     # 溝の底の上で止める高さ
    "cup_drain_at_deg": 2.5,           # 溝へ落とす位置（溝の始まりからの角度）
}

# 材質。接頭辞は det_。すべて不透明。
MATERIALS = {
    "det_panel":      dict(color="#aab1b7", pbr=True, metallic=0.28, roughness=0.55),
    "det_panel_dark": dict(color="#31363b", pbr=True, metallic=0.35, roughness=0.60),
    "det_red":        dict(color="#b5241d", pbr=True, metallic=0.00, roughness=0.45),
    "det_yellow":     dict(color="#d9a90f", pbr=True, metallic=0.00, roughness=0.50),
    "det_lamp":       dict(color="#37a352", pbr=True, metallic=0.00, roughness=0.25),
    "det_adjuster":   dict(color="#8e979f", pbr=True, metallic=0.72, roughness=0.42),
    "det_duct":       dict(color="#c4c9cd", pbr=True, metallic=0.05, roughness=0.70),
    "det_duct_lid":   dict(color="#aeb4b9", pbr=True, metallic=0.05, roughness=0.65),
    "det_tray":       dict(color="#c8d0d6", pbr=True, metallic=0.80, roughness=0.30),
    "det_plate":      dict(color="#dfe3e6", pbr=True, metallic=0.90, roughness=0.18),
    # 締結部品（センタカバーのボルト・つまみ）。磨いた蓋より暗いつや消しの
    # ステンレスにして、真上のフラット照明で白い天板と溶けないようにする。
    "det_bolt":       dict(color="#79828a", pbr=True, metallic=0.58, roughness=0.46),
    "det_pipe_sus":   dict(color="#bcc4cb", pbr=True, metallic=0.85, roughness=0.22),
    "det_tube_air":   dict(color="#6f95b4", pbr=True, metallic=0.00, roughness=0.35),
    # 液の供給系
    "det_tank":       dict(color="#d2d9de", pbr=True, metallic=0.82, roughness=0.26),
    "det_stand":      dict(color="#6e767e", pbr=True, metallic=0.52, roughness=0.66),
    "det_pump":       dict(color="#46545f", pbr=True, metallic=0.40, roughness=0.48),
}

# 描く順に差し込む断片。載せ忘れた群は絵に出ない。
DRAW_ORDER = (
    ("det_adjuster", "det_adjuster"),
    ("det_stand", "det_stand"),
    ("det_duct", "det_duct"),
    ("det_duct_lid", "det_duct_lid"),
    ("det_panel", "det_panel"),
    ("det_panel_dark", "det_panel_dark"),
    ("det_pump", "det_pump"),
    ("det_red", "det_red"),
    ("det_yellow", "det_yellow"),
    ("det_lamp", "det_lamp"),
    ("det_tray", "det_tray"),
    ("det_plate", "det_plate"),
    ("det_bolt", "det_bolt"),
    ("det_tank", "det_tank"),
    ("det_pipe_sus", "det_pipe_sus"),
    ("det_tube_air", "det_tube_air"),
)

# 小物は回らない。
ROTATING = False


# --------------------------------------------------------------------------
# 座標
# --------------------------------------------------------------------------
def _polar(deg, r):
    """世界角 deg [度]・半径 r の点。"""
    a = np.radians(float(deg))
    return (float(r) * np.cos(a), float(r) * np.sin(a))


def _spin_pts(points, deg):
    """(x, y, z) の並びを z 軸まわりに deg [度] 回す。"""
    c, s = np.cos(np.radians(float(deg))), np.sin(np.radians(float(deg)))
    return tuple((x * c - y * s, x * s + y * c, z) for x, y, z in points)


def _short_turn(from_deg, to_deg):
    """from_deg から to_deg への回り方のうち、短いほうの符号付き角度。"""
    d = (float(to_deg) - float(from_deg)) % 360.0
    return d - 360.0 if d > 180.0 else d


def _mid_away_from(a_deg, b_deg, avoid_deg):
    """a から b への 2 通りの弧のうち、avoid を含まないほうの中点。

    操作盤を「2 本のコンベアのレーンに挟まれた側」に置くのに使う。工程角が
    動いても、充填ステーションの居ないほうの中央が付いてくる。
    """
    a, b = float(a_deg), float(b_deg)
    ccw = (b - a) % 360.0                       # a から反時計回りに b まで
    if (float(avoid_deg) - a) % 360.0 < ccw:    # その側に avoid が居る
        return (a - (360.0 - ccw) / 2.0) % 360.0
    return (a + ccw / 2.0) % 360.0


def _clear_of(deg, blockers, gap_deg):
    """blockers のどの角からも gap_deg 以上離れた角へ寄せ直す。

    架台の脚は振れることがある。脚に当たる角へ腕木を立てないための逃がし。
    寄せる向きは、いま近いほうの脚から離れる向き。
    """
    deg = float(deg)
    for _ in range(len(blockers) + 2):
        near = None
        for b in blockers:
            d = _short_turn(b, deg)
            if abs(d) < gap_deg and (near is None or abs(d) < abs(near[0])):
                near = (d, float(b))
        if near is None:
            return deg
        d, b = near
        deg = b + gap_deg * (1.0 if d >= 0.0 else -1.0)
    return deg


def layout(params: dict, lay: dict) -> dict:
    """小物の絶対座標と寸法。キーはすべて det_ 接頭辞つき。

    局所座標のまま返して build() で回すものが 4 系統ある。
      * 架台に付くもの det_frame_spin（脚の振り。腰板・底板・制御箱・銘板・
        下段の配線ダクトはこれを掛けて世界へ回す）
      * 操作盤   det_panel_spin（扉を外へ向けるための回転）
      * 液の供給系 det_tank_spin（タンクを世界角 45 度へ置く回転。+x が
        機械から外へ向かう向き）
      * 充填部の配管まわり det_fill_spin（ノズルの世界角）
    配管の点列だけは折れ線をそのまま渡すので、ここで世界座標に直してある。
    """
    p = PROPORTION

    pipe = lay["frame_pipe"]
    half = lay["frame_span"] / 2.0
    face = half + pipe / 2.0                    # 架台の外側の面までの距離

    # 架台の脚の向き。基盤側の lay が唯一の表。無いときだけ対角（45 度刻み）
    # に戻す。frame_leg_deg_all が 4 本ぶんの世界角、frame_leg_deg は 1 本目
    # だけの数なので、どちらで来ても 4 本の並びに直して使う。
    # 控えの -22.5 は asm_transfer と同じ数にしてある（供給スターホイールの軸を
    # 避けるための振り）。以前ここだけ 0.0 で、キーが消えると受け渡しだけ振れた
    # まま残る食い違いだった。控えを使ったら必ず stderr に出す。
    leg_phase = float(_need(lay, "frame_leg_phase_deg", -22.5,
                            "asm_transfer と同じ数にしてある"))
    leg_deg = lay.get("frame_leg_deg_all", lay.get("frame_leg_deg"))
    if leg_deg is None:
        _warn("lay に frame_leg_deg_all も frame_leg_deg も無い。対角に戻して組む")
        leg_deg = tuple(45.0 + leg_phase + 90.0 * k for k in range(4))
    elif np.ndim(leg_deg) == 0:
        leg_deg = tuple(float(leg_deg) + 90.0 * k for k in range(4))
    leg_deg = tuple(float(v) for v in leg_deg)
    leg_xy = lay.get("frame_leg_xy")
    if leg_xy is None:
        leg_xy = tuple(_polar(a, half * np.sqrt(2.0)) for a in leg_deg)
    leg_xy = tuple((float(x), float(y)) for x, y in leg_xy)
    # 架台の作りは parts.frame() の公開定数を直に読む（写しを持たない）
    foot_h = pipe * parts.FRAME_FOOT_H_X_PIPE   # 脚の下端＝アジャスタの丈
    deck_top = lay["deck_top"]
    deck_bottom = deck_top - lay["deck_t"]
    leg_h = deck_bottom - foot_h
    rail_low_z = foot_h + leg_h * parts.FRAME_RAIL_LOW_X_LEG_H
    rail_high_z = deck_bottom - pipe * parts.FRAME_RAIL_HIGH_X_PIPE
    rail_low_top = rail_low_z + pipe / 2.0

    plate_d = lay["plate_d"]
    plate_r = lay["plate_r"]
    plate_t = lay["plate_t"]
    fill_deg = lay["fill_deg"]
    infeed_deg = float(_need(lay, "infeed_deg", fill_deg - 45.0))
    # 供給は充填の 1 ステーション手前（-45）、排出は 3 手前（-135 = +225）。
    # どちらも fill から工程角ぶん戻る向きで揃える。控えが要るのは params が
    # 工程角を落としたときだけで、そのときは _need が警告する。
    discharge_deg = float(_need(lay, "discharge_deg", fill_deg - 135.0))

    # --- 架台の板金 -------------------------------------------------------
    skirt_t = plate_t * p["skirt_t_x_plate_t"]
    skirt_x = face + skirt_t / 2.0
    pan_t = plate_t * p["pan_t_x_plate_t"]
    # 底板は下桁の内側へ落とし込む。桁の外まで広げると脚 4 本を板が
    # 串刺しにするので、一辺は脚の芯々から角パイプ 1 本ぶん詰める。
    pan_span = lay["frame_span"] - pipe
    pan_top = rail_low_top
    skirt_z = (pan_top, deck_bottom - plate_t * p["skirt_top_gap_x_plate_t"])

    # --- 制御箱（架台の中。扉は -y を向く）--------------------------------
    cab_w = plate_d * p["cab_w_x_plate_d"]
    cab_d = plate_d * p["cab_d_x_plate_d"]
    cab_h = plate_d * p["cab_h_x_plate_d"]
    # 割出しユニットの取付ベースが上に来る。丈はその下で止める
    # （駆動が読み込まれていないときは既定の丈のまま）
    unit_bottom = lay.get("drv_base_z")
    if unit_bottom is not None:
        cab_h = min(cab_h, unit_bottom - pan_top - plate_t * p["cab_gap_x_plate_t"])
    cab_face_y = -(half - pipe / 2.0)           # 脚の内側の面。ここに扉を揃える
    cab_center = (0.0, cab_face_y + cab_d / 2.0, pan_top + cab_h / 2.0)

    # --- 操作盤（局所座標。det_panel_spin で世界へ回す）-------------------
    # 扉を向ける世界角は工程角から出す。排出と供給の 2 本のレーンに挟まれた
    # 側（充填ステーションの居ないほう）の中央。世界角は直書きしない。
    panel_deg = _mid_away_from(discharge_deg, infeed_deg, fill_deg)
    for name, deg in (("充填", fill_deg), ("供給", infeed_deg),
                      ("排出", discharge_deg),
                      ("駆動", lay.get("drv_angle_deg"))):
        if deg is None:
            continue
        if abs(_short_turn(deg, panel_deg)) < p["panel_clear_deg"]:
            _warn(f"操作盤 {panel_deg:.1f} 度が{name} {float(deg):.1f} 度から"
                  f" {abs(_short_turn(deg, panel_deg)):.1f} 度しか離れていない。"
                  "工程角が変わったら置き場所を見直すこと")
    pan_dp = plate_d * p["panel_d_x_plate_d"]
    pan_w = plate_d * p["panel_w_x_plate_d"]
    pan_h = plate_d * p["panel_h_x_plate_d"]
    pan_cx = plate_r * p["panel_at_r_x_plate_r"]
    pan_cy = plate_d * p["panel_off_x_plate_d"]
    pan_cz = deck_top * p["estop_z_x_deck_top"] - pan_h * p["estop_at_x_panel_h"]
    door_x = pan_cx + pan_dp / 2.0 + pipe * p["panel_mount_x_frame_pipe"]
    pan_bottom = pan_cz - pan_h / 2.0
    pan_top_z = pan_cz + pan_h / 2.0

    stand_post = pipe * p["stand_post_x_frame_pipe"]
    stand_base_t = plate_t * p["stand_base_t_x_plate_t"]

    estop_base = plate_d * p["estop_base_x_plate_d"]

    # --- シグナルタワー（操作盤の上。局所座標）---------------------------
    tower_lamp_h = plate_d * p["tower_lamp_h_x_plate_d"]
    tower_z = (pan_top_z,
               pan_top_z + plate_d * p["tower_stem_h_x_plate_d"])

    # --- 配線ダクト -------------------------------------------------------
    # 下段は架台の局所 -y 面（腰板を貼らない面）に沿って走る。そこから
    # 操作盤の側へ横へ抜け、スタンドの柱の機械寄りの面を立ち上がる。
    # 下段だけ架台の局所座標、横と立ち上がりは操作盤の局所座標で持つ。
    duct = pipe * p["duct_x_frame_pipe"]
    duct_embed = duct * p["duct_embed_x_duct"]
    duct_y = -(face + duct / 2.0 - duct_embed)   # 架台の局所 -y 面へ抱かせる
    # 立ち上がりは操作盤の柱の面へ寄せる。宙で切らない
    riser_x = pan_cx - stand_post / 2.0 - duct / 2.0 + duct_embed
    # 横へ抜ける区間の始まり。操作盤の局所 x 軸が架台の局所 -y 面と交わる所。
    # 架台の局所 +y 方向を世界で表した向きに、操作盤の局所軸を投げて出す。
    frame_n = _polar(leg_phase + 90.0, 1.0)
    e1 = _polar(panel_deg, 1.0)
    e2 = _polar(panel_deg + 90.0, 1.0)
    along = e1[0] * frame_n[0] + e1[1] * frame_n[1]
    side = e2[0] * frame_n[0] + e2[1] * frame_n[1]
    if abs(along) < 1e-6:
        cross_x = face + duct                   # 交わらない置き方のときの控え
    else:
        cross_x = (duct_y - pan_cy * side) / along
    # 架台の中を逆走させない。交点が架台の内へ落ちたら面の外まで押し出す
    cross_x = float(np.clip(cross_x, face + duct, riser_x - duct))

    # --- 液受け溝 ---------------------------------------------------------
    gut_ri = plate_r * p["gutter_ri_x_plate_r"]
    gut_ro = plate_r * p["gutter_ro_x_plate_r"]
    gut_drain_xy = _polar(fill_deg + p["gutter_drain_deg"], (gut_ri + gut_ro) / 2.0)
    # 始まりはノズルの支柱のすぐ先。支柱の根元の据付フランジまで逃げる。
    post_dodge = np.degrees(np.arcsin(min(
        1.0, lay["post_r"] * p["gutter_post_dodge_x_post_r"] / lay["post_at_r"])))
    gut_from = fill_deg + post_dodge
    gut_to = fill_deg + p["gutter_to_deg"]
    # 排出のスターホイールの軸が溝の帯（半径 315〜338）を横切る。星車の
    # 寸法から、軸の芯から必要なだけ手前で溝を止める。**控えで黙って
    # 無効にしない。** 以前は控えが 0.0 で、キーが欠けると if が偽になり、
    # 溝が排出星車の軸を横切ったまま組まれた。
    star_c = float(_need(lay, "trf_center_r", 0.0, "星車の軸よけが効かない"))
    star_shaft_r = float(_need(lay, "trf_shaft_d", 0.0,
                               "星車の軸よけが効かない")) / 2.0
    if star_c > 0.0 and star_shaft_r > 0.0:
        keep = star_shaft_r + plate_t * p["gutter_star_gap_x_plate_t"]
        dodge = np.degrees(2.0 * np.arcsin(min(1.0, keep / (2.0 * star_c))))
        limit = (discharge_deg - dodge - gut_from) % 360.0
        gut_to = gut_from + min((gut_to - gut_from) % 360.0, limit)

    # 受け口。溝の内壁をテーブルの縁の近くまで寄せる区間。ノズルの下で
    # テーブルに落ちた滴は、テーブルが反時計回りに送る先で外周から落ちる。
    # 外周ガイドの支柱の据付座が溝の内側の帯を塞ぐので、その手前で閉じる。
    gut_mouth_ri = plate_r * p["gutter_mouth_ri_x_plate_r"]
    gut_mouth_to = gut_to
    ring_posts = lay.get("trf_ring_posts")
    if ring_posts is None:
        _warn("lay に trf_ring_posts が無い。外周ガイドの支柱の座を避けられない"
              "ので、液受け溝の受け口は作らない")
        gut_mouth_to = gut_from
    else:
        foot_r = float(_need(lay, "trf_ring_post_foot_d", 0.0)) / 2.0
        post_at_r = float(_need(lay, "trf_ring_post_at_r", plate_r * 1.055))
        foot_dodge = np.degrees(np.arcsin(min(1.0, foot_r / max(post_at_r, 1e-6)))) \
            + p["gutter_mouth_gap_deg"]
        for x, y in ring_posts:
            a = np.degrees(np.arctan2(float(y), float(x)))
            limit = (a - foot_dodge - gut_from) % 360.0
            gut_mouth_to = gut_from + min((gut_mouth_to - gut_from) % 360.0, limit)

    # --- センタカバー -----------------------------------------------------
    # 段付きのハブキャップ。中心柱に締め、回るハブには触れない。
    hub_r = float(_need(lay, "hub_d", plate_d * 0.2)) / 2.0
    center_z0 = lay["table_top"] + plate_t * p["center_gap_x_plate_t"]
    center_z1 = (center_z0 + plate_t * (p["center_step1_h_x_plate_t"]
                                        + p["center_step2_h_x_plate_t"]))
    # 回るハブの上面（＝中心柱の根元）と、段の内半径。どちらかが破れると
    # 固定のカバーが回るハブを飲む。絵では大径で不透明なので気付けない。
    hub_top = float(_need(lay, "column_base", lay["table_top"] + hub_r / 2.0))
    step1_ri = hub_r * p["center_step1_ri_x_hub_r"]
    if step1_ri <= hub_r and center_z0 < hub_top:
        _warn(f"センタカバーの内半径 {step1_ri:.1f} が回るハブ {hub_r:.1f} の内側で、"
              f"下端 {center_z0:.1f} がハブの上面 {hub_top:.1f} より下。ハブを飲む")
    # 斜め上（仰角 26 度）から奥のボトルの足元を見通す線。段の頭がこれを
    # 超えるとテーブルの向こうのボトルが隠れる。
    sight = np.tan(np.radians(26.0))
    knob_r = step1_ri * p["center_knob_at_x_step2"]
    knob_top = (center_z1 + plate_t * p["center_knob_h_x_plate_t"]
                + plate_t * p["center_knob_d_x_plate_t"]
                * p["center_knob_cap_x_knob_d"] * 0.45)
    for label, rr, zz in (
            ("大径の段", plate_d * p["center_d_x_plate_d"] / 2.0,
             center_z0 + plate_t * p["center_step1_h_x_plate_t"]),
            ("中径の段", step1_ri, center_z1),
            ("つまみ", knob_r, knob_top)):
        limit = lay["table_top"] + (lay["pitch_r"] - rr) * sight
        if zz > limit:
            _warn(f"センタカバーの{label}（半径 {rr:.0f}・頭 {zz:.1f}）が"
                  f"見通し線 {limit:.1f} を超える。奥のボトルの足元を隠す")

    # --- 配管の高さ関係 ---------------------------------------------------
    post_x = lay["post_at_r"]
    post_r = lay["post_r"]
    pipe_d = lay["nozzle_bore"] * p["pipe_d_x_nozzle_bore"]
    tube_d = lay["nozzle_bore"] * p["tube_d_x_nozzle_bore"]
    pipe_y = -post_r * p["pipe_side_x_post_r"]   # 手前側。液配管
    tube_y = post_r * p["tube_side_x_post_r"]    # 奥側。エアチューブ
    deck_in_z = deck_top - lay["deck_t"] * p["pipe_enter_x_deck_t"]
    arm_mid_z = lay["arm_base"] + lay["arm_t"] / 2.0
    collar_z = lay["nozzle_tip"] + lay["nozzle_len"] * p["nozzle_collar_z_x_len"]
    collar_r = lay["nozzle_bore"] * p["nozzle_collar_r_x_bore"]
    post_h = lay["arm_top"] - deck_top
    run_r = plate_r * p["run_r_x_plate_r"]
    run_z = deck_top * p["run_z_x_deck_top"]

    # --- 空気弁（支柱の外側）---------------------------------------------
    vw, vd, vh = (lay["arm_w"] * v for v in p["valve_x_arm_w"])
    valve_cz = deck_top + post_h * p["valve_z_x_post"]
    valve_cx = post_x + post_r + vd / 2.0
    air_z = deck_top + post_h * p["valve_air_z_x_post"]

    # --- タンクとポンプ（局所座標。+x が機械から外へ向かう）---------------
    # 世界角は直書きしない。充填ステーションからの振りで置く。
    tank_deg = fill_deg + p["tank_from_fill_deg"]
    tank_at_r = plate_r * p["tank_at_r_x_plate_r"]
    tank_d = plate_d * p["tank_d_x_plate_d"]
    tank_h = plate_d * p["tank_h_x_plate_d"]
    stand_top = deck_top * p["stand_top_x_deck_top"]
    tank_base = stand_top + tank_d * p["tank_lift_x_tank_d"]
    tank_cone_h = tank_d * p["tank_cone_h_x_tank_d"]
    tank_out_z = tank_base - tank_cone_h        # 絞りの下端＝タンクの出口

    # タンク架台とポンプ棚の脚。丈は parts.frame() の作りに合わせる
    stand_pipe = pipe * p["stand_pipe_x_frame_pipe"]
    stand_foot_h = stand_pipe * parts.FRAME_FOOT_H_X_PIPE
    stand_half = tank_d * p["stand_span_x_tank_d"] / 2.0
    stand_foot_xy = tuple(
        (x, y) for x, y, _ in _spin_pts(
            tuple((tank_at_r + sx * stand_half, sy * stand_half, 0.0)
                  for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)), tank_deg))

    shelf_top = deck_top * p["pump_shelf_top_x_deck_top"]
    shelf_t = plate_t * p["pump_shelf_t_x_plate_t"]
    pump_base_t = plate_t * p["pump_base_t_x_plate_t"]
    pump_body_d = tank_d * p["pump_body_d_x_tank_d"]
    pump_head_d = tank_d * p["pump_head_d_x_tank_d"]
    pump_axis_z = shelf_top + pump_base_t + pump_body_d / 2.0
    pump_head_top = pump_axis_z + pump_head_d / 2.0
    stub_h = pipe_d * p["pump_stub_h_x_pipe_d"]
    pump_in_r = plate_r * p["pump_in_at_r_x_plate_r"]
    pump_out_r = plate_r * p["pump_out_at_r_x_plate_r"]
    cross_z = deck_top * p["tank_out_z_x_deck_top"]
    shelf_foot_xy = tuple(
        (x, y) for x, y, _ in _spin_pts(
            ((plate_r * p["pump_leg_at_r_x_plate_r"],
              sy * (tank_d * p["pump_shelf_w_x_tank_d"] / 2.0
                    - pipe * p["pump_leg_x_frame_pipe"] * 1.2), 0.0)
             for sy in (-1.0, 1.0)), tank_deg))
    # タンクの出口はポンプの吸込み口より上でなければ液が落ちない
    if tank_out_z <= pump_head_top + stub_h:
        _warn(f"タンクの出口 {tank_out_z:.1f} がポンプの吸込み口"
              f" {pump_head_top + stub_h:.1f} より低い。配管が上り勾配になる")

    # --- 配管の点列（世界座標）-------------------------------------------
    # タンク -> ポンプ。絞りの下から降りて水平に渡り、吸込み口へ落とす
    feed_pts = _spin_pts((
        (tank_at_r, 0.0, tank_out_z),
        (tank_at_r, 0.0, cross_z),
        (pump_in_r, 0.0, cross_z),
        (pump_in_r, 0.0, pump_head_top + stub_h),
    ), tank_deg)

    # ポンプ -> 天板の下を回り込み -> 支柱 -> アームの脇 -> ノズルのカラー。
    # 吐出し口の半径は回り込みの半径と同じにしてあるので、立ち上がりが
    # そのまま回り込みの始点になる（pump_out_at_r と run_r は同じ倍率）。
    riser = _spin_pts((
        (pump_out_r, 0.0, pump_head_top + stub_h),
        (pump_out_r, 0.0, run_z),
    ), tank_deg)
    turn = _short_turn(tank_deg, fill_deg)
    n_seg = max(int(abs(turn) / p["run_seg_deg"]), 1)
    arc = tuple(_polar(tank_deg + turn * i / n_seg, run_r) + (run_z,)
                for i in range(1, n_seg + 1))
    # 回り込みが駆動系の帯を突き抜けていないか。ギヤモータの覆いは天板の
    # 下面まで届いていて、この半径では通り抜けられない。タンクの振りを
    # 変えたときに黙って貫通させないための検査。
    drv_deg = lay.get("drv_angle_deg")
    drv_half_w = lay.get("drv_panel_half_w")
    if drv_deg is None or drv_half_w is None:
        _warn("lay に drv_angle_deg / drv_panel_half_w が無い。"
              "回り込みが駆動系を突き抜けていないか確かめられない")
    else:
        drv_half = np.degrees(np.arctan2(float(drv_half_w), run_r))
        for a in [tank_deg] + [tank_deg + turn * i / n_seg
                               for i in range(n_seg + 1)]:
            if abs(_short_turn(float(drv_deg), a)) < drv_half:
                _warn(f"液配管の回り込み（半径 {run_r:.0f}）が駆動系の帯"
                      f"（{float(drv_deg):.1f} 度 ± {drv_half:.1f} 度）に入る。"
                      "タンクの振り tank_from_fill_deg を見直すこと")
                break
    tail = _spin_pts((
        (run_r, pipe_y, run_z),
        (post_x, pipe_y, run_z),
        (post_x, pipe_y, deck_in_z),
        (post_x, pipe_y, arm_mid_z),
        (lay["pitch_r"], pipe_y, arm_mid_z),
        (lay["pitch_r"], pipe_y, collar_z),
        (lay["pitch_r"], -collar_r * 0.9, collar_z),
    ), fill_deg)
    supply_pts = riser + arc + tail

    # エアチューブ。天板 -> 支柱の脇を上がる -> 弁 -> アームの下をノズルへ
    tube_pts = _spin_pts((
        (post_x, tube_y, deck_in_z),
        (post_x, tube_y, valve_cz),
        (valve_cx - vd / 2.0, tube_y, valve_cz),
    ), fill_deg)
    tube_out_pts = _spin_pts((
        (valve_cx, tube_y, valve_cz + vh / 2.0),
        (valve_cx, tube_y, air_z),
        (lay["pitch_r"], tube_y, air_z),
        (lay["pitch_r"], lay["nozzle_bore"] * 0.67, air_z),
    ), fill_deg)

    # 回り込みを受ける腕木の角。架台の脚に当たる角へは立てない
    hanger_degs = tuple(_clear_of(fill_deg + v, leg_deg, p["hanger_leg_gap_deg"])
                        for v in p["hanger_at_deg"])

    clamp = post_r * p["clamp_x_post_r"]
    clamp_z = tuple(deck_top + post_h * v for v in p["clamp_z_x_post"])

    # --- ノズルの滴受けと落とし管 -----------------------------------------
    # 皿はノズルのテーパが終わって胴が立つ高さに締める。ボトルの頭
    # （lay["bottle_top"]）より上にあることを確かめておく。
    cup_z0 = lay["nozzle_tip"] + lay["nozzle_len"] * p["cup_z_x_nozzle_len"]
    cup_ro = lay["nozzle_bore"] * p["cup_ro_x_nozzle_bore"]
    cup_ri = lay["nozzle_bore"] * p["cup_ri_x_nozzle_bore"]
    cup_floor = plate_t * p["cup_floor_x_plate_t"]
    cup_h = lay["nozzle_bore"] * p["cup_h_x_nozzle_bore"]
    if cup_z0 < lay["bottle_top"]:
        _warn(f"ノズルの滴受け皿の底 {cup_z0:.1f} がボトルの頭"
              f" {lay['bottle_top']:.1f} より下。通るボトルに当たる")
    # 落とし管。皿の外壁から半径方向へ出し、支柱と安全カバーの間の帯を
    # 降りて、液受け溝の始まりのすぐ先へ立ち上がりの上から落とす。
    # 半径を一定に保って回る（弦で結ぶと支柱を斜めに串刺しにする）。
    cup_drain_r = gut_ro * p["cup_drain_r_x_gutter_ro"]
    if cup_drain_r - tube_d / 2.0 <= post_x + post_r:
        _warn(f"滴受けの落とし管の帯 {cup_drain_r:.1f} がノズルの支柱"
              f"（外面 {post_x + post_r:.1f}）に掛かる")
    cover_ri = lay.get("cover_r")
    if cover_ri is not None and cup_drain_r + tube_d / 2.0 >= float(cover_ri):
        _warn(f"滴受けの落とし管の帯 {cup_drain_r:.1f} が安全カバーの板"
              f"（内面 {float(cover_ri):.1f}）を貫く")
    gut_mid_r = (gut_ri + gut_ro) / 2.0
    gut_top_z = deck_top + plate_t * (p["gutter_floor_x_plate_t"]
                                      + p["gutter_h_x_plate_t"])
    over_z = gut_top_z + plate_t * p["cup_drain_over_x_plate_t"]
    cup_mid_z = cup_z0 + cup_floor + cup_h * 0.5
    # 降りる位置は充填ステーションから振る。振らないと支柱と空気弁の箱を
    # 串刺しにする（皿から真っすぐ半径方向へ出すと支柱の芯を通る）。
    down_deg = fill_deg - p["cup_drain_side_deg"]
    into_deg = gut_from + p["cup_drain_at_deg"]
    cup_xy = np.array(_polar(fill_deg, lay["pitch_r"]))
    down_xy = np.array(_polar(down_deg, cup_drain_r))
    step = down_xy - cup_xy
    step = step / max(float(np.hypot(*step)), 1e-9)
    start_xy = cup_xy + step * cup_ro
    # 皿から降り口までの一本が支柱をよけているか（芯からの最短距離）
    post_xy = np.array(_polar(fill_deg, post_x))
    seg = down_xy - start_xy
    t = float(np.clip(np.dot(post_xy - start_xy, seg) / np.dot(seg, seg), 0.0, 1.0))
    gap = float(np.hypot(*(start_xy + seg * t - post_xy))) - post_r - tube_d / 2.0
    if gap < 0.0:
        _warn(f"滴受けの落とし管がノズルの支柱に {-gap:.1f} 食い込む。"
              "cup_drain_side_deg を大きくすること")
    n_turn = max(int(abs(_short_turn(down_deg, into_deg))
                     / p["cup_drain_seg_deg"]), 1)
    cup_drain_pts = (
        (tuple(start_xy) + (cup_mid_z,), tuple(down_xy) + (cup_mid_z,),
         tuple(down_xy) + (over_z,))
        + tuple(_polar(down_deg + _short_turn(down_deg, into_deg) * i / n_turn,
                       cup_drain_r) + (over_z,)
                for i in range(1, n_turn + 1))
        + (_polar(into_deg, gut_mid_r) + (over_z,),
           _polar(into_deg, gut_mid_r)
           + (deck_top + plate_t * (p["gutter_floor_x_plate_t"]
                                    + p["cup_drain_in_x_plate_t"]),))
    )

    return {
        # 架台から読み直した値（他でも使えるように出しておく）
        # 脚の位置は基盤側の lay["frame_leg_xy"] をそのまま使う（世界座標）。
        "det_leg_xy": leg_xy,
        "det_frame_spin": leg_phase,
        "det_frame_face": face,
        "det_foot_h": foot_h,
        "det_rail_low_z": rail_low_z,
        "det_rail_high_z": rail_high_z,

        # アジャスタ脚（丸座とロックナットだけ。ねじ軸は架台側の丸柱）
        "det_foot_pad_d": (pipe * p["foot_pad_d_x_frame_pipe"],
                           pipe * p["foot_pad_top_x_frame_pipe"]),
        "det_foot_pad_h": pipe * p["foot_pad_h_x_frame_pipe"],
        "det_foot_nut_d": pipe * p["foot_nut_x_frame_pipe"],
        "det_foot_nut_h": pipe * p["foot_nut_h_x_frame_pipe"],

        # 架台の板金
        "det_skirt_x": skirt_x,
        "det_skirt_size": (skirt_t, lay["frame_span"] + pipe,
                           skirt_z[1] - skirt_z[0]),
        "det_skirt_z": (skirt_z[0] + skirt_z[1]) / 2.0,
        "det_pan_size": (pan_span, pan_span, pan_t),
        "det_pan_z": rail_low_top - pan_t / 2.0,

        # 制御箱
        "det_cab_size": (cab_d, cab_w, cab_h),
        "det_cab_center": cab_center,
        "det_cab_spin": -90.0,                  # 扉（+x）を -y へ向ける

        # 操作盤（局所座標。det_panel_spin で世界へ回す）
        "det_panel_spin": panel_deg,
        "det_panel_size": (pan_dp, pan_w, pan_h),
        "det_panel_center": (pan_cx, pan_cy, pan_cz),
        "det_panel_door_x": door_x,
        "det_stand_post": stand_post,
        "det_stand_post_z": (stand_base_t, pan_bottom),
        "det_stand_base": (plate_d * p["stand_base_x_plate_d"], stand_base_t),

        # シグナルタワー（局所座標）
        "det_tower_stem": (plate_d * p["tower_stem_d_x_plate_d"], tower_z),
        "det_tower_lamp": (plate_d * p["tower_lamp_d_x_plate_d"], tower_lamp_h),
        "det_tower_cap": plate_d * p["tower_cap_h_x_plate_d"],

        # 非常停止と操作器具（局所座標）
        "det_estop_base_d": estop_base,
        "det_estop_button_d": estop_base * p["estop_button_x_estop_base"],
        "det_estop_at": (pan_cy + pan_w * p["estop_at_x_panel_w"],
                         pan_cz + pan_h * p["estop_at_x_panel_h"]),
        "det_lamp_d": estop_base * p["lamp_d_x_estop_base"],
        "det_lamp_out": estop_base * p["lamp_out_x_estop_base"],
        "det_lamp_at": (pan_cy - pan_w * p["op_left_x_panel_w"],
                        pan_cy - pan_w * p["op_right_x_panel_w"],
                        pan_cz + pan_h * p["lamp_at_x_panel_h"]),
        "det_knob_d": estop_base * p["knob_d_x_estop_base"],
        "det_knob_out": estop_base * p["knob_out_x_estop_base"],
        "det_stop_d": estop_base * p["stop_d_x_estop_base"],
        "det_stop_out": estop_base * p["stop_out_x_estop_base"],
        "det_op_at": (pan_cy - pan_w * p["op_left_x_panel_w"],
                      pan_cy - pan_w * p["op_right_x_panel_w"],
                      pan_cz + pan_h * p["op_low_x_panel_h"]),
        "det_label_size": estop_base * p["label_x_estop_base"],
        "det_label_at": (pan_cy + pan_w * p["estop_at_x_panel_w"],
                         pan_cz - pan_h * p["label_at_x_panel_h"]),

        # 配線ダクト。下段は架台の局所座標（det_frame_spin）、横と
        # 立ち上がりは操作盤の局所座標（det_panel_spin）で持つ。
        "det_duct_size": (duct, duct),
        "det_duct_wall": duct * p["duct_wall_x_duct"],
        "det_duct_low": (duct_y, rail_low_z, lay["frame_span"]),
        "det_duct_cross": (cross_x, riser_x, pan_cy, rail_low_z),
        "det_duct_riser": (riser_x, pan_cy, rail_low_z, pan_bottom),

        # 液受け溝（世界座標。角度は充填ステーションからの相対）
        "det_gutter_r": (gut_ri, gut_ro),
        "det_gutter_deg": (gut_from, gut_to),
        "det_gutter_end_deg": p["gutter_end_deg"],
        "det_gutter_z": (deck_top, plate_t * p["gutter_floor_x_plate_t"],
                         plate_t * p["gutter_h_x_plate_t"],
                         plate_t * p["gutter_wall_x_plate_t"]),
        "det_gutter_mouth": (gut_mouth_ri, gut_mouth_to),
        "det_gutter_drain": (gut_drain_xy[0], gut_drain_xy[1],
                             plate_t * p["gutter_drain_d_x_plate_t"],
                             deck_bottom - plate_d * p["gutter_drain_h_x_plate_d"]),
        "det_gutter_valve": (plate_t * p["gutter_valve_d_x_plate_t"],
                             plate_t * p["gutter_valve_h_x_plate_t"]),

        # センタカバー（世界座標）。段付きのハブキャップ。
        # どの段も中空の輪で、内半径は回るハブの外。det_center_step1 =
        # (外半径, 内半径, 下端, 丈)、det_center_step2 も同じ並び。
        "det_center_step1": (
            plate_d * p["center_d_x_plate_d"] / 2.0,
            hub_r * p["center_step1_ri_x_hub_r"],
            center_z0, plate_t * p["center_step1_h_x_plate_t"]),
        "det_center_step2": (
            hub_r * p["center_step1_ri_x_hub_r"],
            lay["column_r"] * p["center_top_x_column_r"],
            center_z0 + plate_t * p["center_step1_h_x_plate_t"],
            plate_t * p["center_step2_h_x_plate_t"]),
        "det_center_collar": (lay["column_r"] * p["center_collar_x_column_r"],
                              lay["column_r"] * p["center_top_x_column_r"],
                              center_z1, plate_t * p["center_collar_h_x_plate_t"]),
        "det_center_bolts": (p["center_bolts"],
                             (hub_r * p["center_step1_ri_x_hub_r"]
                              + plate_d * p["center_d_x_plate_d"] / 2.0) / 2.0,
                             plate_t * p["center_bolt_d_x_plate_t"],
                             plate_t * p["center_bolt_h_x_plate_t"],
                             center_z0 + plate_t * p["center_step1_h_x_plate_t"]),
        "det_center_knobs": (p["center_knobs"],
                             hub_r * p["center_step1_ri_x_hub_r"]
                             * p["center_knob_at_x_step2"],
                             plate_t * p["center_knob_d_x_plate_t"],
                             plate_t * p["center_knob_h_x_plate_t"],
                             p["center_knob_cap_x_knob_d"], center_z1),

        # 銘板（架台の局所座標。上桁の -y 面へ貼る。作業者の立つ 270 度側）
        "det_plate_size": (plate_d * p["plate_w_x_plate_d"],
                           plate_d * p["plate_h_x_plate_d"],
                           plate_t * p["plate_t_x_plate_t"]),
        "det_plate_at": (lay["frame_span"] * p["plate_at_x_frame_span"],
                         -(face + plate_t * p["plate_t_x_plate_t"] / 2.0),
                         rail_high_z),

        # タンクとポンプ（局所座標。det_tank_spin で世界へ回す）
        "det_tank_spin": tank_deg,
        "det_tank_at_r": tank_at_r,
        "det_tank_d": tank_d,
        "det_tank_z": (tank_base, tank_base + tank_h),
        "det_tank_dome": (tank_d * p["tank_dome_h_x_tank_d"],
                          tank_d * p["tank_neck_d_x_tank_d"],
                          tank_d * p["tank_neck_h_x_tank_d"]),
        "det_tank_cone": (tank_cone_h, tank_d * p["tank_cone_d_x_tank_d"]),
        "det_tank_leg": (pipe * p["tank_leg_x_frame_pipe"],
                         tank_d * p["tank_leg_at_x_tank_d"], stand_top, tank_base),
        "det_stand_frame": (tank_d * p["stand_span_x_tank_d"], stand_top,
                            pipe * p["stand_pipe_x_frame_pipe"],
                            tank_d * p["stand_deck_d_x_tank_d"],
                            plate_t * p["stand_deck_t_x_plate_t"]),
        # タンク架台とポンプ棚の脚のアジャスタ。機械側の脚と同じ作りにする
        # （ここだけ床でぶつ切りだと、脚の高さが揃っていないように見える）。
        # (丸座の径, 同 上面の径, 同 厚み, ナットの径, 同 厚み, 丈, 丸柱の半径)
        "det_stand_foot": (stand_pipe * p["foot_pad_d_x_frame_pipe"],
                           stand_pipe * p["foot_pad_top_x_frame_pipe"],
                           stand_pipe * p["foot_pad_h_x_frame_pipe"],
                           stand_pipe * p["foot_nut_x_frame_pipe"],
                           stand_pipe * p["foot_nut_h_x_frame_pipe"],
                           stand_foot_h,
                           stand_pipe * parts.FRAME_FOOT_R_X_PIPE),
        "det_stand_foot_xy": stand_foot_xy,     # 丸柱は parts.frame が作る
        "det_shelf_foot_xy": shelf_foot_xy,     # こちらは丸柱から作る
        "det_shelf_foot_h": stand_foot_h,
        "det_gauge": (pipe_d * p["gauge_d_x_pipe_d"],
                      tank_base + tank_h * p["gauge_z0_x_tank_h"],
                      tank_base + tank_h * p["gauge_z1_x_tank_h"],
                      tank_d / 2.0),
        # 投入口のふた・ベント・出口弁
        "det_tank_lid": (tank_d * p["tank_neck_d_x_tank_d"]
                         * p["tank_lid_d_x_neck_d"] / 2.0,
                         tank_d * p["tank_lid_h_x_tank_d"]),
        "det_tank_vent": (pipe_d * p["tank_vent_d_x_pipe_d"] / 2.0,
                          tank_d * p["tank_vent_h_x_tank_d"],
                          tank_d * p["tank_vent_at_x_tank_d"],
                          p["tank_vent_cap_x_vent_d"]),
        "det_tank_valve": (pipe_d * p["tank_valve_d_x_pipe_d"] / 2.0,
                           pipe_d * p["tank_valve_h_x_pipe_d"],
                           tuple(pipe_d * v for v in p["tank_valve_lever_x_pipe_d"])),

        "det_shelf_size": (plate_r * p["pump_shelf_ri_x_plate_r"],
                           plate_r * p["pump_shelf_ro_x_plate_r"],
                           tank_d * p["pump_shelf_w_x_tank_d"], shelf_t),
        "det_shelf_top": shelf_top,
        "det_shelf_leg": (plate_r * p["pump_leg_at_r_x_plate_r"],
                          pipe * p["pump_leg_x_frame_pipe"],
                          tank_d * p["pump_shelf_w_x_tank_d"] / 2.0
                          - pipe * p["pump_leg_x_frame_pipe"] * 1.2),
        "det_pump_base": (tank_d * p["pump_base_len_x_tank_d"],
                          tank_d * p["pump_base_w_x_tank_d"], pump_base_t),
        "det_pump_body": (pump_body_d, tank_d * p["pump_body_len_x_tank_d"],
                          plate_r * p["pump_body_at_r_x_plate_r"]),
        "det_pump_head": (pump_head_d, tank_d * p["pump_head_len_x_tank_d"],
                          plate_r * p["pump_head_at_r_x_plate_r"]),
        "det_pump_axis_z": pump_axis_z,
        "det_pump_stub": (pipe_d * p["pump_stub_d_x_pipe_d"], stub_h,
                          pump_head_top, pump_in_r, pump_out_r),
        "det_pump_gauge": (tank_d * p["pump_gauge_d_x_tank_d"],
                           plate_t * p["pump_gauge_t_x_plate_t"],
                           pipe_d * p["pump_gauge_h_x_pipe_d"]),

        # 配管（世界座標の点列）
        "det_pipe_d": pipe_d,
        "det_tube_d": tube_d,
        "det_feed_pts": feed_pts,
        "det_supply_pts": supply_pts,
        "det_pipe_fitting": _spin_pts(
            ((lay["pitch_r"], (pipe_y - collar_r * 0.9) / 2.0, collar_z),),
            fill_deg)[0],
        "det_tube_pts": tube_pts,
        "det_tube_out_pts": tube_out_pts,
        "det_hanger": (hanger_degs,
                       plate_r * p["hanger_ri_x_plate_r"], run_r,
                       pipe * p["hanger_w_x_frame_pipe"],
                       plate_t * p["hanger_t_x_plate_t"],
                       run_z - pipe_d / 2.0, deck_bottom),

        # 天板を貫くものの座（世界座標。液配管・エアチューブ・排液口）。
        # (x, y, 座の径, 下端, 上端) の並び。
        "det_bulkheads": tuple(
            _spin_pts(((post_x, y, 0.0),), fill_deg)[0][:2]
            + (d * p["bulkhead_d_x_pipe_d"],
               deck_bottom - d * p["bulkhead_down_x_pipe_d"],
               deck_top + d * p["bulkhead_up_x_pipe_d"])
            for y, d in ((pipe_y, pipe_d), (tube_y, tube_d))),
        "det_drain_bulkhead": (
            gut_drain_xy[0], gut_drain_xy[1],
            plate_t * p["gutter_drain_d_x_plate_t"] * p["bulkhead_d_x_pipe_d"],
            deck_bottom - plate_t * p["gutter_drain_d_x_plate_t"]
            * p["bulkhead_down_x_pipe_d"],
            deck_top + plate_t * p["gutter_floor_x_plate_t"]),

        # 充填部（局所座標。det_fill_spin で世界へ回す）
        "det_fill_spin": fill_deg,
        "det_clamp_size": clamp,
        "det_clamp_z": clamp_z,
        "det_clamp_at": (post_x, pipe_y, tube_y),
        "det_valve_size": (vw, vd, vh),
        "det_valve_center": (valve_cx, tube_y, valve_cz),

        # ノズルの滴受け（局所座標。det_fill_spin で世界へ回す）と落とし管
        "det_cup": (lay["pitch_r"], cup_ro, cup_ri, cup_z0, cup_floor, cup_h,
                    plate_t * p["cup_wall_x_plate_t"]),
        "det_cup_drain_pts": cup_drain_pts,
    }


# --------------------------------------------------------------------------
# 組み立て
# --------------------------------------------------------------------------
def _face_matrix(x, y, z, spin_deg=0.0):
    """z 方向に組んだ回転体を「+x 向き」に倒して (x, y, z) へ運ぶ変換。

    倒して局所座標 (x, y, z) へ置いてから、spin_deg だけ z 軸まわりに回す。
    平行移動そのものも回るので、扉に付ける押しボタンのように「外を向いた面から
    生える」部品を、箱と同じ局所座標のまま置ける。
    """
    local = parts.transform_matrix(translate=(x, y, z), rot_y_deg=90.0)
    return parts.transform_matrix(rot_z_deg=spin_deg) @ local


def build(params: dict, lay: dict) -> dict:
    """小物のメッシュ群。キーは DRAW_ORDER に載せた名前だけ。"""
    g = {k: [] for k, _ in DRAW_ORDER}

    # ---------------------------------------------------------------- 脚
    # 架台側（parts.frame）が脚の下に丸柱を作っている。これをねじ軸に見立て、
    # 床に当たる丸座とロックナットだけを足す。二重に軸を立てない。
    pad_d, pad_top_d = lay["det_foot_pad_d"]
    pad_h = lay["det_foot_pad_h"]
    nut_h = lay["det_foot_nut_h"]
    for (x, y) in lay["det_leg_xy"]:
        at = parts.transform_matrix(translate=(x, y, 0.0))
        g["det_adjuster"].append(
            parts.cone_frustum(pad_d / 2.0, pad_top_d / 2.0, pad_h, base_z=0.0,
                               resolution=parts.RES_COARSE, matrix=at))
        g["det_adjuster"].append(
            parts.cylinder(lay["det_foot_nut_d"] / 2.0, nut_h,
                           base_z=lay["det_foot_h"] - nut_h,
                           resolution=parts.RES_COARSE, matrix=at))

    # ---------------------------------------------------------------- 板金
    # ここから下、架台に付くものは架台の局所座標で組んで frame_spin で回す。
    # 脚が振れても板金・制御箱・ダクト・銘板がそのまま追従する。
    frame_spin = parts.transform_matrix(rot_z_deg=lay["det_frame_spin"])

    # 腰板は局所 +x と -x の 2 面。+y はギヤモータが張り出す面、-y は中の
    # 制御箱を見せる面なので開けておく。
    sk = lay["det_skirt_size"]
    for sx in (-1.0, 1.0):
        g["det_panel"].append(
            parts.box(sk, center=(sx * lay["det_skirt_x"], 0.0, lay["det_skirt_z"]),
                      matrix=frame_spin))
    g["det_panel"].append(
        parts.box(lay["det_pan_size"], center=(0.0, 0.0, lay["det_pan_z"]),
                  matrix=frame_spin))

    # ---------------------------------------------------------------- 制御箱
    # インバータと PLC の箱。架台の中の底板に載せ、扉を局所 -y へ向ける。
    cab_spin = frame_spin @ parts.transform_matrix(rot_z_deg=lay["det_cab_spin"])
    cw, cd, ch = lay["det_cab_size"]
    ccx, ccy, ccz = lay["det_cab_center"]
    # 局所座標（扉が +x）へ直してから回す
    cab = parts.control_box(size=(cw, cd, ch),
                            center=(-ccy, ccx, ccz), matrix=cab_spin)
    g["det_panel"] += [cab["body"], cab["door"]]
    g["det_panel_dark"] += [cab["seam"], cab["latch"]]

    # ---------------------------------------------------------------- 操作盤
    spin = lay["det_panel_spin"]
    to_world = parts.transform_matrix(rot_z_deg=spin)
    pan_d, pan_w, pan_h = lay["det_panel_size"]
    cx, cy, cz = lay["det_panel_center"]
    door_x = lay["det_panel_door_x"]

    cab2 = parts.control_box(size=(pan_d, pan_w, pan_h),
                             center=(cx, cy, cz), matrix=to_world)
    g["det_panel"] += [cab2["body"], cab2["door"]]
    g["det_panel_dark"] += [cab2["seam"], cab2["latch"]]

    # 床置きの柱と据付板。箱が天板より高いので脚の角には抱かせられない
    post_a = lay["det_stand_post"]
    pz0, pz1 = lay["det_stand_post_z"]
    base_a, base_t = lay["det_stand_base"]
    g["det_stand"].append(
        parts.box((post_a, post_a, pz1 - pz0),
                  center=(cx, cy, (pz0 + pz1) / 2.0), matrix=to_world))
    g["det_stand"].append(
        parts.box((base_a, base_a, base_t),
                  center=(cx, cy, base_t / 2.0), matrix=to_world))

    # 積層シグナルタワー。実機写真でいちばん目立つ部品のひとつ
    stem_d, (tz0, tz1) = lay["det_tower_stem"]
    lamp_d, lamp_h = lay["det_tower_lamp"]
    at_tower = parts.transform_matrix(translate=(cx, cy, 0.0))
    g["det_panel_dark"].append(
        parts.cylinder(stem_d / 2.0, tz1 - tz0, base_z=tz0,
                       resolution=parts.RES_COARSE,
                       matrix=to_world @ at_tower))
    # 下から 緑（運転）・黄（注意）・赤（停止）。赤が上
    for i, key in enumerate(("det_lamp", "det_yellow", "det_red")):
        g[key].append(
            parts.cylinder(lamp_d / 2.0, lamp_h, base_z=tz1 + lamp_h * i,
                           resolution=parts.RES_COARSE,
                           matrix=to_world @ at_tower))
    g["det_panel_dark"].append(
        parts.cone_frustum(lamp_d / 2.0, lamp_d * 0.34, lay["det_tower_cap"],
                           base_z=tz1 + lamp_h * 3.0,
                           resolution=parts.RES_COARSE,
                           matrix=to_world @ at_tower))

    # 非常停止。扉の面から赤いきのこが出る
    es_y, es_z = lay["det_estop_at"]
    est = parts.estop_button(base_d=lay["det_estop_base_d"],
                             button_d=lay["det_estop_button_d"],
                             matrix=_face_matrix(door_x, es_y, es_z, spin))
    g["det_yellow"].append(est["base"])
    g["det_red"].append(est["button"])

    # 表示灯 2 つ。運転（緑）と異常（黄）
    lamp_in, lamp_out_y, lamp_z = lay["det_lamp_at"]
    lamp_r = lay["det_lamp_d"] / 2.0
    lamp_len = lay["det_lamp_out"]
    for key, ly in (("det_lamp", lamp_in), ("det_yellow", lamp_out_y)):
        g[key].append(
            parts.cylinder(lamp_r, lamp_len, base_z=0.0,
                           resolution=parts.RES_COARSE,
                           matrix=_face_matrix(door_x, ly, lamp_z, spin)))

    # 運転停止の押しボタンと速度つまみ
    op_in, op_out_y, op_z = lay["det_op_at"]
    g["det_panel_dark"].append(
        parts.cylinder(lay["det_stop_d"] / 2.0, lay["det_stop_out"], base_z=0.0,
                       resolution=parts.RES_COARSE,
                       matrix=_face_matrix(door_x, op_in, op_z, spin)))
    g["det_panel_dark"].append(
        parts.cone_frustum(lay["det_knob_d"] / 2.0, lay["det_knob_d"] * 0.35,
                           lay["det_knob_out"], base_z=0.0,
                           resolution=parts.RES_COARSE,
                           matrix=_face_matrix(door_x, op_out_y, op_z, spin)))

    # 警告ラベル。扉の下側に 1 枚
    lb = lay["det_label_size"]
    lb_y, lb_z = lay["det_label_at"]
    lb_t = lb * 0.04
    g["det_yellow"].append(
        parts.box((lb_t, lb, lb),
                  center=(door_x + lb_t / 2.0, lb_y, lb_z), matrix=to_world))

    # ---------------------------------------------------------------- ダクト
    duct_size = lay["det_duct_size"]
    duct_wall = lay["det_duct_wall"]

    def _duct(length, axis, center, matrix):
        d = parts.cable_duct(length, size=duct_size, axis=axis,
                             center=center, wall=duct_wall, matrix=matrix)
        g["det_duct"].append(d["body"])
        g["det_duct_lid"].append(d["lid"])

    # 下段は架台の局所 -y 面に沿って走る（脚と一緒に振れる）
    dy, low_z, low_len = lay["det_duct_low"]
    _duct(low_len, "x", (0.0, dy, low_z), frame_spin)

    # 横へ抜ける区間と立ち上がりは操作盤の局所座標。柱の面まで繋ぐ
    cx0, cx1, cy0, cz0 = lay["det_duct_cross"]
    _duct(abs(cx1 - cx0), "x", ((cx0 + cx1) / 2.0, cy0, cz0), to_world)

    rx, ry, rz0, rz1 = lay["det_duct_riser"]
    _duct(rz1 - rz0, "z", (rx, ry, (rz0 + rz1) / 2.0), to_world)

    # ---------------------------------------------------------------- 液受け
    # テーブル外周の環状の溝。底板と内外の立ち上がり、端を塞ぐ板、排液口。
    # 充填ステーションの側では内壁をテーブルの縁の近くまで寄せ（受け口）、
    # ノズルの下でテーブルに落ちた滴を、テーブルが送る先で受ける。
    gi, go = lay["det_gutter_r"]
    ga0, ga1 = lay["det_gutter_deg"]
    gz0, floor_t, wall_h, wall_t = lay["det_gutter_z"]
    ge = lay["det_gutter_end_deg"]
    mouth_ri, mouth_a1 = lay["det_gutter_mouth"]
    has_mouth = mouth_a1 > ga0 + 1e-6 and mouth_ri < gi

    g["det_tray"].append(parts.crescent_guide(gi, go, ga0, ga1, floor_t, base_z=gz0))
    g["det_tray"].append(
        parts.crescent_guide(go - wall_t, go, ga0, ga1, wall_h, base_z=gz0 + floor_t))
    if has_mouth:
        # 受け口の底と内壁。溝の内壁はこの区間だけ内側へ移る（同じ所に
        # 2 枚立てると受け口を堰き止める）
        g["det_tray"].append(
            parts.crescent_guide(mouth_ri, gi, ga0, mouth_a1, floor_t, base_z=gz0))
        g["det_tray"].append(
            parts.crescent_guide(mouth_ri, mouth_ri + wall_t, ga0, mouth_a1,
                                 wall_h, base_z=gz0 + floor_t))
        # 受け口の終わりで内壁を元の半径へ戻す仕切り
        g["det_tray"].append(
            parts.crescent_guide(mouth_ri, gi, mouth_a1 - ge, mouth_a1,
                                 wall_h, base_z=gz0 + floor_t))
        g["det_tray"].append(
            parts.crescent_guide(gi, gi + wall_t, mouth_a1, ga1, wall_h,
                                 base_z=gz0 + floor_t))
    else:
        g["det_tray"].append(
            parts.crescent_guide(gi, gi + wall_t, ga0, ga1, wall_h,
                                 base_z=gz0 + floor_t))
    for a, ri in ((ga0, mouth_ri if has_mouth else gi), (ga1 - ge, gi)):
        g["det_tray"].append(
            parts.crescent_guide(ri, go, a, a + ge, wall_h, base_z=gz0 + floor_t))

    # 排液口。溝の底から天板を抜けて下へ出し、下端に排液弁を付ける
    dr_x, dr_y, dr_d, dr_z = lay["det_gutter_drain"]
    val_d, val_h = lay["det_gutter_valve"]
    at_drain = parts.transform_matrix(translate=(dr_x, dr_y, 0.0))
    g["det_tray"].append(
        parts.cylinder(dr_d / 2.0, gz0 + floor_t - dr_z, base_z=dr_z,
                       resolution=parts.RES_COARSE, matrix=at_drain))
    g["det_pipe_sus"].append(
        parts.cylinder(val_d / 2.0, val_h, base_z=dr_z - val_h,
                       resolution=parts.RES_COARSE, matrix=at_drain))
    g["det_panel_dark"].append(
        parts.horizontal_cylinder(val_d * 0.18, val_d * 1.5, axis="x",
                                  center=(dr_x, dr_y, dr_z - val_h * 0.5),
                                  resolution=parts.RES_COARSE))

    # 天板を貫くものの座。板をまたぐカラーを被せて貫通部の座に見せる
    for bx, by, bd, bz0, bz1 in (tuple(lay["det_bulkheads"])
                                 + (tuple(lay["det_drain_bulkhead"]),)):
        g["det_pipe_sus"].append(
            parts.cylinder(bd / 2.0, bz1 - bz0, base_z=bz0,
                           resolution=parts.RES_COARSE,
                           matrix=parts.transform_matrix(translate=(bx, by, 0.0))))

    # ------------------------------------------------------- センタカバー
    # 中心柱に締める段付きのハブキャップ。大径の段・中径の段・締め座の 3 段。
    # **どの段も中空の輪**で、内半径は回るハブの外へ出してある（中実の円板で
    # 組むと、固定のカバーが回るハブを飲む。カバーが不透明なので絵では
    # 気付けない）。上にボルト円と持ち上げ用のつまみを載せる。
    for key in ("det_center_step1", "det_center_step2", "det_center_collar"):
        ro, ri, z0, h = lay[key]
        g["det_tray"].append(parts.tube(ro, ri, h, base_z=z0))

    n_bolt, bolt_r, bolt_d, bolt_h, bolt_z = lay["det_center_bolts"]
    for i in range(int(n_bolt)):
        bx, by = _polar(360.0 * i / max(int(n_bolt), 1), bolt_r)
        g["det_bolt"].append(
            parts.cylinder(bolt_d / 2.0, bolt_h, base_z=bolt_z,
                           resolution=parts.RES_COARSE,
                           matrix=parts.transform_matrix(translate=(bx, by, 0.0))))

    n_knob, knob_r, knob_d, knob_h, cap_x, knob_z = lay["det_center_knobs"]
    for i in range(int(n_knob)):
        kx, ky = _polar(360.0 * i / max(int(n_knob), 1), knob_r)
        at_knob = parts.transform_matrix(translate=(kx, ky, 0.0))
        g["det_bolt"].append(
            parts.cylinder(knob_d / 2.0, knob_h, base_z=knob_z,
                           resolution=parts.RES_COARSE, matrix=at_knob))
        g["det_bolt"].append(
            parts.cone_frustum(knob_d * cap_x / 2.0, knob_d * cap_x * 0.38,
                               knob_d * cap_x * 0.45, base_z=knob_z + knob_h,
                               resolution=parts.RES_COARSE, matrix=at_knob))

    # ---------------------------------------------------------------- 銘板
    # 上桁の -y 面へ貼る。厚み方向（z）を -y へ倒してから架台と一緒に回す
    px, py, pz = lay["det_plate_at"]
    g["det_plate"].append(
        parts.name_plate(size=lay["det_plate_size"],
                         matrix=frame_spin @ parts.transform_matrix(
                             translate=(px, py, pz), rot_x_deg=90.0)))

    # ---------------------------------------------------------------- タンク
    # 据付角は lay["det_tank_spin"]（充填ステーションから +tank_from_fill_deg。
    # 既定諸元では 0 + 45 = 世界角 45 度）。局所座標では +x が機械から外へ向かう。
    tspin = parts.transform_matrix(rot_z_deg=lay["det_tank_spin"])
    t_at = lay["det_tank_at_r"]
    t_d = lay["det_tank_d"]
    t_z0, t_z1 = lay["det_tank_z"]
    dome_h, neck_d, neck_h = lay["det_tank_dome"]
    cone_h, cone_d = lay["det_tank_cone"]
    at_tank = tspin @ parts.transform_matrix(translate=(t_at, 0.0, 0.0))

    g["det_tank"].append(
        parts.cylinder(t_d / 2.0, t_z1 - t_z0, base_z=t_z0, matrix=at_tank))
    g["det_tank"].append(
        parts.cone_frustum(t_d / 2.0, neck_d / 2.0, dome_h, base_z=t_z1,
                           matrix=at_tank))
    g["det_tank"].append(
        parts.cylinder(neck_d / 2.0, neck_h, base_z=t_z1 + dome_h,
                       resolution=parts.RES_COARSE, matrix=at_tank))
    # 下の絞り。ここがタンクの出口で、そのまま配管に繋がる
    g["det_tank"].append(
        parts.cone_frustum(cone_d / 2.0, t_d / 2.0, cone_h, base_z=t_z0 - cone_h,
                           matrix=at_tank))

    # 投入口のふた。開けたまま置かない
    lid_r, lid_h = lay["det_tank_lid"]
    g["det_tank"].append(
        parts.cylinder(lid_r, lid_h, base_z=t_z1 + dome_h + neck_h,
                       resolution=parts.RES_COARSE, matrix=at_tank))
    # ベント（呼吸口）。鏡板に立てて頭に傘を被せる。密閉すると液が落ちない
    vt_r, vt_h, vt_at, vt_cap = lay["det_tank_vent"]
    vt_z = t_z1 + dome_h * (1.0 - vt_at / (t_d / 2.0)) * 0.9
    at_vent = at_tank @ parts.transform_matrix(translate=(vt_at, 0.0, 0.0))
    g["det_pipe_sus"].append(
        parts.cylinder(vt_r, vt_h, base_z=vt_z, resolution=parts.RES_COARSE,
                       matrix=at_vent))
    g["det_pipe_sus"].append(
        parts.cone_frustum(vt_r * vt_cap, vt_r * vt_cap * 0.4, vt_r * vt_cap * 0.7,
                           base_z=vt_z + vt_h, resolution=parts.RES_COARSE,
                           matrix=at_vent))
    # 出口弁。絞りの下に 1 つ。ここを閉めるとポンプを外せる
    vl_r, vl_h, (lv_len, lv_w, lv_t) = lay["det_tank_valve"]
    vl_z = t_z0 - cone_h - vl_h
    g["det_pipe_sus"].append(
        parts.cylinder(vl_r, vl_h, base_z=vl_z, resolution=parts.RES_COARSE,
                       matrix=at_tank))
    g["det_panel_dark"].append(
        parts.box((lv_len, lv_w, lv_t),
                  center=(t_at + vl_r + lv_len / 2.0, 0.0, vl_z + vl_h * 0.5),
                  matrix=tspin))

    # 胴を持ち上げる短い角柱 4 本。絞りの下に配管を通す隙間を作る
    leg_a, leg_at, leg_z0, leg_z1 = lay["det_tank_leg"]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            g["det_stand"].append(
                parts.box((leg_a, leg_a, leg_z1 - leg_z0),
                          center=(t_at + sx * leg_at, sy * leg_at,
                                  (leg_z0 + leg_z1) / 2.0),
                          matrix=tspin))

    # タンク架台。機械の架台と同じ組み方（角パイプ 4 本＋桁＋丸天板）
    st_span, st_top, st_pipe, st_deck_d, st_deck_t = lay["det_stand_frame"]
    stand = parts.frame(span=st_span, deck_top_z=st_top, pipe=st_pipe,
                        deck_diameter=st_deck_d, deck_thickness=st_deck_t,
                        matrix=at_tank)
    g["det_stand"] += [stand["legs"], stand["rails"], stand["deck"]]
    g["det_adjuster"].append(stand["feet"])

    # タンク架台とポンプ棚の脚にもアジャスタ（丸座＋ロックナット）を付ける。
    # 機械側の脚だけ丸座があって、こちらが床でぶつ切りだと不揃いが目立つ。
    (st_pad_d, st_pad_top_d, st_pad_h,
     st_nut_d, st_nut_h, st_foot_h, st_post_r) = lay["det_stand_foot"]
    for (x, y) in tuple(lay["det_stand_foot_xy"]) + tuple(lay["det_shelf_foot_xy"]):
        at = parts.transform_matrix(translate=(x, y, 0.0))
        g["det_adjuster"].append(
            parts.cone_frustum(st_pad_d / 2.0, st_pad_top_d / 2.0, st_pad_h,
                               base_z=0.0, resolution=parts.RES_COARSE, matrix=at))
        g["det_adjuster"].append(
            parts.cylinder(st_nut_d / 2.0, st_nut_h, base_z=st_foot_h - st_nut_h,
                           resolution=parts.RES_COARSE, matrix=at))
    # ポンプ棚の脚は角柱で床まで来ていたので、丸柱（ねじ軸）に載せ替える
    for (x, y) in lay["det_shelf_foot_xy"]:
        g["det_adjuster"].append(
            parts.cylinder(st_post_r, st_foot_h, base_z=0.0,
                           resolution=parts.RES_COARSE,
                           matrix=parts.transform_matrix(translate=(x, y, 0.0))))

    # 液面計。胴の機械寄りの面に沿う細い管と、上下の取り出し
    gg_d, gg_z0, gg_z1, gg_r = lay["det_gauge"]
    gg_x = t_at - gg_r - gg_d
    g["det_tube_air"].append(
        parts.cylinder(gg_d / 2.0, gg_z1 - gg_z0, base_z=gg_z0,
                       resolution=parts.RES_COARSE,
                       matrix=tspin @ parts.transform_matrix(translate=(gg_x, 0.0, 0.0))))
    for z in (gg_z0, gg_z1):
        g["det_pipe_sus"].append(
            parts.horizontal_cylinder(gg_d * 0.62, gg_d * 2.6, axis="x",
                                      center=(gg_x + gg_d * 0.9, 0.0, z),
                                      resolution=parts.RES_COARSE, matrix=tspin))

    # ---------------------------------------------------------------- ポンプ
    sh_ri, sh_ro, sh_w, sh_t = lay["det_shelf_size"]
    sh_top = lay["det_shelf_top"]
    g["det_stand"].append(
        parts.box((sh_ro - sh_ri, sh_w, sh_t),
                  center=((sh_ri + sh_ro) / 2.0, 0.0, sh_top - sh_t / 2.0),
                  matrix=tspin))
    lg_r, lg_a, lg_y = lay["det_shelf_leg"]
    lg_foot_h = lay["det_shelf_foot_h"]
    for sy in (-1.0, 1.0):
        g["det_stand"].append(
            parts.box((lg_a, lg_a, sh_top - sh_t - lg_foot_h),
                      center=(lg_r, sy * lg_y,
                              (sh_top - sh_t + lg_foot_h) / 2.0),
                      matrix=tspin))

    pb_len, pb_w, pb_t = lay["det_pump_base"]
    body_d, body_len, body_r = lay["det_pump_body"]
    head_d, head_len, head_r = lay["det_pump_head"]
    ax_z = lay["det_pump_axis_z"]
    g["det_stand"].append(
        parts.box((pb_len, pb_w, pb_t), center=(body_r, 0.0, sh_top + pb_t / 2.0),
                  matrix=tspin))
    g["det_pump"].append(
        parts.horizontal_cylinder(body_d / 2.0, body_len, axis="x",
                                  center=(body_r, 0.0, ax_z), matrix=tspin))
    g["det_pump"].append(
        parts.horizontal_cylinder(head_d / 2.0, head_len, axis="x",
                                  center=(head_r, 0.0, ax_z), matrix=tspin))
    # 冷却フィンの代わりに胴へ細い輪を数本
    for i in range(4):
        g["det_pump"].append(
            parts.horizontal_cylinder(body_d * 0.54, body_len * 0.05, axis="x",
                                      center=(body_r + body_len * (0.10 * i - 0.15),
                                              0.0, ax_z),
                                      resolution=parts.RES_COARSE, matrix=tspin))

    stub_d, stub_h, head_top, in_r, out_r = lay["det_pump_stub"]
    for r in (in_r, out_r):
        g["det_pipe_sus"].append(
            parts.cylinder(stub_d / 2.0, stub_h, base_z=head_top,
                           resolution=parts.RES_COARSE,
                           matrix=tspin @ parts.transform_matrix(translate=(r, 0.0, 0.0))))
    # 吐出しの圧力計。立ち上がりの管から横へ出す（同軸に置くと管の途中の
    # つばに見える）
    pg_d, pg_t, pg_h = lay["det_pump_gauge"]
    pg_z = head_top + stub_h + pg_h * 2.0
    pg_y = pg_d * 0.55
    g["det_pipe_sus"].append(
        parts.horizontal_cylinder(stub_d * 0.26, pg_y, axis="y",
                                  center=(out_r, pg_y / 2.0, pg_z),
                                  resolution=parts.RES_COARSE, matrix=tspin))
    g["det_plate"].append(
        parts.horizontal_cylinder(pg_d / 2.0, pg_t, axis="y",
                                  center=(out_r, pg_y + pg_t / 2.0, pg_z),
                                  resolution=parts.RES_COARSE, matrix=tspin))

    # ---------------------------------------------------------------- 配管
    # タンク -> ポンプ -> 天板の下を回り込み -> 支柱 -> ノズル。
    # どこも宙で切らずに繋がっている。
    g["det_pipe_sus"].append(
        parts.pipe_run(lay["det_feed_pts"], lay["det_pipe_d"]))
    g["det_pipe_sus"].append(
        parts.pipe_run(lay["det_supply_pts"], lay["det_pipe_d"]))
    # ノズルのカラーへ差した継手。管が刺さっているだけに見えないように太らせる
    fx, fy, fz = lay["det_pipe_fitting"]
    g["det_pipe_sus"].append(
        parts.horizontal_cylinder(lay["det_pipe_d"] * 0.70, lay["det_pipe_d"] * 1.1,
                                  axis="y", center=(fx, fy, fz),
                                  resolution=parts.RES_COARSE,
                                  matrix=parts.transform_matrix(
                                      rot_z_deg=lay["det_fill_spin"])))
    g["det_tube_air"].append(
        parts.pipe_run(lay["det_tube_pts"], lay["det_tube_d"]))
    g["det_tube_air"].append(
        parts.pipe_run(lay["det_tube_out_pts"], lay["det_tube_d"]))

    # 回り込みを受ける腕木。天板の下面から吊って配管を載せる
    h_degs, h_ri, h_ro, h_w, h_t, h_z, h_deck = lay["det_hanger"]
    for a in h_degs:
        hang = parts.transform_matrix(rot_z_deg=a)
        g["det_stand"].append(
            parts.box((h_ro - h_ri + h_w, h_w, h_t),
                      center=((h_ri + h_ro) / 2.0, 0.0, h_z - h_t / 2.0),
                      matrix=hang))
        g["det_stand"].append(
            parts.box((h_t, h_w, h_deck - h_z + h_t),
                      center=(h_ri + h_t / 2.0, 0.0, (h_z + h_deck) / 2.0),
                      matrix=hang))

    # 支柱と配管を挟むクランプ。管が空中を通っていないことを見せる
    fspin = parts.transform_matrix(rot_z_deg=lay["det_fill_spin"])
    cl = lay["det_clamp_size"]
    post_x, pipe_y, tube_y = lay["det_clamp_at"]
    for i, z in enumerate(lay["det_clamp_z"]):
        y = pipe_y if i % 2 == 0 else tube_y
        g["det_panel_dark"].append(
            parts.box((cl, cl * 1.6, cl * 0.55),
                      center=(post_x, y * 0.72, z), matrix=fspin))

    # 支柱の外側に付く空気弁の箱。エアチューブの行き先
    g["det_panel_dark"].append(
        parts.box(lay["det_valve_size"], center=lay["det_valve_center"],
                  matrix=fspin))

    # ノズルの滴受け皿と落とし管。ノズルの胴に締めた輪で、胴を伝って落ちる
    # 液をここで止める。溜まった液は支柱の脇を降りて液受け溝へ落とす。
    # （ノズルの真下には置けない。そこは回るテーブルの通り道で、ボトルの頭と
    #   ノズル先端の間は 12 しか無い。先端から垂れる滴はテーブルに落ち、
    #   テーブルが送る先の受け口で受ける。）
    cup_at_r, cup_ro, cup_ri, cup_z0, cup_floor, cup_h, cup_wall = lay["det_cup"]
    at_cup = fspin @ parts.transform_matrix(translate=(cup_at_r, 0.0, 0.0))
    g["det_tray"].append(parts.tube(cup_ro, cup_ri, cup_floor, base_z=cup_z0,
                                    resolution=parts.RES_COARSE, matrix=at_cup))
    g["det_tray"].append(
        parts.tube(cup_ro, cup_ro - cup_wall, cup_h, base_z=cup_z0 + cup_floor,
                   resolution=parts.RES_COARSE, matrix=at_cup))
    g["det_pipe_sus"].append(
        parts.pipe_run(lay["det_cup_drain_pts"], lay["det_tube_d"]))

    return {k: parts.merge(v) for k, v in g.items() if v}


# --------------------------------------------------------------------------
# 目で見て確かめる
# --------------------------------------------------------------------------
def _merge_groups(*pieces):
    """メッシュ群の辞書をいくつか合流させる。同じ名前は結合する。

    scene.build_static() と scene.build_carousel() はどちらも "steel" を返す。
    dict.update で重ねると静止側が丸ごと消えるので、必ずここを通す。
    scene 側に同じ働きのヘルパがあればそちらを使い、これは控え。
    """
    groups: dict = {}
    for piece in pieces:
        for name, mesh in piece.items():
            groups.setdefault(name, []).append(mesh)
    return {k: parts.merge(v) for k, v in groups.items() if v}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="小物を機械に足して焼く")
    ap.add_argument("--out", default="figs", help="PNG の置き場")
    ap.add_argument("--size", type=int, nargs=2, default=(800, 600),
                    metavar=("W", "H"), help="画素数")
    args = ap.parse_args()

    import pyvista as pv

    import cameras
    import scene

    params = scene.load_params()
    lay = scene.derive_layout(params)

    # 静止側と回る側を合流させる。どちらも "steel" を返すので dict.update で
    # 重ねてはいけない（静止側の固定ノズル・支柱とアーム・旋回軸受リング・
    # 中心柱が丸ごと消える）。基盤側の合成ヘルパがあればそれを使う。
    combine = getattr(scene, "merge_groups", None)
    if not callable(combine):
        combine = _merge_groups
    meshes = combine(scene.build_static(params, lay, with_floor=True),
                     scene.build_carousel(params, lay))

    material = dict(scene.MATERIAL)
    order = list(scene.DRAW_ORDER)
    if "det_panel_size" not in lay:
        # scene がこのモジュールを読み込めていない（書きかけの取り違え）。
        # 自分の分だけ手で合流させて、絵は焼けるようにしておく。
        print("[asm_details] scene 側が読み込んでいない。手で合流させる")
        lay.update(layout(params, lay))
        meshes = combine(meshes, build(params, lay))
        material.update(MATERIALS)
        order += list(DRAW_ORDER)

    # 画角は機械の実測外形に合わせる。タンクを外へ足したので、これを通さないと
    # 枠から外れる（基盤側が cameras.machine_radius をここへ繋いである）。
    if hasattr(scene, "ensure_extent"):
        scene.ensure_extent(lay, meshes, params)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def shot(cam, path):
        pl = pv.Plotter(off_screen=True, window_size=list(args.size))
        pl.set_background(scene.BACKGROUND)
        pl.set_environment_texture(scene.studio_cubemap(), is_srgb=True)
        try:
            pl.renderer.GetEnvMapPrefiltered().SetPrefilterMaxSamples(64)
            pl.renderer.GetEnvMapIrradiance().SetIrradianceSize(32)
        except AttributeError:
            pass
        for name, mat in order:
            mesh = meshes.get(name)
            if mesh is None or mesh.n_points == 0:
                continue
            pl.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True,
                        feature_angle=35.0, **material[mat])
        cameras.apply_resolved(pl, cam)
        pl.show(screenshot=str(path))
        pl.close()
        print(f"書き出し: {path}")

    shot(cameras.resolve("iso", lay, params), out_dir / "det_iso.png")
    shot(cameras.resolve("iso", lay, params, elevation_deg=20.0,
                         azimuth_deg=160.0),
         out_dir / "det_iso_wide.png")
    shot(cameras.resolve("top", lay, params), out_dir / "det_top.png")
    # 真横。床への接地と高さを見る。全高が縦に収まる距離を自分で出す
    view_h = lay["total_height"] * 1.45
    dist = view_h / (2.0 * np.tan(np.radians(24.0) / 2.0))
    az, el = np.radians(203.0), np.radians(6.0)
    focal_s = np.array([0.0, 0.0, lay["total_height"] * 0.45])
    eye_s = focal_s + dist * np.array([np.cos(el) * np.cos(az),
                                       np.cos(el) * np.sin(az), np.sin(el)])
    shot(dict(position=tuple(eye_s), focal_point=tuple(focal_s),
              view_up=(0.0, 0.0, 1.0), view_angle=24.0,
              parallel=False, parallel_scale=None),
         out_dir / "det_side.png")

    # テーブル中央の寄り。真上図でいちばん見られる所。段付きのハブキャップと
    # ボルト円・つまみが読めるか、回るハブと縁が離れているかを見る
    cen_top = lay["det_center_knobs"][5] + lay["det_center_knobs"][3]
    cfocal = np.array([0.0, 0.0, (lay["table_top"] + cen_top) / 2.0])
    ceye = cfocal + 900.0 * np.array([np.cos(np.radians(205.0)),
                                      np.sin(np.radians(205.0)), 0.62])
    shot(dict(position=tuple(ceye), focal_point=tuple(cfocal),
              view_up=(0.0, 0.0, 1.0), view_angle=30.0,
              parallel=False, parallel_scale=None),
         out_dir / "det_center.png")

    # 操作盤の寄り。カメラは自分で組む
    px, py, pz = lay["det_panel_center"]
    th = np.radians(lay["det_panel_spin"])
    focal = np.array([px * np.cos(th) - py * np.sin(th),
                      px * np.sin(th) + py * np.cos(th), pz])
    eye = focal + 1100.0 * np.array([np.cos(th + 0.30), np.sin(th + 0.30), 0.34])
    shot(dict(position=tuple(eye), focal_point=tuple(focal),
              view_up=(0.0, 0.0, 1.0), view_angle=30.0,
              parallel=False, parallel_scale=None),
         out_dir / "det_panel.png")

    # タンクとポンプの寄り。配管がタンクからポンプ、天板の下を回り込んで
    # ノズルまで繋がっているかを見る。コンベアの反対側（+ 側）へ回り込んで
    # 引かないと、手前のコンベアとタンクの胴でポンプが隠れる
    tth = np.radians(lay["det_tank_spin"])
    tside = np.radians(lay["det_tank_spin"] + 52.0)
    tr = lay["det_tank_at_r"] * 0.62
    tfocal = np.array([tr * np.cos(tth), tr * np.sin(tth),
                       lay["det_tank_z"][0] * 0.72])
    tdist = lay["det_tank_at_r"] + lay["det_tank_d"] * 4.2
    teye = np.array([tdist * np.cos(tside), tdist * np.sin(tside),
                     lay["total_height"] * 0.98])
    shot(dict(position=tuple(teye), focal_point=tuple(tfocal),
              view_up=(0.0, 0.0, 1.0), view_angle=30.0,
              parallel=False, parallel_scale=None),
         out_dir / "det_tank.png")

    # 架台の寄り。アジャスタの接地、配線ダクトの通り方、銘板を見る。
    # 架台の局所 -y 面（脚の振りに追従する）の外から低く覗く
    fth = np.radians(lay["det_frame_spin"] - 90.0)
    focal3 = np.array([120.0 * np.cos(fth), 120.0 * np.sin(fth),
                       lay["det_rail_high_z"] * 0.56])
    eye3 = focal3 + 2500.0 * np.array([np.cos(fth), np.sin(fth), 0.30])
    shot(dict(position=tuple(eye3), focal_point=tuple(focal3),
              view_up=(0.0, 0.0, 1.0), view_angle=30.0,
              parallel=False, parallel_scale=None),
         out_dir / "det_foot.png")

    # 充填部の寄り。配管の通り方を見る
    fth = np.radians(lay["det_fill_spin"])
    fr = lay["pitch_r"] + 30.0
    focal2 = np.array([fr * np.cos(fth) + 10.0 * np.sin(fth),
                       fr * np.sin(fth) - 10.0 * np.cos(fth),
                       lay["arm_base"] - 60.0])
    eye2 = focal2 + 620.0 * np.array([np.cos(fth - 0.75), np.sin(fth - 0.75), 0.30])
    shot(dict(position=tuple(eye2), focal_point=tuple(focal2),
              view_up=(0.0, 0.0, 1.0), view_angle=30.0,
              parallel=False, parallel_scale=None),
         out_dir / "det_pipe.png")
