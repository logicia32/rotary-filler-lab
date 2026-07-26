"""視点の表。名前で引いて、実座標のカメラ諸元に直す。

これまで構図は scene.py の CAMERA 1 個だけだった。記事では同じ 1 回の
シミュレーションを複数の視点から同時に見せるので、視点を名前で引ける表に
分けてある。scene.py はここを呼ぶだけでよい。

使い方
------
    import cameras
    lay = scene.derive_layout(params)
    cameras.names("3d")                       # ['iso', 'top', 'nozzle']
    cameras.apply(plotter, lay, "nozzle", params)
    for cam in cameras.turntable(lay, 72):    # ぐるり 1 周
        ...

距離の決め方
------------
どの視点も、実寸を先に決めてから距離を逆算する。だから params.json の寸法が
変わっても構図は崩れない。決め方は 2 通りある。

  1. 機械全体を写す視点（iso）は、機械の外接円柱（実測の最大半径と上下端）が
     枠に収まる距離を _fit_extent() が数値で解く。画面の縦だけでなく横も
     見るので、横に長い機械でも上下が空かない。注視点の高さも同時に決まる。
  2. 寄りの視点（nozzle）と、外から倍率を指定して撮る確認画像は、
     「画面の縦に入れる高さ」を実寸で決めて距離を逆算する（_FIT_BASE）。

数値は params.json に入れない。ここに置いてあるのは「絵にするためだけの値」で、
scene.PROPORTION と同じ扱い。

視点の名前と説明は params.json の viz.cameras が正典。ここの CAMERAS は
その 6 つと同じ名前・同じ役割で並べてある（check_names() で突き合わせる）。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 絵にするためだけの値。解析には出てこない。
# --------------------------------------------------------------------------
# 充填ステーションの世界角 [deg] の控えはここには置かない。
# params が正典で、params を渡されなかったときの控えは scene.station_deg() が
# 1 つだけ持っている（以前は同じ控えが 6 箇所に散っていた）。

# 視線が真上・真下にどれだけ近づいたら view_up を倒すか [deg]。
# 視線と view_up が平行になると向きが決まらず、絵が落ちる。
UP_DEGENERATE_TOL_DEG = 0.5

# 真上視で使う view_up。画面の上が +y、右が +x になる。
# 工程配置（供給 315 / 充填 0 / 排出 225）を数学の角度どおりに読める向き。
TOP_VIEW_UP = (0.0, 1.0, 0.0)

# 画面の横 / 縦。画角は縦で決まるので、横に長い機械を収めるにはこれが要る。
# 焼く絵はどれも 4:3（800x600 / 1600x1200 / 2400x1800）。apply() は Plotter の
# 窓から測り直すので、ここは resolve() を直に呼ぶときの控え。
FRAME_ASPECT = 4.0 / 3.0

# 外形の当てはめ（_fit_extent）で外接円柱の上下の縁を何点で見るか。
# 64 点あれば 1 周のうちどこを向いても投影の端を取り逃がさない。
FIT_RIM_SAMPLES = 64
FIT_ITERATIONS = 60


# --------------------------------------------------------------------------
# 視点の表
# --------------------------------------------------------------------------
# 共通の項目
#   kind          : "3d" = 機械を描く視点 / "2d" = 線図・断面のパネル（別担当）
#   azimuth_deg   : 方位角。x 軸から反時計回り。
#                   azimuth_is_relative=True のときだけ充填ステーションの
#                   半径方向からの振りになる。
#   elevation_deg : 仰角。水平から上向き。
#   view_angle_deg: 画角（縦）。
#   focus         : 注視点の置き方。"axis" = 回転軸上 / "fill" = 充填ステーション
#   parallel      : True で平行投影。False で透視投影。
# 画面に入れる大きさは次のどれか 1 つで指定する。
#   view_fit_x_extent      : **機械全体を写す視点はこれ。** 機械を外接円柱
#                            （実測の最大半径と上下端）で押さえ、それが枠に
#                            収まる距離と注視点の高さを _fit_extent() が
#                            数値で解く。倍率 1.0 で円柱が枠に接する。
#                            縦・横の両方を同時に見るので、横に長い機械でも
#                            上下が空かない。
# 以下は _FIT_BASE のキーで、「画面の縦に入れる高さ」を直に決める古い指定。
# 寄りの視点と、外から倍率を指定して撮る確認画像がまだ使っている。
#   view_span_x_extent_d   : 機械の外接円の直径の何倍を画面の縦に入れるか
#                            （実測。部品を外へ足すと基準ごと大きくなる）
#   view_height_x_workzone : テーブル面から機械の頂点までの何倍を画面の縦に
#   view_span_x_plate_d    : テーブル外径の何倍を画面の縦に入れるか
#   view_height_x_bottle_h : ボトル全高の何倍を画面の縦に入れるか
# view_fit_x_extent を持つ視点に _FIT_BASE のキーを overrides で渡すと、
# 渡した方が勝って古い決め方に戻る（注視点も focus / focus_bottle_ratio に
# 従う）。寄りの絵を外から作っている呼び先はそのまま動く。
CAMERAS = {
    # 機械全体の姿。方位角はノズル（世界角 0 度）が画面の奥・右寄りに
    # 来る値。ノズルの支柱はノズルより外に立つので、手前に置くと必ず手前の
    # ボトルを隠す。奥に回すと支柱がノズルの背後へ抜けて、何も隠さない。
    #
    # 画面に入れる大きさは view_fit_x_extent で決める。以前は「外接円の直径の
    # 1.05 倍を画面の縦に入れる」だった。機械は横 2158 x 高さ 1245 で横に長い
    # のに縦だけで決めていたので、上が大きく空き、しかも注視点がボトルの高さ
    # （z 989）に置いてあったせいで下は逆に切れていた（実測で画面の下端を
    # 12% はみ出していた）。今は外接円柱の上下の縁を実際に投影して、縦と横の
    # 厳しい方で距離を決め、注視点は機械の上下の中央へ置く。方位角に依らない
    # 決め方なので、turntable でぐるり回しても枠の大きさは動かない。
    # 1.0（外接円柱が枠に接する）で、実際の機械は縦 89% 横 71% を占める。
    # 円柱は機械より一回り大きいぶんの余りで、これ以上詰めると隅が切れる。
    #
    # focus / focus_bottle_ratio は、外から _FIT_BASE のキーで大きさを指定
    # されたときだけ効く控え（駆動系・受け渡し系の寄りの絵がそう撮っている）。
    "iso": dict(kind="3d", azimuth_deg=215.0, elevation_deg=26.0, view_angle_deg=20.0,
                focus="axis", focus_bottle_ratio=0.50,
                view_fit_x_extent=1.0, parallel=False,
                desc="斜め上。機械全体の姿。テーブルが間欠で動くことを見せる"),

    # 真上。8 ステーションの割出しと工程配置（供給 315 / 充填 0 / 排出 225）を
    # 読ませる。
    #
    # 倍率は機械の外接円の直径（実測）に対する値。以前はテーブル外径 φ560 の
    # 倍率で、部品を外へ足すたびに枠から外れた（2.0 = 1120 で幅 1420 の機械が
    # はみ出し、2.9 に上げても注視点が軸上なので最大半径の 2 倍には届かない）。
    # 注視点はテーブル軸上なので、外接円がそのまま入る 1.0 を既定にする。
    # 平行投影で画面の縦がそのまま実寸になり、横は縦以上あるので枠に収まる。
    #
    # 仰角は 90 度ちょうど。視線が真下を向くと view_up (0,0,1) と平行になって
    # 向きが決まらないので、真上に近いときは view_up を TOP_VIEW_UP へ倒す
    # （89.x 度で止める手もあるが、真上を真上のまま描けるこちらを採った）。
    #
    # 投影は平行を既定にした。真上から見た図は透視だと外周のボトルが外へ
    # 倒れて見え、ピッチ円が読みにくい。parallel=False を渡せば透視にもできる。
    "top": dict(kind="3d", azimuth_deg=0.0, elevation_deg=90.0, view_angle_deg=20.0,
                focus="axis", focus_bottle_ratio=0.0,
                view_span_x_extent_d=1.0, parallel=True,
                desc="真上。8 ステーションの割出しと工程配置が一目で分かる"),

    # 充填ステーションの寄り。ノズルから液が入って液面が上がるのを読ませる。
    # 注視点はノズル先端とボトル口の中間。そこを画面の中心に置くとボトルは
    # 下半分に来るので、画面の縦にボトル全高の 1.8 倍（この視点で許した上限）を
    # 入れて、底からアームまでを枠に収めてある。底は下端ぎりぎりに来る。
    # 仰角を寝かせるほど底がはみ出すので、28 度より下げない。
    # 方位角は充填ステーションの半径方向からの振り。支柱は真外（振り 0 度）に
    # 立つので、そこから 75 度ずらして支柱を画の脇へ逃がしてある。
    # 隣のステーションの正面（振り 135 度あたり）も、視線がボトルを串刺しに
    # するので避けてある。振り 60 度まで戻すと隣のボトルが主役の前に被る。
    # 75 度なら隣は枠の端に入るだけで、主役には重ならない。
    "nozzle": dict(kind="3d", azimuth_deg=75.0, azimuth_is_relative=True,
                   elevation_deg=28.0, view_angle_deg=20.0,
                   focus="fill", focus_gap_ratio=0.50,
                   view_height_x_bottle_h=1.8, parallel=False,
                   desc="充填ステーションの寄り。液面が上がっていくのを見せる"),

    # ここから下は機械を 3D で描かないパネル。resolve() は受け付けない。
    # 枠の縦横比だけ持たせてある。中身の描き方はパネル側の担当。
    "bottle_xs": dict(kind="2d", aspect=0.75,
                      desc="1 本のボトルの断面。液面の傾きと揺れだけを描く"),
    "cam": dict(kind="2d", aspect=1.60,
                desc="カム入力軸の角度とテーブル角の関係を線図と機械で同時に見せる"),
    "sensors": dict(kind="2d", aspect=2.20,
                    desc="波形パネル。上の絵と時刻を合わせて加速度・ひずみ・電流が走る"),
}

def machine_radius(lay: dict) -> float:
    """テーブル軸から測った機械の水平の最大半径 [mm]。

    正典は実測（scene.ensure_extent() が組み上がったメッシュから測って
    lay["extent_r"] に入れる）。まだ測っていなければテーブル板の半径で代用する。
    """
    r = lay.get("extent_r")
    return float(r) if r else float(lay["plate_r"])


def machine_top(lay: dict) -> float:
    """機械の頂点の高さ [mm]。

    こちらも実測が正典。lay["total_height"] は安全カバーの上端で、それより
    高い部品（弁の箱など）が付くと足りない。
    """
    z = lay.get("extent_top")
    return max(float(z), float(lay["total_height"])) if z else float(lay["total_height"])


def machine_bottom(lay: dict) -> float:
    """機械の下端の高さ [mm]。実測が無ければ床（0）とみなす。"""
    z = lay.get("extent_bottom")
    return float(z) if z is not None else 0.0


# 画面に入れる大きさの基準寸法。値はこれに倍率を掛けたもの。
_FIT_BASE = {
    "view_span_x_extent_d": lambda lay: 2.0 * machine_radius(lay),
    "view_height_x_workzone": lambda lay: machine_top(lay) - lay["table_base"],
    "view_span_x_plate_d": lambda lay: lay["plate_d"],
    "view_height_x_bottle_h": lambda lay: lay["bottle_h"],
}

# 外接円柱を枠に収める指定。_FIT_BASE と違って縦横の両方を見るので、
# 「画面の縦に入れる高さ」ではなく _fit_extent() が距離ごと解く。
FIT_EXTENT_KEY = "view_fit_x_extent"

# resolve() が見るキー。これ以外を overrides で渡したら打ち間違いとみなす。
_KNOWN_KEYS = ({"kind", "azimuth_deg", "azimuth_is_relative", "elevation_deg",
                "view_angle_deg", "focus", "focus_bottle_ratio", "focus_gap_ratio",
                "parallel", "aspect", "desc", FIT_EXTENT_KEY} | set(_FIT_BASE))


# --------------------------------------------------------------------------
# 表を引く
# --------------------------------------------------------------------------
def names(kind=None) -> list:
    """視点の名前を並べる。kind に "3d" / "2d" を渡すと絞る。"""
    if kind is None:
        return list(CAMERAS)
    return [k for k, v in CAMERAS.items() if v["kind"] == kind]


def spec(name: str) -> dict:
    """視点 1 つの諸元。呼び先で書き換えても表が汚れないように写しを返す。"""
    try:
        return dict(CAMERAS[name])
    except KeyError:
        raise KeyError(f"知らない視点: {name!r}（ある視点: {', '.join(CAMERAS)}）") from None


def check_names(params: dict) -> list:
    """params.json の viz.cameras とこの表の食い違いを並べる。

    名前と説明の正典は params.json。空のリストが返れば食い違いなし。
    """
    listed = [c["name"] for c in params.get("viz", {}).get("cameras", [])]
    issues = [f"params.json にあってここに無い: {n}" for n in listed if n not in CAMERAS]
    issues += [f"ここにあって params.json に無い: {n}" for n in CAMERAS if n not in listed]
    return issues


# --------------------------------------------------------------------------
# 実座標へ直す
# --------------------------------------------------------------------------
def _fill_angle_rad(params) -> float:
    """充填ステーションの世界角。ノズルはここに固定してある。

    params が正典。無いときだけ scene の控えを借りる（scene はこの
    モジュールを読み込む側なので、輪にならないよう呼ぶときに解く）。
    """
    if params is not None:
        value = params.get("stations", {}).get("fill_deg")
        if value is not None:
            return np.radians(float(value))
    import scene                                          # noqa: PLC0415
    return np.radians(scene.station_deg(None, "fill"))


def _view_height(sp: dict, lay: dict) -> float:
    """画面の縦に入れる高さ [mm]。指定の仕方は視点ごとに 1 つだけ。"""
    used = [k for k in _FIT_BASE if k in sp]
    if len(used) != 1:
        raise ValueError(f"画面に入れる大きさの指定が {len(used)} 個ある: {used}")
    key = used[0]
    return float(_FIT_BASE[key](lay)) * float(sp[key])


def _focal_point(sp: dict, lay: dict, fill_rad: float) -> np.ndarray:
    """注視点。"""
    focus = sp.get("focus", "axis")
    if focus == "axis":
        # 回転軸上。ボトルのどの高さを見るかだけ決める。
        z = lay["table_top"] + lay["bottle_h"] * float(sp["focus_bottle_ratio"])
        return np.array([0.0, 0.0, z])
    if focus == "fill":
        # 充填ステーションのボトル軸上。高さはボトル口とノズル先端の間。
        z = (lay["bottle_top"]
             + (lay["nozzle_tip"] - lay["bottle_top"]) * float(sp["focus_gap_ratio"]))
        return np.array([lay["pitch_r"] * np.cos(fill_rad),
                         lay["pitch_r"] * np.sin(fill_rad), z])
    raise ValueError(f"知らない注視点の置き方: {focus!r}")


def _screen_axes(az: float, el: float, view_up) -> tuple:
    """カメラの向きと画面の右・上の単位ベクトル。

    d は注視点から視点へ向かう向き（視線はその逆）。u が画面の右、v が上。
    """
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    w = -d
    u = np.cross(w, np.asarray(view_up, dtype=float))
    n = float(np.linalg.norm(u))
    if n < 1e-9:                       # view_up は el で倒してあるので来ない
        u = np.array([-np.sin(az), np.cos(az), 0.0])
        n = 1.0
    u = u / n
    return d, w, u, np.cross(u, w)


def _fit_extent(lay: dict, az: float, el: float, view_angle_deg: float,
                view_up, aspect: float, parallel: bool, scale: float) -> tuple:
    """機械が枠に収まる「画面の縦の実寸」と注視点の高さを解く。

    機械を外接円柱（実測の最大半径 R、上下端 z0..z1）で押さえ、その上下の縁を
    実際に投影して、縦にも横にも枠からはみ出さない最小の距離まで詰める。
    返り値は (画面の縦に入る実寸 [mm], 注視点の z [mm])。

    縦だけで決めない理由。この機械は横 2R が高さ z1-z0 より長い。縦の画角で
    距離を決めると横が余り、逆に横を入れようと引くと上下が空く。画面の横縦比
    aspect で横を縦に直し、厳しい方で決めれば、どちらにも無駄が出ない。

    注視点を機械の上下の中央に置くのも同じ理由。軸上のボトルの高さに置くと、
    見下ろす絵では機械の下半分（脚と床）が画面の下へはみ出し、上は空く。

    円柱で押さえるので方位角に依らない。turntable でぐるり回しても枠の
    大きさが動かず、機械が伸び縮みして見えることがない。透視では手前の縁が
    実寸より大きく写るので、投影を実際に計算して距離を詰める（直交投影の式で
    出すと 2 割足りない）。
    """
    r = machine_radius(lay)
    z0, z1 = machine_bottom(lay), machine_top(lay)
    tan_half = np.tan(np.radians(view_angle_deg) / 2.0)
    d, w, u, v = _screen_axes(az, el, view_up)

    th = np.linspace(0.0, 2.0 * np.pi, FIT_RIM_SAMPLES, endpoint=False)
    rim = np.column_stack([r * np.cos(th), r * np.sin(th)])
    ones = np.ones(len(th))
    pts = np.vstack([np.column_stack([rim, ones * z0]),
                     np.column_stack([rim, ones * z1])])

    zf = 0.5 * (z0 + z1)
    # 直交投影での当たりから始める。透視ではここから少し広がる
    view_h = 2.0 * max(r * abs(np.sin(el)) + 0.5 * (z1 - z0) * abs(np.cos(el)),
                       r / float(aspect))
    for _ in range(FIT_ITERATIONS):
        q = pts - (np.array([0.0, 0.0, zf]) + (view_h / (2.0 * tan_half)) * d)
        if parallel:
            sx = (q @ u) / (0.5 * view_h * aspect)
            sy = (q @ v) / (0.5 * view_h)
        else:
            depth = np.maximum(q @ w, 1e-6)
            sx = (q @ u) / depth / (tan_half * aspect)
            sy = (q @ v) / depth / tan_half
        # 枠は |sx| <= 1 かつ |sy| <= 1。はみ出したぶんだけ引き、足りなければ寄る
        need = float(max(np.abs(sx).max(), np.abs(sy).max()))
        off = float(0.5 * (sy.max() + sy.min()))          # 縦の中心のずれ
        # 注視点を軸に沿って動かすと絵は cos(el) 倍だけ縦に動く。
        # 真上視（cos el = 0）では動かないので触らない。
        if abs(np.cos(el)) > 1e-3:
            zf += off * (0.5 * view_h) / np.cos(el)
        view_h *= need
        if abs(need - 1.0) < 1e-5 and abs(off) < 1e-5:
            break
    return view_h * float(scale), zf


def resolve(name: str, lay: dict, params=None, frame_aspect=None, **overrides) -> dict:
    """カメラ諸元を実座標に直す。

    返り値は
        dict(position=(x, y, z), focal_point=(x, y, z), view_up=(x, y, z),
             view_angle=deg, parallel=False, parallel_scale=None)
    parallel が False のとき parallel_scale は None。

    overrides で諸元を上書きできる（表にあるキーだけ受け付ける）。
    kind="2d" の視点は 3D で描かないので受け付けない。
    frame_aspect は画面の横 / 縦。省くと FRAME_ASPECT（4:3）。外接円柱で
    大きさを決める視点（view_fit_x_extent）だけがこれを見る。
    """
    sp = spec(name)
    if sp["kind"] != "3d":
        raise ValueError(f"視点 {name!r} は kind={sp['kind']!r} のパネルで、"
                         f"3D の描画には使わない（3D は {', '.join(names('3d'))}）")

    bad = set(overrides) - _KNOWN_KEYS
    if bad:
        raise ValueError(f"知らない諸元: {', '.join(sorted(bad))}")
    sp.update(overrides)

    fill_rad = _fill_angle_rad(params)
    az = np.radians(float(sp["azimuth_deg"]))
    if sp.get("azimuth_is_relative", False):
        az += fill_rad
    el = np.radians(float(sp["elevation_deg"]))
    view_angle = float(sp["view_angle_deg"])
    parallel = bool(sp.get("parallel", False))

    # 真上・真下を向くと view_up (0,0,1) が視線と平行になって向きが決まらない。
    # そこまで倒れたら水平な view_up へ差し替える。
    if abs(np.degrees(el)) > 90.0 - UP_DEGENERATE_TOL_DEG:
        view_up = TOP_VIEW_UP
    else:
        view_up = (0.0, 0.0, 1.0)

    # 大きさの決め方。外から「画面の縦に入れる高さ」を指定されたらそちらが
    # 勝つ（寄りの絵はそう撮っている）。無ければ外接円柱を枠に当てはめる。
    if any(k in sp for k in _FIT_BASE) or FIT_EXTENT_KEY not in sp:
        view_h = _view_height(sp, lay)
        focal = _focal_point(sp, lay, fill_rad)
    else:
        view_h, focal_z = _fit_extent(
            lay, az, el, view_angle, view_up,
            FRAME_ASPECT if frame_aspect is None else float(frame_aspect),
            parallel, sp[FIT_EXTENT_KEY])
        focal = np.array([0.0, 0.0, focal_z])

    dist = view_h / (2.0 * np.tan(np.radians(view_angle) / 2.0))
    eye = focal + dist * np.array([np.cos(el) * np.cos(az),
                                   np.cos(el) * np.sin(az),
                                   np.sin(el)])

    return dict(position=tuple(float(v) for v in eye),
                focal_point=tuple(float(v) for v in focal),
                view_up=tuple(float(v) for v in view_up),
                view_angle=view_angle,
                parallel=parallel,
                # 平行投影の倍率は画面の縦半分の実寸。透視と同じ大きさに写る。
                parallel_scale=(view_h / 2.0) if parallel else None)


def frame_aspect_of(plotter) -> float:
    """Plotter の窓の横 / 縦。読めなければ FRAME_ASPECT。"""
    try:
        w, h = (float(v) for v in plotter.window_size)
        if w > 0.0 and h > 0.0:
            return w / h
    except Exception:                                     # noqa: BLE001
        pass
    return FRAME_ASPECT


def apply(plotter, lay: dict, name: str, params=None, **overrides) -> dict:
    """resolve した結果を Plotter に当てる。当てた dict を返す。

    画面の横縦比は Plotter の窓から測る。枠に機械を当てはめる視点は
    横縦比で決め方が変わるので、控えの 4:3 のまま解かない。
    """
    return apply_resolved(plotter, resolve(name, lay, params,
                                           frame_aspect=frame_aspect_of(plotter),
                                           **overrides))


def apply_resolved(plotter, cam: dict) -> dict:
    """resolve() が返した dict をそのまま Plotter に当てる。

    turntable() の並びを 1 コマずつ当てるのはこちら。
    """
    if cam["parallel"]:
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = cam["parallel_scale"]
    else:
        plotter.disable_parallel_projection()
    plotter.camera_position = [cam["position"], cam["focal_point"], cam["view_up"]]
    plotter.camera.view_angle = cam["view_angle"]
    # 視点を手で置くと前後のクリップ面が古いままになり、手前の部品が切れる
    plotter.reset_camera_clipping_range()
    return cam


def turntable(lay: dict, n_frames: int, base="iso", params=None,
              frame_aspect=None) -> list:
    """base の方位角を 360 度ぐるり回した resolve 済みの並び。

    回すのは方位角だけ。仰角・画角・注視点は動かさない。n_frames 枚で
    ちょうど 1 周するので、先頭と末尾は重ならない（そのまま繋げば輪になる）。
    外接円柱で大きさを決める視点は方位角に依らないので、回しても枠の
    大きさは動かない（機械が伸び縮みして見えない）。
    """
    n = int(n_frames)
    if n < 1:
        raise ValueError(f"コマ数は 1 以上: {n_frames}")
    az0 = float(spec(base)["azimuth_deg"])
    return [resolve(base, lay, params, frame_aspect=frame_aspect,
                    azimuth_deg=az0 + 360.0 * i / n)
            for i in range(n)]


# --------------------------------------------------------------------------
# 目で見て確かめる
# --------------------------------------------------------------------------
def _demo_render(out_dir, size=(800, 600)) -> None:
    """3D の 3 視点と turntable を焼いて、構図を目で確かめる。

    scene は「読むだけ」。scene 側がここを取り込むので、輪になる読み込みを
    避けるためにこの中で読む。描き方は scene.render() と同じ順序を使うが、
    カメラだけこちらで当てる。
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import scene

    params = scene.load_params()
    lay = scene.derive_layout(params)
    st = scene.demo_state(params, lay)
    meshes = scene.build(params, lay, st)
    # 画角は機械の実測外形に合わせるので、resolve する前に測っておく
    scene.ensure_extent(lay, meshes, params)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    issues = check_names(params)
    print("params.json との食い違い:", issues if issues else "なし")
    print(f"外形の実測: 最大半径 {lay['extent_r']:.0f} / "
          f"高さ {lay['extent_bottom']:.0f}〜{lay['extent_top']:.0f} mm")

    def shot(cam, path):
        # 描く順も材質も scene.DRAW_ORDER が唯一の表。ここで名前を並べ直すと、
        # 新しい部品を足したときにこの確認画像にだけ写らなくなる。
        plotter = scene.new_plotter(size)
        for name, material in scene.DRAW_ORDER:
            mesh = meshes.get(name)
            if mesh is None or mesh.n_points == 0:
                continue
            plotter.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True,
                             feature_angle=35.0, **scene.MATERIAL[material])
        apply_resolved(plotter, cam)
        plotter.enable_anti_aliasing("ssaa")
        plotter.show(screenshot=str(path))
        plotter.close()

    for name in names("3d"):
        cam = resolve(name, lay, params)
        shot(cam, out_dir / f"cam_{name}.png")
        print(f"{name:7s} pos={tuple(round(v, 1) for v in cam['position'])} "
              f"focal={tuple(round(v, 1) for v in cam['focal_point'])} "
              f"up={cam['view_up']} parallel={cam['parallel']}")

    for i, cam in enumerate(turntable(lay, 8)):
        shot(cam, out_dir / f"turn_{i}.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="視点の表を目で確かめる")
    ap.add_argument("--out", default="figs", help="PNG の置き場")
    ap.add_argument("--size", type=int, nargs=2, default=(800, 600),
                    metavar=("W", "H"), help="画素数")
    args = ap.parse_args()
    _demo_render(args.out, size=tuple(args.size))
