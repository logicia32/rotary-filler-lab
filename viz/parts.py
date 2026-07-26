"""ロータリー充填機の描画用 部品ライブラリ。

方針
----
* メッシュファイルは一切読まない。すべて PyVista のプリミティブと
  「断面プロファイルの回転体」からコードで生成する。
* 単位は mm。params.json がミリ表記なので、そのまま世界座標に使う。
* どの関数も最後に 4x4 の同次変換 `matrix` を掛けられる。位置と姿勢は
  呼び出し側（scene.py）が決め、部品側は原点まわりの素の形だけを持つ。

座標の約束
----------
z 上向き。回転体は z 軸まわりに回した形として作る。
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# 回転体の分割数。上げると滑らかになるが面数が増える。
RES_FINE = 64
RES_COARSE = 32


# --------------------------------------------------------------- 部品の内部比率
#
# 部品の中で形を決めている比率のうち、**外から位置を合わせるのに要るもの**を
# 名前を付けて出す。組み立て側（asm_*.py）はここを読むこと。
# 数値を写すと、この表を直したときに合わせ先だけ古くなって部品が浮く／食い込む。
# 実際に、板厚を 8 -> 12 に変えるとモータのベースが 1.0 mm 浮き、据付段の比を
# 1.07 -> 1.20 に変えるとカバーが食い込む、という壊れ方をした。

# 架台 frame()
FRAME_FOOT_H_X_PIPE = 1.8       # アジャスタの丸柱の丈 / 角パイプ一辺
FRAME_RAIL_LOW_X_LEG_H = 0.18   # 下桁の芯の高さ / 脚の丈
FRAME_RAIL_HIGH_X_PIPE = 1.2    # 上桁の芯を天板の下面から下げる量 / 角パイプ一辺
FRAME_DECK_D_X_SPAN = 1.45      # 天板の直径 / 脚の芯々距離
FRAME_DECK_T_X_PIPE = 0.28      # 天板の板厚 / 角パイプ一辺
FRAME_FOOT_R_X_PIPE = 0.45      # 丸柱の半径 / 角パイプ一辺

# カム式インデックスユニット index_unit()
INDEX_UNIT_INPUT_AXIS_X_BOX_H = 0.55   # 入力軸の芯の高さ / 本体箱の丈
INDEX_UNIT_SKIRT_SCALE = 1.07          # 据付段の平面寸法 / 本体箱の平面寸法
INDEX_UNIT_SKIRT_H_X_BOX_H = 0.09      # 据付段の丈 / 本体箱の丈
INDEX_UNIT_SHOULDER_H_X_BOX_H = 0.10   # 上の段の丈 / 本体箱の丈

# 誘導ギヤモータ gearmotor()
GEARMOTOR_BASE_GAP_MM = 2.0     # ベース上面と胴／ギヤヘッドの下端の逃げ [mm]

# 継手の安全カバー coupling_cover()
COUPLING_COVER_LID_SCALE = 1.08   # 上蓋の平面寸法 / 箱の平面寸法
COUPLING_COVER_LID_T_X_H = 0.08   # 上蓋の板厚 / 箱の丈

# 搬送コンベア conveyor()
CONVEYOR_SIDE_W_MIN_MM = 8.0    # 側枠の板幅の下限 [mm]。枠幅とチェーン幅の差が
                                # これを下回っても、側枠はこの幅で立てる


# ---------------------------------------------------------------- 変換まわり


def transform_matrix(translate=(0.0, 0.0, 0.0), rot_z_deg=0.0,
                     rot_y_deg=0.0, rot_x_deg=0.0) -> np.ndarray:
    """平行移動と各軸まわりの回転から 4x4 の同次変換を組む。

    回転の適用順は x -> y -> z。最後に平行移動。
    """
    def _rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    def _ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    def _rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    rot = _rz(np.radians(rot_z_deg)) @ _ry(np.radians(rot_y_deg)) @ _rx(np.radians(rot_x_deg))
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = np.asarray(translate, dtype=float)
    return mat


def place(mesh: pv.PolyData, matrix=None) -> pv.PolyData:
    """部品に同次変換を掛けた複製を返す。matrix が None ならそのまま。"""
    if matrix is None:
        return mesh
    return mesh.transform(np.asarray(matrix, dtype=float), inplace=False)


def merge(meshes) -> pv.PolyData:
    """複数の部品を 1 つのメッシュにまとめる。材質が同じものをまとめる用。"""
    meshes = [m for m in meshes if m is not None and m.n_points > 0]
    if not meshes:
        return pv.PolyData()
    out = meshes[0].copy()
    if len(meshes) > 1:
        out = out.merge(meshes[1:])
    if not isinstance(out, pv.PolyData):
        out = out.extract_surface(algorithm='dataset_surface')
    return out


def revolve(profile_rz, resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """(r, z) の折れ線を z 軸まわりに一回転させて閉じた面を作る。

    profile_rz は外形を左下から順にたどった点列。始点と終点を r=0 に
    置けば上下の蓋も自然に閉じる。円錐台もボトルも管もこれ 1 本で作れる。
    """
    pts = np.asarray(profile_rz, dtype=float)
    xyz = np.column_stack([pts[:, 0], np.zeros(len(pts)), pts[:, 1]])
    line = pv.lines_from_points(xyz)
    surf = line.extrude_rotate(resolution=resolution, capping=False)
    surf = surf.extract_surface(algorithm='dataset_surface')
    surf.clear_data()
    # 回転軸上（r=0）で重なった点をまとめておかないと閉じた面にならない。
    # 液の切断や透過描画がここで効いてくるので、必ず clean を通す。
    surf = surf.triangulate().clean(tolerance=1e-6)
    surf = _drop_degenerate(surf)
    # 掃引が返す三角形は表裏の向きが揃っていない。揃えておかないと、
    # 平らな蓋の陰影が面ごとに反転する。
    surf = surf.compute_normals(cell_normals=True, point_normals=True,
                                consistent_normals=True, auto_orient_normals=True,
                                split_vertices=False, inplace=False)
    # 面の並びだけ直ればよい。法線そのものは描画時に稜線を割って作り直すので、
    # ここで持たせると平面と側面が混ざった値が残ってしまう。捨てる。
    surf.clear_data()
    return place(surf, matrix)


def _drop_degenerate(mesh: pv.PolyData) -> pv.PolyData:
    """面積ゼロのセルを落とす。

    回転軸上（r=0）では掃引した四角形が潰れて線分になる。そのまま描くと
    円板の中心から放射状に白い筋が出るので、ここで捨てておく。
    """
    sized = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    keep = np.flatnonzero(np.asarray(sized.cell_data["Area"]) > 1e-9)
    if len(keep) == mesh.n_cells:
        return mesh
    out = mesh.extract_cells(keep).extract_surface(algorithm='dataset_surface')
    out.clear_data()
    return out.triangulate().clean(tolerance=1e-6)


# ---------------------------------------------------------------- 基本形状


def box(size, center=(0.0, 0.0, 0.0), matrix=None) -> pv.PolyData:
    """直方体。size は (x, y, z) の辺長。"""
    sx, sy, sz = (float(v) for v in size)
    cx, cy, cz = (float(v) for v in center)
    mesh = pv.Cube(center=(cx, cy, cz), x_length=sx, y_length=sy, z_length=sz)
    return place(mesh.triangulate(), matrix)


def cylinder(radius, height, base_z=0.0, resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """z 軸に沿った円柱。base_z は下端の高さ。"""
    prof = [(0.0, base_z),
            (radius, base_z),
            (radius, base_z + height),
            (0.0, base_z + height)]
    return revolve(prof, resolution=resolution, matrix=matrix)


def cone_frustum(r_bottom, r_top, height, base_z=0.0,
                 resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """円錐台。r_top を 0 にすれば円錐になる。"""
    prof = [(0.0, base_z),
            (r_bottom, base_z),
            (r_top, base_z + height),
            (0.0, base_z + height)]
    return revolve(prof, resolution=resolution, matrix=matrix)


def plate(diameter, thickness, base_z=0.0, resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """円板。テーブルや架台の天板に使う。"""
    return cylinder(diameter / 2.0, thickness, base_z=base_z,
                    resolution=resolution, matrix=matrix)


def tube(outer_radius, inner_radius, height, base_z=0.0,
         resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """中空の円筒（リング押し出し）。軸受リングやカバー上端の枠に使う。"""
    prof = [(inner_radius, base_z),
            (outer_radius, base_z),
            (outer_radius, base_z + height),
            (inner_radius, base_z + height),
            (inner_radius, base_z)]
    return revolve(prof, resolution=resolution, matrix=matrix)


def flat_ring(inner_radius, outer_radius, z=0.0,
              resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """厚みの無い平らな円環。面に貼りつけて色を乗せる用。

    接地部の暗がりのように「面を汚す」だけの用途を想定している。厚みを
    持たせると横から見たときに段が出るので、板ではなく一枚の面にしてある。
    """
    if outer_radius <= inner_radius:
        return pv.PolyData()
    prof = [(float(inner_radius), float(z)), (float(outer_radius), float(z))]
    return revolve(prof, resolution=resolution, matrix=matrix)


def ring(radius, z=0.0, thickness=1.0, resolution=RES_FINE, n_sides=12,
         matrix=None) -> pv.PolyData:
    """水平な円を細い管でなぞった輪。稜線をなぞって光を乗せる用。

    管が細いと分割数の粗さがそのまま数珠つなぎに見えるので、既定より
    細かく刻む。
    """
    if radius <= 0.0 or thickness <= 0.0:
        return pv.PolyData()
    th = np.linspace(0.0, 2.0 * np.pi, int(resolution), endpoint=False)
    pts = np.column_stack([radius * np.cos(th), radius * np.sin(th),
                           np.full(len(th), float(z))])
    pts = np.vstack([pts, pts[:1]])            # 閉じる
    out = pv.lines_from_points(pts).tube(radius=float(thickness), n_sides=int(n_sides))
    out = out.extract_surface(algorithm='dataset_surface')
    out.clear_data()
    return place(out.triangulate(), matrix)


def extrude_polygon(points_xy, height, base_z=0.0, matrix=None) -> pv.PolyData:
    """閉じた 2D 多角形（凹んでいてよい）を z 方向に押し出す。

    切り欠きのある板はこれで作る。円板から円柱を引くやり方は取らない。
    points_xy は反時計回りの (x, y) の列。始点と終点は重ねない。
    自己交差しない単純多角形であることが条件で、そこは点列を作る側が守る。
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return pv.PolyData()

    h = float(height)
    z0 = float(base_z)
    if h == 0.0:
        return pv.PolyData()
    if h < 0.0:                      # 下向きの押し出しは下端を付け替えて上向きに直す
        z0, h = z0 + h, -h

    n = len(pts)
    xyz = np.column_stack([pts[:, 0], pts[:, 1], np.full(n, z0)])
    # セル 1 個の多角形。三角形分割は耳切りなので凹んでいても通る
    face = np.hstack([[n], np.arange(n)])
    cap = pv.PolyData(xyz, faces=face).triangulate()

    solid = cap.extrude((0.0, 0.0, h), capping=True)
    solid = solid.extract_surface(algorithm='dataset_surface')
    solid.clear_data()
    solid = solid.triangulate().clean(tolerance=1e-6)
    solid = _drop_degenerate(solid)
    # 掃引と同じ事情。向きを揃えないと押し出した側面が面ごとに黒く落ちる
    solid = solid.compute_normals(cell_normals=True, point_normals=True,
                                  consistent_normals=True, auto_orient_normals=True,
                                  split_vertices=False, inplace=False)
    solid.clear_data()
    return place(solid, matrix)


# z 方向に組んだ部品／x 方向に組んだ部品を、指定の軸へ倒すための回転。
_AXIS_ROT = {
    ("z", "x"): {"rot_y_deg": 90.0},
    ("z", "y"): {"rot_x_deg": -90.0},
    ("z", "z"): {},
    ("x", "x"): {},
    ("x", "y"): {"rot_z_deg": 90.0},
    ("x", "z"): {"rot_y_deg": -90.0},
}


def _axis_matrix(axis, translate=(0.0, 0.0, 0.0), local_axis="z") -> np.ndarray:
    """local_axis 向きに組んだ部品を axis 向きへ倒し、translate へ運ぶ変換。"""
    key = (str(local_axis).lower(), str(axis).lower())
    if key not in _AXIS_ROT:
        raise ValueError("axis は 'x' / 'y' / 'z' のどれか: %r" % (axis,))
    return transform_matrix(translate=translate, **_AXIS_ROT[key])


def _direction_matrix(direction, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """z 軸に組んだ部品を任意の向き direction へ倒す変換。配管の各区間用。"""
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    mat = np.eye(4)
    mat[:3, 3] = np.asarray(origin, dtype=float)
    if norm < 1e-12:
        return mat
    u = d / norm
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, u)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, u))
    if s < 1e-12:
        # 真上か真下。真下のときは x 軸まわりに半回転させる
        rot = np.eye(3) if c > 0.0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        rot = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    mat[:3, :3] = rot
    return mat


def horizontal_cylinder(radius, length, axis="x", center=(0.0, 0.0, 0.0),
                        resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """水平に寝かせた円柱。軸・モータ胴・カップリングの土台。

    revolve() は z 軸まわりなので、z 方向に作ってから axis へ倒す。
    center は円柱の中心（両端面の中点）。
    """
    r = float(radius)
    ln = float(length)
    if r <= 0.0 or ln <= 0.0:
        return pv.PolyData()
    body = cylinder(r, ln, base_z=-ln / 2.0, resolution=resolution)
    return place(place(body, _axis_matrix(axis, center, local_axis="z")), matrix)


def sphere(radius, center=(0.0, 0.0, 0.0), resolution=RES_COARSE,
           matrix=None) -> pv.PolyData:
    """球。半円の断面を回した回転体として作る。配管の折れ目を隠す用。"""
    r = float(radius)
    if r <= 0.0:
        return pv.PolyData()
    t = np.linspace(np.pi, 0.0, max(int(resolution) // 2, 8))
    prof = np.column_stack([r * np.sin(t), r * np.cos(t)])
    out = revolve(prof, resolution=resolution)
    return place(place(out, transform_matrix(translate=center)), matrix)


# ---------------------------------------------------------------- 機械固有


def bottle(inner_diameter, body_height, shoulder_height,
           neck_diameter, neck_height, wall_thickness,
           base_z=0.0, resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """ボトル外形。胴（円筒）＋肩（円錐台）＋首（円筒）の 3 区間。

    中空にはしない。外形だけを 1 本の回転体にして、余計な内部面が
    半透明表示で濁らないようにしている。寸法は params.json の bottle 節。
    """
    r_body = inner_diameter / 2.0 + wall_thickness
    r_neck = neck_diameter / 2.0 + wall_thickness
    z0 = base_z
    z_shoulder = z0 + body_height
    z_neck = z_shoulder + shoulder_height
    z_top = z_neck + neck_height
    # 底の角と口元をわずかに落として、板金の角のような硬さを消す
    fillet = min(wall_thickness * 2.0, r_body * 0.12)
    prof = [
        (0.0, z0),
        (r_body - fillet, z0),
        (r_body, z0 + fillet),
        (r_body, z_shoulder),          # 胴
        (r_neck, z_neck),              # 肩（円錐台）
        (r_neck, z_top),               # 首
        (0.0, z_top),
    ]
    return revolve(prof, resolution=resolution, matrix=matrix)


def bottle_edges(inner_diameter, body_height, shoulder_height,
                 neck_diameter, neck_height, wall_thickness,
                 base_z=0.0, thickness=None,
                 resolution=RES_FINE * 2, matrix=None) -> pv.PolyData:
    """ボトルの稜線（肩の付け根・首の付け根・口元）をなぞった細い輪。

    半透明のガラスは面の陰影がほとんど出ないので、形の変わり目にだけ
    細い輪を置いて光を乗せる。これがあると胴・肩・首の区切りが読める。
    底は接地部の暗がりで読めるので輪は置かない。寸法の取り方は
    `bottle()` と同じ。
    """
    r_body = inner_diameter / 2.0 + wall_thickness
    r_neck = neck_diameter / 2.0 + wall_thickness
    thick = thickness if thickness is not None else wall_thickness * 0.85

    z_shoulder = base_z + body_height
    z_neck = z_shoulder + shoulder_height
    z_top = z_neck + neck_height

    rings = [
        ring(r_body, z_shoulder, thick, resolution=resolution),
        ring(r_neck, z_neck, thick, resolution=resolution),
        ring(r_neck, z_top, thick, resolution=resolution),
    ]
    return place(merge(rings), matrix)


def liquid(inner_radius, level_mm, base_z=0.0, tilt_rad=0.0, tilt_dir_rad=0.0,
           resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """ボトル内の液。円柱を傾いた平面で切って液面を作る。

    level_mm は静止時の液深。傾きは液面中心を軸に持ち上げ／下げるので、
    体積はおおよそ保たれる。tilt_dir_rad は傾きの向き（水平面内の方位角）。
    """
    level = max(float(level_mm), 0.0)
    if level <= 0.0 or inner_radius <= 0.0:
        return pv.PolyData()

    tilt = float(tilt_rad)
    # 切り代。壁際で液面が上下する分だけ円柱を高くしておく
    margin = abs(inner_radius * np.tan(tilt)) + inner_radius * 0.02 + 0.5
    solid = cylinder(inner_radius, level + margin, base_z=base_z,
                     resolution=resolution)

    # 液面の法線は上向き。clip_closed_surface は法線の向いた側を残すので、
    # 液として残したいのは平面の下側 = 法線を反転して渡す。
    normal = -np.array([
        -np.sin(tilt) * np.cos(tilt_dir_rad),
        -np.sin(tilt) * np.sin(tilt_dir_rad),
        np.cos(tilt),
    ], dtype=float)
    origin = np.array([0.0, 0.0, base_z + level], dtype=float)
    # 傾いた平面で切り、切り口に蓋をして液面にする
    cut = solid.clip_closed_surface(normal=normal, origin=origin)
    cut = cut.extract_surface(algorithm='dataset_surface')
    cut.clear_data()
    cut = cut.triangulate().clean(tolerance=1e-6)
    return place(cut, matrix)


def liquid_rim(inner_radius, level_mm, base_z=0.0, tilt_rad=0.0, tilt_dir_rad=0.0,
               thickness=None, resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """液面の縁を細い管でなぞった輪。液面の楕円を目で追えるようにする。

    液面は `z = z0 + R*tan(tilt)*cos(theta - tilt_dir)` の平面なので、
    壁ぎわの高さはそのまま解析式で書ける。液の内側にわずかに入れて置き、
    液の側面と重なってちらつかないようにしてある。
    """
    level = max(float(level_mm), 0.0)
    if level <= 0.0 or inner_radius <= 0.0:
        return pv.PolyData()

    thick = thickness if thickness is not None else inner_radius * 0.035
    radius = inner_radius - thick * 0.5
    th = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    z = base_z + level + radius * np.tan(float(tilt_rad)) * np.cos(th - float(tilt_dir_rad))
    pts = np.column_stack([radius * np.cos(th), radius * np.sin(th), z])
    pts = np.vstack([pts, pts[:1]])          # 閉じる
    ring = pv.lines_from_points(pts).tube(radius=thick, n_sides=8)
    ring = ring.extract_surface(algorithm='dataset_surface')
    ring.clear_data()
    return place(ring.triangulate(), matrix)


def rotary_table(plate_diameter, plate_thickness, base_z=0.0,
                 hub_diameter=None, hub_height=None,
                 rim_step=None, resolution=RES_FINE, matrix=None) -> dict:
    """割出しテーブル。円板＋外周のわずかな段＋中心ハブ。

    戻り値は部位名 -> メッシュの辞書。材質はすべてステンレス扱い。
    """
    plate_r = plate_diameter / 2.0
    hub_d = hub_diameter if hub_diameter is not None else plate_diameter * 0.20
    hub_h = hub_height if hub_height is not None else plate_thickness * 3.5
    step = rim_step if rim_step is not None else plate_thickness * 0.6

    top_z = base_z + plate_thickness
    parts = {
        "disc": plate(plate_diameter, plate_thickness, base_z=base_z,
                      resolution=resolution),
        # 外周の立ち上がり。ボトルが外へ落ちないためのわずかな縁
        "rim": tube(plate_r, plate_r - step * 1.6, step, base_z=top_z,
                    resolution=resolution),
        "hub": cylinder(hub_d / 2.0, hub_h, base_z=top_z, resolution=resolution),
    }
    return {k: place(v, matrix) for k, v in parts.items()}


def nozzle(bore_diameter, length, base_z=0.0, tip_ratio=0.55,
           collar_ratio=1.5, resolution=RES_COARSE, matrix=None) -> pv.PolyData:
    """充填ノズル。上部の太い胴、下端を絞ったテーパ、途中の取付カラー。

    base_z はノズル先端（下端）の高さ。そこから上へ length だけ伸びる。
    """
    r = bore_diameter / 2.0
    r_tip = r * tip_ratio
    r_collar = r * collar_ratio
    taper = length * 0.22
    collar_h = length * 0.10
    collar_z = base_z + length * 0.62
    prof = [
        (0.0, base_z),
        (r_tip, base_z),
        (r, base_z + taper),
        (r, collar_z),
        (r_collar, collar_z),
        (r_collar, collar_z + collar_h),
        (r, collar_z + collar_h),
        (r, base_z + length),
        (0.0, base_z + length),
    ]
    return revolve(prof, resolution=resolution, matrix=matrix)


def nozzle_gantry(nozzle_r, post_r, post_radius, post_base_z,
                  arm_base_z, arm_thickness, arm_width,
                  angle_rad=0.0, resolution=RES_COARSE, matrix=None) -> dict:
    """ノズル 1 本を吊るための固定支持。柱 1 本＋内側へ張り出した水平アーム。

    テーブルの外に柱を立て、そこから中心へ向かってアームを出し、先端で
    ノズルをくわえる。テーブルとは縁が切れていて一緒には回らない。
    ノズルの世界角は `angle_rad` で固定する。

    nozzle_r  : ノズル中心の半径（ピッチ円の半径と同じにする）
    post_r    : 柱を立てる半径。テーブル外周より外に取る
    """
    length = post_r - nozzle_r + arm_width * 0.6
    arm_center_r = post_r - length / 2.0
    top_z = arm_base_z + arm_thickness

    # 柱。天板の上に立てて、アームの上面まで伸ばす
    post = cylinder(post_radius, top_z - post_base_z, base_z=post_base_z,
                    resolution=resolution)
    # 柱の足元のフランジ。天板に締めてある見た目にする
    flange = cone_frustum(post_radius * 2.0, post_radius * 1.35,
                          post_radius * 0.55, base_z=post_base_z,
                          resolution=resolution)
    post = merge([post, flange])
    post = place(post, transform_matrix(translate=(post_r, 0.0, 0.0)))

    # 水平アーム。先端の下面にノズル取付のボスを出す
    arm = box((length, arm_width, arm_thickness),
              center=(arm_center_r, 0.0, arm_base_z + arm_thickness / 2.0))
    boss = cylinder(arm_width * 0.42, arm_thickness * 0.9,
                    base_z=arm_base_z - arm_thickness * 0.9,
                    resolution=resolution,
                    matrix=transform_matrix(translate=(nozzle_r, 0.0, 0.0)))
    arm = merge([arm, boss])

    spin = transform_matrix(rot_z_deg=np.degrees(float(angle_rad)))
    parts_out = {"post": place(post, spin), "arm": place(arm, spin)}
    return {k: place(v, matrix) for k, v in parts_out.items()}


def frame(span, deck_top_z, pipe, foot_height=None, deck_diameter=None,
          deck_thickness=None, rail_levels=None, deck_bore_d=None,
          matrix=None) -> dict:
    """角パイプを組んだ架台。4 本脚＋上下の桁＋丸い天板＋アジャスタ脚。

    span は脚の芯々距離（正方形配置）。pipe は角パイプの一辺。
    脚は既定で世界角 45/135/225/315 度（対角）に立つ。向きを変えたいときは
    matrix に z まわりの回転を渡す。天板は丸いので見た目は脚と桁しか変わらない。

    deck_bore_d を渡すと天板を円環にして、中心にその径の下穴を開ける。
    テーブルを回す出力軸はここを通る。**円板から円柱を引く（ブール演算）
    やり方は取らない。** 渡さなければ中実の円板のまま。
    """
    foot_h = foot_height if foot_height is not None else pipe * FRAME_FOOT_H_X_PIPE
    deck_d = deck_diameter if deck_diameter is not None else span * FRAME_DECK_D_X_SPAN
    deck_t = deck_thickness if deck_thickness is not None else pipe * FRAME_DECK_T_X_PIPE

    deck_bottom = deck_top_z - deck_t
    leg_top = deck_bottom
    leg_bottom = foot_h
    leg_h = leg_top - leg_bottom
    half = span / 2.0

    legs = []
    feet = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            legs.append(box((pipe, pipe, leg_h),
                            center=(sx * half, sy * half, leg_bottom + leg_h / 2.0)))
            feet.append(cylinder(pipe * FRAME_FOOT_R_X_PIPE, foot_h,
                                 base_z=0.0, resolution=RES_COARSE,
                                 matrix=transform_matrix(translate=(sx * half, sy * half, 0.0))))

    if rail_levels is None:
        rail_levels = (leg_bottom + leg_h * FRAME_RAIL_LOW_X_LEG_H,
                       leg_top - pipe * FRAME_RAIL_HIGH_X_PIPE)

    rails = []
    for z in rail_levels:
        for sy in (-1, 1):
            rails.append(box((span, pipe, pipe), center=(0.0, sy * half, z)))
        for sx in (-1, 1):
            rails.append(box((pipe, span, pipe), center=(sx * half, 0.0, z)))

    bore_r = 0.0 if deck_bore_d is None else float(deck_bore_d) / 2.0
    if bore_r > 0.0:
        deck = tube(deck_d / 2.0, bore_r, deck_t, base_z=deck_bottom)
    else:
        deck = plate(deck_d, deck_t, base_z=deck_bottom)

    parts = {
        "legs": merge(legs),
        "feet": merge(feet),
        "rails": merge(rails),
        "deck": deck,
    }
    return {k: place(v, matrix) for k, v in parts.items()}


def cover(radius, height, base_z=0.0, panels=8, post=None,
          ring_height=None, arcs=None, wall=None, matrix=None) -> dict:
    """安全カバー。半透明のパネル＋角柱の支柱＋上下の枠。

    上面は開けてある。上から見下ろすカメラで中身がそのまま見えるようにし、
    かつ側面のパネル越しにも中が透けるようにするため。
    戻り値は "panels"（半透明）と "posts"（ステンレス）に分けてある。

    arcs に [(始まり, 終わり), ...] を渡すと、その角度範囲だけに板を張る。
    ボトルが出入りする受け渡し部を開けるのに使う。渡さなければ全周。
    支柱は各円弧の両端と、その間を post_span_deg 以下に割った位置に立てる。

    **radius は板の内面（内法）で、板も枠も支柱もそこから外へ積む。**
    以前は radius を板の外面にして、枠（radius ± 支柱幅 0.6 倍）と支柱
    （radius を芯にした角柱）を板の内側にも張り出させていた。テーブル外周の
    液受け溝の立ち上がりがちょうどそこを通るので、下枠と支柱が溝に 9 mm
    食い込んでいた。溝は内へ寄せられない（内側は外周ガイドの据付座で
    止まる）ので、カバーの側を外へ出してある。
    """
    post_w = post if post is not None else radius * 0.09
    ring_h = ring_height if ring_height is not None else post_w * 1.1
    wall_t = wall if wall is not None else radius * 0.012
    # 内側から外へ：板（radius〜skin_ro）、その外に枠と支柱。
    skin_ro = radius + wall_t
    ring_ri = skin_ro
    ring_ro = skin_ro + post_w * 1.2
    post_r = skin_ro + post_w / 2.0                # 支柱の芯。内面が板の外面

    def stand(th_rad):
        """角度 th に支柱を 1 本立てる。"""
        return box((post_w, post_w, height + ring_h),
                   center=(0.0, 0.0, base_z + (height + ring_h) / 2.0),
                   matrix=transform_matrix(
                       translate=(post_r * np.cos(th_rad), post_r * np.sin(th_rad), 0.0),
                       rot_z_deg=np.degrees(th_rad)))

    if arcs is None:
        # 全周。パネルは分割数を panels にした円筒の側面。上下の蓋は付けない
        shells = [pv.Cylinder(center=(0.0, 0.0, base_z + height / 2.0),
                              direction=(0.0, 0.0, 1.0),
                              radius=radius + wall_t / 2.0, height=height,
                              resolution=panels, capping=False).triangulate()]
        posts = [stand(2.0 * np.pi * (i + 0.5) / panels) for i in range(panels)]
        frames = [tube(ring_ro, ring_ri, ring_h, base_z=base_z + height,
                       resolution=panels * 4),
                  tube(ring_ro, ring_ri, ring_h * 0.8, base_z=base_z,
                       resolution=panels * 4)]
    else:
        # 指定の角度範囲だけ。開口のところは板も枠も無い
        span_max = 360.0 / max(int(panels), 1)         # 支柱の間隔の上限 [deg]
        shells, posts, frames = [], [], []
        for a0, a1 in arcs:
            a0, a1 = float(a0), float(a1)
            if a1 <= a0:
                a1 += 360.0
            shells.append(crescent_guide(radius, skin_ro, a0, a1, height,
                                         base_z=base_z))
            frames.append(crescent_guide(ring_ri, ring_ro, a0, a1, ring_h,
                                         base_z=base_z + height))
            frames.append(crescent_guide(ring_ri, ring_ro, a0, a1, ring_h * 0.8,
                                         base_z=base_z))
            n_post = max(int(np.ceil((a1 - a0) / span_max)), 1) + 1
            # 端の支柱は円弧の端に芯を置かず、外側の面を板の端に合わせる。
            # 芯を端に置くと支柱の半分が開口へはみ出し、受け渡しの門型の脚
            # （半径 361.6 の 251.9 / 288.1deg に太さ 12.8）と同じ所を通る。
            inset = np.degrees(np.arctan2(post_w / 2.0, post_r))
            lo, hi = a0 + inset, a1 - inset
            if hi < lo:                        # 円弧が支柱より狭いときは真ん中に 1 本
                lo = hi = 0.5 * (a0 + a1)
            posts += [stand(np.radians(a)) for a in np.linspace(lo, hi, n_post)]

    parts = {
        "panels": merge(shells),
        "posts": merge(posts + frames),
    }
    return {k: place(v, matrix) for k, v in parts.items()}


# ---------------------------------------------------------------- 駆動系


def index_unit(box_size=(290.0, 250.0, 210.0),
               flange_outer_d=210.0, flange_inner_d=150.0,
               flange_t_outer=18.0, flange_t_inner=12.0,
               boss_d=105.0, boss_h=50.0,
               bolt_d=20.0, bolt_h=6.0,
               input_boss_d=70.0, input_boss_len=30.0,
               sight_d=30.0, oil_port_d=26.0,
               base_z=0.0, resolution=RES_FINE, matrix=None) -> dict:
    """カム式インデックスユニット。テーブルの真下・中心に据える。

    箱だけだとただの鋳物の塊にしか見えない。上面の段付き出力フランジ、
    側面の入力軸ボス、油面窓、四隅の取付ボルト頭。この 4 つが姿を決める。
    base_z は本体箱の底面。戻り値は部位名 -> メッシュの辞書。

        housing     本体箱＋上下のリブ段
        flange      上面の段付き出力フランジ
        boss        出力軸ボス（フランジの上）
        input_boss  側面の入力軸ボス（-x 側へ出す）
        bolts       上面四隅の取付ボルト頭
        oil_port    上面の給油口
        sight_glass 側面の油面窓（+y 側。暗い琥珀を当てる）
    """
    bx, by, bz = (float(v) for v in box_size)
    z_top = base_z + bz

    # 本体。下に一回り大きい据付段、上に一回り小さい段を重ねて鋳物の抜けを出す。
    # 段は箱の外へ出す。中に埋めると横から見たときに何も見えない
    skirt_h = bz * INDEX_UNIT_SKIRT_H_X_BOX_H
    shoulder_h = bz * INDEX_UNIT_SHOULDER_H_X_BOX_H
    body_h = bz - shoulder_h
    housing = merge([
        box((bx, by, body_h), center=(0.0, 0.0, base_z + body_h / 2.0)),
        box((bx * INDEX_UNIT_SKIRT_SCALE, by * INDEX_UNIT_SKIRT_SCALE, skirt_h),
            center=(0.0, 0.0, base_z + skirt_h / 2.0)),
        box((bx * 0.86, by * 0.86, shoulder_h),
            center=(0.0, 0.0, z_top - shoulder_h / 2.0)),
    ])

    ro = flange_outer_d / 2.0
    ri = flange_inner_d / 2.0
    z1 = z_top + flange_t_outer
    z2 = z1 + flange_t_inner
    flange = revolve([(0.0, z_top), (ro, z_top), (ro, z1),
                      (ri, z1), (ri, z2), (0.0, z2)], resolution=resolution)

    boss = cylinder(boss_d / 2.0, boss_h, base_z=z2, resolution=resolution)

    # 側面の入力軸ボス。ここへギヤモータの出力軸が入る
    input_boss = horizontal_cylinder(
        input_boss_d / 2.0, input_boss_len, axis="x",
        center=(-(bx / 2.0 + input_boss_len / 2.0), 0.0,
                base_z + bz * INDEX_UNIT_INPUT_AXIS_X_BOX_H),
        resolution=resolution)

    inset = min(bx, by) * 0.10 + bolt_d
    bolts = merge([
        cylinder(bolt_d / 2.0, bolt_h, base_z=z_top, resolution=RES_COARSE,
                 matrix=transform_matrix(translate=(sx * (bx / 2.0 - inset),
                                                    sy * (by / 2.0 - inset), 0.0)))
        for sx in (-1, 1) for sy in (-1, 1)
    ])

    # 給油口。上面の隅寄り。栓の頭が少し太い
    rp = oil_port_d / 2.0
    oil_port = revolve([(0.0, z_top), (rp, z_top), (rp, z_top + 8.0),
                        (rp * 1.25, z_top + 8.0), (rp * 1.25, z_top + 12.0),
                        (0.0, z_top + 12.0)], resolution=RES_COARSE,
                       matrix=transform_matrix(
                           translate=(bx * 0.30, -by * 0.28, 0.0)))

    sight_glass = horizontal_cylinder(
        sight_d / 2.0, 4.0, axis="y",
        center=(bx * 0.18, by / 2.0 + 1.0, base_z + bz * 0.32),
        resolution=RES_COARSE)

    parts_out = {
        "housing": housing,
        "flange": flange,
        "boss": boss,
        "input_boss": input_boss,
        "bolts": bolts,
        "oil_port": oil_port,
        "sight_glass": sight_glass,
    }
    return {k: place(v, matrix) for k, v in parts_out.items()}


def gearmotor(motor_d=88.0, motor_len=145.0, gearhead_size=100.0, gearhead_len=75.0,
              shaft_d=24.0, shaft_len=45.0, fins=7,
              terminal_size=(50.0, 35.0, 35.0), base_size=(220.0, 120.0, 10.0),
              axis="x", origin=(0.0, 0.0, 0.0), resolution=RES_COARSE,
              matrix=None) -> dict:
    """誘導ギヤモータ。平行軸のギヤヘッドを介してカム入力軸に直結する。

    origin は出力軸の付け根（ギヤヘッド前面の中心）で、そこから axis の
    正方向へ軸が出て、負方向へギヤヘッド・モータ胴が続く。
    戻り値は部位名 -> メッシュの辞書。

        motor        胴（横向き円筒）＋後端の端子カバー
        fins         胴の外周に並ぶ冷却フィン
        gearhead     角形のギヤヘッド
        shaft        出力軸とその付け根のボス
        terminal_box 端子箱＋薄い上蓋
        base         取付ベース板
    """
    rm = motor_d / 2.0
    gs = gearhead_size / 2.0
    x_gear = -gearhead_len                    # ギヤヘッド後面 = 胴の前面
    x_motor_end = x_gear - motor_len

    motor = merge([
        horizontal_cylinder(rm, motor_len, axis="x",
                            center=(x_gear - motor_len / 2.0, 0.0, 0.0),
                            resolution=RES_FINE),
        # 後端の端子カバー。薄い円板を重ねるだけで端面が締まる
        horizontal_cylinder(rm * 0.92, 12.0, axis="x",
                            center=(x_motor_end - 5.0, 0.0, 0.0),
                            resolution=RES_FINE),
    ])

    fin_list = []
    n_fin = max(int(fins), 0)
    for i in range(n_fin):
        # 胴の前後を少し空けてフィンを等間隔に並べる
        t = (i + 1.0) / (n_fin + 1.0)
        xc = x_gear - motor_len * t
        fin_list.append(tube(rm + 4.0, rm - 1.0, 5.0, base_z=-2.5,
                             resolution=RES_FINE,
                             matrix=_axis_matrix("x", (xc, 0.0, 0.0))))
    fins_mesh = merge(fin_list)

    gearhead = merge([
        box((gearhead_len, gearhead_size, gearhead_size),
            center=(x_gear / 2.0, 0.0, 0.0)),
        # 前面の取付フランジ。ギヤヘッドより一回り大きい薄板
        box((6.0, gearhead_size * 1.10, gearhead_size * 1.10), center=(-3.0, 0.0, 0.0)),
    ])

    shaft = merge([
        horizontal_cylinder(gs * 0.42, 14.0, axis="x", center=(7.0, 0.0, 0.0),
                            resolution=resolution),
        horizontal_cylinder(shaft_d / 2.0, shaft_len, axis="x",
                            center=(shaft_len / 2.0, 0.0, 0.0), resolution=resolution),
    ])

    tx, ty, tz = (float(v) for v in terminal_size)
    x_term = x_gear - motor_len * 0.45
    terminal_box = merge([
        box((tx, ty, tz), center=(x_term, 0.0, rm + tz / 2.0 - 4.0)),
        box((tx * 1.12, ty * 1.12, 4.0), center=(x_term, 0.0, rm + tz - 2.0)),
    ])

    bx, by, bt = (float(v) for v in base_size)
    base_top = -max(rm, gs) - GEARMOTOR_BASE_GAP_MM
    base = box((bx, by, bt), center=(x_motor_end / 2.0, 0.0, base_top - bt / 2.0))

    parts_out = {
        "motor": motor,
        "fins": fins_mesh,
        "gearhead": gearhead,
        "shaft": shaft,
        "terminal_box": terminal_box,
        "base": base,
    }
    lay = _axis_matrix(axis, origin, local_axis="x")
    return {k: place(place(v, lay), matrix) for k, v in parts_out.items()}


def coupling(diameter=42.0, length=58.0, axis="x", center=(0.0, 0.0, 0.0),
             resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """軸継手。太-細-太の段付き円筒。中央の細い所が締結部の逃げになる。"""
    r = float(diameter) / 2.0
    ln = float(length)
    r_mid = r * 0.72
    z0 = -ln / 2.0
    a = z0 + ln * 0.34
    b = z0 + ln * 0.66
    prof = [(0.0, z0), (r, z0), (r, a), (r_mid, a), (r_mid, b),
            (r, b), (r, z0 + ln), (0.0, z0 + ln)]
    body = revolve(prof, resolution=resolution)
    return place(place(body, _axis_matrix(axis, center, local_axis="z")), matrix)


def coupling_cover(size=(80.0, 80.0, 100.0), center=(0.0, 0.0, 0.0),
                   matrix=None) -> pv.PolyData:
    """カップリング部の安全カバー。角箱＋一回り大きい上蓋。

    回転する継手をむき出しにした機械は据え付けられない。実機の写真には
    必ずこれが写る。
    """
    sx, sy, sz = (float(v) for v in size)
    cx, cy, cz = (float(v) for v in center)
    lid_t = sz * COUPLING_COVER_LID_T_X_H
    body = box((sx, sy, sz), center=(cx, cy, cz))
    lid = box((sx * COUPLING_COVER_LID_SCALE, sy * COUPLING_COVER_LID_SCALE, lid_t),
              center=(cx, cy, cz + sz / 2.0))
    return place(merge([body, lid]), matrix)


# ---------------------------------------------------------------- 受け渡し


def _notched_disc_outline(outer_r, count, pocket_r, pcd_r,
                          resolution=RES_FINE, phase_deg=0.0) -> np.ndarray:
    """外周に丸いポケットを count 個切り欠いた円盤の輪郭。

    反時計回りの (x, y) 列を返す。外周円とポケット円の交点を解いてから
    繋ぐ。ここを目分量で繋ぐと輪郭が自分と交わり、三角形分割が壊れる。

    ポケットが外へ開くには `|外半径 - ポケット半径| < ピッチ円半径 < 外半径
    + ポケット半径` が要る。外周円がポケット円を丸ごと飲み込む寸法を渡された
    ときは、口が開くところまで外半径を落とす（ボトルは外から出入りするので、
    口が塞がった円盤は受け渡しに使えない）。
    """
    r_out = float(outer_r)
    rp = float(pocket_r)
    d = float(pcd_r)
    n = int(count)
    phase = np.radians(float(phase_deg))
    res = max(int(resolution), 24)

    if n < 1 or rp <= 0.0 or d <= 0.0:
        th = np.linspace(0.0, 2.0 * np.pi, res, endpoint=False)
        return np.column_stack([r_out * np.cos(th), r_out * np.sin(th)])

    r_out = min(r_out, d + rp * 0.55)
    r_out = max(r_out, d - rp * 0.50)

    # 外周円（中心 O・半径 r_out）とポケット円（中心 P・半径 rp・|OP| = d）の交点
    x_i = (d * d + r_out * r_out - rp * rp) / (2.0 * d)
    y_i = np.sqrt(max(r_out * r_out - x_i * x_i, 0.0))
    alpha = np.arctan2(y_i, x_i)               # O から見た口の半角
    beta = np.arctan2(y_i, x_i - d)            # P から見た交点の角
    if alpha * n >= np.pi * 0.98:
        raise ValueError("ポケットが隣と重なる。pocket_r を小さくするか pcd を大きくする")

    n_pocket = max(res // 3, 12)
    n_span = max(res // n, 8)
    pts = []
    for k in range(n):
        th = phase + 2.0 * np.pi * k / n
        px, py = d * np.cos(th), d * np.sin(th)
        # ポケット。交点 A(-beta) から円盤の内側を通って B(+beta) へ抜ける。
        # P まわりに時計回りに回れば、外へ開いた口の側は通らない
        g = np.linspace(-beta, beta - 2.0 * np.pi, n_pocket)
        pts.append(np.column_stack([px + rp * np.cos(th + g),
                                    py + rp * np.sin(th + g)]))
        # 次のポケットの入口までの外周。両端は交点そのものなので重ねない
        span = np.linspace(th + alpha, th + 2.0 * np.pi / n - alpha,
                           n_span, endpoint=False)[1:]
        pts.append(np.column_stack([r_out * np.cos(span), r_out * np.sin(span)]))
    return np.vstack(pts)


def star_wheel(outer_d=235.0, pockets=4, pocket_r=40.0, pcd=225.0, thickness=15.0,
               base_z=0.0, hub_d=70.0, hub_h=25.0, shaft_d=30.0, shaft_h=110.0,
               resolution=RES_FINE, matrix=None) -> dict:
    """供給・排出のスターホイール。ボトルを抱えて渡す間欠回転の円盤。

    ポケットは押し出しで本当に切り欠く。既定はボトル外径 68 に対して
    R40（片側 6 mm の逃げ）、ピッチ円 φ225 の 4 ポケット・外径 φ235。
    **呼ぶ側は既定値を使わず、寸法を lay から渡すこと**（契約）。ここの既定は
    単体で形を見るときのためのもので、いまの機械の寸法に合わせてある。
    以前は 6 ポケット・ピッチ円 φ160・外径 φ200 のままで、テーブルのピッチと
    合わせる前の設計を説明していた（4 ポケットにした理由は viz/README.md）。
    外径はポケットが外へ開く寸法に収まるよう `_notched_disc_outline()` 側で
    頭を抑える。
    戻り値は部位名 -> メッシュの辞書。

        disc  ポケット付きの円盤
        hub   中心ボス
        shaft 下へ伸びる軸
    """
    outline = _notched_disc_outline(outer_d / 2.0, pockets, pocket_r, pcd / 2.0,
                                    resolution=resolution)
    disc = extrude_polygon(outline, thickness, base_z=base_z)
    z_top = base_z + thickness
    parts_out = {
        "disc": disc,
        "hub": cylinder(hub_d / 2.0, hub_h, base_z=z_top, resolution=resolution),
        "shaft": cylinder(shaft_d / 2.0, shaft_h, base_z=base_z - shaft_h,
                          resolution=resolution),
    }
    return {k: place(v, matrix) for k, v in parts_out.items()}


def crescent_guide(inner_r, outer_r, start_deg, end_deg, height, base_z=0.0,
                   resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """三日月ガイド。指定した角度範囲だけの環状扇形を立てた板。

    スターホイール外周に沿う固定ガイドにも、テーブル外周のボトル押さえにも
    使う。部分角で作れることが肝で、供給と排出の位置だけガイドを切っておく。
    """
    ri = float(inner_r)
    ro = float(outer_r)
    a0 = np.radians(float(start_deg))
    a1 = np.radians(float(end_deg))
    if ro <= ri:
        return pv.PolyData()
    while a1 <= a0:
        a1 += 2.0 * np.pi
    if a1 - a0 > 2.0 * np.pi - 1e-6:           # 全周は端が重なるので少し開ける
        a1 = a0 + 2.0 * np.pi - 1e-3

    n = max(int(resolution * (a1 - a0) / (2.0 * np.pi)) + 1, 8)
    a = np.linspace(a0, a1, n)
    outer = np.column_stack([ro * np.cos(a), ro * np.sin(a)])
    inner = np.column_stack([ri * np.cos(a[::-1]), ri * np.sin(a[::-1])])
    return extrude_polygon(np.vstack([outer, inner]), height, base_z=base_z,
                           matrix=matrix)


def pocket_plate(plate_r=245.0, pcd=450.0, stations=8, pocket_r=37.0, thickness=10.0,
                 base_z=0.0, resolution=RES_FINE, matrix=None) -> pv.PolyData:
    """テーブル板の上に載るポケット板。外周に U 溝が stations 個開く。

    ボトルを受けて位置を決める板で、テーブルと一緒に回る。溝の中心は
    ピッチ円上。ボトル外径 68 に対して R37 が既定。
    """
    outline = _notched_disc_outline(plate_r, stations, pocket_r, pcd / 2.0,
                                    resolution=resolution)
    return extrude_polygon(outline, thickness, base_z=base_z, matrix=matrix)


# ---------------------------------------------------------------- 搬送


def conveyor(length, belt_width=110.0, frame_width=190.0, top_z=870.0,
             belt_thickness=8.0, frame_height=80.0,
             guide_inner_w=76.0, guide_h=70.0, guide_t=5.0,
             post_d=11.0, posts=3, matrix=None) -> dict:
    """搬送コンベア。x 方向へ流れ、y=0 が搬送の中心線。

    top_z は搬送面（チェーン上面）の高さ。テーブル上面と同じか 0〜2 低く取る。
    戻り値は部位名 -> メッシュの辞書。

        belt        チェーン（搬送面）
        frame       側枠と下の繋ぎ
        guides      サイドガイド 2 枚
        guide_posts ガイド支柱の丸棒とクランプブロック
    """
    ln = float(length)
    bw = float(belt_width)
    fw = float(frame_width)
    bt = float(belt_thickness)
    fh = float(frame_height)
    z_belt = float(top_z)

    belt = box((ln, bw, bt), center=(0.0, 0.0, z_belt - bt / 2.0))

    side_w = max((fw - bw) / 2.0, CONVEYOR_SIDE_W_MIN_MM)
    z_frame_top = z_belt - bt
    frame_parts = []
    for sy in (-1, 1):
        frame_parts.append(box((ln, side_w, fh),
                               center=(0.0, sy * (bw + side_w) / 2.0,
                                       z_frame_top - fh / 2.0)))
    # 下の繋ぎ。側枠だけだと宙に浮いた 2 本の棒に見える
    frame_parts.append(box((ln, fw, 14.0),
                           center=(0.0, 0.0, z_frame_top - fh + 7.0)))
    frame = merge(frame_parts)

    guides = merge([
        box((ln, guide_t, guide_h),
            center=(0.0, sy * (guide_inner_w + guide_t) / 2.0,
                    z_belt + 2.0 + guide_h / 2.0))
        for sy in (-1, 1)
    ])

    post_list = []
    n_post = max(int(posts), 1)
    # 丸棒はガイド板のすぐ外に立て、クランプブロックで板とつなぐ
    y_post = guide_inner_w / 2.0 + guide_t + 14.0
    for i in range(n_post):
        x = -ln / 2.0 + ln * (i + 0.5) / n_post
        for sy in (-1, 1):
            post_list.append(cylinder(post_d / 2.0, guide_h + 26.0,
                                      base_z=z_frame_top - 6.0,
                                      resolution=RES_COARSE,
                                      matrix=transform_matrix(
                                          translate=(x, sy * y_post, 0.0))))
            post_list.append(box((25.0, 25.0, 20.0),
                                 center=(x, sy * (y_post - 5.0),
                                         z_belt + 2.0 + guide_h * 0.6)))
    parts_out = {
        "belt": belt,
        "frame": frame,
        "guides": guides,
        "guide_posts": merge(post_list),
    }
    return {k: place(v, matrix) for k, v in parts_out.items()}


# ---------------------------------------------------------------- 小物


def control_box(size=(250.0, 180.0, 120.0), center=(0.0, 0.0, 0.0),
                matrix=None) -> dict:
    """制御盤。扉は +x 側を向く。合わせ目は薄い直方体の線で出す。

    戻り値は部位名 -> メッシュの辞書。

        body   箱本体
        door   わずかに前へ出た扉板
        seam   扉の合わせ目（暗い色を当てる細い枠）
        latch  取っ手
    """
    sx, sy, sz = (float(v) for v in size)
    cx, cy, cz = (float(v) for v in center)
    x_face = cx + sx / 2.0

    body = box((sx, sy, sz), center=(cx, cy, cz))
    door = box((3.0, sy * 0.90, sz * 0.88), center=(x_face + 1.5, cy, cz))
    # 扉板より一回り大きい枠を、扉より手前に薄く置いて合わせ目の線にする
    seam_w = 3.0
    dy = sy * 0.90 / 2.0
    dz = sz * 0.88 / 2.0
    seam = merge([
        box((1.5, sy * 0.90 + seam_w * 2.0, seam_w),
            center=(x_face + 3.2, cy, cz + sz_off))
        for sz_off in (dz + seam_w / 2.0, -dz - seam_w / 2.0)
    ] + [
        box((1.5, seam_w, sz * 0.88 + seam_w * 2.0),
            center=(x_face + 3.2, cy + sy_off, cz))
        for sy_off in (dy + seam_w / 2.0, -dy - seam_w / 2.0)
    ])
    latch = merge([
        box((8.0, 18.0, 55.0), center=(x_face + 7.0, cy - dy * 0.80, cz)),
        horizontal_cylinder(9.0, 22.0, axis="x",
                            center=(x_face + 15.0, cy - dy * 0.80, cz),
                            resolution=RES_COARSE),
    ])

    parts_out = {"body": body, "door": door, "seam": seam, "latch": latch}
    return {k: place(v, matrix) for k, v in parts_out.items()}


def estop_button(base_d=60.0, button_d=45.0, center=(0.0, 0.0, 0.0),
                 resolution=RES_COARSE, matrix=None) -> dict:
    """非常停止押しボタン。黄色い座と赤いきのこ頭。

    戻り値は部位名 -> メッシュの辞書。

        base   黄色の台座
        button 赤いきのこ頭
    """
    rb = float(base_d) / 2.0
    rk = float(button_d) / 2.0
    base_h = rb * 0.24
    base_mesh = revolve([(0.0, 0.0), (rb, 0.0), (rb, base_h * 0.6),
                         (rb * 0.86, base_h), (0.0, base_h)], resolution=resolution)
    # きのこ頭。首を細くして傘を張り出す
    neck = rk * 0.62
    z1 = base_h + rk * 0.30
    z2 = z1 + rk * 0.42
    button = revolve([(0.0, base_h), (neck, base_h), (neck, z1), (rk, z1),
                      (rk, z2), (rk * 0.80, z2 + rk * 0.16),
                      (0.0, z2 + rk * 0.16)], resolution=resolution)
    lay = transform_matrix(translate=center)
    parts_out = {"base": base_mesh, "button": button}
    return {k: place(place(v, lay), matrix) for k, v in parts_out.items()}


def adjuster_foot(pad_d=60.0, screw_d=18.0, height=60.0, center=(0.0, 0.0, 0.0),
                  resolution=RES_COARSE, matrix=None) -> pv.PolyData:
    """アジャスタ脚。丸座＋ロックナット＋ねじ軸。center は接地面。"""
    rp = float(pad_d) / 2.0
    rs = float(screw_d) / 2.0
    h = float(height)
    pad_h = rp * 0.22
    nut_h = rs * 0.9
    prof = [(0.0, 0.0), (rp, 0.0), (rp, pad_h * 0.7), (rp * 0.72, pad_h),
            (rs * 1.55, pad_h), (rs * 1.55, pad_h + nut_h),
            (rs, pad_h + nut_h), (rs, h), (0.0, h)]
    body = revolve(prof, resolution=resolution,
                   matrix=transform_matrix(translate=center))
    return place(body, matrix)


def cable_duct(length, size=(40.0, 40.0), axis="x", center=(0.0, 0.0, 0.0),
               wall=3.0, matrix=None) -> dict:
    """配線ダクト。U 形の樋と、その上に載る蓋。

    戻り値は部位名 -> メッシュの辞書。

        body 樋（底と側板）
        lid  蓋
    """
    ln = float(length)
    w, h = (float(v) for v in size)
    t = float(wall)
    body = merge([
        box((ln, w, t), center=(0.0, 0.0, -h / 2.0 + t / 2.0)),
        box((ln, t, h - t), center=(0.0, (w - t) / 2.0, t / 2.0)),
        box((ln, t, h - t), center=(0.0, -(w - t) / 2.0, t / 2.0)),
    ])
    lid = box((ln, w + t * 1.4, t * 1.2), center=(0.0, 0.0, h / 2.0 - t * 0.4))
    lay = _axis_matrix(axis, center, local_axis="x")
    parts_out = {"body": body, "lid": lid}
    return {k: place(place(v, lay), matrix) for k, v in parts_out.items()}


def drip_tray(size=(160.0, 100.0, 20.0), drain_d=15.0, wall=2.5,
              center=(0.0, 0.0, 0.0), matrix=None) -> pv.PolyData:
    """液受け皿。浅い箱と、底から下へ出る排液口。center は底板の下面。"""
    sx, sy, sz = (float(v) for v in size)
    t = float(wall)
    cx, cy, cz = (float(v) for v in center)
    parts_list = [box((sx, sy, t), center=(0.0, 0.0, t / 2.0))]
    for sgn in (-1, 1):
        parts_list.append(box((sx, t, sz - t), center=(0.0, sgn * (sy - t) / 2.0,
                                                       t + (sz - t) / 2.0)))
        parts_list.append(box((t, sy - t * 2.0, sz - t),
                              center=(sgn * (sx - t) / 2.0, 0.0, t + (sz - t) / 2.0)))
    drain_h = float(drain_d) * 1.4
    parts_list.append(cylinder(float(drain_d) / 2.0, drain_h + t, base_z=-drain_h,
                               resolution=RES_COARSE))
    body = merge(parts_list)
    return place(place(body, transform_matrix(translate=(cx, cy, cz))), matrix)


def name_plate(size=(60.0, 30.0, 1.5), center=(0.0, 0.0, 0.0),
               matrix=None) -> pv.PolyData:
    """銘板。薄板 1 枚。厚み方向は z なので、貼る面に合わせて matrix で倒す。"""
    return box(size, center=center, matrix=matrix)


def pipe_run(points_xyz, diameter, joint_ratio=1.35, resolution=RES_COARSE,
             matrix=None) -> pv.PolyData:
    """折れ線に沿った配管。液配管（φ12〜20）にもエアチューブ（φ6〜10）にも使う。

    各区間を円柱で埋め、折れ目には少し太い球を置いて継ぎ目を隠す。
    points_xyz は通過点の列で、2 点以上あればよい。
    """
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return pv.PolyData()
    r = float(diameter) / 2.0
    if r <= 0.0:
        return pv.PolyData()

    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        ln = float(np.linalg.norm(d))
        if ln < 1e-9:
            continue
        segs.append(place(cylinder(r, ln, base_z=0.0, resolution=resolution),
                          _direction_matrix(d, a)))
    # 折れ目。端点には置かない（管の口が膨らんで見えるため）
    for p in pts[1:-1]:
        segs.append(sphere(r * float(joint_ratio), center=p, resolution=resolution))
    return place(merge(segs), matrix)


# ---------------------------------------------------------------- 確認用


if __name__ == "__main__":
    # 部品を並べて 1 枚に焼く。書き出し先は引数で受ける（既定の出力先は持たない）。
    #     .venv/bin/python viz/parts.py /path/to/check.png
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("使い方: python viz/parts.py <出力する png のパス>")

    shelf = []

    def _put(obj, dx, dy):
        mat = transform_matrix(translate=(dx, dy, 0.0))
        items = obj.values() if isinstance(obj, dict) else [obj]
        for m in items:
            shelf.append(place(m, mat))

    _put(index_unit(), -700.0, 400.0)
    _put(gearmotor(origin=(0.0, 0.0, 120.0)), -250.0, 400.0)
    _put(coupling(center=(0.0, 0.0, 120.0)), 60.0, 400.0)
    _put(coupling_cover(center=(180.0, 0.0, 120.0)), 60.0, 400.0)
    _put(star_wheel(), 500.0, 400.0)
    _put(crescent_guide(210.0, 226.0, 20.0, 200.0, 90.0), 500.0, 400.0)
    _put(pocket_plate(), -350.0, -300.0)
    _put(conveyor(700.0, top_z=200.0), 400.0, -300.0)
    _put(control_box(center=(0.0, 0.0, 60.0)), 900.0, 300.0)
    _put(estop_button(), 900.0, 100.0)
    _put(adjuster_foot(), 980.0, 100.0)
    _put(cable_duct(400.0, center=(0.0, 0.0, 20.0)), 900.0, -100.0)
    _put(drip_tray(), 900.0, -250.0)
    _put(name_plate(center=(0.0, 0.0, 1.0)), 900.0, -350.0)
    _put(pipe_run([(850.0, -500.0, 0.0), (850.0, -500.0, 160.0),
                   (1050.0, -500.0, 160.0), (1050.0, -420.0, 60.0)], 16.0), 0.0, 0.0)

    pl = pv.Plotter(off_screen=True, window_size=(1200, 900))
    pl.set_background("#e8e8ea")
    for mesh in shelf:
        pl.add_mesh(mesh, color="#b8bcc0", smooth_shading=False)
    pl.camera_position = "iso"
    pl.reset_camera()
    pl.screenshot(sys.argv[1])
    print("wrote", sys.argv[1])
