"""連番 PNG を 1 本の GIF にまとめる。

この環境には ffmpeg も imageio も入っていない。使えるのは PIL だけなので、
GIF は PIL で直に書く。記事に貼る前提なので、大きさを 5 MB 以下に収めることを
目標にしてある。

    .venv/bin/python viz/make_gif.py                       # frames/ -> figs/anim_filler.gif
    .venv/bin/python viz/make_gif.py --step 3 --width 640  # 10 コマ/s に間引いて小さく
    .venv/bin/python viz/make_gif.py --frames frames_cover --out figs/anim_filler_cover.gif

フォルダは何本でも渡せる。視点ごとに分けて焼いた連番や `viz/compose.py` で
貼り合わせた連番を、まとめて GIF にするときはこちら。

    .venv/bin/python viz/make_gif.py --frames frames/iso frames/top frames/nozzle \\
        --out-dir figs --step 1
    .venv/bin/python viz/make_gif.py --frames frames_pair --step 1 \\
        --out figs/anim_pair.gif

複数渡したときの出力名は `--out-dir` の下に `anim_<フォルダ名>.gif` で作る。
`figs/` に書いてよいのは `anim_` で始まる名前だけ、という約束に合わせてある。

小さくする手は 3 つ。効く順に:

1. **コマを間引く**（`--step`）。焼いたコマ数のまま貼る必要はない。10〜20 コマ/s で足りる。
2. **幅を縮める**（`--width`。**縮めるだけで、元より大きい値を渡しても拡大しない**）。
3. **色数を減らす**（`--colors`）。全コマ共通のパレットを 1 つ作って使い回す。
   コマごとにパレットを作ると、同じ画素でも色番号が変わって差分が効かなくなる。

GIF は前のコマとの差分だけを書けるので、動いていない画素が多いほど小さくなる。
`optimize=True` を渡すと PIL が「前のコマと同じ画素」を透過色で塗ってくれる
（そのために色数は 256 未満にして、透過用の番号を 1 つ空けておく）。
背景と架台が 1 画素も動かないのは、同じ状態から同じ絵が出るように作ってあるから。

**差し色は放っておくとパレットから落ちる。** メディアンカットは画素数だけで色を
選ぶので、面積の小さい彩度の高い色（波形パネルの赤い時刻カーソル、シグナルタワーの
橙と黄）は 1 色も割り当てられず、灰色に丸められて消える。実測で 192 色でも 255 色でも
全コマ 0 px になった（3D 3 視点と波形を貼り合わせた 12 コマ・幅 640・192 色で、
時刻カーソル 1092 px -> 0 px、シグナルタワーの橙黄 768 px -> 0 px、液の水色は 2 割まで）。
枠を取ると 1316 px（120%）まで戻り、容量は 0.23 MB で変わらない。
**PNG では合っているので、PNG を見て通すと気付けない。**
なのでパレットの枠を差し色のぶんだけ先に取っておく（下の `accent_colors()` /
`accent_entries()` / `common_palette()`）。**どの色相が差し色か**は
`scene.MATERIAL` と `panels.COLOR` から拾うので、材質の色を振り分け直しても
付いてくる。ここには色を直書きしない。

コマ間隔は `--src-fps` / `--step` で決まる。**既定は 100 ms**（`--step 3` ×
連番 30 コマ/s）。30 コマ/s の連番からは `--step` だけでは 50 ms にできない
（30/step は 30 / 15 / 10 / 7.5 / … としか刻めない）。50 ms が要るなら連番を
20 コマ/s で焼く（`viz/animate.py --fps 20`）か、`--src-fps 20` で 20 コマ/s
として読ませる（後者は再生が 1.5 倍の早回しになる）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy                    # 差し色を数えるところだけで使う
from PIL import Image

VIZ_DIR = Path(__file__).resolve().parent
LAB_ROOT = VIZ_DIR.parent
PARAMS_PATH = LAB_ROOT / "params.json"
DEFAULT_FRAMES = LAB_ROOT / "frames"
DEFAULT_OUT = LAB_ROOT / "figs" / "anim_filler.gif"
DEFAULT_WIDTH = 640           # 記事に貼る幅。これより小さい連番は縮めない
PALETTE_SAMPLES = 12          # 共通パレットを作るのに使うコマ数
SIZE_BUDGET_MB = 5.0          # 記事に貼るときの上限
# params.json が読めなかったときの控え。viz/animate.py の既定と同じ値であること
FALLBACK_SRC_FPS = 30.0
# 差し色とみなす彩度の下限。機械はほぼ無彩色（架台 #7c848c で 0.11、ボトル 0.07）
# なので、0.25 で切ると赤・橙・黄・緑・水色だけが残る。灰色まで枠を取ると
# パレットを無駄に食う。
ACCENT_MIN_SAT = 0.25
# 彩度だけで切ると、暗い青灰（接地の暗がり #242b33 など）が差し色に混ざる。
# 暗い色は相対の彩度が上がりやすいので、最大と最小の差そのものにも下限を置く。
ACCENT_MIN_CHROMA = 40
# 材質の色から何度ぶん離れた色相まで「その差し色」とみなすか [deg]。
# 光が当たると色相も少し振れる（シグナルタワーの赤 #b5241d は 3deg、
# 絵に出てくる画素の平均は 4〜10deg あたり）。
ACCENT_HUE_TOL_DEG = 25.0
# 差し色に取っておくパレットの枠数。差し色 1 つにつき 2 枠（明るい側と暗い側）。
# 192 色のうち 24 枠なら 1 割強で、機械の灰色の階調は目に見えて痩せない。
ACCENT_SLOTS_PER_COLOR = 2


def default_src_fps() -> float:
    """元の連番が何コマ/s か。params.json の sim.frame_rate_hz が正典。

    viz/animate.py が既定でその値で焼くので、ここで別の数字を持つと GIF の
    コマ間隔だけが実時間とずれる（既定が 60 のままで 30 コマ/s の連番を
    渡すと、そのまま 2 倍速の GIF になる）。
    """
    try:
        with open(PARAMS_PATH, encoding="utf-8") as fp:
            return float(json.load(fp)["sim"]["frame_rate_hz"])
    except (OSError, KeyError, ValueError):
        return FALLBACK_SRC_FPS


def load_frames(frame_dir: Path, prefix="f", step=1, width=None,
                warn_enlarge=False) -> list:
    """連番 PNG を読んで、間引いて、縮める。

    `width` は**縮める側にだけ**効く。元より大きい値を渡しても拡大しない
    （拡大しても情報は増えず、GIF が重くなるだけ）。既定より小さい連番を
    渡したときにいちいち言われても困るので、断り書きは自分で `--width` を
    渡したとき（`warn_enlarge`）だけ出す。
    """
    paths = sorted(frame_dir.glob(f"{prefix}*.png"))
    if not paths:
        raise SystemExit(f"{frame_dir} に {prefix}*.png が無い。先に viz/animate.py を回す")
    paths = paths[::max(int(step), 1)]
    out = []
    warned = False
    for p in paths:
        img = Image.open(p).convert("RGB")
        if width and int(width) > img.width:
            if warn_enlarge and not warned:
                sys.stderr.write(
                    f"注意: --width {int(width)} は元の {img.width} px より大きい。"
                    "縮めるだけの指定なので、そのままの大きさで焼く\n")
                warned = True
        elif width and int(width) != img.width:
            h = round(img.height * width / img.width)
            img = img.resize((int(width), h), Image.LANCZOS)
        out.append(img)
    return out


# --------------------------------------------------------------------------
# 差し色
# --------------------------------------------------------------------------
def _to_rgb(spec):
    """色の指定を (R, G, B) の 0..255 に直す。読めなければ None。

    3D 側は `#rrggbb`、2D パネル側は matplotlib の名前（`tab:red` や `"0.45"`）
    なので、matplotlib があればそれに解かせる。無ければ `#rrggbb` だけ読む。
    """
    if isinstance(spec, (tuple, list)) and len(spec) >= 3:
        vals = list(spec)[:3]
        if all(isinstance(v, (int, float)) for v in vals):
            if max(vals) <= 1.0:
                return tuple(int(round(float(v) * 255.0)) for v in vals)
            return tuple(int(v) for v in vals)
        return None
    if not isinstance(spec, str):
        return None
    s = spec.strip()
    if s.startswith("#") and len(s) == 7:
        try:
            return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            return None
    try:
        from matplotlib.colors import to_rgb           # 2D パネルが使う色名
    except ImportError:
        return None
    try:
        return tuple(int(round(v * 255.0)) for v in to_rgb(s))
    except ValueError:
        return None


def _saturation(rgb) -> float:
    hi, lo = max(rgb), min(rgb)
    return 0.0 if hi == 0 else (hi - lo) / float(hi)


def accent_colors(min_sat=ACCENT_MIN_SAT, min_chroma=ACCENT_MIN_CHROMA) -> list:
    """絵に出てくる差し色を集める。**色は直書きしない。**

    出どころは 2 つ。3D は `scene.MATERIAL`（`asm_*.py` の `MATERIALS` も
    読み込み時にここへ合流する）、2D パネルは `panels.COLOR`。材質の色を
    振り分け直しても、ここは何も直さなくてよい。

    どちらかが読めなくても止めない（`asm_*.py` を書き換えている最中でも
    GIF は焼けること）。読めなかったら標準エラーに 1 行出して先へ進む。
    """
    if str(VIZ_DIR) not in sys.path:
        sys.path.insert(0, str(VIZ_DIR))
    specs = []
    try:
        import scene                                   # noqa: PLC0415 重いので必要なときだけ
        specs += [m.get("color") for m in scene.MATERIAL.values()]
    except Exception as exc:                           # noqa: BLE001 書きかけで壊れている
        sys.stderr.write(f"注意: scene の材質が読めない（{type(exc).__name__}: {exc}）。"
                         "3D の差し色が落ちるかもしれない\n")
    try:
        import panels                                  # noqa: PLC0415
        specs += list(panels.COLOR.values())
    except Exception as exc:                           # noqa: BLE001
        sys.stderr.write(f"注意: panels の色が読めない（{type(exc).__name__}: {exc}）。"
                         "波形の時刻カーソルが落ちるかもしれない\n")

    out = []
    for spec in specs:
        rgb = _to_rgb(spec)
        if rgb is None:
            continue
        if _saturation(rgb) < min_sat or (max(rgb) - min(rgb)) < min_chroma:
            continue
        if rgb not in out:
            out.append(rgb)
    return out


def palette_samples(frames: list) -> list:
    """パレットを決めるのに使うコマ。等間隔に抜く。

    液の水色は画面のごく一部なので、コマを絞りすぎると色を取りこぼす。
    """
    n = len(frames)
    return [frames[round(i * (n - 1) / max(PALETTE_SAMPLES - 1, 1))]
            for i in range(min(PALETTE_SAMPLES, n))]


def palette_sheet(frames: list) -> Image.Image:
    """色を選ばせる見本用のシート。抜いたコマを縦につないだ 1 枚。"""
    picks = palette_samples(frames)
    w, h = picks[0].size
    sheet = Image.new("RGB", (w, h * len(picks)))
    for i, img in enumerate(picks):
        sheet.paste(img, (0, i * h))
    return sheet


def _hue_deg(arr):
    """(..., 3) の RGB から色相 [deg] と彩度差（最大-最小）を出す。"""
    a = arr.astype("int16")
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    d = (mx - mn).astype("float32")
    safe = numpy.where(d > 0, d, 1.0)
    h = numpy.zeros(d.shape, dtype="float32")
    h = numpy.where(mx == r, ((g - b) / safe) % 6.0, h)
    h = numpy.where(mx == g, (b - r) / safe + 2.0, h)
    h = numpy.where(mx == b, (r - g) / safe + 4.0, h)
    return (h * 60.0) % 360.0, d


def accent_entries(frames: list, accents, slots: int,
                   hue_tol=ACCENT_HUE_TOL_DEG, min_chroma=ACCENT_MIN_CHROMA) -> list:
    """差し色に取っておく枠へ入れる色を、**絵に出てきた画素から**選ぶ。

    材質の色をそのまま入れても効かない。3D は光が当たって明暗も色相も振れるので、
    材質の #b5241d は絵の中に現れない（実測で、シグナルタワーの赤い画素の平均は
    #a85b56）。材質の色をそのまま枠に入れると、絵の側の赤はその枠にも届かず、
    結局いちばん近い灰色へ落ちた。

    そこで材質の色は**どの色相が差し色か**の目印にだけ使い、その色相の近く
    （±`hue_tol`）にある実際の画素を集めて、`slots` 色にまとめる。

    **枠は色相ごとに配る。** まとめて 1 回で選ぶと、面積の大きい差し色
    （波形の青い曲線、液の水色）が枠を全部取り、シグナルタワーの橙黄
    （4 面貼り合わせで 1 コマ 64 px）には 1 枠も回らなかった。
    """
    accents = list(accents or [])
    slots = int(slots)
    if not accents or slots < 1:
        return []
    per = max(1, min(slots // len(accents), slots))
    picks = palette_samples(frames)
    flat = numpy.concatenate([numpy.asarray(im).reshape(-1, 3) for im in picks])
    hue, chroma = _hue_deg(flat)
    enough = chroma >= min_chroma
    out = []
    for rgb in accents:
        h0 = _hue_deg(numpy.asarray(rgb, dtype="int16").reshape(1, 3))[0][0]
        delta = numpy.abs((hue - h0 + 180.0) % 360.0 - 180.0)
        sel = flat[enough & (delta <= hue_tol)]
        if sel.shape[0] < per * 4:                    # その色は写っていない
            continue
        strip = Image.fromarray(sel.reshape(-1, 1, 3).astype("uint8"), "RGB")
        q = strip.quantize(colors=per, method=Image.Quantize.MEDIANCUT,
                           dither=Image.Dither.NONE)
        pal = list(q.getpalette() or [])
        for i in sorted(set(numpy.asarray(q).reshape(-1).tolist())):
            entry = tuple(pal[3 * i:3 * i + 3])
            if len(entry) == 3 and entry not in out:
                out.append(entry)
    return out[:slots]


def common_palette(frames: list, colors: int, accents=None) -> Image.Image:
    """全コマ共通のパレットを作る。**差し色には先に枠を取っておく。**

    メディアンカットは画素数だけで色を選ぶので、面積の小さい彩度の高い色は
    1 枠ももらえない（実測で、赤い時刻カーソル 1092 px が全コマ 0 px、
    シグナルタワーの橙黄 768 px も 0 px、液の水色は 2 割まで痩せた）。
    絵から選ばせるのは `colors - 差し色の枠数` で、残りは `accent_entries()` が
    絵の中の差し色から埋める。
    """
    accents = list(accents if accents is not None else [])
    colors = max(int(colors), 2)
    slots = min(ACCENT_SLOTS_PER_COLOR * len(accents), max(colors // 4, 1))
    extra = accent_entries(frames, accents, slots)
    n_base = colors - len(extra)
    base = palette_sheet(frames).quantize(colors=n_base,
                                          method=Image.Quantize.MEDIANCUT,
                                          dither=Image.Dither.NONE)
    entries = list(base.getpalette() or [])[:3 * n_base]
    for rgb in extra:
        entries += [int(v) for v in rgb]
    # 余った枠は最後の色で埋める。0 で埋めると黒が増えて、暗い所がそこへ落ちる
    tail = entries[-3:] if entries else [0, 0, 0]
    entries += tail * ((768 - len(entries)) // 3)
    pal = Image.new("P", (1, 1))
    pal.putpalette(entries[:768])
    return pal


def write_gif(frames: list, out_path: Path, fps: float, colors=192,
              dither=False, accents=None) -> Path:
    """パレットをそろえてから 1 本の GIF にする。"""
    pal = common_palette(frames, colors, accents=accents)
    mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    quant = [f.quantize(palette=pal, dither=mode) for f in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quant[0].save(out_path, save_all=True, append_images=quant[1:],
                  duration=round(1000.0 / fps), loop=0, optimize=True)
    return out_path


def one_gif(frame_dir: Path, out_path: Path, prefix="f", src_fps=None, step=3,
            width=DEFAULT_WIDTH, colors=192, dither=False, accents=None,
            warn_enlarge=False) -> Path:
    """フォルダ 1 つぶんを GIF にして、大きさを標準エラーに出す。

    `accents` は差し色の並び。省略すると `accent_colors()` で集める
    （空の並びを渡せば枠を取らない。差し色が落ちる様子を見るとき用）。
    """
    if src_fps is None:
        src_fps = default_src_fps()
    if accents is None:
        accents = accent_colors()
    frames = load_frames(frame_dir, prefix=prefix, step=step, width=width or None,
                         warn_enlarge=warn_enlarge)
    fps = src_fps / max(int(step), 1)
    # GIF のコマ間隔は 10 ms 刻みしか持てない。PIL は指定 [ms] を 10 で割って
    # 切り捨てるので、割り切れないコマ数を選ぶと再生が速くなる（30 コマ/s =
    # 33.3 ms は 30 ms になって 11% 早回し）。20 コマ/s か 10 コマ/s なら丁度。
    ms = round(1000.0 / fps)
    if ms % 10 != 0:
        sys.stderr.write(
            f"注意: {fps:g} コマ/s は 1 コマ {1000.0 / fps:.1f} ms で、"
            f"10 ms 刻みに乗らない。GIF では {ms // 10 * 10} ms になり "
            f"{(ms / (ms // 10 * 10) - 1.0) * 100:.0f}% 早回しになる"
            "（20 コマ/s か 10 コマ/s にする）\n")

    out = write_gif(frames, Path(out_path), fps, colors=colors, dither=dither,
                    accents=accents)
    mb = out.stat().st_size / 1e6
    w, h = frames[0].size
    sys.stderr.write(
        f"{out}: {len(frames)} コマ / {w}x{h} / {fps:g} コマ/s / "
        f"{len(frames) / fps:.1f} s / {colors} 色 / 差し色 {len(accents)} / "
        f"{ms // 10 * 10} ms/コマ / {mb:.2f} MB\n")
    if mb > SIZE_BUDGET_MB:
        sys.stderr.write(
            f"注意: {SIZE_BUDGET_MB:g} MB を超えている。"
            " --step を増やす / --width を下げる / --colors を減らす\n")
    return out


def main(argv=None) -> int:
    src_fps = default_src_fps()
    ap = argparse.ArgumentParser(
        description="連番 PNG を GIF にまとめる",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "コマ間隔について:\n"
            f"  既定は {round(1000.0 * 3 / src_fps)} ms（--step 3 x 連番 {src_fps:g} コマ/s）。"
            "50 ms ではない。\n"
            f"  {src_fps:g} コマ/s の連番からは --step だけでは 50 ms にできない"
            f"（{src_fps:g}/step は "
            f"{' / '.join(f'{src_fps / s:g}' for s in (1, 2, 3, 4, 5, 6))} コマ/s）。\n"
            "  50 ms（20 コマ/s）が要るなら、連番を 20 コマ/s で焼く\n"
            "    .venv/bin/python viz/animate.py --fps 20 ...\n"
            "    .venv/bin/python viz/make_gif.py --src-fps 20 --step 1\n"
            "  か、30 コマ/s の連番を 20 コマ/s として読ませる（--src-fps 20 --step 1）。\n"
            "  後者は 1.5 倍の早回しになるので、時刻を書いた絵には使わない。\n"))
    ap.add_argument("--frames", type=Path, nargs="+", default=[DEFAULT_FRAMES],
                    help="PNG の置き場。複数渡せる")
    ap.add_argument("--prefix", default="f", help="連番の頭")
    ap.add_argument("--out", type=Path, default=None,
                    help="出力 GIF。フォルダを 1 つだけ渡したときに使う")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT.parent,
                    help="複数フォルダのときの置き場。anim_<フォルダ名>.gif で作る")
    ap.add_argument("--src-fps", type=float, default=None,
                    help="元の連番が何コマ/s か。"
                         f"既定は params.json の sim.frame_rate_hz（{src_fps:g}）")
    ap.add_argument("--step", type=int, default=3,
                    help=f"何コマに 1 枚使うか。既定 3 は {src_fps:g} コマ/s の連番で "
                         f"{round(1000.0 * 3 / src_fps)} ms/コマ（下の注記）")
    ap.add_argument("--width", type=int, default=None,
                    help=f"横幅 [px]。既定 {DEFAULT_WIDTH}、0 でそのまま。"
                         "縮めるだけで、元より大きい値は効かない")
    ap.add_argument("--colors", type=int, default=192,
                    help="パレットの色数。256 未満にすること（透過用に 1 つ空ける）")
    ap.add_argument("--no-accents", action="store_true",
                    help="差し色に枠を取らない。赤い時刻カーソルもシグナルタワーも "
                         "GIF から消えるので、比べるとき以外は使わない")
    ap.add_argument("--dither", action="store_true",
                    help="誤差拡散を掛ける（滑らかになるが 2〜3 倍に膨らむ）")
    args = ap.parse_args(argv)

    if args.colors >= 256:
        raise SystemExit("--colors は 255 以下にする（差分を透過で埋める番号が要る）")
    if args.out is not None and len(args.frames) > 1:
        raise SystemExit("--out はフォルダ 1 つのときだけ。複数なら --out-dir を使う")

    # 差し色は 1 回だけ集めて全部のフォルダで使い回す（scene の読み込みが重い）。
    accents = [] if args.no_accents else accent_colors()

    for frame_dir in args.frames:
        if args.out is not None:
            out_path = args.out
        elif len(args.frames) == 1 and frame_dir == DEFAULT_FRAMES:
            out_path = DEFAULT_OUT
        else:
            out_path = Path(args.out_dir) / f"anim_{Path(frame_dir).name}.gif"
        one_gif(frame_dir, out_path, prefix=args.prefix, src_fps=args.src_fps,
                step=args.step,
                width=DEFAULT_WIDTH if args.width is None else args.width,
                colors=args.colors, dither=args.dither, accents=accents,
                warn_enlarge=args.width is not None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
