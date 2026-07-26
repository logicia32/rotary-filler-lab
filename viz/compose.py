"""連番 PNG のフォルダを何本か受けて、1 本の連番に貼り合わせる。

この環境には ffmpeg も imageio も入っていない。使えるのは PIL だけなので、
貼り合わせも PIL で直に書く。出した連番はそのまま `viz/make_gif.py` に渡せる。

    .venv/bin/python viz/compose.py frames/iso frames/sensors --out frames_pair
    .venv/bin/python viz/compose.py frames/top frames/iso frames/nozzle \\
        --out frames_three --labels top,iso,nozzle
    .venv/bin/python viz/compose.py frames/normal frames/fault --out frames_ab \\
        --labels normal,fault

用途は 3 つ。

1. **3D の絵と波形パネル。** 記事の主役。同じ時刻のコマどうしを並べるので、
   連番の i 番目が同じ時刻を指していることが前提になる。時刻を揃えるのは
   焼く側（`viz/animate.py`）の仕事で、ここは番号でしか揃えない。
2. **3D の複数視点の横並び。** 同じ 1 回の動きを別の角度から同時に見せる。
3. **正常と異常の左右並置。** 同じ時間軸の 2 本を並べて差を見る。

大きさの違う連番を並べられるようにしてある。3D は縦長、波形パネルは横長で、
そもそも縦横比が違うのが普通。小さいほうを引き伸ばすか（`fit="scale"`）、
余白で埋めるか（`fit="pad"`）を選べる。

コマ数が食い違うときは短いほうに合わせて切るが、**何コマ捨てたかは必ず
標準エラーに出す。** 黙って切ると、片方の焼き直しに失敗していても気付けない。

図に載せる文字は英字だけにしてある。PIL の既定フォントに日本語が無く、
そのまま書くと豆腐になる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ラベルの帯。既定フォントは小さいので、字の高さに少しだけ余白を足す。
LABEL_PAD_PX = 4
LABEL_COLOR = "#20262c"
DEFAULT_BG = "#cfd6dc"      # scene.BACKGROUND と同じ。継ぎ目が目立たない


def frame_lists(dirs, prefix="f") -> list:
    """各フォルダの連番を並べて返す。1 枚も無いフォルダがあれば止める。"""
    out = []
    for d in dirs:
        d = Path(d)
        paths = sorted(d.glob(f"{prefix}*.png"))
        if not paths:
            raise SystemExit(f"{d} に {prefix}*.png が無い")
        out.append(paths)
    return out


def _font():
    """PIL の既定フォント。外部のフォントファイルは読まない。"""
    try:
        return ImageFont.load_default(size=16)
    except TypeError:
        # 古い PIL には size が無い
        return ImageFont.load_default()


def _text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _check_labels(labels, n: int) -> list:
    if labels is None:
        return [None] * n
    labels = list(labels)
    if len(labels) != n:
        raise ValueError(f"ラベルの数が合わない: {len(labels)} 個 / パネル {n} 枚")
    for lab in labels:
        if lab is None:
            continue
        if not str(lab).isascii():
            raise ValueError(f"ラベルは英字と数字だけ（既定フォントに日本語が無い）: {lab!r}")
    return labels


def _cell_boxes(sizes, layout: str, cols: int | None):
    """パネルの並べ方を決める。返すのは (行, 列) の割り当てと、行列ごとの大きさ。"""
    n = len(sizes)
    if layout == "h":
        grid = [(0, i) for i in range(n)]
        n_rows, n_cols = 1, n
    elif layout == "v":
        grid = [(i, 0) for i in range(n)]
        n_rows, n_cols = n, 1
    elif layout == "grid":
        n_cols = int(cols) if cols else int(n ** 0.5 + 0.999)
        n_cols = max(1, min(n_cols, n))
        n_rows = (n + n_cols - 1) // n_cols
        grid = [(i // n_cols, i % n_cols) for i in range(n)]
    else:
        raise ValueError(f"知らない並べ方: {layout!r}（h / v / grid）")
    return grid, n_rows, n_cols


def _scaled(sizes, layout: str, fit: str):
    """`fit="scale"` のとき、小さいほうを引き伸ばした後の大きさを返す。

    横並びなら高さを、縦並びなら幅を、いちばん大きいものに揃える。縦横比は
    変えない（変えると機械の丸いテーブルが楕円になる）。
    """
    if fit != "scale":
        return list(sizes)
    if layout == "h":
        h = max(s[1] for s in sizes)
        return [(max(1, round(w * h / hh)), h) for w, hh in sizes]
    if layout == "v":
        w = max(s[0] for s in sizes)
        return [(w, max(1, round(hh * w / ww))) for ww, hh in sizes]
    # 格子は面積で揃えると崩れるので、いちばん背の高いものに高さを合わせる
    h = max(s[1] for s in sizes)
    return [(max(1, round(w * h / hh)), h) for w, hh in sizes]


def side_by_side(dirs, out_dir, layout="h", gap=8, bg=DEFAULT_BG,
                 labels=None, align="center", fit="pad", prefix="f",
                 out_prefix="f", cols=None) -> int:
    """連番のフォルダを貼り合わせて、1 本の連番として書き出す。コマ数を返す。

    dirs   : 連番 PNG のフォルダの並び。i 番目どうしが同じ時刻である前提。
    layout : "h" 横並び / "v" 縦並び / "grid" 格子
    gap    : パネルの間と外周の余白 [px]
    bg     : 余白の色
    labels : パネルの上に載せる英字ラベル。None で載せない
    align  : 大きさが違うときの寄せ方。"center" / "start" / "end"
             （横並びなら上下方向、縦並びなら左右方向に効く）
    fit    : "pad" 余白で埋める / "scale" 小さいほうを引き伸ばして揃える
    cols   : layout="grid" のときの列数。None なら正方形に近い形にする
    """
    dirs = [Path(d) for d in dirs]
    if not dirs:
        raise ValueError("フォルダが 1 つも無い")
    if align not in ("center", "start", "end"):
        raise ValueError(f"知らない寄せ方: {align!r}（center / start / end）")
    if fit not in ("pad", "scale"):
        raise ValueError(f"知らない揃え方: {fit!r}（pad / scale）")

    lists = frame_lists(dirs, prefix=prefix)
    counts = [len(p) for p in lists]
    n = min(counts)
    if max(counts) != n:
        # 黙って切らない。片方の焼き直しが途中で落ちていてもここで気付ける
        for d, c in zip(dirs, counts):
            if c > n:
                sys.stderr.write(f"注意: {d} は {c} コマあり、{c - n} コマ捨てて {n} に合わせた\n")
    labels = _check_labels(labels, len(dirs))

    # 貼り位置は先頭のコマの大きさで決める。途中で大きさが変わっている連番は
    # 焼き直しが混ざっているので、先頭に合わせて縮めたうえで標準エラーに出す。
    with Image.open(lists[0][0]) as im0:
        base_sizes = [im0.size]
    for paths in lists[1:]:
        with Image.open(paths[0]) as im:
            base_sizes.append(im.size)
    sizes = _scaled(base_sizes, layout, fit)

    font = _font()
    lab_h = 0
    if any(lab for lab in labels):
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        lab_h = max(_text_size(probe, str(lab), font)[1]
                    for lab in labels if lab) + LABEL_PAD_PX * 2

    grid, n_rows, n_cols = _cell_boxes(sizes, layout, cols)
    col_w = [0] * n_cols
    row_h = [0] * n_rows
    for (r, c), (w, h) in zip(grid, sizes):
        col_w[c] = max(col_w[c], w)
        row_h[r] = max(row_h[r], h + lab_h)

    total_w = sum(col_w) + gap * (n_cols + 1)
    total_h = sum(row_h) + gap * (n_rows + 1)
    x0 = [gap + sum(col_w[:c]) + gap * c for c in range(n_cols)]
    y0 = [gap + sum(row_h[:r]) + gap * r for r in range(n_rows)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in sorted(out_dir.glob(f"{out_prefix}*.png")):
        old.unlink()

    warned = set()
    for i in range(n):
        canvas = Image.new("RGB", (total_w, total_h), bg)
        draw = ImageDraw.Draw(canvas)
        for k, paths in enumerate(lists):
            r, c = grid[k]
            w, h = sizes[k]
            img = Image.open(paths[i]).convert("RGB")
            if img.size != (w, h):
                if img.size != base_sizes[k] and k not in warned:
                    sys.stderr.write(
                        f"注意: {dirs[k]} は途中で大きさが変わっている"
                        f"（{base_sizes[k]} -> {img.size}）。先頭に合わせて縮めた\n")
                    warned.add(k)
                img = img.resize((w, h), Image.LANCZOS)
            # 余った幅・高さの寄せ方。横並びでは上下、縦並びでは左右に効く
            free_x = col_w[c] - w
            free_y = row_h[r] - lab_h - h
            if align == "center":
                dx, dy = free_x // 2, free_y // 2
            elif align == "start":
                dx, dy = 0, 0
            else:
                dx, dy = free_x, free_y
            px, py = x0[c] + dx, y0[r] + lab_h + dy
            canvas.paste(img, (px, py))
            img.close()
            if labels[k]:
                tw, th = _text_size(draw, str(labels[k]), font)
                draw.text((px + (w - tw) // 2, y0[r] + LABEL_PAD_PX),
                          str(labels[k]), fill=LABEL_COLOR, font=font)
        canvas.save(out_dir / f"{out_prefix}{i:04d}.png")
        canvas.close()

    sys.stderr.write(
        f"{out_dir}: {n} コマ / {total_w}x{total_h} / {len(dirs)} パネル "
        f"({layout}, fit={fit}, align={align})\n")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="連番 PNG のフォルダを貼り合わせる")
    ap.add_argument("dirs", nargs="+", type=Path, help="連番 PNG のフォルダ")
    ap.add_argument("--out", type=Path, required=True, help="貼り合わせた連番の置き場")
    ap.add_argument("--layout", choices=("h", "v", "grid"), default="h",
                    help="並べ方。h 横 / v 縦 / grid 格子")
    ap.add_argument("--cols", type=int, default=None, help="grid の列数")
    ap.add_argument("--gap", type=int, default=8, help="パネルの間と外周の余白 [px]")
    ap.add_argument("--bg", default=DEFAULT_BG, help="余白の色")
    ap.add_argument("--labels", default=None,
                    help="パネルの上に載せる英字ラベル（カンマ区切り）")
    ap.add_argument("--align", choices=("center", "start", "end"), default="center",
                    help="大きさが違うときの寄せ方")
    ap.add_argument("--fit", choices=("pad", "scale"), default="pad",
                    help="pad 余白で埋める / scale 小さいほうを引き伸ばす")
    ap.add_argument("--prefix", default="f", help="読む連番の頭")
    ap.add_argument("--out-prefix", default="f", help="書く連番の頭")
    args = ap.parse_args(argv)

    labels = args.labels.split(",") if args.labels else None
    side_by_side(args.dirs, args.out, layout=args.layout, gap=args.gap, bg=args.bg,
                 labels=labels, align=args.align, fit=args.fit,
                 prefix=args.prefix, out_prefix=args.out_prefix, cols=args.cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
