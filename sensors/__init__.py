"""仮想センサ層。

物理コアの出力（core/FORMAT.md のバイナリ）から、センサが実際に返すであろう
数字を作る。理想の物理量ではなく、取り付けの伝達・帯域制限・サンプリング・
ノイズ・量子化・レンジ飽和を通したものを出す。

- :mod:`sensors.chain`     連鎖の部品（段ごとに単体で試験できる）
- :mod:`sensors.virtual`   真値の合成と、連鎖を通した各センサの出力
- :mod:`sensors.read_dump` 物理コアのバイナリ読み込み
"""

__all__ = ["chain", "virtual", "read_dump"]
