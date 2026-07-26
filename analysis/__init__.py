"""検出層。仮想センサの信号から、異常を特徴量で拾えるかを試す。

* `features.py`   回転角リサンプル・次数比・包絡線・帯域パワー・衝撃検出
* `detect.py`     正常データから基準を作ってしきい値判定
* `dataset.py`    物理コアの実行と、センサ信号の合成
* `config.py`     解析側の取り決め（窓の長さ、帯域、何 σ か）
* `run_matrix.py` 故障 × センサ × 特徴量の総当たり
* `figures.py`    図

結果は `analysis/RESULTS.md` と `figs/analysis_*.png`。
"""
