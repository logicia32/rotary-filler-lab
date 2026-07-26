"""判定。正常データから基準を作り、しきい値で異常を出す。

考え方
------
特徴量ごとに、正常運転のデータを 1 回転ぶんずつの窓に切って値を並べ、
その平均と標準偏差を基準にする。試験データの値がしきい値

    しきい値 = 平均 + k * 標準偏差

を超えたら異常とする。片側にしてあるのは、ここで使う特徴量が
「振幅」「実効値」「回数」で、どれも大きくなる方向にしか異常が出ないため。

k をどう決めたか
----------------
特徴量は 1 回転（19.2 s）に 1 個出るので、1 日あたり 4500 個。
誤警報を月 1 回未満に抑えたいなら、1 回あたりの片側確率は

    1 / (4500 * 30) = 7.4e-6

が要る。基準の平均と標準偏差は有限個（既定 22 個）の標本から推定した値なので、
正規分布の分位点ではなく予測区間を使う。自由度 n-1 の t 分布で

    k = t(1 - p, n-1) * sqrt(1 + 1/n)

自由度 21・p = 7.4e-6 で k = 5.7。丸めて 6 σ を既定にする。
基準の個数が変わっても k が揺れないように、既定は固定値の 6 σ にして、
上の計算値は報告に併記する。

この誤警報率の見積もりが成り立つ前提（記事にも必ず書くこと）
------------------------------------------------------------
1. 正常の特徴量が正規分布に従うこと。振幅や実効値は 0 以上なので、
   本当は右に裾を引く。`normality_note` で歪度を出して、目安が壊れていないか見る。
2. 窓どうしが独立であること。ここでは重ならない窓を使っているので概ね成り立つ。
3. 物理コアは決定論的なので、正常データどうしの違いはセンサ雑音しかない。
   実機なら、液の入り方・ホルダの個体差・温度で、これよりずっと大きく散る。
   したがってここで出る誤警報率は楽観側の値で、実機の値ではない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Baseline:
    """正常データから作った基準。"""

    name: str
    unit: str
    values: np.ndarray
    mean: float
    std: float
    n: int

    @property
    def usable(self) -> bool:
        return self.n >= 3 and np.isfinite(self.mean) and self.std > 0

    def threshold(self, k_sigma: float) -> float:
        return self.mean + k_sigma * self.std

    def zscore(self, value: float) -> float:
        if self.std <= 0:
            return float("inf") if value > self.mean else 0.0
        return (float(value) - self.mean) / self.std

    def snr_db(self, value: float) -> float:
        """基準の平均に対する比 [dB]。雑音床からどれだけ出ているか。"""
        if self.mean <= 0 or not np.isfinite(value) or value <= 0:
            return float("nan")
        return 20.0 * np.log10(value / self.mean)

    def skew(self) -> float:
        return float(stats.skew(self.values)) if self.n >= 3 else float("nan")


def build_baseline(values, name: str = "", unit: str = "") -> Baseline:
    """正常データの窓ごとの値から基準を作る。"""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return Baseline(name, unit, v, float("nan"), float("nan"), 0)
    return Baseline(name=name, unit=unit, values=v,
                    mean=float(np.mean(v)),
                    std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    n=len(v))


def sigma_for_false_alarm(n_baseline: int, decisions_per_day: float,
                          alarms_per_month: float = 1.0) -> tuple[float, float]:
    """誤警報の目標から、何 σ にすべきかを出す。

    返り値は (k, 片側確率 p)。標本数 n の推定を使うので t 分布の予測区間。
    """
    p = alarms_per_month / (decisions_per_day * 30.0)
    if n_baseline < 3:
        return float("nan"), p
    k = float(stats.t.ppf(1.0 - p, df=n_baseline - 1) * np.sqrt(1.0 + 1.0 / n_baseline))
    return k, p


def false_alarm_rate(k_sigma: float, n_baseline: int) -> float:
    """しきい値 k σ のときの、1 回の判定あたりの誤警報確率（片側）。"""
    if n_baseline < 3:
        return float("nan")
    t = k_sigma / np.sqrt(1.0 + 1.0 / n_baseline)
    return float(stats.t.sf(t, df=n_baseline - 1))


@dataclass
class Verdict:
    """1 つの特徴量に対する判定。"""

    feature: str
    channel: str
    unit: str
    value: float
    baseline_mean: float
    baseline_std: float
    threshold: float
    z: float
    snr_db: float
    detected: bool
    note: str = ""

    def mark(self) -> str:
        if self.note and not np.isfinite(self.value):
            return "測れない"
        return "検出" if self.detected else "見えない"


def judge(value: float, baseline: Baseline, k_sigma: float,
          feature: str = "", channel: str = "", note: str = "") -> Verdict:
    """基準に照らして 1 個の値を判定する。"""
    if not baseline.usable or not np.isfinite(value):
        return Verdict(feature, channel, baseline.unit, float(value),
                       baseline.mean, baseline.std, float("nan"),
                       float("nan"), float("nan"), False,
                       note=note or "基準が作れない")
    thr = baseline.threshold(k_sigma)
    z = baseline.zscore(value)
    return Verdict(feature=feature, channel=channel, unit=baseline.unit,
                   value=float(value), baseline_mean=baseline.mean,
                   baseline_std=baseline.std, threshold=thr, z=z,
                   snr_db=baseline.snr_db(value), detected=bool(value > thr),
                   note=note)


def normality_note(baseline: Baseline) -> str:
    """基準の分布が正規からどれだけ外れているかを一言で返す。"""
    if baseline.n < 8:
        return f"標本 {baseline.n} 個。正規性は判断できない"
    sk = baseline.skew()
    try:
        _, p = stats.shapiro(baseline.values)
    except Exception:
        p = float("nan")
    verdict = "正規から大きく外れる" if (np.isfinite(p) and p < 0.01) else "正規として扱ってよい範囲"
    return f"歪度 {sk:+.2f} / Shapiro-Wilk p={p:.3g} / {verdict}"
