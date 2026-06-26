"""
モンテカルロ法 / ランジュバンシミュレーションデータの統計解析ツール。

ジャックナイフ再標本化、ビニング解析、自己相関関数、積分自己相関時間の
推定を提供する。これらは、確率過程量子化シミュレーションで生成される
相関を持った時系列から信頼できる誤差を見積もるための標準的な道具である。
"""

import numpy as np


# ---------------------------------------------------------------------------
# ジャックナイフ再標本化
# ---------------------------------------------------------------------------

def jackknife(data, func=np.mean):
    """ジャックナイフ再標本化により誤差を推定する。

    引数
    ----
    data : 1 次元 np.ndarray
        測定値の配列。
    func : 呼び出し可能
        配列をスカラーに写す関数（デフォルト: np.mean）。

    戻り値
    ------
    (estimate, error) : float の組
        推定値 = func(data)
        誤差   = sqrt((N-1)/N * sum((1 点除外推定値_i - 再標本平均)^2))
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 2:
        return float(func(data)), 0.0

    estimate = float(func(data))

    # 1 点除外した再標本ごとの推定値
    resampled = np.empty(n)
    for i in range(n):
        resampled[i] = func(np.delete(data, i))

    mean_resampled = np.mean(resampled)
    error = np.sqrt((n - 1) / n * np.sum((resampled - mean_resampled) ** 2))

    return estimate, float(error)


# ---------------------------------------------------------------------------
# ビニング解析
# ---------------------------------------------------------------------------

def binning_analysis(data, max_bin_size=None):
    """自己相関を検出するためのビニング解析を行う。

    ビンサイズを 1, 2, 4, 8, ... と幾何級数的に増やしながら平均値の標準誤差を
    計算する。誤差がプラトーに達したら、そのビンサイズは連続する測定値を
    十分に無相関化できる大きさだと考えられる。

    引数
    ----
    data : 1 次元 np.ndarray
        測定値の配列。
    max_bin_size : int, 省略可
        試す最大のビンサイズ。デフォルトは len(data) // 4
        （意味のある誤差推定には少なくとも 4 個のビンが必要）。

    戻り値
    ------
    (bin_sizes, errors) : tuple of np.ndarrays
        bin_sizes : 試したビンサイズの配列（1, 2, 4, 8, ...）
        errors    : 対応する平均値の標準誤差
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 2:
        return np.array([1]), np.array([0.0])

    if max_bin_size is None:
        max_bin_size = max(1, n // 4)

    bin_sizes = []
    errors = []

    bs = 1
    while bs <= max_bin_size:
        # 完全なビンの個数
        n_bins = n // bs
        if n_bins < 2:
            break

        # データを再形成してビン平均を作る（末尾の不完全なビンは捨てる）
        binned = data[: n_bins * bs].reshape(n_bins, bs).mean(axis=1)

        sem = np.std(binned, ddof=1) / np.sqrt(n_bins)
        bin_sizes.append(bs)
        errors.append(sem)

        bs *= 2

    return np.array(bin_sizes), np.array(errors)


# ---------------------------------------------------------------------------
# 自己相関関数
# ---------------------------------------------------------------------------

def autocorrelation(data, max_lag=None):
    """規格化された自己相関関数を計算する。

    A(k) = <(x_i - <x>)(x_{i+k} - <x>)> / <(x_i - <x>)^2>

    この定義により A(0) = 1 となる。

    引数
    ----
    data : 1 次元 np.ndarray
        測定値の時系列。
    max_lag : int, 省略可
        計算する最大ラグ。デフォルトは len(data) // 4。

    戻り値
    ------
    np.ndarray
        ラグ 0, 1, ..., max_lag に対する自己相関値。
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 2:
        return np.array([1.0])

    if max_lag is None:
        max_lag = max(1, n // 4)
    max_lag = min(max_lag, n - 1)

    mean = np.mean(data)
    fluctuations = data - mean
    variance = np.dot(fluctuations, fluctuations) / n

    # 分散が 0 のデータに対する保護
    if variance == 0.0:
        return np.ones(max_lag + 1)

    acf = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        acf[k] = np.dot(fluctuations[: n - k], fluctuations[k:]) / (n - k)

    acf /= variance
    return acf


# ---------------------------------------------------------------------------
# 積分自己相関時間
# ---------------------------------------------------------------------------

def integrated_autocorrelation_time(data, max_lag=None):
    """積分自己相関時間を推定する。

    tau_int = 0.5 + sum_{k=1}^{max_lag} A(k)

    和は A(k) が初めて 0 未満になるラグで打ち切る。これは、裾のノイズが
    推定値を過大評価するのを避けるためである。

    引数
    ----
    data : 1 次元 np.ndarray
        測定値の時系列。
    max_lag : int, 省略可
        autocorrelation にそのまま渡される。

    戻り値
    ------
    float
        積分自己相関時間。
    """
    acf = autocorrelation(data, max_lag=max_lag)

    tau = 0.5
    for k in range(1, len(acf)):
        if acf[k] < 0.0:
            break
        tau += acf[k]

    return float(tau)
