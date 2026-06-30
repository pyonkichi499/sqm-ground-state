"""
サンプリングされた格子配位から物理量を抽出する関数群。

確率過程量子化（Parisi-Wu のランジュバン発展）で生成された
ユークリッド時間経路のアンサンブルに対して、以下を計算する。

- 2 点ユークリッド相関関数  C(tau) = <x(tau) x(0)>
- 有効質量（エネルギーギャップ） m_eff(tau) = (1/a) ln[C(tau)/C(tau+1)]
- 位置ヒストグラム  |psi_0(x)|^2 の近似
- <x^2>  ビリアル定理を通して基底状態エネルギーと関係する量
"""

import numpy as np


def correlator(configs, max_tau=None):
    """ユークリッド 2 点相関関数 C(tau) = <x(tau) x(0)> を計算する。

    並進不変性を利用し、すべての基準点とすべての配位について平均する。

        C[k] = configs と格子点 i に関する x[i] * x[(i+k) % N] の平均

    引数
    ----
    configs : np.ndarray, 形状 (n_configs, n_lattice)
        格子上の経路配位のアンサンブル。
    max_tau : int, 省略可
        計算する最大の距離。デフォルトは n_lattice // 2。

    戻り値
    ------
    np.ndarray, 形状 (max_tau + 1,)
        C[0], C[1], ..., C[max_tau]。
    """
    n_configs, n_lattice = configs.shape

    if max_tau is None:
        max_tau = n_lattice // 2

    corr = np.empty(max_tau + 1)
    for k in range(max_tau + 1):
        shifted = np.roll(configs, -k, axis=1)
        corr[k] = np.mean(configs * shifted)

    return corr


def effective_mass(corr, a=0.1):
    """相関関数から有効質量（エネルギーギャップ）を計算する。

        m_eff(tau) = (1/a) * ln(C(tau) / C(tau+1))

    tau が大きい領域では E_1 - E_0 に収束する。質量 m = 1 の調和振動子では、
    この値は omega に等しい。

    引数
    ----
    corr : np.ndarray
        correlator が返す相関関数 C(tau)。
    a : float
        ユークリッド時間方向の格子間隔。

    戻り値
    ------
    np.ndarray, 形状 (len(corr) - 1,)
        各 tau における有効質量。C(tau) <= 0 または C(tau+1) <= 0 の要素は
        np.nan になる。
    """
    c_tau = corr[:-1]
    c_tau1 = corr[1:]

    valid = (c_tau > 0) & (c_tau1 > 0)

    m_eff = np.full_like(c_tau, np.nan)
    m_eff[valid] = np.log(c_tau[valid] / c_tau1[valid]) / a

    return m_eff


def position_histogram(configs, n_bins=50):
    """x の値のヒストグラムを計算する（|psi_0(x)|^2 の近似）。

    すべての配位・すべての格子点の x の値を 1 つのサンプルに平坦化して
    ビン分けする。ヒストグラムは積分が 1 になるように規格化される
    （つまり sum(hist * bin_width) == 1）。

    引数
    ----
    configs : np.ndarray, 形状 (n_configs, n_lattice)
        格子上の経路配位のアンサンブル。
    n_bins : int
        ヒストグラムのビン数。

    戻り値
    ------
    bin_centers : np.ndarray, 形状 (n_bins,)
    hist_values : np.ndarray, 形状 (n_bins,)
        各ビン中心における規格化された確率密度。
    """
    x_flat = configs.ravel()
    hist_values, bin_edges = np.histogram(x_flat, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, hist_values


def mean_x_squared(configs):
    """すべての格子点と配位で平均した <x^2> を計算する。

    質量 m = 1 の調和振動子では、T -> 0 極限で次が成り立つ。

        <x^2> = 1 / (2 omega)

    引数
    ----
    configs : np.ndarray, 形状 (n_configs, n_lattice)
        格子上の経路配位のアンサンブル。

    戻り値
    ------
    float
        配位および格子点で平均した <x^2>。
    """
    return float(np.mean(configs**2))
