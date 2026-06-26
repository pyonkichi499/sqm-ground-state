"""
ユークリッド時間格子上の作用とドリフト力の数値カーネル。

ここはポテンシャルに依存しない汎用処理である。ポテンシャルやその導関数は
呼び出し可能オブジェクト（callable）として渡す。

規約
----
- 格子点数は N、格子間隔は a とし、周期境界条件 x[N] = x[0] を課す。
- ユークリッド作用:
      S_E = sum_i [ m/(2a) * (x[i+1] - x[i])^2  +  a * V(x[i]) ]
- 格子点 j におけるドリフト力:
      f_j = -dS_E/dx_j = m/a * (x[j+1] + x[j-1] - 2*x[j])  -  a * V'(x[j])
- 自然単位系 hbar = 1 を用いる。
"""

import numpy as np


def euclidean_action(x, potential_func, mass=1.0, a=0.1, **pot_params):
    """経路 x に対するユークリッド作用 S_E を計算する。

    引数
    ----
    x : 1 次元 numpy 配列
        N 個の格子点を持つ経路。周期境界条件 x[N] = x[0] を仮定する。
    potential_func : 呼び出し可能
        ポテンシャル関数 V(x)。
    mass : float
        粒子の質量。
    a : float
        ユークリッド時間方向の格子間隔。
    **pot_params
        potential_func に渡す追加のキーワード引数。

    戻り値
    ------
    float
        離散化されたユークリッド作用。
    """
    dx = np.roll(x, -1) - x          # 周期境界条件つきの x[i+1] - x[i]
    kinetic = mass / (2.0 * a) * np.sum(dx**2)
    potential = a * np.sum(potential_func(x, **pot_params))
    return kinetic + potential


def drift_force(x, force_func, mass=1.0, a=0.1, **pot_params):
    """全格子点に対するランジュバンのドリフト力を計算する。

    引数
    ----
    x : 1 次元 numpy 配列
        N 個の格子点を持つ経路。周期境界条件 x[N] = x[0] を仮定する。
    force_func : 呼び出し可能
        ポテンシャルの導関数 dV/dx。
    mass : float
        粒子の質量。
    a : float
        ユークリッド時間方向の格子間隔。
    **pot_params
        force_func に渡す追加のキーワード引数。

    戻り値
    ------
    numpy 配列（x と同じ形状）
        f_j = (m/a)*(x[j+1] + x[j-1] - 2*x[j])  -  a * V'(x[j])
    """
    laplacian = np.roll(x, -1) + np.roll(x, 1) - 2.0 * x
    return (mass / a) * laplacian - a * force_func(x, **pot_params)
