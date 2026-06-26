"""
離散化されたユークリッド時間格子上での確率過程量子化に用いる、
ポテンシャル、導関数、ユークリッド作用、ランジュバンのドリフト力を定義する。

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


# ---------------------------------------------------------------------------
# ポテンシャル V(x)
# ---------------------------------------------------------------------------

def harmonic_potential(x, omega=1.0):
    """調和振動子ポテンシャル V(x) = 0.5 * omega^2 * x^2 を返す。"""
    return 0.5 * omega**2 * x**2


def anharmonic_potential(x, omega=1.0, lam=1.0):
    """非調和振動子ポテンシャル V(x) = 0.5 * omega^2 * x^2 + lam * x^4 を返す。"""
    return 0.5 * omega**2 * x**2 + lam * x**4


def double_well_potential(x, lam=1.0, v=1.0):
    """二重井戸ポテンシャル V(x) = lam * (x^2 - v^2)^2 を返す。"""
    return lam * (x**2 - v**2)**2


# ---------------------------------------------------------------------------
# ポテンシャルの導関数 V'(x)
# ---------------------------------------------------------------------------

def harmonic_force(x, omega=1.0):
    """調和振動子ポテンシャルの導関数 dV/dx を返す。"""
    return omega**2 * x


def anharmonic_force(x, omega=1.0, lam=1.0):
    """非調和振動子ポテンシャルの導関数 dV/dx を返す。"""
    return omega**2 * x + 4.0 * lam * x**3


def double_well_force(x, lam=1.0, v=1.0):
    """二重井戸ポテンシャルの導関数 dV/dx を返す。"""
    return 4.0 * lam * x * (x**2 - v**2)


# ---------------------------------------------------------------------------
# ユークリッド作用
# ---------------------------------------------------------------------------

def euclidean_action(x, potential_func, mass=1.0, a=0.1, **pot_params):
    """経路 x に対するユークリッド作用 S_E を計算する。

    引数
    ----
    x : 1 次元 numpy 配列
        N 個の格子点を持つ経路。周期境界条件 x[N] = x[0] を仮定する。
    potential_func : 呼び出し可能
        上で定義したポテンシャル関数のいずれか。
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


# ---------------------------------------------------------------------------
# ドリフト力 f_j = -dS_E / dx_j（格子全体をベクトル化して計算）
# ---------------------------------------------------------------------------

def drift_force(x, force_func, mass=1.0, a=0.1, **pot_params):
    """全格子点に対するランジュバンのドリフト力を計算する。

    引数
    ----
    x : 1 次元 numpy 配列
        N 個の格子点を持つ経路。周期境界条件 x[N] = x[0] を仮定する。
    force_func : 呼び出し可能
        ポテンシャルの導関数 dV/dx（例: harmonic_force）。
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
