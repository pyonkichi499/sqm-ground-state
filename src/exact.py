"""
1 次元量子力学系の検証用の厳密対角化（有限差分）ツール。

確率過程量子化の結果を既知解または独立な数値計算と比較するため、
位置空間を有限区間で格子化し、シュレーディンガー方程式のハミルトニアンを
行列として構成して対角化する。
"""

import numpy as np


def finite_difference_hamiltonian(x_grid, potential_func, mass=1.0, **pot_params):
    """1 次元ハミルトニアン行列を 2 次精度の有限差分で構成する。

    連続系のハミルトニアン

        H = -1/(2m) d^2/dx^2 + V(x)

    を、等間隔の位置格子上で離散化する。境界は箱の端で波動関数が 0 になる
    ディリクレ境界条件として扱う。

    引数
    ----
    x_grid : np.ndarray
        等間隔の位置格子。
    potential_func : callable
        ポテンシャル関数 V(x)。
    mass : float
        粒子の質量。
    **pot_params
        potential_func に渡す追加パラメータ。

    戻り値
    ------
    np.ndarray
        離散化されたハミルトニアン行列。
    """
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1:
        raise ValueError("x_grid は 1 次元配列である必要がある")
    if len(x_grid) < 3:
        raise ValueError("x_grid には少なくとも 3 点が必要")

    dx_values = np.diff(x_grid)
    dx = dx_values[0]
    if not np.allclose(dx_values, dx):
        raise ValueError("x_grid は等間隔である必要がある")

    n = len(x_grid)

    kinetic_diag = np.full(n, 1.0 / (mass * dx**2))
    kinetic_offdiag = np.full(n - 1, -1.0 / (2.0 * mass * dx**2))
    potential_diag = potential_func(x_grid, **pot_params)

    hamiltonian = np.diag(kinetic_diag + potential_diag)
    hamiltonian += np.diag(kinetic_offdiag, k=1)
    hamiltonian += np.diag(kinetic_offdiag, k=-1)
    return hamiltonian


def solve_spectrum(
    potential_func,
    x_min=-8.0,
    x_max=8.0,
    n_grid=400,
    mass=1.0,
    n_levels=4,
    **pot_params,
):
    """有限差分ハミルトニアンを対角化して低エネルギー準位を求める。

    引数
    ----
    potential_func : callable
        ポテンシャル関数 V(x)。
    x_min, x_max : float
        位置空間の計算範囲。
    n_grid : int
        位置格子点数。
    mass : float
        粒子の質量。
    n_levels : int
        返す低エネルギー準位の数。
    **pot_params
        potential_func に渡す追加パラメータ。

    戻り値
    ------
    (energies, x_grid, wavefunctions) : tuple
        energies は低い順の固有エネルギー、wavefunctions は対応する固有ベクトル。
        波動関数は sum(|psi|^2) dx = 1 となるように規格化する。
    """
    if x_max <= x_min:
        raise ValueError("x_max は x_min より大きい必要がある")
    if n_grid < 3:
        raise ValueError("n_grid は 3 以上である必要がある")
    if n_levels < 1:
        raise ValueError("n_levels は 1 以上である必要がある")

    x_grid = np.linspace(x_min, x_max, n_grid)
    hamiltonian = finite_difference_hamiltonian(
        x_grid,
        potential_func,
        mass=mass,
        **pot_params,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    n_keep = min(n_levels, len(eigenvalues))
    energies = eigenvalues[:n_keep]
    wavefunctions = eigenvectors[:, :n_keep]

    dx = x_grid[1] - x_grid[0]
    for i in range(n_keep):
        norm = np.sqrt(np.sum(np.abs(wavefunctions[:, i]) ** 2) * dx)
        wavefunctions[:, i] /= norm

    return energies, x_grid, wavefunctions


def ground_state_energy(potential_func, **kwargs):
    """有限差分対角化で基底状態エネルギーを返す。"""
    energies, _, _ = solve_spectrum(potential_func, n_levels=1, **kwargs)
    return float(energies[0])
