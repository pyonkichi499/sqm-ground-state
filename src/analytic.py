"""
解析的に分かる量をまとめるヘルパー。

完全な解析解がある系ではエネルギーや期待値を返す。
完全な解析解がない系では、古典的に分かる量や局所近似で分かる量を返し、
スペクトルの厳密な解析解は利用できないことを明示する。
"""

import numpy as np


def harmonic_analytic_results(omega=1.0, mass=1.0):
    """調和振動子の解析解を返す。

    このプロジェクトの調和振動子ポテンシャルは

        V(x) = 1/2 * omega^2 * x^2

    である。標準的な V(x) = 1/2 * m * Omega^2 * x^2 と比べると、
    物理的な角振動数は Omega = omega / sqrt(mass) である。
    mass=1 では Omega = omega になる。
    """
    physical_omega = omega / np.sqrt(mass)
    alpha = mass * physical_omega
    return {
        "available": True,
        "description": "調和振動子の厳密解",
        "E0": 0.5 * physical_omega,
        "gap": physical_omega,
        "x2": 1.0 / (2.0 * alpha),
        "density_prefactor": np.sqrt(alpha / np.pi),
        "density_exponent": alpha,
    }


def harmonic_wavefunction_density(x, omega=1.0, mass=1.0):
    """調和振動子の基底状態確率密度 |psi_0(x)|^2 を返す。"""
    result = harmonic_analytic_results(omega=omega, mass=mass)
    return result["density_prefactor"] * np.exp(-result["density_exponent"] * x**2)


def unavailable_analytic_results(reason):
    """閉じた解析解がないことを表す結果辞書を返す。"""
    return {
        "available": False,
        "reason": reason,
        "E0": None,
        "gap": None,
        "x2": None,
    }


def anharmonic_analytic_results():
    """非調和振動子の解析解の有無を返す。"""
    return unavailable_analytic_results(
        "一般の lambda に対する閉じた形の厳密な解析スペクトルはない"
    )


def double_well_analytic_results(lam=1.0, v=1.0, mass=1.0):
    """二重井戸ポテンシャルで解析的に分かる古典的な量を返す。

    ポテンシャルは

        V(x) = lam * (x^2 - v^2)^2

    である。完全な量子スペクトルは閉じた形では得られないが、極小点、
    障壁高さ、井戸底での局所調和振動子近似の角振動数は解析的に分かる。
    """
    local_curvature = 8.0 * lam * v**2
    local_omega = np.sqrt(local_curvature / mass)
    return {
        "available": False,
        "reason": "二重井戸の量子スペクトルは一般に閉じた解析解を持たない",
        "E0": None,
        "gap": None,
        "x2": None,
        "classical_minima": [-v, v],
        "classical_minimum_energy": 0.0,
        "barrier_position": 0.0,
        "barrier_height": lam * v**4,
        "local_harmonic_omega": local_omega,
        "local_harmonic_E0_approx": 0.5 * local_omega,
    }
