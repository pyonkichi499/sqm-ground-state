"""
解析的に分かる量への後方互換レイヤー。

設計の単一の真実の源は ``src.potentials`` の各 Potential クラスの
``analytic(mass)`` メソッドに移った。このモジュールは従来の関数ベース API を
維持するための薄いラッパーである。新しいコードでは Potential オブジェクトの
``analytic`` を直接使うことを推奨する。
"""

import numpy as np

from src.potentials import (
    AnharmonicPotential,
    DoubleWellPotential,
    HarmonicPotential,
)


def harmonic_analytic_results(omega=1.0, mass=1.0):
    """調和振動子の解析解を返す。"""
    return HarmonicPotential(omega=omega).analytic(mass=mass)


def harmonic_wavefunction_density(x, omega=1.0, mass=1.0):
    """調和振動子の基底状態確率密度 |psi_0(x)|^2 を返す。"""
    result = HarmonicPotential(omega=omega).analytic(mass=mass)
    return result["density_prefactor"] * np.exp(-result["density_exponent"] * np.asarray(x) ** 2)


def unavailable_analytic_results(reason):
    """閉じた解析解がないことを表す結果辞書を返す。"""
    return {
        "available": False,
        "reason": reason,
        "E0": None,
        "gap": None,
        "x2": None,
    }


def anharmonic_analytic_results(omega=1.0, lam=1.0):
    """非調和振動子の解析解の有無を返す。"""
    return AnharmonicPotential(omega=omega, lam=lam).analytic()


def double_well_analytic_results(lam=1.0, v=1.0, mass=1.0):
    """二重井戸ポテンシャルで解析的に分かる古典的な量を返す。"""
    return DoubleWellPotential(lam=lam, v=v).analytic(mass=mass)
