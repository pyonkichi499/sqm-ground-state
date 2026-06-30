"""
ポテンシャルと作用カーネルへの後方互換レイヤー。

設計の単一の真実の源は次の 2 つに移った。

- ``sqm_ground_state.potentials`` : ポテンシャル V(x)・導関数 V'(x)・解析的に分かる量
- ``sqm_ground_state.lattice``    : ユークリッド作用とドリフト力の数値カーネル

このモジュールは、従来からの関数ベースの API（``harmonic_potential`` など）を
維持し、既存の呼び出し側やテストを壊さないために残している。新しいコードでは
``sqm_ground_state.potentials.Potential`` オブジェクトを直接使うことを推奨する。
"""

from sqm_ground_state.lattice import drift_force, euclidean_action  # noqa: F401  （後方互換の再エクスポート）
from sqm_ground_state.potentials import (
    AnharmonicPotential,
    DoubleWellPotential,
    HarmonicPotential,
)


# ---------------------------------------------------------------------------
# ポテンシャル V(x)
# ---------------------------------------------------------------------------

def harmonic_potential(x, omega=1.0):
    """調和振動子ポテンシャル V(x) = 0.5 * omega^2 * x^2 を返す。"""
    return HarmonicPotential(omega=omega).potential(x)


def anharmonic_potential(x, omega=1.0, lam=1.0):
    """非調和振動子ポテンシャル V(x) = 0.5 * omega^2 * x^2 + lam * x^4 を返す。"""
    return AnharmonicPotential(omega=omega, lam=lam).potential(x)


def double_well_potential(x, lam=1.0, v=1.0):
    """二重井戸ポテンシャル V(x) = lam * (x^2 - v^2)^2 を返す。"""
    return DoubleWellPotential(lam=lam, v=v).potential(x)


# ---------------------------------------------------------------------------
# ポテンシャルの導関数 V'(x)
# ---------------------------------------------------------------------------

def harmonic_force(x, omega=1.0):
    """調和振動子ポテンシャルの導関数 dV/dx を返す。"""
    return HarmonicPotential(omega=omega).force(x)


def anharmonic_force(x, omega=1.0, lam=1.0):
    """非調和振動子ポテンシャルの導関数 dV/dx を返す。"""
    return AnharmonicPotential(omega=omega, lam=lam).force(x)


def double_well_force(x, lam=1.0, v=1.0):
    """二重井戸ポテンシャルの導関数 dV/dx を返す。"""
    return DoubleWellPotential(lam=lam, v=v).force(x)
