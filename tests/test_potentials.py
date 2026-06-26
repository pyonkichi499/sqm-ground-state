import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.potentials import (
    AnharmonicPotential,
    DoubleWellPotential,
    HarmonicPotential,
    anharmonic,
    double_well,
    harmonic,
)


def test_constructors_return_expected_types():
    assert isinstance(harmonic(), HarmonicPotential)
    assert isinstance(anharmonic(), AnharmonicPotential)
    assert isinstance(double_well(), DoubleWellPotential)


def test_potential_names():
    assert harmonic().name == "harmonic"
    assert anharmonic().name == "anharmonic"
    assert double_well().name == "double_well"


def test_display_params_order_and_values():
    assert list(harmonic(omega=2.0).display_params.items()) == [("omega", 2.0)]
    assert list(anharmonic(omega=1.0, lam=0.5).display_params.items()) == [
        ("omega", 1.0), ("lambda", 0.5)
    ]
    assert list(double_well(lam=2.0, v=3.0).display_params.items()) == [
        ("lambda", 2.0), ("v", 3.0)
    ]


@pytest.mark.parametrize("x_val", [-2.0, -0.5, 0.0, 0.7, 3.0])
def test_force_is_derivative_of_potential(x_val):
    """各ポテンシャルで V' が V の中心差分微分に一致する。"""
    eps = 1e-6
    for pot in [harmonic(omega=1.3), anharmonic(omega=1.0, lam=0.4),
                double_well(lam=1.0, v=1.2)]:
        deriv = (pot.potential(x_val + eps) - pot.potential(x_val - eps)) / (2 * eps)
        assert pot.force(x_val) == pytest.approx(deriv, abs=1e-5)


def test_potential_vectorized():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(harmonic().potential(x), 0.5 * x**2)


def test_harmonic_analytic_available():
    result = harmonic(omega=2.0).analytic(mass=1.0)
    assert result["available"] is True
    assert result["E0"] == pytest.approx(1.0)
    assert result["gap"] == pytest.approx(2.0)
    assert result["x2"] == pytest.approx(0.25)
    # density は callable で、解析的な確率密度を返す
    assert result["density"](0.0) == pytest.approx(result["density_prefactor"])


def test_anharmonic_analytic_unavailable():
    result = anharmonic().analytic()
    assert result["available"] is False
    assert result["E0"] is None


def test_double_well_classical_quantities():
    result = double_well(lam=2.0, v=3.0).analytic(mass=1.0)
    assert result["available"] is False
    assert result["classical_minima"] == [-3.0, 3.0]
    assert result["barrier_height"] == pytest.approx(162.0)
    assert result["local_harmonic_omega"] == pytest.approx(np.sqrt(8.0 * 2.0 * 9.0))
