import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from sqm_ground_state.analytic import (
    anharmonic_analytic_results,
    double_well_analytic_results,
    harmonic_analytic_results,
    harmonic_wavefunction_density,
)


def test_harmonic_analytic_results_mass_one():
    """mass=1 の調和振動子では E0=omega/2, gap=omega, <x^2>=1/(2omega)。"""
    result = harmonic_analytic_results(omega=2.0, mass=1.0)

    assert result["available"] is True
    assert result["E0"] == pytest.approx(1.0)
    assert result["gap"] == pytest.approx(2.0)
    assert result["x2"] == pytest.approx(0.25)


def test_harmonic_wavefunction_density_integrates_to_one():
    """調和振動子の解析的な |psi_0|^2 は 1 に規格化されている。"""
    x = np.linspace(-8.0, 8.0, 2000)
    psi2 = harmonic_wavefunction_density(x, omega=1.0, mass=1.0)
    integral = np.trapezoid(psi2, x)

    assert integral == pytest.approx(1.0, rel=1e-5)


def test_anharmonic_reports_unavailable_exact_solution():
    """非調和振動子は閉じた解析スペクトルなしとして扱う。"""
    result = anharmonic_analytic_results()

    assert result["available"] is False
    assert result["E0"] is None
    assert result["gap"] is None


def test_double_well_classical_quantities():
    """二重井戸の極小点・障壁高さ・局所振動数は解析的に得られる。"""
    result = double_well_analytic_results(lam=2.0, v=3.0, mass=1.0)

    assert result["available"] is False
    assert result["classical_minima"] == [-3.0, 3.0]
    assert result["barrier_height"] == pytest.approx(162.0)
    assert result["local_harmonic_omega"] == pytest.approx(np.sqrt(144.0))
