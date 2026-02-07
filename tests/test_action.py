import sys
sys.path.insert(0, "/home/user/sqm-ground-state")

import numpy as np
import pytest
from src.action import (
    harmonic_potential,
    anharmonic_potential,
    double_well_potential,
    harmonic_force,
    anharmonic_force,
    double_well_force,
    euclidean_action,
    drift_force,
)


# ── Potentials at scalar inputs ──────────────────────────────────────────

class TestHarmonicPotential:
    def test_zero(self):
        assert harmonic_potential(0.0) == 0.0

    def test_positive(self):
        assert harmonic_potential(2.0) == pytest.approx(2.0)

    def test_omega(self):
        assert harmonic_potential(1.0, omega=3.0) == pytest.approx(4.5)

    def test_array(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        expected = 0.5 * x**2
        np.testing.assert_allclose(harmonic_potential(x), expected)


class TestAnharmonicPotential:
    def test_zero(self):
        assert anharmonic_potential(0.0) == 0.0

    def test_positive(self):
        # 0.5*1*1 + 1*1 = 1.5
        assert anharmonic_potential(1.0) == pytest.approx(1.5)

    def test_params(self):
        # 0.5*4*4 + 2*16 = 8 + 32 = 40
        assert anharmonic_potential(2.0, omega=2.0, lam=2.0) == pytest.approx(40.0)

    def test_array(self):
        x = np.array([0.0, 1.0, -1.0])
        expected = 0.5 * x**2 + x**4
        np.testing.assert_allclose(anharmonic_potential(x), expected)


class TestDoubleWellPotential:
    def test_at_minimum(self):
        # Minima at x = +/- v, V = 0
        assert double_well_potential(1.0, lam=1.0, v=1.0) == pytest.approx(0.0)
        assert double_well_potential(-1.0, lam=1.0, v=1.0) == pytest.approx(0.0)

    def test_at_origin(self):
        # V(0) = lam * v^4
        assert double_well_potential(0.0, lam=2.0, v=3.0) == pytest.approx(2.0 * 81.0)

    def test_array(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = (x**2 - 1.0) ** 2
        np.testing.assert_allclose(double_well_potential(x), expected)


# ── Forces at scalar inputs ──────────────────────────────────────────────

class TestHarmonicForce:
    def test_zero(self):
        assert harmonic_force(0.0) == 0.0

    def test_value(self):
        assert harmonic_force(3.0, omega=2.0) == pytest.approx(12.0)

    def test_array(self):
        x = np.array([-1.0, 0.0, 2.0])
        np.testing.assert_allclose(harmonic_force(x, omega=2.0), 4.0 * x)


class TestAnharmonicForce:
    def test_zero(self):
        assert anharmonic_force(0.0) == 0.0

    def test_value(self):
        # omega^2 * x + 4*lam*x^3 = 1*2 + 4*1*8 = 34
        assert anharmonic_force(2.0) == pytest.approx(34.0)


class TestDoubleWellForce:
    def test_zero(self):
        # 4*lam*0*(0 - v^2) = 0
        assert double_well_force(0.0) == pytest.approx(0.0)

    def test_at_minimum(self):
        # 4*1*1*(1-1) = 0
        assert double_well_force(1.0) == pytest.approx(0.0)

    def test_value(self):
        # 4*1*2*(4-1) = 24
        assert double_well_force(2.0) == pytest.approx(24.0)


# ── Force is numerical derivative of potential ───────────────────────────

@pytest.mark.parametrize("x_val", [-2.0, -0.5, 0.0, 0.7, 3.0])
class TestForceIsDerivative:
    """Check dV/dx via centred finite differences for all three potentials."""

    eps = 1e-6

    def test_harmonic(self, x_val):
        deriv = (harmonic_potential(x_val + self.eps) - harmonic_potential(x_val - self.eps)) / (2 * self.eps)
        assert harmonic_force(x_val) == pytest.approx(deriv, abs=1e-5)

    def test_anharmonic(self, x_val):
        deriv = (anharmonic_potential(x_val + self.eps) - anharmonic_potential(x_val - self.eps)) / (2 * self.eps)
        assert anharmonic_force(x_val) == pytest.approx(deriv, abs=1e-5)

    def test_double_well(self, x_val):
        deriv = (double_well_potential(x_val + self.eps) - double_well_potential(x_val - self.eps)) / (2 * self.eps)
        assert double_well_force(x_val) == pytest.approx(deriv, abs=1e-5)


# ── Euclidean action ─────────────────────────────────────────────────────

class TestEuclideanAction:
    def test_constant_path_harmonic(self):
        """Constant path: kinetic = 0, S_E = N * a * V(c)."""
        N, a, c = 20, 0.1, 2.0
        x = np.full(N, c)
        expected = N * a * harmonic_potential(c)
        result = euclidean_action(x, harmonic_potential, mass=1.0, a=a)
        assert result == pytest.approx(expected)

    def test_constant_path_double_well(self):
        N, a, c = 10, 0.2, 0.5
        x = np.full(N, c)
        expected = N * a * double_well_potential(c, lam=2.0, v=1.0)
        result = euclidean_action(x, double_well_potential, mass=1.0, a=a, lam=2.0, v=1.0)
        assert result == pytest.approx(expected)

    def test_two_site_kinetic(self):
        """Two-site path: kinetic contribution from periodic differences."""
        x = np.array([0.0, 1.0])
        a, mass = 0.5, 2.0
        # dx = [1-0, 0-1] = [1, -1], kinetic = m/(2a)*sum(dx^2) = 2/1*2 = 4
        kinetic = mass / (2 * a) * (1.0 + 1.0)
        potential = a * (harmonic_potential(0.0) + harmonic_potential(1.0))
        expected = kinetic + potential
        result = euclidean_action(x, harmonic_potential, mass=mass, a=a)
        assert result == pytest.approx(expected)

    def test_zero_path(self):
        """All-zero path gives zero action for harmonic potential."""
        x = np.zeros(50)
        assert euclidean_action(x, harmonic_potential, a=0.1) == pytest.approx(0.0)


# ── Drift force ──────────────────────────────────────────────────────────

class TestDriftForce:
    def test_constant_path_harmonic(self):
        """Constant path: laplacian = 0, drift = -a * V'(c) for every site."""
        N, a, c = 16, 0.1, 2.0
        x = np.full(N, c)
        expected = -a * harmonic_force(c) * np.ones(N)
        result = drift_force(x, harmonic_force, mass=1.0, a=a)
        np.testing.assert_allclose(result, expected)

    def test_constant_path_anharmonic(self):
        N, a, c = 8, 0.2, 1.5
        x = np.full(N, c)
        expected = -a * anharmonic_force(c, omega=2.0, lam=0.5) * np.ones(N)
        result = drift_force(x, anharmonic_force, mass=1.0, a=a, omega=2.0, lam=0.5)
        np.testing.assert_allclose(result, expected)

    def test_zero_path(self):
        """Zero path gives zero drift for harmonic potential."""
        x = np.zeros(10)
        result = drift_force(x, harmonic_force, a=0.1)
        np.testing.assert_allclose(result, np.zeros(10))


# ── Periodic boundary conditions ─────────────────────────────────────────

class TestPeriodicBC:
    def test_drift_force_3site(self):
        """Verify periodic BC on a small 3-site path [1, 2, 4]."""
        x = np.array([1.0, 2.0, 4.0])
        mass, a = 1.0, 0.1

        # Site 0: neighbours are x[1]=2 and x[2]=4 (periodic)
        lap_0 = 2.0 + 4.0 - 2.0 * 1.0   # 4
        f_0 = (mass / a) * lap_0 - a * harmonic_force(1.0)

        # Site 1: neighbours are x[2]=4 and x[0]=1
        lap_1 = 4.0 + 1.0 - 2.0 * 2.0   # 1
        f_1 = (mass / a) * lap_1 - a * harmonic_force(2.0)

        # Site 2: neighbours are x[0]=1 and x[1]=2 (periodic)
        lap_2 = 1.0 + 2.0 - 2.0 * 4.0   # -5
        f_2 = (mass / a) * lap_2 - a * harmonic_force(4.0)

        expected = np.array([f_0, f_1, f_2])
        result = drift_force(x, harmonic_force, mass=mass, a=a)
        np.testing.assert_allclose(result, expected)

    def test_action_periodic_wrap(self):
        """Action kinetic term wraps x[N-1] -> x[0]."""
        x = np.array([0.0, 0.0, 3.0])
        a, mass = 1.0, 1.0
        # dx = [0-0, 3-0, 0-3] = [0, 3, -3], kinetic = 0.5*(0+9+9) = 9
        kinetic = mass / (2.0 * a) * (0.0 + 9.0 + 9.0)
        potential = a * np.sum(harmonic_potential(x))
        expected = kinetic + potential
        result = euclidean_action(x, harmonic_potential, mass=mass, a=a)
        assert result == pytest.approx(expected)
