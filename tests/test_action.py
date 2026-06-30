import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from sqm_ground_state.action import (
    harmonic_potential,
    anharmonic_potential,
    double_well_potential,
    harmonic_force,
    anharmonic_force,
    double_well_force,
    euclidean_action,
    drift_force,
)


# ── スカラー入力に対するポテンシャル ───────────────────────────────────────

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
        # 極小点は x = +/- v で、このとき V = 0
        assert double_well_potential(1.0, lam=1.0, v=1.0) == pytest.approx(0.0)
        assert double_well_potential(-1.0, lam=1.0, v=1.0) == pytest.approx(0.0)

    def test_at_origin(self):
        # V(0) = lam * v^4
        assert double_well_potential(0.0, lam=2.0, v=3.0) == pytest.approx(2.0 * 81.0)

    def test_array(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = (x**2 - 1.0) ** 2
        np.testing.assert_allclose(double_well_potential(x), expected)


# ── スカラー入力に対する力 ─────────────────────────────────────────────────

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


# ── 力がポテンシャルの数値微分になっていること ─────────────────────────────

@pytest.mark.parametrize("x_val", [-2.0, -0.5, 0.0, 0.7, 3.0])
class TestForceIsDerivative:
    """3 種類すべてのポテンシャルについて、中心差分で dV/dx を確認する。"""

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


# ── ユークリッド作用 ───────────────────────────────────────────────────────

class TestEuclideanAction:
    def test_constant_path_harmonic(self):
        """定数経路では 運動項 = 0, S_E = N * a * V(c) となる。"""
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
        """2 格子点の経路で、周期的な差分から運動項を計算する。"""
        x = np.array([0.0, 1.0])
        a, mass = 0.5, 2.0
        # dx = [1-0, 0-1] = [1, -1], 運動項 = m/(2a)*sum(dx^2) = 2/1*2 = 4
        kinetic = mass / (2 * a) * (1.0 + 1.0)
        potential = a * (harmonic_potential(0.0) + harmonic_potential(1.0))
        expected = kinetic + potential
        result = euclidean_action(x, harmonic_potential, mass=mass, a=a)
        assert result == pytest.approx(expected)

    def test_zero_path(self):
        """全ゼロ経路では、調和振動子ポテンシャルに対する作用は 0 になる。"""
        x = np.zeros(50)
        assert euclidean_action(x, harmonic_potential, a=0.1) == pytest.approx(0.0)


# ── ドリフト力 ─────────────────────────────────────────────────────────────

class TestDriftForce:
    def test_constant_path_harmonic(self):
        """定数経路では ラプラシアン = 0 なので、各格子点で ドリフト力 = -a * V'(c) となる。"""
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
        """ゼロ経路では、調和振動子ポテンシャルに対するドリフト力は 0 になる。"""
        x = np.zeros(10)
        result = drift_force(x, harmonic_force, a=0.1)
        np.testing.assert_allclose(result, np.zeros(10))


# ── 周期境界条件 ───────────────────────────────────────────────────────────

class TestPeriodicBC:
    def test_drift_force_3site(self):
        """小さな 3 格子点経路 [1, 2, 4] で周期境界条件を確認する。"""
        x = np.array([1.0, 2.0, 4.0])
        mass, a = 1.0, 0.1

        # 格子点 0: 隣接点は x[1]=2 と x[2]=4（周期境界）
        lap_0 = 2.0 + 4.0 - 2.0 * 1.0   # 4
        f_0 = (mass / a) * lap_0 - a * harmonic_force(1.0)

        # 格子点 1: 隣接点は x[2]=4 と x[0]=1
        lap_1 = 4.0 + 1.0 - 2.0 * 2.0   # 1
        f_1 = (mass / a) * lap_1 - a * harmonic_force(2.0)

        # 格子点 2: 隣接点は x[0]=1 と x[1]=2（周期境界）
        lap_2 = 1.0 + 2.0 - 2.0 * 4.0   # -5
        f_2 = (mass / a) * lap_2 - a * harmonic_force(4.0)

        expected = np.array([f_0, f_1, f_2])
        result = drift_force(x, harmonic_force, mass=mass, a=a)
        np.testing.assert_allclose(result, expected)

    def test_action_periodic_wrap(self):
        """作用の運動項では x[N-1] -> x[0] の折り返しを含める。"""
        x = np.array([0.0, 0.0, 3.0])
        a, mass = 1.0, 1.0
        # dx = [0-0, 3-0, 0-3] = [0, 3, -3], 運動項 = 0.5*(0+9+9) = 9
        kinetic = mass / (2.0 * a) * (0.0 + 9.0 + 9.0)
        potential = a * np.sum(harmonic_potential(x))
        expected = kinetic + potential
        result = euclidean_action(x, harmonic_potential, mass=mass, a=a)
        assert result == pytest.approx(expected)
