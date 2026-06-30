import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from sqm_ground_state.action import harmonic_potential
from sqm_ground_state.exact import (
    finite_difference_hamiltonian,
    ground_state_energy,
    solve_spectrum,
)


def test_finite_difference_hamiltonian_shape():
    """ハミルトニアン行列は位置格子点数に対応する正方行列になる。"""
    x_grid = np.linspace(-2.0, 2.0, 21)
    hamiltonian = finite_difference_hamiltonian(x_grid, harmonic_potential)

    assert hamiltonian.shape == (21, 21)


def test_finite_difference_hamiltonian_is_symmetric():
    """有限差分ハミルトニアンは実対称行列になる。"""
    x_grid = np.linspace(-2.0, 2.0, 21)
    hamiltonian = finite_difference_hamiltonian(x_grid, harmonic_potential)

    np.testing.assert_allclose(hamiltonian, hamiltonian.T)


def test_harmonic_ground_state_energy():
    """調和振動子の基底状態エネルギーは omega/2 に近い。"""
    omega = 1.0
    energy = ground_state_energy(
        harmonic_potential,
        x_min=-8.0,
        x_max=8.0,
        n_grid=250,
        omega=omega,
    )

    assert energy == pytest.approx(omega / 2.0, rel=2e-3)


def test_harmonic_energy_gap():
    """調和振動子の E1 - E0 は omega に近い。"""
    omega = 1.0
    energies, _, _ = solve_spectrum(
        harmonic_potential,
        x_min=-8.0,
        x_max=8.0,
        n_grid=250,
        n_levels=2,
        omega=omega,
    )

    assert energies[1] - energies[0] == pytest.approx(omega, rel=2e-3)


def test_wavefunctions_are_normalized():
    """返される波動関数は積分規格化されている。"""
    energies, x_grid, wavefunctions = solve_spectrum(
        harmonic_potential,
        x_min=-6.0,
        x_max=6.0,
        n_grid=150,
        n_levels=3,
    )
    dx = x_grid[1] - x_grid[0]

    for i in range(len(energies)):
        norm = np.sum(np.abs(wavefunctions[:, i]) ** 2) * dx
        assert norm == pytest.approx(1.0)


def test_invalid_grid_raises():
    """不正な位置格子は ValueError になる。"""
    with pytest.raises(ValueError):
        finite_difference_hamiltonian(np.array([0.0, 0.5, 2.0]), harmonic_potential)
