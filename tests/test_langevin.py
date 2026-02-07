"""Tests for the LangevinSimulation class in src/langevin.py."""

import sys
sys.path.insert(0, '/home/user/sqm-ground-state')

import numpy as np
import pytest

from src.langevin import LangevinSimulation
from src.action import harmonic_force


# -- Shared helpers ----------------------------------------------------------

def make_sim(**overrides):
    """Create a LangevinSimulation with sensible defaults for testing."""
    defaults = dict(
        n_lattice=20,
        a=0.1,
        mass=1.0,
        force_func=harmonic_force,
        epsilon=0.01,
        rng_seed=123,
        omega=1.0,
    )
    defaults.update(overrides)
    return LangevinSimulation(**defaults)


# -- 1. Initialization -------------------------------------------------------

class TestInitialization:
    def test_x_is_zeros(self):
        sim = make_sim(n_lattice=32)
        np.testing.assert_array_equal(sim.x, np.zeros(32))

    def test_x_shape(self):
        sim = make_sim(n_lattice=64)
        assert sim.x.shape == (64,)

    def test_attributes_stored(self):
        sim = make_sim(n_lattice=10, a=0.2, mass=2.0, epsilon=0.05)
        assert sim.n_lattice == 10
        assert sim.a == 0.2
        assert sim.mass == 2.0
        assert sim.epsilon == 0.05


# -- 2. Reproducibility ------------------------------------------------------

class TestReproducibility:
    def test_same_seed_gives_same_path(self):
        sim1 = make_sim(rng_seed=99)
        sim1.thermalize(100)
        x1 = sim1.x.copy()

        sim2 = make_sim(rng_seed=99)
        sim2.thermalize(100)
        x2 = sim2.x.copy()

        np.testing.assert_array_equal(x1, x2)

    def test_different_seed_gives_different_path(self):
        sim1 = make_sim(rng_seed=10)
        sim1.thermalize(100)

        sim2 = make_sim(rng_seed=20)
        sim2.thermalize(100)

        assert not np.allclose(sim1.x, sim2.x)


# -- 3. Step modifies the path -----------------------------------------------

class TestStep:
    def test_step_changes_path(self):
        sim = make_sim()
        assert np.all(sim.x == 0.0)
        sim.step()
        assert not np.all(sim.x == 0.0), "Path should change after a Langevin step"

    def test_step_updates_all_sites(self):
        sim = make_sim(n_lattice=16)
        sim.step()
        # Every site receives independent noise, so all should be nonzero.
        assert np.count_nonzero(sim.x) == 16


# -- 4. generate_configurations shape ----------------------------------------

class TestGenerateConfigurations:
    def test_shape(self):
        n_configs, n_lattice, n_skip = 15, 24, 5
        sim = make_sim(n_lattice=n_lattice)
        configs = sim.generate_configurations(n_configs=n_configs, n_skip=n_skip)
        assert configs.shape == (n_configs, n_lattice)

    def test_configs_are_distinct(self):
        sim = make_sim(n_lattice=10)
        sim.thermalize(200)
        configs = sim.generate_configurations(n_configs=5, n_skip=10)
        # Successive saved configurations should not be identical.
        for i in range(len(configs) - 1):
            assert not np.array_equal(configs[i], configs[i + 1])


# -- 5. run returns correct shape --------------------------------------------

class TestRun:
    def test_shape(self):
        n_configs, n_lattice = 10, 30
        sim = make_sim(n_lattice=n_lattice)
        configs = sim.run(n_therm=50, n_configs=n_configs, n_skip=5)
        assert configs.shape == (n_configs, n_lattice)

    def test_run_equivalent_to_thermalize_then_generate(self):
        """run() should produce the same result as calling thermalize + generate
        with the same seed."""
        seed = 77
        sim1 = make_sim(rng_seed=seed, n_lattice=16)
        configs1 = sim1.run(n_therm=100, n_configs=8, n_skip=5)

        sim2 = make_sim(rng_seed=seed, n_lattice=16)
        sim2.thermalize(100)
        configs2 = sim2.generate_configurations(n_configs=8, n_skip=5)

        np.testing.assert_array_equal(configs1, configs2)


# -- 6. Physics: harmonic oscillator <x^2> -----------------------------------

class TestPhysics:
    def test_harmonic_x_squared_expectation(self):
        """For a harmonic oscillator with omega=1, mass=1 the exact ground-
        state expectation value is <x^2> = 1/(2*omega) = 0.5.

        We verify the lattice simulation reproduces this within 20% tolerance.
        """
        sim = LangevinSimulation(
            n_lattice=100,
            a=0.1,
            mass=1.0,
            force_func=harmonic_force,
            epsilon=0.01,
            rng_seed=42,
            omega=1.0,
        )
        configs = sim.run(n_therm=5000, n_configs=2000, n_skip=20)

        # <x^2> averaged over all sites and all configurations
        x_sq_mean = np.mean(configs ** 2)
        expected = 0.5  # 1 / (2 * omega)

        assert x_sq_mean == pytest.approx(expected, rel=0.20), (
            f"<x^2> = {x_sq_mean:.4f}, expected ~{expected} within 20%"
        )
