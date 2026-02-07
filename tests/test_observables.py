import sys
sys.path.insert(0, '/home/user/sqm-ground-state')

import numpy as np
import pytest
from src.observables import correlator, effective_mass, position_histogram, mean_x_squared


# ---------- correlator ----------

def test_correlator_all_ones():
    """Configs of all ones should give C[k] = 1 for every k."""
    configs = np.ones((5, 16))
    corr = correlator(configs)
    np.testing.assert_allclose(corr, 1.0)


def test_correlator_default_shape():
    """Shape should be (n_lattice//2 + 1,) when max_tau is not given."""
    configs = np.random.randn(10, 20)
    corr = correlator(configs)
    assert corr.shape == (11,)


def test_correlator_custom_max_tau():
    """Explicit max_tau should be respected."""
    configs = np.random.randn(10, 20)
    corr = correlator(configs, max_tau=5)
    assert corr.shape == (6,)


# ---------- effective_mass ----------

def test_effective_mass_pure_exponential():
    """For C[k] = exp(-m*a*k), effective mass should recover m."""
    m_true = 3.0
    a = 0.1
    taus = np.arange(20)
    corr = np.exp(-m_true * a * taus)
    m_eff = effective_mass(corr, a=a)
    np.testing.assert_allclose(m_eff, m_true, atol=1e-12)


def test_effective_mass_negative_values():
    """Negative correlator entries should produce nan."""
    corr = np.array([1.0, -0.5, 0.3])
    m_eff = effective_mass(corr, a=0.1)
    # corr[0]>0 but corr[1]<0 -> m_eff[0] is nan
    assert np.isnan(m_eff[0])
    # corr[1]<0 but corr[2]>0 -> m_eff[1] is nan
    assert np.isnan(m_eff[1])


def test_effective_mass_shape():
    """Output length should be len(corr) - 1."""
    corr = np.ones(10)
    m_eff = effective_mass(corr)
    assert m_eff.shape == (9,)


# ---------- position_histogram ----------

def test_position_histogram_shapes():
    """bin_centers and hist_values should both have length n_bins."""
    configs = np.random.randn(50, 30)
    centers, values = position_histogram(configs, n_bins=40)
    assert centers.shape == (40,)
    assert values.shape == (40,)


def test_position_histogram_integrates_to_one():
    """density=True histogram should integrate to approximately 1."""
    configs = np.random.randn(200, 100)
    centers, values = position_histogram(configs, n_bins=80)
    bin_width = centers[1] - centers[0]
    integral = np.sum(values * bin_width)
    np.testing.assert_allclose(integral, 1.0, atol=1e-10)


# ---------- mean_x_squared ----------

def test_mean_x_squared_constant():
    """Configs of all 2.0 should give <x^2> = 4.0."""
    configs = np.full((8, 12), 2.0)
    result = mean_x_squared(configs)
    assert result == pytest.approx(4.0)


def test_mean_x_squared_returns_float():
    """Return value must be a plain Python float."""
    configs = np.ones((3, 4))
    result = mean_x_squared(configs)
    assert isinstance(result, float)
