import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.analysis import (
    jackknife,
    binning_analysis,
    autocorrelation,
    integrated_autocorrelation_time,
)


# ==========================================================================
# jackknife
# ==========================================================================

def test_jackknife_known_mean():
    """data=[1,2,3,4,5] with np.mean should give estimate=3.0."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    estimate, error = jackknife(data, func=np.mean)
    assert estimate == pytest.approx(3.0)
    # The jackknife error for the mean of equally-spaced data is well-defined
    # and should be a small positive number.
    assert error > 0.0
    assert error < 2.0  # sanity upper bound


def test_jackknife_single_element():
    """Single-element data should return (value, 0.0)."""
    estimate, error = jackknife([42.0])
    assert estimate == pytest.approx(42.0)
    assert error == 0.0


def test_jackknife_custom_func_var():
    """Jackknife with np.var should apply the custom function."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    estimate, error = jackknife(data, func=np.var)
    assert estimate == pytest.approx(np.var(data))
    # The error should be non-negative
    assert error >= 0.0


def test_jackknife_constant_data():
    """Constant data should give zero error for the mean."""
    data = np.array([7.0, 7.0, 7.0, 7.0])
    estimate, error = jackknife(data, func=np.mean)
    assert estimate == pytest.approx(7.0)
    assert error == pytest.approx(0.0, abs=1e-15)


def test_jackknife_returns_tuple_of_floats():
    """Both estimate and error must be plain Python floats."""
    estimate, error = jackknife([1.0, 2.0, 3.0])
    assert isinstance(estimate, float)
    assert isinstance(error, float)


def test_jackknife_two_elements():
    """Two elements should still work and produce a valid error."""
    estimate, error = jackknife([10.0, 20.0])
    assert estimate == pytest.approx(15.0)
    assert error >= 0.0


def test_jackknife_accepts_list():
    """Should accept a plain Python list, not just numpy arrays."""
    estimate, error = jackknife([1, 2, 3, 4, 5])
    assert estimate == pytest.approx(3.0)


def test_jackknife_custom_func_sum():
    """Jackknife with np.sum should return the sum as the estimate."""
    data = [2.0, 4.0, 6.0]
    estimate, error = jackknife(data, func=np.sum)
    assert estimate == pytest.approx(12.0)


# ==========================================================================
# binning_analysis
# ==========================================================================

def test_binning_uncorrelated_errors_roughly_constant():
    """For uncorrelated data the SEM should be roughly flat across bin sizes."""
    rng = np.random.default_rng(12345)
    data = rng.standard_normal(4096)
    bin_sizes, errors = binning_analysis(data)
    # All errors should be within a factor of 3 of each other for IID data
    assert errors.max() / errors.min() < 3.0


def test_binning_bin_sizes_are_powers_of_two():
    """Bin sizes should be 1, 2, 4, 8, ..."""
    data = np.arange(64, dtype=float)
    bin_sizes, errors = binning_analysis(data)
    for i, bs in enumerate(bin_sizes):
        assert bs == 2 ** i


def test_binning_single_element():
    """Single-element data returns ([1], [0.0])."""
    bin_sizes, errors = binning_analysis([99.0])
    np.testing.assert_array_equal(bin_sizes, [1])
    np.testing.assert_array_equal(errors, [0.0])


def test_binning_shape_consistency():
    """bin_sizes and errors must have the same length."""
    data = np.arange(256, dtype=float)
    bin_sizes, errors = binning_analysis(data)
    assert len(bin_sizes) == len(errors)
    assert len(bin_sizes) > 0


def test_binning_returns_ndarrays():
    """Both returned arrays should be numpy ndarrays."""
    data = np.arange(32, dtype=float)
    bin_sizes, errors = binning_analysis(data)
    assert isinstance(bin_sizes, np.ndarray)
    assert isinstance(errors, np.ndarray)


def test_binning_errors_non_negative():
    """All errors should be non-negative."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal(512)
    _, errors = binning_analysis(data)
    assert np.all(errors >= 0.0)


def test_binning_custom_max_bin_size():
    """Passing max_bin_size should limit the largest bin size tested."""
    data = np.arange(1024, dtype=float)
    bin_sizes, errors = binning_analysis(data, max_bin_size=8)
    assert bin_sizes[-1] <= 8


def test_binning_empty_like():
    """For n < 2 the trivial branch should be taken."""
    bin_sizes, errors = binning_analysis(np.array([]))
    np.testing.assert_array_equal(bin_sizes, [1])
    np.testing.assert_array_equal(errors, [0.0])


# ==========================================================================
# autocorrelation
# ==========================================================================

def test_autocorrelation_white_noise_lag_zero():
    """A(0) must be exactly 1.0 for any data."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(10000)
    acf = autocorrelation(data)
    assert acf[0] == pytest.approx(1.0, abs=1e-14)


def test_autocorrelation_white_noise_higher_lags_near_zero():
    """For white noise, A(k>0) should be approximately 0."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(10000)
    acf = autocorrelation(data)
    # Skip lag-0; all others should be close to zero
    assert np.all(np.abs(acf[1:]) < 0.05)


def test_autocorrelation_constant_data():
    """Constant data (zero variance) should return all ones."""
    data = np.full(20, 5.0)
    acf = autocorrelation(data)
    np.testing.assert_array_equal(acf, np.ones_like(acf))


def test_autocorrelation_single_element():
    """Single-element data should return [1.0]."""
    acf = autocorrelation([3.14])
    np.testing.assert_array_equal(acf, [1.0])


def test_autocorrelation_lag_zero_always_one():
    """A(0) == 1 for arbitrary non-constant data."""
    data = np.array([1.0, -2.0, 3.5, 0.0, -1.1])
    acf = autocorrelation(data)
    assert acf[0] == pytest.approx(1.0, abs=1e-14)


def test_autocorrelation_returns_ndarray():
    """Return type should be a numpy ndarray."""
    acf = autocorrelation([1.0, 2.0, 3.0])
    assert isinstance(acf, np.ndarray)


def test_autocorrelation_max_lag_respected():
    """Explicit max_lag should control the output length."""
    data = np.arange(100, dtype=float)
    acf = autocorrelation(data, max_lag=10)
    assert len(acf) == 11  # lags 0..10


def test_autocorrelation_two_elements():
    """Two-element data should return a valid autocorrelation."""
    acf = autocorrelation([0.0, 1.0])
    assert acf[0] == pytest.approx(1.0)
    assert len(acf) == 2  # lags 0 and 1


def test_autocorrelation_correlated_signal():
    """A slowly varying signal should have positive autocorrelation at small lags."""
    # Sine wave has strong positive correlation at small lags
    t = np.linspace(0, 4 * np.pi, 1000)
    data = np.sin(t)
    acf = autocorrelation(data, max_lag=50)
    # At small lags the autocorrelation should be positive and large
    assert acf[1] > 0.9


# ==========================================================================
# integrated_autocorrelation_time
# ==========================================================================

def test_iat_white_noise():
    """For uncorrelated data, tau_int should be approximately 0.5."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(10000)
    tau = integrated_autocorrelation_time(data)
    assert tau == pytest.approx(0.5, abs=0.1)


def test_iat_correlated_data():
    """For a random walk (cumulative sum), tau_int should be >> 0.5."""
    rng = np.random.default_rng(42)
    increments = rng.standard_normal(5000)
    random_walk = np.cumsum(increments)
    tau = integrated_autocorrelation_time(random_walk)
    assert tau > 5.0  # much larger than 0.5


def test_iat_returns_float():
    """Return value must be a plain Python float."""
    tau = integrated_autocorrelation_time([1.0, 2.0, 3.0])
    assert isinstance(tau, float)


def test_iat_at_least_half():
    """tau_int is always >= 0.5 because the formula starts at 0.5."""
    rng = np.random.default_rng(99)
    data = rng.standard_normal(500)
    tau = integrated_autocorrelation_time(data)
    assert tau >= 0.5


def test_iat_single_element():
    """Single-element data: autocorrelation is [1.0], no k>=1 terms, so tau=0.5."""
    tau = integrated_autocorrelation_time([7.0])
    assert tau == pytest.approx(0.5)


def test_iat_constant_data():
    """Constant data has all-ones autocorrelation, so tau_int should be large."""
    data = np.full(100, 3.0)
    tau = integrated_autocorrelation_time(data)
    # With all-ones ACF, tau = 0.5 + (max_lag) which is big
    assert tau > 10.0
