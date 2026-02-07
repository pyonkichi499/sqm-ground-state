"""
Statistical analysis tools for Monte Carlo / Langevin simulation data.

Provides jackknife resampling, binning analysis, autocorrelation
functions, and integrated autocorrelation time estimation -- the
standard toolkit for extracting reliable error bars from correlated
time-series produced by stochastic quantization simulations.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Jackknife resampling
# ---------------------------------------------------------------------------

def jackknife(data, func=np.mean):
    """Jackknife resampling for error estimation.

    Parameters
    ----------
    data : 1-D np.ndarray
        Array of measurements.
    func : callable
        Function that maps an array to a scalar (default: np.mean).

    Returns
    -------
    (estimate, error) : tuple of floats
        estimate = func(data)
        error    = sqrt((N-1)/N * sum((func(leave-one-out_i) - mean_resampled)^2))
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 2:
        return float(func(data)), 0.0

    estimate = float(func(data))

    # Leave-one-out resampled estimates
    resampled = np.empty(n)
    for i in range(n):
        resampled[i] = func(np.delete(data, i))

    mean_resampled = np.mean(resampled)
    error = np.sqrt((n - 1) / n * np.sum((resampled - mean_resampled) ** 2))

    return estimate, float(error)


# ---------------------------------------------------------------------------
# Binning analysis
# ---------------------------------------------------------------------------

def binning_analysis(data, max_bin_size=None):
    """Binning analysis to detect autocorrelation.

    Computes the standard error of the mean for geometrically increasing
    bin sizes (1, 2, 4, 8, ...).  When the error plateaus the bin size
    is large enough to decorrelate successive measurements.

    Parameters
    ----------
    data : 1-D np.ndarray
        Array of measurements.
    max_bin_size : int, optional
        Largest bin size to test.  Defaults to len(data) // 4 (need at
        least 4 bins for a meaningful error estimate).

    Returns
    -------
    (bin_sizes, errors) : tuple of np.ndarrays
        bin_sizes : array of bin sizes tested (1, 2, 4, 8, ...)
        errors    : corresponding standard error of the mean
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 2:
        return np.array([1]), np.array([0.0])

    if max_bin_size is None:
        max_bin_size = max(1, n // 4)

    bin_sizes = []
    errors = []

    bs = 1
    while bs <= max_bin_size:
        # Number of complete bins
        n_bins = n // bs
        if n_bins < 2:
            break

        # Bin the data by reshaping (drop trailing incomplete bin)
        binned = data[: n_bins * bs].reshape(n_bins, bs).mean(axis=1)

        sem = np.std(binned, ddof=1) / np.sqrt(n_bins)
        bin_sizes.append(bs)
        errors.append(sem)

        bs *= 2

    return np.array(bin_sizes), np.array(errors)


# ---------------------------------------------------------------------------
# Autocorrelation function
# ---------------------------------------------------------------------------

def autocorrelation(data, max_lag=None):
    """Compute the normalized autocorrelation function.

    A(k) = <(x_i - <x>)(x_{i+k} - <x>)> / <(x_i - <x>)^2>

    so that A(0) = 1.

    Parameters
    ----------
    data : 1-D np.ndarray
        Time series of measurements.
    max_lag : int, optional
        Maximum lag to compute.  Defaults to len(data) // 4.

    Returns
    -------
    np.ndarray
        Autocorrelation values for lags 0, 1, ..., max_lag.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 2:
        return np.array([1.0])

    if max_lag is None:
        max_lag = max(1, n // 4)
    max_lag = min(max_lag, n - 1)

    mean = np.mean(data)
    fluctuations = data - mean
    variance = np.dot(fluctuations, fluctuations) / n

    # Guard against zero-variance data
    if variance == 0.0:
        return np.ones(max_lag + 1)

    acf = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        acf[k] = np.dot(fluctuations[: n - k], fluctuations[k:]) / (n - k)

    acf /= variance
    return acf


# ---------------------------------------------------------------------------
# Integrated autocorrelation time
# ---------------------------------------------------------------------------

def integrated_autocorrelation_time(data, max_lag=None):
    """Estimate the integrated autocorrelation time.

    tau_int = 0.5 + sum_{k=1}^{max_lag} A(k)

    The summation is truncated at the first lag where A(k) drops below
    zero, since noise in the tail biases the estimate upward.

    Parameters
    ----------
    data : 1-D np.ndarray
        Time series of measurements.
    max_lag : int, optional
        Passed through to :func:`autocorrelation`.

    Returns
    -------
    float
        The integrated autocorrelation time.
    """
    acf = autocorrelation(data, max_lag=max_lag)

    tau = 0.5
    for k in range(1, len(acf)):
        if acf[k] < 0.0:
            break
        tau += acf[k]

    return float(tau)
