import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.observables import correlator, effective_mass, position_histogram, mean_x_squared


# ---------- 相関関数 ----------

def test_correlator_all_ones():
    """すべて 1 の配位では、すべての k について C[k] = 1 になる。"""
    configs = np.ones((5, 16))
    corr = correlator(configs)
    np.testing.assert_allclose(corr, 1.0)


def test_correlator_default_shape():
    """max_tau を指定しない場合、配列の形状は (n_lattice//2 + 1,) になる。"""
    configs = np.random.randn(10, 20)
    corr = correlator(configs)
    assert corr.shape == (11,)


def test_correlator_custom_max_tau():
    """明示的に指定した max_tau が尊重される。"""
    configs = np.random.randn(10, 20)
    corr = correlator(configs, max_tau=5)
    assert corr.shape == (6,)


# ---------- 有効質量 ----------

def test_effective_mass_pure_exponential():
    """C[k] = exp(-m*a*k) の場合、有効質量は m を再現する。"""
    m_true = 3.0
    a = 0.1
    taus = np.arange(20)
    corr = np.exp(-m_true * a * taus)
    m_eff = effective_mass(corr, a=a)
    np.testing.assert_allclose(m_eff, m_true, atol=1e-12)


def test_effective_mass_negative_values():
    """相関関数に負の要素がある場合、対応する有効質量は nan になる。"""
    corr = np.array([1.0, -0.5, 0.3])
    m_eff = effective_mass(corr, a=0.1)
    # corr[0]>0 だが corr[1]<0 なので m_eff[0] は nan
    assert np.isnan(m_eff[0])
    # corr[1]<0 だが corr[2]>0 なので m_eff[1] は nan
    assert np.isnan(m_eff[1])


def test_effective_mass_shape():
    """出力長は len(corr) - 1 になる。"""
    corr = np.ones(10)
    m_eff = effective_mass(corr)
    assert m_eff.shape == (9,)


# ---------- 位置ヒストグラム ----------

def test_position_histogram_shapes():
    """bin_centers と hist_values はどちらも長さ n_bins になる。"""
    configs = np.random.randn(50, 30)
    centers, values = position_histogram(configs, n_bins=40)
    assert centers.shape == (40,)
    assert values.shape == (40,)


def test_position_histogram_integrates_to_one():
    """密度規格化したヒストグラムは、おおよそ 1 に積分される。"""
    configs = np.random.randn(200, 100)
    centers, values = position_histogram(configs, n_bins=80)
    bin_width = centers[1] - centers[0]
    integral = np.sum(values * bin_width)
    np.testing.assert_allclose(integral, 1.0, atol=1e-10)


# ---------- <x^2> の平均 ----------

def test_mean_x_squared_constant():
    """すべて 2.0 の配位では <x^2> = 4.0 になる。"""
    configs = np.full((8, 12), 2.0)
    result = mean_x_squared(configs)
    assert result == pytest.approx(4.0)


def test_mean_x_squared_returns_float():
    """戻り値は通常の Python の float である。"""
    configs = np.ones((3, 4))
    result = mean_x_squared(configs)
    assert isinstance(result, float)
