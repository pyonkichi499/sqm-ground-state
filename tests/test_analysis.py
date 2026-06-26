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
# ジャックナイフ
# ===========================================================================

def test_jackknife_known_mean():
    """data=[1,2,3,4,5] に np.mean を適用すると、推定値は 3.0 になる。"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    estimate, error = jackknife(data, func=np.mean)
    assert estimate == pytest.approx(3.0)
    # 等間隔データの平均に対するジャックナイフ誤差は明確に定義され、
    # 小さい正の値になるはず。
    assert error > 0.0
    assert error < 2.0  # 妥当性確認のための上限


def test_jackknife_single_element():
    """1 要素のデータでは (value, 0.0) を返す。"""
    estimate, error = jackknife([42.0])
    assert estimate == pytest.approx(42.0)
    assert error == 0.0


def test_jackknife_custom_func_var():
    """np.var を指定したジャックナイフでは、その関数が適用される。"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    estimate, error = jackknife(data, func=np.var)
    assert estimate == pytest.approx(np.var(data))
    # 誤差は非負であるはず。
    assert error >= 0.0


def test_jackknife_constant_data():
    """定数データの平均に対する誤差は 0 になる。"""
    data = np.array([7.0, 7.0, 7.0, 7.0])
    estimate, error = jackknife(data, func=np.mean)
    assert estimate == pytest.approx(7.0)
    assert error == pytest.approx(0.0, abs=1e-15)


def test_jackknife_returns_tuple_of_floats():
    """estimate と error はどちらも通常の Python の float である。"""
    estimate, error = jackknife([1.0, 2.0, 3.0])
    assert isinstance(estimate, float)
    assert isinstance(error, float)


def test_jackknife_two_elements():
    """2 要素でも動作し、妥当な誤差を返す。"""
    estimate, error = jackknife([10.0, 20.0])
    assert estimate == pytest.approx(15.0)
    assert error >= 0.0


def test_jackknife_accepts_list():
    """numpy 配列だけでなく、通常の Python リストも受け付ける。"""
    estimate, error = jackknife([1, 2, 3, 4, 5])
    assert estimate == pytest.approx(3.0)


def test_jackknife_custom_func_sum():
    """np.sum を指定したジャックナイフでは、推定値として和を返す。"""
    data = [2.0, 4.0, 6.0]
    estimate, error = jackknife(data, func=np.sum)
    assert estimate == pytest.approx(12.0)


# ==========================================================================
# ビニング解析
# ===========================================================================

def test_binning_uncorrelated_errors_roughly_constant():
    """無相関データでは、ビンサイズを変えても平均値の標準誤差はおおよそ一定になる。"""
    rng = np.random.default_rng(12345)
    data = rng.standard_normal(4096)
    bin_sizes, errors = binning_analysis(data)
    # 独立同分布のデータでは、すべての誤差が互いに 3 倍以内に収まるはず。
    assert errors.max() / errors.min() < 3.0


def test_binning_bin_sizes_are_powers_of_two():
    """ビンサイズは 1, 2, 4, 8, ... になる。"""
    data = np.arange(64, dtype=float)
    bin_sizes, errors = binning_analysis(data)
    for i, bs in enumerate(bin_sizes):
        assert bs == 2 ** i


def test_binning_single_element():
    """1 要素データでは ([1], [0.0]) を返す。"""
    bin_sizes, errors = binning_analysis([99.0])
    np.testing.assert_array_equal(bin_sizes, [1])
    np.testing.assert_array_equal(errors, [0.0])


def test_binning_shape_consistency():
    """bin_sizes と errors は同じ長さを持つ。"""
    data = np.arange(256, dtype=float)
    bin_sizes, errors = binning_analysis(data)
    assert len(bin_sizes) == len(errors)
    assert len(bin_sizes) > 0


def test_binning_returns_ndarrays():
    """返される 2 つの配列はどちらも numpy.ndarray である。"""
    data = np.arange(32, dtype=float)
    bin_sizes, errors = binning_analysis(data)
    assert isinstance(bin_sizes, np.ndarray)
    assert isinstance(errors, np.ndarray)


def test_binning_errors_non_negative():
    """すべての誤差は非負である。"""
    rng = np.random.default_rng(7)
    data = rng.standard_normal(512)
    _, errors = binning_analysis(data)
    assert np.all(errors >= 0.0)


def test_binning_custom_max_bin_size():
    """max_bin_size を渡すと、試す最大ビンサイズが制限される。"""
    data = np.arange(1024, dtype=float)
    bin_sizes, errors = binning_analysis(data, max_bin_size=8)
    assert bin_sizes[-1] <= 8


def test_binning_empty_like():
    """n < 2 の場合は自明な分岐が使われる。"""
    bin_sizes, errors = binning_analysis(np.array([]))
    np.testing.assert_array_equal(bin_sizes, [1])
    np.testing.assert_array_equal(errors, [0.0])


# ==========================================================================
# 自己相関関数
# ===========================================================================

def test_autocorrelation_white_noise_lag_zero():
    """任意のデータで A(0) は厳密に 1.0 になる。"""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(10000)
    acf = autocorrelation(data)
    assert acf[0] == pytest.approx(1.0, abs=1e-14)


def test_autocorrelation_white_noise_higher_lags_near_zero():
    """白色雑音では、A(k>0) はおおよそ 0 になる。"""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(10000)
    acf = autocorrelation(data)
    # ラグ 0 を除くすべての値は 0 に近いはず。
    assert np.all(np.abs(acf[1:]) < 0.05)


def test_autocorrelation_constant_data():
    """定数データ（分散 0）では、すべて 1 の配列を返す。"""
    data = np.full(20, 5.0)
    acf = autocorrelation(data)
    np.testing.assert_array_equal(acf, np.ones_like(acf))


def test_autocorrelation_single_element():
    """1 要素データでは [1.0] を返す。"""
    acf = autocorrelation([3.14])
    np.testing.assert_array_equal(acf, [1.0])


def test_autocorrelation_lag_zero_always_one():
    """任意の非定数データで A(0) == 1 になる。"""
    data = np.array([1.0, -2.0, 3.5, 0.0, -1.1])
    acf = autocorrelation(data)
    assert acf[0] == pytest.approx(1.0, abs=1e-14)


def test_autocorrelation_returns_ndarray():
    """戻り値の型は numpy.ndarray である。"""
    acf = autocorrelation([1.0, 2.0, 3.0])
    assert isinstance(acf, np.ndarray)


def test_autocorrelation_max_lag_respected():
    """明示的に指定した max_lag が出力長を決める。"""
    data = np.arange(100, dtype=float)
    acf = autocorrelation(data, max_lag=10)
    assert len(acf) == 11  # ラグ 0..10


def test_autocorrelation_two_elements():
    """2 要素データでも妥当な自己相関を返す。"""
    acf = autocorrelation([0.0, 1.0])
    assert acf[0] == pytest.approx(1.0)
    assert len(acf) == 2  # ラグ 0 と 1


def test_autocorrelation_correlated_signal():
    """ゆっくり変化する信号は、小さいラグで正の自己相関を持つ。"""
    # サイン波は小さいラグで強い正の相関を持つ。
    t = np.linspace(0, 4 * np.pi, 1000)
    data = np.sin(t)
    acf = autocorrelation(data, max_lag=50)
    # 小さいラグでは、自己相関は正かつ大きいはず。
    assert acf[1] > 0.9


# ==========================================================================
# 積分自己相関時間
# ===========================================================================

def test_iat_white_noise():
    """無相関データでは、tau_int はおおよそ 0.5 になる。"""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(10000)
    tau = integrated_autocorrelation_time(data)
    assert tau == pytest.approx(0.5, abs=0.1)


def test_iat_correlated_data():
    """ランダムウォーク（累積和）では、tau_int は 0.5 よりかなり大きくなる。"""
    rng = np.random.default_rng(42)
    increments = rng.standard_normal(5000)
    random_walk = np.cumsum(increments)
    tau = integrated_autocorrelation_time(random_walk)
    assert tau > 5.0  # 0.5 より十分大きい


def test_iat_returns_float():
    """戻り値は通常の Python の float である。"""
    tau = integrated_autocorrelation_time([1.0, 2.0, 3.0])
    assert isinstance(tau, float)


def test_iat_at_least_half():
    """式が 0.5 から始まるため、tau_int は常に 0.5 以上である。"""
    rng = np.random.default_rng(99)
    data = rng.standard_normal(500)
    tau = integrated_autocorrelation_time(data)
    assert tau >= 0.5


def test_iat_single_element():
    """1 要素データでは自己相関は [1.0] で、k>=1 の項がないため tau=0.5 になる。"""
    tau = integrated_autocorrelation_time([7.0])
    assert tau == pytest.approx(0.5)


def test_iat_constant_data():
    """定数データの自己相関はすべて 1 なので、tau_int は大きくなる。"""
    data = np.full(100, 3.0)
    tau = integrated_autocorrelation_time(data)
    # 自己相関関数がすべて 1 なので、tau = 0.5 + max_lag となり大きい。
    assert tau > 10.0
