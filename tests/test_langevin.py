"""src/langevin.py の LangevinSimulation クラスに対するテスト。"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.langevin import LangevinSimulation
from src.action import harmonic_force


# -- 共通ヘルパー ------------------------------------------------------------

def make_sim(**overrides):
    """テスト用の妥当なデフォルト値で LangevinSimulation を作成する。"""
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


# -- 1. 初期化 ---------------------------------------------------------------

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


# -- 2. 再現性 ---------------------------------------------------------------

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


# -- 3. 1 ステップで経路が変更されること -------------------------------------------

class TestStep:
    def test_step_changes_path(self):
        sim = make_sim()
        assert np.all(sim.x == 0.0)
        sim.step()
        assert not np.all(sim.x == 0.0), "ランジュバンステップ後は経路が変化するはず"

    def test_step_updates_all_sites(self):
        sim = make_sim(n_lattice=16)
        sim.step()
        # 各格子点には独立なノイズが加わるため、すべて非ゼロになるはず。
        assert np.count_nonzero(sim.x) == 16


# -- 4. 生成される配位配列の形状 ------------------------------------

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
        # 連続して保存された配位は同一ではないはず。
        for i in range(len(configs) - 1):
            assert not np.array_equal(configs[i], configs[i + 1])


# -- 5. run が正しい 形状 を返すこと ---------------------------------------

class TestRun:
    def test_shape(self):
        n_configs, n_lattice = 10, 30
        sim = make_sim(n_lattice=n_lattice)
        configs = sim.run(n_therm=50, n_configs=n_configs, n_skip=5)
        assert configs.shape == (n_configs, n_lattice)

    def test_run_equivalent_to_thermalize_then_generate(self):
        """同じシードなら、run() は熱化後に配位生成した場合と同じ結果を返すはず。"""
        seed = 77
        sim1 = make_sim(rng_seed=seed, n_lattice=16)
        configs1 = sim1.run(n_therm=100, n_configs=8, n_skip=5)

        sim2 = make_sim(rng_seed=seed, n_lattice=16)
        sim2.thermalize(100)
        configs2 = sim2.generate_configurations(n_configs=8, n_skip=5)

        np.testing.assert_array_equal(configs1, configs2)


# -- 6. 物理: 調和振動子の <x^2> -------------------------------------------

class TestPhysics:
    def test_harmonic_x_squared_expectation(self):
        """omega=1, mass=1 の調和振動子では、厳密な基底状態期待値は
        <x^2> = 1/(2*omega) = 0.5 である。

        格子シミュレーションがこの値を 20% の相対誤差以内で再現することを確認する。
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

        # すべての格子点・すべての配位で平均した <x^2>
        x_sq_mean = np.mean(configs ** 2)
        expected = 0.5  # 1 / (2 * omega)

        assert x_sq_mean == pytest.approx(expected, rel=0.20), (
            f"<x^2> = {x_sq_mean:.4f}, 期待値 ~{expected} から 20% 以内に入るはず"
        )
