"""確率過程量子化（Parisi-Wu 法）のためのランジュバン方程式ソルバー。

周期境界条件を持つ離散化ユークリッド時間格子上で、次のランジュバン方程式を
オイラー・丸山法で積分する:

    x_j(t + eps) = x_j(t) + eps * f_j + sqrt(2 * eps) * eta_j

ここで f_j = -dS_E/dx_j はドリフト力、eta_j は標準正規分布 N(0, 1) に従う
ノイズである。
"""

import numpy as np

from sqm_ground_state.lattice import drift_force


class LangevinSimulation:
    """ユークリッド時間格子上のランジュバン力学シミュレーション。

    引数
    ----
    n_lattice : int
        格子点数（ユークリッド時間方向の離散化数）。
    a : float
        格子間隔。
    mass : float
        粒子の質量。
    force_func : 呼び出し可能
        ポテンシャルの導関数 dV/dx。drift_force に渡される。
    epsilon : float
        ランジュバンステップ幅（架空の確率時間の刻み幅）。
    rng_seed : int, 省略可
        乱数生成器のシード（再現性のため）。
    **pot_params
        drift_force に渡す追加のキーワード引数（例: omega, lam）。
    """

    def __init__(self, n_lattice, a, mass, force_func, epsilon,
                 rng_seed=None, **pot_params):
        self.n_lattice = n_lattice
        self.a = a
        self.mass = mass
        self.force_func = force_func
        self.epsilon = epsilon
        self.pot_params = pot_params

        self.rng = np.random.default_rng(rng_seed)
        self.x = np.zeros(n_lattice)

    def step(self):
        """全格子点に対してオイラー・丸山法によるランジュバンステップを 1 回進める。"""
        force = drift_force(self.x, self.force_func,
                            mass=self.mass, a=self.a, **self.pot_params)
        noise = self.rng.standard_normal(self.n_lattice)
        self.x = self.x + self.epsilon * force + np.sqrt(2.0 * self.epsilon) * noise

    def thermalize(self, n_therm):
        """熱平衡に近づけるため、n_therm 回のランジュバンステップを実行する。"""
        for _ in range(n_therm):
            self.step()

    def generate_configurations(self, n_configs, n_skip=10):
        """n_skip ステップ間隔で格子配位を収集する。

        引数
        ----
        n_configs : int
            収集する配位数。
        n_skip : int
            連続するサンプルの間に進めるランジュバンステップ数
            （自己相関を減らすため）。

        戻り値
        ------
        configs : np.ndarray, 形状 (n_configs, n_lattice)
            サンプリングされた経路配位の配列。
        """
        configs = np.empty((n_configs, self.n_lattice))
        for i in range(n_configs):
            for _ in range(n_skip):
                self.step()
            configs[i] = self.x.copy()
        return configs

    def run(self, n_therm, n_configs, n_skip=10):
        """熱化を行った後、配位を生成する。

        thermalize を呼んだ後に generate_configurations を呼ぶための便利メソッド。

        引数
        ----
        n_therm : int
            熱化ステップ数。
        n_configs : int
            収集する配位数。
        n_skip : int
            測定間にスキップするステップ数。

        戻り値
        ------
        configs : np.ndarray, 形状 (n_configs, n_lattice)
            熱化後にサンプリングされた経路配位。
        """
        self.thermalize(n_therm)
        return self.generate_configurations(n_configs, n_skip)
