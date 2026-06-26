#!/usr/bin/env python3
"""
二重井戸ポテンシャルの確率過程量子化の実行例。

ポテンシャルは V(x) = lambda (x^2 - v^2)^2。量子スペクトルの閉じた解析解は
一般にはないが、極小点・障壁高さ・井戸底の局所振動数は解析的に分かる。
有限差分対角化を独立な数値解として Parisi-Wu の結果と比較する。

使い方:
    uv run python examples/double_well.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiment import ExperimentConfig, run_experiment
from src.potentials import double_well


def main():
    config = ExperimentConfig(
        potential=double_well(lam=1.0, v=1.0),
        mass=1.0,
        n_lattice=100,
        a=0.1,
        epsilon=0.003,   # 二重井戸では小さめにする
        n_therm=9000,
        n_configs=5000,
        n_skip=25,
        rng_seed=321,
        # 二重井戸では今回のパラメータだと比較的短い tau でプラトーが見える。
        # 大きすぎる tau では相関関数が小さくなり統計ノイズの影響が増える。
        plateau_start=2,
        plateau_end=8,
        exact_x_min=-5.0,
        exact_x_max=5.0,
        exact_n_grid=350,
        n_bins=70,
        title="二重井戸ポテンシャル",
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
