#!/usr/bin/env python3
"""
非調和振動子の確率過程量子化の実行例。

ポテンシャルは V(x) = (1/2) omega^2 x^2 + lambda x^4。単純な解析解はないため、
有限差分対角化を独立な数値解として Parisi-Wu の結果と比較する。

使い方:
    uv run python examples/anharmonic.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiment import ExperimentConfig, run_experiment
from src.potentials import anharmonic


def main():
    config = ExperimentConfig(
        potential=anharmonic(omega=1.0, lam=0.1),
        mass=1.0,
        n_lattice=100,
        a=0.1,
        epsilon=0.005,   # 非調和項があるため少し小さめ
        n_therm=7000,
        n_configs=5000,
        n_skip=25,
        rng_seed=123,
        plateau_start=5,
        plateau_end=20,
        exact_x_min=-6.0,
        exact_x_max=6.0,
        exact_n_grid=350,
        n_bins=70,
        title="非調和振動子",
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
