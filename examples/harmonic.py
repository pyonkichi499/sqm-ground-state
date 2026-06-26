#!/usr/bin/env python3
"""
量子調和振動子の確率過程量子化の実行例。

調和振動子 V(x) = (1/2) omega^2 x^2 について、次の 3 つを比較する。

1. 解析解
2. 非 Parisi-Wu の数値計算（有限差分対角化）
3. Parisi-Wu 確率過程量子化

共通の実行フローは src.experiment.run_experiment にまとまっているため、
この example はパラメータを設定して呼び出すだけである。

使い方:
    uv run python examples/harmonic.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiment import ExperimentConfig, run_experiment
from src.potentials import harmonic


def main():
    config = ExperimentConfig(
        potential=harmonic(omega=1.0),
        mass=1.0,
        n_lattice=100,
        a=0.1,
        epsilon=0.01,
        n_therm=5000,
        n_configs=5000,
        n_skip=20,
        rng_seed=42,
        plateau_start=5,
        plateau_end=20,
        exact_x_min=-8.0,
        exact_x_max=8.0,
        exact_n_grid=350,
        n_bins=60,
        title="調和振動子",
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
