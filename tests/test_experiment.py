import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.experiment import ExperimentConfig, run_experiment
from src.potentials import double_well, harmonic


def _fast_config(potential, base_dir, **overrides):
    """テスト用に小さく速い設定を作る。"""
    params = dict(
        potential=potential,
        mass=1.0,
        n_lattice=20,
        a=0.1,
        epsilon=0.01,
        n_therm=50,
        n_configs=40,
        n_skip=5,
        rng_seed=7,
        exact_n_grid=120,
        base_output_dir=str(base_dir),
        verbose=False,
        make_plots=False,
    )
    params.update(overrides)
    return ExperimentConfig(**params)


def test_run_experiment_harmonic_outputs(tmp_path):
    """調和振動子では解析解・有限差分・Parisi-Wu の 3 つが揃う。"""
    result = run_experiment(_fast_config(harmonic(omega=1.0), tmp_path))

    assert result["analytic"]["available"] is True
    assert result["finite_difference"]["E0"] > 0
    assert np.isfinite(result["parisi_wu"]["x2"])

    # summary.json が書かれている
    assert os.path.exists(result["summary_path"])
    with open(result["summary_path"], encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["results"]["x2_analytic"] is not None
    assert summary["results"]["x2_exact_diagonalization"] is not None
    assert summary["results"]["x2_measured"] is not None


def test_run_experiment_double_well_outputs(tmp_path):
    """二重井戸では解析解なしだが古典量と数値解は summary に入る。"""
    result = run_experiment(
        _fast_config(double_well(lam=1.0, v=1.0), tmp_path,
                     plateau_start=2, plateau_end=8)
    )

    assert result["analytic"]["available"] is False
    with open(result["summary_path"], encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["results"]["x2_analytic"] is None
    assert summary["results"]["analytic_reason"]
    assert summary["results"]["barrier_height"] == 1.0


def test_output_dir_name_includes_potential_and_params(tmp_path):
    """出力ディレクトリ名に物理系名と主要パラメータが含まれる。"""
    result = run_experiment(_fast_config(harmonic(omega=1.0), tmp_path))
    output_dir = result["output_dir"]
    assert "harmonic" in output_dir
    assert "omega_1" in output_dir


@pytest.mark.parametrize(
    "overrides, error_type",
    [
        ({"potential": object()}, TypeError),
        ({"mass": 0.0}, ValueError),
        ({"n_lattice": 1}, ValueError),
        ({"a": 0.0}, ValueError),
        ({"epsilon": 0.0}, ValueError),
        ({"n_therm": -1}, ValueError),
        ({"n_configs": 0}, ValueError),
        ({"n_skip": -1}, ValueError),
        ({"plateau_start": -1}, ValueError),
        ({"plateau_start": 5, "plateau_end": 5}, ValueError),
        ({"exact_x_min": 1.0, "exact_x_max": 1.0}, ValueError),
        ({"exact_n_grid": 2}, ValueError),
        ({"n_bins": 0}, ValueError),
    ],
)
def test_experiment_config_rejects_invalid_values(overrides, error_type):
    """明らかに不正な設定値は、計算前に例外で止める。"""
    params = {"potential": harmonic(omega=1.0)}
    params.update(overrides)

    with pytest.raises(error_type):
        ExperimentConfig(**params)
