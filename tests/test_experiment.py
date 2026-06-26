import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.experiment import (
    PLOT_FONT_FAMILY,
    ExperimentConfig,
    PlotFontNotFoundError,
    _configure_plot_fonts,
    _configure_plot_style,
    _resolve_plot_font,
    run_experiment,
)
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


def test_plot_font_family_is_noto_sans_cjk_jp():
    """プロット用フォントは Noto Sans CJK JP に決め打ちする。"""
    assert PLOT_FONT_FAMILY == "Noto Sans CJK JP"


def test_resolve_plot_font_registers_system_font_when_needed():
    """OS に fonts-noto-cjk がある場合、matplotlib 未登録でも解決できる。"""
    import src.experiment as experiment

    experiment._resolved_plot_font = None
    resolved = _resolve_plot_font()
    assert resolved in experiment._PLOT_FONT_ALIASES


def test_resolve_plot_font_raises_when_missing(monkeypatch):
    """日本語フォントが無い環境では PlotFontNotFoundError を出す。"""
    import src.experiment as experiment

    monkeypatch.setattr(experiment, "_lookup_plot_font", lambda _fm: None)
    monkeypatch.setattr(experiment, "_register_plot_font_from_system", lambda _fm: None)

    with pytest.raises(PlotFontNotFoundError, match=PLOT_FONT_FAMILY):
        _resolve_plot_font()


def test_configure_plot_fonts_sets_single_resolved_font(monkeypatch):
    """rcParams には解決済みフォント名を 1 つだけ設定する。"""
    import matplotlib.pyplot as plt

    import src.experiment as experiment

    monkeypatch.setattr(experiment, "_resolved_plot_font", "Noto Sans CJK JP")

    _configure_plot_fonts(plt)

    family = plt.rcParams["font.family"]
    if isinstance(family, list):
        family = family[0]
    assert family == "Noto Sans CJK JP"
    assert plt.rcParams["font.sans-serif"] == ["Noto Sans CJK JP"]
    assert plt.rcParams["axes.unicode_minus"] is False


def test_configure_plot_style_uses_whitegrid():
    """seaborn が利用できる環境では whitegrid スタイルを使う。"""
    import matplotlib.pyplot as plt

    import src.experiment as experiment

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiment, "_resolved_plot_font", "Noto Sans CJK JP")
    try:
        style_name = _configure_plot_style(plt)

        assert style_name in {
            "seaborn-whitegrid",
            "matplotlib-seaborn-v0_8-whitegrid",
            "matplotlib-default",
        }
        family = plt.rcParams["font.family"]
        if isinstance(family, list):
            family = family[0]
        assert family == "Noto Sans CJK JP"
        assert plt.rcParams["axes.unicode_minus"] is False
        if style_name != "matplotlib-default":
            assert plt.rcParams["axes.grid"] is True
    finally:
        monkeypatch.undo()

