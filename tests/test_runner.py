import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.runner import (
    format_param_value,
    make_output_dir,
    make_run_name,
    save_summary,
    to_builtin,
)


def test_format_param_value_float():
    """小数点は p、負号は m に変換される。"""
    assert format_param_value(0.01) == "0p01"
    assert format_param_value(-0.5) == "m0p5"
    assert format_param_value(1.0) == "1"


def test_format_param_value_int_and_string():
    """整数や文字列もディレクトリ名に使える形式へ変換される。"""
    assert format_param_value(100) == "100"
    assert format_param_value("double-well") == "doublemwell"


def test_make_run_name_preserves_parameter_order():
    """パラメータ辞書の挿入順を保った run 名を作る。"""
    params = {"omega": 1.0, "N": 100, "a": 0.1}
    assert make_run_name("harmonic", params) == "harmonic_omega_1_N_100_a_0p1"


def test_make_output_dir_creates_directory(tmp_path):
    """指定した base_dir の下に出力ディレクトリが作成される。"""
    params = {"omega": 1.0, "N": 16}
    output_dir = make_output_dir("harmonic", params, base_dir=tmp_path)

    assert os.path.isdir(output_dir)
    assert output_dir.endswith("harmonic/harmonic_omega_1_N_16")


def test_to_builtin_numpy_values():
    """numpy の値は JSON 保存可能な Python 標準型へ変換される。"""
    assert isinstance(to_builtin(np.float64(1.5)), float)
    assert to_builtin(np.array([1, 2, 3])) == [1, 2, 3]


def test_save_summary_writes_json(tmp_path):
    """summary.json にパラメータと結果が保存される。"""
    params = {"omega": np.float64(1.0), "N": 100}
    results = {"x2": np.float64(0.5), "corr": np.array([1.0, 0.8])}

    summary_path = save_summary(tmp_path, params, results)

    assert os.path.exists(summary_path)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["parameters"] == {"omega": 1.0, "N": 100}
    assert summary["results"] == {"x2": 0.5, "corr": [1.0, 0.8]}
