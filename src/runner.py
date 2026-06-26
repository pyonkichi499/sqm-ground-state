"""
example スクリプト間で共有する出力・実行ヘルパー。

確率過程量子化の各 example（調和振動子、非調和振動子 など）は、
結果を outputs/ 以下のパラメータ別ディレクトリに保存する。
その出力先の生成・命名規則・summary.json への保存処理をここにまとめる。
"""

import json
import os

import numpy as np


def format_param_value(value):
    """出力ディレクトリ名に使いやすい形へパラメータ値を変換する。

    小数点はファイル名で扱いやすいように p に、負号は m に置き換える。
    例: 0.01 -> 0p01, 1.0 -> 1, -0.5 -> m0p5
    """
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return str(value).replace(".", "p").replace("-", "m")


def make_run_name(system_name, params):
    """物理系の名前とパラメータから、実行結果用のディレクトリ名を作る。

    命名規則:

        <物理系名>_<key1>_<value1>_<key2>_<value2>_...

    params は挿入順を保つ辞書として扱う。重要なパラメータから順に並べることで、
    ディレクトリ一覧を見たときに比較しやすくする。
    """
    parts = [system_name]
    for key, value in params.items():
        parts.append(f"{key}_{format_param_value(value)}")
    return "_".join(parts)


def repo_root():
    """このリポジトリのルートディレクトリの絶対パスを返す。"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_output_dir(system_name, params, base_dir=None):
    """物理系別・パラメータ別の出力先ディレクトリを作成して返す。

    例:

        outputs/harmonic/harmonic_omega_1_N_100_a_0p1/

    base_dir を省略するとリポジトリ直下の outputs/ を使う。outputs/ は
    .gitignore で無視されるため、多数のパラメータを試しても Git の
    管理対象には入らない。

    引数
    ----
    system_name : str
        物理系の名前（例: "harmonic", "anharmonic"）。
    params : dict
        ディレクトリ名に埋め込むパラメータ。挿入順が保たれる。
    base_dir : str, 省略可
        出力のベースディレクトリ。省略時は <repo>/outputs。

    戻り値
    ------
    str
        作成された出力ディレクトリの絶対パス。
    """
    if base_dir is None:
        base_dir = os.path.join(repo_root(), "outputs")
    run_name = make_run_name(system_name, params)
    output_dir = os.path.join(base_dir, system_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def to_builtin(value):
    """JSON に保存できる Python 標準型へ変換する。

    numpy のスカラー・配列はそれぞれ Python の数値・リストに変換する。
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_summary(output_dir, params, results):
    """実行パラメータと主要な測定結果を summary.json に保存する。

    引数
    ----
    output_dir : str
        保存先ディレクトリ。
    params : dict
        実行パラメータ。
    results : dict
        主要な測定結果。

    戻り値
    ------
    str
        保存した summary.json の絶対パス。
    """
    summary = {
        "parameters": {key: to_builtin(value) for key, value in params.items()},
        "results": {key: to_builtin(value) for key, value in results.items()},
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path
