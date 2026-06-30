"""
3 者比較（解析解 / 有限差分対角化 / Parisi-Wu）の実行オーケストレーション。

調和振動子・非調和振動子・二重井戸の example はほぼ同じ流れを持っていた。

1. 解析的に分かる量を表示
2. 有限差分対角化で独立な数値解を計算
3. ランジュバンシミュレーションで Parisi-Wu の結果を計算
4. <x^2>・エネルギーギャップ・波動関数を比較
5. summary.json とプロットを保存

その共通フローをここに 1 か所だけ実装する。各 example は ``ExperimentConfig``
を用意して ``run_experiment`` を呼ぶだけでよい。
"""

from dataclasses import dataclass

import numpy as np

from sqm_ground_state.analysis import jackknife
from sqm_ground_state.exact import solve_spectrum
from sqm_ground_state.langevin import LangevinSimulation
from sqm_ground_state.observables import correlator, effective_mass, position_histogram
from sqm_ground_state.potentials import Potential
from sqm_ground_state.runner import make_output_dir, save_summary


@dataclass
class ExperimentConfig:
    """1 回の 3 者比較実験の設定。

    引数
    ----
    potential : Potential
        対象のポテンシャル（V, V', 解析的に分かる量を保持する）。
    mass : float
        粒子の質量。
    n_lattice, a : int, float
        ユークリッド時間格子の点数と格子間隔。
    epsilon, n_therm, n_configs, n_skip, rng_seed :
        ランジュバンシミュレーションのパラメータ。
    plateau_start, plateau_end : int
        有効質量プラトーを平均する tau/a の範囲 [start, end)。
        系によって良いプラトー領域が異なるので調整できるようにする。
    exact_x_min, exact_x_max, exact_n_grid :
        有限差分対角化の位置空間の範囲と格子点数。
    n_bins : int
        波動関数ヒストグラムのビン数。
    title : str
        プロットのタイトルに使う表示名。
    """

    potential: Potential
    mass: float = 1.0
    n_lattice: int = 100
    a: float = 0.1
    epsilon: float = 0.01
    n_therm: int = 5000
    n_configs: int = 5000
    n_skip: int = 20
    rng_seed: int = 42
    plateau_start: int = 5
    plateau_end: int = 20
    exact_x_min: float = -8.0
    exact_x_max: float = 8.0
    exact_n_grid: int = 350
    n_bins: int = 60
    title: str = ""

    #: 出力ディレクトリの基底（テストで差し替え可能）。None なら <repo>/outputs。
    base_output_dir: str = None

    #: 標準出力へログを出すかどうか。
    verbose: bool = True

    #: プロットを生成するかどうか。
    make_plots: bool = True

    def __post_init__(self):
        """設定値の明らかな誤りを早い段階で検出する。"""
        if not isinstance(self.potential, Potential):
            raise TypeError("potential は sqm_ground_state.potentials.Potential のインスタンスである必要がある")
        if self.mass <= 0:
            raise ValueError("mass は正である必要がある")
        if self.n_lattice < 2:
            raise ValueError("n_lattice は 2 以上である必要がある")
        if self.a <= 0:
            raise ValueError("a は正である必要がある")
        if self.epsilon <= 0:
            raise ValueError("epsilon は正である必要がある")
        if self.n_therm < 0:
            raise ValueError("n_therm は 0 以上である必要がある")
        if self.n_configs < 1:
            raise ValueError("n_configs は 1 以上である必要がある")
        if self.n_skip < 0:
            raise ValueError("n_skip は 0 以上である必要がある")
        if self.plateau_start < 0:
            raise ValueError("plateau_start は 0 以上である必要がある")
        if self.plateau_end <= self.plateau_start:
            raise ValueError("plateau_end は plateau_start より大きい必要がある")
        if self.exact_x_max <= self.exact_x_min:
            raise ValueError("exact_x_max は exact_x_min より大きい必要がある")
        if self.exact_n_grid < 3:
            raise ValueError("exact_n_grid は 3 以上である必要がある")
        if self.n_bins < 1:
            raise ValueError("n_bins は 1 以上である必要がある")

    def output_params(self):
        """出力ディレクトリ名・summary に埋め込む順序つきパラメータ。"""
        params = dict(self.potential.display_params)
        params.update(
            {
                "mass": self.mass,
                "N": self.n_lattice,
                "a": self.a,
                "eps": self.epsilon,
                "therm": self.n_therm,
                "configs": self.n_configs,
                "skip": self.n_skip,
                "seed": self.rng_seed,
            }
        )
        return params


def _log(config, *args):
    if config.verbose:
        print(*args)


def _run_finite_difference(config):
    """有限差分対角化で E0・ギャップ・<x^2>・基底波動関数を求める。"""
    energies, x_grid, wavefunctions = solve_spectrum(
        lambda x: config.potential.potential(x),
        x_min=config.exact_x_min,
        x_max=config.exact_x_max,
        n_grid=config.exact_n_grid,
        mass=config.mass,
        n_levels=2,
    )
    E0 = float(energies[0])
    gap = float(energies[1] - energies[0])
    psi0 = wavefunctions[:, 0]
    dx = x_grid[1] - x_grid[0]
    x2 = float(np.sum((x_grid**2) * np.abs(psi0) ** 2) * dx)
    return {
        "E0": E0,
        "gap": gap,
        "x2": x2,
        "x_grid": x_grid,
        "psi0_density": np.abs(psi0) ** 2,
    }


def _run_parisi_wu(config):
    """ランジュバンシミュレーションで <x^2>・相関関数・有効質量を求める。"""
    sim = LangevinSimulation(
        n_lattice=config.n_lattice,
        a=config.a,
        mass=config.mass,
        force_func=lambda x: config.potential.force(x),
        epsilon=config.epsilon,
        rng_seed=config.rng_seed,
    )
    configs = sim.run(
        n_therm=config.n_therm,
        n_configs=config.n_configs,
        n_skip=config.n_skip,
    )

    x2_per_config = np.mean(configs**2, axis=1)
    x2_est, x2_err = jackknife(x2_per_config, func=np.mean)

    max_tau = config.n_lattice // 2
    corr = correlator(configs, max_tau=max_tau)
    m_eff = effective_mass(corr, a=config.a)

    plateau_end = min(config.plateau_end, len(m_eff))
    plateau_vals = m_eff[config.plateau_start:plateau_end]
    plateau_vals = plateau_vals[np.isfinite(plateau_vals)]
    gap = float(np.mean(plateau_vals)) if len(plateau_vals) > 0 else float("nan")

    return {
        "configs": configs,
        "x2": float(x2_est),
        "x2_error": float(x2_err),
        "corr": corr,
        "m_eff": m_eff,
        "gap": gap,
        "plateau_start": config.plateau_start,
        "plateau_end": plateau_end,
    }


def _format(value, spec="{:.6f}"):
    """None は「なし」、数値は整形して返す表示ヘルパー。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "なし"
    return spec.format(value)


# プロットの日本語ラベル用フォント。OS にインストール済みであることを前提とする。
# インストール手順は README.md の「日本語フォント」を参照。
PLOT_FONT_FAMILY = "Noto Sans CJK JP"

# fontconfig 上の登録名の揺れを吸収するための別名（rcParams には解決後の 1 名だけ渡す）。
_PLOT_FONT_ALIASES = (
    PLOT_FONT_FAMILY,
    "Noto Sans CJK",
)

_resolved_plot_font = None


class PlotFontNotFoundError(RuntimeError):
    """日本語プロット用フォントが OS に見つからない。"""


def _plot_font_not_found_message():
    return (
        f"日本語プロット用フォント '{PLOT_FONT_FAMILY}' が見つかりません。\n"
        "README.md の「日本語フォント」を参照して OS にインストールしてください。\n"
        "例 (Debian/Ubuntu/WSL):\n"
        "  sudo apt install fonts-noto-cjk\n"
        "  fc-cache -fv"
    )


def _register_plot_font_from_system(fm):
    """OS に fonts-noto-cjk があるが matplotlib 未登録の場合に addfont する。"""
    for path in fm.findSystemFonts():
        if path.endswith("NotoSansCJK-Regular.ttc"):
            fm.fontManager.addfont(path)
            return path
    return None


def _lookup_plot_font(fm):
    """登録済みフォント名から Noto Sans CJK JP を探す。"""
    for name in _PLOT_FONT_ALIASES:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return None


def _resolve_plot_font():
    """日本語プロット用フォントを 1 回だけ解決する。

    戻り値
    ------
    str
        matplotlib に渡す font family 名。
    """
    from matplotlib import font_manager as fm

    resolved = _lookup_plot_font(fm)
    if resolved is not None:
        return resolved

    # apt で fonts-noto-cjk を入れても、matplotlib の font cache に未登録のことがある。
    if _register_plot_font_from_system(fm) is not None:
        resolved = _lookup_plot_font(fm)
        if resolved is not None:
            return resolved

    raise PlotFontNotFoundError(_plot_font_not_found_message())


def _configure_plot_fonts(plt):
    """matplotlib の描画フォントを設定する。

    日本語ラベル表示のため ``PLOT_FONT_FAMILY`` を前提とする。
    未インストールの場合は :class:`PlotFontNotFoundError` を送出する。
    """
    global _resolved_plot_font
    if _resolved_plot_font is None:
        _resolved_plot_font = _resolve_plot_font()

    plt.rcParams["font.family"] = _resolved_plot_font
    plt.rcParams["font.sans-serif"] = [_resolved_plot_font]
    plt.rcParams["axes.unicode_minus"] = False


def _configure_plot_style(plt):
    """白基調のプロットスタイルを設定する。

    seaborn が利用できる場合は ``whitegrid`` を使う。利用できない場合でも、
    matplotlib 側の白基調スタイルへフォールバックする。

    戻り値
    ------
    str
        実際に適用したスタイル名。
    """
    _configure_plot_fonts(plt)

    try:
        import seaborn as sns

        sns.set_theme(
            style="whitegrid",
            context="notebook",
            palette="deep",
        )
        _configure_plot_fonts(plt)
        return "seaborn-whitegrid"
    except ImportError:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            style_name = "matplotlib-seaborn-v0_8-whitegrid"
        except OSError:
            plt.style.use("default")
            style_name = "matplotlib-default"
        _configure_plot_fonts(plt)
        return style_name


def _print_header(config, output_dir):
    _log(config, "=" * 60)
    _log(config, f"  確率過程量子化 -- {config.title or config.potential.name}")
    _log(config, "=" * 60)
    for key, value in config.output_params().items():
        _log(config, f"  {key:8s} = {value}")
    _log(config, f"  T        = {config.n_lattice * config.a:.1f}")
    _log(config, f"  output   = {output_dir}")
    _log(config, "=" * 60)


def _print_comparison(config, analytic, fd, pw):
    _log(config, "\n--- 解析的に分かる量 ---")
    if analytic["available"]:
        _log(config, f"  E0      = {_format(analytic['E0'])}")
        _log(config, f"  E1 - E0 = {_format(analytic['gap'])}")
        _log(config, f"  <x^2>   = {_format(analytic['x2'])}")
    else:
        _log(config, f"  量子スペクトルの解析解: なし（{analytic['reason']}）")
        if "classical_minima" in analytic:
            _log(config, f"  古典的極小点:       x = {analytic['classical_minima']}")
            _log(config, f"  障壁高さ:           {_format(analytic['barrier_height'])}")
            _log(config, f"  井戸底の局所振動数: {_format(analytic['local_harmonic_omega'])}")

    _log(config, "\n--- <x^2> ---")
    _log(config, f"  解析解:         {_format(analytic['x2'])}")
    _log(config, f"  有限差分対角化: {_format(fd['x2'])}")
    _log(config, f"  Parisi-Wu:      {_format(pw['x2'])} +/- {_format(pw['x2_error'])}")

    _log(config, "\n--- エネルギーギャップ E1 - E0 ---")
    _log(config, f"  解析解:         {_format(analytic['gap'])}")
    _log(config, f"  有限差分対角化: {_format(fd['gap'])}")
    _log(config, f"  Parisi-Wu:      {_format(pw['gap'])}")

    _log(config, "\n--- 基底状態エネルギー E0 ---")
    _log(config, f"  解析解:         {_format(analytic['E0'])}")
    _log(config, f"  有限差分対角化: {_format(fd['E0'])}")
    _log(config, "  Parisi-Wu:      この相関関数からは直接は測定していない")


def _make_plots(config, output_dir, analytic, fd, pw):
    """有効質量と波動関数のプロットを保存する。"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # ヘッドレス環境向けの非対話バックエンド
        import matplotlib.pyplot as plt
    except ImportError:
        _log(config, "\n  [matplotlib が利用できないため、プロットをスキップします]")
        return {}

    try:
        _configure_plot_style(plt)
    except PlotFontNotFoundError as exc:
        _log(config, f"\n  [プロットをスキップします: {exc}]")
        return {}

    import os

    paths = {}
    m_eff = pw["m_eff"]
    tau_vals = config.a * np.arange(len(m_eff))
    valid = np.isfinite(m_eff)

    # ---- 有効質量 vs tau ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau_vals[valid], m_eff[valid], "o-", markersize=3, label=r"$m_{\rm eff}(\tau)$")
    if analytic["available"] and analytic["gap"] is not None:
        ax.axhline(analytic["gap"], color="red", linestyle="--", linewidth=1.5,
                   label=rf"$E_1-E_0={analytic['gap']:.3f}$（解析解）")
    ax.axhline(fd["gap"], color="purple", linestyle=":", linewidth=1.5,
               label=rf"$E_1-E_0={fd['gap']:.3f}$（有限差分）")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$m_{\rm eff}(\tau)$")
    ax.set_title(f"{config.title or config.potential.name} の有効質量")
    ax.set_ylim(0, max(2.5 * fd["gap"], 1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    eff_mass_path = os.path.join(output_dir, "effective_mass.png")
    fig.savefig(eff_mass_path, dpi=150)
    plt.close(fig)
    paths["effective_mass"] = eff_mass_path
    _log(config, f"\n  有効質量のプロットを保存しました -> {eff_mass_path}")

    # ---- 波動関数ヒストグラム ----
    bin_centers, hist_vals = position_histogram(pw["configs"], n_bins=config.n_bins)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(bin_centers, hist_vals, width=(bin_centers[1] - bin_centers[0]),
           alpha=0.6, label="シミュレーションのヒストグラム")
    if analytic["available"] and "density" in analytic:
        x_fine = np.linspace(bin_centers[0], bin_centers[-1], 300)
        ax.plot(x_fine, analytic["density"](x_fine), "r-", linewidth=2,
                label=r"$|\psi_0(x)|^2$（解析解）")
    ax.plot(fd["x_grid"], fd["psi0_density"], color="purple", linestyle=":", linewidth=2,
            label=r"$|\psi_0(x)|^2$（有限差分）")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\psi_0(x)|^2$")
    ax.set_title(f"{config.title or config.potential.name} の基底状態波動関数")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    wf_path = os.path.join(output_dir, "wavefunction.png")
    fig.savefig(wf_path, dpi=150)
    plt.close(fig)
    paths["wavefunction"] = wf_path
    _log(config, f"  波動関数のプロットを保存しました -> {wf_path}")

    return paths


def run_experiment(config):
    """3 者比較実験を実行し、結果辞書を返す。

    解析解・有限差分対角化・Parisi-Wu の結果を計算し、ログ出力・summary.json
    保存・プロット保存を行う。

    戻り値
    ------
    dict
        ``analytic`` / ``finite_difference`` / ``parisi_wu`` / ``output_dir`` /
        ``summary_path`` を含む辞書。
    """
    params = config.output_params()
    output_dir = make_output_dir(
        config.potential.name, params, base_dir=config.base_output_dir
    )
    analytic = config.potential.analytic(mass=config.mass)

    _print_header(config, output_dir)

    _log(config, "\n有限差分対角化で比較用のスペクトルを計算中 ...")
    fd = _run_finite_difference(config)

    _log(config, "\nランジュバンシミュレーションを実行中 ...")
    pw = _run_parisi_wu(config)
    _log(config, f"  {pw['configs'].shape[1]} 個の格子点上で "
                 f"{pw['configs'].shape[0]} 個の配位を生成しました。")

    _print_comparison(config, analytic, fd, pw)

    summary_results = {
        "x2_analytic": analytic["x2"],
        "x2_exact_diagonalization": fd["x2"],
        "x2_measured": pw["x2"],
        "x2_error": pw["x2_error"],
        "energy_gap_analytic": analytic["gap"],
        "energy_gap_exact_diagonalization": fd["gap"],
        "effective_mass_plateau": pw["gap"],
        "E0_analytic": analytic["E0"],
        "E0_exact_diagonalization": fd["E0"],
        "plateau_start": pw["plateau_start"],
        "plateau_end": pw["plateau_end"],
        "exact_x_min": config.exact_x_min,
        "exact_x_max": config.exact_x_max,
        "exact_n_grid": config.exact_n_grid,
    }
    if not analytic["available"]:
        summary_results["analytic_reason"] = analytic["reason"]
    for key in ("classical_minima", "barrier_height",
                "local_harmonic_omega", "local_harmonic_E0_approx"):
        if key in analytic:
            summary_results[key] = analytic[key]

    summary_path = save_summary(output_dir, params, summary_results)
    _log(config, f"\n  実行条件と主要結果を保存しました -> {summary_path}")

    plot_paths = {}
    if config.make_plots:
        plot_paths = _make_plots(config, output_dir, analytic, fd, pw)

    _log(config, "\n完了しました。")

    return {
        "analytic": analytic,
        "finite_difference": fd,
        "parisi_wu": pw,
        "output_dir": output_dir,
        "summary_path": summary_path,
        "plot_paths": plot_paths,
    }
