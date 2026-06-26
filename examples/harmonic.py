#!/usr/bin/env python3
"""
量子調和振動子の確率過程量子化の実行例。

単純な調和振動子 V(x) = (1/2) omega^2 x^2 に対して、Parisi-Wu の
確率過程量子化をユークリッド時間格子上で適用する例を示す。
このシミュレーションでは、以下の厳密な解析結果を再現する。

    <x^2>           = 1 / (2 omega)     = 0.5   （omega = 1 の場合）
    E_1 - E_0       = omega              = 1.0   （有効質量のプラトー）
    |psi_0(x)|^2    = sqrt(omega/pi) * exp(-omega x^2)

使い方:
    uv run python examples/harmonic.py
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# リポジトリルートまたは examples/ ディレクトリから単独実行できるようにする。
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.action import harmonic_force, harmonic_potential
from src.analytic import harmonic_analytic_results, harmonic_wavefunction_density
from src.langevin import LangevinSimulation
from src.exact import solve_spectrum
from src.observables import correlator, effective_mass, position_histogram
from src.analysis import jackknife
from src.runner import make_output_dir, save_summary


def main():
    # ------------------------------------------------------------------
    # 1. シミュレーションパラメータ
    # ------------------------------------------------------------------
    omega = 1.0          # 振動子の角振動数
    mass = 1.0           # 粒子の質量
    n_lattice = 100      # ユークリッド時間方向の格子点数
    a = 0.1              # 格子間隔（ユークリッド時間）
    epsilon = 0.01       # ランジュバンステップ幅（架空の確率時間）
    n_therm = 5000       # 熱化スイープ数
    n_configs = 5000     # 測定する配位数
    n_skip = 20          # 測定間に進めるランジュバンステップ数
    rng_seed = 42        # 再現性のための乱数シード
    max_tau = n_lattice // 2

    # 有限差分対角化のパラメータ
    x_min = -8.0
    x_max = 8.0
    n_grid = 350

    output_params = {
        "omega": omega,
        "mass": mass,
        "N": n_lattice,
        "a": a,
        "eps": epsilon,
        "therm": n_therm,
        "configs": n_configs,
        "skip": n_skip,
        "seed": rng_seed,
    }
    output_dir = make_output_dir("harmonic", output_params)
    analytic = harmonic_analytic_results(omega=omega, mass=mass)

    print("=" * 60)
    print("  確率過程量子化 -- 調和振動子")
    print("=" * 60)
    print(f"  omega     = {omega}")
    print(f"  mass      = {mass}")
    print(f"  N_lattice = {n_lattice},  a = {a},  T = {n_lattice * a:.1f}")
    print(f"  epsilon   = {epsilon}")
    print(f"  n_therm   = {n_therm}")
    print(f"  n_configs = {n_configs},  n_skip = {n_skip}")
    print(f"  rng_seed  = {rng_seed}")
    print(f"  output    = {output_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. 解析解と有限差分対角化による比較用の数値解
    # ------------------------------------------------------------------
    print("\n解析解:")
    print(f"  E0       = {analytic['E0']:.6f}")
    print(f"  E1 - E0  = {analytic['gap']:.6f}")
    print(f"  <x^2>    = {analytic['x2']:.6f}")

    print("\n有限差分対角化で比較用のスペクトルを計算中 ...")
    exact_energies, x_grid, exact_wavefunctions = solve_spectrum(
        harmonic_potential,
        x_min=x_min,
        x_max=x_max,
        n_grid=n_grid,
        mass=mass,
        n_levels=2,
        omega=omega,
    )
    exact_E0 = exact_energies[0]
    exact_gap = exact_energies[1] - exact_energies[0]
    exact_psi0 = exact_wavefunctions[:, 0]
    dx = x_grid[1] - x_grid[0]
    exact_x2 = np.sum((x_grid**2) * np.abs(exact_psi0) ** 2) * dx

    print(f"  E0（有限差分）      = {exact_E0:.6f}")
    print(f"  E1 - E0（有限差分） = {exact_gap:.6f}")
    print(f"  <x^2>（有限差分）   = {exact_x2:.6f}")

    # ------------------------------------------------------------------
    # 3. ランジュバンシミュレーションを実行
    # ------------------------------------------------------------------
    print("\nランジュバンシミュレーションを実行中 ...")
    sim = LangevinSimulation(
        n_lattice=n_lattice,
        a=a,
        mass=mass,
        force_func=harmonic_force,
        epsilon=epsilon,
        rng_seed=rng_seed,
        omega=omega,
    )
    configs = sim.run(n_therm=n_therm, n_configs=n_configs, n_skip=n_skip)
    print(f"  {configs.shape[1]} 個の格子点上で {configs.shape[0]} 個の配位を生成しました。")

    # ------------------------------------------------------------------
    # 4a. 物理量: <x^2> とジャックナイフ誤差
    # ------------------------------------------------------------------
    # 各配位ごとの <x^2>（各配位内で格子点平均を取る）
    x2_per_config = np.mean(configs ** 2, axis=1)

    x2_est, x2_err = jackknife(x2_per_config, func=np.mean)

    print("\n--- <x^2> ---")
    print(f"  解析解:         {analytic['x2']:.6f}")
    print(f"  有限差分対角化: {exact_x2:.6f}")
    print(f"  Parisi-Wu:      {x2_est:.6f} +/- {x2_err:.6f}")
    if x2_err > 0:
        print(f"  Parisi-Wu と解析解のずれ: {abs(x2_est - analytic['x2']) / x2_err:.1f} sigma")

    # ------------------------------------------------------------------
    # 4b. 相関関数と有効質量
    # ------------------------------------------------------------------
    corr = correlator(configs, max_tau=max_tau)
    m_eff = effective_mass(corr, a=a)

    print("\n--- 相関関数 C(tau) と有効質量 m_eff(tau) ---")
    print(f"  {'tau':>5s}  {'C(tau)':>12s}  {'m_eff':>12s}")
    # 代表的な tau の値を表示する
    tau_show = [0, 1, 2, 3, 5, 10, 15, 20]
    for t in tau_show:
        if t < len(corr):
            meff_str = f"{m_eff[t]:.6f}" if t < len(m_eff) and np.isfinite(m_eff[t]) else "---"
            print(f"  {t * a:5.1f}  {corr[t]:12.6f}  {meff_str:>12s}")

    # ------------------------------------------------------------------
    # 4c. 有効質量のプラトーからエネルギーギャップを確認
    # ------------------------------------------------------------------
    # 有効質量 m_eff(tau) は大きな tau で E_1 - E_0 = omega に近づく。
    # 小さい tau と大きすぎる tau を避け、プラトー領域で平均する。
    plateau_start = 5
    plateau_end = min(20, len(m_eff))
    plateau_vals = m_eff[plateau_start:plateau_end]
    plateau_vals = plateau_vals[np.isfinite(plateau_vals)]

    if len(plateau_vals) > 0:
        delta_E = np.mean(plateau_vals)
        print("\n--- エネルギーギャップ E1 - E0 ---")
        print(f"  解析解:         {analytic['gap']:.6f}")
        print(f"  有限差分対角化: {exact_gap:.6f}")
        print(f"  Parisi-Wu:      {delta_E:.6f}")
        print("\n--- 基底状態エネルギー E0 ---")
        print(f"  解析解:         {analytic['E0']:.6f}")
        print(f"  有限差分対角化: {exact_E0:.6f}")
        print("  Parisi-Wu:      この相関関数からは直接は測定していない")
    else:
        delta_E = np.nan
        print("\n  [警告] 安定した有効質量のプラトーを抽出できませんでした。")

    summary_path = save_summary(
        output_dir,
        output_params,
        {
            "x2_measured": x2_est,
            "x2_error": x2_err,
            "x2_analytic": analytic["x2"],
            "x2_exact_diagonalization": exact_x2,
            "effective_mass_plateau": delta_E,
            "energy_gap_analytic": analytic["gap"],
            "energy_gap_exact_diagonalization": exact_gap,
            "E0_analytic": analytic["E0"],
            "E0_exact_diagonalization": exact_E0,
            "plateau_start": plateau_start,
            "plateau_end": plateau_end,
            "exact_x_min": x_min,
            "exact_x_max": x_max,
            "exact_n_grid": n_grid,
        },
    )
    print(f"\n  実行条件と主要結果を保存しました -> {summary_path}")

    # ------------------------------------------------------------------
    # 4. プロット（任意。matplotlib が必要）
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")  # ヘッドレス環境向けの非対話バックエンド
        import matplotlib.pyplot as plt

        # ---- 有効質量 vs tau ----
        tau_vals = a * np.arange(len(m_eff))
        valid = np.isfinite(m_eff)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(tau_vals[valid], m_eff[valid], "o-", markersize=3, label=r"$m_{\rm eff}(\tau)$")
        ax.axhline(analytic["gap"], color="red", linestyle="--", linewidth=1.5,
                   label=rf"$E_1-E_0={analytic['gap']:.3f}$（解析解）")
        ax.axhline(exact_gap, color="purple", linestyle=":", linewidth=1.5,
                   label=rf"$E_1-E_0={exact_gap:.3f}$（有限差分）")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$m_{\rm eff}(\tau)$")
        ax.set_title("ユークリッド相関関数から求めた有効質量")
        ax.set_ylim(0, 2.5 * omega)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        eff_mass_path = os.path.join(output_dir, "effective_mass.png")
        fig.savefig(eff_mass_path, dpi=150)
        plt.close(fig)
        print(f"\n  有効質量のプロットを保存しました -> {eff_mass_path}")

        # ---- 波動関数ヒストグラム vs 厳密な |psi_0|^2 ----
        bin_centers, hist_vals = position_histogram(configs, n_bins=60)
        x_fine = np.linspace(bin_centers[0], bin_centers[-1], 300)
        psi2_analytic = harmonic_wavefunction_density(x_fine, omega=omega, mass=mass)
        psi2_exact = np.abs(exact_psi0) ** 2

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(bin_centers, hist_vals, width=(bin_centers[1] - bin_centers[0]),
               alpha=0.6, label="シミュレーションのヒストグラム")
        ax.plot(x_fine, psi2_analytic, "r-", linewidth=2,
                label=r"$|\psi_0(x)|^2$（解析解）")
        ax.plot(x_grid, psi2_exact, color="purple", linestyle=":", linewidth=2,
                label=r"$|\psi_0(x)|^2$（有限差分）")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|\psi_0(x)|^2$")
        ax.set_title("基底状態の波動関数")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        wf_path = os.path.join(output_dir, "wavefunction.png")
        fig.savefig(wf_path, dpi=150)
        plt.close(fig)
        print(f"  波動関数のプロットを保存しました -> {wf_path}")

    except ImportError:
        print("\n  [matplotlib が利用できないため、プロットをスキップします]")
    except Exception as e:
        print(f"\n  [プロットに失敗しました: {e}]")

    print("\n完了しました。")


if __name__ == "__main__":
    main()
