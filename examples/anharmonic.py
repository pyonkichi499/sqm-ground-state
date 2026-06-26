#!/usr/bin/env python3
"""
非調和振動子の確率過程量子化の実行例。

対象とするポテンシャルは

    V(x) = (1/2) omega^2 x^2 + lambda x^4

である。調和振動子とは異なり単純な解析解はないため、この example では
1 次元シュレーディンガー方程式を有限差分で対角化した結果と比較する。

使い方:
    uv run python examples/anharmonic.py
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# リポジトリルートまたは examples/ ディレクトリから単独実行できるようにする。
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.action import anharmonic_force, anharmonic_potential
from src.analysis import jackknife
from src.exact import solve_spectrum
from src.langevin import LangevinSimulation
from src.observables import correlator, effective_mass, position_histogram
from src.runner import make_output_dir, save_summary


def main():
    # ------------------------------------------------------------------
    # 1. シミュレーションパラメータ
    # ------------------------------------------------------------------
    omega = 1.0          # 二次項の角振動数
    lam = 0.1            # 四次相互作用の強さ
    mass = 1.0           # 粒子の質量
    n_lattice = 100      # ユークリッド時間方向の格子点数
    a = 0.1              # 格子間隔（ユークリッド時間）
    epsilon = 0.005      # ランジュバンステップ幅（非調和項があるため少し小さめ）
    n_therm = 7000       # 熱化スイープ数
    n_configs = 5000     # 測定する配位数
    n_skip = 25          # 測定間に進めるランジュバンステップ数
    rng_seed = 123       # 再現性のための乱数シード
    max_tau = n_lattice // 2

    # 有限差分対角化のパラメータ
    x_min = -6.0
    x_max = 6.0
    n_grid = 350

    output_params = {
        "omega": omega,
        "lambda": lam,
        "mass": mass,
        "N": n_lattice,
        "a": a,
        "eps": epsilon,
        "therm": n_therm,
        "configs": n_configs,
        "skip": n_skip,
        "seed": rng_seed,
    }
    output_dir = make_output_dir("anharmonic", output_params)

    print("=" * 60)
    print("  確率過程量子化 -- 非調和振動子")
    print("=" * 60)
    print(f"  omega     = {omega}")
    print(f"  lambda    = {lam}")
    print(f"  mass      = {mass}")
    print(f"  N_lattice = {n_lattice},  a = {a},  T = {n_lattice * a:.1f}")
    print(f"  epsilon   = {epsilon}")
    print(f"  n_therm   = {n_therm}")
    print(f"  n_configs = {n_configs},  n_skip = {n_skip}")
    print(f"  rng_seed  = {rng_seed}")
    print(f"  output    = {output_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. 有限差分対角化による比較用の数値解
    # ------------------------------------------------------------------
    print("\n有限差分対角化で比較用のスペクトルを計算中 ...")
    exact_energies, x_grid, exact_wavefunctions = solve_spectrum(
        anharmonic_potential,
        x_min=x_min,
        x_max=x_max,
        n_grid=n_grid,
        mass=mass,
        n_levels=2,
        omega=omega,
        lam=lam,
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
        force_func=anharmonic_force,
        epsilon=epsilon,
        rng_seed=rng_seed,
        omega=omega,
        lam=lam,
    )
    configs = sim.run(n_therm=n_therm, n_configs=n_configs, n_skip=n_skip)
    print(f"  {configs.shape[1]} 個の格子点上で {configs.shape[0]} 個の配位を生成しました。")

    # ------------------------------------------------------------------
    # 4a. 物理量: <x^2> とジャックナイフ誤差
    # ------------------------------------------------------------------
    x2_per_config = np.mean(configs ** 2, axis=1)
    x2_est, x2_err = jackknife(x2_per_config, func=np.mean)

    print("\n--- <x^2> ---")
    print(f"  測定値:         {x2_est:.6f} +/- {x2_err:.6f}")
    print(f"  有限差分対角化: {exact_x2:.6f}")
    if x2_err > 0:
        print(f"  ずれ:           {abs(x2_est - exact_x2) / x2_err:.1f} sigma")

    # ------------------------------------------------------------------
    # 4b. 相関関数と有効質量
    # ------------------------------------------------------------------
    corr = correlator(configs, max_tau=max_tau)
    m_eff = effective_mass(corr, a=a)

    print("\n--- 相関関数 C(tau) と有効質量 m_eff(tau) ---")
    print(f"  {'tau':>5s}  {'C(tau)':>12s}  {'m_eff':>12s}")
    tau_show = [0, 1, 2, 3, 5, 10, 15, 20]
    for t in tau_show:
        if t < len(corr):
            meff_str = f"{m_eff[t]:.6f}" if t < len(m_eff) and np.isfinite(m_eff[t]) else "---"
            print(f"  {t * a:5.1f}  {corr[t]:12.6f}  {meff_str:>12s}")

    plateau_start = 5
    plateau_end = min(20, len(m_eff))
    plateau_vals = m_eff[plateau_start:plateau_end]
    plateau_vals = plateau_vals[np.isfinite(plateau_vals)]

    if len(plateau_vals) > 0:
        delta_E = np.mean(plateau_vals)
        print("\n--- エネルギーギャップ ---")
        print(f"  E1 - E0（プラトー平均） = {delta_E:.6f}")
        print(f"  E1 - E0（有限差分）     = {exact_gap:.6f}")
    else:
        delta_E = np.nan
        print("\n  [警告] 安定した有効質量のプラトーを抽出できませんでした。")

    summary_path = save_summary(
        output_dir,
        output_params,
        {
            "x2_measured": x2_est,
            "x2_error": x2_err,
            "x2_exact_diagonalization": exact_x2,
            "effective_mass_plateau": delta_E,
            "energy_gap_exact_diagonalization": exact_gap,
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
    # 5. プロット（任意。matplotlib が必要）
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ---- 有効質量 vs tau ----
        tau_vals = a * np.arange(len(m_eff))
        valid = np.isfinite(m_eff)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(tau_vals[valid], m_eff[valid], "o-", markersize=3, label=r"$m_{\rm eff}(\tau)$")
        ax.axhline(exact_gap, color="red", linestyle="--", linewidth=1.5,
                   label=rf"$E_1 - E_0 = {exact_gap:.3f}$（有限差分）")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$m_{\rm eff}(\tau)$")
        ax.set_title("非調和振動子の有効質量")
        ax.set_ylim(0, 2.5 * exact_gap)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        eff_mass_path = os.path.join(output_dir, "effective_mass.png")
        fig.savefig(eff_mass_path, dpi=150)
        plt.close(fig)
        print(f"\n  有効質量のプロットを保存しました -> {eff_mass_path}")

        # ---- 波動関数ヒストグラム vs 有限差分の |psi_0|^2 ----
        bin_centers, hist_vals = position_histogram(configs, n_bins=70)
        psi2_exact = np.abs(exact_psi0) ** 2

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(bin_centers, hist_vals, width=(bin_centers[1] - bin_centers[0]),
               alpha=0.6, label="シミュレーションのヒストグラム")
        ax.plot(x_grid, psi2_exact, "r-", linewidth=2,
                label=r"$|\psi_0(x)|^2$（有限差分）")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|\psi_0(x)|^2$")
        ax.set_title("非調和振動子の基底状態波動関数")
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
