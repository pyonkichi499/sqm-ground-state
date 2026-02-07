#!/usr/bin/env python3
"""
Stochastic quantization of the quantum harmonic oscillator.

Demonstrates the Parisi-Wu stochastic quantization method applied to the
simple harmonic oscillator V(x) = (1/2) omega^2 x^2 on a Euclidean time
lattice.  The simulation recovers several exact analytic results:

    <x^2>           = 1 / (2 omega)     = 0.5   (for omega = 1)
    E_1 - E_0       = omega              = 1.0   (effective mass plateau)
    |psi_0(x)|^2    = sqrt(omega/pi) * exp(-omega x^2)

Usage:
    python examples/harmonic.py
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Allow standalone execution from the repository root or the examples/ folder.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.action import harmonic_force, harmonic_potential
from src.langevin import LangevinSimulation
from src.observables import correlator, effective_mass, position_histogram, mean_x_squared
from src.analysis import jackknife


def main():
    # ------------------------------------------------------------------
    # 1.  Simulation parameters
    # ------------------------------------------------------------------
    omega = 1.0          # oscillator frequency
    mass = 1.0           # particle mass
    n_lattice = 100      # number of Euclidean time sites
    a = 0.1              # lattice spacing (Euclidean time)
    epsilon = 0.01       # Langevin step size (fictitious time)
    n_therm = 5000       # thermalization sweeps
    n_configs = 5000     # measurement configurations
    n_skip = 20          # Langevin steps between measurements
    rng_seed = 42        # for reproducibility

    print("=" * 60)
    print("  Stochastic Quantization -- Harmonic Oscillator")
    print("=" * 60)
    print(f"  omega     = {omega}")
    print(f"  mass      = {mass}")
    print(f"  N_lattice = {n_lattice},  a = {a},  T = {n_lattice * a:.1f}")
    print(f"  epsilon   = {epsilon}")
    print(f"  n_therm   = {n_therm}")
    print(f"  n_configs = {n_configs},  n_skip = {n_skip}")
    print(f"  rng_seed  = {rng_seed}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2.  Run the Langevin simulation
    # ------------------------------------------------------------------
    print("\nRunning Langevin simulation ...")
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
    print(f"  Generated {configs.shape[0]} configurations "
          f"on {configs.shape[1]} lattice sites.")

    # ------------------------------------------------------------------
    # 3a.  Observable: <x^2> with jackknife error
    # ------------------------------------------------------------------
    # Per-configuration <x^2> (averaged over lattice sites within each config)
    x2_per_config = np.mean(configs ** 2, axis=1)

    x2_est, x2_err = jackknife(x2_per_config, func=np.mean)
    x2_exact = 1.0 / (2.0 * omega)

    print("\n--- <x^2> ---")
    print(f"  Measured:  {x2_est:.6f} +/- {x2_err:.6f}")
    print(f"  Exact:     {x2_exact:.6f}")
    print(f"  Deviation: {abs(x2_est - x2_exact) / x2_err:.1f} sigma")

    # ------------------------------------------------------------------
    # 3b.  Correlator and effective mass
    # ------------------------------------------------------------------
    max_tau = n_lattice // 2
    corr = correlator(configs, max_tau=max_tau)
    m_eff = effective_mass(corr, a=a)

    print("\n--- Correlator C(tau) and effective mass m_eff(tau) ---")
    print(f"  {'tau':>5s}  {'C(tau)':>12s}  {'m_eff':>12s}")
    # Show a selection of tau values
    tau_show = [0, 1, 2, 3, 5, 10, 15, 20]
    for t in tau_show:
        if t < len(corr):
            meff_str = f"{m_eff[t]:.6f}" if t < len(m_eff) and np.isfinite(m_eff[t]) else "---"
            print(f"  {t * a:5.1f}  {corr[t]:12.6f}  {meff_str:>12s}")

    # ------------------------------------------------------------------
    # 3c.  Ground state energy from effective mass plateau
    # ------------------------------------------------------------------
    # The effective mass m_eff(tau) -> E_1 - E_0 = omega at large tau.
    # Average over a plateau region (avoid small and large tau).
    plateau_start = 5
    plateau_end = min(20, len(m_eff))
    plateau_vals = m_eff[plateau_start:plateau_end]
    plateau_vals = plateau_vals[np.isfinite(plateau_vals)]

    if len(plateau_vals) > 0:
        delta_E = np.mean(plateau_vals)
        # Ground state energy: E_0 = omega/2 is inferred because
        # E_1 - E_0 = omega (the gap), which confirms quantization.
        E0_exact = omega / 2.0
        print(f"\n--- Ground state energy ---")
        print(f"  E_1 - E_0 (plateau avg, tau/a in [{plateau_start},{plateau_end})): "
              f"{delta_E:.4f}  (exact: {omega:.4f})")
        print(f"  => E_0 = omega/2 = {E0_exact:.4f}")
    else:
        print("\n  [Warning] Could not extract a stable effective mass plateau.")

    # ------------------------------------------------------------------
    # 4.  Plotting (optional -- requires matplotlib)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for headless use
        import matplotlib.pyplot as plt

        output_dir = os.path.dirname(__file__)

        # ---- Effective mass vs tau ----
        tau_vals = a * np.arange(len(m_eff))
        valid = np.isfinite(m_eff)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(tau_vals[valid], m_eff[valid], "o-", markersize=3, label=r"$m_{\rm eff}(\tau)$")
        ax.axhline(omega, color="red", linestyle="--", linewidth=1.5, label=rf"$\omega = {omega}$")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$m_{\rm eff}(\tau)$")
        ax.set_title("Effective mass from Euclidean correlator")
        ax.set_ylim(0, 2.5 * omega)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        eff_mass_path = os.path.join(output_dir, "effective_mass.png")
        fig.savefig(eff_mass_path, dpi=150)
        plt.close(fig)
        print(f"\n  Saved effective mass plot -> {eff_mass_path}")

        # ---- Wave function histogram vs exact |psi_0|^2 ----
        bin_centers, hist_vals = position_histogram(configs, n_bins=60)
        x_fine = np.linspace(bin_centers[0], bin_centers[-1], 300)
        psi2_exact = np.sqrt(omega / np.pi) * np.exp(-omega * x_fine ** 2)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(bin_centers, hist_vals, width=(bin_centers[1] - bin_centers[0]),
               alpha=0.6, label="Simulation histogram")
        ax.plot(x_fine, psi2_exact, "r-", linewidth=2,
                label=r"$|\psi_0(x)|^2$ (exact)")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|\psi_0(x)|^2$")
        ax.set_title("Ground state wave function")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        wf_path = os.path.join(output_dir, "wavefunction.png")
        fig.savefig(wf_path, dpi=150)
        plt.close(fig)
        print(f"  Saved wave function plot  -> {wf_path}")

    except ImportError:
        print("\n  [matplotlib not available -- skipping plots]")
    except Exception as e:
        print(f"\n  [Plotting failed: {e}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
