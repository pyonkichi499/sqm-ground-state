"""Langevin equation solver for stochastic quantization (Parisi-Wu method).

Implements Euler-Maruyama integration of the Langevin equation on a
discretized Euclidean time lattice with periodic boundary conditions:

    x_j(t + eps) = x_j(t) + eps * f_j + sqrt(2 * eps) * eta_j

where f_j = -dS_E/dx_j is the drift force and eta_j ~ N(0, 1).
"""

import numpy as np

from src.action import drift_force


class LangevinSimulation:
    """Langevin dynamics simulation on a Euclidean time lattice.

    Parameters
    ----------
    n_lattice : int
        Number of lattice sites (Euclidean time discretization).
    a : float
        Lattice spacing.
    mass : float
        Particle mass.
    force_func : callable
        Derivative of the potential dV/dx, passed through to drift_force.
    epsilon : float
        Langevin step size (fictitious-time discretization).
    rng_seed : int, optional
        Seed for the random number generator (for reproducibility).
    **pot_params
        Additional keyword arguments forwarded to drift_force
        (e.g. omega, lam).
    """

    def __init__(self, n_lattice, a, mass, force_func, epsilon,
                 rng_seed=None, **pot_params):
        self.n_lattice = n_lattice
        self.a = a
        self.mass = mass
        self.force_func = force_func
        self.epsilon = epsilon
        self.pot_params = pot_params

        self.rng = np.random.default_rng(rng_seed)
        self.x = np.zeros(n_lattice)

    def step(self):
        """Perform one Euler-Maruyama Langevin step on all lattice sites."""
        force = drift_force(self.x, self.force_func,
                            mass=self.mass, a=self.a, **self.pot_params)
        noise = self.rng.standard_normal(self.n_lattice)
        self.x = self.x + self.epsilon * force + np.sqrt(2.0 * self.epsilon) * noise

    def thermalize(self, n_therm):
        """Run *n_therm* Langevin steps to reach thermal equilibrium."""
        for _ in range(n_therm):
            self.step()

    def generate_configurations(self, n_configs, n_skip=10):
        """Collect lattice configurations separated by *n_skip* steps.

        Parameters
        ----------
        n_configs : int
            Number of configurations to collect.
        n_skip : int
            Number of Langevin steps between successive samples
            (to reduce autocorrelation).

        Returns
        -------
        configs : np.ndarray, shape (n_configs, n_lattice)
            Array of sampled path configurations.
        """
        configs = np.empty((n_configs, self.n_lattice))
        for i in range(n_configs):
            for _ in range(n_skip):
                self.step()
            configs[i] = self.x.copy()
        return configs

    def run(self, n_therm, n_configs, n_skip=10):
        """Thermalize and then generate configurations.

        Convenience wrapper that calls :meth:`thermalize` followed by
        :meth:`generate_configurations`.

        Parameters
        ----------
        n_therm : int
            Number of thermalization steps.
        n_configs : int
            Number of configurations to collect.
        n_skip : int
            Steps skipped between measurements.

        Returns
        -------
        configs : np.ndarray, shape (n_configs, n_lattice)
            Sampled path configurations after thermalization.
        """
        self.thermalize(n_therm)
        return self.generate_configurations(n_configs, n_skip)
