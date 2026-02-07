"""
Potentials, derivatives, Euclidean action, and Langevin drift force
for stochastic quantization on a discretized Euclidean time lattice.

Conventions
-----------
- N lattice sites with spacing *a* and periodic boundary conditions x[N] = x[0].
- Euclidean action:
      S_E = sum_i [ m/(2a) * (x[i+1] - x[i])^2  +  a * V(x[i]) ]
- Drift force for site j:
      f_j = -dS_E/dx_j = m/a * (x[j+1] + x[j-1] - 2*x[j])  -  a * V'(x[j])
- Natural units with hbar = 1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Potentials V(x)
# ---------------------------------------------------------------------------

def harmonic_potential(x, omega=1.0):
    """V(x) = 0.5 * omega^2 * x^2"""
    return 0.5 * omega**2 * x**2


def anharmonic_potential(x, omega=1.0, lam=1.0):
    """V(x) = 0.5 * omega^2 * x^2 + lam * x^4"""
    return 0.5 * omega**2 * x**2 + lam * x**4


def double_well_potential(x, lam=1.0, v=1.0):
    """V(x) = lam * (x^2 - v^2)^2"""
    return lam * (x**2 - v**2)**2


# ---------------------------------------------------------------------------
# Potential derivatives V'(x)
# ---------------------------------------------------------------------------

def harmonic_force(x, omega=1.0):
    """dV/dx for the harmonic potential."""
    return omega**2 * x


def anharmonic_force(x, omega=1.0, lam=1.0):
    """dV/dx for the anharmonic potential."""
    return omega**2 * x + 4.0 * lam * x**3


def double_well_force(x, lam=1.0, v=1.0):
    """dV/dx for the double-well potential."""
    return 4.0 * lam * x * (x**2 - v**2)


# ---------------------------------------------------------------------------
# Euclidean action
# ---------------------------------------------------------------------------

def euclidean_action(x, potential_func, mass=1.0, a=0.1, **pot_params):
    """Compute the Euclidean action S_E for a path *x*.

    Parameters
    ----------
    x : 1-D numpy array
        Lattice path with N sites; periodic BC x[N] = x[0].
    potential_func : callable
        One of the potential functions defined above.
    mass : float
        Particle mass.
    a : float
        Lattice spacing in Euclidean time.
    **pot_params
        Extra keyword arguments forwarded to *potential_func*.

    Returns
    -------
    float
        The discretized Euclidean action.
    """
    dx = np.roll(x, -1) - x          # x[i+1] - x[i], periodic
    kinetic = mass / (2.0 * a) * np.sum(dx**2)
    potential = a * np.sum(potential_func(x, **pot_params))
    return kinetic + potential


# ---------------------------------------------------------------------------
# Drift force  f_j = -dS_E / dx_j  (full lattice, vectorised)
# ---------------------------------------------------------------------------

def drift_force(x, force_func, mass=1.0, a=0.1, **pot_params):
    """Compute the Langevin drift force for every lattice site.

    Parameters
    ----------
    x : 1-D numpy array
        Lattice path with N sites; periodic BC x[N] = x[0].
    force_func : callable
        Derivative of the potential, dV/dx (e.g. *harmonic_force*).
    mass : float
        Particle mass.
    a : float
        Lattice spacing in Euclidean time.
    **pot_params
        Extra keyword arguments forwarded to *force_func*.

    Returns
    -------
    numpy array (same shape as *x*)
        f_j = (m/a)*(x[j+1] + x[j-1] - 2*x[j])  -  a * V'(x[j])
    """
    laplacian = np.roll(x, -1) + np.roll(x, 1) - 2.0 * x
    return (mass / a) * laplacian - a * force_func(x, **pot_params)
