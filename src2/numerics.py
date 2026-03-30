"""Numerical helper routines shared across the reflector solver package."""

from __future__ import annotations

import numpy as np

try:
    from src.utils.math_utils import cheb_first_kind, hankel1
except ImportError:
    from scipy.special import jv, yv

    def hankel1(v: int | np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the Hankel function of the first kind using Bessel J and Y."""

        return jv(v, z) + 1j * yv(v, z)

    def cheb_first_kind(n: int) -> np.ndarray:
        """Return the first-kind Gauss-Chebyshev nodes on ``[-1, 1]``."""

        i = np.arange(1, n + 1)
        return np.cos((2 * i - 1) * np.pi / (2 * n))


def as_float_array(t: float | np.ndarray) -> np.ndarray:
    """Convert a scalar or array-like input to a ``float64`` NumPy array."""

    return np.asarray(t, dtype=np.float64)


def broadcast_constant_like(t: float | np.ndarray, value: float) -> np.ndarray:
    """Broadcast a scalar ``value`` to the shape of ``t`` as ``float64``."""

    arr = as_float_array(t)
    return np.full_like(arr, fill_value=value, dtype=np.float64)


def collocation_nodes_no_overlap(n: int) -> np.ndarray:
    """Return off-grid Chebyshev-like collocation nodes used by the solver.

    Parameters
    ----------
    n:
        Number of nodes per reflector.
    """

    j = np.arange(1, n + 1, dtype=np.float64)
    return np.cos(j * np.pi / (n + 1))
