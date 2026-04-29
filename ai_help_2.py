from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from scipy.linalg import solve


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


# =========================
# Parametrized geometry ABC
# =========================

class ParamCurve(ABC):
    """
    Parametrized curve in the plane: (x(t), y(t)), typically with t in [-1, 1].
    Implementations must be vectorized (accept numpy arrays).
    """

    @abstractmethod
    def x(self, t: FloatArray) -> FloatArray: ...

    @abstractmethod
    def y(self, t: FloatArray) -> FloatArray: ...

    def R(self, t: FloatArray, t0: FloatArray) -> FloatArray:
        """
        Pairwise distance R(t, t0) = sqrt( (x(t)-x(t0))^2 + (y(t)-y(t0))^2 )

        Supports broadcasting: t and t0 can be (n,1) and (1,m).
        """
        xt = self.x(t)
        yt = self.y(t)
        xt0 = self.x(t0)
        yt0 = self.y(t0)
        return np.sqrt((xt - xt0) ** 2 + (yt - yt0) ** 2)


# OK
@dataclass(frozen=True)
class LineSegment(ParamCurve):
    """
    Straight line from P0=(x0,y0) to P1=(x1,y1) with t in [-1,1].

    Mapping: s=(t+1)/2 in [0,1], then P(t)=P0 + s*(P1-P0).
    """
    x0: float
    y0: float
    x1: float
    y1: float

    def x(self, t: FloatArray) -> FloatArray:
        s = (t + 1.0) * 0.5
        return self.x0 + s * (self.x1 - self.x0)

    def y(self, t: FloatArray) -> FloatArray:
        s = (t + 1.0) * 0.5
        return self.y0 + s * (self.y1 - self.y0)


# =========================
# Nodes / quadrature helpers
# =========================

def collocation_nodes(n: int, kind: Literal["chebyshev", "uniform"] = "chebyshev") -> FloatArray:
    """
    Returns t_i in [-1,1].

    - chebyshev: t_i = cos((2i-1)/(2n) * pi), i=1..n  (good for endpoint behavior)
    - uniform: equally spaced (includes endpoints)
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    if kind == "chebyshev":
        i = np.arange(1, n + 1, dtype=np.float64)
        return np.cos((2.0 * i - 1.0) * np.pi / (2.0 * n))
    elif kind == "uniform":
        return np.linspace(-1.0, 1.0, n, dtype=np.float64)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def gauss_chebyshev_nodes_weights(m: int) -> tuple[FloatArray, FloatArray]:
    """
    Gauss–Chebyshev (1st kind) quadrature for:
        ∫_{-1}^1 g(t0) / sqrt(1 - t0^2) dt0  ≈  Σ w_m * g(t0_m)

    nodes: t0_m = cos((2m-1)/(2M) * pi), m=1..M
    weights: all equal to pi/M
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    j = np.arange(1, m + 1, dtype=np.float64)
    nodes = np.cos((2.0 * j - 1.0) * np.pi / (2.0 * m))
    weights = (np.pi / m) * np.ones(m, dtype=np.float64)
    return nodes, weights


# =========================
# Kernels M and K
# =========================

def M_vector(
    curve: ParamCurve,
    t: FloatArray,
    k: float,
    m_quad: int = 256,
    r_eps: float = 1e-12,
) -> ComplexArray:
    """
    M(t) = ∫_{-1}^1 H_0^(1)(k R(t, t0)) dt0 / sqrt(1 - t0^2)

    Uses Gauss–Chebyshev quadrature.
    Returns M evaluated for each t (vector).
    """
    t0, w = gauss_chebyshev_nodes_weights(m_quad)  # (m,), (m,)
    t_col = t.reshape(-1, 1)                        # (n,1)
    t0_row = t0.reshape(1, -1)                      # (1,m)

    R = curve.R(t_col, t0_row)                      # (n,m)
    R = np.maximum(R, r_eps)                        # avoid H0^(1)(0)
    z = k * R

    H0 = hankel1(0, z)                              # (n,m) complex
    # weights broadcast: (1,m)
    return np.sum(H0 * w.reshape(1, -1), axis=1).astype(np.complex128)  # (n,)


def K_matrix(
    curve: ParamCurve,
    t: FloatArray,
    k: float,
    t0: FloatArray,
    principal_value: bool = True,
    dt_eps: float = 1e-14,
    r_eps: float = 1e-12,
) -> ComplexArray:
    """
    K(t, t0) = -H_1^(1)(k R(t, t0)) - 1/(t - t0)

    Returns pairwise matrix K_ij = K(t_i, t0_j) with broadcasting.

    Notes on singularities:
    - 1/(t-t0) is singular on diagonal; if principal_value=True, diagonal is set to 0 for that term.
    - H_1^(1)(kR) is also singular at R=0; we clamp R >= r_eps.
    """
    t_col = t.reshape(-1, 1)     # (n,1)
    t0_row = t0.reshape(1, -1)   # (1,n)

    R = curve.R(t_col, t0_row)                 # (n,n)
    R = np.maximum(R, r_eps)
    H1 = hankel1(1, k * R).astype(np.complex128)

    dt = (t_col - t0_row).astype(np.float64)   # (n,n)

    inv_dt = np.zeros_like(dt, dtype=np.float64)
    if principal_value:
        mask = np.abs(dt) > dt_eps
        inv_dt[mask] = 1.0 / dt[mask]
        # diagonal (and near-diagonal) stays 0
    else:
        # "unsafe" but sometimes useful
        inv_dt = 1.0 / (dt + np.sign(dt) * dt_eps)

    return (-H1 - inv_dt).astype(np.complex128)


# =========================
# Linear system assembly/solve
# =========================

@dataclass(frozen=True)
class IEParams:
    n: int
    k: float
    C: complex
    node_kind: Literal["chebyshev", "uniform"] = "chebyshev"
    m_quad: int = 256

    # singular handling knobs
    principal_value: bool = True
    dt_eps: float = 1e-14
    r_eps: float = 1e-12


def solve_for_v(
    curve: ParamCurve,
    f: Callable[[FloatArray], ComplexArray] | Callable[[FloatArray], FloatArray],
    params: IEParams,
) -> tuple[FloatArray, ComplexArray]:
    """
    Solves the linear system for v_i = v_p^n(t_i).

    Equations (as in your screenshot):
      For j = 1..n-1:
        -(1/n) Σ_i v_i / (t_i - t_j)  + (1/n) Σ_i K(t_i, t_j) v_i  = f(t_j)

      For j = n:
        Σ_i M(t_i) v_i = C

    Returns:
      t (n,), v (n,) complex
    """
    n = params.n
    t = collocation_nodes(n, params.node_kind)  # (n,)

    # Build A and b
    A = np.zeros((n, n), dtype=np.complex128)
    b = np.zeros(n, dtype=np.complex128)

    # Precompute K_ij where i indexes t_i (rows), j indexes t_j (cols) but we need K(t_i, t_j)
    Kij = K_matrix(
        curve=curve,
        t=t, t0=t,
        k=params.k,
        principal_value=params.principal_value,
        dt_eps=params.dt_eps,
        r_eps=params.r_eps,
    )  # (n,n): K(t_i, t_j)

    # Precompute inv(t_i - t_j) for the first term
    t_col = t.reshape(-1, 1)
    t_row = t.reshape(1, -1)
    dt = (t_col - t_row)  # (n,n): (t_i - t_j)

    inv_dt = np.zeros_like(dt, dtype=np.float64)
    if params.principal_value:
        mask = np.abs(dt) > params.dt_eps
        inv_dt[mask] = 1.0 / dt[mask]
    else:
        inv_dt = 1.0 / (dt + np.sign(dt) * params.dt_eps)

    # For each equation j=0..n-2:
    # A[j, i] = -(1/n)*1/(t_i - t_j) + (1/n)*K(t_i, t_j)
    # Note: our dt matrix is (i,j). Kij is (i,j).
    for j in range(n - 1):
        A[j, :] = (-(1.0 / n) * inv_dt[:, j] + (1.0 / n) * Kij[:, j])
        b[j] = np.asarray(f(t[j]), dtype=np.complex128)

    # Last equation j=n:
    Mti = M_vector(curve, t, k=params.k, m_quad=params.m_quad, r_eps=params.r_eps)  # (n,)
    A[n - 1, :] = Mti
    b[n - 1] = np.complex128(params.C)

    # Solve
    v = solve(A, b).astype(np.complex128)
    return t, v


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # Geometry: a simple line segment
    curve = LineSegment(x0=0.0, y0=0.0, x1=1.0, y1=0.0)

    # RHS f(t): example (replace with your actual f)
    def f(t: FloatArray) -> ComplexArray:
        # vectorized: if t is scalar, still works
        t = np.asarray(t, dtype=np.float64)
        return (np.exp(1j * t)).astype(np.complex128)

    params = IEParams(
        n=64,
        k=10.0,
        C=1.0 + 0.0j,
        node_kind="chebyshev",
        m_quad=256,
        principal_value=True,
    )

    t_nodes, v = solve_for_v(curve, f, params)
    print("t:", t_nodes)
    print("v:", v)
