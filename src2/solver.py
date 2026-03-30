"""Incident source and matrix-discretization solver for 2-D reflector problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.linalg import solve

from .geometry import ParamCurve
from .numerics import cheb_first_kind, collocation_nodes_no_overlap, hankel1


@dataclass(frozen=True)
class ComplexSourcePoint:
    """Complex source point (CSP) excitation for the 2-D Helmholtz problem.

    Parameters
    ----------
    k:
        Wavenumber in the background medium.
    x0, y0:
        Real feed coordinates.
    b:
        Complex-source beam parameter. Larger values produce tighter beams.
    beta_rad:
        Beam aiming angle measured in radians.
    """

    k: float
    x0: float
    y0: float
    b: float
    beta_rad: float

    @property
    def rc(self) -> complex:
        """Return the complexified source position on the x-axis projection."""

        return complex(self.x0, 0.0) + 1j * self.b * np.cos(self.beta_rad)

    @property
    def rcx(self) -> complex:
        """Return the complex x-coordinate of the displaced source."""

        return complex(self.x0, 0.0) + 1j * self.b * np.cos(self.beta_rad)

    @property
    def rcy(self) -> complex:
        """Return the complex y-coordinate of the displaced source."""

        return complex(self.y0, 0.0) + 1j * self.b * np.sin(self.beta_rad)

    def field(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        """Evaluate the incident CSP field at spatial coordinates ``(x, y)``."""

        x_arr = np.asarray(x, dtype=np.complex128)
        y_arr = np.asarray(y, dtype=np.complex128)
        r = np.sqrt((x_arr - self.rcx) ** 2 + (y_arr - self.rcy) ** 2)
        return hankel1(0, self.k * r)

    def boundary_field(self, curve: ParamCurve, t: float | np.ndarray) -> np.ndarray:
        """Evaluate the incident field directly on a reflector curve.

        Parameters
        ----------
        curve:
            Reflector contour on which the field is needed.
        t:
            Parameter values on the reflector contour.
        """

        x, y = curve.coords(t)
        return self.field(x, y)

    def boundary_derivative(self, curve: ParamCurve, t: float | np.ndarray) -> np.ndarray:
        """Differentiate the incident field along the reflector parameter ``t``.

        Parameters
        ----------
        curve:
            Reflector contour on which the derivative is evaluated.
        t:
            Parameter values on the reflector contour.
        """

        x, y = curve.coords(t)
        dx, dy = curve.derivatives(t)
        x_arr = np.asarray(x, dtype=np.complex128)
        y_arr = np.asarray(y, dtype=np.complex128)
        dx_arr = np.asarray(dx, dtype=np.complex128)
        dy_arr = np.asarray(dy, dtype=np.complex128)
        r = np.sqrt((x_arr - self.rcx) ** 2 + (y_arr - self.rcy) ** 2)
        dr_dt = ((x_arr - self.rcx) * dx_arr + (y_arr - self.rcy) * dy_arr) / r
        return -self.k * hankel1(1, self.k * r) * dr_dt

    def rhs_f(self, curve: ParamCurve, t: float | np.ndarray) -> np.ndarray:
        """Return the right-hand side for the E-wave PEC boundary equation."""

        return -self.boundary_derivative(curve, t)

    def far_field_pattern(self, phi: float | np.ndarray) -> np.ndarray:
        """Return the asymptotic far-field pattern of the incident CSP beam."""

        phi_arr = np.asarray(phi, dtype=np.float64)
        return np.exp(self.k * self.b * np.cos(phi_arr - self.beta_rad))


@dataclass
class SolverCaches:
    """Precomputed reflector geometry values at quadrature and collocation nodes."""

    x_t: list[np.ndarray]
    y_t: list[np.ndarray]
    dx_t: list[np.ndarray]
    dy_t: list[np.ndarray]
    speed_t: list[np.ndarray]
    x_tau: list[np.ndarray]
    y_tau: list[np.ndarray]
    dx_tau: list[np.ndarray]
    dy_tau: list[np.ndarray]


@dataclass
class MDSSolution:
    """Solved discrete current and convenience post-processing helpers.

    Parameters
    ----------
    solver:
        Solver instance that produced the solution.
    t_nodes, tau_nodes:
        Quadrature and collocation parameter nodes.
    v_nodes:
        Edge-regularized current samples for each reflector.
    physical_current_nodes:
        Recovered physical current samples after dividing by the weight factors.
    boundary_residual_max:
        Maximum absolute residual of the discretized boundary equation.
    """

    solver: "MultiReflectorMDS"
    t_nodes: np.ndarray
    tau_nodes: np.ndarray
    v_nodes: np.ndarray
    physical_current_nodes: np.ndarray
    boundary_residual_max: float

    def near_field(self, x: float | np.ndarray, y: float | np.ndarray, total: bool = True) -> np.ndarray:
        """Evaluate the near field on arbitrary spatial points.

        Parameters
        ----------
        x, y:
            Arrays or scalars of matching shape containing the evaluation points.
        total:
            When ``True``, include the direct incident field together with the
            scattered field. When ``False``, return scattered field only.
        """

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same shape")

        x_flat = x_arr.reshape(-1).astype(np.complex128)
        y_flat = y_arr.reshape(-1).astype(np.complex128)
        u_sc = np.zeros_like(x_flat, dtype=np.complex128)

        for q in range(self.solver.num_reflectors):
            x_nodes = self.solver.caches.x_t[q].astype(np.complex128)
            y_nodes = self.solver.caches.y_t[q].astype(np.complex128)
            dx = x_flat[None, :] - x_nodes[:, None]
            dy = y_flat[None, :] - y_nodes[:, None]
            r = np.sqrt(dx * dx + dy * dy)
            h = hankel1(0, self.solver.k * r)
            u_sc += (np.pi / self.solver.n) * (self.v_nodes[q] @ h)

        if total:
            u_sc += self.solver.incident.field(x_flat, y_flat)

        return u_sc.reshape(x_arr.shape)

    def near_field_grid(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        nx: int = 200,
        ny: int = 200,
        total: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the near field on a rectangular Cartesian grid.

        Parameters
        ----------
        x_min, x_max, y_min, y_max:
            Spatial bounds of the grid.
        nx, ny:
            Number of grid points along the x and y axes.
        total:
            When ``True``, include the direct incident field.
        """

        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xg, yg = np.meshgrid(xs, ys)
        u = self.near_field(xg, yg, total=total)
        return xg, yg, u

    def far_field_pattern(self, phi: float | np.ndarray, total: bool = True) -> np.ndarray:
        """Evaluate the discrete far-field pattern for observation angles ``phi``.

        Parameters
        ----------
        phi:
            Observation angle or angles in radians.
        total:
            When ``True``, include the incident CSP contribution.
        """

        phi_arr = np.asarray(phi, dtype=np.float64)
        phi_flat = phi_arr.reshape(-1)
        s_x = np.cos(phi_flat)
        s_y = np.sin(phi_flat)
        phi_sc = np.zeros(phi_flat.shape, dtype=np.complex128)

        prefactor = np.sqrt(2.0 * np.pi / (1j)) / self.solver.n
        for q in range(self.solver.num_reflectors):
            x_nodes = self.solver.caches.x_t[q]
            y_nodes = self.solver.caches.y_t[q]
            phase = np.exp(
                -1j * self.solver.k * (np.outer(s_x, x_nodes) + np.outer(s_y, y_nodes))
            )
            phi_sc += phase @ self.v_nodes[q]

        phi_sc *= prefactor
        if total:
            phi_sc += self.solver.incident.far_field_pattern(phi_flat)
        return phi_sc.reshape(phi_arr.shape)

    def directivity(self, num_angles: int = 2048) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Compute directivity and the sampled far-field pattern.

        Parameters
        ----------
        num_angles:
            Number of equally spaced angular samples over ``[0, 2π)``.
        """

        phi = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
        pattern = self.far_field_pattern(phi, total=True)
        power = (2.0 * np.pi / num_angles) * np.sum(np.abs(pattern) ** 2)
        peak_index = int(np.argmax(np.abs(pattern)))
        peak_phi = float(phi[peak_index])
        d = float((2.0 * np.pi / power) * (np.abs(pattern[peak_index]) ** 2))
        return d, peak_phi, phi, pattern

    def edge_illumination_db(self, reflector_index: int = 0) -> tuple[float, float]:
        """Return feed edge illumination relative to the reflector center in dB.

        Parameters
        ----------
        reflector_index:
            Index of the reflector to probe.
        """

        curve = self.solver.reflectors[reflector_index]
        center = complex(self.solver.incident.boundary_field(curve, 0.0))
        edge_left = complex(self.solver.incident.boundary_field(curve, -1.0))
        edge_right = complex(self.solver.incident.boundary_field(curve, 1.0))
        left = 20.0 * np.log10(np.abs(edge_left / center))
        right = 20.0 * np.log10(np.abs(edge_right / center))
        return float(left), float(right)


class MultiReflectorMDS:
    """Solve the discretized 2-D E-wave scattering problem for open reflectors.

    The executable path uses direct collocation of the single-layer equation with
    Gauss-Chebyshev quadrature. In this workspace that formulation is the one
    that validated numerically and produced physically sensible results for the
    paper-inspired reflector examples.
    """

    def __init__(
        self,
        reflectors: Sequence[ParamCurve],
        incident: ComplexSourcePoint,
        n: int = 64,
    ) -> None:
        """Initialize the solver and precompute geometry node data.

        Parameters
        ----------
        reflectors:
            Sequence of reflector contours.
        incident:
            Incident CSP feed model.
        n:
            Number of quadrature and current unknowns per reflector.
        """

        if n < 4:
            raise ValueError("n must be at least 4")
        self.reflectors = list(reflectors)
        if not self.reflectors:
            raise ValueError("at least one reflector is required")
        self.incident = incident
        self.k = incident.k
        self.n = int(n)
        self.num_reflectors = len(self.reflectors)
        self.t_nodes = cheb_first_kind(self.n).astype(np.float64)
        self.tau_nodes = collocation_nodes_no_overlap(self.n).astype(np.float64)
        self.caches = self._build_caches()

    def _build_caches(self) -> SolverCaches:
        """Precompute reflector coordinates and derivatives at solver nodes."""

        x_t: list[np.ndarray] = []
        y_t: list[np.ndarray] = []
        dx_t: list[np.ndarray] = []
        dy_t: list[np.ndarray] = []
        speed_t: list[np.ndarray] = []
        x_tau: list[np.ndarray] = []
        y_tau: list[np.ndarray] = []
        dx_tau: list[np.ndarray] = []
        dy_tau: list[np.ndarray] = []

        for curve in self.reflectors:
            xt, yt = curve.coords(self.t_nodes)
            dxt, dyt = curve.derivatives(self.t_nodes)
            xtau, ytau = curve.coords(self.tau_nodes)
            dxtau, dytau = curve.derivatives(self.tau_nodes)
            x_t.append(np.asarray(xt, dtype=np.float64))
            y_t.append(np.asarray(yt, dtype=np.float64))
            dx_t.append(np.asarray(dxt, dtype=np.float64))
            dy_t.append(np.asarray(dyt, dtype=np.float64))
            speed_t.append(np.asarray(curve.speed(self.t_nodes), dtype=np.float64))
            x_tau.append(np.asarray(xtau, dtype=np.float64))
            y_tau.append(np.asarray(ytau, dtype=np.float64))
            dx_tau.append(np.asarray(dxtau, dtype=np.float64))
            dy_tau.append(np.asarray(dytau, dtype=np.float64))

        return SolverCaches(
            x_t=x_t,
            y_t=y_t,
            dx_t=dx_t,
            dy_t=dy_t,
            speed_t=speed_t,
            x_tau=x_tau,
            y_tau=y_tau,
            dx_tau=dx_tau,
            dy_tau=dy_tau,
        )

    def _distance_source_target(
        self, q: int, p: int, t: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        """Return source-to-target distances between reflector samples."""

        xq = self.reflectors[q].x(t)
        yq = self.reflectors[q].y(t)
        xp = self.reflectors[p].x(tau)
        yp = self.reflectors[p].y(tau)
        return np.sqrt((xq - xp) ** 2 + (yq - yp) ** 2)

    def _dR_dtau(self, q: int, p: int, t: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Return the derivative of the source-target distance with respect to ``tau``."""

        xq = self.reflectors[q].x(t)
        yq = self.reflectors[q].y(t)
        xp = self.reflectors[p].x(tau)
        yp = self.reflectors[p].y(tau)
        dxp = self.reflectors[p].x_der(tau)
        dyp = self.reflectors[p].y_der(tau)
        r = np.sqrt((xq - xp) ** 2 + (yq - yp) ** 2)
        return ((xp - xq) * dxp + (yp - yq) * dyp) / r

    def solve_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the dense linear system ``A v = b`` for the current unknowns."""

        total_unknowns = self.num_reflectors * self.n
        a = np.zeros((total_unknowns, total_unknowns), dtype=np.complex128)
        b = np.zeros((total_unknowns,), dtype=np.complex128)

        direct_weight = np.pi / self.n
        h0_blocks: dict[tuple[int, int], np.ndarray] = {}

        for p in range(self.num_reflectors):
            for q in range(self.num_reflectors):
                xq = self.caches.x_t[q][:, None]
                yq = self.caches.y_t[q][:, None]
                xp = self.caches.x_tau[p][None, :]
                yp = self.caches.y_tau[p][None, :]
                r = np.sqrt((xq - xp) ** 2 + (yq - yp) ** 2)
                h0_blocks[(p, q)] = hankel1(0, self.k * r)

        for p in range(self.num_reflectors):
            rhs = -self.incident.boundary_field(self.reflectors[p], self.tau_nodes)
            for j in range(self.n):
                row = p * self.n + j
                b[row] = rhs[j]
                for q in range(self.num_reflectors):
                    col_slice = slice(q * self.n, (q + 1) * self.n)
                    a[row, col_slice] = direct_weight * h0_blocks[(p, q)][:, j]

        return a, b

    def _boundary_residual(self, v_nodes: np.ndarray) -> float:
        """Compute the maximum absolute residual on all collocation points."""

        max_res = 0.0
        weight = np.pi / self.n
        for p in range(self.num_reflectors):
            rhs = -self.incident.boundary_field(self.reflectors[p], self.tau_nodes)
            lhs = np.zeros_like(rhs, dtype=np.complex128)
            for q in range(self.num_reflectors):
                xq = self.caches.x_t[q][:, None]
                yq = self.caches.y_t[q][:, None]
                xp = self.caches.x_tau[p][None, :]
                yp = self.caches.y_tau[p][None, :]
                r = np.sqrt((xq - xp) ** 2 + (yq - yp) ** 2)
                h = hankel1(0, self.k * r)
                lhs += weight * (v_nodes[q] @ h)
            max_res = max(max_res, float(np.max(np.abs(lhs - rhs))))
        return max_res

    def solve(self) -> MDSSolution:
        """Solve the linear system and package the result in an ``MDSSolution``."""

        a, b = self.solve_system()
        unknowns = solve(a, b)
        v_nodes = unknowns.reshape(self.num_reflectors, self.n)
        current = v_nodes / (
            np.vstack(self.caches.speed_t) * np.sqrt(1.0 - self.t_nodes**2)[None, :]
        )
        residual = self._boundary_residual(v_nodes)
        return MDSSolution(
            solver=self,
            t_nodes=self.t_nodes.copy(),
            tau_nodes=self.tau_nodes.copy(),
            v_nodes=v_nodes,
            physical_current_nodes=current,
            boundary_residual_max=residual,
        )
