"""Incident source and matrix-discretization solver for 2-D reflector problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.linalg import solve
from numpy.polynomial.legendre import leggauss

from .geometry import ParamCurve
from .numerics import cheb_first_kind, collocation_nodes_no_overlap, hankel1


EULER_GAMMA = 0.5772156649015329


class IncidentField(Protocol):
    """Protocol implemented by incident-field models used by the solver."""

    k: float

    def field(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        """Evaluate the incident field at spatial coordinates ``(x, y)``."""

    def boundary_field(self, curve: ParamCurve, t: float | np.ndarray) -> np.ndarray:
        """Evaluate the incident field directly on a reflector curve."""

    def boundary_derivative(self, curve: ParamCurve, t: float | np.ndarray) -> np.ndarray:
        """Differentiate the incident field along the reflector parameter ``t``."""

    def rhs_f(self, curve: ParamCurve, t: float | np.ndarray) -> np.ndarray:
        """Return the boundary-equation right-hand side induced by the source."""

    def far_field_pattern(self, phi: float | np.ndarray) -> np.ndarray:
        """Return the asymptotic far-field pattern of the incident field."""


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


@dataclass(frozen=True)
class PlaneWave:
    """Plane-wave excitation with phase ``exp(-i k (x cos beta + y sin beta))``."""

    k: float
    beta_rad: float

    @property
    def direction_x(self) -> float:
        """Return the x-component of the propagation direction."""

        return float(np.cos(self.beta_rad))

    @property
    def direction_y(self) -> float:
        """Return the y-component of the propagation direction."""

        return float(np.sin(self.beta_rad))

    def field(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        """Evaluate the plane wave at spatial coordinates ``(x, y)``."""

        x_arr = np.asarray(x, dtype=np.complex128)
        y_arr = np.asarray(y, dtype=np.complex128)
        phase = self.k * (self.direction_x * x_arr + self.direction_y * y_arr)
        return np.exp(-1j * phase)

    def boundary_field(self, curve: ParamCurve, t: float | np.ndarray) -> np.ndarray:
        """Evaluate the plane wave directly on a reflector curve."""

        x, y = curve.coords(t)
        return self.field(x, y)

    def boundary_derivative(self, curve: ParamCurve, t: float | np.ndarray) -> np.ndarray:
        """Differentiate the plane wave along the reflector parameter ``t``."""

        x, y = curve.coords(t)
        dx, dy = curve.derivatives(t)
        field = self.field(x, y)
        dx_arr = np.asarray(dx, dtype=np.complex128)
        dy_arr = np.asarray(dy, dtype=np.complex128)
        direction_dot_tangent = self.direction_x * dx_arr + self.direction_y * dy_arr
        return -1j * self.k * field * direction_dot_tangent

    def rhs_f(self, curve: ParamCurve, t: float | np.ndarray) -> np.ndarray:
        """Return the right-hand side for derivative-based formulations."""

        return -self.boundary_derivative(curve, t)

    def far_field_pattern(self, phi: float | np.ndarray) -> np.ndarray:
        """Return zeros because a total plane wave has no finite 2-D angular pattern."""

        phi_arr = np.asarray(phi, dtype=np.float64)
        return np.zeros_like(phi_arr, dtype=np.complex128)


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
            u_sc += self.solver.near_field_weight() * (self.v_nodes[q] @ h)

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
            When ``True``, include the incident contribution when it has a finite
            angular far-field representation.
        """

        phi_arr = np.asarray(phi, dtype=np.float64)
        phi_flat = phi_arr.reshape(-1)
        s_x = np.cos(phi_flat)
        s_y = np.sin(phi_flat)
        phi_sc = np.zeros(phi_flat.shape, dtype=np.complex128)

        prefactor = self.solver.far_field_prefactor()
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

    def directivity(
        self,
        num_angles: int = 2048,
        total: bool = True,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Compute directivity and the sampled far-field pattern.

        Parameters
        ----------
        num_angles:
            Number of equally spaced angular samples over ``[0, 2π)``.
        """

        phi = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
        pattern = self.far_field_pattern(phi, total=total)
        max_mag = float(np.max(np.abs(pattern)))
        if max_mag == 0.0:
            return 0.0, 0.0, phi, pattern
        power = (2.0 * np.pi / num_angles) * np.sum(np.abs(pattern) ** 2)
        if power == 0.0:
            return 0.0, 0.0, phi, pattern
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


@dataclass
class MoMSolution:
    """Solved pulse-basis MoM current with shared near/far-field helpers."""

    solver: "MultiReflectorMoM"
    t_nodes: np.ndarray
    tau_nodes: np.ndarray
    v_nodes: np.ndarray
    physical_current_nodes: np.ndarray
    boundary_residual_max: float

    def near_field(self, x: float | np.ndarray, y: float | np.ndarray, total: bool = True) -> np.ndarray:
        """Evaluate the near field using midpoint pulse weights."""

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same shape")

        x_flat = x_arr.reshape(-1).astype(np.complex128)
        y_flat = y_arr.reshape(-1).astype(np.complex128)
        u_sc = np.zeros_like(x_flat, dtype=np.complex128)

        for q in range(self.solver.num_reflectors):
            x_nodes = self.solver.center_x[q].astype(np.complex128)
            y_nodes = self.solver.center_y[q].astype(np.complex128)
            amplitudes = self.v_nodes[q] * self.solver.center_source_weights
            dx = x_flat[None, :] - x_nodes[:, None]
            dy = y_flat[None, :] - y_nodes[:, None]
            r = np.sqrt(dx * dx + dy * dy)
            h = hankel1(0, self.solver.k * r)
            u_sc += amplitudes @ h

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
        """Sample the near field on a Cartesian grid."""

        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xg, yg = np.meshgrid(xs, ys)
        u = self.near_field(xg, yg, total=total)
        return xg, yg, u

    def far_field_pattern(self, phi: float | np.ndarray, total: bool = True) -> np.ndarray:
        """Evaluate the discrete far-field pattern for observation angles ``phi``."""

        phi_arr = np.asarray(phi, dtype=np.float64)
        phi_flat = phi_arr.reshape(-1)
        s_x = np.cos(phi_flat)
        s_y = np.sin(phi_flat)
        phi_sc = np.zeros(phi_flat.shape, dtype=np.complex128)

        prefactor = np.sqrt(2.0 / (np.pi * 1j))
        for q in range(self.solver.num_reflectors):
            x_nodes = self.solver.center_x[q]
            y_nodes = self.solver.center_y[q]
            amplitudes = self.v_nodes[q] * self.solver.center_source_weights
            phase = np.exp(
                -1j * self.solver.k * (np.outer(s_x, x_nodes) + np.outer(s_y, y_nodes))
            )
            phi_sc += phase @ amplitudes

        phi_sc *= prefactor
        if total:
            phi_sc += self.solver.incident.far_field_pattern(phi_flat)
        return phi_sc.reshape(phi_arr.shape)

    def directivity(
        self,
        num_angles: int = 2048,
        total: bool = True,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Compute directivity and the sampled far-field pattern."""

        phi = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
        pattern = self.far_field_pattern(phi, total=total)
        max_mag = float(np.max(np.abs(pattern)))
        if max_mag == 0.0:
            return 0.0, 0.0, phi, pattern
        power = (2.0 * np.pi / num_angles) * np.sum(np.abs(pattern) ** 2)
        if power == 0.0:
            return 0.0, 0.0, phi, pattern
        peak_index = int(np.argmax(np.abs(pattern)))
        peak_phi = float(phi[peak_index])
        d = float((2.0 * np.pi / power) * (np.abs(pattern[peak_index]) ** 2))
        return d, peak_phi, phi, pattern

    def edge_illumination_db(self, reflector_index: int = 0) -> tuple[float, float]:
        """Return feed edge illumination relative to the reflector center in dB."""

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
        incident: IncidentField,
        n: int = 64,
    ) -> None:
        """Initialize the solver and precompute geometry node data.

        Parameters
        ----------
        reflectors:
            Sequence of reflector contours.
        incident:
            Incident-field model.
        n:
            Number of quadrature and current unknowns per reflector.
        """

        if n < 4:
            raise ValueError("n must be at least 4")
        self.reflectors = list(reflectors)
        self.incident = incident
        self.k = incident.k
        self.n = int(n)
        self.num_reflectors = len(self.reflectors)
        self.t_nodes = self._build_t_nodes()
        self.tau_nodes = self._build_tau_nodes()
        self.caches = self._build_caches()

    def _build_t_nodes(self) -> np.ndarray:
        """Return the quadrature nodes used by the solver."""

        return cheb_first_kind(self.n).astype(np.float64)

    def _build_tau_nodes(self) -> np.ndarray:
        """Return the collocation nodes used by the solver."""

        return collocation_nodes_no_overlap(self.n).astype(np.float64)

    def near_field_weight(self) -> float:
        """Return the discrete near-field quadrature weight."""

        return float(np.pi / self.n)

    def far_field_prefactor(self) -> complex:
        """Return the discrete far-field prefactor."""

        return complex(np.sqrt(2.0 * np.pi / (1j)) / self.n)

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

        if self.num_reflectors == 0:
            empty = np.empty((0, self.n), dtype=np.complex128)
            return MDSSolution(
                solver=self,
                t_nodes=self.t_nodes.copy(),
                tau_nodes=self.tau_nodes.copy(),
                v_nodes=empty.copy(),
                physical_current_nodes=empty,
                boundary_residual_max=0.0,
            )

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


class MultiReflectorMoM:
    """Pulse-basis method of moments for the weighted first-kind reflector SIE."""

    def __init__(
        self,
        reflectors: Sequence[ParamCurve],
        incident: IncidentField,
        n: int = 64,
        panel_quad_order: int = 12,
        self_panel_quad_order: int = 20,
    ) -> None:
        if n < 4:
            raise ValueError("n must be at least 4")
        if panel_quad_order < 2:
            raise ValueError("panel_quad_order must be at least 2")
        if self_panel_quad_order < 4:
            raise ValueError("self_panel_quad_order must be at least 4")

        self.reflectors = list(reflectors)
        self.incident = incident
        self.k = incident.k
        self.n = int(n)
        self.num_reflectors = len(self.reflectors)
        self.panel_quad_order = int(panel_quad_order)
        self.self_panel_quad_order = int(self_panel_quad_order)

        self.theta_edges = np.linspace(0.0, np.pi, self.n + 1, dtype=np.float64)
        self.theta_centers = 0.5 * (self.theta_edges[:-1] + self.theta_edges[1:])
        self.panel_widths = np.diff(self.theta_edges)
        self.t_nodes = np.cos(self.theta_centers)
        self.tau_nodes = self.t_nodes.copy()
        self.center_source_weights = self.panel_widths.copy()

        self._regular_panel_nodes, self._regular_panel_weights = self._build_regular_panel_quadrature()
        self._singular_panel_nodes, self._singular_panel_weights = self._build_singular_panel_quadrature()
        self.center_x, self.center_y, self.center_speed = self._build_center_geometry()
        self._regular_source_x, self._regular_source_y = self._build_panel_source_geometry(
            self._regular_panel_nodes
        )
        self._singular_source_x, self._singular_source_y = self._build_panel_source_geometry(
            self._singular_panel_nodes
        )

    def _build_regular_panel_quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-panel Gauss-Legendre nodes in ``t = cos(theta)`` and ``dtheta`` weights."""

        xi, wi = leggauss(self.panel_quad_order)
        nodes = np.empty((self.n, self.panel_quad_order), dtype=np.float64)
        weights = np.empty_like(nodes)
        for panel_index in range(self.n):
            left = self.theta_edges[panel_index]
            right = self.theta_edges[panel_index + 1]
            theta_nodes = 0.5 * (right - left) * xi + 0.5 * (right + left)
            mapped_weights = 0.5 * (right - left) * wi
            nodes[panel_index] = np.cos(theta_nodes)
            weights[panel_index] = mapped_weights
        return nodes, weights

    def _build_singular_panel_quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-panel split quadrature in ``theta`` that avoids the self point."""

        xi, wi = leggauss(self.self_panel_quad_order)
        s = 0.5 * (xi + 1.0)
        s_weights = 0.5 * wi
        nodes = np.empty((self.n, 2 * self.self_panel_quad_order), dtype=np.float64)
        weights = np.empty_like(nodes)

        for panel_index in range(self.n):
            left = self.theta_edges[panel_index]
            center = self.theta_centers[panel_index]
            right = self.theta_edges[panel_index + 1]

            left_theta = center - (center - left) * s**2
            left_weights = 2.0 * (center - left) * s_weights * s
            right_theta = center + (right - center) * s**2
            right_weights = 2.0 * (right - center) * s_weights * s

            nodes[panel_index, : self.self_panel_quad_order] = np.cos(left_theta)
            nodes[panel_index, self.self_panel_quad_order :] = np.cos(right_theta)
            weights[panel_index, : self.self_panel_quad_order] = left_weights
            weights[panel_index, self.self_panel_quad_order :] = right_weights

        return nodes, weights

    def _build_center_geometry(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Return reflector geometry sampled at the pulse centers."""

        center_x: list[np.ndarray] = []
        center_y: list[np.ndarray] = []
        center_speed: list[np.ndarray] = []
        for curve in self.reflectors:
            x_nodes, y_nodes = curve.coords(self.t_nodes)
            center_x.append(np.asarray(x_nodes, dtype=np.float64))
            center_y.append(np.asarray(y_nodes, dtype=np.float64))
            center_speed.append(np.asarray(curve.speed(self.t_nodes), dtype=np.float64))
        return center_x, center_y, center_speed

    def _build_panel_source_geometry(
        self,
        panel_nodes: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return reflector coordinates at the quadrature nodes of every panel."""

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for curve in self.reflectors:
            x_panels = np.empty_like(panel_nodes)
            y_panels = np.empty_like(panel_nodes)
            for panel_index in range(self.n):
                x_panel, y_panel = curve.coords(panel_nodes[panel_index])
                x_panels[panel_index] = np.asarray(x_panel, dtype=np.float64)
                y_panels[panel_index] = np.asarray(y_panel, dtype=np.float64)
            xs.append(x_panels)
            ys.append(y_panels)
        return xs, ys

    def _panel_kernel_integral(
        self,
        target_reflector: int,
        source_reflector: int,
        target_index: int,
        source_panel_index: int,
    ) -> complex:
        """Return one pulse-basis matrix entry."""

        if target_reflector == source_reflector and target_index == source_panel_index:
            x_source = self._singular_source_x[source_reflector][source_panel_index]
            y_source = self._singular_source_y[source_reflector][source_panel_index]
            weights = self._singular_panel_weights[source_panel_index]
        else:
            x_source = self._regular_source_x[source_reflector][source_panel_index]
            y_source = self._regular_source_y[source_reflector][source_panel_index]
            weights = self._regular_panel_weights[source_panel_index]

        x_target = self.center_x[target_reflector][target_index]
        y_target = self.center_y[target_reflector][target_index]
        r = np.sqrt((x_source - x_target) ** 2 + (y_source - y_target) ** 2)
        return complex(np.sum(weights * hankel1(0, self.k * r)))

    def solve_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the dense MoM point-matching system ``A v = b``."""

        total_unknowns = self.num_reflectors * self.n
        a = np.zeros((total_unknowns, total_unknowns), dtype=np.complex128)
        b = np.zeros((total_unknowns,), dtype=np.complex128)

        for target_reflector in range(self.num_reflectors):
            rhs = -self.incident.boundary_field(self.reflectors[target_reflector], self.tau_nodes)
            for target_index in range(self.n):
                row = target_reflector * self.n + target_index
                b[row] = rhs[target_index]
                for source_reflector in range(self.num_reflectors):
                    for source_panel_index in range(self.n):
                        col = source_reflector * self.n + source_panel_index
                        a[row, col] = self._panel_kernel_integral(
                            target_reflector=target_reflector,
                            source_reflector=source_reflector,
                            target_index=target_index,
                            source_panel_index=source_panel_index,
                        )

        return a, b

    def solve(self) -> MoMSolution:
        """Solve the MoM system and return pulse coefficients and postprocessing helpers."""

        if self.num_reflectors == 0:
            empty = np.empty((0, self.n), dtype=np.complex128)
            return MoMSolution(
                solver=self,
                t_nodes=self.t_nodes.copy(),
                tau_nodes=self.tau_nodes.copy(),
                v_nodes=empty.copy(),
                physical_current_nodes=empty,
                boundary_residual_max=0.0,
            )

        a, b = self.solve_system()
        unknowns = solve(a, b)
        residual = float(np.max(np.abs(a @ unknowns - b))) if unknowns.size else 0.0
        v_nodes = unknowns.reshape(self.num_reflectors, self.n)
        current = v_nodes / (
            np.vstack(self.center_speed) * np.sqrt(1.0 - self.t_nodes**2)[None, :]
        )
        return MoMSolution(
            solver=self,
            t_nodes=self.t_nodes.copy(),
            tau_nodes=self.tau_nodes.copy(),
            v_nodes=v_nodes,
            physical_current_nodes=current,
            boundary_residual_max=residual,
        )


class MultiReflectorMAR(MultiReflectorMDS):
    """Solve the analytically regularized Cauchy-type SIE system.

    This backend follows the regularized formulation documented in the local
    derivation notes: the original weighted first-kind equation is
    differentiated, its self-singularity is split into an explicit Cauchy term,
    and one supplementary equation per reflector restores the lost integration
    constant.
    """

    cauchy_singularity_coeff = -2j / np.pi
    log_singularity_coeff = 2j / np.pi

    def _mar_boundary_nodes(self) -> np.ndarray:
        """Return the second-kind Chebyshev zeros used in the MAR rows."""

        j = np.arange(1, self.n, dtype=np.float64)
        return np.cos(j * np.pi / self.n)

    def _regularized_self_m(self, reflector_index: int) -> np.ndarray:
        """Return the analytically regularized self ``M_pp(t_i)`` samples."""

        x_nodes = self.caches.x_t[reflector_index]
        y_nodes = self.caches.y_t[reflector_index]
        dx = x_nodes[:, None] - x_nodes[None, :]
        dy = y_nodes[:, None] - y_nodes[None, :]
        r = np.sqrt(dx * dx + dy * dy)
        diff = self.t_nodes[:, None] - self.t_nodes[None, :]

        smooth = np.empty((self.n, self.n), dtype=np.complex128)
        off_diagonal = ~np.eye(self.n, dtype=bool)
        smooth[off_diagonal] = (
            hankel1(0, self.k * r[off_diagonal])
            - self.log_singularity_coeff * np.log(np.abs(diff[off_diagonal]))
        )

        diagonal_limit = 1.0 + self.log_singularity_coeff * (
            np.log(0.5 * self.k * self.caches.speed_t[reflector_index]) + EULER_GAMMA
        )
        smooth[np.diag_indices(self.n)] = diagonal_limit
        return (np.pi / self.n) * np.sum(smooth, axis=1) - 2j * np.log(2.0)

    def _m_samples(self, target_reflector: int, source_reflector: int) -> np.ndarray:
        """Return ``M_pq(t_i)`` sampled at the source reflector quadrature nodes."""

        if target_reflector == source_reflector:
            return self._regularized_self_m(target_reflector)

        x_source = self.caches.x_t[source_reflector][:, None]
        y_source = self.caches.y_t[source_reflector][:, None]
        x_target = self.caches.x_t[target_reflector][None, :]
        y_target = self.caches.y_t[target_reflector][None, :]
        r = np.sqrt((x_source - x_target) ** 2 + (y_source - y_target) ** 2)
        return (np.pi / self.n) * np.sum(hankel1(0, self.k * r), axis=1)

    def _k_block(
        self,
        target_reflector: int,
        source_reflector: int,
        boundary_nodes: np.ndarray,
    ) -> np.ndarray:
        """Return the regularized derivative-kernel block ``K_pq(t_i, tau_j)``."""

        x_source = self.caches.x_t[source_reflector][:, None]
        y_source = self.caches.y_t[source_reflector][:, None]
        x_target = self.reflectors[target_reflector].x(boundary_nodes)[None, :]
        y_target = self.reflectors[target_reflector].y(boundary_nodes)[None, :]
        dx_target = self.reflectors[target_reflector].x_der(boundary_nodes)[None, :]
        dy_target = self.reflectors[target_reflector].y_der(boundary_nodes)[None, :]

        dx = x_source - x_target
        dy = y_source - y_target
        r = np.sqrt(dx * dx + dy * dy)
        d_r_dtau = -((dx * dx_target) + (dy * dy_target)) / r
        derivative_kernel = -self.k * hankel1(1, self.k * r) * d_r_dtau

        if target_reflector != source_reflector:
            return derivative_kernel

        diff = self.t_nodes[:, None] - boundary_nodes[None, :]
        return derivative_kernel - self.cauchy_singularity_coeff / diff

    def solve_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the MAR linear system ``A v = b`` for the current unknowns."""

        total_unknowns = self.num_reflectors * self.n
        a = np.zeros((total_unknowns, total_unknowns), dtype=np.complex128)
        b = np.zeros((total_unknowns,), dtype=np.complex128)

        direct_weight = np.pi / self.n
        boundary_nodes = self._mar_boundary_nodes()
        k_blocks: dict[tuple[int, int], np.ndarray] = {}
        m_samples: dict[tuple[int, int], np.ndarray] = {}

        for target_reflector in range(self.num_reflectors):
            for source_reflector in range(self.num_reflectors):
                k_blocks[(target_reflector, source_reflector)] = self._k_block(
                    target_reflector=target_reflector,
                    source_reflector=source_reflector,
                    boundary_nodes=boundary_nodes,
                )
                m_samples[(target_reflector, source_reflector)] = self._m_samples(
                    target_reflector=target_reflector,
                    source_reflector=source_reflector,
                )

        for target_reflector in range(self.num_reflectors):
            curve = self.reflectors[target_reflector]
            rhs_boundary = -self.incident.boundary_derivative(curve, boundary_nodes)
            rhs_constant = -direct_weight * np.sum(
                self.incident.boundary_field(curve, self.t_nodes)
            )

            for boundary_index in range(self.n - 1):
                row = target_reflector * self.n + boundary_index
                b[row] = rhs_boundary[boundary_index]
                for source_reflector in range(self.num_reflectors):
                    col_slice = slice(
                        source_reflector * self.n,
                        (source_reflector + 1) * self.n,
                    )
                    block = direct_weight * k_blocks[(target_reflector, source_reflector)][
                        :, boundary_index
                    ]
                    if target_reflector == source_reflector:
                        diff = self.t_nodes - boundary_nodes[boundary_index]
                        block = block + direct_weight * (
                            self.cauchy_singularity_coeff / diff
                        )
                    a[row, col_slice] = block

            supplementary_row = target_reflector * self.n + (self.n - 1)
            b[supplementary_row] = rhs_constant
            for source_reflector in range(self.num_reflectors):
                col_slice = slice(
                    source_reflector * self.n,
                    (source_reflector + 1) * self.n,
                )
                a[supplementary_row, col_slice] = (
                    direct_weight * m_samples[(target_reflector, source_reflector)]
                )

        return a, b


class MultiReflectorPaperMDS(MultiReflectorMDS):
    """Direct implementation of the paper's Eq. (11) MDS linear system.

    This backend follows the derivation notes literally:

    - quadrature nodes are the first-kind Chebyshev roots ``t_i``
    - boundary rows use the second-kind zeros ``tau_j = cos(j*pi/n)``
    - the corrected Kronecker-delta Cauchy term is applied only for ``p = q``
    - supplementary rows use high-order weighted quadrature for ``M_pq`` and ``c_p``
    """

    log_singularity_coeff = 2j / np.pi

    def __init__(
        self,
        reflectors: Sequence[ParamCurve],
        incident: IncidentField,
        n: int = 64,
        aux_quad_order: int | None = None,
    ) -> None:
        self.aux_quad_order = int(aux_quad_order) if aux_quad_order is not None else None
        super().__init__(reflectors=reflectors, incident=incident, n=n)
        if self.aux_quad_order is None:
            self.aux_quad_order = max(4 * self.n, 128)
        if self.aux_quad_order < 4:
            raise ValueError("aux_quad_order must be at least 4")
        self.aux_nodes = cheb_first_kind(self.aux_quad_order).astype(np.float64)
        self.aux_weight = float(np.pi / self.aux_quad_order)
        self.aux_x: list[np.ndarray] = []
        self.aux_y: list[np.ndarray] = []
        self.aux_dx: list[np.ndarray] = []
        self.aux_dy: list[np.ndarray] = []
        self.aux_speed: list[np.ndarray] = []
        for curve in self.reflectors:
            x_aux, y_aux = curve.coords(self.aux_nodes)
            dx_aux, dy_aux = curve.derivatives(self.aux_nodes)
            self.aux_x.append(np.asarray(x_aux, dtype=np.float64))
            self.aux_y.append(np.asarray(y_aux, dtype=np.float64))
            self.aux_dx.append(np.asarray(dx_aux, dtype=np.float64))
            self.aux_dy.append(np.asarray(dy_aux, dtype=np.float64))
            self.aux_speed.append(np.asarray(curve.speed(self.aux_nodes), dtype=np.float64))

    def _build_tau_nodes(self) -> np.ndarray:
        """Return the paper's second-kind Chebyshev boundary nodes."""

        j = np.arange(1, self.n, dtype=np.float64)
        return np.cos(j * np.pi / self.n)

    def near_field_weight(self) -> float:
        """Return the paper-normalized near-field quadrature weight."""

        return float(1.0 / self.n)

    def far_field_prefactor(self) -> complex:
        """Return the paper-normalized far-field prefactor."""

        return complex(np.sqrt(2.0 / (np.pi * 1j)) / self.n)

    def _paper_k_block(
        self,
        target_reflector: int,
        source_reflector: int,
    ) -> np.ndarray:
        """Return the paper kernel samples ``K_pq(t_i, tau_j)``."""

        x_source = self.caches.x_t[source_reflector][:, None]
        y_source = self.caches.y_t[source_reflector][:, None]
        x_target = self.caches.x_tau[target_reflector][None, :]
        y_target = self.caches.y_tau[target_reflector][None, :]
        dx_target = self.caches.dx_tau[target_reflector][None, :]
        dy_target = self.caches.dy_tau[target_reflector][None, :]

        dx = x_source - x_target
        dy = y_source - y_target
        r = np.sqrt(dx * dx + dy * dy)
        d_r_dtau = -((dx * dx_target) + (dy * dy_target)) / r
        derivative_kernel = -self.k * hankel1(1, self.k * r) * d_r_dtau

        if target_reflector != source_reflector:
            return derivative_kernel

        diff = self.t_nodes[:, None] - self.tau_nodes[None, :]
        return derivative_kernel - 1.0 / diff

    def _paper_m_samples(self, target_reflector: int, source_reflector: int) -> np.ndarray:
        """Return the supplementary-kernel samples ``M_pq(t_i)``."""

        x_source = self.caches.x_t[source_reflector][:, None]
        y_source = self.caches.y_t[source_reflector][:, None]
        x_target = self.aux_x[target_reflector][None, :]
        y_target = self.aux_y[target_reflector][None, :]
        r = np.sqrt((x_source - x_target) ** 2 + (y_source - y_target) ** 2)

        if target_reflector != source_reflector:
            return self.aux_weight * np.sum(hankel1(0, self.k * r), axis=1)

        diff = np.abs(self.t_nodes[:, None] - self.aux_nodes[None, :])
        coincident = diff == 0.0
        diff_safe = np.where(coincident, 1.0, diff)
        smooth = hankel1(0, self.k * r) - self.log_singularity_coeff * np.log(diff_safe)
        if np.any(coincident):
            diagonal_limit = 1.0 + self.log_singularity_coeff * (
                np.log(0.5 * self.k * self.caches.speed_t[target_reflector]) + EULER_GAMMA
            )
            smooth[coincident] = np.broadcast_to(diagonal_limit[:, None], smooth.shape)[coincident]
        return self.aux_weight * np.sum(smooth, axis=1) - 2j * np.log(2.0)

    def _paper_rhs_constant(self, reflector_index: int) -> complex:
        """Return the supplementary right-hand-side constant ``c_p``."""

        field = self.incident.boundary_field(self.reflectors[reflector_index], self.aux_nodes)
        return complex(-self.aux_weight * np.sum(field))

    def solve_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the paper-faithful Eq. (11) linear system."""

        total_unknowns = self.num_reflectors * self.n
        a = np.zeros((total_unknowns, total_unknowns), dtype=np.complex128)
        b = np.zeros((total_unknowns,), dtype=np.complex128)

        boundary_nodes = self.tau_nodes
        k_blocks: dict[tuple[int, int], np.ndarray] = {}
        m_samples: dict[tuple[int, int], np.ndarray] = {}

        for target_reflector in range(self.num_reflectors):
            for source_reflector in range(self.num_reflectors):
                k_blocks[(target_reflector, source_reflector)] = self._paper_k_block(
                    target_reflector=target_reflector,
                    source_reflector=source_reflector,
                )
                m_samples[(target_reflector, source_reflector)] = self._paper_m_samples(
                    target_reflector=target_reflector,
                    source_reflector=source_reflector,
                )

        for target_reflector in range(self.num_reflectors):
            curve = self.reflectors[target_reflector]
            rhs_boundary = self.incident.rhs_f(curve, boundary_nodes)
            rhs_constant = self._paper_rhs_constant(target_reflector)

            for boundary_index in range(self.n - 1):
                row = target_reflector * self.n + boundary_index
                b[row] = rhs_boundary[boundary_index]
                for source_reflector in range(self.num_reflectors):
                    col_slice = slice(
                        source_reflector * self.n,
                        (source_reflector + 1) * self.n,
                    )
                    block = k_blocks[(target_reflector, source_reflector)][:, boundary_index].copy()
                    if target_reflector == source_reflector:
                        diff = self.t_nodes - boundary_nodes[boundary_index]
                        block += 1.0 / diff
                    a[row, col_slice] = block / self.n

            supplementary_row = target_reflector * self.n + (self.n - 1)
            b[supplementary_row] = rhs_constant
            for source_reflector in range(self.num_reflectors):
                col_slice = slice(
                    source_reflector * self.n,
                    (source_reflector + 1) * self.n,
                )
                a[supplementary_row, col_slice] = (
                    m_samples[(target_reflector, source_reflector)] / self.n
                )

        return a, b

    def solve(self) -> MDSSolution:
        """Solve the paper Eq. (11) system and return the recovered current."""

        if self.num_reflectors == 0:
            empty = np.empty((0, self.n), dtype=np.complex128)
            return MDSSolution(
                solver=self,
                t_nodes=self.t_nodes.copy(),
                tau_nodes=self.tau_nodes.copy(),
                v_nodes=empty.copy(),
                physical_current_nodes=empty,
                boundary_residual_max=0.0,
            )

        a, b = self.solve_system()
        unknowns = solve(a, b)
        residual = float(np.max(np.abs(a @ unknowns - b))) if unknowns.size else 0.0
        v_nodes = unknowns.reshape(self.num_reflectors, self.n)
        current = v_nodes / (
            np.vstack(self.caches.speed_t) * np.sqrt(1.0 - self.t_nodes**2)[None, :]
        )
        return MDSSolution(
            solver=self,
            t_nodes=self.t_nodes.copy(),
            tau_nodes=self.tau_nodes.copy(),
            v_nodes=v_nodes,
            physical_current_nodes=current,
            boundary_residual_max=residual,
        )
