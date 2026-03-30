from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
from numpy.linalg import solve

try:
    from src.utils.math_utils import (
        cheb_first_kind,
        hankel1,
    )
except ImportError:
    from scipy.special import jv, yv

    def hankel1(v: int | np.ndarray, z: np.ndarray) -> np.ndarray:
        return jv(v, z) + 1j * yv(v, z)

    def cheb_first_kind(n: int) -> np.ndarray:
        i = np.arange(1, n + 1)
        return np.cos((2 * i - 1) * np.pi / (2 * n))

EULER_GAMMA = 0.5772156649015329


def _as_float_array(t: float | np.ndarray) -> np.ndarray:
    return np.asarray(t, dtype=np.float64)


def _broadcast_constant_like(t: float | np.ndarray, value: float) -> np.ndarray:
    arr = _as_float_array(t)
    return np.full_like(arr, fill_value=value, dtype=np.float64)


def collocation_nodes_no_overlap(n: int) -> np.ndarray:
    j = np.arange(1, n + 1, dtype=np.float64)
    return np.cos(j * np.pi / (n + 1))


class ParamCurve(ABC):
    @abstractmethod
    def x(self, t: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def y(self, t: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def coords(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.x(t), self.y(t)

    def derivatives(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.x_der(t), self.y_der(t)

    def speed(self, t: float | np.ndarray) -> np.ndarray:
        dx, dy = self.derivatives(t)
        return np.sqrt(dx * dx + dy * dy)


@dataclass(frozen=True)
class LineSegment(ParamCurve):
    x1: float
    y1: float
    x2: float
    y2: float

    def x(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = _as_float_array(t)
        s = 0.5 * (t_arr + 1.0)
        return self.x1 + s * (self.x2 - self.x1)

    def y(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = _as_float_array(t)
        s = 0.5 * (t_arr + 1.0)
        return self.y1 + s * (self.y2 - self.y1)

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        return _broadcast_constant_like(t, 0.5 * (self.x2 - self.x1))

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return _broadcast_constant_like(t, 0.5 * (self.y2 - self.y1))


@dataclass(frozen=True)
class ParabolaArc(ParamCurve):
    f: float
    y1: float
    y2: float

    @classmethod
    def symmetric(cls, f: float, aperture: float) -> "ParabolaArc":
        half = 0.5 * aperture
        return cls(f=f, y1=-half, y2=half)

    def y(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = _as_float_array(t)
        mid = 0.5 * (self.y1 + self.y2)
        half_span = 0.5 * (self.y2 - self.y1)
        return mid + half_span * t_arr

    def x(self, t: float | np.ndarray) -> np.ndarray:
        y = self.y(t)
        return (y * y) / (4.0 * self.f)

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return _broadcast_constant_like(t, 0.5 * (self.y2 - self.y1))

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        y = self.y(t)
        return y * self.y_der(t) / (2.0 * self.f)


@dataclass(frozen=True)
class HyperbolaArc(ParamCurve):
    a: float
    b: float
    center_x: float
    y1: float
    y2: float
    branch: str = "left"

    @classmethod
    def confocal(
        cls,
        focus_left_x: float,
        focus_right_x: float,
        vertex_x: float,
        y1: float,
        y2: float,
        branch: str = "left",
    ) -> "HyperbolaArc":
        if focus_right_x <= focus_left_x:
            raise ValueError("focus_right_x must be greater than focus_left_x")
        center_x = 0.5 * (focus_left_x + focus_right_x)
        c = 0.5 * (focus_right_x - focus_left_x)
        if branch == "left":
            a = center_x - vertex_x
        elif branch == "right":
            a = vertex_x - center_x
        else:
            raise ValueError("branch must be 'left' or 'right'")
        if a <= 0.0 or a >= c:
            raise ValueError("vertex_x must define 0 < a < c for a confocal hyperbola")
        b = np.sqrt(c * c - a * a)
        return cls(a=a, b=float(b), center_x=center_x, y1=y1, y2=y2, branch=branch)

    def _branch_sign(self) -> float:
        return -1.0 if self.branch == "left" else 1.0

    def y(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = _as_float_array(t)
        mid = 0.5 * (self.y1 + self.y2)
        half_span = 0.5 * (self.y2 - self.y1)
        return mid + half_span * t_arr

    def x(self, t: float | np.ndarray) -> np.ndarray:
        y = self.y(t)
        root = np.sqrt(1.0 + (y / self.b) ** 2)
        return self.center_x + self._branch_sign() * self.a * root

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return _broadcast_constant_like(t, 0.5 * (self.y2 - self.y1))

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        y = self.y(t)
        dy_dt = self.y_der(t)
        root = np.sqrt(1.0 + (y / self.b) ** 2)
        dx_dy = self._branch_sign() * self.a * y / (self.b * self.b * root)
        return dx_dy * dy_dt


@dataclass(frozen=True)
class EllipseArc(ParamCurve):
    center_x: float
    center_y: float
    a: float
    b: float
    theta1_rad: float
    theta2_rad: float
    rotation_rad: float = 0.0

    @classmethod
    def from_foci(
        cls,
        focus_1: tuple[float, float],
        focus_2: tuple[float, float],
        semi_major: float,
        theta1_deg: float,
        theta2_deg: float,
    ) -> "EllipseArc":
        x1, y1 = focus_1
        x2, y2 = focus_2
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        dx = x2 - x1
        dy = y2 - y1
        c = 0.5 * float(np.hypot(dx, dy))
        if semi_major <= c:
            raise ValueError("semi_major must be larger than half the focal separation")
        b = float(np.sqrt(semi_major * semi_major - c * c))
        rotation_rad = float(np.arctan2(dy, dx))
        return cls(
            center_x=float(center_x),
            center_y=float(center_y),
            a=float(semi_major),
            b=b,
            theta1_rad=float(np.deg2rad(theta1_deg)),
            theta2_rad=float(np.deg2rad(theta2_deg)),
            rotation_rad=rotation_rad,
        )

    def _theta(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = _as_float_array(t)
        mid = 0.5 * (self.theta1_rad + self.theta2_rad)
        half_span = 0.5 * (self.theta2_rad - self.theta1_rad)
        return mid + half_span * t_arr

    def _dtheta_dt(self) -> float:
        return 0.5 * (self.theta2_rad - self.theta1_rad)

    def _local_coords(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = self._theta(t)
        return self.a * np.cos(theta), self.b * np.sin(theta)

    def _local_derivatives(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = self._theta(t)
        dtheta_dt = self._dtheta_dt()
        return -self.a * np.sin(theta) * dtheta_dt, self.b * np.cos(theta) * dtheta_dt

    def _rotate(self, x_local: np.ndarray, y_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.cos(self.rotation_rad)
        s = np.sin(self.rotation_rad)
        x_world = self.center_x + c * x_local - s * y_local
        y_world = self.center_y + s * x_local + c * y_local
        return x_world, y_world

    def _rotate_derivative(
        self, dx_local: np.ndarray, dy_local: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        c = np.cos(self.rotation_rad)
        s = np.sin(self.rotation_rad)
        dx_world = c * dx_local - s * dy_local
        dy_world = s * dx_local + c * dy_local
        return dx_world, dy_world

    def x(self, t: float | np.ndarray) -> np.ndarray:
        x_local, y_local = self._local_coords(t)
        x_world, _ = self._rotate(x_local, y_local)
        return x_world

    def y(self, t: float | np.ndarray) -> np.ndarray:
        x_local, y_local = self._local_coords(t)
        _, y_world = self._rotate(x_local, y_local)
        return y_world

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        dx_local, dy_local = self._local_derivatives(t)
        dx_world, _ = self._rotate_derivative(dx_local, dy_local)
        return dx_world

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        dx_local, dy_local = self._local_derivatives(t)
        _, dy_world = self._rotate_derivative(dx_local, dy_local)
        return dy_world


@dataclass(frozen=True)
class TranslatedCurve(ParamCurve):
    base: ParamCurve
    dx: float = 0.0
    dy: float = 0.0

    def x(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.x(t) + self.dx

    def y(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.y(t) + self.dy

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.x_der(t)

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.y_der(t)


@dataclass(frozen=True)
class ReflectedCurveX(ParamCurve):
    base: ParamCurve
    mirror_x: float

    def x(self, t: float | np.ndarray) -> np.ndarray:
        return 2.0 * self.mirror_x - self.base.x(t)

    def y(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.y(t)

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        return -self.base.x_der(t)

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.y_der(t)


@dataclass(frozen=True)
class ComplexSourcePoint:
    k: float
    x0: float
    y0: float
    b: float
    beta_rad: float

    @property
    def rc(self) -> complex:
        return complex(self.x0, 0.0) + 1j * self.b * np.cos(self.beta_rad)

    @property
    def rcx(self) -> complex:
        return complex(self.x0, 0.0) + 1j * self.b * np.cos(self.beta_rad)

    @property
    def rcy(self) -> complex:
        return complex(self.y0, 0.0) + 1j * self.b * np.sin(self.beta_rad)

    def field(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.complex128)
        y_arr = np.asarray(y, dtype=np.complex128)
        r = np.sqrt((x_arr - self.rcx) ** 2 + (y_arr - self.rcy) ** 2)
        return hankel1(0, self.k * r)

    def boundary_field(
        self, curve: ParamCurve, t: float | np.ndarray
    ) -> np.ndarray:
        x, y = curve.coords(t)
        return self.field(x, y)

    def boundary_derivative(
        self, curve: ParamCurve, t: float | np.ndarray
    ) -> np.ndarray:
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
        return -self.boundary_derivative(curve, t)

    def far_field_pattern(self, phi: float | np.ndarray) -> np.ndarray:
        phi_arr = np.asarray(phi, dtype=np.float64)
        return np.exp(self.k * self.b * np.cos(phi_arr - self.beta_rad))


@dataclass
class SolverCaches:
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
    solver: "MultiReflectorMDS"
    t_nodes: np.ndarray
    tau_nodes: np.ndarray
    v_nodes: np.ndarray
    physical_current_nodes: np.ndarray
    boundary_residual_max: float

    def near_field(
        self, x: float | np.ndarray, y: float | np.ndarray, total: bool = True
    ) -> np.ndarray:
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
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xg, yg = np.meshgrid(xs, ys)
        u = self.near_field(xg, yg, total=total)
        return xg, yg, u

    def far_field_pattern(
        self, phi: float | np.ndarray, total: bool = True
    ) -> np.ndarray:
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
                -1j
                * self.solver.k
                * (np.outer(s_x, x_nodes) + np.outer(s_y, y_nodes))
            )
            phi_sc += phase @ self.v_nodes[q]

        phi_sc *= prefactor
        if total:
            phi_sc += self.solver.incident.far_field_pattern(phi_flat)
        return phi_sc.reshape(phi_arr.shape)

    def directivity(
        self, num_angles: int = 2048
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        phi = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
        pattern = self.far_field_pattern(phi, total=True)
        power = (2.0 * np.pi / num_angles) * np.sum(np.abs(pattern) ** 2)
        peak_index = int(np.argmax(np.abs(pattern)))
        peak_phi = float(phi[peak_index])
        d = float((2.0 * np.pi / power) * (np.abs(pattern[peak_index]) ** 2))
        return d, peak_phi, phi, pattern

    def edge_illumination_db(self, reflector_index: int = 0) -> tuple[float, float]:
        curve = self.solver.reflectors[reflector_index]
        center = complex(self.solver.incident.boundary_field(curve, 0.0))
        edge_left = complex(self.solver.incident.boundary_field(curve, -1.0))
        edge_right = complex(self.solver.incident.boundary_field(curve, 1.0))
        left = 20.0 * np.log10(np.abs(edge_left / center))
        right = 20.0 * np.log10(np.abs(edge_right / center))
        return float(left), float(right)


class MultiReflectorMDS:
    """
    Practical implementation of the edge-corrected weighted integral equation for
    E-wave scattering. The unknown is the edge-regularized current v(t).

    The executable solve path uses direct collocation of Eq. (7) with Gauss-Chebyshev
    quadrature, because this produces stable and physically sensible results in the
    current workspace and reproduces the paper's single-reflector edge-illumination
    optimum near -10 dB. The more compressed regularized matrix form printed in the
    paper is left documented in the markdown notes, but its normalization is easy to
    misread and did not validate numerically here without additional source detail.
    """

    def __init__(
        self,
        reflectors: Sequence[ParamCurve],
        incident: ComplexSourcePoint,
        n: int = 64,
    ) -> None:
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
        xq = self.reflectors[q].x(t)
        yq = self.reflectors[q].y(t)
        xp = self.reflectors[p].x(tau)
        yp = self.reflectors[p].y(tau)
        return np.sqrt((xq - xp) ** 2 + (yq - yp) ** 2)

    def _dR_dtau(self, q: int, p: int, t: np.ndarray, tau: np.ndarray) -> np.ndarray:
        xq = self.reflectors[q].x(t)
        yq = self.reflectors[q].y(t)
        xp = self.reflectors[p].x(tau)
        yp = self.reflectors[p].y(tau)
        dxp = self.reflectors[p].x_der(tau)
        dyp = self.reflectors[p].y_der(tau)
        r = np.sqrt((xq - xp) ** 2 + (yq - yp) ** 2)
        return ((xp - xq) * dxp + (yp - yq) * dyp) / r

    def solve_system(self) -> tuple[np.ndarray, np.ndarray]:
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
        a, b = self.solve_system()
        unknowns = solve(a, b)
        v_nodes = unknowns.reshape(self.num_reflectors, self.n)
        current = v_nodes / (
            np.vstack(self.caches.speed_t)
            * np.sqrt(1.0 - self.t_nodes**2)[None, :]
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


def reflector_distance_mask(
    reflectors: Sequence[ParamCurve],
    xg: np.ndarray,
    yg: np.ndarray,
    threshold: float,
    samples_per_reflector: int = 512,
) -> np.ndarray:
    x_flat = xg.reshape(-1)
    y_flat = yg.reshape(-1)
    min_dist2 = np.full(x_flat.shape, np.inf, dtype=np.float64)

    for curve in reflectors:
        t = np.linspace(-1.0, 1.0, samples_per_reflector)
        xc, yc = curve.coords(t)
        xc = np.asarray(xc, dtype=np.float64)
        yc = np.asarray(yc, dtype=np.float64)
        chunk = 128
        for start in range(0, samples_per_reflector, chunk):
            stop = min(start + chunk, samples_per_reflector)
            dx = x_flat[:, None] - xc[None, start:stop]
            dy = y_flat[:, None] - yc[None, start:stop]
            dist2 = np.min(dx * dx + dy * dy, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)

    return (min_dist2 <= threshold * threshold).reshape(xg.shape)


def plot_near_field(
    solution: MDSSolution,
    xg: np.ndarray,
    yg: np.ndarray,
    total_field: np.ndarray,
    field_label: str = "Total",
    plot_style: str = "standard",
    paper_bar_max: float = 9.0,
    paper_percentile_low: float = 2.0,
    paper_percentile_high: float = 99.7,
    mask_distance: float | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    intensity = np.abs(total_field) ** 2
    phase = np.angle(total_field)

    mask = None
    effective_mask_distance = mask_distance
    if plot_style == "paper" and effective_mask_distance is not None:
        effective_mask_distance = min(effective_mask_distance, 0.06)
    if plot_style == "figure" and effective_mask_distance is not None:
        effective_mask_distance = min(effective_mask_distance, 0.08)

    if effective_mask_distance is not None and effective_mask_distance > 0.0:
        mask = reflector_distance_mask(
            solution.solver.reflectors, xg, yg, threshold=effective_mask_distance
        )
        phase = np.ma.array(phase, mask=mask)

    extent = [float(xg.min()), float(xg.max()), float(yg.min()), float(yg.max())]

    if plot_style == "paper":
        eps = np.finfo(np.float64).tiny
        log_intensity = np.log10(np.maximum(intensity, eps))
        if mask is not None:
            visible = log_intensity[~mask]
        else:
            visible = log_intensity.reshape(-1)
        if visible.size == 0:
            visible = log_intensity.reshape(-1)
        low = float(np.percentile(visible, paper_percentile_low))
        high = float(np.percentile(visible, paper_percentile_high))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            high = low + 1.0
        paper_map = paper_bar_max * (log_intensity - low) / (high - low)
        paper_map = np.clip(paper_map, 0.0, paper_bar_max)
        if mask is not None:
            paper_map = np.ma.array(paper_map, mask=mask)

        fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
        im = ax.imshow(
            paper_map,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="gray",
            vmin=0.0,
            vmax=paper_bar_max,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cbar = fig.colorbar(im, ax=ax, shrink=0.92)
        cbar.set_ticks(np.arange(1.0, np.floor(paper_bar_max) + 1.0, 1.0))

        for curve in solution.solver.reflectors:
            t_plot = np.linspace(-1.0, 1.0, 400)
            xc, yc = curve.coords(t_plot)
            ax.plot(xc, yc, color="white", linewidth=1.0)
    elif plot_style == "figure":
        intensity_db = 10.0 * np.log10(intensity / np.nanmax(intensity))
        if mask is not None:
            intensity_db = np.ma.array(intensity_db, mask=mask)

        fig, ax = plt.subplots(figsize=(6.8, 6.8), constrained_layout=True)
        ax.imshow(
            intensity_db,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="jet",
            vmin=-45.0,
            vmax=0.0,
        )

        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.9, alpha=0.75)

        reflector_colors = ["#ff55ff", "#55ff55", "#ffd166", "#7df9ff"]
        for idx, curve in enumerate(solution.solver.reflectors):
            t_plot = np.linspace(-1.0, 1.0, 500)
            xc, yc = curve.coords(t_plot)
            color = reflector_colors[idx % len(reflector_colors)]
            ax.plot(xc, yc, color="white", linewidth=2.2, solid_capstyle="round")
            ax.plot(xc, yc, color=color, linewidth=1.4, solid_capstyle="round")

        ax.plot(
            [solution.solver.incident.x0],
            [solution.solver.incident.y0],
            marker="o",
            markersize=4.0,
            markerfacecolor="#00ff66",
            markeredgecolor="#00ff66",
            linestyle="None",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        intensity_db = 10.0 * np.log10(intensity / np.nanmax(intensity))
        if mask is not None:
            intensity_db = np.ma.array(intensity_db, mask=mask)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        im0 = axes[0].imshow(
            intensity_db,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="jet",
            vmin=-60.0,
            vmax=0.0,
        )
        axes[0].set_title(f"Near-Field {field_label} Intensity [dB]")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[0], shrink=0.9)

        im1 = axes[1].imshow(
            phase,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="twilight",
        )
        axes[1].set_title(f"Near-Field {field_label} Phase [rad]")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im1, ax=axes[1], shrink=0.9)

        for ax in axes:
            for curve in solution.solver.reflectors:
                t_plot = np.linspace(-1.0, 1.0, 400)
                xc, yc = curve.coords(t_plot)
                ax.plot(xc, yc, "k-", linewidth=1.0)

    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_far_field(
    phi: np.ndarray,
    pattern: np.ndarray,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    pattern_db = 20.0 * np.log10(np.abs(pattern) / np.max(np.abs(pattern)))
    phi_deg = np.rad2deg(phi)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(phi_deg, pattern_db, linewidth=1.2)
    ax.set_xlim(float(phi_deg.min()), float(phi_deg.max()))
    ax.set_ylim(-60.0, 1.0)
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Normalized Pattern [dB]")
    ax.set_title("Far-Field Pattern")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_focus_centered_parabola(
    aperture: float,
    focal_ratio: float,
    x_focus: float = 0.0,
    y_focus: float = 0.0,
) -> ParamCurve:
    f = focal_ratio * aperture
    base = ParabolaArc.symmetric(f=f, aperture=aperture)
    return TranslatedCurve(base=base, dx=x_focus - f, dy=y_focus)


def build_shifted_parabola_from_bounds(
    focal_length: float,
    y1: float,
    y2: float,
    x_shift: float = 0.0,
    y_shift: float = 0.0,
) -> ParamCurve:
    base = ParabolaArc(f=focal_length, y1=y1, y2=y2)
    return TranslatedCurve(base=base, dx=x_shift, dy=y_shift)


def build_shifted_parabola(
    aperture: float,
    focal_ratio: float,
    x_shift: float,
    y_shift: float = 0.0,
) -> ParamCurve:
    f = focal_ratio * aperture
    base = ParabolaArc.symmetric(f=f, aperture=aperture)
    return TranslatedCurve(base=base, dx=x_shift, dy=y_shift)


def symmetric_bounds(aperture: float) -> tuple[float, float]:
    half = 0.5 * aperture
    return -half, half


def build_single_shifted_parabolic_example(
    n: int,
    aperture: float,
    focal_ratio: float,
    kb: float,
    beta_deg: float,
) -> MultiReflectorMDS:
    wavelength = 1.0
    k = 2.0 * np.pi / wavelength
    curve = build_focus_centered_parabola(
        aperture=aperture,
        focal_ratio=focal_ratio,
        x_focus=0.0,
        y_focus=0.0,
    )
    csp = ComplexSourcePoint(
        k=k,
        x0=0.0,
        y0=0.0,
        b=kb / k,
        beta_rad=np.deg2rad(beta_deg),
    )
    return MultiReflectorMDS(reflectors=[curve], incident=csp, n=n)


def build_axis_aligned_ellipse_side(
    center_x: float,
    center_y: float,
    semi_x: float,
    semi_y: float,
    half_angle_deg: float,
    side: str,
) -> EllipseArc:
    if side == "right":
        theta1_deg = -half_angle_deg
        theta2_deg = half_angle_deg
    elif side == "left":
        theta1_deg = 180.0 - half_angle_deg
        theta2_deg = 180.0 + half_angle_deg
    else:
        raise ValueError("side must be 'left' or 'right'")
    return EllipseArc(
        center_x=float(center_x),
        center_y=float(center_y),
        a=float(semi_x),
        b=float(semi_y),
        theta1_rad=float(np.deg2rad(theta1_deg)),
        theta2_rad=float(np.deg2rad(theta2_deg)),
        rotation_rad=0.0,
    )


def build_confocal_elliptic_example(
    n: int,
    kb: float,
    beta_deg: float,
    ellipse_upper_mode: str,
    ellipse_d: float | None,
    ellipse_feed_x: float | None,
    ellipse_feed_y: float | None,
    ellipse_common_focus_x: float | None,
    ellipse_common_focus_y: float | None,
    ellipse_lower_center_x: float | None,
    ellipse_lower_center_y: float | None,
    ellipse_mirror_x: float | None,
    ellipse_upper_shift_y: float | None,
    ellipse_output_focus_x: float | None,
    ellipse_output_focus_y: float | None,
    ellipse_lower_a: float | None,
    ellipse_lower_b: float | None,
    ellipse_upper_a: float | None,
    ellipse_lower_theta1: float | None,
    ellipse_lower_theta2: float | None,
    ellipse_upper_theta1: float | None,
    ellipse_upper_theta2: float | None,
    ellipse_half_angle_deg: float | None,
) -> MultiReflectorMDS:
    wavelength = 1.0
    k = 2.0 * np.pi / wavelength

    d = 20.0 if ellipse_d is None else float(ellipse_d)

    feed_x = 0.34 * d if ellipse_feed_x is None else float(ellipse_feed_x)
    feed_y = -0.37 * d if ellipse_feed_y is None else float(ellipse_feed_y)
    common_focus_x = 0.34 * d if ellipse_common_focus_x is None else float(ellipse_common_focus_x)
    common_focus_y = 0.36 * d if ellipse_common_focus_y is None else float(ellipse_common_focus_y)
    lower_center_x = 0.125 * d if ellipse_lower_center_x is None else float(ellipse_lower_center_x)
    lower_center_y = -0.12 * d if ellipse_lower_center_y is None else float(ellipse_lower_center_y)
    output_focus_x = 0.50 * d if ellipse_output_focus_x is None else float(ellipse_output_focus_x)
    output_focus_y = 1.00 * d if ellipse_output_focus_y is None else float(ellipse_output_focus_y)

    lower_a_default = 0.19 * d if ellipse_upper_mode == "mirror_vertical" else 0.56 * d
    lower_a = lower_a_default if ellipse_lower_a is None else float(ellipse_lower_a)
    lower_b = 0.43 * d if ellipse_lower_b is None else float(ellipse_lower_b)
    upper_a = lower_a if ellipse_upper_a is None else float(ellipse_upper_a)

    lower_theta1 = 56.0 if ellipse_lower_theta1 is None else float(ellipse_lower_theta1)
    lower_theta2 = 176.0 if ellipse_lower_theta2 is None else float(ellipse_lower_theta2)
    upper_theta1 = lower_theta1 + 180.0 if ellipse_upper_theta1 is None else float(ellipse_upper_theta1)
    upper_theta2 = lower_theta2 + 180.0 if ellipse_upper_theta2 is None else float(ellipse_upper_theta2)
    half_angle_deg = 130.0 if ellipse_half_angle_deg is None else float(ellipse_half_angle_deg)
    mirror_x = 0.375 * d if ellipse_mirror_x is None else float(ellipse_mirror_x)
    upper_shift_y = 0.85 * d if ellipse_upper_shift_y is None else float(ellipse_upper_shift_y)

    if ellipse_upper_mode == "mirror_vertical":
        lower_reflector = build_axis_aligned_ellipse_side(
            center_x=lower_center_x,
            center_y=lower_center_y,
            semi_x=lower_a,
            semi_y=lower_b,
            half_angle_deg=half_angle_deg,
            side="right",
        )
    else:
        lower_reflector = EllipseArc.from_foci(
            focus_1=(feed_x, feed_y),
            focus_2=(common_focus_x, common_focus_y),
            semi_major=lower_a,
            theta1_deg=lower_theta1,
            theta2_deg=lower_theta2,
        )

    if ellipse_upper_mode == "mirror_vertical":
        upper_reflector = TranslatedCurve(
            base=ReflectedCurveX(base=lower_reflector, mirror_x=mirror_x),
            dy=upper_shift_y,
        )
    elif ellipse_upper_mode == "focus_pair":
        upper_reflector = EllipseArc.from_foci(
            focus_1=(common_focus_x, common_focus_y),
            focus_2=(output_focus_x, output_focus_y),
            semi_major=upper_a,
            theta1_deg=upper_theta1,
            theta2_deg=upper_theta2,
        )
    else:
        raise ValueError(f"Unsupported ellipse_upper_mode: {ellipse_upper_mode}")

    csp = ComplexSourcePoint(
        k=k,
        x0=feed_x,
        y0=feed_y,
        b=kb / k,
        beta_rad=np.deg2rad(beta_deg),
    )
    return MultiReflectorMDS(
        reflectors=[lower_reflector, upper_reflector],
        incident=csp,
        n=n,
    )


def build_cassegrain_example(
    n: int,
    main_aperture: float,
    sub_aperture: float,
    focal_ratio: float,
    kb: float,
    beta_deg: float,
    main_vertex_x: float,
    feed_x: float | None,
    subreflector_vertex_x: float | None,
    main_y1: float | None,
    main_y2: float | None,
    sub_y1: float | None,
    sub_y2: float | None,
    sub_branch: str,
) -> MultiReflectorMDS:
    wavelength = 1.0
    k = 2.0 * np.pi / wavelength
    focal_length = focal_ratio * main_aperture

    main_y1_resolved, main_y2_resolved = (
        symmetric_bounds(main_aperture)
        if main_y1 is None or main_y2 is None
        else (main_y1, main_y2)
    )
    sub_y1_resolved, sub_y2_resolved = (
        symmetric_bounds(sub_aperture)
        if sub_y1 is None or sub_y2 is None
        else (sub_y1, sub_y2)
    )

    main_curve = build_shifted_parabola_from_bounds(
        focal_length=focal_length,
        y1=main_y1_resolved,
        y2=main_y2_resolved,
        x_shift=main_vertex_x,
        y_shift=0.0,
    )
    main_focus_x = main_vertex_x + focal_length

    feed_x_resolved = (
        main_vertex_x + 0.30 * focal_length if feed_x is None else float(feed_x)
    )
    sub_vertex_x_resolved = (
        main_vertex_x + 0.64 * focal_length
        if subreflector_vertex_x is None
        else float(subreflector_vertex_x)
    )

    sub_curve = HyperbolaArc.confocal(
        focus_left_x=min(feed_x_resolved, main_focus_x),
        focus_right_x=max(feed_x_resolved, main_focus_x),
        vertex_x=sub_vertex_x_resolved,
        y1=sub_y1_resolved,
        y2=sub_y2_resolved,
        branch=sub_branch,
    )

    csp = ComplexSourcePoint(
        k=k,
        x0=feed_x_resolved,
        y0=0.0,
        b=kb / k,
        beta_rad=np.deg2rad(beta_deg),
    )
    return MultiReflectorMDS(
        reflectors=[main_curve, sub_curve],
        incident=csp,
        n=n,
    )


def build_two_bracket_reflector_example(
    n: int,
    aperture: float,
    focal_ratio: float,
    kb: float,
    beta_deg: float,
    secondary_scale: float,
    small_vertex_x: float | None,
    large_vertex_x: float | None,
) -> MultiReflectorMDS:
    if secondary_scale <= 1.0:
        raise ValueError("secondary_scale must be greater than 1.0")

    wavelength = 1.0
    k = 2.0 * np.pi / wavelength
    small_aperture = aperture
    large_aperture = secondary_scale * aperture
    small_x = 0.35 * small_aperture if small_vertex_x is None else small_vertex_x
    large_x = -0.55 * large_aperture if large_vertex_x is None else large_vertex_x

    small_right = build_shifted_parabola(
        aperture=small_aperture,
        focal_ratio=focal_ratio,
        x_shift=small_x,
        y_shift=0.0,
    )
    large_left = build_shifted_parabola(
        aperture=large_aperture,
        focal_ratio=focal_ratio,
        x_shift=large_x,
        y_shift=0.0,
    )
    csp = ComplexSourcePoint(
        k=k,
        x0=0.0,
        y0=0.0,
        b=kb / k,
        beta_rad=np.deg2rad(beta_deg),
    )
    return MultiReflectorMDS(
        reflectors=[small_right, large_left],
        incident=csp,
        n=n,
    )


def resolved_beta_deg(args: argparse.Namespace) -> float:
    if args.beta_deg is not None:
        return args.beta_deg
    if args.scene == "single_shifted":
        return 180.0
    if args.scene == "cassegrain":
        return 0.0
    if args.scene == "two_brackets":
        return 0.0
    if args.scene == "confocal_elliptic":
        return 140.0
    raise ValueError(f"Unsupported scene: {args.scene}")


def build_scene(args: argparse.Namespace) -> MultiReflectorMDS:
    beta_deg = resolved_beta_deg(args)
    if args.scene == "single_shifted":
        return build_single_shifted_parabolic_example(
            n=args.n,
            aperture=args.aperture,
            focal_ratio=args.focal_ratio,
            kb=args.kb,
            beta_deg=beta_deg,
        )
    if args.scene == "confocal_elliptic":
        return build_confocal_elliptic_example(
            n=args.n,
            kb=args.kb,
            beta_deg=beta_deg,
            ellipse_upper_mode=args.ellipse_upper_mode,
            ellipse_d=args.ellipse_d,
            ellipse_feed_x=args.ellipse_feed_x,
            ellipse_feed_y=args.ellipse_feed_y,
            ellipse_common_focus_x=args.ellipse_common_focus_x,
            ellipse_common_focus_y=args.ellipse_common_focus_y,
            ellipse_lower_center_x=args.ellipse_lower_center_x,
            ellipse_lower_center_y=args.ellipse_lower_center_y,
            ellipse_mirror_x=args.ellipse_mirror_x,
            ellipse_upper_shift_y=args.ellipse_upper_shift_y,
            ellipse_output_focus_x=args.ellipse_output_focus_x,
            ellipse_output_focus_y=args.ellipse_output_focus_y,
            ellipse_lower_a=args.ellipse_lower_a,
            ellipse_lower_b=args.ellipse_lower_b,
            ellipse_upper_a=args.ellipse_upper_a,
            ellipse_lower_theta1=args.ellipse_lower_theta1,
            ellipse_lower_theta2=args.ellipse_lower_theta2,
            ellipse_upper_theta1=args.ellipse_upper_theta1,
            ellipse_upper_theta2=args.ellipse_upper_theta2,
            ellipse_half_angle_deg=args.ellipse_half_angle_deg,
        )
    if args.scene == "cassegrain":
        return build_cassegrain_example(
            n=args.n,
            main_aperture=args.aperture,
            sub_aperture=args.sub_aperture,
            focal_ratio=args.focal_ratio,
            kb=args.kb,
            beta_deg=beta_deg,
            main_vertex_x=args.main_vertex_x,
            feed_x=args.feed_x,
            subreflector_vertex_x=args.subreflector_vertex_x,
            main_y1=args.main_y1,
            main_y2=args.main_y2,
            sub_y1=args.sub_y1,
            sub_y2=args.sub_y2,
            sub_branch=args.sub_branch,
        )
    if args.scene == "two_brackets":
        return build_two_bracket_reflector_example(
            n=args.n,
            aperture=args.aperture,
            focal_ratio=args.focal_ratio,
            kb=args.kb,
            beta_deg=beta_deg,
            secondary_scale=args.secondary_scale,
            small_vertex_x=args.small_vertex_x,
            large_vertex_x=args.large_vertex_x,
        )
    raise ValueError(f"Unsupported scene: {args.scene}")


def resolved_plot_window(
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    x_min = args.x_min
    x_max = args.x_max
    y_min = args.y_min
    y_max = args.y_max

    if args.scene == "confocal_elliptic":
        if x_min == -5.0:
            x_min = -12.0
        if x_max == 25.0:
            x_max = 28.0
        if y_min == -15.0:
            y_min = -14.0
        if y_max == 15.0:
            y_max = 29.0

    return x_min, x_max, y_min, y_max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MDS implementation for 2-D E-wave reflector scattering with CSP feed."
    )
    parser.add_argument("--n", type=int, default=120, help="Interpolation order.")
    parser.add_argument(
        "--scene",
        choices=("single_shifted", "cassegrain", "two_brackets", "confocal_elliptic"),
        default="cassegrain",
        help="Geometry preset. 'cassegrain' uses a parabolic main reflector and a hyperbolic subreflector, while 'confocal_elliptic' uses two ellipse arcs with a shared intermediate focus.",
    )
    parser.add_argument(
        "--aperture", type=float, default=30.0, help="Primary reflector aperture in wavelengths."
    )
    parser.add_argument(
        "--sub-aperture",
        type=float,
        default=7.0,
        help="Subreflector aperture in wavelengths for the Cassegrain scene.",
    )
    parser.add_argument(
        "--focal-ratio",
        type=float,
        default=0.5,
        help="Focal distance divided by aperture.",
    )
    parser.add_argument(
        "--main-vertex-x",
        type=float,
        default=0.0,
        help="Main-reflector vertex x-position for the Cassegrain scene.",
    )
    parser.add_argument(
        "--feed-x",
        type=float,
        default=None,
        help="Feed x-position for the Cassegrain scene. Default is a tuned 0.30*f from the main vertex.",
    )
    parser.add_argument(
        "--subreflector-vertex-x",
        type=float,
        default=None,
        help="Hyperbolic subreflector vertex x-position for the Cassegrain scene. Default is a tuned 0.64*f from the main vertex.",
    )
    parser.add_argument(
        "--sub-branch",
        choices=("left", "right"),
        default="left",
        help="Hyperbola branch used for the Cassegrain subreflector.",
    )
    parser.add_argument(
        "--main-y1",
        type=float,
        default=None,
        help="Lower y-bound of the main reflector. Defaults to -aperture/2.",
    )
    parser.add_argument(
        "--main-y2",
        type=float,
        default=None,
        help="Upper y-bound of the main reflector. Defaults to +aperture/2.",
    )
    parser.add_argument(
        "--sub-y1",
        type=float,
        default=None,
        help="Lower y-bound of the subreflector. Defaults to -sub_aperture/2.",
    )
    parser.add_argument(
        "--sub-y2",
        type=float,
        default=None,
        help="Upper y-bound of the subreflector. Defaults to +sub_aperture/2.",
    )
    parser.add_argument(
        "--secondary-scale",
        type=float,
        default=2.0,
        help="Secondary reflector aperture divided by primary aperture for the two-reflector scene.",
    )
    parser.add_argument(
        "--ellipse-d",
        type=float,
        default=None,
        help="Reference size for the confocal-elliptic scene. The default reproduces the sent figure with d=20.",
    )
    parser.add_argument(
        "--ellipse-feed-x",
        type=float,
        default=None,
        help="Feed x-position for the confocal-elliptic scene.",
    )
    parser.add_argument(
        "--ellipse-feed-y",
        type=float,
        default=None,
        help="Feed y-position for the confocal-elliptic scene.",
    )
    parser.add_argument(
        "--ellipse-common-focus-x",
        type=float,
        default=None,
        help="Intermediate shared-focus x-position for the confocal-elliptic scene.",
    )
    parser.add_argument(
        "--ellipse-common-focus-y",
        type=float,
        default=None,
        help="Intermediate shared-focus y-position for the confocal-elliptic scene.",
    )
    parser.add_argument(
        "--ellipse-upper-mode",
        choices=("mirror_vertical", "focus_pair"),
        default="mirror_vertical",
        help="How the upper elliptic reflector is constructed. 'mirror_vertical' makes it an exact mirror of the lower reflector across x=const and then shifts it upward.",
    )
    parser.add_argument(
        "--ellipse-mirror-x",
        type=float,
        default=None,
        help="Vertical mirror line x-position used when --ellipse-upper-mode=mirror_vertical. Defaults to the lower ellipse axis.",
    )
    parser.add_argument(
        "--ellipse-lower-center-x",
        type=float,
        default=None,
        help="Center x-position of the lower bracket-like ellipse used in mirror_vertical mode.",
    )
    parser.add_argument(
        "--ellipse-lower-center-y",
        type=float,
        default=None,
        help="Center y-position of the lower bracket-like ellipse used in mirror_vertical mode.",
    )
    parser.add_argument(
        "--ellipse-upper-shift-y",
        type=float,
        default=None,
        help="Vertical shift applied to the mirrored upper reflector when --ellipse-upper-mode=mirror_vertical.",
    )
    parser.add_argument(
        "--ellipse-output-focus-x",
        type=float,
        default=None,
        help="Second-focus x-position of the upper ellipse in the confocal-elliptic scene.",
    )
    parser.add_argument(
        "--ellipse-output-focus-y",
        type=float,
        default=None,
        help="Second-focus y-position of the upper ellipse in the confocal-elliptic scene.",
    )
    parser.add_argument(
        "--ellipse-lower-a",
        type=float,
        default=None,
        help="Semi-major axis of the lower elliptic reflector.",
    )
    parser.add_argument(
        "--ellipse-lower-b",
        type=float,
        default=None,
        help="Semi-minor axis of the lower bracket-like ellipse used in mirror_vertical mode.",
    )
    parser.add_argument(
        "--ellipse-upper-a",
        type=float,
        default=None,
        help="Semi-major axis of the upper elliptic reflector.",
    )
    parser.add_argument(
        "--ellipse-lower-theta1",
        type=float,
        default=None,
        help="Start angle in degrees for the lower elliptic reflector arc.",
    )
    parser.add_argument(
        "--ellipse-lower-theta2",
        type=float,
        default=None,
        help="End angle in degrees for the lower elliptic reflector arc.",
    )
    parser.add_argument(
        "--ellipse-upper-theta1",
        type=float,
        default=None,
        help="Start angle in degrees for the upper elliptic reflector arc.",
    )
    parser.add_argument(
        "--ellipse-upper-theta2",
        type=float,
        default=None,
        help="End angle in degrees for the upper elliptic reflector arc.",
    )
    parser.add_argument(
        "--ellipse-half-angle-deg",
        type=float,
        default=None,
        help="Half-angle span for the bracket-like ellipse arc in mirror_vertical mode. This enforces equal x-coordinates at both arc endpoints.",
    )
    parser.add_argument("--kb", type=float, default=9.0, help="CSP beam parameter kb.")
    parser.add_argument(
        "--beta-deg",
        type=float,
        default=None,
        help="CSP beam aiming angle in degrees. Defaults to 180 deg for 'single_shifted', 0 deg for 'cassegrain' and 'two_brackets', and 140 deg for 'confocal_elliptic'.",
    )
    parser.add_argument(
        "--small-vertex-x",
        type=float,
        default=None,
        help="Vertex x-position of the small right '(' reflector in the two-reflector scene.",
    )
    parser.add_argument(
        "--large-vertex-x",
        type=float,
        default=None,
        help="Vertex x-position of the large left '(' reflector in the two-reflector scene.",
    )
    parser.add_argument(
        "--field-kind",
        choices=("total", "scattered"),
        default="total",
        help="Near-field quantity to plot. 'total' shows incident plus scattered field, while 'scattered' suppresses the direct CSP beam.",
    )
    parser.add_argument(
        "--near-plot-style",
        choices=("standard", "paper", "figure"),
        default="paper",
        help="Near-field plot style. 'paper' uses grayscale with a paper-like positive-value bar, and 'figure' makes a single-panel color chart similar to the reference figure style.",
    )
    parser.add_argument(
        "--paper-bar-max",
        type=float,
        default=9.0,
        help="Maximum value shown on the paper-style grayscale bar.",
    )
    parser.add_argument(
        "--paper-percentile-low",
        type=float,
        default=2.0,
        help="Lower percentile used to scale paper-style grayscale images.",
    )
    parser.add_argument(
        "--paper-percentile-high",
        type=float,
        default=99.7,
        help="Upper percentile used to scale paper-style grayscale images.",
    )
    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=25.0)
    parser.add_argument("--y-min", type=float, default=-15.0)
    parser.add_argument("--y-max", type=float, default=15.0)
    parser.add_argument("--nx", type=int, default=220)
    parser.add_argument("--ny", type=int, default=220)
    parser.add_argument(
        "--mask-distance",
        type=float,
        default=0.35,
        help="Mask distance around the reflector for field plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Compute results without opening matplotlib windows.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=str(datetime.now()).replace("-", "").replace(" ", "").replace(":", "").replace(".", ""),
        help="If given, save '<prefix>_near.png' and '<prefix>_far.png'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solver = build_scene(args)
    solution = solver.solve()
    directivity, peak_phi, phi, pattern = solution.directivity()
    beta_deg = resolved_beta_deg(args)

    print(f"scene = {args.scene}")
    print(f"n = {solver.n}")
    print(f"k = {solver.k:.12g}")
    print(f"kb = {args.kb:.12g}")
    print(f"feed = ({solver.incident.x0:.6f}, {solver.incident.y0:.6f})")
    print(f"beta [deg] = {beta_deg:.6f}")
    print(f"Boundary residual max = {solution.boundary_residual_max:.6e}")
    print(f"Directivity = {directivity:.6f}")
    print(f"Peak angle [deg] = {np.rad2deg(peak_phi):.6f}")
    for reflector_index in range(solver.num_reflectors):
        edge_left_db, edge_right_db = solution.edge_illumination_db(
            reflector_index=reflector_index
        )
        print(
            f"Reflector {reflector_index} edge illumination left/right [dB] = "
            f"{edge_left_db:.6f}, {edge_right_db:.6f}"
        )

    plot_total_field = args.field_kind == "total"
    x_min, x_max, y_min, y_max = resolved_plot_window(args)
    xg, yg, u = solution.near_field_grid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        nx=args.nx,
        ny=args.ny,
        total=plot_total_field,
    )

    near_path = (
        None
        if args.save_prefix is None
        else f"{args.save_prefix}_{args.field_kind}_near.png"
    )
    far_path = None if args.save_prefix is None else f"{args.save_prefix}_far.png"

    plot_near_field(
        solution=solution,
        xg=xg,
        yg=yg,
        total_field=u,
        field_label="Total" if plot_total_field else "Scattered",
        plot_style=args.near_plot_style,
        paper_bar_max=args.paper_bar_max,
        paper_percentile_low=args.paper_percentile_low,
        paper_percentile_high=args.paper_percentile_high,
        mask_distance=args.mask_distance,
        save_path=near_path,
        show=not args.no_show,
    )
    plot_far_field(
        phi=phi,
        pattern=pattern,
        save_path=far_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
