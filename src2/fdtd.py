"""Simple 2-D FDTD backend for plane-wave scattering comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ParamCurve
from .solver import MultiReflectorMDS, PlaneWave


@dataclass(frozen=True)
class FDTDConfig:
    """Numerical controls for the time-domain comparison backend."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    points_per_wavelength: float = 20.0
    padding_wavelengths: float = 3.0
    absorber_cells: int = 24
    ramp_cycles: float = 3.0
    settle_cycles: float = 10.0
    sample_cycles: float = 12.0
    courant: float = 0.45
    damping_strength: float = 3.0


@dataclass
class FDTDSolution:
    """Frequency-domain field recovered from a time-domain simulation."""

    solver: MultiReflectorMDS
    xg: np.ndarray
    yg: np.ndarray
    incident_field: np.ndarray
    total_field: np.ndarray
    scattered_field: np.ndarray
    dx: float
    dy: float
    dt: float
    steps: int
    sample_steps: int
    points_per_wavelength: float
    absorber_cells: int
    boundary_residual_max: float = float("nan")

    def field(self, kind: str) -> np.ndarray:
        """Return the requested field array on the plotting grid."""

        if kind == "incident":
            return self.incident_field
        if kind == "scattered":
            return self.scattered_field
        if kind == "total":
            return self.total_field
        raise ValueError(f"Unsupported field kind: {kind}")


def _curve_length(curve: ParamCurve, samples: int = 2048) -> float:
    """Estimate curve length by uniform parameter sampling."""

    t = np.linspace(-1.0, 1.0, samples)
    x, y = curve.coords(t)
    ds = np.hypot(np.diff(x), np.diff(y))
    return float(np.sum(ds))


def _mark_segment(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, radius: int) -> None:
    """Rasterize a segment on the mask and thicken it by ``radius`` cells."""

    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(np.int64)
    ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(np.int64)
    ny, nx = mask.shape
    for ix, iy in zip(xs, ys):
        x_start = max(ix - radius, 0)
        x_stop = min(ix + radius + 1, nx)
        y_start = max(iy - radius, 0)
        y_stop = min(iy + radius + 1, ny)
        mask[y_start:y_stop, x_start:x_stop] = True


def _build_reflector_mask(
    reflectors: list[ParamCurve],
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    cells_per_wavelength: float,
) -> np.ndarray:
    """Rasterize reflector curves onto the FDTD grid."""

    mask = np.zeros((ys.size, xs.size), dtype=bool)
    x_min = float(xs[0])
    y_min = float(ys[0])
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    grid_step = min(dx, dy)
    for curve in reflectors:
        length = max(_curve_length(curve), grid_step)
        samples = max(512, int(np.ceil(6.0 * length / grid_step)))
        t = np.linspace(-1.0, 1.0, samples)
        xc, yc = curve.coords(t)
        ix = np.clip(np.rint((xc - x_min) / dx).astype(np.int64), 0, xs.size - 1)
        iy = np.clip(np.rint((yc - y_min) / dy).astype(np.int64), 0, ys.size - 1)
        radius = max(1, int(np.ceil(cells_per_wavelength / 40.0)))
        for idx in range(samples - 1):
            _mark_segment(mask, int(ix[idx]), int(iy[idx]), int(ix[idx + 1]), int(iy[idx + 1]), radius)
    return mask


def _absorber_sigma(nx: int, ny: int, cells: int, omega: float, strength: float) -> np.ndarray:
    """Build a quadratic damping profile near the simulation box edges."""

    if cells <= 0:
        return np.zeros((ny, nx), dtype=np.float64)
    sx = np.zeros(nx, dtype=np.float64)
    sy = np.zeros(ny, dtype=np.float64)
    edge_scale = float(strength) * omega
    for idx in range(cells):
        weight = ((cells - idx) / cells) ** 2
        value = edge_scale * weight
        sx[idx] = max(sx[idx], value)
        sx[-idx - 1] = max(sx[-idx - 1], value)
        sy[idx] = max(sy[idx], value)
        sy[-idx - 1] = max(sy[-idx - 1], value)
    return sy[:, None] + sx[None, :]


def _bilinear_resample(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """Resample a field from a regular Cartesian grid using bilinear weights."""

    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    fx = np.clip((xq - xs[0]) / dx, 0.0, xs.size - 1.000001)
    fy = np.clip((yq - ys[0]) / dy, 0.0, ys.size - 1.000001)
    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    ix1 = np.clip(ix0 + 1, 0, xs.size - 1)
    iy1 = np.clip(iy0 + 1, 0, ys.size - 1)
    tx = fx - ix0
    ty = fy - iy0

    v00 = values[iy0, ix0]
    v10 = values[iy0, ix1]
    v01 = values[iy1, ix0]
    v11 = values[iy1, ix1]
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def solve_fdtd(
    solver: MultiReflectorMDS,
    config: FDTDConfig,
) -> FDTDSolution:
    """Approximate the steady-state field using a scalar 2-D FDTD scheme.

    This backend is intentionally scoped to plane-wave incidence. It computes a
    real-valued time-domain scattered field with time-varying Dirichlet data on
    the reflector cells, then recovers the complex phasor on the requested plot
    grid by temporal demodulation.
    """

    if not isinstance(solver.incident, PlaneWave):
        raise ValueError("The FDTD backend currently supports only plane-wave incidence.")
    if config.points_per_wavelength < 8.0:
        raise ValueError("fdtd points_per_wavelength must be at least 8")
    if config.absorber_cells < 4:
        raise ValueError("fdtd absorber_cells must be at least 4")

    wavelength = 2.0 * np.pi / solver.k
    padding = config.padding_wavelengths * wavelength
    sim_x_min = config.x_min - padding
    sim_x_max = config.x_max + padding
    sim_y_min = config.y_min - padding
    sim_y_max = config.y_max + padding

    dx_target = wavelength / config.points_per_wavelength
    nx_sim = max(int(np.ceil((sim_x_max - sim_x_min) / dx_target)) + 1, config.nx + 2)
    ny_sim = max(int(np.ceil((sim_y_max - sim_y_min) / dx_target)) + 1, config.ny + 2)
    xs = np.linspace(sim_x_min, sim_x_max, nx_sim)
    ys = np.linspace(sim_y_min, sim_y_max, ny_sim)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])

    courant_limit = 1.0 / np.sqrt((1.0 / dx**2) + (1.0 / dy**2))
    dt = config.courant * courant_limit
    omega = solver.k
    period = 2.0 * np.pi / omega
    ramp_steps = max(1, int(np.ceil(config.ramp_cycles * period / dt)))
    settle_steps = max(ramp_steps + 1, int(np.ceil(config.settle_cycles * period / dt)))
    sample_steps = max(8, int(np.ceil(config.sample_cycles * period / dt)))
    total_steps = settle_steps + sample_steps

    xg_sim, yg_sim = np.meshgrid(xs, ys)
    phase = solver.k * (
        solver.incident.direction_x * xg_sim + solver.incident.direction_y * yg_sim
    )
    sigma = _absorber_sigma(
        nx=nx_sim,
        ny=ny_sim,
        cells=config.absorber_cells,
        omega=omega,
        strength=config.damping_strength,
    )

    reflector_mask = _build_reflector_mask(
        solver.reflectors,
        xs,
        ys,
        cells_per_wavelength=float(wavelength / min(dx, dy)),
    )

    if not np.any(reflector_mask):
        plot_xs = np.linspace(config.x_min, config.x_max, config.nx)
        plot_ys = np.linspace(config.y_min, config.y_max, config.ny)
        plot_xg, plot_yg = np.meshgrid(plot_xs, plot_ys)
        incident_field = solver.incident.field(plot_xg, plot_yg)
        scattered_field = np.zeros_like(incident_field)
        return FDTDSolution(
            solver=solver,
            xg=plot_xg,
            yg=plot_yg,
            incident_field=incident_field,
            total_field=incident_field.copy(),
            scattered_field=scattered_field,
            dx=dx,
            dy=dy,
            dt=dt,
            steps=0,
            sample_steps=0,
            points_per_wavelength=float(wavelength / min(dx, dy)),
            absorber_cells=config.absorber_cells,
        )

    u_prev = np.zeros((ny_sim, nx_sim), dtype=np.float64)
    u_curr = np.zeros_like(u_prev)
    phasor_acc = np.zeros_like(u_prev, dtype=np.complex128)

    lap_scale_x = 1.0 / dx**2
    lap_scale_y = 1.0 / dy**2

    for step in range(total_steps):
        t_prev = max(step - 1, 0) * dt
        t_curr = step * dt
        ramp_prev = np.sin(0.5 * np.pi * min(1.0, max(step - 1, 0) / ramp_steps)) ** 2
        ramp_curr = np.sin(0.5 * np.pi * min(1.0, step / ramp_steps)) ** 2
        lap = np.zeros_like(u_curr)
        boundary_prev = -ramp_prev * np.cos(omega * t_prev - phase)
        boundary_curr = -ramp_curr * np.cos(omega * t_curr - phase)
        u_prev[reflector_mask] = boundary_prev[reflector_mask]
        u_curr[reflector_mask] = boundary_curr[reflector_mask]
        lap[1:-1, 1:-1] = (
            (u_curr[1:-1, 2:] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) * lap_scale_x
            + (u_curr[2:, 1:-1] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]) * lap_scale_y
        )

        sigma_dt_half = 0.5 * sigma * dt
        u_next = (
            2.0 * u_curr
            - (1.0 - sigma_dt_half) * u_prev
            + (dt**2) * lap
        ) / (1.0 + sigma_dt_half)

        t_next = (step + 1) * dt
        ramp = np.sin(0.5 * np.pi * min(1.0, (step + 1) / ramp_steps)) ** 2
        boundary_next = -ramp * np.cos(omega * t_next - phase)
        u_next[0, :] = 0.0
        u_next[-1, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[reflector_mask] = boundary_next[reflector_mask]

        if step >= settle_steps:
            phasor_acc += u_curr * np.exp(-1j * omega * t_curr)

        u_prev, u_curr = u_curr, u_next

    scattered_phasor_sim = 2.0 * phasor_acc / sample_steps
    plot_xs = np.linspace(config.x_min, config.x_max, config.nx)
    plot_ys = np.linspace(config.y_min, config.y_max, config.ny)
    plot_xg, plot_yg = np.meshgrid(plot_xs, plot_ys)
    incident_field = solver.incident.field(plot_xg, plot_yg)
    scattered_field = _bilinear_resample(xs, ys, scattered_phasor_sim, plot_xg, plot_yg)
    total_field = incident_field + scattered_field

    return FDTDSolution(
        solver=solver,
        xg=plot_xg,
        yg=plot_yg,
        incident_field=incident_field,
        total_field=total_field,
        scattered_field=scattered_field,
        dx=dx,
        dy=dy,
        dt=dt,
        steps=total_steps,
        sample_steps=sample_steps,
        points_per_wavelength=float(wavelength / min(dx, dy)),
        absorber_cells=config.absorber_cells,
    )
