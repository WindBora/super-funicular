"""Regenerate the publication figures and numerical revision results.

The model is a unit-half-length PEC strip

    x(t) = L t,  y(t) = h cos(pi P t),  -1 <= t <= 1,

with ``L=1`` and ``P=5``.  It is represented by :class:`SinusoidalStrip`
using total length ``2L``, spatial frequency ``P/(2L)``, and phase ``pi/2``.
All computations use a unit-amplitude plane wave incident along ``+y``.

Running this file writes eleven standalone PDF figures plus the current
manuscript's combined ``fig2_verification.pdf`` and
``fig3_field_pattern.pdf``, ``revision_results.csv``, and the LaTeX macros in
``revision_results.tex``.  Pass ``--build`` to run two ``pdflatex`` passes
after the numerical products have been validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
import scipy
from scipy.fft import dct


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src2.geometry import SinusoidalStrip
from src2.solver import (
    DifferentiatedNystromSolver,
    MultiReflectorMAR,
    MultiReflectorMoM,
    PlaneWave,
    differential_scattering_cross_section,
    total_scattering_cross_section,
)


# Publication configuration.  Lengths are expressed in units of L.
L = 1.0
P = 5
BETA_RAD = np.pi / 2.0
H_REPRESENTATIVE = 0.10
KL_REPRESENTATIVE = 20.0
N_PRODUCTION = 512
N_REFERENCE = 800
N_CONVERGENCE = (32, 48, 64, 96, 128, 192, 256, 384, 512)
N_ANGLES = 4096
THETA_SAMPLES = 4096
MAR_MODES = 256
MAR_PROJECTION_ORDER = 2048
MAR_FIELD_ORDER = 4096
MAR_RESIDUAL_ORDER = 2 * MAR_MODES + 1
MAR_DOUBLED_MODES = 512
MAR_DOUBLED_PROJECTION_ORDER = 4096
MAR_SOURCE_DOIS = "10.1109/74.775246; 10.1002/2016RS006044"
FLAT_KL = np.linspace(0.25, 20.0, 41)
POLAR_DB_MIN = -30.0
POLAR_DB_MAX = 15.0

FIGURE_PATHS = {
    "geometry": HERE / "fig1_geometry.pdf",
    "convergence": HERE / "fig2_convergence.pdf",
    "flat_validation": HERE / "fig3_flat_validation.pdf",
    "near_field": HERE / "fig4_near_field.pdf",
    "representative_polar": HERE / "fig5_representative_polar.pdf",
    "height_flat": HERE / "fig6_height_flat.pdf",
    "height_005": HERE / "fig7_height_005.pdf",
    "height_010": HERE / "fig8_height_010.pdf",
    "frequency_12": HERE / "fig9_frequency_12.pdf",
    "frequency_16": HERE / "fig10_frequency_16.pdf",
    "frequency_20": HERE / "fig11_frequency_20.pdf",
}
CSV_PATH = HERE / "revision_results.csv"
TEX_PATH = HERE / "revision_results.tex"
MANIFEST_PATH = HERE / "publication_manifest.json"
# The current five-figure manuscript still includes this combined result.
# Keep it regenerable while the standalone publication figures above remain
# available for the fully split layout.
CURRENT_MANUSCRIPT_VERIFICATION_PATH = HERE / "fig2_verification.pdf"
CURRENT_MANUSCRIPT_FIELD_PATTERN_PATH = HERE / "fig3_field_pattern.pdf"

# Match the physical polar-axes diameter used by the three-panel polar figures
# in the current manuscript (Figs. 4 and 5), rather than allowing the two-panel
# GridSpec to enlarge the representative pattern.
COMPOSITE_POLAR_DIAMETER_INCHES = 1.73

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222",
    "gray": "#777777",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        # IEEE artwork should use a compact Times-compatible serif face.  STIX
        # supplies a metrically compatible fallback and matching math glyphs.
        "font.serif": ["Times New Roman", "STIXGeneral", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "lines.linewidth": 1.25,
        "axes.linewidth": 0.7,
        "axes.unicode_minus": False,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.28,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,
    }
)

POLAR_QUANTITY_LABEL = (
    r"$10\log_{10}[|\Phi_{\rm sc}(\varphi)|^2/(kL)]$ (dB)"
)
POLAR_COMMON_SCALE_LABEL = (
    POLAR_QUANTITY_LABEL + r"; common radial scale: $-30$ to $+15$ dB"
)


def log(message: str, started: float) -> None:
    """Print a timestamped progress message and flush it immediately."""

    print(f"[{time.perf_counter() - started:7.1f} s] {message}", flush=True)


def make_curve(h_over_l: float) -> SinusoidalStrip:
    """Return ``x=Lt, y=h cos(pi P t)`` using the shared geometry class."""

    return SinusoidalStrip(
        x_center=0.0,
        y_base=0.0,
        length=2.0 * L,
        amplitude=float(h_over_l) * L,
        frequency=P / (2.0 * L),
        phase_rad=np.pi / 2.0,
    )


def make_incident(k_l: float) -> PlaneWave:
    """Return the unit plane wave for a specified dimensionless ``kL``."""

    return PlaneWave(k=float(k_l) / L, beta_rad=BETA_RAD)


def floquet_reference_angles(
    normalized_spacing: float,
    max_order: int = 8,
) -> tuple[list[int], list[float], list[float]]:
    """Return normal-incidence upper/lower Floquet reference angles in degrees.

    ``normalized_spacing`` is the tangential increment multiplying the order
    (``pi*P/(kL)`` for the revised profile).  The helper is retained for the
    repository's time-convention regression tests and uses the strict
    propagating condition requested in the manuscript.
    """

    orders: list[int] = []
    upper: list[float] = []
    lower: list[float] = []
    for order in range(-max_order, max_order + 1):
        direction_cosine = order * float(normalized_spacing)
        if abs(direction_cosine) < 1.0:
            angle = float(
                np.degrees(np.arccos(np.clip(direction_cosine, -1.0, 1.0)))
            )
            orders.append(order)
            upper.append(angle)
            lower.append(360.0 - angle)
    return orders, upper, lower


@dataclass
class SolutionCache:
    """Cache canonical production solutions reused by several figures."""

    values: dict[tuple[float, float, int], Any]

    def __init__(self) -> None:
        self.values = {}

    def solve(self, h_over_l: float, k_l: float, n: int) -> Any:
        key = (float(h_over_l), float(k_l), int(n))
        if key not in self.values:
            self.values[key] = DifferentiatedNystromSolver(
                reflectors=[make_curve(h_over_l)],
                incident=make_incident(k_l),
                n=n,
            ).solve()
        return self.values[key]


def chebyshev_interpolate(values: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Interpolate samples at first-kind Gauss-Chebyshev nodes."""

    samples = np.asarray(values, dtype=np.complex128)
    coefficients = dct(samples, type=2) / samples.size
    coefficients[0] *= 0.5
    return np.polynomial.chebyshev.chebval(t_eval, coefficients)


def weighted_density_error(solution: Any, reference: np.ndarray, theta: np.ndarray) -> float:
    """Return the relative weighted L2 error of the smooth edge density.

    With ``t=cos(theta)``, the Chebyshev weight ``dt/sqrt(1-t^2)`` is exactly
    ``dtheta``.  The midpoint theta grid therefore gives an equal-weight norm.
    ``v_nodes`` carries an implementation pi scale, removed here for clarity;
    that common factor would cancel in the relative error.
    """

    density = chebyshev_interpolate(solution.v_nodes[0] / np.pi, np.cos(theta))
    numerator = np.sum(np.abs(density - reference) ** 2)
    denominator = np.sum(np.abs(reference) ** 2)
    return float(np.sqrt(numerator / denominator))


def pattern_data(solution: Any, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return far-field pattern, differential TSCS/L in dB, and total TSCS."""

    pattern = solution.far_field_pattern(phi, total=False)
    differential_over_l = differential_scattering_cross_section(
        phi, pattern, solution.solver.k
    ) / L
    with np.errstate(divide="ignore"):
        differential_db = 10.0 * np.log10(np.maximum(differential_over_l, 1.0e-300))
    total = total_scattering_cross_section(phi, pattern, solution.solver.k)
    return pattern, differential_db, total


def total_field_chunked(solution: Any, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Evaluate a large near-field grid without allocating a huge kernel matrix."""

    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    result = np.empty(x_flat.size, dtype=np.complex128)
    chunk = 2048
    for start in range(0, x_flat.size, chunk):
        stop = min(start + chunk, x_flat.size)
        result[start:stop] = solution.near_field(
            x_flat[start:stop], y_flat[start:stop], total=True
        )
    return result.reshape(x_grid.shape)


def panel_label(ax: Any, label: str, *, outside: bool = False) -> None:
    """Place a consistent panel identifier inside or just above an axes."""

    x_position, y_position = ((0.01, 1.02) if outside else (0.015, 0.975))
    ax.text(
        x_position,
        y_position,
        label,
        transform=ax.transAxes,
        ha="left",
        # For outside labels, bottom alignment keeps the complete glyph above
        # the top spine instead of letting it descend into the plotted data.
        va="bottom" if outside else "top",
        fontweight="bold",
        fontsize=8.0,
        zorder=20,
    )


def configure_polar(ax: Any) -> None:
    """Apply the common absolute differential-TSCS polar scale."""

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(
        [0, 90, 180, 270],
        labels=[r"$0^\circ$", r"$90^\circ$", r"$180^\circ$", r"$270^\circ$"],
    )
    ax.set_rlim(POLAR_DB_MIN, POLAR_DB_MAX)
    ax.set_rticks([-30, -15, 0, 15])
    ax.set_rlabel_position(25)
    ax.grid(True)


def save_figure(fig: Any, path: Path) -> None:
    """Write a tightly cropped vector PDF and release its memory."""

    # Matplotlib otherwise inserts the wall-clock creation time, which makes
    # byte-identical figures acquire different SHA-256 hashes on every run.
    fig.savefig(
        path,
        format="pdf",
        # Vector artists remain vector.  This controls only deliberately
        # rasterized dense meshes, keeping them print-sharp without PDF seam
        # artifacts between adjacent pcolormesh cells.
        dpi=600,
        metadata={
            "Creator": "ukraine_microwave_week/generate_figures.py",
            "Producer": f"Matplotlib {matplotlib.__version__}",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def figure_geometry() -> None:
    """Create the geometry and angular-convention schematic."""

    curve = make_curve(H_REPRESENTATIVE)
    t = np.linspace(-1.0, 1.0, 1601)
    x, y = curve.coords(t)

    fig, ax = plt.subplots(figsize=(3.45, 2.25), constrained_layout=True)
    ax.plot(x / L, y / L, color=COLORS["blue"], lw=1.7, zorder=3)
    ax.plot([-1.0, 1.0], [0.0, 0.0], "o", ms=2.8, color=COLORS["black"])

    # Use a tick-ended dimension away from the centerline.  The former short
    # double-headed arrow collapsed into a dot at publication scale.
    height_dimension_x = 0.14
    ax.vlines(
        height_dimension_x,
        0.0,
        H_REPRESENTATIVE,
        colors=COLORS["vermillion"],
        lw=0.75,
        zorder=4,
    )
    ax.hlines(
        [0.0, H_REPRESENTATIVE],
        height_dimension_x - 0.025,
        height_dimension_x + 0.025,
        colors=COLORS["vermillion"],
        lw=0.75,
        zorder=4,
    )
    ax.text(
        height_dimension_x + 0.04,
        0.5 * H_REPRESENTATIVE,
        r"$h$",
        color=COLORS["vermillion"],
        ha="left",
        va="center",
    )

    # The schematic shows the general incidence convention beta; the numerical
    # figures below specialize it to normal incidence, beta=pi/2.
    incidence_origin = np.array([-0.94, -0.225])
    incidence_angle = np.deg2rad(48.0)
    incidence_length = 0.30
    incidence_end = incidence_origin + incidence_length * np.array(
        [np.cos(incidence_angle), np.sin(incidence_angle)]
    )
    ax.annotate(
        "",
        xy=incidence_end,
        xytext=incidence_origin,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLORS["green"]},
    )
    ax.plot(
        [incidence_origin[0], incidence_origin[0] + 0.23],
        [incidence_origin[1], incidence_origin[1]],
        color=COLORS["gray"],
        lw=0.55,
        ls=":",
    )
    ax.add_patch(
        Arc(
            incidence_origin,
            0.22,
            0.22,
            theta1=0.0,
            theta2=np.rad2deg(incidence_angle),
            color=COLORS["green"],
            lw=0.75,
        )
    )
    ax.text(-0.78, -0.145, r"$\beta$", color=COLORS["green"], ha="center")
    ax.text(-0.67, -0.18, r"$\mathbf{k}_{\rm i}$", color=COLORS["green"], va="center")

    ray_origin = np.array([0.40, H_REPRESENTATIVE])
    ray_angle = np.deg2rad(38.0)
    ray_length = 0.29
    ray_end = ray_origin + ray_length * np.array(
        [np.cos(ray_angle), np.sin(ray_angle)]
    )
    ax.plot(
        [ray_origin[0], ray_origin[0] + 0.24],
        [ray_origin[1], ray_origin[1]],
        color=COLORS["gray"],
        lw=0.55,
        ls=":",
    )
    ax.annotate(
        "",
        xy=ray_end,
        xytext=ray_origin,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLORS["purple"]},
    )
    ax.add_patch(
        Arc(
            ray_origin,
            0.24,
            0.24,
            theta1=0.0,
            theta2=np.rad2deg(ray_angle),
            color=COLORS["purple"],
            lw=0.8,
        )
    )
    ax.text(0.54, 0.135, r"$\varphi$", color=COLORS["purple"])

    ax.text(
        0.98,
        0.97,
        r"$x=Lt,\quad y=h\cos(\pi Pt),\quad P=5$",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.text(-1.02, 0.075, r"$-L$", ha="center", va="bottom")
    ax.text(1.02, 0.075, r"$+L$", ha="center", va="bottom")
    ax.set_xlim(-1.16, 1.16)
    # The removed 2L dimension occupied the former lower margin.  Retain enough
    # room for the incidence convention while tightening the unused whitespace.
    ax.set_ylim(-0.29, 0.39)
    ax.set_aspect("equal", adjustable="box")

    # The pale symmetry and mean lines are the actual coordinate axes.  Move
    # the left/bottom spines to x=0 and y=0 instead of drawing a second pair of
    # frame axes along the plot boundary.
    axis_color = "0.82"
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["left"].set_position(("data", 0.0))
    for spine_name in ("bottom", "left"):
        ax.spines[spine_name].set_color(axis_color)
        ax.spines[spine_name].set_linewidth(0.65)
        ax.spines[spine_name].set_zorder(1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["-1.0", "-0.5", "0", "0.5", "1.0"])
    ax.set_yticks([-0.2, 0.0, 0.2])
    ax.set_yticklabels(["-0.2", "", "0.2"])
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        color=axis_color,
        labelcolor=COLORS["black"],
        width=0.55,
        length=2.5,
        pad=2.0,
    )
    axis_arrow = {
        "arrowstyle": "-|>",
        "color": axis_color,
        "lw": 0.65,
        "mutation_scale": 8.0,
        "shrinkA": 0.0,
        "shrinkB": 0.0,
    }
    ax.annotate(
        "",
        xy=(1.155, 0.0),
        xytext=(1.07, 0.0),
        arrowprops=axis_arrow,
        annotation_clip=False,
        zorder=1,
    )
    ax.annotate(
        "",
        xy=(0.0, 0.385),
        xytext=(0.0, 0.31),
        arrowprops=axis_arrow,
        annotation_clip=False,
        zorder=1,
    )
    ax.annotate(
        r"$x/L$",
        xy=(1.0, 0.0),
        xycoords=("axes fraction", "data"),
        xytext=(-1.0, -12.0),
        textcoords="offset points",
        ha="right",
        va="top",
    )
    ax.annotate(
        r"$y/L$",
        xy=(0.0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(7.0, -1.0),
        textcoords="offset points",
        ha="left",
        va="top",
        rotation=90,
    )
    ax.grid(False)
    save_figure(fig, FIGURE_PATHS["geometry"])


def compute_convergence(cache: SolutionCache, started: float) -> dict[int, float]:
    """Compute the nine-point density convergence sequence against N=800."""

    theta = (np.arange(THETA_SAMPLES, dtype=float) + 0.5) * np.pi / THETA_SAMPLES
    log(f"convergence reference N={N_REFERENCE}", started)
    reference_solution = cache.solve(H_REPRESENTATIVE, KL_REPRESENTATIVE, N_REFERENCE)
    reference = chebyshev_interpolate(
        reference_solution.v_nodes[0] / np.pi, np.cos(theta)
    )

    errors: dict[int, float] = {}
    for n in N_CONVERGENCE:
        solution = cache.solve(H_REPRESENTATIVE, KL_REPRESENTATIVE, n)
        errors[n] = weighted_density_error(solution, reference, theta)
        log(f"convergence N={n:3d}: relative weighted L2={errors[n]:.6e}", started)
    return errors


def compute_flat_sweep(
    phi: np.ndarray,
    started: float,
) -> dict[str, np.ndarray]:
    """Compute the three-backend flat-strip TSCS verification sweep."""

    backend_specs: tuple[tuple[str, Callable[..., Any], int], ...] = (
        ("Differentiated Nystrom", DifferentiatedNystromSolver, 256),
        ("MAR", MultiReflectorMAR, MAR_MODES),
        ("MoM", MultiReflectorMoM, 192),
    )
    ratios = {name: np.empty(FLAT_KL.size, dtype=float) for name, _, _ in backend_specs}
    flat_curve = make_curve(0.0)

    log("flat-strip TSCS sweep: 41 frequencies x 3 backends", started)
    for index, k_l in enumerate(FLAT_KL):
        for name, solver_type, n in backend_specs:
            solver_kwargs: dict[str, Any] = {}
            if solver_type is MultiReflectorMAR:
                solver_kwargs["quadrature_order"] = MAR_PROJECTION_ORDER
                solver_kwargs["field_order"] = MAR_FIELD_ORDER
                solver_kwargs["residual_order"] = MAR_RESIDUAL_ORDER
            solution = solver_type(
                reflectors=[flat_curve],
                incident=make_incident(k_l),
                n=n,
                **solver_kwargs,
            ).solve()
            pattern = solution.far_field_pattern(phi, total=False)
            sigma = total_scattering_cross_section(phi, pattern, solution.solver.k)
            ratios[name][index] = sigma / (4.0 * L)
        if index == 0 or (index + 1) % 5 == 0 or index + 1 == FLAT_KL.size:
            log(
                f"flat sweep {index + 1:2d}/{FLAT_KL.size}: kL={k_l:5.2f}",
                started,
            )
    return ratios


def compute_backend_order_checks(
    phi: np.ndarray,
    flat_ratios: dict[str, np.ndarray],
    started: float,
) -> dict[str, dict[str, float | int]]:
    """Check flat-strip backend convergence at the demanding ``kL=20`` point.

    For MAR, modal truncation and compact-kernel projection are varied
    separately so that cancellation between the two errors cannot masquerade
    as convergence.  Both TSCS and the full complex far-field pattern are
    compared on the independent publication field grid.
    """

    backend_specs: tuple[tuple[str, Callable[..., Any], int, int], ...] = (
        ("Differentiated Nystrom", DifferentiatedNystromSolver, 256, 512),
        ("MAR", MultiReflectorMAR, MAR_MODES, MAR_DOUBLED_MODES),
        ("MoM", MultiReflectorMoM, 192, 384),
    )
    checks: dict[str, dict[str, float | int]] = {}
    flat_curve = make_curve(0.0)
    for name, solver_type, base_n, doubled_n in backend_specs:
        if solver_type is MultiReflectorMAR:
            base_solution = solver_type(
                reflectors=[flat_curve],
                incident=make_incident(KL_REPRESENTATIVE),
                n=base_n,
                quadrature_order=MAR_PROJECTION_ORDER,
                field_order=MAR_FIELD_ORDER,
                residual_order=MAR_RESIDUAL_ORDER,
            ).solve()
            mode_solution = solver_type(
                reflectors=[flat_curve],
                incident=make_incident(KL_REPRESENTATIVE),
                n=doubled_n,
                quadrature_order=MAR_PROJECTION_ORDER,
                field_order=MAR_FIELD_ORDER,
                residual_order=MAR_RESIDUAL_ORDER,
            ).solve()
            projection_solution = solver_type(
                reflectors=[flat_curve],
                incident=make_incident(KL_REPRESENTATIVE),
                n=base_n,
                quadrature_order=MAR_DOUBLED_PROJECTION_ORDER,
                field_order=MAR_FIELD_ORDER,
                residual_order=MAR_RESIDUAL_ORDER,
            ).solve()
            base_pattern = base_solution.far_field_pattern(phi, total=False)
            mode_pattern = mode_solution.far_field_pattern(phi, total=False)
            projection_pattern = projection_solution.far_field_pattern(
                phi, total=False
            )

            def normalized_tscs(solution: Any, pattern: np.ndarray) -> float:
                return total_scattering_cross_section(
                    phi, pattern, solution.solver.k
                ) / (4.0 * L)

            base_ratio = normalized_tscs(base_solution, base_pattern)
            mode_ratio = normalized_tscs(mode_solution, mode_pattern)
            projection_ratio = normalized_tscs(
                projection_solution, projection_pattern
            )
            mode_tscs_change = abs(mode_ratio - base_ratio) / abs(mode_ratio)
            mode_pattern_change = float(
                np.linalg.norm(mode_pattern - base_pattern)
                / np.linalg.norm(mode_pattern)
            )
            projection_tscs_change = (
                abs(projection_ratio - base_ratio) / abs(projection_ratio)
            )
            projection_pattern_change = float(
                np.linalg.norm(projection_pattern - base_pattern)
                / np.linalg.norm(projection_pattern)
            )
            checks[name] = {
                "base_n": base_n,
                "doubled_n": doubled_n,
                "base_ratio": base_ratio,
                "doubled_ratio": mode_ratio,
                "relative_change": mode_tscs_change,
                "pattern_change": mode_pattern_change,
                "base_projection_order": MAR_PROJECTION_ORDER,
                "doubled_projection_order": MAR_DOUBLED_PROJECTION_ORDER,
                "projection_ratio": projection_ratio,
                "projection_relative_change": projection_tscs_change,
                "projection_pattern_change": projection_pattern_change,
                "projection_residual": float(
                    projection_solution.boundary_residual_max
                ),
            }
            log(
                "separate MAR convergence: "
                f"modes {base_n}->{doubled_n} at Q={MAR_PROJECTION_ORDER}, "
                f"TSCS={mode_tscs_change:.3e}, pattern={mode_pattern_change:.3e}; "
                f"Q={MAR_PROJECTION_ORDER}->{MAR_DOUBLED_PROJECTION_ORDER} "
                f"at {base_n} modes, TSCS={projection_tscs_change:.3e}, "
                f"pattern={projection_pattern_change:.3e}, "
                f"residual={projection_solution.boundary_residual_max:.3e}",
                started,
            )
            continue

        solver_kwargs: dict[str, Any] = {}
        solution = solver_type(
            reflectors=[flat_curve],
            incident=make_incident(KL_REPRESENTATIVE),
            n=doubled_n,
            **solver_kwargs,
        ).solve()
        pattern = solution.far_field_pattern(phi, total=False)
        doubled_ratio = total_scattering_cross_section(
            phi, pattern, solution.solver.k
        ) / (4.0 * L)
        base_ratio = float(flat_ratios[name][-1])
        relative_change = abs(doubled_ratio - base_ratio) / abs(doubled_ratio)
        checks[name] = {
            "base_n": base_n,
            "doubled_n": doubled_n,
            "base_ratio": base_ratio,
            "doubled_ratio": doubled_ratio,
            "relative_change": relative_change,
        }
        log(
            f"order doubling {name}: N={base_n}->{doubled_n}, "
            f"relative TSCS change={relative_change:.6e}",
            started,
        )
    return checks


def draw_convergence(ax: Any, errors: dict[int, float]) -> None:
    """Draw the differentiated-Nystrom convergence result on one axes."""

    n_values = np.asarray(N_CONVERGENCE)
    error_values = np.asarray([errors[n] for n in N_CONVERGENCE])
    ax.loglog(
        n_values,
        error_values,
        "o-",
        color=COLORS["blue"],
        ms=3.4,
        mfc="white",
        mew=0.8,
    )
    ax.set_xlabel("Nystr\N{LATIN SMALL LETTER O WITH DIAERESIS}m order $N$")
    ax.set_ylabel(r"Relative weighted $L^2$ error")
    ax.set_xticks([32, 64, 128, 256, 512], labels=["32", "64", "128", "256", "512"])
    ax.set_ylim(1.0e-5, 1.0)
    ax.grid(True, which="both")
    exponent = int(np.floor(np.log10(abs(errors[512]))))
    mantissa = errors[512] / (10.0**exponent)
    ax.text(
        0.06,
        0.08,
        rf"$N=512:\;{mantissa:.2f}\times 10^{{{exponent}}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.75", "lw": 0.5},
    )


def figure_convergence(errors: dict[int, float]) -> None:
    """Create the standalone differentiated-Nystrom convergence figure."""

    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    draw_convergence(ax, errors)
    save_figure(fig, FIGURE_PATHS["convergence"])


def draw_flat_validation(ax: Any, ratios: dict[str, np.ndarray]) -> None:
    """Draw the flat-strip independent-backend comparison on one axes."""

    styles = {
        "Differentiated Nystrom": (COLORS["blue"], "-"),
        "MAR": (COLORS["vermillion"], "--"),
        "MoM": (COLORS["green"], ":"),
    }
    labels = {
        "Differentiated Nystrom": "Nystr\N{LATIN SMALL LETTER O WITH DIAERESIS}m ($N=256$)",
        "MAR": "MAR (256 modes)",
        "MoM": "Pulse MoM (192 panels)",
    }
    for name in ("Differentiated Nystrom", "MAR", "MoM"):
        color, linestyle = styles[name]
        values = np.asarray(ratios[name])
        # Plot the complete computed path.  The fixed ordinate clips only the
        # off-scale part of the kL=0.25-to-0.74375 segment, preserving a visible
        # and truthful connection into the displayed range.
        ax.plot(
            FLAT_KL,
            values,
            color=color,
            ls=linestyle,
            label=labels[name],
        )
    ax.axhline(
        1.0,
        color=COLORS["black"],
        lw=0.9,
        ls="-.",
        label="Geometrical optics (GO)",
    )
    ax.set_xlabel(r"$kL$")
    ax.set_ylabel(r"$\sigma/(4L)$")
    ax.set_xlim(0.0, 20.0)
    ax.set_xticks(np.arange(0.0, 20.1, 2.5))
    # Suppress the isolated low-frequency excursion above 1.1 and stretch the
    # near-unity range that carries the useful cross-backend comparison.
    ax.set_ylim(0.98, 1.10)
    ax.set_yticks(np.arange(0.98, 1.101, 0.02))
    ax.grid(True)
    ax.legend(loc="upper right", ncol=1, handlelength=2.3, borderaxespad=0.5)


def figure_flat_validation(ratios: dict[str, np.ndarray]) -> None:
    """Create the standalone flat-strip independent-backend comparison."""

    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    draw_flat_validation(ax, ratios)
    save_figure(fig, FIGURE_PATHS["flat_validation"])


def figure_verification(
    errors: dict[int, float], ratios: dict[str, np.ndarray]
) -> None:
    """Create the current manuscript's two-panel verification Figure 2."""

    fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.72), constrained_layout=True)
    draw_convergence(axes[0], errors)
    panel_label(axes[0], "(a)", outside=True)
    draw_flat_validation(axes[1], ratios)
    panel_label(axes[1], "(b)", outside=True)
    save_figure(fig, CURRENT_MANUSCRIPT_VERIFICATION_PATH)


def compute_publication_cases(
    cache: SolutionCache,
    phi: np.ndarray,
    started: float,
) -> tuple[dict[float, dict[str, Any]], dict[float, dict[str, Any]]]:
    """Solve the amplitude and frequency cases at production order N=512."""

    amplitudes: dict[float, dict[str, Any]] = {}
    for h_over_l in (0.0, 0.05, 0.10):
        solution = cache.solve(h_over_l, KL_REPRESENTATIVE, N_PRODUCTION)
        pattern, differential_db, total = pattern_data(solution, phi)
        amplitudes[h_over_l] = {
            "solution": solution,
            "pattern": pattern,
            "differential_db": differential_db,
            "sigma": total,
        }
        log(
            f"production h/L={h_over_l:.2f}, kL=20: sigma/(4L)={total / (4.0 * L):.8f}",
            started,
        )

    frequencies: dict[float, dict[str, Any]] = {}
    for k_l in (12.0, 16.0, 20.0):
        solution = cache.solve(H_REPRESENTATIVE, k_l, N_PRODUCTION)
        pattern, differential_db, total = pattern_data(solution, phi)
        frequencies[k_l] = {
            "solution": solution,
            "pattern": pattern,
            "differential_db": differential_db,
            "sigma": total,
        }
        log(
            f"production h/L=0.10, kL={k_l:g}: sigma/(4L)={total / (4.0 * L):.8f}",
            started,
        )
    return amplitudes, frequencies


def compute_publication_reference_checks(
    cache: SolutionCache,
    phi: np.ndarray,
    amplitudes: dict[float, dict[str, Any]],
    frequencies: dict[float, dict[str, Any]],
    started: float,
) -> list[dict[str, float | str]]:
    """Compare every unique N=512 publication case with an N=800 solution."""

    production: dict[tuple[float, float], tuple[str, dict[str, Any]]] = {}
    for h_over_l, case in amplitudes.items():
        production[(h_over_l, KL_REPRESENTATIVE)] = (
            f"h/L={h_over_l:.2f}, kL=20",
            case,
        )
    for k_l, case in frequencies.items():
        production.setdefault(
            (H_REPRESENTATIVE, k_l),
            (f"h/L=0.10, kL={k_l:g}", case),
        )

    checks: list[dict[str, float | str]] = []
    for (h_over_l, k_l), (label, case) in production.items():
        reference_solution = cache.solve(h_over_l, k_l, N_REFERENCE)
        reference_pattern = reference_solution.far_field_pattern(phi, total=False)
        reference_sigma = total_scattering_cross_section(
            phi, reference_pattern, reference_solution.solver.k
        )
        production_pattern = np.asarray(case["pattern"])
        pattern_change = float(
            np.sqrt(
                np.sum(np.abs(production_pattern - reference_pattern) ** 2)
                / np.sum(np.abs(reference_pattern) ** 2)
            )
        )
        tscs_change = float(abs(case["sigma"] - reference_sigma) / abs(reference_sigma))
        checks.append(
            {
                "label": label,
                "h_over_l": h_over_l,
                "k_l": k_l,
                "tscs_change": tscs_change,
                "pattern_change": pattern_change,
            }
        )
        log(
            f"N=512 vs 800 {label}: TSCS={tscs_change:.3e}, "
            f"full-pattern L2={pattern_change:.3e}",
            started,
        )
    return checks


def compute_mar_publication_checks(
    cache: SolutionCache,
    phi: np.ndarray,
    amplitudes: dict[float, dict[str, Any]],
    frequencies: dict[float, dict[str, Any]],
    started: float,
) -> list[dict[str, float | str | int]]:
    """Cross-check every unique publication case with genuine corrugated MAR."""

    production: dict[tuple[float, float], str] = {
        (h_over_l, KL_REPRESENTATIVE): f"h/L={h_over_l:.2f}, kL=20"
        for h_over_l in amplitudes
    }
    for k_l in frequencies:
        production.setdefault(
            (H_REPRESENTATIVE, k_l),
            f"h/L=0.10, kL={k_l:g}",
        )

    checks: list[dict[str, float | str | int]] = []
    for (h_over_l, k_l), label in production.items():
        reference_solution = cache.solve(h_over_l, k_l, N_REFERENCE)
        reference_pattern = reference_solution.far_field_pattern(phi, total=False)
        reference_sigma = total_scattering_cross_section(
            phi, reference_pattern, reference_solution.solver.k
        )
        mar_solution = MultiReflectorMAR(
            reflectors=[make_curve(h_over_l)],
            incident=make_incident(k_l),
            n=MAR_MODES,
            quadrature_order=MAR_PROJECTION_ORDER,
            field_order=MAR_FIELD_ORDER,
            residual_order=MAR_RESIDUAL_ORDER,
        ).solve()
        mar_pattern = mar_solution.far_field_pattern(phi, total=False)
        mar_sigma = total_scattering_cross_section(
            phi, mar_pattern, mar_solution.solver.k
        )
        pattern_change = float(
            np.linalg.norm(mar_pattern - reference_pattern)
            / np.linalg.norm(reference_pattern)
        )
        tscs_change = float(abs(mar_sigma - reference_sigma) / abs(reference_sigma))
        checks.append(
            {
                "label": label,
                "h_over_l": h_over_l,
                "k_l": k_l,
                "tscs_change": tscs_change,
                "pattern_change": pattern_change,
                "boundary_residual": float(mar_solution.boundary_residual_max),
                "n": MAR_MODES,
                "projection_order": MAR_PROJECTION_ORDER,
            }
        )
        log(
            f"MAR vs N=800 {label}: TSCS={tscs_change:.3e}, "
            f"full-pattern L2={pattern_change:.3e}, "
            f"IE residual={mar_solution.boundary_residual_max:.3e}",
            started,
        )
    return checks


def compute_near_field_plot_data(representative: dict[str, Any]) -> dict[str, Any]:
    """Compute the common near-field data used by standalone/composite plots."""

    solution = representative["solution"]
    xs = np.linspace(-1.30 * L, 1.30 * L, 321)
    ys = np.linspace(-0.62 * L, 0.82 * L, 241)
    x_grid, y_grid = np.meshgrid(xs, ys)
    total_field = total_field_chunked(solution, x_grid, y_grid)
    # Unit incidence makes this exactly |U_tot|^2/|U_inc|^2.
    relative_intensity = np.abs(total_field) ** 2

    curve = make_curve(H_REPRESENTATIVE)
    inside = np.abs(x_grid) <= L
    curve_y = curve.y(np.clip(x_grid / L, -1.0, 1.0))
    strip_mask = inside & (np.abs(y_grid - curve_y) < 0.008 * L)
    relative_intensity[strip_mask | ~np.isfinite(relative_intensity)] = np.nan
    color_max = max(1.0, float(np.nanpercentile(relative_intensity, 99.7)))

    return {
        "xs": xs,
        "ys": ys,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "relative_intensity": relative_intensity,
        "color_max": color_max,
        "curve": curve,
    }


def draw_near_field(
    ax_field: Any,
    plot_data: dict[str, Any],
    *,
    rasterized: bool = False,
) -> Any:
    """Draw the common near-field panel and return its scalar mappable."""

    image = ax_field.pcolormesh(
        plot_data["x_grid"] / L,
        plot_data["y_grid"] / L,
        plot_data["relative_intensity"],
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=plot_data["color_max"],
        rasterized=rasterized,
    )
    t_plot = np.linspace(-1.0, 1.0, 1601)
    x_curve, y_curve = plot_data["curve"].coords(t_plot)
    ax_field.plot(x_curve / L, y_curve / L, color="white", lw=2.2, zorder=5)
    ax_field.plot(x_curve / L, y_curve / L, color=COLORS["black"], lw=0.8, zorder=6)
    ax_field.annotate(
        "",
        xy=(-1.12, -0.32),
        xytext=(-1.12, -0.52),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "white"},
    )
    ax_field.text(-1.06, -0.43, r"$\mathbf{k}_{\rm i}$", color="white", va="center")
    ax_field.set_xlabel(r"$x/L$")
    ax_field.set_ylabel(r"$y/L$")
    ax_field.set_aspect("equal", adjustable="box")
    ax_field.set_xlim(plot_data["xs"][0] / L, plot_data["xs"][-1] / L)
    ax_field.set_ylim(plot_data["ys"][0] / L, plot_data["ys"][-1] / L)
    return image


def figure_near_field(representative: dict[str, Any]) -> None:
    """Create the standalone absolute total near-field figure."""

    plot_data = compute_near_field_plot_data(representative)
    fig, ax_field = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    image = draw_near_field(ax_field, plot_data)
    colorbar = fig.colorbar(image, ax=ax_field, pad=0.02, shrink=0.91)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label(r"$|U_{\rm tot}|^2/|U_{\rm inc}|^2$")
    save_figure(fig, FIGURE_PATHS["near_field"])


def figure_field_pattern(phi: np.ndarray, representative: dict[str, Any]) -> None:
    """Create the current manuscript's two-panel representative Figure 3."""

    plot_data = compute_near_field_plot_data(representative)
    fig = plt.figure(figsize=(7.10, 3.05), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_field = fig.add_subplot(grid[0, 0])
    ax_polar = fig.add_subplot(grid[0, 1], projection="polar")

    # A dense vector pcolormesh can show false hairline seams in PDF viewers.
    # Rasterize only this mesh at save_figure's 600 dpi; curves, axes, labels,
    # colorbar, and the polar panel remain vector.
    image = draw_near_field(ax_field, plot_data, rasterized=True)

    # Anchor the colorbar to the actual heatmap axes.  Because equal aspect
    # shrinks the heatmap inside its GridSpec cell, fig.colorbar(..., shrink=)
    # cannot robustly align the two vertical extents.
    colorbar_ax = ax_field.inset_axes([1.025, 0.0, 0.035, 1.0])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label(r"$|U_{\rm tot}|^2/|U_{\rm inc}|^2$")
    panel_label(ax_field, "(a)", outside=True)

    configure_polar(ax_polar)
    ax_polar.plot(phi, representative["differential_db"], color=COLORS["blue"])
    ax_polar.set_title(POLAR_QUANTITY_LABEL, pad=10)
    panel_label(ax_polar, "(b)", outside=True)

    # Let constrained layout place all labels first, then freeze the layout and
    # set the polar axes to the same 1.73-in diameter as Figs. 4 and 5.
    fig.canvas.draw()
    polar_position = ax_polar.get_position()
    polar_center_x = 0.5 * (polar_position.x0 + polar_position.x1)
    polar_center_y = 0.5 * (polar_position.y0 + polar_position.y1)
    polar_width = COMPOSITE_POLAR_DIAMETER_INCHES / fig.get_figwidth()
    polar_height = COMPOSITE_POLAR_DIAMETER_INCHES / fig.get_figheight()
    fig.set_layout_engine(None)
    ax_polar.set_position(
        [
            polar_center_x - 0.5 * polar_width,
            polar_center_y - 0.5 * polar_height,
            polar_width,
            polar_height,
        ]
    )

    save_figure(fig, CURRENT_MANUSCRIPT_FIELD_PATTERN_PATH)


def figure_representative_polar(phi: np.ndarray, representative: dict[str, Any]) -> None:
    """Create the standalone representative absolute polar pattern."""

    fig, ax_polar = plt.subplots(
        figsize=(2.55, 2.45),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    configure_polar(ax_polar)
    ax_polar.plot(phi, representative["differential_db"], color=COLORS["blue"])
    ax_polar.set_title(POLAR_QUANTITY_LABEL, pad=10)
    save_figure(fig, FIGURE_PATHS["representative_polar"])


def figure_amplitude(phi: np.ndarray, cases: dict[float, dict[str, Any]]) -> None:
    """Create three standalone differential-TSCS height polar figures."""

    styles = (
        (0.0, COLORS["black"], "-", "height_flat"),
        (0.05, COLORS["orange"], "--", "height_005"),
        (0.10, COLORS["blue"], "-", "height_010"),
    )
    for h_over_l, color, linestyle, path_key in styles:
        fig, ax = plt.subplots(
            figsize=(2.55, 2.45),
            subplot_kw={"projection": "polar"},
            constrained_layout=True,
        )
        configure_polar(ax)
        ax.plot(
            phi,
            cases[h_over_l]["differential_db"],
            color=color,
            ls=linestyle,
        )
        ax.set_title(rf"$h/L={h_over_l:.2f}$", pad=7)
        save_figure(fig, FIGURE_PATHS[path_key])


def figure_frequency(phi: np.ndarray, cases: dict[float, dict[str, Any]]) -> None:
    """Create three standalone differential-TSCS frequency polar figures."""

    styles = (
        (12.0, COLORS["green"], ":", "frequency_12"),
        (16.0, COLORS["orange"], "--", "frequency_16"),
        (20.0, COLORS["blue"], "-", "frequency_20"),
    )
    for k_l, color, linestyle, path_key in styles:
        fig, ax = plt.subplots(
            figsize=(2.55, 2.45),
            subplot_kw={"projection": "polar"},
            constrained_layout=True,
        )
        configure_polar(ax)
        ax.plot(
            phi,
            cases[k_l]["differential_db"],
            color=color,
            ls=linestyle,
        )
        ax.set_title(rf"$kL={k_l:g}$", pad=7)
        save_figure(fig, FIGURE_PATHS[path_key])


CSV_FIELDS = (
    "dataset",
    "series",
    "x_name",
    "x_value",
    "y_name",
    "y_value",
    "units",
    "n",
    "projection_order",
    "field_order",
    "residual_target_count",
    "small_argument_threshold",
    "small_argument_terms",
    "panel_quad_order",
    "self_panel_quad_order",
    "angular_samples",
    "source_doi",
    "h_over_L",
    "kL",
    "P",
    "beta_deg",
    "notes",
)


def base_row(**updates: Any) -> dict[str, Any]:
    """Return one consistently shaped result row."""

    row = {field: "" for field in CSV_FIELDS}
    row.update({"P": P, "beta_deg": 90.0})
    row.update(updates)
    return row


def write_csv(
    errors: dict[int, float],
    flat_ratios: dict[str, np.ndarray],
    order_checks: dict[str, dict[str, float | int]],
    amplitudes: dict[float, dict[str, Any]],
    frequencies: dict[float, dict[str, Any]],
    reference_checks: list[dict[str, float | str]],
    mar_checks: list[dict[str, float | str | int]],
    phi: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """Persist scalar results and the one-dimensional data behind every plot."""

    rows: list[dict[str, Any]] = []
    for macro_name, value in metrics.items():
        units = "dimensionless"
        notes = ""
        if macro_name == "BackendMaxDiff":
            notes = "maximum absolute inter-backend spread in sigma/(4L) over 41 kL values"
        elif macro_name == "ConvErrorNFiveTwelve":
            notes = "relative L2_{1/sqrt(1-t^2)} density error against N=800"
        elif macro_name == "FirstOrderCutoff":
            units = "kL"
            notes = "normal-incidence first grating-order threshold pi*P"
        elif macro_name == "FlatOrderDoublingMaxChange":
            notes = "worst relative TSCS change in the three kL=20 backend order-doubling checks"
        elif macro_name == "ProductionMaxTSCSChange":
            notes = "worst relative TSCS change for N=512 versus N=800 over all publication cases"
        elif macro_name == "ProductionMaxPatternChange":
            notes = "worst full-complex-pattern relative L2 change for N=512 versus N=800"
        elif macro_name == "MARCorrugatedMaxTSCSChange":
            notes = "worst genuine-MAR versus N=800 relative TSCS change over publication cases"
        elif macro_name == "MARCorrugatedMaxPatternChange":
            notes = "worst genuine-MAR versus N=800 full-complex-pattern relative L2 change"
        elif macro_name == "MARCorrugatedMaxResidual":
            notes = "worst independently evaluated original-IE maximum absolute MAR residual"
        elif macro_name == "MARModeDoublingTSCSChange":
            notes = "MAR relative TSCS change for 256 to 512 modes at fixed Q=2048"
        elif macro_name == "MARModeDoublingPatternChange":
            notes = "MAR full-complex-pattern relative L2 change for 256 to 512 modes at fixed Q=2048"
        elif macro_name == "MARProjectionDoublingTSCSChange":
            notes = "MAR relative TSCS change for Q=2048 to 4096 at fixed 256 modes"
        elif macro_name == "MARProjectionDoublingPatternChange":
            notes = "MAR full-complex-pattern relative L2 change for Q=2048 to 4096 at fixed 256 modes"
        elif macro_name == "MARProjectionDoubledResidual":
            notes = "original-IE maximum absolute residual for flat-strip MAR at 256 modes and Q=4096"
        elif "TSCS" in macro_name:
            notes = "sigma/(4L)"
        rows.append(
            base_row(
                dataset="metric",
                series=macro_name,
                y_name="value",
                y_value=f"{value:.16g}",
                units=units,
                notes=notes,
            )
        )

    for n, error in errors.items():
        rows.append(
            base_row(
                dataset="density_convergence",
                series="Differentiated Nystrom",
                x_name="N",
                x_value=n,
                y_name="relative_weighted_L2_error",
                y_value=f"{error:.16g}",
                units="dimensionless",
                n=n,
                h_over_L=H_REPRESENTATIVE,
                kL=KL_REPRESENTATIVE,
                notes=f"N_ref={N_REFERENCE}; {THETA_SAMPLES} midpoint theta samples",
            )
        )

    backend_n = {"Differentiated Nystrom": 256, "MAR": 256, "MoM": 192}
    mar_common_metadata = {
        "field_order": MAR_FIELD_ORDER,
        "residual_target_count": MAR_RESIDUAL_ORDER,
        "small_argument_threshold": MultiReflectorMAR.small_argument_threshold,
        "small_argument_terms": MultiReflectorMAR.small_argument_terms,
        "source_doi": MAR_SOURCE_DOIS,
    }
    backend_metadata = {
        "Differentiated Nystrom": {},
        "MAR": {
            "projection_order": MAR_PROJECTION_ORDER,
            **mar_common_metadata,
        },
        "MoM": {
            "panel_quad_order": 12,
            "self_panel_quad_order": 20,
            "source_doi": "10.2528/PIER07122502",
        },
    }
    for name, values in flat_ratios.items():
        for k_l, value in zip(FLAT_KL, values, strict=True):
            rows.append(
                base_row(
                    dataset="flat_tscs_sweep",
                    series=name,
                    x_name="kL",
                    x_value=f"{k_l:.16g}",
                    y_name="sigma_over_4L",
                    y_value=f"{value:.16g}",
                    units="dimensionless",
                    n=backend_n[name],
                    angular_samples=N_ANGLES,
                    h_over_L=0.0,
                    kL=f"{k_l:.16g}",
                    notes=f"{N_ANGLES} endpoint-excluded azimuth samples",
                    **backend_metadata[name],
                )
            )
    for k_l in FLAT_KL:
        rows.append(
            base_row(
                dataset="flat_tscs_sweep",
                series="GO",
                x_name="kL",
                x_value=f"{k_l:.16g}",
                y_name="sigma_over_4L",
                y_value="1",
                units="dimensionless",
                h_over_L=0.0,
                kL=f"{k_l:.16g}",
                notes="geometrical-optics limit sigma=4L",
            )
        )

    for name, check in order_checks.items():
        order_metadata = backend_metadata[name]
        for order_key, ratio_key in (
            ("base_n", "base_ratio"),
            ("doubled_n", "doubled_ratio"),
        ):
            rows.append(
                base_row(
                    dataset="flat_order_doubling",
                    series=name,
                    x_name="N",
                    x_value=check[order_key],
                    y_name="sigma_over_4L",
                    y_value=f"{float(check[ratio_key]):.16g}",
                    units="dimensionless",
                    n=check[order_key],
                    angular_samples=N_ANGLES,
                    h_over_L=0.0,
                    kL=KL_REPRESENTATIVE,
                    notes="flat strip at kL=20",
                    **order_metadata,
                )
            )
        rows.append(
            base_row(
                dataset="flat_order_doubling",
                series=name,
                x_name="N_pair",
                x_value=f"{check['base_n']}->{check['doubled_n']}",
                y_name="relative_tscs_change",
                y_value=f"{float(check['relative_change']):.16g}",
                units="dimensionless",
                angular_samples=N_ANGLES,
                h_over_L=0.0,
                kL=KL_REPRESENTATIVE,
                notes="abs(sigma_N-sigma_2N)/abs(sigma_2N)",
                **order_metadata,
            )
        )

    mar_check = order_checks["MAR"]
    separated_mar_rows = (
        (
            "MAR modes",
            "modes_pair",
            f"{MAR_MODES}->{MAR_DOUBLED_MODES}",
            "relative_tscs_change",
            mar_check["relative_change"],
            MAR_DOUBLED_MODES,
            MAR_PROJECTION_ORDER,
            "fixed compact-kernel projection order Q=2048",
        ),
        (
            "MAR modes",
            "modes_pair",
            f"{MAR_MODES}->{MAR_DOUBLED_MODES}",
            "relative_full_pattern_L2_change",
            mar_check["pattern_change"],
            MAR_DOUBLED_MODES,
            MAR_PROJECTION_ORDER,
            "fixed compact-kernel projection order Q=2048",
        ),
        (
            "MAR projection",
            "projection_pair",
            f"{MAR_PROJECTION_ORDER}->{MAR_DOUBLED_PROJECTION_ORDER}",
            "relative_tscs_change",
            mar_check["projection_relative_change"],
            MAR_MODES,
            MAR_DOUBLED_PROJECTION_ORDER,
            "fixed modal truncation at 256 modes",
        ),
        (
            "MAR projection",
            "projection_pair",
            f"{MAR_PROJECTION_ORDER}->{MAR_DOUBLED_PROJECTION_ORDER}",
            "relative_full_pattern_L2_change",
            mar_check["projection_pattern_change"],
            MAR_MODES,
            MAR_DOUBLED_PROJECTION_ORDER,
            "fixed modal truncation at 256 modes",
        ),
        (
            "MAR projection",
            "projection_order",
            str(MAR_DOUBLED_PROJECTION_ORDER),
            "original_IE_max_abs_residual",
            mar_check["projection_residual"],
            MAR_MODES,
            MAR_DOUBLED_PROJECTION_ORDER,
            "513 independent target points; Q=4096 source quadrature",
        ),
    )
    for (
        series,
        x_name,
        x_value,
        y_name,
        value,
        modes,
        projection_order,
        notes,
    ) in separated_mar_rows:
        rows.append(
            base_row(
                dataset="mar_flat_separated_convergence",
                series=series,
                x_name=x_name,
                x_value=x_value,
                y_name=y_name,
                y_value=f"{float(value):.16g}",
                units="dimensionless",
                n=modes,
                projection_order=projection_order,
                angular_samples=N_ANGLES,
                h_over_L=0.0,
                kL=KL_REPRESENTATIVE,
                notes=notes,
                **mar_common_metadata,
            )
        )

    for check in reference_checks:
        for y_name in ("tscs_change", "pattern_change"):
            rows.append(
                base_row(
                    dataset="production_reference_check",
                    series=check["label"],
                    x_name="N_pair",
                    x_value=f"{N_PRODUCTION}->{N_REFERENCE}",
                    y_name=(
                        "relative_tscs_change"
                        if y_name == "tscs_change"
                        else "relative_full_pattern_L2_change"
                    ),
                    y_value=f"{float(check[y_name]):.16g}",
                    units="dimensionless",
                    n=N_PRODUCTION,
                    angular_samples=N_ANGLES,
                    h_over_L=check["h_over_l"],
                    kL=check["k_l"],
                    notes="N=800 reference; 4096 endpoint-excluded azimuth samples",
                )
            )

    for check in mar_checks:
        for y_name, csv_name in (
            ("tscs_change", "relative_tscs_change"),
            ("pattern_change", "relative_full_pattern_L2_change"),
            ("boundary_residual", "original_IE_max_abs_residual"),
        ):
            rows.append(
                base_row(
                    dataset="mar_publication_crosscheck",
                    series=check["label"],
                    x_name="method_pair",
                    x_value="MAR->Nystrom N=800",
                    y_name=csv_name,
                    y_value=f"{float(check[y_name]):.16g}",
                    units="dimensionless",
                    n=check["n"],
                    projection_order=check["projection_order"],
                    angular_samples=N_ANGLES,
                    h_over_L=check["h_over_l"],
                    kL=check["k_l"],
                    notes="genuine Chebyshev-Galerkin MAR of the original first-kind IE",
                    **mar_common_metadata,
                )
            )

    phi_deg = np.rad2deg(phi)
    for h_over_l, case in amplitudes.items():
        for angle, value in zip(phi_deg, case["differential_db"], strict=True):
            rows.append(
                base_row(
                    dataset="amplitude_polar",
                    series=f"h/L={h_over_l:.2f}",
                    x_name="phi_deg",
                    x_value=f"{angle:.16g}",
                    y_name="differential_tscs_over_L_db",
                    y_value=f"{value:.16g}",
                    units="dB",
                    n=N_PRODUCTION,
                    angular_samples=N_ANGLES,
                    h_over_L=f"{h_over_l:.2f}",
                    kL=KL_REPRESENTATIVE,
                    notes="absolute; not peak-normalized",
                )
            )
    for k_l, case in frequencies.items():
        for angle, value in zip(phi_deg, case["differential_db"], strict=True):
            rows.append(
                base_row(
                    dataset="frequency_polar",
                    series=f"kL={k_l:g}",
                    x_name="phi_deg",
                    x_value=f"{angle:.16g}",
                    y_name="differential_tscs_over_L_db",
                    y_value=f"{value:.16g}",
                    units="dB",
                    n=N_PRODUCTION,
                    angular_samples=N_ANGLES,
                    h_over_L=H_REPRESENTATIVE,
                    kL=f"{k_l:g}",
                    notes="absolute; not peak-normalized",
                )
            )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latex_number(value: float) -> str:
    """Format a finite scalar for robust use in text or math mode."""

    if value == 0.0:
        return r"\ensuremath{0}"
    magnitude = abs(value)
    if 1.0e-3 <= magnitude < 1.0e4:
        return rf"\ensuremath{{{value:.7g}}}"
    exponent = int(np.floor(np.log10(magnitude)))
    mantissa = value / (10.0**exponent)
    return rf"\ensuremath{{{mantissa:.6g}\times 10^{{{exponent}}}}}"


def write_tex(metrics: dict[str, float]) -> None:
    """Write publication macros with deterministic names and precision."""

    comments = (
        "% Generated by generate_figures.py; do not edit by hand.",
        "% TSCS result macros are dimensionless sigma/(4L) ratios.",
        "% BackendMaxDiff is an absolute difference in sigma/(4L).",
        "% FirstOrderCutoff is the dimensionless kL threshold pi*P.",
    )
    lines = [*comments]
    for name, value in metrics.items():
        lines.append(rf"\newcommand{{\{name}}}{{{latex_number(value)}}}")
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one reproducibility artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    """Write deterministic software, solver, source, and artifact metadata."""

    source_paths = (
        ROOT / ".python-version",
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "requirements-publication.txt",
        ROOT / "src2" / "__init__.py",
        ROOT / "src2" / "geometry.py",
        ROOT / "src2" / "numerics.py",
        ROOT / "src2" / "solver.py",
        ROOT / "tests" / "test_mar_solver.py",
        ROOT / "tests" / "test_publication_revision.py",
        HERE / "generate_figures.py",
        HERE / "NOTES.md",
        HERE / "main.tex",
    )
    artifact_paths = (
        *FIGURE_PATHS.values(),
        CURRENT_MANUSCRIPT_VERIFICATION_PATH,
        CURRENT_MANUSCRIPT_FIELD_PATTERN_PATH,
        CSV_PATH,
        TEX_PATH,
    )
    paths = (*source_paths, *artifact_paths)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "cannot write publication manifest; missing: "
            + ", ".join(path.name for path in missing)
        )

    manifest = {
        "license": "MIT",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "publication_configuration": {
            "L": L,
            "P": P,
            "beta_rad": BETA_RAD,
            "n_angles": N_ANGLES,
            "n_production": N_PRODUCTION,
            "n_reference": N_REFERENCE,
            "mar_modes": MAR_MODES,
            "mar_projection_order": MAR_PROJECTION_ORDER,
            "mar_field_order": MAR_FIELD_ORDER,
            "mar_residual_target_count": MAR_RESIDUAL_ORDER,
            "mar_small_argument_threshold": MultiReflectorMAR.small_argument_threshold,
            "mar_small_argument_terms": MultiReflectorMAR.small_argument_terms,
            "mar_doubled_modes": MAR_DOUBLED_MODES,
            "mar_doubled_projection_order": MAR_DOUBLED_PROJECTION_ORDER,
            "mom_panels": 192,
            "mom_regular_panel_gauss_order": 12,
            "mom_self_panel_gauss_order": 20,
        },
        "method_sources": {
            "MAR": "https://doi.org/10.1109/74.775246",
            "MAR_review_2016": "https://doi.org/10.1002/2016RS006044",
            "MoM": "https://doi.org/10.2528/PIER07122502",
            "MoM_foundation": "https://doi.org/10.1109/PROC.1967.5433",
        },
        "sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in paths
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate(
    errors: dict[int, float],
    flat_ratios: dict[str, np.ndarray],
    order_checks: dict[str, dict[str, float | int]],
    reference_checks: list[dict[str, float | str]],
    mar_checks: list[dict[str, float | str | int]],
    metrics: dict[str, float],
) -> list[str]:
    """Return all failed numerical or artifact validation checks."""

    failures: list[str] = []

    if not np.isfinite(errors[512]) or errors[512] > 1.0e-4:
        failures.append(
            f"N=512 weighted L2 error {errors[512]:.3e} exceeds 1.0e-4"
        )
    error_sequence = np.asarray([errors[n] for n in N_CONVERGENCE])
    if np.any(np.diff(error_sequence) >= 0.0):
        failures.append("density convergence is not strictly decreasing over the requested N sequence")
    if abs(metrics["FlatTSCSRatioTwenty"] - 1.0) > 5.0e-3:
        failures.append(
            "flat-strip Nystrom sigma/(4L) at kL=20 differs from GO by more than 0.005"
        )
    if metrics["BackendMaxDiff"] >= 1.0e-3:
        failures.append(
            f"maximum backend sigma/(4L) spread {metrics['BackendMaxDiff']:.3e} is not below 0.001"
        )
    for name, check in order_checks.items():
        change = float(check["relative_change"])
        if not np.isfinite(change) or change >= 1.0e-3:
            failures.append(
                f"{name} order-doubling TSCS change {change:.3e} is not below 0.001"
            )
    mar_limits = (
        ("MARModeDoublingTSCSChange", 1.0e-6),
        ("MARModeDoublingPatternChange", 1.0e-5),
        ("MARProjectionDoublingTSCSChange", 1.0e-6),
        ("MARProjectionDoublingPatternChange", 1.0e-6),
        ("MARProjectionDoubledResidual", 1.0e-7),
    )
    for metric_name, limit in mar_limits:
        value = metrics[metric_name]
        if not np.isfinite(value) or value >= limit:
            failures.append(
                f"{metric_name}={value:.3e} is not below {limit:.0e}"
            )
    for check in reference_checks:
        tscs_change = float(check["tscs_change"])
        pattern_change = float(check["pattern_change"])
        if not np.isfinite(tscs_change) or tscs_change >= 1.0e-3:
            failures.append(
                f"{check['label']} N=512/800 TSCS change {tscs_change:.3e} is not below 0.001"
            )
        if not np.isfinite(pattern_change) or pattern_change >= 1.0e-3:
            failures.append(
                f"{check['label']} N=512/800 pattern L2 change {pattern_change:.3e} is not below 0.001"
            )
    for check in mar_checks:
        tscs_change = float(check["tscs_change"])
        pattern_change = float(check["pattern_change"])
        residual = float(check["boundary_residual"])
        if not np.isfinite(tscs_change) or tscs_change >= 1.0e-3:
            failures.append(
                f"{check['label']} MAR/N=800 TSCS change {tscs_change:.3e} is not below 0.001"
            )
        if not np.isfinite(pattern_change) or pattern_change >= 1.0e-3:
            failures.append(
                f"{check['label']} MAR/N=800 pattern L2 change {pattern_change:.3e} is not below 0.001"
            )
        if not np.isfinite(residual) or residual >= 1.0e-5:
            failures.append(
                f"{check['label']} MAR original-IE residual {residual:.3e} is not below 1e-5"
            )
    all_ratios = np.concatenate(tuple(flat_ratios.values()))
    if not np.all(np.isfinite(all_ratios)) or np.any(all_ratios <= 0.0):
        failures.append("flat-strip sweep contains non-finite or non-positive TSCS values")

    for path in (
        *FIGURE_PATHS.values(),
        CURRENT_MANUSCRIPT_VERIFICATION_PATH,
        CURRENT_MANUSCRIPT_FIELD_PATTERN_PATH,
        CSV_PATH,
        TEX_PATH,
        MANIFEST_PATH,
    ):
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty output: {path.name}")
    return failures


def build_paper(started: float) -> None:
    """Compile the IEEE manuscript twice with pdflatex."""

    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex was not found on PATH")
    command = [executable, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    for pass_number in (1, 2):
        log(f"pdflatex pass {pass_number}/2", started)
        subprocess.run(command, cwd=HERE, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build",
        action="store_true",
        help="run two pdflatex passes after regenerating and validating all outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    phi = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
    cache = SolutionCache()

    log("writing geometry schematic", started)
    figure_geometry()
    errors = compute_convergence(cache, started)
    flat_ratios = compute_flat_sweep(phi, started)
    order_checks = compute_backend_order_checks(phi, flat_ratios, started)
    log("writing convergence and flat-validation figures", started)
    figure_convergence(errors)
    figure_flat_validation(flat_ratios)
    figure_verification(errors, flat_ratios)

    log("solving N=512 publication cases", started)
    amplitudes, frequencies = compute_publication_cases(cache, phi, started)
    reference_checks = compute_publication_reference_checks(
        cache, phi, amplitudes, frequencies, started
    )
    mar_checks = compute_mar_publication_checks(
        cache, phi, amplitudes, frequencies, started
    )
    log("writing field and polar figures", started)
    figure_near_field(amplitudes[H_REPRESENTATIVE])
    figure_representative_polar(phi, amplitudes[H_REPRESENTATIVE])
    figure_field_pattern(phi, amplitudes[H_REPRESENTATIVE])
    figure_amplitude(phi, amplitudes)
    figure_frequency(phi, frequencies)

    backend_stack = np.vstack(
        [
            flat_ratios["Differentiated Nystrom"],
            flat_ratios["MAR"],
            flat_ratios["MoM"],
        ]
    )
    metrics = {
        "ConvErrorNFiveTwelve": errors[512],
        "FlatTSCSRatioTwenty": float(flat_ratios["Differentiated Nystrom"][-1]),
        "BackendMaxDiff": float(np.max(np.ptp(backend_stack, axis=0))),
        "AmpTSCSFlat": amplitudes[0.0]["sigma"] / (4.0 * L),
        "AmpTSCSFive": amplitudes[0.05]["sigma"] / (4.0 * L),
        "AmpTSCSTen": amplitudes[0.10]["sigma"] / (4.0 * L),
        "FreqTSCSTwelve": frequencies[12.0]["sigma"] / (4.0 * L),
        "FreqTSCSSixteen": frequencies[16.0]["sigma"] / (4.0 * L),
        "FreqTSCSTwenty": frequencies[20.0]["sigma"] / (4.0 * L),
        "FirstOrderCutoff": float(np.pi * P),
        "FlatOrderDoublingMaxChange": max(
            float(check["relative_change"]) for check in order_checks.values()
        ),
        "ProductionMaxTSCSChange": max(
            float(check["tscs_change"]) for check in reference_checks
        ),
        "ProductionMaxPatternChange": max(
            float(check["pattern_change"]) for check in reference_checks
        ),
        "MARCorrugatedMaxTSCSChange": max(
            float(check["tscs_change"]) for check in mar_checks
        ),
        "MARCorrugatedMaxPatternChange": max(
            float(check["pattern_change"]) for check in mar_checks
        ),
        "MARCorrugatedMaxResidual": max(
            float(check["boundary_residual"]) for check in mar_checks
        ),
        "MARModeDoublingTSCSChange": float(
            order_checks["MAR"]["relative_change"]
        ),
        "MARModeDoublingPatternChange": float(
            order_checks["MAR"]["pattern_change"]
        ),
        "MARProjectionDoublingTSCSChange": float(
            order_checks["MAR"]["projection_relative_change"]
        ),
        "MARProjectionDoublingPatternChange": float(
            order_checks["MAR"]["projection_pattern_change"]
        ),
        "MARProjectionDoubledResidual": float(
            order_checks["MAR"]["projection_residual"]
        ),
    }
    write_csv(
        errors,
        flat_ratios,
        order_checks,
        amplitudes,
        frequencies,
        reference_checks,
        mar_checks,
        phi,
        metrics,
    )
    write_tex(metrics)
    write_manifest()

    failures = validate(
        errors,
        flat_ratios,
        order_checks,
        reference_checks,
        mar_checks,
        metrics,
    )
    print("\nRevision metrics", flush=True)
    for name, value in metrics.items():
        print(f"  {name:26s} = {value:.10g}", flush=True)
    if failures:
        print("\nVALIDATION FAILURES", flush=True)
        for failure in failures:
            print(f"  FAIL: {failure}", flush=True)
        return 2

    print("\nAll numerical and artifact thresholds passed.", flush=True)
    if args.build:
        build_paper(started)
    log("publication pipeline complete", started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
