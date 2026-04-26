"""Command-line entrypoint for the reflector solver and plotting workflow."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Sequence

import numpy as np

from .fdtd import FDTDConfig, solve_fdtd
from .plotting import (
    plot_comparison_overview,
    plot_solution_overview,
    plot_all_comparison_overview,
    reflector_distance_mask,
)
from .scenes import build_scene, resolved_beta_deg, resolved_plot_window
from .solver import MultiReflectorMAR, MultiReflectorMoM, MultiReflectorPaperMDS


def pattern_directivity(phi: np.ndarray, pattern: np.ndarray) -> tuple[float, float]:
    """Return directivity and peak angle for a sampled far-field pattern."""

    if pattern.size == 0:
        return 0.0, 0.0
    peak_magnitude = float(np.max(np.abs(pattern)))
    if peak_magnitude == 0.0:
        return 0.0, 0.0
    power = (2.0 * np.pi / pattern.size) * np.sum(np.abs(pattern) ** 2)
    if power == 0.0:
        return 0.0, 0.0
    peak_index = int(np.argmax(np.abs(pattern)))
    peak_phi = float(phi[peak_index])
    directivity = float((2.0 * np.pi / power) * (np.abs(pattern[peak_index]) ** 2))
    return directivity, peak_phi


def comparison_metrics(
    reference_field: np.ndarray,
    candidate_field: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return relative L2, max absolute, and max relative error metrics."""

    reference = np.asarray(reference_field)
    candidate = np.asarray(candidate_field)
    diff = candidate - reference
    if mask is not None:
        valid = ~mask
        reference = reference[valid]
        diff = diff[valid]
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    diff_norm = float(np.linalg.norm(diff.reshape(-1)))
    max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    reference_peak = float(np.max(np.abs(reference))) if reference.size else 0.0
    rel_l2 = 0.0 if reference_norm == 0.0 else diff_norm / reference_norm
    max_rel = 0.0 if reference_peak == 0.0 else max_abs_error / reference_peak
    return rel_l2, max_abs_error, max_rel


def compute_far_field_report(args: argparse.Namespace, solver, solution):
    """Evaluate the far-field reporting quantities for a frequency-domain solution."""

    beta_deg = resolved_beta_deg(args)
    pure_free_space_plane_wave = args.scene == "free_space" and args.incident_kind == "plane_wave"
    phi = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    far_field_note = None
    if args.field_kind == "incident":
        pattern = solver.incident.far_field_pattern(phi)
        directivity, peak_phi = pattern_directivity(phi, pattern)
        if args.incident_kind == "plane_wave":
            phi = np.array([], dtype=np.float64)
            pattern = np.array([], dtype=np.complex128)
            directivity = 0.0
            peak_phi = 0.0
            far_field_note = "Far-field plot = skipped for a plane-wave incident field."
    elif pure_free_space_plane_wave:
        phi = np.array([], dtype=np.float64)
        pattern = np.array([], dtype=np.complex128)
        directivity = 0.0
        peak_phi = 0.0
        far_field_note = "Far-field plot = skipped for a free-space plane wave."
    else:
        include_total_far_field = args.field_kind == "total" and args.incident_kind != "plane_wave"
        pattern = solution.far_field_pattern(phi, total=include_total_far_field)
        directivity, peak_phi = pattern_directivity(phi, pattern)
        if args.field_kind == "total" and args.incident_kind == "plane_wave":
            far_field_note = "Far-field/directivity report = scattered field only for plane-wave incidence."
    return beta_deg, phi, pattern, directivity, peak_phi, far_field_note


def build_incident_grid(
    solver,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the exact incident field on a Cartesian grid."""

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    xg, yg = np.meshgrid(xs, ys)
    return xg, yg, solver.incident.field(xg, yg)


def build_solution_grid(
    args: argparse.Namespace,
    solver,
    solution,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the plotted field grid for a frequency-domain solver result."""

    if args.field_kind == "incident":
        return build_incident_grid(
            solver,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            nx=args.nx,
            ny=args.ny,
        )
    return solution.near_field_grid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        nx=args.nx,
        ny=args.ny,
        total=args.field_kind == "total",
    )


def comparison_observable(args: argparse.Namespace, solver, field: np.ndarray) -> np.ndarray:
    """Return the scalar near-field quantity shown in the comparison plots."""

    plane_wave_real_plot = (
        args.near_plot_style == "figure"
        and args.incident_kind == "plane_wave"
        and (args.field_kind == "incident" or solver.num_reflectors == 0)
    )
    if plane_wave_real_plot:
        return np.real(field)
    return np.abs(field) ** 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the solver application.

    Parameters
    ----------
    argv:
        Optional explicit argument list. When omitted, ``argparse`` reads from
        ``sys.argv``.
    """

    parser = argparse.ArgumentParser(
        description="MDS implementation for 2-D E-wave reflector scattering with CSP or plane-wave incidence."
    )
    parser.add_argument(
        "--solver-kind",
        choices=(
            "nystrom",
            "mom",
            "mar",
            "paper_mds",
            "fdtd",
            "compare",
            "compare_mom",
            "compare_mar",
            "compare_paper_mds",
            "compare_all",
        ),
        default="nystrom",
        help="Numerical backend. 'mom' uses pulse-basis method of moments, 'mar' uses the analytically regularized Cauchy-SIE system, and 'paper_mds' assembles the Nosich 2007 Eq. (11) MDS system directly. 'compare' runs Nystrom and FDTD, 'compare_mom' runs Nystrom and MoM, 'compare_mar' runs Nystrom and MAR, 'compare_paper_mds' runs Nystrom and the paper-faithful MDS, and 'compare_all' runs Paper MDS, MoM, and MAR on the same plot grid.",
    )
    parser.add_argument("--n", type=int, default=120, help="Interpolation order.")
    parser.add_argument(
        "--scene",
        choices=(
            "free_space",
            "single_shifted",
            "cassegrain",
            "two_brackets",
            "confocal_elliptic",
            "sinusoidal_strip",
        ),
        default="cassegrain",
        help="Geometry preset. 'free_space' uses no reflector at all, 'cassegrain' uses a parabolic main reflector and a hyperbolic subreflector, 'confocal_elliptic' uses two ellipse arcs with a shared intermediate focus, and 'sinusoidal_strip' builds a bottom sinusoidal reflector strip.",
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
        "--strip-length",
        type=float,
        default=24.0,
        help="End-to-end length of the sinusoidal strip in wavelengths.",
    )
    parser.add_argument(
        "--strip-amplitude",
        type=float,
        default=1.5,
        help="Vertical oscillation amplitude of the sinusoidal strip in wavelengths.",
    )
    parser.add_argument(
        "--strip-frequency",
        type=float,
        default=0.2,
        help="Spatial frequency of the sinusoidal strip in cycles per wavelength unit.",
    )
    parser.add_argument(
        "--strip-base-y",
        type=float,
        default=-10.0,
        help="Baseline y-coordinate of the sinusoidal strip.",
    )
    parser.add_argument(
        "--strip-center-x",
        type=float,
        default=0.0,
        help="Midpoint x-coordinate of the sinusoidal strip.",
    )
    parser.add_argument(
        "--strip-phase-deg",
        type=float,
        default=0.0,
        help="Phase offset of the sinusoidal strip in degrees.",
    )
    parser.add_argument(
        "--strip-feed-x",
        type=float,
        default=None,
        help="Optional CSP x-position for the sinusoidal strip scene. Default is the strip center x.",
    )
    parser.add_argument(
        "--strip-feed-y",
        type=float,
        default=None,
        help="Optional CSP y-position for the sinusoidal strip scene. Default is above the strip.",
    )
    parser.add_argument(
        "--strip-target-x",
        type=float,
        default=None,
        help="Optional x-coordinate on the sinusoidal strip used as the automatic CSP aiming point. Default is the strip midpoint.",
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
        "--ellipse-lower-side",
        choices=("left", "right"),
        default="left",
        help="Which side of the lower bracket-like ellipse is kept in mirror_vertical mode. 'left' makes the lower reflector open to the right.",
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
        "--ellipse-lower-rotation-deg",
        type=float,
        default=None,
        help="Rotation angle of the lower ellipse local frame. The ellipse is defined as (x'/a1)^2 + (y'/a2)^2 = 1 before this rotation.",
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
        help="Lower-ellipse local semi-axis a1 from the paper-style equation (x'/a1)^2 + (y'/a2)^2 = 1. If smaller than --ellipse-lower-b, the code swaps them so a1 >= a2.",
    )
    parser.add_argument(
        "--ellipse-lower-b",
        type=float,
        default=None,
        help="Lower-ellipse local semi-axis a2 from the paper-style equation (x'/a1)^2 + (y'/a2)^2 = 1.",
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
        help="Half-angle span of the retained open ellipse arc in mirror_vertical mode. Values below 90 degrees give a clearly open reflector, while symmetric angles still enforce equal endpoint x-coordinates.",
    )
    parser.add_argument(
        "--incident-kind",
        choices=("csp", "plane_wave"),
        default="csp",
        help="Incident field model. 'plane_wave' uses exp(-i*k*(x*cos(beta)+y*sin(beta))).",
    )
    parser.add_argument("--kb", type=float, default=9.0, help="CSP beam parameter kb. Used only for --incident-kind=csp.")
    parser.add_argument(
        "--beta-deg",
        type=float,
        default=None,
        help="Incident angle in degrees. For --incident-kind=plane_wave this is the B in exp(-i*k*(x*cos(B)+y*sin(B))). Defaults remain scene-dependent.",
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
        choices=("total", "scattered", "incident"),
        default="total",
        help="Field quantity to plot. 'total' shows incident plus scattered field, 'scattered' suppresses the direct incident field, and 'incident' plots only the incident field.",
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
        "--fdtd-ppw",
        type=float,
        default=20.0,
        help="FDTD points per wavelength used on the internal time-domain grid.",
    )
    parser.add_argument(
        "--fdtd-padding-wavelengths",
        type=float,
        default=3.0,
        help="Extra free-space padding around the plotted window for the FDTD simulation box.",
    )
    parser.add_argument(
        "--fdtd-absorber-cells",
        type=int,
        default=24,
        help="Number of sponge-layer cells on each side of the FDTD box.",
    )
    parser.add_argument(
        "--fdtd-ramp-cycles",
        type=float,
        default=3.0,
        help="Number of source cycles used to ramp the FDTD harmonic drive.",
    )
    parser.add_argument(
        "--fdtd-settle-cycles",
        type=float,
        default=10.0,
        help="Number of pre-sampling cycles used to let FDTD transients decay.",
    )
    parser.add_argument(
        "--fdtd-sample-cycles",
        type=float,
        default=12.0,
        help="Number of cycles accumulated to recover the steady-state FDTD phasor.",
    )
    parser.add_argument(
        "--fdtd-courant",
        type=float,
        default=0.45,
        help="Courant factor for the FDTD update. Must be below the 2-D stability limit.",
    )
    parser.add_argument(
        "--fdtd-damping-strength",
        type=float,
        default=3.0,
        help="Strength multiplier for the FDTD sponge layer.",
    )
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
        help="If given, save generated plot images using this filename prefix.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the full solve-and-plot workflow from parsed command-line arguments.

    Parameters
    ----------
    argv:
        Optional explicit argument list. When omitted, ``argparse`` reads from
        ``sys.argv``.
    """

    args = parse_args(argv)
    solver = build_scene(args)
    mom_solver = None
    mar_solver = None
    paper_solver = None
    beta_deg = resolved_beta_deg(args)
    x_min, x_max, y_min, y_max = resolved_plot_window(args)
    nystrom_solution = None
    mom_solution = None
    mar_solution = None
    paper_solution = None
    fdtd_solution = None
    phi = np.array([], dtype=np.float64)
    pattern = np.array([], dtype=np.complex128)
    far_field_note = None
    directivity = 0.0
    peak_phi = 0.0
    mom_phi = np.array([], dtype=np.float64)
    mom_pattern = np.array([], dtype=np.complex128)
    mom_far_field_note = None
    mom_directivity = 0.0
    mom_peak_phi = 0.0
    mar_phi = np.array([], dtype=np.float64)
    mar_pattern = np.array([], dtype=np.complex128)
    mar_far_field_note = None
    mar_directivity = 0.0
    mar_peak_phi = 0.0
    paper_phi = np.array([], dtype=np.float64)
    paper_pattern = np.array([], dtype=np.complex128)
    paper_far_field_note = None
    paper_directivity = 0.0
    paper_peak_phi = 0.0

    if args.solver_kind in {"fdtd", "compare"} and args.incident_kind != "plane_wave":
        raise ValueError(
            "The FDTD backend is currently applicable only for --incident-kind plane_wave."
        )

    if args.solver_kind in {"nystrom", "compare", "compare_mom", "compare_mar", "compare_paper_mds"}:
        nystrom_solution = solver.solve()
        beta_deg, phi, pattern, directivity, peak_phi, far_field_note = compute_far_field_report(
            args, solver, nystrom_solution
        )

    if args.solver_kind in {"mom", "compare_mom", "compare_all"}:
        mom_solver = MultiReflectorMoM(
            reflectors=solver.reflectors,
            incident=solver.incident,
            n=solver.n,
        )
        mom_solution = mom_solver.solve()
        (
            beta_deg,
            mom_phi,
            mom_pattern,
            mom_directivity,
            mom_peak_phi,
            mom_far_field_note,
        ) = compute_far_field_report(args, mom_solver, mom_solution)

    if args.solver_kind in {"mar", "compare_mar", "compare_all"}:
        mar_solver = MultiReflectorMAR(
            reflectors=solver.reflectors,
            incident=solver.incident,
            n=solver.n,
        )
        mar_solution = mar_solver.solve()
        (
            beta_deg,
            mar_phi,
            mar_pattern,
            mar_directivity,
            mar_peak_phi,
            mar_far_field_note,
        ) = compute_far_field_report(args, mar_solver, mar_solution)

    if args.solver_kind in {"paper_mds", "compare_paper_mds", "compare_all"}:
        paper_solver = MultiReflectorPaperMDS(
            reflectors=solver.reflectors,
            incident=solver.incident,
            n=solver.n,
        )
        paper_solution = paper_solver.solve()
        (
            beta_deg,
            paper_phi,
            paper_pattern,
            paper_directivity,
            paper_peak_phi,
            paper_far_field_note,
        ) = compute_far_field_report(args, paper_solver, paper_solution)

    if args.solver_kind in {"fdtd", "compare"}:
        fdtd_solution = solve_fdtd(
            solver,
            FDTDConfig(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                nx=args.nx,
                ny=args.ny,
                points_per_wavelength=args.fdtd_ppw,
                padding_wavelengths=args.fdtd_padding_wavelengths,
                absorber_cells=args.fdtd_absorber_cells,
                ramp_cycles=args.fdtd_ramp_cycles,
                settle_cycles=args.fdtd_settle_cycles,
                sample_cycles=args.fdtd_sample_cycles,
                courant=args.fdtd_courant,
                damping_strength=args.fdtd_damping_strength,
            ),
        )

    print(f"scene = {args.scene}")
    print(f"solver kind = {args.solver_kind}")
    print(f"incident kind = {args.incident_kind}")
    print(f"n = {solver.n}")
    print(f"k = {solver.k:.12g}")
    if args.incident_kind == "csp":
        print(f"kb = {args.kb:.12g}")
        print(f"feed = ({solver.incident.x0:.6f}, {solver.incident.y0:.6f})")
    else:
        print("plane wave = exp(-i*k*(x*cos(beta)+y*sin(beta)))")
    print(f"beta [deg] = {beta_deg:.6f}")
    print(f"field kind = {args.field_kind}")
    if nystrom_solution is not None:
        if args.solver_kind in {"compare_mom", "compare_mar", "compare_paper_mds", "compare_all"}:
            print(f"Nystrom boundary residual max = {nystrom_solution.boundary_residual_max:.6e}")
        else:
            print(f"Boundary residual max = {nystrom_solution.boundary_residual_max:.6e}")
        if pattern.size == 0:
            if args.solver_kind not in {"compare_mom", "compare_mar", "compare_paper_mds", "compare_all"}:
                print("Directivity = not defined for this field selection.")
                print("Peak angle [deg] = not defined for this field selection.")
        else:
            if args.solver_kind in {"compare_mom", "compare_mar", "compare_paper_mds", "compare_all"}:
                print(f"Nystrom directivity = {directivity:.6f}")
                print(f"Nystrom peak angle [deg] = {np.rad2deg(peak_phi):.6f}")
            else:
                print(f"Directivity = {directivity:.6f}")
                print(f"Peak angle [deg] = {np.rad2deg(peak_phi):.6f}")
        if far_field_note is not None:
            print(
                far_field_note
                if args.solver_kind not in {"compare_mom", "compare_mar", "compare_paper_mds", "compare_all"}
                else f"Nystrom: {far_field_note}"
            )
        for reflector_index in range(solver.num_reflectors):
            edge_left_db, edge_right_db = nystrom_solution.edge_illumination_db(
                reflector_index=reflector_index
            )
            print(
                f"{'Nystrom ' if args.solver_kind in {'compare_mom', 'compare_mar', 'compare_paper_mds', 'compare_all'} else ''}Reflector {reflector_index} edge illumination left/right [dB] = "
                f"{edge_left_db:.6f}, {edge_right_db:.6f}"
            )
    if mom_solution is not None:
        if args.solver_kind in {"compare_mom", "compare_all"}:
            print(f"MoM boundary residual max = {mom_solution.boundary_residual_max:.6e}")
            if mom_pattern.size != 0:
                print(f"MoM directivity = {mom_directivity:.6f}")
                print(f"MoM peak angle [deg] = {np.rad2deg(mom_peak_phi):.6f}")
            if mom_far_field_note is not None:
                print(f"MoM: {mom_far_field_note}")
        else:
            print(f"Boundary residual max = {mom_solution.boundary_residual_max:.6e}")
            if mom_pattern.size == 0:
                print("Directivity = not defined for this field selection.")
                print("Peak angle [deg] = not defined for this field selection.")
            else:
                print(f"Directivity = {mom_directivity:.6f}")
                print(f"Peak angle [deg] = {np.rad2deg(mom_peak_phi):.6f}")
            if mom_far_field_note is not None:
                print(mom_far_field_note)
        for reflector_index in range(solver.num_reflectors):
            edge_left_db, edge_right_db = mom_solution.edge_illumination_db(
                reflector_index=reflector_index
            )
            print(
                f"{'MoM ' if args.solver_kind in {'compare_mom', 'compare_all'} else ''}Reflector {reflector_index} edge illumination left/right [dB] = "
                f"{edge_left_db:.6f}, {edge_right_db:.6f}"
            )
    if mar_solution is not None:
        if args.solver_kind in {"compare_mar", "compare_all"}:
            print(f"MAR boundary residual max = {mar_solution.boundary_residual_max:.6e}")
            if mar_pattern.size != 0:
                print(f"MAR directivity = {mar_directivity:.6f}")
                print(f"MAR peak angle [deg] = {np.rad2deg(mar_peak_phi):.6f}")
            if mar_far_field_note is not None:
                print(f"MAR: {mar_far_field_note}")
        else:
            print(f"Boundary residual max = {mar_solution.boundary_residual_max:.6e}")
            if mar_pattern.size == 0:
                print("Directivity = not defined for this field selection.")
                print("Peak angle [deg] = not defined for this field selection.")
            else:
                print(f"Directivity = {mar_directivity:.6f}")
                print(f"Peak angle [deg] = {np.rad2deg(mar_peak_phi):.6f}")
            if mar_far_field_note is not None:
                print(mar_far_field_note)
        for reflector_index in range(solver.num_reflectors):
            edge_left_db, edge_right_db = mar_solution.edge_illumination_db(
                reflector_index=reflector_index
            )
            print(
                f"{'MAR ' if args.solver_kind in {'compare_mar', 'compare_all'} else ''}Reflector {reflector_index} edge illumination left/right [dB] = "
                f"{edge_left_db:.6f}, {edge_right_db:.6f}"
            )
    if paper_solution is not None:
        if args.solver_kind in {"compare_paper_mds", "compare_all"}:
            print(f"Paper MDS boundary residual max = {paper_solution.boundary_residual_max:.6e}")
            if paper_pattern.size != 0:
                print(f"Paper MDS directivity = {paper_directivity:.6f}")
                print(f"Paper MDS peak angle [deg] = {np.rad2deg(paper_peak_phi):.6f}")
            if paper_far_field_note is not None:
                print(f"Paper MDS: {paper_far_field_note}")
        else:
            print(f"Boundary residual max = {paper_solution.boundary_residual_max:.6e}")
            if paper_pattern.size == 0:
                print("Directivity = not defined for this field selection.")
                print("Peak angle [deg] = not defined for this field selection.")
            else:
                print(f"Directivity = {paper_directivity:.6f}")
                print(f"Peak angle [deg] = {np.rad2deg(paper_peak_phi):.6f}")
            if paper_far_field_note is not None:
                print(paper_far_field_note)
        for reflector_index in range(solver.num_reflectors):
            edge_left_db, edge_right_db = paper_solution.edge_illumination_db(
                reflector_index=reflector_index
            )
            print(
                f"{'Paper MDS ' if args.solver_kind in {'compare_paper_mds', 'compare_all'} else ''}Reflector {reflector_index} edge illumination left/right [dB] = "
                f"{edge_left_db:.6f}, {edge_right_db:.6f}"
            )
    if fdtd_solution is not None:
        print(
            "FDTD grid: "
            f"dx={fdtd_solution.dx:.6f}, dy={fdtd_solution.dy:.6f}, "
            f"ppw={fdtd_solution.points_per_wavelength:.2f}, dt={fdtd_solution.dt:.6f}, "
            f"steps={fdtd_solution.steps}, sample_steps={fdtd_solution.sample_steps}"
        )
        print("FDTD far-field report = not implemented; comparison uses near fields.")

    nystrom_xg = None
    nystrom_yg = None
    nystrom_field = None
    if nystrom_solution is not None:
        nystrom_xg, nystrom_yg, nystrom_field = build_solution_grid(
            args,
            solver,
            nystrom_solution,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    mom_xg = None
    mom_yg = None
    mom_field = None
    if mom_solution is not None and mom_solver is not None:
        mom_xg, mom_yg, mom_field = build_solution_grid(
            args,
            mom_solver,
            mom_solution,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    mar_xg = None
    mar_yg = None
    mar_field = None
    if mar_solution is not None and mar_solver is not None:
        mar_xg, mar_yg, mar_field = build_solution_grid(
            args,
            mar_solver,
            mar_solution,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    paper_xg = None
    paper_yg = None
    paper_field = None
    if paper_solution is not None and paper_solver is not None:
        paper_xg, paper_yg, paper_field = build_solution_grid(
            args,
            paper_solver,
            paper_solution,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    fdtd_field = None
    if fdtd_solution is not None:
        fdtd_field = fdtd_solution.field(args.field_kind)

    if args.solver_kind == "nystrom":
        overview_path = None if args.save_prefix is None else f"{args.save_prefix}_overview.png"
        note_lines = [far_field_note] if far_field_note is not None else []
        plot_solution_overview(
            solution=nystrom_solution,
            xg=nystrom_xg,
            yg=nystrom_yg,
            total_field=nystrom_field,
            field_label=args.field_kind.capitalize(),
            plot_style=args.near_plot_style,
            paper_bar_max=args.paper_bar_max,
            paper_percentile_low=args.paper_percentile_low,
            paper_percentile_high=args.paper_percentile_high,
            mask_distance=args.mask_distance,
            phi=phi,
            pattern=pattern,
            figure_title=f"Nystrom | {args.field_kind.capitalize()} field",
            note_lines=note_lines,
            save_path=overview_path,
            show=not args.no_show,
        )
        return

    if args.solver_kind == "mom":
        overview_path = None if args.save_prefix is None else f"{args.save_prefix}_mom_overview.png"
        note_lines = [mom_far_field_note] if mom_far_field_note is not None else []
        plot_solution_overview(
            solution=mom_solution,
            xg=mom_xg,
            yg=mom_yg,
            total_field=mom_field,
            field_label=args.field_kind.capitalize(),
            plot_style=args.near_plot_style,
            paper_bar_max=args.paper_bar_max,
            paper_percentile_low=args.paper_percentile_low,
            paper_percentile_high=args.paper_percentile_high,
            mask_distance=args.mask_distance,
            phi=mom_phi,
            pattern=mom_pattern,
            figure_title=f"MoM | {args.field_kind.capitalize()} field",
            note_lines=note_lines,
            save_path=overview_path,
            show=not args.no_show,
        )
        return

    if args.solver_kind == "mar":
        overview_path = None if args.save_prefix is None else f"{args.save_prefix}_mar_overview.png"
        note_lines = [mar_far_field_note] if mar_far_field_note is not None else []
        plot_solution_overview(
            solution=mar_solution,
            xg=mar_xg,
            yg=mar_yg,
            total_field=mar_field,
            field_label=args.field_kind.capitalize(),
            plot_style=args.near_plot_style,
            paper_bar_max=args.paper_bar_max,
            paper_percentile_low=args.paper_percentile_low,
            paper_percentile_high=args.paper_percentile_high,
            mask_distance=args.mask_distance,
            phi=mar_phi,
            pattern=mar_pattern,
            figure_title=f"MAR | {args.field_kind.capitalize()} field",
            note_lines=note_lines,
            save_path=overview_path,
            show=not args.no_show,
        )
        return

    if args.solver_kind == "paper_mds":
        overview_path = None if args.save_prefix is None else f"{args.save_prefix}_paper_mds_overview.png"
        note_lines = [paper_far_field_note] if paper_far_field_note is not None else []
        plot_solution_overview(
            solution=paper_solution,
            xg=paper_xg,
            yg=paper_yg,
            total_field=paper_field,
            field_label=args.field_kind.capitalize(),
            plot_style=args.near_plot_style,
            paper_bar_max=args.paper_bar_max,
            paper_percentile_low=args.paper_percentile_low,
            paper_percentile_high=args.paper_percentile_high,
            mask_distance=args.mask_distance,
            phi=paper_phi,
            pattern=paper_pattern,
            figure_title=f"Paper MDS | {args.field_kind.capitalize()} field",
            note_lines=note_lines,
            save_path=overview_path,
            show=not args.no_show,
        )
        return

    if args.solver_kind == "fdtd":
        overview_path = None if args.save_prefix is None else f"{args.save_prefix}_fdtd_overview.png"
        plot_solution_overview(
            solution=fdtd_solution,
            xg=fdtd_solution.xg,
            yg=fdtd_solution.yg,
            total_field=fdtd_field,
            field_label=args.field_kind.capitalize(),
            plot_style=args.near_plot_style,
            paper_bar_max=args.paper_bar_max,
            paper_percentile_low=args.paper_percentile_low,
            paper_percentile_high=args.paper_percentile_high,
            mask_distance=args.mask_distance,
            phi=np.array([], dtype=np.float64),
            pattern=np.array([], dtype=np.complex128),
            figure_title=f"FDTD | {args.field_kind.capitalize()} field",
            note_lines=["FDTD far-field report is not implemented."],
            save_path=overview_path,
            show=not args.no_show,
        )
        return

    if args.solver_kind == "compare_all":
        compare_mask = None
        if args.mask_distance is not None and args.mask_distance > 0.0:
            compare_mask = reflector_distance_mask(
                solver.reflectors,
                paper_xg,
                paper_yg,
                threshold=args.mask_distance,
            )
        reference_observable = comparison_observable(args, paper_solver, paper_field)
        reference_scale = float(np.max(np.abs(reference_observable))) if reference_observable.size else 0.0

        mom_observable = comparison_observable(args, mom_solver, mom_field)
        mom_diff_raw = np.abs(mom_observable - reference_observable)
        mom_diff = mom_diff_raw / reference_scale if reference_scale > 0.0 else mom_diff_raw

        mar_observable = comparison_observable(args, mar_solver, mar_field)
        mar_diff_raw = np.abs(mar_observable - reference_observable)
        mar_diff = mar_diff_raw / reference_scale if reference_scale > 0.0 else mar_diff_raw

        rel_l2_mom, max_abs_error_mom, max_rel_error_mom = comparison_metrics(
            reference_field=reference_observable,
            candidate_field=mom_observable,
            mask=compare_mask,
        )
        rel_l2_mar, max_abs_error_mar, max_rel_error_mar = comparison_metrics(
            reference_field=reference_observable,
            candidate_field=mar_observable,
            mask=compare_mask,
        )
        print(f"Compare MoM observable relative L2 error = {rel_l2_mom:.6e}")
        print(f"Compare MoM observable max absolute error = {max_abs_error_mom:.6e}")
        print(f"Compare MoM observable max relative error = {max_rel_error_mom:.6e}")
        print(f"Compare MAR observable relative L2 error = {rel_l2_mar:.6e}")
        print(f"Compare MAR observable max absolute error = {max_abs_error_mar:.6e}")
        print(f"Compare MAR observable max relative error = {max_rel_error_mar:.6e}")

        overview_path = None if args.save_prefix is None else f"{args.save_prefix}_compare_all_overview.png"
        metrics_lines = [
            f"MoM Rel L2 error: {rel_l2_mom:.6e}",
            f"MAR Rel L2 error: {rel_l2_mar:.6e}",
        ]
        if paper_far_field_note is not None:
            metrics_lines.append(paper_far_field_note)

        plot_all_comparison_overview(
            mds_solution=paper_solution,
            mds_field=paper_field,
            mom_solution=mom_solution,
            mom_field=mom_field,
            mar_solution=mar_solution,
            mar_field=mar_field,
            diff_mom_mds=mom_diff,
            diff_mar_mds=mar_diff,
            xg=paper_xg,
            yg=paper_yg,
            field_label=args.field_kind.capitalize(),
            plot_style=args.near_plot_style,
            paper_bar_max=args.paper_bar_max,
            paper_percentile_low=args.paper_percentile_low,
            paper_percentile_high=args.paper_percentile_high,
            mask_distance=args.mask_distance,
            mds_label="Paper MDS",
            phi=paper_phi,
            mds_pattern=paper_pattern,
            mom_pattern=mom_pattern,
            mar_pattern=mar_pattern,
            metrics_lines=metrics_lines,
            figure_title=f"Paper MDS vs MoM vs MAR | {args.field_kind.capitalize()} field",
            save_path=overview_path,
            show=not args.no_show,
        )
        return

    if args.solver_kind == "compare_mom":
        compare_candidate_label = "MoM"
        compare_candidate_field = mom_field
        compare_candidate_solution = mom_solution
        compare_candidate_xg = mom_xg
        compare_candidate_yg = mom_yg
        compare_candidate_prefix = "mom"
        error_path = None if args.save_prefix is None else f"{args.save_prefix}_compare_mom_error.png"
    elif args.solver_kind == "compare_mar":
        compare_candidate_label = "MAR"
        compare_candidate_field = mar_field
        compare_candidate_solution = mar_solution
        compare_candidate_xg = mar_xg
        compare_candidate_yg = mar_yg
        compare_candidate_prefix = "mar"
        error_path = None if args.save_prefix is None else f"{args.save_prefix}_compare_mar_error.png"
    elif args.solver_kind == "compare_paper_mds":
        compare_candidate_label = "Paper MDS"
        compare_candidate_field = paper_field
        compare_candidate_solution = paper_solution
        compare_candidate_xg = paper_xg
        compare_candidate_yg = paper_yg
        compare_candidate_prefix = "paper_mds"
        error_path = None if args.save_prefix is None else f"{args.save_prefix}_compare_paper_mds_error.png"
    else:
        compare_candidate_label = "FDTD"
        compare_candidate_field = fdtd_field
        compare_candidate_solution = fdtd_solution
        compare_candidate_xg = fdtd_solution.xg
        compare_candidate_yg = fdtd_solution.yg
        compare_candidate_prefix = "fdtd"
        error_path = None if args.save_prefix is None else f"{args.save_prefix}_compare_error.png"

    compare_mask = None
    if args.mask_distance is not None and args.mask_distance > 0.0:
        compare_mask = reflector_distance_mask(
            solver.reflectors,
            nystrom_xg,
            nystrom_yg,
            threshold=args.mask_distance,
        )
    reference_observable = comparison_observable(args, solver, nystrom_field)
    candidate_observable = comparison_observable(args, solver, compare_candidate_field)
    observable_diff = np.abs(candidate_observable - reference_observable)
    reference_scale = float(np.max(np.abs(reference_observable))) if reference_observable.size else 0.0
    if reference_scale > 0.0:
        normalized_diff = observable_diff / reference_scale
    else:
        normalized_diff = observable_diff
    rel_l2, max_abs_error, max_rel_error = comparison_metrics(
        reference_field=reference_observable,
        candidate_field=candidate_observable,
        mask=compare_mask,
    )
    print(f"Compare {compare_candidate_label} observable relative L2 error = {rel_l2:.6e}")
    print(f"Compare {compare_candidate_label} observable max absolute error = {max_abs_error:.6e}")
    print(f"Compare {compare_candidate_label} observable max relative error = {max_rel_error:.6e}")

    overview_suffix = (
        f"{compare_candidate_prefix}_comparison_overview.png"
        if compare_candidate_prefix != "fdtd"
        else "comparison_overview.png"
    )
    overview_path = None if args.save_prefix is None else f"{args.save_prefix}_{overview_suffix}"
    metrics_lines = [
        f"Relative L2 error: {rel_l2:.6e}",
        f"Max abs error:     {max_abs_error:.6e}",
        f"Max rel error:     {max_rel_error:.6e}",
    ]
    plot_comparison_overview(
        reference_solution=nystrom_solution,
        reference_xg=nystrom_xg,
        reference_yg=nystrom_yg,
        reference_field=nystrom_field,
        reference_label=f"Nystrom {args.field_kind.capitalize()}",
        candidate_solution=compare_candidate_solution,
        candidate_xg=compare_candidate_xg,
        candidate_yg=compare_candidate_yg,
        candidate_field=compare_candidate_field,
        candidate_label=f"{compare_candidate_label} {args.field_kind.capitalize()}",
        difference_map=normalized_diff,
        difference_title=f"Normalized difference ({compare_candidate_label} vs Nystrom)",
        plot_style=args.near_plot_style,
        paper_bar_max=args.paper_bar_max,
        paper_percentile_low=args.paper_percentile_low,
        paper_percentile_high=args.paper_percentile_high,
        mask_distance=args.mask_distance,
        phi=phi,
        pattern=pattern,
        far_field_note=far_field_note,
        metrics_lines=metrics_lines,
        figure_title=f"Nystrom vs {compare_candidate_label} | {args.field_kind.capitalize()} field",
        save_path=overview_path,
        show=not args.no_show,
    )
