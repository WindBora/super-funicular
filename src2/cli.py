"""Command-line entrypoint for the reflector solver and plotting workflow."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Sequence

import numpy as np

from .plotting import plot_far_field, plot_near_field
from .scenes import build_scene, resolved_beta_deg, resolved_plot_window


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the solver application.

    Parameters
    ----------
    argv:
        Optional explicit argument list. When omitted, ``argparse`` reads from
        ``sys.argv``.
    """

    parser = argparse.ArgumentParser(
        description="MDS implementation for 2-D E-wave reflector scattering with CSP feed."
    )
    parser.add_argument("--n", type=int, default=120, help="Interpolation order.")
    parser.add_argument(
        "--scene",
        choices=("single_shifted", "cassegrain", "two_brackets", "confocal_elliptic", "sinusoidal_strip"),
        default="cassegrain",
        help="Geometry preset. 'cassegrain' uses a parabolic main reflector and a hyperbolic subreflector, 'confocal_elliptic' uses two ellipse arcs with a shared intermediate focus, and 'sinusoidal_strip' builds a bottom sinusoidal reflector strip.",
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
    parser.add_argument("--kb", type=float, default=9.0, help="CSP beam parameter kb.")
    parser.add_argument(
        "--beta-deg",
        type=float,
        default=None,
        help="CSP beam aiming angle in degrees. Defaults to 180 deg for 'single_shifted', 0 deg for 'cassegrain' and 'two_brackets', 140 deg for 'confocal_elliptic', and automatically points to the strip target for 'sinusoidal_strip'.",
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

    near_path = None if args.save_prefix is None else f"{args.save_prefix}_{args.field_kind}_near.png"
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
