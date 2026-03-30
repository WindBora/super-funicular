"""Scene builders that assemble reflector geometries and feeds for the solver."""

from __future__ import annotations

import argparse

import numpy as np

from .geometry import (
    EllipseArc,
    HyperbolaArc,
    ReflectedCurveX,
    SinusoidalStrip,
    TranslatedCurve,
    build_ellipse_arc_from_equation,
    build_focus_centered_parabola,
    build_shifted_parabola,
    build_shifted_parabola_from_bounds,
    ellipse_foci,
    symmetric_bounds,
)
from .solver import ComplexSourcePoint, MultiReflectorMDS


def build_single_shifted_parabolic_example(
    n: int,
    aperture: float,
    focal_ratio: float,
    kb: float,
    beta_deg: float,
) -> MultiReflectorMDS:
    """Build the single parabolic reflector example centered on the feed focus.

    Parameters
    ----------
    n:
        Number of current unknowns per reflector.
    aperture:
        Reflector aperture in wavelengths.
    focal_ratio:
        Ratio ``f / aperture``.
    kb:
        CSP beam parameter.
    beta_deg:
        Feed aiming angle in degrees.
    """

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


def sinusoidal_strip_target_point(
    strip_center_x: float,
    strip_base_y: float,
    strip_amplitude: float,
    strip_frequency: float,
    strip_phase_deg: float,
    strip_target_x: float | None,
) -> tuple[float, float]:
    """Return the point on the sinusoidal strip used as the default aiming target.

    Parameters
    ----------
    strip_center_x:
        Midpoint x-coordinate of the strip.
    strip_base_y:
        Baseline y-coordinate of the strip.
    strip_amplitude:
        Vertical oscillation amplitude.
    strip_frequency:
        Spatial frequency in cycles per wavelength unit.
    strip_phase_deg:
        Phase offset of the sinusoid in degrees.
    strip_target_x:
        Optional x-coordinate of the target point. When omitted, the strip
        midpoint is used.
    """

    target_x = strip_center_x if strip_target_x is None else float(strip_target_x)
    phase_rad = np.deg2rad(strip_phase_deg)
    arg = 2.0 * np.pi * strip_frequency * (target_x - strip_center_x) + phase_rad
    target_y = strip_base_y + strip_amplitude * np.sin(arg)
    return float(target_x), float(target_y)


def build_sinusoidal_strip_example(
    n: int,
    strip_length: float,
    strip_amplitude: float,
    strip_frequency: float,
    strip_base_y: float,
    strip_center_x: float,
    strip_phase_deg: float,
    strip_feed_x: float | None,
    strip_feed_y: float | None,
    strip_target_x: float | None,
    kb: float,
    beta_deg: float,
) -> MultiReflectorMDS:
    """Build a scene with a single open sinusoidal reflector strip.

    Parameters
    ----------
    n:
        Number of current unknowns on the strip.
    strip_length:
        Horizontal end-to-end strip length in wavelengths.
    strip_amplitude:
        Vertical oscillation amplitude in wavelengths.
    strip_frequency:
        Spatial frequency in cycles per wavelength unit.
    strip_base_y:
        Baseline y-coordinate of the strip.
    strip_center_x:
        Midpoint x-coordinate of the strip.
    strip_phase_deg:
        Sinusoidal phase offset in degrees.
    strip_feed_x, strip_feed_y:
        Optional feed coordinates. Defaults place the feed above the strip center.
    strip_target_x:
        Optional x-coordinate on the strip used as the default aiming point.
    kb:
        CSP beam parameter.
    beta_deg:
        Feed aiming angle in degrees.
    """

    if strip_length <= 0.0:
        raise ValueError("strip_length must be positive")

    wavelength = 1.0
    k = 2.0 * np.pi / wavelength
    phase_rad = float(np.deg2rad(strip_phase_deg))
    target_x, target_y = sinusoidal_strip_target_point(
        strip_center_x=strip_center_x,
        strip_base_y=strip_base_y,
        strip_amplitude=strip_amplitude,
        strip_frequency=strip_frequency,
        strip_phase_deg=strip_phase_deg,
        strip_target_x=strip_target_x,
    )

    feed_x = strip_center_x if strip_feed_x is None else float(strip_feed_x)
    if strip_feed_y is None:
        feed_y = strip_base_y + max(0.35 * strip_length, 4.0 * abs(strip_amplitude), 8.0)
    else:
        feed_y = float(strip_feed_y)

    curve = SinusoidalStrip(
        x_center=float(strip_center_x),
        y_base=float(strip_base_y),
        length=float(strip_length),
        amplitude=float(strip_amplitude),
        frequency=float(strip_frequency),
        phase_rad=phase_rad,
    )
    csp = ComplexSourcePoint(
        k=k,
        x0=feed_x,
        y0=feed_y,
        b=kb / k,
        beta_rad=np.deg2rad(beta_deg),
    )
    return MultiReflectorMDS(reflectors=[curve], incident=csp, n=n)


def build_confocal_elliptic_example(
    n: int,
    kb: float,
    beta_deg: float,
    ellipse_upper_mode: str,
    ellipse_lower_side: str,
    ellipse_d: float | None,
    ellipse_feed_x: float | None,
    ellipse_feed_y: float | None,
    ellipse_common_focus_x: float | None,
    ellipse_common_focus_y: float | None,
    ellipse_lower_center_x: float | None,
    ellipse_lower_center_y: float | None,
    ellipse_lower_rotation_deg: float | None,
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
    """Build the generalized two-ellipse scene used for the figure-style plots.

    Parameters
    ----------
    n:
        Number of current unknowns per reflector.
    kb:
        CSP beam parameter.
    beta_deg:
        Feed aiming angle in degrees.
    ellipse_upper_mode:
        How the upper reflector is constructed. ``"mirror_vertical"`` makes it a
        vertical-line mirror of the lower reflector; ``"focus_pair"`` uses a
        separate focus-pair ellipse definition.
    ellipse_lower_side:
        Which side of the lower bracket-like ellipse is retained in
        ``mirror_vertical`` mode. ``"left"`` produces a reflector whose opening
        faces to the right, while ``"right"`` produces a reflector whose opening
        faces to the left.
    ellipse_d:
        Reference size parameter used by the tuned defaults.
    ellipse_feed_x, ellipse_feed_y:
        Feed location overrides.
    ellipse_common_focus_x, ellipse_common_focus_y:
        Shared intermediate focus used by the tuned defaults.
    ellipse_lower_center_x, ellipse_lower_center_y:
        Center of the lower ellipse used in ``mirror_vertical`` mode.
    ellipse_lower_rotation_deg:
        Rotation angle of the lower ellipse local frame. The ellipse itself is
        defined by ``(x'/a1)^2 + (y'/a2)^2 = 1`` before this rotation.
    ellipse_mirror_x:
        x-coordinate of the vertical reflection line for the upper reflector.
    ellipse_upper_shift_y:
        Upward translation applied to the mirrored upper reflector.
    ellipse_output_focus_x, ellipse_output_focus_y:
        Auxiliary focus for the ``focus_pair`` upper-ellipse mode.
    ellipse_lower_a, ellipse_lower_b:
        Lower-reflector semi-axis lengths. In ``mirror_vertical`` mode these are
        the horizontal and vertical semi-axes of the bracket-like ellipse.
    ellipse_upper_a:
        Semi-major axis for the upper reflector in ``focus_pair`` mode.
    ellipse_lower_theta1, ellipse_lower_theta2:
        Lower-arc angle limits used in ``focus_pair`` mode.
    ellipse_upper_theta1, ellipse_upper_theta2:
        Upper-arc angle limits used in ``focus_pair`` mode.
    ellipse_half_angle_deg:
        Half-angle span of the bracket-like lower reflector in ``mirror_vertical`` mode.
    """

    wavelength = 1.0
    k = 2.0 * np.pi / wavelength

    d = 20.0 if ellipse_d is None else float(ellipse_d)

    common_focus_x = 0.34 * d if ellipse_common_focus_x is None else float(ellipse_common_focus_x)
    common_focus_y = 0.36 * d if ellipse_common_focus_y is None else float(ellipse_common_focus_y)
    lower_center_x = 0.125 * d if ellipse_lower_center_x is None else float(ellipse_lower_center_x)
    lower_center_y = -0.12 * d if ellipse_lower_center_y is None else float(ellipse_lower_center_y)
    lower_rotation_deg = 0.0 if ellipse_lower_rotation_deg is None else float(ellipse_lower_rotation_deg)
    output_focus_x = 0.50 * d if ellipse_output_focus_x is None else float(ellipse_output_focus_x)
    output_focus_y = 1.00 * d if ellipse_output_focus_y is None else float(ellipse_output_focus_y)

    lower_a_default = 0.2 * d if ellipse_upper_mode == "mirror_vertical" else 0.1 * d
    lower_a_raw = lower_a_default if ellipse_lower_a is None else float(ellipse_lower_a)
    lower_b_raw = 0.9 * d if ellipse_lower_b is None else float(ellipse_lower_b)
    lower_a = max(lower_a_raw, lower_b_raw)
    lower_b = min(lower_a_raw, lower_b_raw)
    upper_a = lower_a if ellipse_upper_a is None else float(ellipse_upper_a)

    if ellipse_upper_mode == "mirror_vertical" and ellipse_feed_x is None and ellipse_feed_y is None:
        lower_focus_1, _ = ellipse_foci(
            center_x=lower_center_x,
            center_y=lower_center_y,
            a1=lower_a,
            a2=lower_b,
            rotation_deg=lower_rotation_deg,
        )
        feed_x = float(lower_focus_1[0])
        feed_y = float(lower_focus_1[1])
    else:
        feed_x = 0.34 * d if ellipse_feed_x is None else float(ellipse_feed_x)
        feed_y = -0.37 * d if ellipse_feed_y is None else float(ellipse_feed_y)

    lower_theta1 = 16.0 if ellipse_lower_theta1 is None else float(ellipse_lower_theta1)
    lower_theta2 = 120.0 if ellipse_lower_theta2 is None else float(ellipse_lower_theta2)
    upper_theta1 = lower_theta1 + 180.0 if ellipse_upper_theta1 is None else float(ellipse_upper_theta1)
    upper_theta2 = lower_theta2 + 180.0 if ellipse_upper_theta2 is None else float(ellipse_upper_theta2)
    half_angle_deg = 70.0 if ellipse_half_angle_deg is None else float(ellipse_half_angle_deg)
    mirror_x = 0.375 * d if ellipse_mirror_x is None else float(ellipse_mirror_x)
    upper_shift_y = 0.85 * d if ellipse_upper_shift_y is None else float(ellipse_upper_shift_y)

    if ellipse_upper_mode == "mirror_vertical":
        lower_reflector = build_ellipse_arc_from_equation(
            center_x=lower_center_x,
            center_y=lower_center_y,
            a1=lower_a,
            a2=lower_b,
            half_angle_deg=half_angle_deg,
            side=ellipse_lower_side,
            rotation_deg=lower_rotation_deg,
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
    """Build the generalized Cassegrain scene with parabolic and hyperbolic arcs.

    Parameters
    ----------
    n:
        Number of current unknowns per reflector.
    main_aperture, sub_aperture:
        Main-reflector and subreflector apertures in wavelengths.
    focal_ratio:
        Ratio ``f / main_aperture`` for the parabolic main reflector.
    kb:
        CSP beam parameter.
    beta_deg:
        Feed aiming angle in degrees.
    main_vertex_x:
        x-coordinate of the main-reflector vertex.
    feed_x:
        Optional feed x-position override.
    subreflector_vertex_x:
        Optional subreflector branch vertex x-position override.
    main_y1, main_y2:
        Optional main-reflector truncation limits.
    sub_y1, sub_y2:
        Optional subreflector truncation limits.
    sub_branch:
        Hyperbola branch to use for the subreflector.
    """

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

    feed_x_resolved = main_vertex_x + 0.30 * focal_length if feed_x is None else float(feed_x)
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
    """Build the two-parabola bracket scene used during geometry experiments.

    Parameters
    ----------
    n:
        Number of current unknowns per reflector.
    aperture:
        Aperture of the smaller reflector.
    focal_ratio:
        Ratio ``f / aperture`` used for both reflectors.
    kb:
        CSP beam parameter.
    beta_deg:
        Feed aiming angle in degrees.
    secondary_scale:
        Ratio of large-reflector aperture to small-reflector aperture.
    small_vertex_x, large_vertex_x:
        Optional vertex x-positions for the small and large reflectors.
    """

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
    """Return the scene-dependent default feed aiming angle in degrees."""

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
    if args.scene == "sinusoidal_strip":
        strip_center_x = 0.0 if args.strip_center_x is None else float(args.strip_center_x)
        strip_base_y = -10.0 if args.strip_base_y is None else float(args.strip_base_y)
        strip_amplitude = 1.5 if args.strip_amplitude is None else float(args.strip_amplitude)
        strip_frequency = 0.2 if args.strip_frequency is None else float(args.strip_frequency)
        strip_phase_deg = 0.0 if args.strip_phase_deg is None else float(args.strip_phase_deg)
        feed_x = strip_center_x if args.strip_feed_x is None else float(args.strip_feed_x)
        if args.strip_feed_y is None:
            strip_length = 24.0 if args.strip_length is None else float(args.strip_length)
            feed_y = strip_base_y + max(0.35 * strip_length, 4.0 * abs(strip_amplitude), 8.0)
        else:
            feed_y = float(args.strip_feed_y)
        target_x, target_y = sinusoidal_strip_target_point(
            strip_center_x=strip_center_x,
            strip_base_y=strip_base_y,
            strip_amplitude=strip_amplitude,
            strip_frequency=strip_frequency,
            strip_phase_deg=strip_phase_deg,
            strip_target_x=args.strip_target_x,
        )
        return float(np.rad2deg(np.arctan2(target_y - feed_y, target_x - feed_x)))
    raise ValueError(f"Unsupported scene: {args.scene}")


def build_scene(args: argparse.Namespace) -> MultiReflectorMDS:
    """Build the solver object corresponding to the parsed command-line options."""

    beta_deg = resolved_beta_deg(args)
    if args.scene == "single_shifted":
        return build_single_shifted_parabolic_example(
            n=args.n,
            aperture=args.aperture,
            focal_ratio=args.focal_ratio,
            kb=args.kb,
            beta_deg=beta_deg,
        )
    if args.scene == "sinusoidal_strip":
        return build_sinusoidal_strip_example(
            n=args.n,
            strip_length=args.strip_length,
            strip_amplitude=args.strip_amplitude,
            strip_frequency=args.strip_frequency,
            strip_base_y=args.strip_base_y,
            strip_center_x=args.strip_center_x,
            strip_phase_deg=args.strip_phase_deg,
            strip_feed_x=args.strip_feed_x,
            strip_feed_y=args.strip_feed_y,
            strip_target_x=args.strip_target_x,
            kb=args.kb,
            beta_deg=beta_deg,
        )
    if args.scene == "confocal_elliptic":
        return build_confocal_elliptic_example(
            n=args.n,
            kb=args.kb,
            beta_deg=beta_deg,
            ellipse_upper_mode=args.ellipse_upper_mode,
            ellipse_lower_side=args.ellipse_lower_side,
            ellipse_d=args.ellipse_d,
            ellipse_feed_x=args.ellipse_feed_x,
            ellipse_feed_y=args.ellipse_feed_y,
            ellipse_common_focus_x=args.ellipse_common_focus_x,
            ellipse_common_focus_y=args.ellipse_common_focus_y,
            ellipse_lower_center_x=args.ellipse_lower_center_x,
            ellipse_lower_center_y=args.ellipse_lower_center_y,
            ellipse_lower_rotation_deg=args.ellipse_lower_rotation_deg,
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


def resolved_plot_window(args: argparse.Namespace) -> tuple[float, float, float, float]:
    """Return scene-aware plot bounds using the CLI values as the starting point."""

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
    if args.scene == "sinusoidal_strip":
        strip_center_x = args.strip_center_x
        strip_length = args.strip_length
        strip_base_y = args.strip_base_y
        strip_amplitude = args.strip_amplitude
        feed_y = args.strip_feed_y
        if x_min == -5.0:
            x_min = strip_center_x - 0.7 * strip_length
        if x_max == 25.0:
            x_max = strip_center_x + 0.7 * strip_length
        if y_min == -15.0:
            y_min = strip_base_y - max(2.5 * abs(strip_amplitude), 3.0)
        if y_max == 15.0:
            default_feed_y = (
                strip_base_y + max(0.35 * strip_length, 4.0 * abs(strip_amplitude), 8.0)
                if feed_y is None
                else feed_y
            )
            y_max = max(default_feed_y + 3.0, strip_base_y + 2.5 * abs(strip_amplitude) + 3.0)

    return x_min, x_max, y_min, y_max
