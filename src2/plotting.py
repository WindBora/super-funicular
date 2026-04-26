"""Plotting helpers for near-field and far-field visualizations."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import ParamCurve
from .solver import MDSSolution, PlaneWave


def reflector_distance_mask(
    reflectors: Sequence[ParamCurve],
    xg: np.ndarray,
    yg: np.ndarray,
    threshold: float,
    samples_per_reflector: int = 512,
) -> np.ndarray:
    """Mask points that lie too close to any reflector contour.

    Parameters
    ----------
    reflectors:
        Reflector contours used to compute distances.
    xg, yg:
        Meshgrid-style coordinate arrays.
    threshold:
        Distance below which points are masked.
    samples_per_reflector:
        Number of points used to sample each reflector for the distance test.
    """

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


def _effective_mask_distance(plot_style: str, mask_distance: float | None) -> float | None:
    """Return the mask distance adjusted for the chosen plot style."""

    effective_mask_distance = mask_distance
    if plot_style == "paper" and effective_mask_distance is not None:
        effective_mask_distance = min(effective_mask_distance, 0.06)
    if plot_style == "figure" and effective_mask_distance is not None:
        effective_mask_distance = min(effective_mask_distance, 0.08)
    return effective_mask_distance


def _scene_extent(xg: np.ndarray, yg: np.ndarray) -> list[float]:
    """Return the image extent used by all spatial plots."""

    return [float(xg.min()), float(xg.max()), float(yg.min()), float(yg.max())]


def _reflector_curves(solution) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return dense reflector samples for plot overlays."""

    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for curve in solution.solver.reflectors:
        t_plot = np.linspace(-1.0, 1.0, 500)
        xc, yc = curve.coords(t_plot)
        curves.append((np.asarray(xc, dtype=np.float64), np.asarray(yc, dtype=np.float64)))
    return curves


def _draw_plane_wave_direction(
    ax,
    solution,
    xg: np.ndarray,
    yg: np.ndarray,
    *,
    color: str,
    outline_color: str,
) -> None:
    """Overlay a short arrow showing the propagation direction of the plane wave."""

    incident = solution.solver.incident
    if not isinstance(incident, PlaneWave):
        return

    x_min = float(xg.min())
    x_max = float(xg.max())
    y_min = float(yg.min())
    y_max = float(yg.max())
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0.0 or height <= 0.0:
        return

    direction = np.array([incident.direction_x, incident.direction_y], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return
    direction /= norm

    length = 0.18 * min(width, height)
    margin_x = 0.08 * width
    margin_y = 0.08 * height
    center_x = x_min + margin_x + 0.5 * length * abs(direction[0])
    center_y = y_max - margin_y - 0.5 * length * abs(direction[1])
    start = np.array([center_x, center_y]) - 0.5 * length * direction
    end = np.array([center_x, center_y]) + 0.5 * length * direction

    for line_width, line_color in ((3.6, outline_color), (2.2, color)):
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=line_color,
                linewidth=line_width,
                shrinkA=0.0,
                shrinkB=0.0,
            ),
            zorder=8,
        )

    label_x = 0.5 * (start[0] + end[0])
    label_y = 0.5 * (start[1] + end[1]) + 0.06 * height
    ax.text(
        label_x,
        label_y,
        "Plane wave",
        color=color,
        fontsize=9,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=outline_color, edgecolor="none", alpha=0.75),
        zorder=8,
    )


def _decorate_spatial_axis(
    ax,
    solution,
    xg: np.ndarray,
    yg: np.ndarray,
    *,
    axis_color: str,
    outline_color: str,
    show_direction: bool = True,
) -> None:
    """Add common overlays such as axes, reflectors, feed marker, and plane-wave arrow."""

    ax.axvline(0.0, color=axis_color, linestyle="--", linewidth=0.9, alpha=0.75)
    ax.axhline(0.0, color=axis_color, linestyle="--", linewidth=0.9, alpha=0.75)

    reflector_colors = ["#ff55ff", "#55ff55", "#ffd166", "#7df9ff"]
    for idx, (xc, yc) in enumerate(_reflector_curves(solution)):
        color = reflector_colors[idx % len(reflector_colors)]
        ax.plot(xc, yc, color="white", linewidth=2.2, solid_capstyle="round", zorder=6)
        ax.plot(xc, yc, color=color, linewidth=1.4, solid_capstyle="round", zorder=7)

    incident_x = getattr(solution.solver.incident, "x0", None)
    incident_y = getattr(solution.solver.incident, "y0", None)
    if incident_x is not None and incident_y is not None:
        ax.plot(
            [incident_x],
            [incident_y],
            marker="o",
            markersize=4.0,
            markerfacecolor="#00ff66",
            markeredgecolor="#00ff66",
            linestyle="None",
            zorder=7,
        )

    if show_direction:
        _draw_plane_wave_direction(
            ax,
            solution,
            xg,
            yg,
            color=axis_color,
            outline_color=outline_color,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_single_quantity_near_field(
    ax,
    solution,
    xg: np.ndarray,
    yg: np.ndarray,
    total_field: np.ndarray,
    *,
    field_label: str,
    plot_style: str,
    paper_bar_max: float,
    paper_percentile_low: float,
    paper_percentile_high: float,
    mask_distance: float | None,
) -> None:
    """Draw a one-panel near-field view for ``figure`` or ``paper`` style."""

    intensity = np.abs(total_field) ** 2
    max_intensity = float(np.nanmax(intensity))
    is_free_space_plane_wave = (
        solution.solver.num_reflectors == 0 and isinstance(solution.solver.incident, PlaneWave)
    )
    is_incident_plane_wave = (
        isinstance(solution.solver.incident, PlaneWave) and "incident" in field_label.lower()
    )
    extent = _scene_extent(xg, yg)
    effective_mask_distance = _effective_mask_distance(plot_style, mask_distance)
    mask = None
    if effective_mask_distance is not None and effective_mask_distance > 0.0:
        mask = reflector_distance_mask(
            solution.solver.reflectors, xg, yg, threshold=effective_mask_distance
        )

    if plot_style == "paper":
        eps = np.finfo(np.float64).tiny
        log_intensity = np.log10(np.maximum(intensity, eps))
        visible = log_intensity[~mask] if mask is not None else log_intensity.reshape(-1)
        if visible.size == 0:
            visible = log_intensity.reshape(-1)
        low = float(np.percentile(visible, paper_percentile_low))
        high = float(np.percentile(visible, paper_percentile_high))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            high = low + 1.0
        image = paper_bar_max * (log_intensity - low) / (high - low)
        image = np.clip(image, 0.0, paper_bar_max)
        if mask is not None:
            image = np.ma.array(image, mask=mask)
        ax.imshow(
            image,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="gray",
            vmin=0.0,
            vmax=paper_bar_max,
        )
        ax.set_title(f"{field_label} Field (Paper Map)")
        for xc, yc in _reflector_curves(solution):
            ax.plot(xc, yc, color="white", linewidth=1.0, zorder=6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if isinstance(solution.solver.incident, PlaneWave):
            _draw_plane_wave_direction(
                ax,
                solution,
                xg,
                yg,
                color="black",
                outline_color="white",
            )
        return

    if is_free_space_plane_wave or is_incident_plane_wave:
        image = np.real(total_field)
        ax.imshow(
            image,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_title(f"{field_label} Field (Real Part)")
    else:
        if max_intensity > 0.0:
            image = 10.0 * np.log10(intensity / max_intensity)
        else:
            image = np.full(intensity.shape, -45.0, dtype=np.float64)
        if mask is not None:
            image = np.ma.array(image, mask=mask)
        ax.imshow(
            image,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="jet",
            vmin=-45.0,
            vmax=0.0,
        )
        ax.set_title(f"{field_label} Field (Intensity dB)")

    _decorate_spatial_axis(
        ax,
        solution,
        xg,
        yg,
        axis_color="black",
        outline_color="white",
    )


def _plot_standard_near_field_axes(
    axes,
    solution,
    xg: np.ndarray,
    yg: np.ndarray,
    total_field: np.ndarray,
    *,
    field_label: str,
    mask_distance: float | None,
) -> None:
    """Draw the two-panel intensity/phase near-field view used by ``standard`` style."""

    intensity = np.abs(total_field) ** 2
    phase = np.angle(total_field)
    max_intensity = float(np.nanmax(intensity))
    if max_intensity > 0.0:
        intensity_db = 10.0 * np.log10(intensity / max_intensity)
    else:
        intensity_db = np.full(intensity.shape, -60.0, dtype=np.float64)

    effective_mask_distance = _effective_mask_distance("figure", mask_distance)
    mask = None
    if effective_mask_distance is not None and effective_mask_distance > 0.0:
        mask = reflector_distance_mask(
            solution.solver.reflectors, xg, yg, threshold=effective_mask_distance
        )
        intensity_db = np.ma.array(intensity_db, mask=mask)
        phase = np.ma.array(phase, mask=mask)

    extent = _scene_extent(xg, yg)
    axes[0].imshow(
        intensity_db,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap="jet",
        vmin=-60.0,
        vmax=0.0,
    )
    axes[0].set_title(f"{field_label} Intensity [dB]")
    _decorate_spatial_axis(
        axes[0],
        solution,
        xg,
        yg,
        axis_color="black",
        outline_color="white",
    )

    axes[1].imshow(
        phase,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap="twilight",
    )
    axes[1].set_title(f"{field_label} Phase [rad]")
    _decorate_spatial_axis(
        axes[1],
        solution,
        xg,
        yg,
        axis_color="black",
        outline_color="white",
    )


def _plot_far_field_axis(ax, phi: np.ndarray, pattern: np.ndarray, *, title: str) -> None:
    """Draw a far-field line plot on a provided axis."""

    max_pattern = float(np.max(np.abs(pattern)))
    if max_pattern > 0.0:
        pattern_db = 20.0 * np.log10(np.abs(pattern) / max_pattern)
    else:
        pattern_db = np.full(pattern.shape, -60.0, dtype=np.float64)
    ax.plot(np.rad2deg(phi), pattern_db, "b-")
    ax.set_title(title)
    ax.set_xlabel("phi [deg]")
    ax.set_ylabel("Normalized |Phi| [dB]")
    ax.set_ylim(-60.0, 1.0)
    ax.grid(True, alpha=0.3)


def _plot_difference_axis(
    ax,
    solution,
    xg: np.ndarray,
    yg: np.ndarray,
    difference_map: np.ndarray,
    *,
    title: str,
    mask_distance: float | None,
) -> None:
    """Draw a normalized scalar difference map on a provided axis."""

    error = np.asarray(difference_map, dtype=np.float64)
    mask = None
    if mask_distance is not None and mask_distance > 0.0:
        effective_mask_distance = min(mask_distance, 0.08)
        mask = reflector_distance_mask(
            solution.solver.reflectors, xg, yg, threshold=effective_mask_distance
        )
        error = np.ma.array(error, mask=mask)

    error_values = error.filled(np.nan) if np.ma.isMaskedArray(error) else error
    vmax = float(np.nanpercentile(error_values.reshape(-1), 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    ax.imshow(
        error,
        extent=_scene_extent(xg, yg),
        origin="lower",
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(title)
    _decorate_spatial_axis(
        ax,
        solution,
        xg,
        yg,
        axis_color="white",
        outline_color="black",
    )


def _fill_note_axis(ax, *, title: str, lines: Sequence[str]) -> None:
    """Render a text-only panel for notes or metrics."""

    ax.set_title(title)
    ax.axis("off")
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )


def plot_solution_overview(
    solution,
    xg: np.ndarray,
    yg: np.ndarray,
    total_field: np.ndarray,
    *,
    field_label: str,
    plot_style: str,
    paper_bar_max: float,
    paper_percentile_low: float,
    paper_percentile_high: float,
    mask_distance: float | None,
    phi: np.ndarray | None = None,
    pattern: np.ndarray | None = None,
    figure_title: str | None = None,
    note_lines: Sequence[str] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the solver outputs in one overview window."""

    import matplotlib.pyplot as plt

    phi_arr = np.array([], dtype=np.float64) if phi is None else np.asarray(phi)
    pattern_arr = np.array([], dtype=np.complex128) if pattern is None else np.asarray(pattern)
    notes = list(note_lines or [])

    if plot_style == "standard":
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), constrained_layout=True)
        _plot_standard_near_field_axes(
            axes[:2],
            solution,
            xg,
            yg,
            total_field,
            field_label=field_label,
            mask_distance=mask_distance,
        )
        if pattern_arr.size != 0:
            _plot_far_field_axis(axes[2], phi_arr, pattern_arr, title="Far Field")
            if notes:
                axes[2].text(
                    0.03,
                    0.03,
                    "\n".join(notes),
                    transform=axes[2].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5", alpha=0.9),
                )
        else:
            _fill_note_axis(axes[2], title="Notes", lines=notes or ["No additional plot available."])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), constrained_layout=True)
        _plot_single_quantity_near_field(
            axes[0],
            solution,
            xg,
            yg,
            total_field,
            field_label=field_label,
            plot_style=plot_style,
            paper_bar_max=paper_bar_max,
            paper_percentile_low=paper_percentile_low,
            paper_percentile_high=paper_percentile_high,
            mask_distance=mask_distance,
        )
        if pattern_arr.size != 0:
            _plot_far_field_axis(axes[1], phi_arr, pattern_arr, title="Far Field")
            if notes:
                axes[1].text(
                    0.03,
                    0.03,
                    "\n".join(notes),
                    transform=axes[1].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5", alpha=0.9),
                )
        else:
            _fill_note_axis(axes[1], title="Notes", lines=notes or ["No additional plot available."])

    if figure_title is not None:
        fig.suptitle(figure_title)
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison_overview(
    reference_solution,
    reference_xg: np.ndarray,
    reference_yg: np.ndarray,
    reference_field: np.ndarray,
    *,
    reference_label: str,
    candidate_solution,
    candidate_xg: np.ndarray,
    candidate_yg: np.ndarray,
    candidate_field: np.ndarray,
    candidate_label: str,
    difference_map: np.ndarray,
    difference_title: str,
    plot_style: str,
    paper_bar_max: float,
    paper_percentile_low: float,
    paper_percentile_high: float,
    mask_distance: float | None,
    phi: np.ndarray | None = None,
    pattern: np.ndarray | None = None,
    far_field_note: str | None = None,
    metrics_lines: Sequence[str] | None = None,
    figure_title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot comparison panels in a single figure window."""

    import matplotlib.pyplot as plt

    phi_arr = np.array([], dtype=np.float64) if phi is None else np.asarray(phi)
    pattern_arr = np.array([], dtype=np.complex128) if pattern is None else np.asarray(pattern)
    notes = list(metrics_lines or [])
    if far_field_note:
        notes.append(far_field_note)

    if plot_style == "standard":
        fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.6), constrained_layout=True)
        axes_flat = axes.reshape(-1)
        _plot_standard_near_field_axes(
            axes_flat[0:2],
            reference_solution,
            reference_xg,
            reference_yg,
            reference_field,
            field_label=reference_label,
            mask_distance=mask_distance,
        )
        _plot_standard_near_field_axes(
            axes_flat[2:4],
            candidate_solution,
            candidate_xg,
            candidate_yg,
            candidate_field,
            field_label=candidate_label,
            mask_distance=mask_distance,
        )
        _plot_difference_axis(
            axes_flat[4],
            reference_solution,
            reference_xg,
            reference_yg,
            difference_map,
            title=difference_title,
            mask_distance=mask_distance,
        )
        if pattern_arr.size != 0:
            _plot_far_field_axis(axes_flat[5], phi_arr, pattern_arr, title="Nystrom Far Field")
            if notes:
                axes_flat[5].text(
                    0.03,
                    0.03,
                    "\n".join(notes),
                    transform=axes_flat[5].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5", alpha=0.9),
                )
        else:
            _fill_note_axis(axes_flat[5], title="Notes", lines=notes or ["No far-field data available."])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14.2, 11.0), constrained_layout=True)
        _plot_single_quantity_near_field(
            axes[0, 0],
            reference_solution,
            reference_xg,
            reference_yg,
            reference_field,
            field_label=reference_label,
            plot_style=plot_style,
            paper_bar_max=paper_bar_max,
            paper_percentile_low=paper_percentile_low,
            paper_percentile_high=paper_percentile_high,
            mask_distance=mask_distance,
        )
        _plot_single_quantity_near_field(
            axes[0, 1],
            candidate_solution,
            candidate_xg,
            candidate_yg,
            candidate_field,
            field_label=candidate_label,
            plot_style=plot_style,
            paper_bar_max=paper_bar_max,
            paper_percentile_low=paper_percentile_low,
            paper_percentile_high=paper_percentile_high,
            mask_distance=mask_distance,
        )
        _plot_difference_axis(
            axes[1, 0],
            reference_solution,
            reference_xg,
            reference_yg,
            difference_map,
            title=difference_title,
            mask_distance=mask_distance,
        )
        if pattern_arr.size != 0:
            _plot_far_field_axis(axes[1, 1], phi_arr, pattern_arr, title="Nystrom Far Field")
            if notes:
                axes[1, 1].text(
                    0.03,
                    0.03,
                    "\n".join(notes),
                    transform=axes[1, 1].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5", alpha=0.9),
                )
        else:
            _fill_note_axis(axes[1, 1], title="Notes", lines=notes or ["No far-field data available."])

    if figure_title is not None:
        fig.suptitle(figure_title)
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


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
    """Plot the near field using one of the supported visualization styles.

    Parameters
    ----------
    solution:
        Solved reflector problem, used for overlays and masking.
    xg, yg:
        Meshgrid coordinate arrays corresponding to ``total_field``.
    total_field:
        Field values on the grid.
    field_label:
        Human-readable label used in the standard plot style.
    plot_style:
        One of ``"standard"``, ``"paper"``, or ``"figure"``.
    paper_bar_max:
        Upper end of the grayscale bar for the paper-style plot.
    paper_percentile_low, paper_percentile_high:
        Robust percentile limits used to map log-intensity to the grayscale bar.
    mask_distance:
        Distance from the reflector contour inside which grid points are masked.
    save_path:
        Optional file path for saving the plot.
    show:
        When ``True``, display the plot in an interactive window.
    """

    import matplotlib.pyplot as plt

    intensity = np.abs(total_field) ** 2
    phase = np.angle(total_field)
    max_intensity = float(np.nanmax(intensity))
    is_free_space_plane_wave = (
        solution.solver.num_reflectors == 0 and isinstance(solution.solver.incident, PlaneWave)
    )
    is_incident_plane_wave = (
        isinstance(solution.solver.incident, PlaneWave) and "incident" in field_label.lower()
    )

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
        fig, ax = plt.subplots(figsize=(6.8, 6.8), constrained_layout=True)
        if is_free_space_plane_wave or is_incident_plane_wave:
            real_field = np.real(total_field)
            ax.imshow(
                real_field,
                extent=extent,
                origin="lower",
                aspect="equal",
                cmap="RdBu_r",
                vmin=-1.0,
                vmax=1.0,
            )
            ax.set_title(f"{field_label} Field Real Part")
        else:
            if max_intensity > 0.0:
                intensity_db = 10.0 * np.log10(intensity / max_intensity)
            else:
                intensity_db = np.full(intensity.shape, -45.0, dtype=np.float64)
            if mask is not None:
                intensity_db = np.ma.array(intensity_db, mask=mask)
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

        incident_x = getattr(solution.solver.incident, "x0", None)
        incident_y = getattr(solution.solver.incident, "y0", None)
        if incident_x is not None and incident_y is not None:
            ax.plot(
                [incident_x],
                [incident_y],
                marker="o",
                markersize=4.0,
                markerfacecolor="#00ff66",
                markeredgecolor="#00ff66",
                linestyle="None",
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        if max_intensity > 0.0:
            intensity_db = 10.0 * np.log10(intensity / max_intensity)
        else:
            intensity_db = np.full(intensity.shape, -60.0, dtype=np.float64)
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
    """Plot the normalized far-field magnitude in decibels.

    Parameters
    ----------
    phi:
        Observation angles in radians.
    pattern:
        Complex far-field samples associated with ``phi``.
    save_path:
        Optional file path for saving the plot.
    show:
        When ``True``, display the plot in an interactive window.
    """

    import matplotlib.pyplot as plt

    if pattern.size == 0:
        return
    max_pattern = float(np.max(np.abs(pattern)))
    if max_pattern > 0.0:
        pattern_db = 20.0 * np.log10(np.abs(pattern) / max_pattern)
    else:
        pattern_db = np.full(pattern.shape, -60.0, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(np.rad2deg(phi), pattern_db, "b-")
    ax.set_xlabel("phi [deg]")
    ax.set_ylabel("Normalized |Phi| [dB]")
    ax.set_ylim(-60.0, 1.0)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_field_difference(
    solution: MDSSolution,
    xg: np.ndarray,
    yg: np.ndarray,
    difference_map: np.ndarray,
    title: str = "Normalized difference",
    mask_distance: float | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot a normalized scalar difference map on the field grid."""

    import matplotlib.pyplot as plt

    error = np.asarray(difference_map, dtype=np.float64)
    mask = None
    if mask_distance is not None and mask_distance > 0.0:
        effective_mask_distance = min(mask_distance, 0.08)
        mask = reflector_distance_mask(
            solution.solver.reflectors, xg, yg, threshold=effective_mask_distance
        )
        error = np.ma.array(error, mask=mask)

    error_values = error.filled(np.nan) if np.ma.isMaskedArray(error) else error
    vmax = float(np.nanpercentile(error_values.reshape(-1), 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    extent = [float(xg.min()), float(xg.max()), float(yg.min()), float(yg.max())]
    fig, ax = plt.subplots(figsize=(6.8, 6.8), constrained_layout=True)
    ax.imshow(
        error,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.axvline(0.0, color="white", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.axhline(0.0, color="white", linestyle="--", linewidth=0.9, alpha=0.5)
    for curve in solution.solver.reflectors:
        t_plot = np.linspace(-1.0, 1.0, 500)
        xc, yc = curve.coords(t_plot)
        ax.plot(xc, yc, color="white", linewidth=1.8, solid_capstyle="round")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_comparison_overview(
    mds_solution,
    mds_field: np.ndarray,
    mom_solution,
    mom_field: np.ndarray,
    mar_solution,
    mar_field: np.ndarray,
    diff_mom_mds: np.ndarray,
    diff_mar_mds: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    *,
    field_label: str,
    plot_style: str,
    paper_bar_max: float,
    paper_percentile_low: float,
    paper_percentile_high: float,
    mask_distance: float | None,
    mds_label: str = "MDS",
    phi: np.ndarray | None = None,
    mds_pattern: np.ndarray | None = None,
    mom_pattern: np.ndarray | None = None,
    mar_pattern: np.ndarray | None = None,
    metrics_lines: Sequence[str] | None = None,
    figure_title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot MDS, MoM, and MAR fields and their differences in a single figure window."""

    import matplotlib.pyplot as plt

    phi_arr = np.array([], dtype=np.float64) if phi is None else np.asarray(phi)
    mds_pat = np.array([], dtype=np.complex128) if mds_pattern is None else np.asarray(mds_pattern)
    mom_pat = np.array([], dtype=np.complex128) if mom_pattern is None else np.asarray(mom_pattern)
    mar_pat = np.array([], dtype=np.complex128) if mar_pattern is None else np.asarray(mar_pattern)
    notes = list(metrics_lines or [])

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.0), constrained_layout=True)

    _plot_single_quantity_near_field(
        axes[0, 0],
        mds_solution,
        xg,
        yg,
        mds_field,
        field_label=mds_label + " " + field_label,
        plot_style=plot_style,
        paper_bar_max=paper_bar_max,
        paper_percentile_low=paper_percentile_low,
        paper_percentile_high=paper_percentile_high,
        mask_distance=mask_distance,
    )
    _plot_single_quantity_near_field(
        axes[0, 1],
        mom_solution,
        xg,
        yg,
        mom_field,
        field_label="MoM " + field_label,
        plot_style=plot_style,
        paper_bar_max=paper_bar_max,
        paper_percentile_low=paper_percentile_low,
        paper_percentile_high=paper_percentile_high,
        mask_distance=mask_distance,
    )
    _plot_single_quantity_near_field(
        axes[0, 2],
        mar_solution,
        xg,
        yg,
        mar_field,
        field_label="MAR " + field_label,
        plot_style=plot_style,
        paper_bar_max=paper_bar_max,
        paper_percentile_low=paper_percentile_low,
        paper_percentile_high=paper_percentile_high,
        mask_distance=mask_distance,
    )

    _plot_difference_axis(
        axes[1, 0],
        mds_solution,
        xg,
        yg,
        diff_mom_mds,
        title=f"Normalized Diff (MoM vs {mds_label})",
        mask_distance=mask_distance,
    )
    _plot_difference_axis(
        axes[1, 1],
        mds_solution,
        xg,
        yg,
        diff_mar_mds,
        title=f"Normalized Diff (MAR vs {mds_label})",
        mask_distance=mask_distance,
    )

    if mds_pat.size != 0 or mom_pat.size != 0 or mar_pat.size != 0:
        ax_ff = axes[1, 2]
        ax_ff.set_title("Far Field Comparison")
        ax_ff.set_xlabel("phi [deg]")
        ax_ff.set_ylabel("Normalized |Phi| [dB]")
        ax_ff.set_ylim(-60.0, 1.0)
        ax_ff.grid(True, alpha=0.3)
        phi_deg = np.rad2deg(phi_arr)

        def _add_pattern(p, color, label, lw=1.5):
            if p.size > 0:
                max_p = float(np.max(np.abs(p)))
                pdb = 20.0 * np.log10(np.abs(p) / max_p) if max_p > 0.0 else np.full(p.shape, -60.0, dtype=np.float64)
                ax_ff.plot(phi_deg, pdb, color, label=label, linewidth=lw)

        _add_pattern(mds_pat, "k-", mds_label, lw=2.5)
        _add_pattern(mom_pat, "r--", "MoM", lw=1.5)
        _add_pattern(mar_pat, "b:", "MAR", lw=1.5)
        ax_ff.legend(loc="upper right")

        if notes:
            ax_ff.text(
                0.03,
                0.03,
                "\n".join(notes),
                transform=ax_ff.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5", alpha=0.9),
            )
    else:
        _fill_note_axis(axes[1, 2], title="Notes", lines=notes or ["No far-field data available."])

    if figure_title is not None:
        fig.suptitle(figure_title)
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
