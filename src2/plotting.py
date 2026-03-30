"""Plotting helpers for near-field and far-field visualizations."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import ParamCurve
from .solver import MDSSolution


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

    pattern_db = 20.0 * np.log10(np.abs(pattern) / np.max(np.abs(pattern)))
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
