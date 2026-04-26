"""Plot a 2D plane wave using Euler's formula.

Run from the repository root with:

    python plane_wave
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


SPEED_OF_LIGHT = 299_792_458.0


def main() -> None:
    """Animate a 2D plane wave from Euler's formula."""
    wavelength_nm = 600.0
    wavelength = wavelength_nm * 1e-9
    beta = np.deg2rad(35.0)

    k = 2.0 * np.pi / wavelength
    omega = k * SPEED_OF_LIGHT
    period = 2.0 * np.pi / omega
    time_step = period / 60.0

    half_width = 4.0 * wavelength
    x = np.linspace(-half_width, half_width, 500)
    y = np.linspace(-half_width, half_width, 500)
    xx, yy = np.meshgrid(x, y)
    x_um = x * 1e6
    y_um = y * 1e6

    spatial_phase = k * (xx * np.cos(beta) + yy * np.sin(beta))

    def plane_wave(time: float) -> np.ndarray:
        """Return Re{exp(-i * (k(x cos beta + y sin beta) + omega t))}."""
        complex_wave = np.exp(-1j * (spatial_phase + omega * time))
        return np.real(complex_wave)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        plane_wave(time=0.0),
        extent=[x_um.min(), x_um.max(), y_um.min(), y_um.max()],
        origin="lower",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )

    ax.arrow(
        0,
        0,
        -1.5 * wavelength * 1e6 * np.cos(beta),
        -1.5 * wavelength * 1e6 * np.sin(beta),
        width=0.015,
        color="black",
        length_includes_head=True,
    )
    ax.text(
        -2.3 * wavelength * 1e6,
        -2.6 * wavelength * 1e6,
        "propagation direction",
        color="black",
    )

    title = ax.set_title("")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Electric field amplitude")

    def update(frame: int):
        time = frame * time_step
        image.set_data(plane_wave(time))
        title.set_text(
            rf"2D plane wave, $\lambda={wavelength_nm:.0f}\,\mathrm{{nm}}$: "
            rf"$\mathrm{{Re}}\{{e^{{-i(k(x\cos\beta+y\sin\beta)+\omega t)}}\}}$, "
            rf"$t={time * 1e15:.2f}\,\mathrm{{fs}}$"
        )
        return image, title

    animation = FuncAnimation(fig, update, frames=300, interval=30, blit=False)
    fig.plane_wave_animation = animation

    fig.tight_layout()
    fig.canvas.draw()
    plt.show()


if __name__ == "__main__":
    main()
