"""Generate publication figures for the Ukrainian Microwave Week paper.

Problem: E-wave plane-wave scattering by a sinusoidal open PEC strip,
analysed by the Method of Discrete Singularities (MDS).

Figures produced
----------------
fig1_flatstrip_near.png      -- flat strip near-field validation
fig2_flatstrip_far.png       -- flat strip far-field validation
fig3_sinusoidal_near.png     -- sinusoidal strip near field
fig4_sinusoidal_far.png      -- sinusoidal strip far field with Floquet markers
fig5_amplitude_sweep.png     -- reflected-hemisphere evolution with A
fig6_frequency_sweep.png     -- reflected-hemisphere evolution with nu
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src2.geometry import SinusoidalStrip
from src2.solver import PlaneWave, MultiReflectorPaperMDS


OUT = Path(__file__).parent
K = 2.0 * np.pi       # wavenumber (lambda = 1)
# With the suppressed exp(-i*omega*t) factor, beta is the physical direction
# of exp(+i*k*d.r).  This publication case illuminates the strip from below.
BETA_INC = np.pi / 2  # normal incidence: wave propagates in +y direction
L = 10.0              # strip horizontal length (lambda)
Y_BASE = 0.0          # strip baseline y-coordinate
N_MDS = 200           # MDS unknowns per reflector

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.2,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_solver(A: float, nu: float, n: int = N_MDS) -> MultiReflectorPaperMDS:
    curve = SinusoidalStrip(
        x_center=0.0, y_base=Y_BASE, length=L,
        amplitude=A, frequency=nu, phase_rad=0.0,
    )
    return MultiReflectorPaperMDS(
        reflectors=[curve],
        incident=PlaneWave(k=K, beta_rad=BETA_INC),
        n=n,
    )


def draw_plane_wave_arrow(ax, incident: PlaneWave, *, color: str = 'white') -> None:
    """Draw ``k`` from the incident side in the physical propagation direction.

    Arrow geometry is derived from the same direction components used by
    ``PlaneWave.field``.  This keeps the figure correct if the incidence angle
    changes and avoids a hard-coded arrow that can disagree with the phasor.
    """

    x_min, x_max = (float(value) for value in ax.get_xlim())
    y_min, y_max = (float(value) for value in ax.get_ylim())
    width = x_max - x_min
    height = y_max - y_min
    direction = np.array([incident.direction_x, incident.direction_y], dtype=float)
    direction /= np.linalg.norm(direction)

    margin_x = 0.04 * width
    margin_y = 0.04 * height
    length = 0.18 * min(width, height)
    start_x = (
        0.5 * (x_min + x_max)
        if abs(direction[0]) <= 1.0e-12
        else (x_min + margin_x if direction[0] > 0.0 else x_max - margin_x)
    )
    start_y = (
        0.5 * (y_min + y_max)
        if abs(direction[1]) <= 1.0e-12
        else (y_min + margin_y if direction[1] > 0.0 else y_max - margin_y)
    )
    start = np.array([start_x, start_y])
    end = start + length * direction

    ax.annotate(
        '',
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
    )
    midpoint = 0.5 * (start + end)
    label_offset = 0.035 * min(width, height) * np.array([-direction[1], direction[0]])
    ax.text(*(midpoint + label_offset), r'$\mathbf{k}$', color=color, fontsize=9)


def intensity_db(u: np.ndarray, floor: float = -45.0) -> np.ndarray:
    I = np.abs(u) ** 2
    Imax = float(np.nanmax(I))
    if Imax == 0.0:
        return np.full(I.shape, floor)
    with np.errstate(divide='ignore', invalid='ignore'):
        db = 10.0 * np.log10(I / Imax)
    return np.maximum(db, floor)


def scattering_width(sol, n_phi: int = 4096) -> float:
    """Compute normalised total 2-D scattering width sigma (in lambda units).

    sigma = (1/pi) * integral |Phi_sc(phi)|^2 d_phi.
    This equals the standard 2-D total scattering cross-width per unit amplitude
    for a unity-amplitude plane wave (|U0|=1).
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    psc = sol.far_field_pattern(phi, total=False)
    dphi = 2.0 * np.pi / n_phi
    return float(np.sum(np.abs(psc) ** 2) * dphi) / np.pi


def floquet_reference_angles(nu: float, max_order: int = 8):
    """Return propagating forward and backward Floquet-reference angles.

    Angles are measured counter-clockwise from +x.  For incidence along +y,
    ``acos(m*nu)`` is the forward (+y) branch and ``360-acos(m*nu)`` is
    the reflected/backward (-y) branch of ``cos(phi_m) = m*nu``.
    """
    orders, forward_angles, backward_angles = [], [], []
    for m in range(-max_order, max_order + 1):
        if abs(m * nu) < 1.0:
            phi_forward = float(np.degrees(np.arccos(np.clip(m * nu, -1.0, 1.0))))
            orders.append(m)
            forward_angles.append(phi_forward)
            backward_angles.append(360.0 - phi_forward)
    return orders, forward_angles, backward_angles


# ---------------------------------------------------------------------------
# Figure 1: flat strip (A = 0) — validation
# ---------------------------------------------------------------------------

def fig1_flatstrip():
    print('[fig1] solving flat strip (A=0) ...')
    sol = make_solver(A=0.0, nu=0.20).solve()          # nu irrelevant for A=0
    print(f'[fig1] boundary residual = {sol.boundary_residual_max:.2e}')
    sigma = scattering_width(sol)
    print(f'[fig1] sigma_total = {sigma:.4f} lambda')

    # near-field grid
    xs = np.linspace(-6.5, 6.5, 440)
    ys = np.linspace(-4.2, 5.0, 380)
    xg, yg = np.meshgrid(xs, ys)
    u_tot = sol.near_field(xg, yg, total=True)

    # far-field
    phi = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    psc = sol.far_field_pattern(phi, total=False)
    pmax = float(np.max(np.abs(psc)))
    pdb = 20.0 * np.log10(np.maximum(np.abs(psc) / pmax, 1e-7))

    # --- near field ---
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.8), constrained_layout=True)
    im = ax.imshow(intensity_db(u_tot),
                   extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   origin='lower', aspect='equal',
                   cmap='jet', vmin=-45, vmax=0, interpolation='bilinear')
    t_plot = np.linspace(-1.0, 1.0, 400)
    xc, yc = sol.solver.reflectors[0].coords(t_plot)
    ax.plot(xc, yc, 'w-', linewidth=1.8, zorder=6, label='Strip')
    draw_plane_wave_arrow(ax, sol.solver.incident)
    ax.set_xlabel(r'$x/\lambda$')
    ax.set_ylabel(r'$y/\lambda$')
    ax.set_title(r'Flat strip: $|U_{\rm tot}|^2$ (dB)')
    fig.colorbar(im, ax=ax, shrink=0.90, pad=0.01, label='dB')
    fig.savefig(OUT / 'fig1_flatstrip_near.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- far field ---
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.55), constrained_layout=True)
    ax.plot(np.rad2deg(phi), pdb, 'b-', linewidth=0.9)
    ax.axvline(90,  color='k', linestyle='--', lw=0.9, alpha=0.7, label=r'Forward ($\varphi=90^\circ$)')
    ax.axvline(270, color='r', linestyle='--', lw=0.9, alpha=0.7, label=r'Backward ($\varphi=270^\circ$)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-50, 1)
    ax.set_xlabel(r'$\varphi$ (deg)')
    ax.set_ylabel(r'Norm. $|\Phi_{\rm sc}|$ (dB)')
    ax.set_title('Flat strip: far-field pattern')
    ax.legend(fontsize=6.5, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 90, 180, 270, 360])

    fig.savefig(OUT / 'fig2_flatstrip_far.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[fig1] saved  fig1_flatstrip_near.png, fig2_flatstrip_far.png')
    return sigma


# ---------------------------------------------------------------------------
# Figure 2: sinusoidal strip best case — near field + far field
# ---------------------------------------------------------------------------

def fig2_sinusoidal(A: float = 1.5, nu: float = 0.20):
    print(f'[fig2] solving sinusoidal strip A={A}, nu={nu} ...')
    sol = make_solver(A, nu).solve()
    print(f'[fig2] boundary residual = {sol.boundary_residual_max:.2e}')

    xs = np.linspace(-6.5, 6.5, 440)
    ys = np.linspace(-A - 2.0, A + 4.5, 380)
    xg, yg = np.meshgrid(xs, ys)
    u_tot = sol.near_field(xg, yg, total=True)

    phi = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    psc = sol.far_field_pattern(phi, total=False)
    pmax = float(np.max(np.abs(psc)))
    pdb = 20.0 * np.log10(np.maximum(np.abs(psc) / pmax, 1e-7))

    # --- near field ---
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.8), constrained_layout=True)
    im = ax.imshow(intensity_db(u_tot),
                   extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   origin='lower', aspect='equal',
                   cmap='jet', vmin=-45, vmax=0, interpolation='bilinear')
    t_plot = np.linspace(-1.0, 1.0, 600)
    xc, yc = sol.solver.reflectors[0].coords(t_plot)
    ax.plot(xc, yc, 'w-', linewidth=2.0, zorder=6)
    ax.plot(xc, yc, color='#ff55ff', linewidth=1.3, zorder=7)
    draw_plane_wave_arrow(ax, sol.solver.incident)
    ax.set_xlabel(r'$x/\lambda$')
    ax.set_ylabel(r'$y/\lambda$')
    ax.set_title(fr'Sinusoidal: $A={A}\lambda$, $\nu={nu}\,\lambda^{{-1}}$')
    fig.colorbar(im, ax=ax, shrink=0.90, pad=0.01, label='dB')
    fig.savefig(OUT / 'fig3_sinusoidal_near.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- far field with Floquet-order markers ---
    fig, ax2 = plt.subplots(1, 1, figsize=(3.45, 2.55), constrained_layout=True)
    phi_deg = np.rad2deg(phi)
    ax2.plot(phi_deg, pdb, 'b-', linewidth=0.9, label='Scattered field', zorder=3)

    marker_colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    orders, _, backward_angles = floquet_reference_angles(nu, max_order=3)
    label_set = set()
    for m, phi_m in zip(orders, backward_angles):
        if m == 0:
            continue
        col = marker_colors[abs(m) - 1] if abs(m) <= len(marker_colors) else 'gray'
        sign = '+' if m > 0 else ''
        lbl = fr'$m={sign}{m}$ refl. ({phi_m:.0f}$^\circ$)'
        if lbl not in label_set:
            ax2.axvline(phi_m, color=col, linestyle=':', lw=1.1, alpha=0.85, label=lbl)
            label_set.add(lbl)

    ax2.axvline(90,  color='k', linestyle='--', lw=0.9, alpha=0.6, label=r'$m=0$ fwd (90$^\circ$)')
    ax2.axvline(270, color='gray', linestyle='--', lw=0.9, alpha=0.6, label=r'$m=0$ bwd (270$^\circ$)')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-50, 1)
    ax2.set_xlabel(r'$\varphi$ (deg)')
    ax2.set_ylabel(r'Norm. $|\Phi_{\rm sc}|$ (dB)')
    ax2.set_title('Far-field pattern with Floquet markers')
    ax2.legend(fontsize=6.5, loc='lower left', ncol=1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0, 90, 180, 270, 360])

    fig.savefig(OUT / 'fig4_sinusoidal_far.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[fig2] saved  fig3_sinusoidal_near.png, fig4_sinusoidal_far.png')


# ---------------------------------------------------------------------------
# Figures 5 and 6: reflected-pattern evolution with A and nu
# ---------------------------------------------------------------------------

def fig3_pattern_evolution():
    """Reflected-hemisphere patterns showing finite-strip Floquet lobes.

    Figure 5 varies A at fixed nu=0.20; Figure 6 varies nu at fixed A=1.5.
    """
    print('[fig3] pattern-evolution sweep ...')
    # For +y incidence, pi <= phi <= 2*pi is the reflected/backward side.
    phi = np.linspace(np.pi, 2.0 * np.pi, 2048)
    phi_deg = np.rad2deg(phi)

    palette_A  = ['#444444', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    palette_nu = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # ---- Left: vary A, nu fixed ----
    nu_fixed = 0.20
    amplitudes_plot = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    fig, ax1 = plt.subplots(1, 1, figsize=(3.45, 2.55), constrained_layout=True)
    for A, col in zip(amplitudes_plot, palette_A):
        sol = make_solver(A, nu_fixed).solve()
        psc = sol.far_field_pattern(phi, total=False)
        pmax = float(np.max(np.abs(psc)))
        pdb = 20.0 * np.log10(np.maximum(np.abs(psc) / pmax, 1e-7))
        lbl = fr'$A={A}\lambda$' if A > 0 else r'$A=0$ (flat)'
        print(f'  [fig3-left]  A={A:.1f}, nu={nu_fixed:.2f}:  res={sol.boundary_residual_max:.1e}')
        ax1.plot(phi_deg, pdb, color=col, lw=0.95, label=lbl)

    orders, _, backward_angles = floquet_reference_angles(nu_fixed, max_order=3)
    for m, phi_m in zip(orders, backward_angles):
        if abs(m) == 1:
            ax1.axvline(phi_m, color='gray', linestyle=':', lw=0.85, alpha=0.8)
            ax1.text(
                phi_m,
                -48,
                fr'$m={m:+d}$',
                fontsize=6,
                color='gray',
                rotation=90,
                ha='center',
                va='bottom',
            )
    ax1.axvline(270.0, color='k', linestyle='--', lw=0.85, alpha=0.7)
    ax1.text(270, -48, r'$m=0$', fontsize=6, color='k', rotation=90, ha='center', va='bottom')
    ax1.set_xlim(180, 360)
    ax1.set_ylim(-50, 1)
    ax1.set_xlabel(r'$\varphi$ (deg)')
    ax1.set_ylabel(r'Norm. $|\Phi_{\rm sc}|$ (dB)')
    ax1.set_title(fr'Fixed $\nu={nu_fixed}\,\lambda^{{-1}}$, vary $A$')
    ax1.legend(fontsize=6.5, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([180, 225, 270, 315, 360])
    fig.savefig(OUT / 'fig5_amplitude_sweep.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---- Right: vary nu, A fixed ----
    A_fixed = 1.5
    freqs_plot = [0.10, 0.15, 0.20, 0.25, 0.30]
    fig, ax2 = plt.subplots(1, 1, figsize=(3.45, 2.55), constrained_layout=True)
    for nu, col in zip(freqs_plot, palette_nu):
        sol = make_solver(A_fixed, nu).solve()
        psc = sol.far_field_pattern(phi, total=False)
        pmax = float(np.max(np.abs(psc)))
        pdb = 20.0 * np.log10(np.maximum(np.abs(psc) / pmax, 1e-7))
        lbl = fr'$\nu={nu}\,\lambda^{{-1}}$'
        print(f'  [fig3-right] A={A_fixed:.1f}, nu={nu:.2f}:  res={sol.boundary_residual_max:.1e}')
        ax2.plot(phi_deg, pdb, color=col, lw=0.95, label=lbl)
        # Mark both reflected first-order references for this frequency.
        if abs(nu) < 1.0:
            phi_plus_1 = 360.0 - float(np.degrees(np.arccos(min(nu, 0.9999))))
            phi_minus_1 = 360.0 - float(np.degrees(np.arccos(max(-nu, -0.9999))))
            ax2.axvline(phi_plus_1, color=col, linestyle=':', lw=0.7, alpha=0.6)
            ax2.axvline(phi_minus_1, color=col, linestyle=':', lw=0.7, alpha=0.6)

    ax2.axvline(270.0, color='k', linestyle='--', lw=0.85, alpha=0.7, label=r'$m=0$ (270$^\circ$)')
    ax2.set_xlim(180, 360)
    ax2.set_ylim(-50, 1)
    ax2.set_xlabel(r'$\varphi$ (deg)')
    ax2.set_title(fr'Fixed $A={A_fixed}\,\lambda$, vary $\nu$')
    ax2.legend(fontsize=6.5, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([180, 225, 270, 315, 360])

    fig.savefig(OUT / 'fig6_frequency_sweep.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[fig3] saved  fig5_amplitude_sweep.png, fig6_frequency_sweep.png')


# ---------------------------------------------------------------------------
# N-convergence of sigma for the flat strip (for paper Table I)
# ---------------------------------------------------------------------------

def print_convergence_table():
    print('\n=== N-convergence: flat strip, A=0, L=10 lambda ===')
    print(f'{"N":>6}  {"sigma (lambda)":>16}  {"rel. change":>12}  {"residual":>12}')
    prev = None
    for n in [32, 64, 100, 150, 200]:
        sol = make_solver(A=0.0, nu=0.20, n=n).solve()
        sig = scattering_width(sol)
        rel = abs(sig - prev) / abs(prev) if prev is not None else float('nan')
        print(f'{n:>6}  {sig:>16.6f}  {rel:>12.2e}  {sol.boundary_residual_max:>12.2e}')
        prev = sig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Ukrainian Microwave Week — plane-wave figures ===\n')

    print('--- Figure 1: flat strip (validation) ---')
    sigma_flat = fig1_flatstrip()

    print('\n--- Figure 2: sinusoidal strip best case ---')
    fig2_sinusoidal(A=1.5, nu=0.20)

    print('\n--- Figure 3: pattern evolution ---')
    fig3_pattern_evolution()

    print_convergence_table()

    print('\n=== Done. Figures saved to', OUT, '===')
