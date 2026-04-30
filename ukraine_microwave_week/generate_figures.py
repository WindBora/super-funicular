"""Generate publication figures for the Ukrainian Microwave Week paper.

Problem: E-wave plane-wave scattering by a sinusoidal open PEC strip,
analysed by the Method of Discrete Singularities (MDS).

Figures produced
----------------
fig1_flatstrip.png  -- flat strip (A=0) near-field + far-field  (validation)
fig2_sinusoidal.png -- sinusoidal strip near-field + far-field with grating markers
fig3_sweep.png      -- forward far-field pattern evolution with A and nu
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src2.geometry import SinusoidalStrip
from src2.solver import PlaneWave, MultiReflectorMDS

OUT = Path(__file__).parent
K = 2.0 * np.pi      # wavenumber (lambda = 1)
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

def make_solver(A: float, nu: float, n: int = N_MDS) -> MultiReflectorMDS:
    curve = SinusoidalStrip(
        x_center=0.0, y_base=Y_BASE, length=L,
        amplitude=A, frequency=nu, phase_rad=0.0,
    )
    return MultiReflectorMDS(
        reflectors=[curve],
        incident=PlaneWave(k=K, beta_rad=BETA_INC),
        n=n,
    )


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


def forward_grating_angles(nu: float, max_order: int = 8):
    """Return propagating forward (upper half-space) grating-order (m, phi_deg) pairs.

    For normal incidence (beta = pi/2) on a horizontal grating with period 1/nu:
      phi_m = arccos(m * nu)   (in [0, pi] for forward hemisphere).
    """
    orders, angles = [], []
    for m in range(-max_order, max_order + 1):
        if abs(m * nu) < 1.0:
            phi_m = float(np.degrees(np.arccos(np.clip(m * nu, -1.0, 1.0))))
            orders.append(m)
            angles.append(phi_m)
    return orders, angles


# ---------------------------------------------------------------------------
# Figure 1: flat strip (A = 0) — validation
# ---------------------------------------------------------------------------

def fig1_flatstrip():
    print('[fig1] solving flat strip (A=0) ...')
    sol = make_solver(A=0.0, nu=0.20).solve()          # nu irrelevant for A=0
    print(f'[fig1] boundary residual = {sol.boundary_residual_max:.2e}')
    sigma = scattering_width(sol)
    print(f'[fig1] sigma_total = {sigma:.4f} lambda  (PO limit = {2*L:.1f})')

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

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    # --- near field ---
    ax = axes[0]
    im = ax.imshow(intensity_db(u_tot),
                   extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   origin='lower', aspect='equal',
                   cmap='jet', vmin=-45, vmax=0, interpolation='bilinear')
    t_plot = np.linspace(-1.0, 1.0, 400)
    xc, yc = sol.solver.reflectors[0].coords(t_plot)
    ax.plot(xc, yc, 'w-', linewidth=1.8, zorder=6, label='Strip')
    # plane-wave direction arrow (coming from below)
    ax.annotate('', xy=(0.0, -1.5), xytext=(0.0, -4.0),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.text(0.3, -2.8, r'$\mathbf{k}$', color='white', fontsize=9)
    ax.set_xlabel(r'$x/\lambda$')
    ax.set_ylabel(r'$y/\lambda$')
    ax.set_title(r'(a) Flat strip: $|U_{\rm tot}|^2$ (dB)')
    fig.colorbar(im, ax=ax, shrink=0.90, pad=0.01, label='dB')

    # --- far field ---
    ax2 = axes[1]
    ax2.plot(np.rad2deg(phi), pdb, 'b-', linewidth=0.9)
    ax2.axvline(90,  color='k', linestyle='--', lw=0.9, alpha=0.7, label=r'Forward ($\varphi=90°$)')
    ax2.axvline(270, color='r', linestyle='--', lw=0.9, alpha=0.7, label=r'Backward ($\varphi=270°$)')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-50, 1)
    ax2.set_xlabel(r'$\varphi$ (deg)')
    ax2.set_ylabel(r'Norm. $|\Phi_{\rm sc}|$ (dB)')
    ax2.set_title('(b) Flat strip: far-field pattern')
    ax2.legend(fontsize=7, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0, 90, 180, 270, 360])

    fig.savefig(OUT / 'fig1_flatstrip.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[fig1] saved  fig1_flatstrip.png')
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

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    # --- near field ---
    ax = axes[0]
    im = ax.imshow(intensity_db(u_tot),
                   extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   origin='lower', aspect='equal',
                   cmap='jet', vmin=-45, vmax=0, interpolation='bilinear')
    t_plot = np.linspace(-1.0, 1.0, 600)
    xc, yc = sol.solver.reflectors[0].coords(t_plot)
    ax.plot(xc, yc, 'w-', linewidth=2.0, zorder=6)
    ax.plot(xc, yc, color='#ff55ff', linewidth=1.3, zorder=7)
    ax.annotate('', xy=(0.0, -A - 0.5), xytext=(0.0, -A - 2.0),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.text(0.3, -A - 1.3, r'$\mathbf{k}$', color='white', fontsize=9)
    ax.set_xlabel(r'$x/\lambda$')
    ax.set_ylabel(r'$y/\lambda$')
    ax.set_title(fr'(a) Sinusoidal: $A={A}\lambda$, $\nu={nu}\,\lambda^{{-1}}$')
    fig.colorbar(im, ax=ax, shrink=0.90, pad=0.01, label='dB')

    # --- far field with grating-order markers ---
    ax2 = axes[1]
    phi_deg = np.rad2deg(phi)
    ax2.plot(phi_deg, pdb, 'b-', linewidth=0.9, label='Scattered field', zorder=3)

    marker_colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    orders, angles = forward_grating_angles(nu, max_order=3)
    label_set = set()
    for m, phi_m in zip(orders, angles):
        if m == 0:
            continue
        col = marker_colors[abs(m) - 1] if abs(m) <= len(marker_colors) else 'gray'
        sign = '+' if m > 0 else ''
        lbl = fr'$m={sign}{m}$ fwd ({phi_m:.0f}°)'
        if lbl not in label_set:
            ax2.axvline(phi_m, color=col, linestyle=':', lw=1.1, alpha=0.85, label=lbl)
            label_set.add(lbl)

    ax2.axvline(90,  color='k', linestyle='--', lw=0.9, alpha=0.6, label=r'$m=0$ fwd (90°)')
    ax2.axvline(270, color='gray', linestyle='--', lw=0.9, alpha=0.6, label=r'$m=0$ bwd (270°)')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-50, 1)
    ax2.set_xlabel(r'$\varphi$ (deg)')
    ax2.set_ylabel(r'Norm. $|\Phi_{\rm sc}|$ (dB)')
    ax2.set_title('(b) Far-field pattern with grating orders')
    ax2.legend(fontsize=6.5, loc='lower left', ncol=1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0, 90, 180, 270, 360])

    fig.savefig(OUT / 'fig2_sinusoidal.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[fig2] saved  fig2_sinusoidal.png')


# ---------------------------------------------------------------------------
# Figure 3: parametric sweep sigma_norm(A, nu)
# ---------------------------------------------------------------------------

def fig3_pattern_evolution():
    """Forward-hemisphere far-field patterns showing grating-order emergence.

    Left panel: vary A at fixed nu=0.20.
    Right panel: vary nu at fixed A=1.5.
    """
    print('[fig3] pattern-evolution sweep ...')
    phi = np.linspace(0.0, np.pi, 2048)          # forward hemisphere only
    phi_deg = np.rad2deg(phi)

    palette_A  = ['#444444', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    palette_nu = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    # ---- Left: vary A, nu fixed ----
    nu_fixed = 0.20
    amplitudes_plot = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    ax1 = axes[0]
    for A, col in zip(amplitudes_plot, palette_A):
        sol = make_solver(A, nu_fixed).solve()
        psc = sol.far_field_pattern(phi, total=False)
        pmax = float(np.max(np.abs(psc)))
        pdb = 20.0 * np.log10(np.maximum(np.abs(psc) / pmax, 1e-7))
        lbl = fr'$A={A}\lambda$' if A > 0 else r'$A=0$ (flat)'
        print(f'  [fig3-left]  A={A:.1f}, nu={nu_fixed:.2f}:  res={sol.boundary_residual_max:.1e}')
        ax1.plot(phi_deg, pdb, color=col, lw=0.95, label=lbl)

    orders, angles = forward_grating_angles(nu_fixed, max_order=3)
    for m, phi_m in zip(orders, angles):
        if abs(m) == 1:
            ax1.axvline(phi_m, color='gray', linestyle=':', lw=0.85, alpha=0.8)
            ax1.text(phi_m + 0.8, -46, fr'$m={m:+d}$', fontsize=6, color='gray', ha='left')
    ax1.axvline(90.0, color='k', linestyle='--', lw=0.85, alpha=0.7)
    ax1.text(91, -46, r'$m=0$', fontsize=6, color='k', ha='left')
    ax1.set_xlim(0, 180)
    ax1.set_ylim(-50, 1)
    ax1.set_xlabel(r'$\varphi$ (deg)')
    ax1.set_ylabel(r'Norm. $|\Phi_{\rm sc}|$ (dB)')
    ax1.set_title(fr'(a) Fixed $\nu={nu_fixed}\,\lambda^{{-1}}$, vary $A$')
    ax1.legend(fontsize=6.5, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([0, 45, 90, 135, 180])

    # ---- Right: vary nu, A fixed ----
    A_fixed = 1.5
    freqs_plot = [0.10, 0.15, 0.20, 0.25, 0.30]
    ax2 = axes[1]
    for nu, col in zip(freqs_plot, palette_nu):
        sol = make_solver(A_fixed, nu).solve()
        psc = sol.far_field_pattern(phi, total=False)
        pmax = float(np.max(np.abs(psc)))
        pdb = 20.0 * np.log10(np.maximum(np.abs(psc) / pmax, 1e-7))
        lbl = fr'$\nu={nu}\,\lambda^{{-1}}$'
        print(f'  [fig3-right] A={A_fixed:.1f}, nu={nu:.2f}:  res={sol.boundary_residual_max:.1e}')
        ax2.plot(phi_deg, pdb, color=col, lw=0.95, label=lbl)
        # mark first-order angle
        if abs(nu) < 1.0:
            phi_1 = float(np.degrees(np.arccos(min(nu, 0.9999))))
            ax2.axvline(phi_1, color=col, linestyle=':', lw=0.7, alpha=0.6)

    ax2.axvline(90.0, color='k', linestyle='--', lw=0.85, alpha=0.7, label=r'$m=0$ (90°)')
    ax2.set_xlim(0, 180)
    ax2.set_ylim(-50, 1)
    ax2.set_xlabel(r'$\varphi$ (deg)')
    ax2.set_title(fr'(b) Fixed $A={A_fixed}\,\lambda$, vary $\nu$')
    ax2.legend(fontsize=6.5, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0, 45, 90, 135, 180])

    fig.savefig(OUT / 'fig3_sweep.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[fig3] saved  fig3_sweep.png')


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
    print(f'   PO limit: {2*L:.4f}')


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
