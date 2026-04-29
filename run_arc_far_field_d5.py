"""
Standalone far-field radiation pattern evaluation for a small parabolic antenna.
Aperture d = 5 wavelengths.
"""

import numpy as np
import matplotlib.pyplot as plt

from src2.geometry import ParabolaArc
from src2.solver import ComplexSourcePoint, MultiReflectorMAR as MultiReflectorMoM # MultiReflectorMoM


def main():
    # 1. Antenna parameters
    aperture = 5.0
    focal_ratio = 0.5
    f = focal_ratio * aperture
    wavelength = 1.0
    k = 2.0 * np.pi / wavelength
    
    # CSP parameters: kb = 2.5 provides ~ -10 dB edge illumination
    kb = 2.5 
    
    # 2. Setup the Arc Geometry and Feed
    arc = ParabolaArc.symmetric(f=f, aperture=aperture)
    csp = ComplexSourcePoint(
        k=k, x0=f, y0=0.0, b=kb / k, beta_rad=np.pi 
    )
    
    # 3. Initialize and run the MoM solver
    print(f"Solving MoM for Parabolic Arc (aperture={aperture}λ, f={f}λ)...")
    solver = MultiReflectorMoM(reflectors=[arc], incident=csp, n=120)
    solution = solver.solve()
    
    # 4. Extract far-field metrics
    directivity, peak_phi, phi, pattern = solution.directivity(num_angles=2048)
    edge_left, edge_right = solution.edge_illumination_db(reflector_index=0)
    
    print(f"Solver finished. Max boundary residual: {solution.boundary_residual_max:.4e}")
    print(f"Directivity: {directivity:.2f}")
    print(f"Edge illumination: {edge_left:.2f} dB")
    
    # 5. Plotting the Radiation Pattern
    # Shift angles from [0, 360) to [-180, 180) to center the main beam
    phi_deg = np.rad2deg(phi)
    phi_deg[phi_deg > 180.0] -= 360.0
    
    # Sort the arrays so the line plots correctly across the -180 to 180 range
    sort_idx = np.argsort(phi_deg)
    phi_deg = phi_deg[sort_idx]
    pattern = pattern[sort_idx]
    
    # Normalize pattern to 0 dB at the peak
    max_mag = np.max(np.abs(pattern))
    pattern_db = 20.0 * np.log10(np.maximum(np.abs(pattern) / max_mag, 1e-12))
    
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(phi_deg, pattern_db, color="blue", linewidth=1.5, label=f"d = {aperture}λ, kb = {kb}")
    
    ax.set_title("Radiation Pattern (Диаграмма Направленности)")
    ax.set_xlabel("Angle [degrees]")
    ax.set_ylabel("Normalized |Phi| [dB]")
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-60.0, 0.0)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.legend(loc="upper right")
    
    plt.savefig("d:\\uni\\PhD\\paper_1\\far_field_d5.png", dpi=200)
    print("Plot saved to far_field_d5.png")
    plt.show()

if __name__ == "__main__":
    main()