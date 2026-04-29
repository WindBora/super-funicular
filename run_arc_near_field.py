"""
Standalone near-field evaluation for a parabolic arc reflector.
Based on the canonical setup from Nosich & Gandel (2007).
"""

import numpy as np
import matplotlib.pyplot as plt

from src2.geometry import ParabolaArc
from src2.solver import ComplexSourcePoint, MultiReflectorMAR as MultiReflectorMoM # MultiReflectorMoM


def main():
    # 1. Canonical parameters from the nosich2007 paper
    aperture = 30.0
    focal_ratio = 0.5
    f = focal_ratio * aperture
    wavelength = 1.0
    k = 2.0 * np.pi / wavelength
    
    # CSP parameters: kb = 2.5 provides ~ -10 dB edge illumination in this setup
    kb = 2.5 
    
    # 2. Setup the Arc Geometry
    # ParabolaArc.symmetric creates x = y^2 / (4f)
    # Vertex is at (0, 0) and the geometric focus is at (f, 0)
    arc = ParabolaArc.symmetric(f=f, aperture=aperture)
    
    # 3. Setup the Incident Field (CSP Feed)
    # Place the feed at the geometric focus, aimed towards the vertex (180 degrees)
    csp = ComplexSourcePoint(
        k=k,
        x0=f,
        y0=0.0,
        b=kb / k,
        beta_rad=np.pi 
    )
    
    # 4. Initialize and run the MoM solver
    print(f"Solving MoM for Parabolic Arc (aperture={aperture}λ, f={f}λ)...")
    solver = MultiReflectorMoM(reflectors=[arc], incident=csp, n=200)
    solution = solver.solve()
    print(f"Solver finished. Max boundary residual: {solution.boundary_residual_max:.4e}")
    
    # 5. Evaluate Near Field on a Cartesian Grid
    print("Evaluating near field...")
    nx, ny = 400, 400
    x_min, x_max = -5.0, 25.0
    y_min, y_max = -20.0, 20.0
    
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    xg, yg = np.meshgrid(xs, ys)
    
    # Leverage the existing near-field evaluation logic
    u_total = solution.near_field(xg, yg, total=True)
    
    # 6. Custom Plotting Routine (independent of src2.plotting)
    intensity = np.abs(u_total)**2
    
    # Fix: Instead of normalizing to the absolute mathematical peak (which occurs
    # exactly at the CSP feed and crushes the dynamic range), we normalize to the 
    # 99.5th percentile. This brings the incident/reflected beams back into view.
    raw_db = 10.0 * np.log10(np.maximum(intensity, 1e-12))
    vmax = np.percentile(raw_db, 99.5)
    intensity_db = raw_db - vmax
    
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    cax = ax.imshow(
        np.clip(intensity_db, -40.0, 0.0), # Clip at -40 dB for clean visualization
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="viridis", # Using a different colormap to distinguish from existing plots
        aspect="equal"
    )
    
    fig.colorbar(cax, ax=ax, label="Normalized Total Field Intensity [dB]")
    
    # Overlay the reflector contour
    t_plot = np.linspace(-1.0, 1.0, 500)
    xc, yc = arc.coords(t_plot)
    ax.plot(xc, yc, color="white", linewidth=2.5, label="Parabolic Arc")
    ax.plot(xc, yc, color="black", linewidth=1.0)
    
    # Overlay the feed focus
    ax.plot(f, 0.0, marker="*", color="cyan", markersize=10, markeredgecolor="black", label="CSP Feed")
    
    ax.set_title("Standalone Near-Field (Nosich 2007 Benchmark)")
    ax.set_xlabel("x [wavelengths]")
    ax.set_ylabel("y [wavelengths]")
    ax.legend(loc="upper right")
    
    plt.savefig("d:\\uni\\PhD\\paper_1\\custom_arc_near_field.png", dpi=200)
    print("Plot saved to custom_arc_near_field.png")
    plt.show()

if __name__ == "__main__":
    main()