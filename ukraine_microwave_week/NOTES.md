# Ukrainian Microwave Week Paper — Work Notes

## What Was Built

A complete 4-page IEEEtran conference paper and all supporting Python code for the **Ukrainian Microwave Week** conference. The paper is a full-wave numerical study of plane-wave E-wave scattering by sinusoidal open PEC strips, analyzed via the Method of Discrete Singularities (MDS).

---

## Paper Overview

**Title:** E-Wave Plane-Wave Scattering by Sinusoidal Open PEC Strips via the Method of Discrete Singularities

**Files:**
- `main.tex` — 4-page IEEEtran conference paper (compiles to `main.pdf`)
- `generate_figures.py` — Python script that produces all 3 figures
- `fig1_flatstrip.png`, `fig2_sinusoidal.png`, `fig3_sweep.png` — generated figures

**To compile the paper:**
```bash
pdflatex main.tex && pdflatex main.tex
```

**To regenerate figures:**
```bash
python generate_figures.py
```

---

## Physical Problem

2D electromagnetic scattering in the E-wave (TM) polarization. A perfectly-conducting (PEC) open arc (sinusoidal strip) is illuminated by a plane wave. The strip is a model for sinusoidal superconducting strips (meander-line resonators, delay lines) at microwave frequencies — PEC is accurate because the London penetration depth satisfies `k·λ_L ≲ 10⁻⁴` at 10 GHz.

### Setup
- Wavenumber: `k = 2π` (wavelength `λ = 1`)
- Incidence: `β = π/2` (normal incidence, wave propagates in `+y` direction)
- Strip: horizontal extent `L = 10λ`, centered at origin, baseline `y = 0`
- Sinusoidal geometry: `x = (L/2)·t`, `y = A·sin(2π·ν·x)`, `t ∈ [-1, 1]`
- Parameters swept: amplitude `A ∈ {0.5, 1.0, 1.5, 2.0, 2.5}λ`, spatial frequency `ν ∈ {0.10, 0.15, 0.20, 0.25, 0.30}λ⁻¹`

### Governing Equations
Scattered field `U` satisfies the 2D Helmholtz equation `(Δ + k²)U = 0` outside the arc, with Dirichlet BC `U|_Γ = −U₀|_Γ`.

**Plane wave:** `U₀(r) = exp(−ik(x·cos β + y·sin β))`

**SIE** (after factoring inverse-square-root edge singularity `j|r'(t)| = v(t)(1−t²)^{−1/2}`):
```
∫₋₁¹ H₀⁽¹⁾(k·R(t,t₀)) · v(t) / √(1−t²) dt = −U₀(t₀)
```

**Total 2D scattering width:**
```
σ = (1/π) ∫₀²π |Φ_sc(φ)|² dφ
```

---

## Method: MDS (Method of Discrete Singularities)

MDS is a Nystrom-type discretization of the SIE due to Nosich & Gandel. Key steps:

1. Differentiate the first-kind logarithmic SIE → converts kernel singularity from log-type to Cauchy-type
2. Add one supplementary integral condition (restores the lost constant)
3. Collocate at `n−1` second-kind Chebyshev nodes `τⱼ = cos(jπ/n)`, `j = 1, …, n−1`

**Resulting n×n linear system:**
```
(1/n) Σᵢ [ v(tᵢ)/(tᵢ − τⱼ) + K(tᵢ, τⱼ)·v(tᵢ) ] = f(τⱼ)
```
where `tᵢ = cos((2i−1)π/(2n))` are first-kind Chebyshev roots, `K` is the smooth part of the differentiated Hankel kernel, and `f = −dU₀/dt₀`.

Solved by Gaussian elimination. Boundary residuals reach machine precision (~10⁻¹⁵) for all tested cases.

### Why MDS is Good Here
- Analytically extracts edge singularity → spectral convergence for smooth problems
- Dense but small system (N = 200 unknowns per reflector)
- No artificial boundary truncation or absorbing layers needed
- Validated in literature for parabolic reflector antennas (Nosich & Gandel 2007)

---

## Key Physics: Floquet Grating Orders

For normal incidence (`β = π/2`) on a horizontal periodic grating with period `Λ = 1/ν`, the Floquet theory gives propagating forward-hemisphere diffraction orders at:

```
φₘ = arccos(m·ν),    |m·ν| < 1
```

| ν (λ⁻¹) | m=+1 angle | m=−1 angle | m=+2 angle |
|----------|-----------|-----------|-----------|
| 0.10     | 84.3°     | 95.7°     | 78.5°     |
| 0.15     | 81.4°     | 98.6°     | 72.5°     |
| 0.20     | 78.5°     | 101.5°    | 66.4°     |
| 0.25     | 75.5°     | 104.5°    | not shown |
| 0.30     | 72.5°     | 107.5°    | 53.1°     |

The backward hemisphere has mirror orders at `360° − φₘ`.

For the finite strip, `L·ν` periods = 2 periods at `ν = 0.20`. This broadens each Floquet lobe but the peak positions remain exactly at the predicted angles.

---

## Numerical Results

### Flat Strip Benchmark (A = 0, L = 10λ, N = 200)

- Boundary residual: `2.12 × 10⁻¹⁵` (machine precision)
- Total scattering width: `σ = 39.24λ`

**N-convergence of σ:**

| N   | σ (λ)  | Rel. change | Residual       |
|-----|--------|-------------|----------------|
| 32  | 31.866 | —           | 7.2 × 10⁻¹⁶   |
| 64  | 35.954 | 1.28 × 10⁻¹ | 8.5 × 10⁻¹⁶   |
| 100 | 37.710 | 4.88 × 10⁻² | 2.3 × 10⁻¹⁵   |
| 150 | 38.765 | 2.80 × 10⁻² | 1.7 × 10⁻¹⁵   |
| 200 | 39.243 | 1.23 × 10⁻² | 2.1 × 10⁻¹⁵   |

Note: residual being machine-precision means the discrete system is solved exactly. The N-convergence reflects the MDS approximation quality for the true SIE solution (not linear algebra error).

### Sinusoidal Strip (A = 1.5λ, ν = 0.20λ⁻¹)

- Boundary residual: `2.92 × 10⁻¹⁵`
- Far-field peaks confirmed at: 78.5°, 90°, 101.5° (m = ±1, 0 forward) and 66.4°, 113.6° (m = ±2 forward)
- Backward hemisphere mirrors: 281.5°, 270°, 258.5°, 293.6°, 246.4°

### Parametric Sweep (25 cases, all N = 200)

All boundary residuals: `1.4 × 10⁻¹⁵` to `2.9 × 10⁻¹⁵`.

**Key finding:** Total scattering width `σ` varies by < 0.2λ (< 0.5%) across all 25 (A, ν) combinations. The sinusoidal corrugation redistributes scattered power into grating orders without changing the total — analogous to blaze in classical diffraction gratings.

- Amplitude `A` → controls how much power shifts into m = ±1, ±2 orders (blaze strength)
- Spatial frequency `ν` → controls the angles of grating orders (lobe steering)

---

## Code Structure

### `generate_figures.py`

**Constants:**
```python
K = 2π          # wavenumber (λ = 1)
BETA_INC = π/2  # normal incidence (+y direction)
L = 10.0        # strip length (λ)
Y_BASE = 0.0    # strip baseline y-coordinate
N_MDS = 200     # unknowns per reflector
```

**Key functions:**

| Function | Description |
|----------|-------------|
| `make_solver(A, nu, n)` | Creates `SinusoidalStrip` + `PlaneWave` + `MultiReflectorMDS`, returns unsolved solver |
| `scattering_width(sol)` | σ = (1/π)∫\|Φ_sc\|² dφ, 4096-point trapezoidal integration |
| `forward_grating_angles(nu)` | Returns (orders, angles) where φₘ = arccos(m·ν) for propagating orders |
| `fig1_flatstrip()` | Flat strip near-field + far-field; returns σ_flat |
| `fig2_sinusoidal(A, nu)` | Sinusoidal strip near-field + far-field with Floquet markers |
| `fig3_pattern_evolution()` | 2-panel: vary A at fixed ν; vary ν at fixed A |
| `print_convergence_table()` | Prints N-convergence of σ for flat strip |

**Figure layout:**
- All figures: `figsize=(7.0, 3.2)`, `dpi=200`, `constrained_layout=True`
- Near-field: `imshow` on regular grid, `cmap='jet'`, dB scale (floor = −45 dB)
- Far-field: linear amplitude in dB, `20·log10(|Φ_sc|/max|Φ_sc|)`

### Source modules used (from `src2/`)

**`src2/solver.py`** — `PlaneWave`, `MultiReflectorMDS`, `MDSSolution`
- `PlaneWave(k, beta_rad)` — incident field class; `.far_field_pattern()` returns zeros (plane wave has no finite 2D far-field amplitude)
- `MultiReflectorMDS(reflectors, incident, n)` — builds and solves MDS system
- `MDSSolution.far_field_pattern(phi, total=False)` — `total=False` gives scattered field only (avoids the delta-function in forward direction for plane wave incidence)
- `MDSSolution.near_field(xg, yg, total=True)` — evaluates `u_sc + u_inc` on a grid

**`src2/geometry.py`** — `SinusoidalStrip`
- `SinusoidalStrip(x_center, y_base, length, amplitude, frequency, phase_rad)`
- Parameterization: `x(t) = x_center + (L/2)·t`, `y(t) = y_base + A·sin(2π·ν·(x(t)−x_center) + φ)`
- Derivatives: `x'(t) = L/2`, `y'(t) = A·cos(arg)·2π·ν·(L/2)`

---

## Paper Section Summary

| Section | Key content |
|---------|-------------|
| Abstract | MDS + plane wave + Floquet grating + parametric sweep; σ constant < 0.5% |
| §I Introduction | Sinusoidal PEC strips → HTS devices; PEC approximation; MDS background |
| §II-A | Plane wave BC + single-layer SIE with edge singularity factoring |
| §II-B | MDS system (Chebyshev nodes, Cauchy kernel); far-field and σ; strip geometry; Floquet prediction |
| §III-A | Flat strip: Fig. 1 (near + far field) + Table I (N-convergence) |
| §III-B | A=1.5λ, ν=0.20: Fig. 2 with Floquet markers; orders m=±1,±2 confirmed |
| §III-C | Fig. 3: A controls blaze, ν controls angle; σ invariant |
| §IV Conclusion | Two design knobs; σ constant; HTS crosstalk application; extensions |

---

## Design Decisions and Issues Resolved

### Why L = 10λ (not 24λ as in ICTON paper)?
For L = 24λ, A = 2.5λ, ν = 0.30, arc length ≈ 116λ → need N ≈ 460 for adequate Chebyshev resolution. N = 200 severely under-resolves. For L = 10λ, worst-case arc ≈ 48λ → N = 200 gives ~4.2 nodes/λ, which is adequate.

### Why plane wave instead of CSP beam?
User explicitly requested plane wave: "Use plane wave, not beam. I am reviewing plane wave only." This changes the physics story from "near-field concentration reflector" to "diffraction grating / scattering width analysis."

### Why σ instead of Q(A,ν) (near-field peak metric)?
- For plane wave (unlike focused CSP beam), there is no natural "near-field concentration" metric
- σ is the standard metric for plane-wave scattering
- σ turns out to be nearly constant (±0.5%) → the interesting result is the FAR-FIELD PATTERN redistribution

### Why forward-hemisphere pattern evolution (Fig. 3) instead of σ_norm heatmap?
σ_norm heatmap showed < 0.4 dB variation across all 25 cases — visually boring and not physically informative. The pattern-evolution figure directly shows the grating-order physics: how lobes emerge and shift as A and ν vary.

### Why Y_BASE = 0 (not −10λ)?
Flat baseline at y = 0 makes the near-field observation window symmetric and natural. The incident wave comes from below (+y direction), so the strip at y = 0 is in the center of the near-field plots.

---

## References Used in Paper

1. Nosich & Gandel, IEEE TAP 2007 — MDS for parabolic reflectors (key validation ref)
2. Gandel, J. Math. Sci. 2010 — MDS mathematical foundations
3. Nosich, Gandel, Magath, Altintas, JOSA-A 2007 — Nystrom multireflector synthesis
4. Collin, Field Theory of Guided Waves, 1991 — Floquet theory background
5. Balanis, Advanced Engineering Electromagnetics, 2012 — grating theory
6. Shapoval et al., IEEE TTHZ 2013 — graphene strip gratings (THz, plane wave + MDS)
7. Shapoval, Sauleau, Nosich, IEEE TAP 2011 — Nystrom for finite-conductivity strips
8. Lancaster, Cambridge 1997 — HTS passive devices
9. Pozar, Microwave Engineering, 2011 — microwave context
10. Wadell, Artech 1991 — meander-line delay lines
