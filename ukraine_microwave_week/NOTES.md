# Ukrainian Microwave Week Paper — Work Notes

## What Was Built

A complete 5-page IEEEtran conference paper and all supporting Python code for the **Ukrainian Microwave Week** conference. The paper is a full-wave numerical study of plane-wave E-wave scattering by sinusoidal open PEC strips, analyzed via the Method of Discrete Singularities (MDS).

---

## Paper Overview

**Title:** E-Wave Plane-Wave Scattering by Sinusoidal Open PEC Strips via the Method of Discrete Singularities

**Files:**
- `main.tex` — 5-page IEEEtran conference paper (compiles to `main.pdf`)
- `generate_figures.py` — Python script that produces all split figure panels
- `fig1_flatstrip_near.png`, `fig2_flatstrip_far.png`,
  `fig3_sinusoidal_near.png`, `fig4_sinusoidal_far.png`,
  `fig5_amplitude_sweep.png`, `fig6_frequency_sweep.png` — generated figures

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
- Euler time factor: `exp(−iωt)` (suppressed in frequency-domain equations)
- Incidence: `β = π/2` (normal incidence, wave propagates in `+y` direction)
- Strip: horizontal extent `L = 10λ`, centered at origin, baseline `y = 0`
- Sinusoidal geometry: `x = (L/2)·t`, `y = A·sin(2π·ν·x)`, `t ∈ [-1, 1]`
- Parameters swept: amplitude `A ∈ {0.5, 1.0, 1.5, 2.0, 2.5}λ`, spatial frequency `ν ∈ {0.10, 0.15, 0.20, 0.25, 0.30}λ⁻¹`

### Governing Equations
Scattered field `U` satisfies the 2D Helmholtz equation `(Δ + k²)U = 0` outside the arc, with Dirichlet BC `U|_Γ = −U₀|_Γ`.

**Physical field:** `u_tot(r,t) = Re{U_tot(r) exp(−iωt)}`, where
`U_tot = U₀ + U`. The compatible outgoing condition is
`√r(∂U/∂r − ikU) → 0`, and `H₀⁽¹⁾` is therefore the outgoing kernel.

**Plane wave:** `U₀(r) = exp(+ik(x·cos β + y·sin β))`. With the
`exp(−iωt)` time factor, constant-phase fronts move in the `+β` direction.

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
where `tᵢ = cos((2i−1)π/(2n))` are first-kind Chebyshev roots, `K` is the smooth part of the differentiated Hankel kernel, and
`f(τ) = −dU₀/dτ = −ik U₀(r(τ)) d̂·r′(τ)`.

Solved by Gaussian elimination. Boundary residuals remain near `10⁻¹³` for
the Paper-MDS systems used in the figures.

### Why MDS is Good Here
- Analytically extracts edge singularity → spectral convergence for smooth problems
- Dense but small system (N = 200 unknowns per reflector)
- No artificial boundary truncation or absorbing layers needed
- Validated in literature for parabolic reflector antennas (Nosich & Gandel 2007)

---

## Key Physics: Floquet Reference Orders

Angles are measured counterclockwise from `+x`. For normal incidence
(`β = π/2`) on the infinite periodic continuation of the sinusoidal strip
with period `Λ = 1/ν`, Floquet theory gives two branches:

```
φₘᶠ = arccos(m·ν)             forward / +y branch
φₘᵇ = 360° − arccos(m·ν)      backward / reflected −y branch
|m·ν| < 1
```

| ν (λ⁻¹) | m=+1 forward | m=−1 forward | m=+1 reflected | m=−1 reflected |
|----------|--------------|--------------|----------------|----------------|
| 0.10     | 84.3°        | 95.7°        | 275.7°         | 264.3°         |
| 0.15     | 81.4°        | 98.6°        | 278.6°         | 261.4°         |
| 0.20     | 78.5°        | 101.5°       | 281.5°         | 258.5°         |
| 0.25     | 75.5°        | 104.5°       | 284.5°         | 255.5°         |
| 0.30     | 72.5°        | 107.5°       | 287.5°         | 252.5°         |

For `ν = 0.20`, the reflected `m=+2` and `m=−2` references are
`293.6°` and `246.4°`. The infinite PEC continuation supports the
reflected branch; the finite strip also produces forward shadow-side
endpoint diffraction.

For the finite strip, `L·ν` periods = 2 periods at `ν = 0.20`. This broadens each Floquet lobe, so maxima can shift slightly from the periodic-reference angles.

---

## Numerical Results

### Flat Strip Benchmark (A = 0, L = 10λ, N = 200)

- Boundary residual: approximately `10⁻¹³`
- Total scattering width: `σ = 40.0λ`

**N-convergence of σ:**

| N   | σ (λ)  | Rel. change | Residual       |
|-----|--------|-------------|----------------|
| 32  | 39.857 | —           | `O(10⁻¹⁴)` |
| 64  | 39.999 | 3.56 × 10⁻³ | `O(10⁻¹⁴)` |
| 100 | 40.000 | 2.78 × 10⁻⁵ | `O(10⁻¹⁴)` |
| 150 | 40.000 | 1.18 × 10⁻⁶ | `O(10⁻¹³)` |
| 200 | 40.000 | 2.23 × 10⁻⁸ | `O(10⁻¹³)` |

The residual measures algebraic satisfaction of the discrete system; the
change in `σ` measures convergence of the reported physical observable.

### Sinusoidal Strip (A = 1.5λ, ν = 0.20λ⁻¹)

- Boundary residual: approximately `1.5 × 10⁻¹³`
- Total scattering width: `σ = 40.011λ`
- The forward finite-aperture lobe is at `90°`; it is not a reflected order.
- Strong reflected first-order peaks occur at approximately `258.2°` and
  `281.7°`, matching the `m=−1` and `m=+1` references `258.5°` and `281.5°`.
- Weaker reflected second-order structure is organized by the `m=−2` and
  `m=+2` references `246.4°` and `293.6°`.

### Parametric Sweep (25 cases, all N = 200)

All boundary residuals remain of order `10⁻¹³`.

**Key finding:** Total scattering width `σ` spans `38.940--40.656λ` across
all 25 `(A, ν)` combinations. The sinusoidal corrugation changes both total
scattering width and the angular distribution of scattered power.

- Amplitude `A` → controls the relative strength of off-normal lobes
- Spatial frequency `ν` → controls the Floquet-order reference angles

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
| `make_solver(A, nu, n)` | Creates `SinusoidalStrip` + `PlaneWave` + `MultiReflectorPaperMDS`, returns unsolved solver |
| `scattering_width(sol)` | σ = (1/π)∫\|Φ_sc\|² dφ, 4096-point trapezoidal integration |
| `floquet_reference_angles(nu)` | Returns both forward and reflected periodic-reference branches |
| `fig1_flatstrip()` | Flat strip near-field + far-field; returns σ_flat |
| `fig2_sinusoidal(A, nu)` | Sinusoidal strip near-field + far-field with Floquet markers |
| `fig3_pattern_evolution()` | Generates separate reflected-pattern figures for the A and ν sweeps |
| `print_convergence_table()` | Prints N-convergence of σ for flat strip |

**Figure layout:**
- Single-column figures use approximately 3.45-inch widths, `dpi=200`, and `constrained_layout=True`
- Near-field: `imshow` on regular grid, `cmap='jet'`, dB scale (floor = −45 dB)
- Far-field: linear amplitude in dB, `20·log10(|Φ_sc|/max|Φ_sc|)`

### Source modules used (from `src2/`)

**`src2/solver.py`** — `PlaneWave`, `MultiReflectorPaperMDS`, `MDSSolution`
- `PlaneWave(k, beta_rad)` — incident field class; `.far_field_pattern()` returns zeros (plane wave has no finite 2D far-field amplitude)
- `MultiReflectorPaperMDS(reflectors, incident, n)` — builds and solves the paper-faithful MDS system
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
| Abstract | MDS + plane wave + Floquet reference orders + parametric sweep; σ range 38.940--40.656λ |
| §I Introduction | Sinusoidal PEC strips → HTS devices; PEC approximation; MDS background |
| §II-A | Plane wave BC + single-layer SIE with edge singularity factoring |
| §II-B | MDS system (Chebyshev nodes, Cauchy kernel); far-field and σ; strip geometry; forward/reflected Floquet branches |
| §III-A | Flat strip: Fig. 1 (near + far field) + Table I (N-convergence) |
| §III-B | A=1.5λ, ν=0.20: Fig. 2 with directional markers; reflected lobes near backward m=±1,±2 references |
| §III-C | Fig. 3: A controls off-normal lobe strength, ν controls reference angle; σ is not invariant |
| §IV Conclusion | Two design knobs; σ varies across the sweep; HTS crosstalk application; extensions |

---

## Design Decisions and Issues Resolved

### Why L = 10λ (not 24λ as in ICTON paper)?
For L = 24λ, A = 2.5λ, ν = 0.30, arc length ≈ 116λ → need N ≈ 460 for adequate Chebyshev resolution. N = 200 severely under-resolves. For L = 10λ, worst-case arc ≈ 32.31λ → N = 200 gives ~6.2 nodes/λ, which is adequate.

### Why plane wave instead of CSP beam?
User explicitly requested plane wave: "Use plane wave, not beam. I am reviewing plane wave only." This changes the physics story from "near-field concentration reflector" to finite-strip plane-wave scattering and Floquet-reference angle analysis.

### Why σ instead of Q(A,ν) (near-field peak metric)?
- For plane wave (unlike focused CSP beam), there is no natural "near-field concentration" metric
- σ is the standard metric for plane-wave scattering
- σ is not invariant in the current verified sweep; the interesting result is both FAR-FIELD PATTERN redistribution and total-width variation

### Why reflected-hemisphere pattern evolution (Fig. 3) instead of σ_norm heatmap?
The reflected-hemisphere figure directly shows how the backward Floquet-reference lobes emerge and shift as `A` and `ν` vary. The forward `90°` lobe is retained in the full-pattern figure for directional context.

### Why Y_BASE = 0 (not −10λ)?
Flat baseline at y = 0 makes the near-field observation window symmetric and natural. The incident wave comes from below (+y direction), so the strip at y = 0 is in the center of the near-field plots.

---

## References Used in Paper

1. Nosich & Gandel, IEEE TAP 2007 — MDS for parabolic reflectors (key validation ref)
2. Gandel, J. Math. Sci. 2010 — MDS mathematical foundations
3. Nosich, Gandel, Magath, Altintas, JOSA-A 2007 — Nystrom multireflector synthesis
4. Collin, Field Theory of Guided Waves, 1991 — Floquet theory background
5. Balanis, Advanced Engineering Electromagnetics, 2012 — periodic/Floquet background
6. Lancaster, Cambridge 1997 — HTS passive devices
7. Pozar, Microwave Engineering, 2011 — microwave context
8. Wadell, Artech 1991 — meander-line delay lines
