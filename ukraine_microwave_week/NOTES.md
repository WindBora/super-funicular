# Publication computation and figure pipeline

`generate_figures.py` is the reproducible source for the numerical results and
all eleven publication figures in this directory. It solves the scattering
problems, writes the underlying one-dimensional plot data, emits LaTeX result
macros, applies hard numerical acceptance checks, and can compile the paper.

## One-command regeneration and build

From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -c "import sys; assert sys.version_info[:3] == (3, 13, 1), sys.version"
.\.venv\Scripts\python.exe -m pip install -r .\requirements-publication.txt
.\.venv\Scripts\python.exe .\ukraine_microwave_week\generate_figures.py --build
```

The required interpreter patch version is recorded in the repository-root
`.python-version` file; exact package versions are recorded in
`requirements-publication.txt`.

This regenerates every figure and result file, validates them, and then runs
two `pdflatex` passes on `main.tex`. The build step starts only if all numerical
and artifact checks pass. `pdflatex` must be on `PATH`.

To regenerate and validate the numerical products without compiling the paper:

```powershell
.\.venv\Scripts\python.exe .\ukraine_microwave_week\generate_figures.py
```

A full run currently takes about twelve minutes on the development machine. The
script prints timestamped progress during the 41-frequency backend sweep.

## Fixed physical configuration

Lengths are nondimensionalized by the strip half-length `L=1`. The symmetric
five-period strip is

```text
x(t) = L t
y(t) = h cos(pi P t)
-1 <= t <= 1,  P = 5.
```

It is represented without a separate geometry implementation as:

```python
SinusoidalStrip(
    x_center=0,
    y_base=0,
    length=2*L,
    amplitude=h,
    frequency=P/(2*L),
    phase_rad=pi/2,
)
```

The numerical experiments use a unit-amplitude plane wave propagating along
`+y`, so `beta=pi/2`. Figure 1 also shows the general arbitrary-incidence
definition of `beta`. Observation angle `phi` is counterclockwise from `+x`.

The representative corrugated case is `h/L=0.10`, `kL=20`, and `P=5`. The
normal-incidence first grating order becomes propagating at

```text
kL = pi P = 5 pi = 15.707963...
```

because the period is `Lambda=2L/P` and the first-order condition is
`2*pi/Lambda <= k`.

## Solvers, orders, and angular sampling

- Publication solutions: `DifferentiatedNystromSolver`, `N=512`.
- Reference solutions: the same formulation, `N_ref=800`.
- All full-circle far fields and TSCS integrals: 4096 uniformly spaced,
  endpoint-excluded observation angles.
- Convergence orders: `32, 48, 64, 96, 128, 192, 256, 384, 512`.
- Convergence reference comparison: 4096 midpoint samples in Chebyshev angle
  `theta`.
- Flat-strip verification sweep: 41 uniformly spaced `kL` values from `0.25`
  through `20`, inclusive.
- Independent flat-strip backends: differentiated Nystrom `N=256`, genuine
  Chebyshev-Galerkin analytical regularization (MAR) with 256 modal
  coefficients, 2048 projection nodes, an independent 4096-node source/field
  quadrature, and pulse MoM with 192 panels.
- Order-doubling checks at `kL=20`: Nystrom `256 -> 512`; MAR modes
  `256 -> 512` at fixed projection order `Q=2048`; MAR projection order
  `2048 -> 4096` at fixed 256 modes; and pulse MoM `192 -> 384`. Both the
  relative TSCS change and full complex-pattern relative L2 change are checked
  separately for each MAR refinement.
- MAR analytically inverts the logarithmic static operator before truncation,
  following A. I. Nosich, IEEE Antennas and Propagation Magazine 41(3), 1999,
  DOI `10.1109/74.775246`, and Radio Science 51(8), 2016, DOI
  `10.1002/2016RS006044`. It is separate from the differentiated Nystrom
  solver and supports the full smooth sinusoidal geometry. Its compact-kernel
  cancellation series is used for `kappa*rho < 0.5` with at most 24 terms.
  Publication solutions are substituted into the original IE at 513
  independent target points.
- MoM uses constant pulses on equal `theta=acos(t)` panels and midpoint
  testing, adapted from the block-pulse method in S. Hatamzadeh-Varmazyar
  et al., PIER 81, 2008, DOI `10.2528/PIER07122502`. Regular panels use
  12-point Gauss-Legendre quadrature; split self panels use a quadratic map
  and 20-point Gauss-Legendre quadrature.

The smooth edge density is interpolated from its first-kind Gauss-Chebyshev
nodes. With `t=cos(theta)`, the weighted density norm becomes an ordinary
theta norm:

```text
error_N = sqrt(
    sum_j |v_N(cos(theta_j)) - v_800(cos(theta_j))|^2
    / sum_j |v_800(cos(theta_j))|^2
)
```

where `theta_j=(j+1/2)pi/4096`.

The physical two-dimensional cross sections are computed only through the
shared solver helpers:

```text
d sigma / d phi = |Phi_sc(phi)|^2 / k
sigma = integral_0^(2 pi) (d sigma / d phi) d phi.
```

All polar figures plot the absolute, non-peak-normalized quantity
`10 log10[|Phi_sc(phi)|^2/(kL)]` on the common range `-30` to `+15 dB`. Its angular
integral in linear units gives `sigma/L`. The near-field figure plots the
relative intensity `|U_tot|^2/|U_inc|^2`; unit incidence makes the denominator
one. The flat-strip geometrical-optics reference is `sigma=4L`.

## Generated products

- `fig1_geometry.pdf`: geometry, arbitrary-incidence `beta`, and observation
  angle convention; strip extent is identified by the `-L` and `+L` endpoints
  without a separate `2L` dimension line.
- `fig2_convergence.pdf`: weighted density convergence of the differentiated
  Nystrom solution.
- `fig3_flat_validation.pdf`: flat-strip `sigma/(4L)` across all three
  numerical backends plus GO.
- `fig4_near_field.pdf`: unit-incident relative near-field intensity.
- `fig5_representative_polar.pdf`: representative absolute differential-TSCS
  polar pattern.
- `fig6_height_flat.pdf`: polar pattern at `h/L=0` and `kL=20`.
- `fig7_height_005.pdf`: polar pattern at `h/L=0.05` and `kL=20`.
- `fig8_height_010.pdf`: polar pattern at `h/L=0.10` and `kL=20`.
- `fig9_frequency_12.pdf`: polar pattern at `h/L=0.10` and `kL=12`.
- `fig10_frequency_16.pdf`: polar pattern at `h/L=0.10` and `kL=16`.
- `fig11_frequency_20.pdf`: polar pattern at `h/L=0.10` and `kL=20`.
- `fig2_verification.pdf`: current-manuscript compatibility composite of the
  convergence and flat-strip validation plots. Panel identifiers are placed
  above the axes so they do not cover data; the validation ordinate displays
  `0.98` through `1.10` to expand the near-unity comparison. The coincident
  `kL=0.25` values above the visible range remain in the complete plotted path;
  only their off-scale portion is clipped by the axes, so the connecting line
  remains visible. The horizontal axis begins at `kL=0`, and the composite
  height remains 2.72 in.
- `fig3_field_pattern.pdf`: current-manuscript compatibility composite of the
  representative near-field and polar pattern. Its colorbar is locked to the
  heatmap height, and its polar axes match the physical size used in the
  current three-panel polar figures; panel identifiers are above the axes.
- `revision_results.csv`: scalar metrics, convergence data, the 41-point
  backend sweep, order-doubling checks, every N=512/N=800 reference check,
  every MAR/N=800 publication-case check and original-IE residual, and all
  plotted polar samples.
- `revision_results.tex`: deterministic LaTeX macros for manuscript claims.
- `publication_manifest.json`: exact software versions, numerical settings,
  method-source DOIs, and SHA-256 hashes of source and generated artifacts.

The CSV is in tidy long form. Important columns are `dataset`, `series`,
`x_name`, `x_value`, `y_name`, `y_value`, `n`, `h_over_L`, and `kL`. Polar CSV
values are the unclipped absolute dB data; the displayed radial range is a
plotting choice.

## Hard validation thresholds

Generation returns a nonzero exit code if any of these checks fails:

- the N=512 weighted density error exceeds `1e-4`;
- the requested convergence sequence is not strictly decreasing;
- the flat-strip Nystrom `sigma/(4L)` at `kL=20` differs from GO by more than
  `0.005`;
- the maximum three-backend absolute spread in `sigma/(4L)` over all 41
  frequencies is not below `1e-3`;
- any backend's order-doubling relative TSCS change at `kL=20` is not below
  `1e-3`;
- the fixed-`Q` MAR mode-doubling change is not below `1e-6` in relative TSCS
  or `1e-5` in full complex-pattern relative L2 norm;
- the fixed-mode MAR projection-doubling change is not below `1e-6` in either
  relative TSCS or full complex-pattern relative L2 norm, or the
  projection-doubled flat-strip original-IE residual is not below `1e-7`;
- any publication case has an N=512/N=800 relative TSCS change or full complex
  pattern relative L2 change not below `1e-3`;
- any 256-mode, `Q=2048` MAR publication-case result differs from the N=800
  Nystrom reference by `1e-3` or more in relative TSCS or full complex-pattern
  L2 norm, or has an independently evaluated original-IE maximum absolute
  residual of `1e-5` or more;
- a sweep value is non-finite/non-positive, or a required artifact is missing.

The MAR original-IE residual is an internal implementation-consistency check,
not the primary evidence of accuracy. Accuracy is established by solution
convergence, independent-backend agreement, and order/reference comparisons.

The four legacy composite figure PDFs retained beside the manuscript are
archival companions to the untouched `main copy*.tex` sources. They are not
publication outputs and are excluded from `publication_manifest.json`.

## Current accepted metrics

The current deterministic run produced:

| Metric | Value |
|---|---:|
| N=512 weighted density error vs N=800 | `2.811422277e-05` |
| Flat Nystrom `sigma/(4L)` at `kL=20` | `0.9999611930` |
| Maximum three-backend spread | `1.592975342e-05` |
| Maximum flat-backend order-doubling change | `5.038187853e-06` |
| Maximum publication TSCS change, N=512 vs N=800 | `7.533153066e-06` |
| Maximum publication full-pattern L2 change, N=512 vs N=800 | `2.306639575e-05` |
| Maximum publication TSCS change, MAR vs N=800 Nystrom | `2.427893786e-06` |
| Maximum publication full-pattern L2 change, MAR vs N=800 Nystrom | `7.643997800e-06` |
| Maximum MAR original-IE absolute residual | `9.658357547e-07` |
| MAR 256-to-512-mode relative TSCS change at fixed Q | `3.330798483e-16` |
| MAR 256-to-512-mode full-pattern L2 change at fixed Q | `6.254606797e-16` |
| MAR Q=2048-to-4096 relative TSCS change at fixed 256 modes | `2.963788901e-11` |
| MAR Q=2048-to-4096 full-pattern L2 change at fixed 256 modes | `8.265571720e-08` |
| Projection-doubled flat-strip MAR original-IE residual | `3.042777402e-08` |

The TSCS ratios used in the amplitude and frequency discussion are read from
`revision_results.tex`, not typed into `main.tex`. That generated file defines:

```text
ConvErrorNFiveTwelve
FlatTSCSRatioTwenty
BackendMaxDiff
AmpTSCSFlat
AmpTSCSFive
AmpTSCSTen
FreqTSCSTwelve
FreqTSCSSixteen
FreqTSCSTwenty
FirstOrderCutoff
FlatOrderDoublingMaxChange
ProductionMaxTSCSChange
ProductionMaxPatternChange
MARCorrugatedMaxTSCSChange
MARCorrugatedMaxPatternChange
MARCorrugatedMaxResidual
MARModeDoublingTSCSChange
MARModeDoublingPatternChange
MARProjectionDoublingTSCSChange
MARProjectionDoublingPatternChange
MARProjectionDoubledResidual
```
