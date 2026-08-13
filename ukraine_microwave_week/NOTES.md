# Publication computation and figure pipeline

`generate_figures.py` is the reproducible source for the numerical results and
all five publication figures in this directory. It solves the scattering
problems, writes the underlying one-dimensional plot data, emits LaTeX result
macros, applies hard numerical acceptance checks, and can compile the paper.

## One-command regeneration and build

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\ukraine_microwave_week\generate_figures.py --build
```

This regenerates every figure and result file, validates them, and then runs
two `pdflatex` passes on `main.tex`. The build step starts only if all numerical
and artifact checks pass. `pdflatex` must be on `PATH`.

To regenerate and validate the numerical products without compiling the paper:

```powershell
.\.venv\Scripts\python.exe .\ukraine_microwave_week\generate_figures.py
```

A full run currently takes about four minutes on the development machine. The
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
- Independent flat-strip backends: differentiated Nystrom `N=256`, analytical
  regularization (MAR) `N=256`, and pulse MoM `N=192`.
- Order-doubling checks at `kL=20`: Nystrom `256 -> 512`, MAR `256 -> 512`,
  and pulse MoM `192 -> 384`.

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

All polar panels plot the absolute, non-peak-normalized quantity
`10 log10[|Phi_sc(phi)|^2/(kL)]` on the common range `-30` to `+15 dB`. Its angular
integral in linear units gives `sigma/L`. The near-field panel plots the
relative intensity `|U_tot|^2/|U_inc|^2`; unit incidence makes the denominator
one. The flat-strip geometrical-optics reference is `sigma=4L`.

## Generated products

- `fig1_geometry.pdf`: geometry, arbitrary-incidence `beta`, and observation
  angle convention.
- `fig2_verification.pdf`: weighted density convergence and flat-strip
  `sigma/(4L)` across all three numerical backends plus GO.
- `fig3_field_pattern.pdf`: unit-incident relative near-field intensity and the
  representative absolute differential-TSCS polar pattern.
- `fig4_amplitude_polar.pdf`: three polar panels at `h/L=0, 0.05, 0.10`, one
  curve per panel, with `kL=20`.
- `fig5_frequency_polar.pdf`: three polar panels at `kL=12, 16, 20`, one curve
  per panel, with `h/L=0.10`.
- `revision_results.csv`: scalar metrics, convergence data, the 41-point
  backend sweep, order-doubling checks, every N=512/N=800 reference check, and
  all plotted polar samples.
- `revision_results.tex`: deterministic LaTeX macros for manuscript claims.

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
- any publication case has an N=512/N=800 relative TSCS change or full complex
  pattern relative L2 change not below `1e-3`;
- a sweep value is non-finite/non-positive, or a required artifact is missing.

Algebraic residuals are not used as accuracy evidence. Accuracy is established
by solution convergence, independent-backend agreement, and order/reference
comparisons.

## Current accepted metrics

The current deterministic run produced:

| Metric | Value |
|---|---:|
| N=512 weighted density error vs N=800 | `2.811422277e-05` |
| Flat Nystrom `sigma/(4L)` at `kL=20` | `0.9999611930` |
| Maximum three-backend spread | `1.589017047e-05` |
| Maximum flat-backend order-doubling change | `5.038187853e-06` |
| Maximum publication TSCS change, N=512 vs N=800 | `7.533153066e-06` |
| Maximum publication full-pattern L2 change, N=512 vs N=800 | `2.306639575e-05` |

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
```
