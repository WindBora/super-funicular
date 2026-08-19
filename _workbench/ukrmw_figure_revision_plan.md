# Eleven-Figure Revision with Reproducible MAR and MoM Validation

## Current-Manuscript Figure 3 Follow-up

The current `main.tex` revision uses the two-panel compatibility Figure 3.
For that artifact, the colorbar is anchored to the exact heatmap height and the
right polar axes use the same 1.73-in physical diameter as the corresponding
polar panels in current Figures 4 and 5. The dense heatmap alone is embedded at
600 dpi to avoid false vector-cell seams; axes, labels, curves, colorbar, and
the polar diagram remain vector. In current compatibility Figures 2 and 3,
panel identifiers `(a)` and `(b)` sit just above their axes rather than covering
the plotted data.

Figure 1 no longer carries the separate `2L` dimension line, extension marks,
or `2L` label; the endpoint markers and `-L`/`+L` labels still identify the
physical strip ends. In current Figure 2(b), the displayed TSCS axis is limited
to `0.98 <= sigma/(4L) <= 1.10`, which excludes the isolated value above 1.1
from view and stretches the near-unity comparison without changing source data.
The three coincident off-scale backend samples at `kL=0.25` remain in the CSV
and in the plotted paths, and are disclosed in the caption. The y-limit clips
only the off-scale portions while retaining the visible connecting segments.
The composite height is 2.72 in, matching the other full-width result figures
while avoiding an orphaned bibliography page.
Its horizontal axis starts at `kL=0` so the first visible segment is separated
from the left frame; the physical figure height remains unchanged.

## Objective

Revise `ukraine_microwave_week/main.tex` so the paper itself is the source of
truth for every plotted result, numerical backend, and reproducibility choice.
Use only standard IEEEtran floats and allow the paper to grow naturally.

## Figure Structure

The final manuscript contains eleven independently numbered figures:

1. Geometry and angle convention.
2. Nystrom density convergence.
3. Flat-strip three-backend validation.
4. Representative near-field intensity.
5. Representative far-field polar pattern.
6. Height sweep, `h/L=0`.
7. Height sweep, `h/L=0.05`.
8. Height sweep, `h/L=0.10`.
9. Frequency sweep, `kL=12`.
10. Frequency sweep, `kL=16`.
11. Frequency sweep, `kL=20`.

Required figure changes:

- Figure 1 uses the pale solid `x=0` and `y=0` centerlines as the actual
  coordinate axes; the separate bottom and left frame axes are removed. The
  former overlapping height-arrow heads are replaced by a thin ticked dimension
  line, so no unexplained central dot remains.
- Figure 1 is declared as an ordinary one-column `figure[!t]` early enough for
  IEEEtran to place it at the top right of page 1. No wrapping, absolute
  placement, negative spacing, forced break, or output-routine package is used.
- Every former panel is emitted as its own vector PDF and ordinary single-column
  float. There are no `figure*` environments and no panel labels.
- Every caption, callout, table reference, reviewer-response statement,
  filename, generated-product list, and manifest entry follows the eleven-
  figure numbering.
- `main.tex` is the sole build manuscript. The `main copy*.tex` files and their
  legacy composite PDFs remain untouched archival material and are excluded
  from the publication manifest.

## Genuine MAR and Documented MoM

- Replace the former pseudo-MAR backend, which duplicated the differentiated
  Nystrom system, with an independent coefficient-space Chebyshev-Galerkin
  method of analytical regularization for the original first-kind IE.
- For `g=r/L`, `kappa=kL`, and `a=2i/pi`, split

  \[
  H_0^{(1)}(\kappa\rho)
  =a\log|t-\tau|+C_\kappa+K_c(t,\tau),\qquad
  C_\kappa=1+a[\log(\kappa/2)+\gamma_E],
  \]

  with `K_c(t,t)=a log|g'(t)|`. In the orthonormal weighted Chebyshev basis,
  use the exact static eigenvalues

  \[
  \lambda_0=C_\kappa-a\log2,\qquad \lambda_n=-a/n,
  \]

  and solve

  \[
  [I+\Lambda^{-1}R]c=\Lambda^{-1}b.
  \]

- Published MAR values use 256 modes and `Q=2048` midpoint-Chebyshev DCT
  projection nodes. A separate 4096-node midpoint-Chebyshev rule reconstructs
  the density and evaluates source, near-field, far-field, and residual
  integrals. Original-IE residuals use 513 independent target points.
- Evaluate the compact kernel with its exact geometry-dependent diagonal and a
  cancellation-safe, at-most-24-term series for `kappa*rho < 0.5`.
- Base the implementation on Nosich's 1999 and 2016 analytical-regularization
  framework. Cite Vinogradova et al. as a prior sinusoidal-grating MAR
  application, not as the exact algorithm reproduced here.
- Document the independent MoM backend as an edge-weighted block-pulse
  point-matching scheme: 192 equal panels in `t=cos(theta)`, constant smooth
  density per panel, midpoint testing, 12-point Gauss-Legendre regular-panel
  integration, and a quadratically mapped 20-point rule on each half of a self
  panel. Cite Harrington and the block-pulse scattering source.

## Reproducibility Contract

- `generate_figures.py` produces eleven vector PDFs, `revision_results.csv`,
  `revision_results.tex`, and `publication_manifest.json`, then optionally
  compiles the paper with `--build`.
- Matplotlib PDF metadata is fixed so repeated runs are byte-identical and
  figure SHA-256 hashes are deterministic.
- CSV rows record mode/panel count, projection order, independent field order,
  residual target count, small-argument settings, MoM quadrature orders,
  angular samples, and both MAR implementation-source DOIs.
- `.python-version` records Python 3.13.1. `requirements-publication.txt` pins
  NumPy 2.3.3, SciPy 1.16.2, and Matplotlib 3.10.7. The README creates and
  verifies `.venv` before installation or regeneration.
- The manifest records the full imported implementation source set, numerical
  configuration, environment, method-source DOIs, and hashes of all current
  source/data/figure artifacts.
- The source archive accompanying the manuscript is the revision artifact; the
  project URL is identified separately without claiming that uncommitted local
  changes already exist on the remote.

## Numerical Verification

- Check MAR static eigenvalues, compact-kernel diagonal, small-argument
  evaluation, kernel symmetry, broadside parity, oblique incidence, analytic
  flat-strip Bessel reconstruction, multiple reflectors, and the independent
  original-IE residual.
- At flat-strip `kL=20`, vary the controls separately:
  - 256 to 512 modes at fixed `Q=2048`: relative TSCS below `1e-6` and complex-
    pattern relative L2 below `1e-5`.
  - `Q=2048` to `Q=4096` at fixed 256 modes: relative TSCS and complex-pattern
    relative L2 below `1e-6`; projection-doubled absolute IE residual below
    `1e-7`.
- Compare every unique corrugated publication case against Nystrom `N=800` in
  relative TSCS and full complex-pattern L2 norm. Require both below `1e-3` and
  each production MAR absolute original-IE residual below `1e-5`.
- Verify MoM 192-to-384 convergence and the exact documented panel/quadrature
  construction.
- Verify generated-data/macro/manifest synchronization, every manifest hash,
  deterministic figure metadata, all-vector Figure 4 output, bibliography
  ordering, and the eleven-float manuscript contract.

## Final Acceptance

- Run the full numerical generator and its hard validation checks.
- Run the complete unit-test suite.
- Compile the manuscript to `output/pdf/UkrMW_2026_revised.pdf`.
- Render and inspect every page. Confirm Figure 1 is top right on page 1, all
  eleven figures are independent and legible, references resolve, fonts are
  embedded, no content is clipped or overlapping, and no figure enters the
  bibliography.

## Assumptions

- “Full corrugated MAR” means a transparent genuine MAR derived for this
  manuscript's smooth sinusoidal-strip IE under Nosich's framework, not an
  undocumented reproduction of Vinogradova et al.'s implementation.
- Natural page growth is acceptable; no manual float-placement trick or page
  enlargement is used.
