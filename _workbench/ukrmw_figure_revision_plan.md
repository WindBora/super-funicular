# Seven-Figure Revision with Genuine Corrugated MAR

## Summary

Revise the manuscript to contain seven numbered figures:

1. Geometry
2. Nyström convergence
3. Flat-strip three-backend validation
4. Representative near field
5. Representative polar pattern
6. Existing three-panel height comparison
7. Existing three-panel frequency comparison

Allow natural page growth. Use only standard IEEEtran floats and remove the existing manual page enlargement.

## Figure and Manuscript Changes

- Redraw Fig. 1 with pale solid \(x=0\) and \(y=0\) reference lines behind the strip; identify \(x=0\) as the symmetry axis and \(y=0\) as the mean line.
- Remove the overlapping double-arrowheads that currently resemble an unexplained dot. Show \(h\) with a thin offset dimension line and end ticks, and define it in the caption.
- Add the Fig. 1 callout to the opening Introduction paragraph and declare a normal one-column `figure[!t]` immediately after that paragraph. Verify that IEEEtran places it at the top-right of page 1; use no wrapping, absolute positioning, negative spacing, forced breaks, or output-routine packages.
- Split current Fig. 2 into two one-column figures: convergence and flat-strip validation. Remove `(a)/(b)` labels and give each an independent caption and reference.
- Split current Fig. 3 into two one-column figures: near-field intensity and representative polar pattern.
- Preserve the current Figs. 4 and 5 as unchanged three-panel full-width comparisons, renumbered as Figs. 6 and 7.
- Update every callout, caption, table reference, reviewer-response reference, filename, and generated-artifact list. Remove `\enlargethispage` and obsolete float-scheduling comments.

## Genuine MAR and Documented MoM

- Replace the current pseudo-MAR implementation in `src2/solver.py`; it duplicates the differentiated Nyström system and must not remain in results or archived CSV data.
- Reimplement `MultiReflectorMAR` as an independent coefficient-space Chebyshev–Galerkin analytical regularization of the original first-kind IE. It must not subclass the MDS/Nyström solver.
- For \(\boldsymbol g=\boldsymbol r/L\), \(\kappa=kL\), and \(a=2i/\pi\), use
  \[
  H_0^{(1)}(\kappa\rho)
  =a\log|t-\tau|+C_\kappa+K_c(t,\tau),\qquad
  C_\kappa=1+a[\log(\kappa/2)+\gamma_E],
  \]
  with \(K_c(t,t)=a\log|\boldsymbol g'(t)|\). In the orthonormal weighted Chebyshev basis, analytically invert
  \[
  \lambda_0=C_\kappa-a\log2,\qquad \lambda_n=-a/n,
  \]
  and solve
  \[
  [I+\Lambda^{-1}R]c=\Lambda^{-1}b.
  \]
- Assemble \(R\) using midpoint Chebyshev angles and an orthonormal two-dimensional DCT. Use 256 modal coefficients and \(Q=2048\) projection points for published values; verify with 512 coefficients and \(Q=4096\).
- Evaluate the compact kernel using its exact geometry-dependent diagonal and a cancellation-safe small-argument series below \(z=0.5\). Reconstruct current, near field, and far field from modal coefficients using an independent 4096-point weighted rule.
- Introduce `MARSolution`, exposing modal coefficients, density/current evaluation, near field, far field, and an independently evaluated original-IE residual. Keep existing CLI MAR modes, but route them to the genuine backend and correct scripts that currently import MAR under a MoM name.
- Base the MAR description on [Nosich’s analytical-regularization framework](https://www.ire.kharkov.ua/wp-content/uploads/2018/12/apmag1999-mareg.pdf). Cite Vinogradova et al. as prior sinusoidal-grating MAR work, not as the exact algorithm reproduced here.
- Describe the existing MoM as an edge-weighted adaptation of the [block-pulse point-matching method](https://www.jpier.org/issues/volume.html?paper=07122502): 192 equal panels in \(t=\cos\theta\), constant \(v\) per panel, midpoint testing, 12-point Gauss–Legendre regular-panel integration, and split quadratic-map 20-point self-panel integration. Also cite Harrington’s foundational MoM paper.
- Add the complete MAR split, eigenvalues, matrix equation, quadrature orders, MoM panel rules, time convention, and stopping criteria to the manuscript. Remove `[[xxx]]` and `[[yyy]]`.
- Retain Fig. 3’s flat-strip comparison using Nyström \(N=256\), genuine MAR with 256 modes/\(Q=2048\), and MoM with 192 panels. Additionally report generated maximum MAR–Nyström TSCS and complex-pattern differences over every unique corrugated publication case.

## Reproducibility and Interfaces

- Refactor `generate_figures.py` to use named paths and generate seven vector PDFs, separate plotting functions for Figs. 2–5, and unchanged three-panel generators for Figs. 6–7.
- Regenerate all CSV values rather than relabeling old pseudo-MAR data. Record solver order, MAR projection order, MoM quadrature orders, angular samples, source DOI, residuals, and order-doubling metrics.
- Add exact publication dependency pins for Python 3.13.1, NumPy 2.3.3, SciPy 1.16.2, and Matplotlib 3.10.7, plus a deterministic manifest containing parameters and SHA-256 hashes of generated artifacts.
- Add a code/data-availability paragraph pointing to the public repository, the generator command, generated CSV/macros, and dependency file. Update the README, publication notes, tests, and reviewer response consistently.
- Treat `ukraine_microwave_week/main.tex` as the sole manuscript source; leave the archival `main copy*.tex` files untouched.

## Verification

- Test the MAR static eigenvalues, compact-kernel diagonal for flat and sinusoidal strips, matrix symmetry, small-argument evaluation, broadside parity, oblique incidence, analytic flat-strip Bessel far field, and original-IE residual.
- Require MAR mode doubling to change TSCS by less than \(10^{-6}\) and the complex pattern by less than \(10^{-5}\); require \(Q\)-doubling changes below \(10^{-7}\) and \(10^{-6}\), respectively, and independent relative boundary residual below \(10^{-7}\).
- Cross-check flat cases at \(kL=0.25,1,5,20\), corrugated cases at \(P=5\), \(h/L=0.10\), \(kL=12,16,20\), and one oblique-incidence case. Keep cross-backend published discrepancies below \(10^{-3}\).
- Verify MoM \(192\rightarrow384\) convergence and the exact documented panel/quadrature construction.
- Update generated-data and bibliography tests, including the currently stale bibliography-order expectation.
- Run the full numerical generator, unit tests, and two LaTeX passes. Render every final PDF page and visually confirm Fig. 1 is top-right on page 1, Figs. 2–5 are independent one-column figures, Figs. 6–7 retain their comparison layouts, all references resolve, and no figures enter the bibliography.

## Assumptions

- “Full corrugated MAR” means a genuine MAR derived transparently for the manuscript’s smooth sinusoidal-strip IE under Nosich’s framework, not an undocumented reproduction of Vinogradova et al.’s particular implementation.
- Additional pages are acceptable; no manual layout compression will be used.
