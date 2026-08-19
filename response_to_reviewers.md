# Response to Reviewers

We thank both reviewers for their careful reading and constructive recommendations. The manuscript has been substantially rewritten. It now uses Nyström terminology throughout, a symmetric five-period geometry scaled by its half-span $L$, an arbitrary-incidence formulation, absolute polar scattering plots, density-based convergence, and a three-backend numerical comparison. The revision contains eleven standalone, single-column figures: Fig. 1 is the geometry, Figs. 2--5 show convergence, validation, and representative fields, Figs. 6--8 show the height sweep, and Figs. 9--11 show the electrical-size sweep.

We also audited every field normalization. That audit identified an additional issue not raised explicitly in the reports: the former TSCS expression lacked its dimensional factor $1/k$. The revised definition is

$$
\sigma=\frac{1}{k}\int_0^{2\pi}|\Phi_{\rm sc}(\varphi)|^2\,d\varphi.
$$

The resulting TSCS has units of length and can be compared meaningfully with the geometrical-optics value $4L$.

For reproducibility, the validation backends are now specified in both the manuscript and the source archive accompanying it (`src2/solver.py`). The analytical-regularization backend is a genuine coefficient-space method of analytical regularization (MAR) based on static-part inversion as presented by Nosich (1999, DOI 10.1109/74.775246) and Nosich (2016, DOI 10.1002/2016RS006044); it is not a reproduction of the sinusoidal-grating implementation of Vinogradova *et al.* (2019). The published MAR calculation uses 256 Chebyshev modes and a $Q=2048$ midpoint-Chebyshev DCT projection, an independent 4096-node midpoint-Chebyshev source/field quadrature, 513 independent original-IE target points, and a cancellation-safe small-argument series with threshold $\kappa\rho=0.5$ and at most 24 terms. At $kL=20$, mode convergence is checked by doubling 256 to 512 modes at fixed $Q=2048$, while projection convergence is checked by doubling $Q=2048$ to 4096 at fixed 256 modes; both relative TSCS and full complex-pattern changes are tested separately. The acceptance limits are $10^{-6}$ in TSCS and $10^{-5}$ in pattern for mode doubling, and $10^{-6}$ in both quantities for projection doubling. The method-of-moments (MoM) backend is an edge-weighted block-pulse point-matching implementation adapted from Harrington (1967) and Hatamzadeh *et al.* (2008): it uses 192 panels, 12-point Gauss--Legendre integration on regular panels, and a split, quadratically mapped 20-point rule on each half of a self panel; the flat-strip order-doubling check uses 384 panels. The reproducible driver `ukraine_microwave_week/generate_figures.py` writes the numerical CSV, manuscript macros, eleven figure PDFs, and `publication_manifest.json`; the manifest records configurations, source DOIs, software versions, and SHA-256 hashes. Python 3.13.1 is recorded in `.python-version`, and exact package versions are pinned in `requirements-publication.txt`; `README.md` gives the fresh-clone environment-creation and regeneration commands. The source archive accompanies the manuscript, and the project repository is `https://github.com/WindBora/super-funicular`.

## Reviewer 1

### Numbered comments

1. **Title and method name.** We agree. The title is now "E-Polarized Plane-Wave Scattering by a Finite Sinusoidal PEC Grating via the Nyström Method." "Method of discrete singularities" is no longer used as the name of the present method; "Nyström method" or "Nyström technique" is used throughout for the proposed discretization. MDS remains only in the Introduction where the historical literature is described.

2. **Author names and affiliation.** Corrected. The ordinals were removed, the authors are listed as A. Petryshyn, S. V. Dukhopelnykov, and T. L. Zinenko, and the common affiliation is "O. Y. Usikov Institute for Radiophysics and Electronics of the National Academy of Sciences of Ukraine."

3. **Abstract terminology and residual claims.** Corrected. The abstract no longer presents a discrete-system residual as evidence of accuracy. It now states that interpolation-order convergence and agreement among independently implemented validation methods were demonstrated; the quantitative density and backend comparisons are reported in the Numerical Results section. "Finite-strip lobes" has been replaced by "finite-grating lobes."

4. **Index terms.** Corrected. "Nyström method" replaces "method of discrete singularities."

5. **Endpoint behavior.** Corrected. The manuscript now says explicitly that the inverse-square-root endpoint singularity is *extracted as a factor* and is not removed. The remaining scaled density is smooth.

6. **Incidence angle and time convention.** The formulation now retains an arbitrary incidence angle $\beta$; normal incidence is selected only for the numerical examples. The working source supplied for revision had already been corrected to the $e^{-i\omega t}$ convention, with $U^{\rm inc}=e^{ik\widehat{\boldsymbol d}\cdot\boldsymbol r}$, the outgoing $H_0^{(1)}$ kernel, and the matching Sommerfeld condition. We rechecked this convention end to end and regenerated the directional graphics so the incident and shadow sides are unambiguous. During that audit we additionally found and fixed the missing $1/k$ in the TSCS, as noted above.

7. **Choice of normalization.** Corrected. Geometry is no longer specified in wavelengths. The half-span $L$ is the reference length, the profile is described by $h/L$ and $P$, and frequency is represented by $kL$.

8. **Order of the numerical procedure.** Corrected. Section II now states: first, the logarithmically singular integral equation is differentiated to obtain a Cauchy-singular equation; second, that equation is discretized. The exact Cauchy coefficient $-2i/\pi$, the smooth remainder, the interlaced Chebyshev nodes, and the supplementary equation are all given explicitly.

9. **Scattering terminology and dimensions.** Corrected. "Scattering width" has been replaced by "total scattering cross section (TSCS)." The revised differential and total cross sections include the factor $1/k$ and therefore have dimensions of length.

10. **Residuals versus convergence.** Algebraic residuals of the solved discrete system were removed as evidence of accuracy. Standalone Fig. 2 instead plots the relative weighted $L^2$ error of the edge-regularized density as a function of interpolation order, using an $N=800$ reference solution. The error at $N=512$ is reported in the Numerical Results section; the abstract is intentionally free of mathematical notation and detailed convergence values. As a separate implementation check, each publication-case MAR solution is substituted into the original continuous IE at 513 independent target points. The production maximum absolute original-IE residual is required to remain below $10^{-5}$; the projection-doubled flat-strip check is required to remain below $10^{-7}$. This original-equation defect is explicitly distinguished from a residual of the linear system used to obtain the solution.

11. **Geometry and placement.** Corrected. The geometry is now introduced at the beginning of the problem formulation as $\boldsymbol r(t)=(Lt,h\cos(\pi Pt))$, $-1\le t\le1$, and the new Fig. 1 defines $L$, $h$, $P$, $\beta$, and $\varphi$. The pale $x=0$ and $y=0$ centerlines are the actual coordinate axes and show the symmetry axis and mean line; no separate bottom or left frame axes remain. The unexplained central marker has been removed, and a tick-ended dimension indicates $h$. The figure is placed by an ordinary single-column top-float environment at the upper right of the first page, without manual overlay or spacing tricks.

12. **Number of corrugations.** Corrected. All corrugated examples now use $P=5$ complete periods. The earlier two-period example and its figures were removed.

13. **Figure labels and number of curves.** Corrected. The revision now has eleven standalone figures. Fig. 2 is the Nyström-convergence plot, Fig. 3 is the flat-strip three-backend validation, Fig. 4 is the representative near field, and Fig. 5 is the representative polar pattern. Figs. 6--8 separately show $h/L=0$, $0.05$, and $0.10$ at $kL=20$; Figs. 9--11 separately show $kL=12$, $16$, and $20$ at $h/L=0.10$. Every polar figure contains one curve.

14. **Halfspace terminology and polar plots.** Corrected. "Hemisphere" has been replaced by "halfspace." All angular results in Figs. 5--11 are plotted in polar coordinates over $0\le\varphi<2\pi$. Their radial coordinate is the absolute, non-peak-normalized quantity $10\log_{10}[(d\sigma/d\varphi)/L]$.

15. **Visibility wording.** Corrected. The manuscript now says that a harmonic "becomes propagating" and describes the finite-grating features as broad maxima that develop near infinite-grating reference directions, not as exact diffraction orders.

16. **Former Section IV wording.** The discussion was reorganized and the unclear opening sentence removed. "Aperture diffraction" is not used. The forward feature is called the "shadow lobe," and every reference to an integrated measure uses "TSCS."

16. **Conclusion (duplicate numbering in the report).** Corrected. The conclusion now begins "An edge-conditioned integral equation, discretized with the Nyström technique..." It contains no residual-based accuracy claim and does not state that the TSCS "converges to" an exact value. Instead, it distinguishes numerical density convergence from the finite-frequency approach of $\sigma/(4L)$ to its high-frequency limit.

17. **References and relation to prior work.** Added and discussed in the Introduction: Nazarchuk (1989); Kobayashi and Eizawa (1991); Eizawa and Kobayashi, *Progress In Electromagnetics Research* (2014), DOI 10.2528/PIER14063007; Vinogradova, Kobayashi, and Eizawa, *Wave Motion* (2019), DOI 10.1016/j.wavemoti.2018.12.006; Vinogradova and Kobayashi, *IET Microwaves, Antennas & Propagation* (2021), DOI 10.1049/mia2.12166; and Shapoval, Sauleau, and Nosich, *IEEE Transactions on Antennas and Propagation* (2011), DOI 10.1109/TAP.2011.2161547. The Independent Validation Backends subsection additionally cites the actual methodological sources for the implementations used here: Nosich (1999, 2016) for coefficient-space MAR, Harrington (1967) for the general moment construction, and Hatamzadeh *et al.* (2008) for the block-pulse scattering treatment. The manuscript states explicitly that the MAR backend applies the general Nosich framework and is not a reproduction of the implementation in Vinogradova *et al.* (2019).

### Additional recommendations A-E

A. **Time dependence.** Adopted and verified throughout. As explained in response 6, the revision source already used $e^{-i\omega t}$; the final text now makes the propagation direction and the outgoing-kernel choice explicit. The same audit led us to additionally correct the TSCS factor $1/k$.

B. **Symmetric, $L$-scaled geometry and new Fig. 1.** Adopted. The profile is $(Lt,h\cos(\pi Pt))$, all geometrical ratios use $L$, and the new geometry figure defines every symbol used in the computation. Pale coordinate axes at $x=0$ and $y=0$ make the strip symmetry visible without separate bottom/left frame axes, the unexplained center marker is absent, and the figure uses the standard top-right first-page float placement described in response 11.

C. **Convergence evidence.** Adopted. Algebraic residuals of the solved Nyström system were removed as convergence evidence. The standalone Fig. 2 reports the relative $L^2_{1/\sqrt{1-t^2}}$ error of the interpolated smooth density for $N=32,48,64,96,128,192,256,384,512$, relative to $N=800$.

D. **Independent validation.** Adopted. Standalone Fig. 3 compares $\sigma/(4L)$ from differentiated Nyström, genuine coefficient-space MAR, and adapted block-pulse MoM backends at 41 values over $0.25\le kL\le20$. The manuscript reports the maximum spread among all three results and separate mode- and projection-doubling checks at $kL=20$. The fixed-$Q$ mode test and fixed-mode $Q$ test each compare both TSCS and the full complex pattern, so the effects of the two discretization controls are not conflated. It also gives the implementation provenance and numerical details summarized above. The five unique publication parameter cases are cross-checked by comparing 256-mode MAR with the $N=800$ Nyström reference in both TSCS and full complex-pattern relative $L^2$ norm, and by evaluating the MAR solution in the original IE at 513 independent targets. These values are written to the `mar_publication_crosscheck` data set in `revision_results.csv`.

E. **Geometrical-optics check.** Adopted. For a normally illuminated flat PEC strip of projected width $2L$, the manuscript states the asymptotic result $\sigma/(4L)\to1$ as $kL\to\infty$. The computed value at $kL=20$ is reported as a finite-frequency result and is deliberately not set equal to one. Fig. 3 shows its approach over the complete frequency sweep.

## Reviewer 2

1. **Institutional affiliation and author numbering.** Corrected exactly as requested: author ordinals were removed and the official institutional name is used in full.

2. **Undefined abbreviation.** Corrected. The abstract no longer uses "SIE." "Integral equation (IE)" is defined at first use in the body, and subsequent terminology is consistent.

## Revision-location map

The following map identifies where each response can be checked in the revised manuscript. Equation and figure labels are the stable LaTeX labels in `main.tex`.

| Comment | Revised location |
|---|---|
| Reviewer 1, #1 | Title and all manuscript-facing method terminology; Section "Formulation and Nyström Discretization". |
| Reviewer 1, #2 | Author/affiliation block on the title page. |
| Reviewer 1, #3 | Abstract; Section "Convergence and Cross-Validation"; Figs. 2 and 3. |
| Reviewer 1, #4 | IEEE keywords. |
| Reviewer 1, #5 | Abstract and Eq. `eq:edgefactor` in Section "Log-Singular Integral Equation". |
| Reviewer 1, #6 | Eqs. `eq:incident`--`eq:pec`; Fig. 1 and the near- and far-field results in Figs. 4--11. |
| Reviewer 1, #7 | Eq. `eq:geometry` and every numerical parameter statement in Section "Numerical Results". |
| Reviewer 1, #8 | The two-step paragraph preceding Eqs. `eq:ch`--`eq:cauchyie`, followed by Eqs. `eq:nodes`--`eq:supprow`. |
| Reviewer 1, #9 | Section "Far field and TSCS", especially Eq. `eq:tscs`. |
| Reviewer 1, #10 | Eq. `eq:l2error` and standalone Fig. 2; the former discrete-system residual table and claims are absent from the Abstract and Conclusion. The independent original-IE check for MAR is described in "Independent Validation Backends." |
| Reviewer 1, #11 | Opening of Section "Geometry and Boundary-Value Problem", Eq. `eq:geometry`, and Fig. 1. |
| Reviewer 1, #12 | Sections "Convergence and Cross-Validation" and "Representative Five-Period Grating"; corrugated cases in Figs. 2 and 4--11 use $P=5$. |
| Reviewer 1, #13 | Standalone Figs. 2--11; each polar figure in Figs. 5--11 contains one curve. |
| Reviewer 1, #14 | Sections "Geometry and Boundary-Value Problem" and "Far Field and TSCS"; the halfspaces in Fig. 4 and full-circle polar Figs. 5--11. |
| Reviewer 1, #15 | Section "Height and Electrical-Size Dependence", Eq. `eq:cutoff`, and Figs. 9--11. |
| Reviewer 1, #16 (former Section IV) | Sections "Representative Five-Period Grating" and "Height and Electrical-Size Dependence"; Figs. 4--11. |
| Reviewer 1, #16 (Conclusion) | Revised Conclusion. |
| Reviewer 1, #17 | Introduction and Refs. `nazarchuk1989`, `kobayashi1991`, `eizawa2014`, `vinogradova2019`, `vinogradova2021`, and `shapoval2011`; "Independent Validation Backends" and Refs. `nosich1999`, `nosich2016`, `harrington1967`, and `hatamzadeh2008`. |
| Recommendation A | Eq. `eq:incident`, the outgoing-kernel statement after Eq. `eq:pec`, and Eq. `eq:farfield`. |
| Recommendation B | Eq. `eq:geometry` and Fig. 1. |
| Recommendation C | Eq. `eq:l2error` and standalone Fig. 2. |
| Recommendation D | Sections "Independent Validation Backends" and "Convergence and Cross-Validation"; standalone Fig. 3; and the `flat_tscs_sweep`, `flat_order_doubling`, and `mar_publication_crosscheck` rows in `revision_results.csv`. |
| Recommendation E | Eq. `eq:tscs` and the GO line and discussion accompanying Fig. 3. |
| Reviewer 2, #1 | Author/affiliation block on the title page. |
| Reviewer 2, #2 | Abstract and the first definition of "integral equation (IE)" before Eq. `eq:ie`. |
