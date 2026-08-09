# Response to Reviewers

We thank both reviewers for their careful reading and constructive recommendations. The manuscript has been substantially rewritten. It now uses Nyström terminology throughout, a symmetric five-period geometry scaled by its half-span $L$, an arbitrary-incidence formulation, absolute polar scattering plots, density-based convergence, and a three-backend numerical comparison.

We also audited every field normalization. That audit identified an additional issue not raised explicitly in the reports: the former TSCS expression lacked its dimensional factor $1/k$. The revised definition is

$$
\sigma=\frac{1}{k}\int_0^{2\pi}|\Phi_{\rm sc}(\varphi)|^2\,d\varphi.
$$

The resulting TSCS has units of length and can be compared meaningfully with the geometrical-optics value $4L$.

## Reviewer 1

### Numbered comments

1. **Title and method name.** We agree. The title is now "E-Polarized Plane-Wave Scattering by a Finite Sinusoidal PEC Grating via the Nyström Method." "Method of discrete singularities" has been removed from the title and manuscript; "Nyström method" or "Nyström technique" is used throughout.

2. **Author names and affiliation.** Corrected. The ordinals were removed, the authors are listed as A. Petryshyn, S. Dukhopelnykov, and T. Zinenko, and the common affiliation is "O. Ya. Usikov Institute for Radiophysics and Electronics of the National Academy of Sciences of Ukraine."

3. **Abstract terminology and residual claims.** Corrected. The abstract no longer presents a discrete-system residual as evidence of accuracy. It now reports the relative $L^2$ density error and the spread among three separately implemented numerical backends. "Finite-strip lobes" has been replaced by "finite-grating lobes."

4. **Index terms.** Corrected. "Nyström method" replaces "method of discrete singularities."

5. **Endpoint behavior.** Corrected. The manuscript now says explicitly that the inverse-square-root endpoint singularity is *extracted as a factor* and is not removed. The remaining scaled density is smooth.

6. **Incidence angle and time convention.** The formulation now retains an arbitrary incidence angle $\beta$; normal incidence is selected only for the numerical examples. The working source supplied for revision had already been corrected to the $e^{-i\omega t}$ convention, with $U^{\rm inc}=e^{ik\widehat{\boldsymbol d}\cdot\boldsymbol r}$, the outgoing $H_0^{(1)}$ kernel, and the matching Sommerfeld condition. We rechecked this convention end to end and regenerated the directional graphics so the incident and shadow sides are unambiguous. During that audit we additionally found and fixed the missing $1/k$ in the TSCS, as noted above.

7. **Choice of normalization.** Corrected. Geometry is no longer specified in wavelengths. The half-span $L$ is the reference length, the profile is described by $h/L$ and $P$, and frequency is represented by $kL$.

8. **Order of the numerical procedure.** Corrected. Section II now states: first, the logarithmically singular integral equation is differentiated to obtain a Cauchy-singular equation; second, that equation is discretized. The exact Cauchy coefficient $-2i/\pi$, the smooth remainder, the interlaced Chebyshev nodes, and the supplementary equation are all given explicitly.

9. **Scattering terminology and dimensions.** Corrected. "Scattering width" has been replaced by "total scattering cross section (TSCS)." The revised differential and total cross sections include the factor $1/k$ and therefore have dimensions of length.

10. **Residuals versus convergence.** All residual definitions, values, and claims were removed. Fig. 2(a) instead plots the relative weighted $L^2$ error of the edge-regularized density as a function of interpolation order, using an $N=800$ reference solution. The error at $N=512$ is reported in the numerical results; the abstract is intentionally free of mathematical notation and detailed convergence values.

11. **Geometry and placement.** Corrected. The geometry is now introduced at the beginning of the problem formulation as $\boldsymbol r(t)=(Lt,h\cos(\pi Pt))$, $-1\le t\le1$, and the new Fig. 1 defines $L$, $h$, $P$, $\beta$, and $\varphi$.

12. **Number of corrugations.** Corrected. All corrugated examples now use $P=5$ complete periods. The earlier two-period example and its figures were removed.

13. **Figure labels and number of curves.** Corrected. Labels and markers were repositioned to avoid data. Figures 4 and 5 each contain three polar panels with one curve per panel. The five revised figures replace the earlier six.

14. **Halfspace terminology and polar plots.** Corrected. "Hemisphere" has been replaced by "halfspace." All angular results are plotted in polar coordinates over $0\le\varphi<2\pi$. Their radial coordinate is the absolute, non-peak-normalized quantity $10\log_{10}[(d\sigma/d\varphi)/L]$.

15. **Visibility wording.** Corrected. The manuscript now says that a harmonic "becomes propagating" and describes the finite-grating features as broad maxima that develop near infinite-grating reference directions, not as exact diffraction orders.

16. **Former Section IV wording.** The discussion was reorganized and the unclear opening sentence removed. "Aperture diffraction" is not used. The forward feature is called the "shadow lobe," and every reference to an integrated measure uses "TSCS."

16. **Conclusion (duplicate numbering in the report).** Corrected. The conclusion now begins "An edge-conditioned integral equation, discretized with the Nyström technique..." It contains no residual-based accuracy claim and does not state that the TSCS "converges to" an exact value. Instead, it distinguishes numerical density convergence from the finite-frequency approach of $\sigma/(4L)$ to its high-frequency limit.

17. **References and relation to prior work.** Added and discussed in the Introduction: Nazarchuk (1989); Kobayashi and Eizawa (1991); Eizawa and Kobayashi, *Progress In Electromagnetics Research* (2014), DOI 10.2528/PIER14063007; Vinogradova, Kobayashi, and Eizawa, *Wave Motion* (2019), DOI 10.1016/j.wavemoti.2018.12.006; Vinogradova and Kobayashi, *IET Microwaves, Antennas & Propagation* (2021), DOI 10.1049/mia2.12166; and Shapoval, Sauleau, and Nosich, *IEEE Transactions on Antennas and Propagation* (2011), DOI 10.1109/TAP.2011.2161547.

### Additional recommendations A-E

A. **Time dependence.** Adopted and verified throughout. As explained in response 6, the revision source already used $e^{-i\omega t}$; the final text now makes the propagation direction and the outgoing-kernel choice explicit. The same audit led us to additionally correct the TSCS factor $1/k$.

B. **Symmetric, $L$-scaled geometry and new Fig. 1.** Adopted. The profile is $(Lt,h\cos(\pi Pt))$, all geometrical ratios use $L$, and the new geometry figure defines every symbol used in the computation.

C. **Convergence evidence.** Adopted. Algebraic residuals were removed. The new Fig. 2(a) reports the relative $L^2_{1/\sqrt{1-t^2}}$ error of the interpolated smooth density for $N=32,48,64,96,128,192,256,384,512$, relative to $N=800$.

D. **Independent validation.** Adopted. Fig. 2(b) compares $\sigma/(4L)$ from differentiated Nyström, analytical-regularization, and pulse-basis method-of-moments backends at 41 values over $0.25\le kL\le20$. The manuscript reports the maximum spread among all three results and order-doubling checks at $kL=20$.

E. **Geometrical-optics check.** Adopted. For a normally illuminated flat PEC strip of projected width $2L$, the manuscript states the asymptotic result $\sigma/(4L)\to1$ as $kL\to\infty$. The computed value at $kL=20$ is reported as a finite-frequency result and is deliberately not set equal to one. Fig. 2(b) shows its approach over the complete frequency sweep.

## Reviewer 2

1. **Institutional affiliation and author numbering.** Corrected exactly as requested: author ordinals were removed and the official institutional name is used in full.

2. **Undefined abbreviation.** Corrected. The abstract no longer uses "SIE." "Integral equation (IE)" is defined at first use in the body, and subsequent terminology is consistent.

## Revision-location map

The following map identifies where each response can be checked in the revised manuscript. Equation and figure labels are the stable LaTeX labels in `main.tex`.

| Comment | Revised location |
|---|---|
| Reviewer 1, #1 | Title and all manuscript-facing method terminology; Section "Formulation and Nyström Discretization". |
| Reviewer 1, #2 | Author/affiliation block on the title page. |
| Reviewer 1, #3 | Abstract; Section "Convergence and cross-validation"; Fig. 2. |
| Reviewer 1, #4 | IEEE keywords. |
| Reviewer 1, #5 | Abstract and Eq. `eq:edgefactor` in Section "Edge-conditioned integral equation". |
| Reviewer 1, #6 | Eqs. `eq:incident`--`eq:pec`; Figs. 1 and 3. |
| Reviewer 1, #7 | Eq. `eq:geometry` and every numerical parameter statement in Section "Numerical Results". |
| Reviewer 1, #8 | The two-step paragraph preceding Eqs. `eq:ch`--`eq:cauchyie`, followed by Eqs. `eq:nodes`--`eq:supprow`. |
| Reviewer 1, #9 | Section "Far field and TSCS", especially Eq. `eq:tscs`. |
| Reviewer 1, #10 | Eq. `eq:l2error` and Fig. 2(a); the former residual table and claims are absent from the Abstract and Conclusion. |
| Reviewer 1, #11 | Opening of Section "Geometry and Boundary-Value Problem", Eq. `eq:geometry`, and Fig. 1. |
| Reviewer 1, #12 | Section "Representative five-period grating" and Figs. 3--5, all with $P=5$. |
| Reviewer 1, #13 | Figs. 4 and 5: three panels, one curve per panel, with labels outside the data. |
| Reviewer 1, #14 | Sections "Geometry and boundary-value problem" and "Far field and TSCS"; full-circle polar Figs. 3--5. |
| Reviewer 1, #15 | Section "Height and Electrical-Size Dependence", Eq. `eq:cutoff`, and Fig. 5. |
| Reviewer 1, #16 (former Section IV) | Sections "Representative five-period grating" and "Height and electrical-size dependence"; Figs. 3--5. |
| Reviewer 1, #16 (Conclusion) | Revised Conclusion. |
| Reviewer 1, #17 | Introduction and Refs. `nazarchuk1989`, `kobayashi1991`, `eizawa2014`, `vinogradova2019`, `vinogradova2021`, and `shapoval2011`. |
| Recommendation A | Eq. `eq:incident`, the outgoing-kernel statement after Eq. `eq:pec`, and Eq. `eq:farfield`. |
| Recommendation B | Eq. `eq:geometry` and Fig. 1. |
| Recommendation C | Eq. `eq:l2error` and Fig. 2(a). |
| Recommendation D | Section "Convergence and Cross-Validation", Fig. 2(b), and the `flat_tscs_sweep`/`flat_order_doubling` rows in `revision_results.csv`. |
| Recommendation E | Eq. `eq:tscs` and the GO line and discussion accompanying Fig. 2(b). |
| Reviewer 2, #1 | Author/affiliation block on the title page. |
| Reviewer 2, #2 | Abstract and the first definition of "integral equation (IE)" before Eqs. `eq:potential`--`eq:ie`. |
