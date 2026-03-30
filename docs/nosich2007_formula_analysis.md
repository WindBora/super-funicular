# Analysis of `nosich2007.pdf`

Source: `docs/nosich2007.pdf`

Status:

- Full paper read.
- Numbered equations (1) to (14) cross-checked against rendered page images.
- Some inline geometry formulas in the later numerical sections are only partly recoverable from the PDF text layer; where needed, I reconstructed them from the page images and from the notation already established earlier in the paper.
- I do not see a mathematical inconsistency in the derivation. The main places that need care are notation compression and normalization changes between the integral equations.

## 1. What the paper does

The paper develops an accurate full-wave method for 2-D PEC multireflector antennas in the E-polarized case. The logic is:

1. Formulate the open-arc Helmholtz scattering problem.
2. Represent the scattered field by coupled singular integral equations (SIEs) for the induced surface currents.
3. Reparameterize the reflector contours and factor out the edge singularity of the current.
4. Convert the logarithmic first-kind SIEs into a Cauchy-type system plus supplementary conditions.
5. Discretize with a Nystrom/MDS scheme at Chebyshev nodes.
6. Recover the currents, then compute the far-field pattern and directivity.

This chain is mathematically coherent.

## 2. Core formulas

### 2.1 Governing scattering problem

The reflectors are smooth open PEC arcs

```math
L = \{L_q\}_{q=1}^Q \subset \mathbb{R}^2,
```

with the total field written as

```math
U^{\mathrm{tot}} = U_0 + U, \qquad U = \sum_{q=1}^Q U_q.
```

The scattered field satisfies the 2-D Helmholtz equation outside the reflector set:

```math
(\Delta + k^2) U(x,y) = 0, \qquad (x,y)\in \mathbb{R}^2 \setminus L. \tag{1}
```

Here

```math
k = \omega/c = 2\pi/\lambda.
```

For E-polarization, the PEC condition is Dirichlet:

```math
U(x,y)\big|_{L_p} = -\,U_0(x,y)\big|_{L_p}. \tag{2}
```

Radiation condition:

```math
\frac{\partial U(x,y)}{\partial r} - i k\,U(x,y) = o(r^{-1/2}), \qquad r\to\infty. \tag{3}
```

Local energy finiteness near the edges:

```math
\int_{\Omega} \left(k^2 |U|^2 + |\nabla U|^2\right)\,d\sigma < \infty, \tag{4}
```

with `\Omega` any bounded domain containing the endpoints of the open arcs.

### 2.2 Complex-source-point feed

The illuminating field is the 2-D complex-source-point (CSP) field

```math
U_0(\mathbf r) = H_0^{(1)}\!\left(k\left|\mathbf r - \widetilde{\mathbf r}_0\right|\right),
\qquad
\mathbf r=(x,y), \tag{5}
```

with

```math
\widetilde{\mathbf r}_0 = \mathbf r_0 + i\mathbf b,
\qquad
\mathbf r_0=(x_0,y_0),
\qquad
\mathbf b=(b\cos\beta,\; b\sin\beta).
```

What this means:

- `b` controls beam width. Larger `kb` means a narrower beam.
- `\beta` is the beam looking direction.
- The field is an exact Helmholtz solution, not a paraxial approximation.
- In the near zone it behaves like a Gaussian beam; in the far zone it becomes cylindrical.
- Because of the complex source shift, a branch cut of length `2b` is required and must not intersect any reflector.

### 2.3 First-kind SIE for the induced currents

Using the paper's current normalization, the problem is reduced to coupled electric-field SIEs:

```math
\sum_{q=1}^{Q}
\int_{L_q}
H_0^{(1)}\!\left(k\left|\mathbf r_q(s_q)-\mathbf r_p(s_{p,0})\right|\right)
j_q(s_q)\,ds_q

= -\,U_0\!\left(\mathbf r_p(s_{p,0})\right),
\qquad p=1,\dots,Q. \tag{6}
```

This is the correct 2-D single-layer representation for a Dirichlet problem on open PEC arcs.

### 2.4 Reparameterization and edge-singularity extraction

Each reflector is parameterized by a smooth map

```math
\mathbf r_q(t) = (x_q(t), y_q(t)), \qquad t\in[-1,1].
```

The arc-length Jacobian is

```math
\left|\mathbf r_q'(t)\right|
= \sqrt{\dot x_q(t)^2 + \dot y_q(t)^2}.
```

The current has the expected open-edge singularity, so the paper factors it out by introducing a smooth unknown `v_q(t)`:

```math
j_q\!\big(s_q(t)\big)\,\left|\mathbf r_q'(t)\right|
= v_q(t)\,(1-t^2)^{-1/2}.
```

This step is mathematically important and correct:

- The physical current on an open PEC arc has inverse-square-root endpoint behavior.
- After factoring `(1-t^2)^{-1/2}`, the new unknown `v_q` is smooth enough for a high-order Nystrom discretization.

The reparameterized canonical system is

```math
\sum_{q=1}^{Q}
\int_{-1}^{1}
H_0^{(1)}\!\left(k R_{pq}(t,t_0)\right)\,
v_q(t)\,
\frac{dt}{(1-t^2)^{1/2}}

= -\,U_{p,0}(t_0), \tag{7}
```

where

```math
R_{pq}(t,t_0)
= \left(
[x_q(t)-x_p(t_0)]^2 + [y_q(t)-y_p(t_0)]^2
\right)^{1/2},
```

and `U_{p,0}(t_0)` denotes the incident field evaluated on reflector `p`.

Note:

- Between (6) and (7) the paper compresses notation.
- In practice, one must keep track of both the Jacobian and the Green-function prefactor consistently.
- The derivation itself is sound; the normalization is just packed into the new notation.

### 2.5 Cauchy-type SIE system used by MDS

Direct discretization of (7) is inefficient because of the logarithmic singularity. The paper differentiates (7) with respect to `t_0` and adds supplementary integral conditions. The resulting system is

```math
\sum_{q=1}^{Q}
\left[
\delta_{pq}
\int_{-1}^{1}
\frac{v_q(t)}{t-t_0}\,
\frac{dt}{(1-t^2)^{1/2}}
\;+\;
\frac{1}{k}
\int_{-1}^{1}
K_{pq}(t,t_0)\,
\frac{v_q(t)\,dt}{(1-t^2)^{1/2}}
\right]
= f_p(t_0),
```

```math
\sum_{q=1}^{Q}
\left[
\frac{1}{k}
\int_{-1}^{1}
M_{pq}(t)\,v_q(t)\,
\frac{dt}{(1-t^2)^{1/2}}
\right]
= c_p,
\qquad p=1,\dots,Q. \tag{8}
```

The paper then defines the smooth kernels

```math
K_{pq}(t,t_0)
= \frac{\partial}{\partial t_0}
H_0^{(1)}\!\left(kR_{pq}(t,t_0)\right)
- \frac{\delta_{pq}}{t-t_0}, \tag{9}
```

```math
M_{pq}(t)
= \int_{-1}^{1}
H_0^{(1)}\!\left(kR_{pq}(t,t_0)\right)\,
\frac{dt_0}{(1-t_0^2)^{1/2}}. \tag{10}
```

The paper also uses

```math
f_p(t_0) = -\,\frac{dU_{p,0}(t_0)}{dt_0},
\qquad
c_p = - \int_{-1}^{1} U_{p,0}(t_0)\,(1-t_0^2)^{-1/2}\,dt_0.
```

About `\delta_{pq}`:

- It is the Kronecker delta.
- Only the self interaction `q=p` carries the Cauchy singular part.
- This is exactly what one expects physically and mathematically.

Why this step is logical:

- Differentiating the logarithmic kernel raises the singularity to Cauchy type.
- Differentiation alone would lose one integral constant; the supplementary conditions restore equivalence.
- This is exactly the right way to turn a first-kind logarithmic equation into a numerically stronger system for open-arc scattering.

### 2.6 MDS / Nystrom discretization

The quadrature nodes are the Chebyshev roots

```math
t_i^{(n)} = \cos\!\frac{(2i-1)\pi}{2n}, \qquad i=1,\dots,n,
```

and

```math
t_{0j}^{(n)} = \cos\!\frac{j\pi}{n}, \qquad j=1,\dots,n-1.
```

The discrete linear system is

```math
\frac{1}{n}
\sum_{q=1}^{Q}
\left[
\delta_{pq}
\sum_{i=1}^{n}
\frac{v_q\!\left(t_i^{(n)}\right)}{t_i^{(n)}-t_{0j}^{(n)}}
\;+\;
\sum_{i=1}^{n}
K_{pq}\!\left(t_i^{(n)},t_{0j}^{(n)}\right)
v_q\!\left(t_i^{(n)}\right)
\right]
= f_p\!\left(t_{0j}^{(n)}\right),
```

```math
\frac{1}{n}
\sum_{q=1}^{Q}
\left[
\sum_{i=1}^{n}
M_{pq}\!\left(t_i^{(n)}\right)
v_q\!\left(t_i^{(n)}\right)
\right]
= c_p,
\qquad
p=1,\dots,Q, \quad j=1,\dots,n-1. \tag{11}
```

This is the main computational formula of the paper.

What to be careful about in implementation:

- The first line gives `(n-1)` equations per reflector.
- The second line gives the missing one equation per reflector.
- Together they produce `nQ` unknowns and `nQ` equations.
- This closure is mathematically correct and is exactly why the supplementary condition is necessary.

### 2.7 Far-field pattern

The total radiation pattern is

```math
\Phi(\varphi_0)
= \Phi_0(\varphi_0) + \Phi_{\mathrm{sc}}(\varphi_0)
= \lim_{kr\to\infty}
\sqrt{kr}\,e^{-ikr}
\left[
U(r,\varphi_0)+U_0(r,\varphi_0)
\right]. \tag{12}
```

For the CSP source in free space,

```math
\Phi_0(\varphi_0) = e^{kb\cos(\varphi_0-\beta)}.
```

The scattered-field contribution is approximated by

```math
\Phi_{\mathrm{sc}}(\varphi_0)
\approx
\frac{1}{n}\left(\frac{2}{\pi}\right)^{1/2} e^{-i\pi/4}
\sum_{q=1}^{Q}
\left[
\sum_{i=1}^{n}
e^{-ik r_q(t_i^{(n)})\cos(\varphi_q(t_i^{(n)})-\varphi_0)}
\,v_q(t_i^{(n)})
\right]. \tag{13}
```

This is the correct asymptotic far-field formula after replacing the boundary integral by the same interpolation-type quadrature.

### 2.8 Directivity

The antenna directivity is

```math
D
= \frac{2\pi}{P}\,|\Phi(\varphi_{\max})|^2,
\qquad
P = \int_{0}^{2\pi} |\Phi(\varphi_0)|^2\,d\varphi_0. \tag{14}
```

The paper also gives the directivity of the free-space CSP feed:

```math
D_0 = \frac{e^{2kb}}{I_0(2kb)},
```

where `I_0` is the modified Bessel function.

This formula is correct, because

```math
|\Phi_0(\varphi)|^2 = e^{2kb\cos(\varphi-\beta)},
```

and

```math
\int_0^{2\pi} e^{2kb\cos(\varphi-\beta)}\,d\varphi = 2\pi I_0(2kb).
```

## 3. Geometry and design formulas from the numerical sections

### 3.1 Symmetric parabolic reflector

The paper defines the symmetric parabolic reflector by the parabola

```math
L = \left\{(x,y)\in\mathbb R^2:\; x=\frac{y^2}{4f},\; y_1<y<y_2 \right\},
\qquad y_1=-y_2.
```

The feed is placed at the geometric focus:

```math
(x_0,y_0)=(f,0),
```

and is aimed toward the reflector center (`\beta = 180^\circ` in the paper's setup).

### 3.2 Edge illumination

The edge-illumination parameter is defined as the ratio of incident-field magnitude at the reflector edge to that at the reflector center:

```math
E_{\mathrm{edge}}
= 20\log_{10}\left|
\frac{U_0^{\mathrm{edge}}}{U_0^{\mathrm{center}}}
\right| \ \text{dB}.
```

This is the correct amplitude definition in decibels. The main numerical conclusion of the paper is:

- single reflectors are typically optimal near `-10 dB`,
- two-reflector systems often want a stricter illumination.

### 3.3 Offset parabolic reflector

The offset reflector uses the same parent parabola,

```math
x = \frac{y^2}{4f},
```

but with asymmetric truncation:

```math
y_1 \neq -y_2.
```

The paper uses the aperture measures

```math
d = y_2 - y_1,
\qquad
a = 2y_2,
```

where `a` is the full aperture of the corresponding symmetric "virtual" parabola.

### 3.4 Two-reflector examples

The later sections instantiate the same solver for:

- Cassegrain antennas: parabolic main reflector plus hyperbolic subreflector.
- PACO antennas: parabolic subreflector plus a corner/conical reflector.

Those contour formulas are geometry inputs to the same MDS machinery. The derivation of the solver itself does not change.

## 4. Logical verification of the derivation

### 4.1 Equation (1) to (4)

No issue.

- Standard open-arc exterior Helmholtz scattering problem.
- Dirichlet boundary condition matches the E-polarized PEC case.
- Radiation and local-energy conditions are the correct uniqueness conditions for open arcs.

### 4.2 Equation (5)

No issue.

- The CSP field is an exact Helmholtz solution.
- The branch-cut discussion is mathematically necessary.
- The beam-sharpening role of `kb` is correct.

### 4.3 Equation (6)

No issue.

- In this paper, the usual scalar Green-function prefactor is absorbed into the current normalization, so Eq. (6) is written without an explicit `i/4`.
- Unknowns are the electric surface currents on open PEC arcs.
- This is a first-kind SIE with logarithmic kernel singularity, as expected.

### 4.4 Equation (6) to (7)

Logical, but easy to misread if one implements it too quickly.

- The contour parameterization changes variables from arc length to `t\in[-1,1]`.
- The edge singularity is factored into `(1-t^2)^{-1/2}`.
- The paper compresses the normalization and Jacobian bookkeeping into new symbols; this is acceptable mathematically, but in code one must keep those factors explicit.

### 4.5 Equation (7) to (8)

Logical and important.

- Differentiating a logarithmic kernel gives a Cauchy singularity.
- The missing constant of integration is restored by the supplementary condition.
- The new kernels `K_{pq}` and `M_{pq}` are smooth after singular-part subtraction.

This is the key analytical regularization step of the paper.

### 4.6 Equation (11)

Logical and dimensionally correct.

- Chebyshev first-kind nodes handle the weighted integrals well.
- Second-kind-node testing points avoid singular coincidence.
- The count of equations equals the count of unknowns once the supplementary equation is added.

### 4.7 Equation (12) to (14)

No issue.

- The far-field definition is standard for 2-D problems.
- The asymptotic factor `\left(2/\pi\right)^{1/2} e^{-i\pi/4}` is the correct Hankel far-field prefactor.
- The directivity definition is standard.
- The CSP-feed directivity formula checks out analytically.

## 5. Final formulas you would actually use in calculations

If the goal is implementation or reproduction of the paper's numerical results, the essential formula chain is:

### Step A. Define the feed

```math
U_0(\mathbf r) = H_0^{(1)}\!\left(k\left|\mathbf r - \widetilde{\mathbf r}_0\right|\right),
\qquad
\widetilde{\mathbf r}_0 = \mathbf r_0 + i\mathbf b.
```

### Step B. Parameterize each reflector

```math
\mathbf r_q(t)=(x_q(t),y_q(t)), \qquad t\in[-1,1].
```

### Step C. Use the edge-corrected current representation

```math
j_q\!\big(s_q(t)\big)\,\left|\mathbf r_q'(t)\right|
= v_q(t)\,(1-t^2)^{-1/2}.
```

### Step D. Assemble and solve the MDS matrix system

```math
\text{Solve (11) for } v_q\!\left(t_i^{(n)}\right).
```

### Step E. Compute far-field pattern

```math
\Phi(\varphi_0)=\Phi_0(\varphi_0)+\Phi_{\mathrm{sc}}(\varphi_0),
```

with

```math
\Phi_0(\varphi_0)=e^{kb\cos(\varphi_0-\beta)},
```

and `\Phi_{\mathrm{sc}}` from (13).

### Step F. Compute total radiated power and directivity

```math
P=\int_0^{2\pi}|\Phi(\varphi_0)|^2\,d\varphi_0,
\qquad
D=\frac{2\pi}{P}|\Phi(\varphi_{\max})|^2.
```

### Step G. Compute edge illumination

```math
E_{\mathrm{edge}}
=20\log_{10}\left|
\frac{U_0^{\mathrm{edge}}}{U_0^{\mathrm{center}}}
\right| \ \text{dB}.
```

## 6. Bottom-line assessment

I do not find a broken mathematical step in the paper.

The derivation is internally consistent:

- PDE formulation is standard.
- The integral-equation representation is standard.
- The edge singularity extraction is physically and mathematically correct.
- The differentiation plus supplementary condition is the right regularizing move.
- The Chebyshev-node Nystrom discretization matches the weighted singular structure.
- The radiation-pattern and directivity formulas are correct.

The only real caution is notational:

- the paper compresses some normalization factors when moving from (6) to (7) and then to (8),
- so an implementation should keep those factors explicit rather than relying on visual memory of the printed formulas.

If you want, I can next produce a second markdown file that turns this into an implementation-ready derivation with code-oriented notation, one equation per line and no prose compression.
