# Implementation-Ready Derivation for `nosich2007.pdf`

Source: `docs/nosich2007.pdf`

Verification status:

- Equations (1) to (14) rechecked against rendered page images.
- The implementation below follows the paper's normalization exactly.
- Important consistency fix: in the printed Eq. (11), the first discrete Cauchy sum is shown without `\delta_{pq}`; consistency with Eq. (8) requires multiplying that term by `\delta_{pq}`.

## 0. Conventions

```math
i^2 = -1.
```

```math
\text{paper } j \text{ as imaginary unit } \longrightarrow \text{ code } i \text{ or } 1j.
```

```math
Q = \text{number of reflectors}.
```

```math
n = \text{MDS interpolation order}.
```

```math
k = 2\pi/\lambda.
```

```math
\mathbf r = (x,y).
```

```math
\mathbf r_q(t) = (x_q(t),y_q(t)).
```

```math
\mathbf r_q'(t) = (x_q'(t),y_q'(t)).
```

```math
\gamma_q(t) = \sqrt{x_q'(t)^2 + y_q'(t)^2}.
```

```math
\omega(t) = (1-t^2)^{-1/2}.
```

## 1. Geometry input

```math
t \in [-1,1].
```

```math
s_q(t) = \frac{|L_q|}{2}(t+1).
```

```math
t = \frac{2s_q}{|L_q|} - 1.
```

```math
|L_q| = \int_{-1}^{1} \gamma_q(t)\,dt.
```

```math
\mathbf r_q(t) = (x_q(t),y_q(t)).
```

```math
\gamma_q(t) = \left|\mathbf r_q'(t)\right|.
```

## 2. Incident CSP field

```math
\mathbf r_c = \mathbf r_0 + i\mathbf b.
```

```math
\mathbf r_0 = (x_0,y_0).
```

```math
\mathbf b = (b\cos\beta,\; b\sin\beta).
```

```math
U_0(\mathbf r) = H_0^{(1)}\!\left(k\left|\mathbf r-\mathbf r_c\right|\right).
```

```math
\rho(\mathbf r) = \left|\mathbf r-\mathbf r_c\right|.
```

```math
\Phi_0(\varphi) = e^{kb\cos(\varphi-\beta)}.
```

```math
D_0 = \frac{e^{2kb}}{I_0(2kb)}.
```

## 3. Scattering problem

```math
(\Delta + k^2)U(x,y) = 0.
```

```math
U(x,y)\big|_{L} = -U_0(x,y)\big|_{L}.
```

```math
\frac{\partial U}{\partial r} - ikU = o(r^{-1/2}).
```

```math
\int_{\Omega} \left(k^2|U|^2 + |\nabla U|^2\right)\,d\sigma < \infty.
```

## 4. Boundary integral equation

```math
\sum_{q=1}^{Q}
\int_{L_q}
H_0^{(1)}\!\left(k\left|\mathbf r_p(s_p)-\mathbf r_q(s_q)\right|\right)
j_q(s_q)\,ds_q
= -U_0\!\left(\mathbf r_p(s_p)\right).
```

## 5. Edge-singularity factorization

```math
j_q\!\big(s_q(t)\big)\,\gamma_q(t) = v_q(t)\,\omega(t).
```

```math
j_q\!\big(s_q(t)\big) = \frac{v_q(t)}{\gamma_q(t)\sqrt{1-t^2}}.
```

```math
R_{pq}(t,\tau) = \left|\mathbf r_q(t)-\mathbf r_p(\tau)\right|.
```

```math
U_{p,0}(\tau) = U_0(\mathbf r_p(\tau)).
```

```math
\sum_{q=1}^{Q}
\int_{-1}^{1}
H_0^{(1)}\!\left(kR_{pq}(t,\tau)\right)
v_q(t)\,\omega(t)\,dt
= -U_{p,0}(\tau).
```

## 6. Differentiated system

```math
f_p(\tau) = -\frac{dU_{p,0}(\tau)}{d\tau}.
```

```math
c_p = -\int_{-1}^{1} U_{p,0}(\tau)\,\omega(\tau)\,d\tau.
```

```math
K_{pq}(t,\tau) = \frac{\partial}{\partial \tau}H_0^{(1)}\!\left(kR_{pq}(t,\tau)\right) - \frac{\delta_{pq}}{t-\tau}.
```

```math
M_{pq}(t) = \int_{-1}^{1} H_0^{(1)}\!\left(kR_{pq}(t,\tau)\right)\,\omega(\tau)\,d\tau.
```

```math
\sum_{q=1}^{Q}
\left[
\delta_{pq}
\int_{-1}^{1}
\frac{v_q(t)}{t-\tau}\,\omega(t)\,dt
+
\int_{-1}^{1}
K_{pq}(t,\tau)\,v_q(t)\,\omega(t)\,dt
\right]
= f_p(\tau).
```

```math
\sum_{q=1}^{Q}
\int_{-1}^{1}
M_{pq}(t)\,v_q(t)\,\omega(t)\,dt
= c_p.
```

## 7. Explicit derivatives for code

```math
\frac{\partial}{\partial \tau}H_0^{(1)}\!\left(kR_{pq}(t,\tau)\right)
= -k\,H_1^{(1)}\!\left(kR_{pq}(t,\tau)\right)\,\frac{\partial R_{pq}(t,\tau)}{\partial \tau}.
```

```math
\frac{\partial R_{pq}(t,\tau)}{\partial \tau}
= -\frac{
\left(x_q(t)-x_p(\tau)\right)x_p'(\tau)
+
\left(y_q(t)-y_p(\tau)\right)y_p'(\tau)
}{
R_{pq}(t,\tau)
}.
```

```math
\rho_p(\tau) = \left|\mathbf r_p(\tau)-\mathbf r_c\right|.
```

```math
\frac{d\rho_p(\tau)}{d\tau}
= \frac{
\left(\mathbf r_p(\tau)-\mathbf r_c\right)\cdot \mathbf r_p'(\tau)
}{
\rho_p(\tau)
}.
```

```math
\frac{dU_{p,0}(\tau)}{d\tau}
= -k\,H_1^{(1)}\!\left(k\rho_p(\tau)\right)\,\frac{d\rho_p(\tau)}{d\tau}.
```

```math
f_p(\tau)
= k\,H_1^{(1)}\!\left(k\rho_p(\tau)\right)\,
\frac{
\left(\mathbf r_p(\tau)-\mathbf r_c\right)\cdot \mathbf r_p'(\tau)
}{
\rho_p(\tau)
}.
```

## 8. Discretization nodes

```math
t_i = \cos\!\frac{(2i-1)\pi}{2n}.
```

```math
i = 1,2,\dots,n.
```

```math
\tau_j = \cos\!\frac{j\pi}{n}.
```

```math
j = 1,2,\dots,n-1.
```

```math
u_{q,i} = v_q(t_i).
```

## 9. Discrete linear system

Boundary rows:

```math
A_{(p,j),(q,i)} = \frac{1}{n}\left[\delta_{pq}\frac{1}{t_i-\tau_j} + K_{pq}(t_i,\tau_j)\right].
```

```math
b_{(p,j)} = f_p(\tau_j).
```

```math
p = 1,2,\dots,Q.
```

```math
j = 1,2,\dots,n-1.
```

Supplementary rows:

```math
A_{(p,n),(q,i)} = \frac{1}{n} M_{pq}(t_i).
```

```math
b_{(p,n)} = c_p.
```

```math
p = 1,2,\dots,Q.
```

Unknown packing:

```math
m(q,i) = (q-1)n + i.
```

```math
\mathbf u_{m(q,i)} = u_{q,i}.
```

System solve:

```math
A\mathbf u = \mathbf b.
```

Implementation correction to the printed paper:

```math
\text{Use } \delta_{pq}\frac{1}{t_i-\tau_j} \text{ in } A_{(p,j),(q,i)}.
```

```math
\text{Do not apply the Cauchy self-term to } q \ne p.
```

## 10. Continuous terms needed by the matrix

For `p \ne q`:

```math
M_{pq}(t_i) = \int_{-1}^{1} H_0^{(1)}\!\left(kR_{pq}(t_i,\tau)\right)\,\omega(\tau)\,d\tau.
```

For `p = q`:

```math
H_0^{(1)}(z) = 1 + \frac{2i}{\pi}\left[\ln\!\left(\frac{z}{2}\right) + \gamma_{\mathrm E}\right] + O(z^2\ln z).
```

For `p = q`:

```math
M_{pp}(t_i)
= \int_{-1}^{1}
\left[
H_0^{(1)}\!\left(kR_{pp}(t_i,\tau)\right)
- \frac{2i}{\pi}\ln|t_i-\tau|
\right]
\omega(\tau)\,d\tau
+
\frac{2i}{\pi}\int_{-1}^{1}\ln|t_i-\tau|\,\omega(\tau)\,d\tau.
```

For `|t_i| < 1`:

```math
\int_{-1}^{1}\ln|t_i-\tau|\,\omega(\tau)\,d\tau = -\pi\ln 2.
```

Stable self-term formula:

```math
M_{pp}(t_i)
= \int_{-1}^{1}
\left[
H_0^{(1)}\!\left(kR_{pp}(t_i,\tau)\right)
- \frac{2i}{\pi}\ln|t_i-\tau|
\right]
\omega(\tau)\,d\tau
- 2i\ln 2.
```

Right-hand-side constant:

```math
c_p = -\int_{-1}^{1} U_{p,0}(\tau)\,\omega(\tau)\,d\tau.
```

Recommended evaluation rule:

```math
\text{evaluate } c_p \text{ and } M_{pq}(t_i) \text{ with a high-accuracy weighted quadrature}.
```

## 11. Recover physical current

```math
j_q\!\big(s_q(t_i)\big) = \frac{u_{q,i}}{\gamma_q(t_i)\sqrt{1-t_i^2}}.
```

## 12. Near-field evaluation

Continuous scattered field:

```math
U_{\mathrm sc}(\mathbf r)
= \sum_{q=1}^{Q}
\int_{-1}^{1}
H_0^{(1)}\!\left(k\left|\mathbf r-\mathbf r_q(t)\right|\right)
v_q(t)\,\omega(t)\,dt.
```

Discrete scattered field:

```math
U_{\mathrm sc}^{(n)}(\mathbf r)
\approx
\frac{1}{n}
\sum_{q=1}^{Q}
\sum_{i=1}^{n}
H_0^{(1)}\!\left(k\left|\mathbf r-\mathbf r_q(t_i)\right|\right)
u_{q,i}.
```

Total field:

```math
U_{\mathrm tot}^{(n)}(\mathbf r) = U_0(\mathbf r) + U_{\mathrm sc}^{(n)}(\mathbf r).
```

Cartesian grid point:

```math
\mathbf r = (x,y).
```

Near-field intensity map:

```math
I(x,y) = \left|U_{\mathrm tot}^{(n)}(x,y)\right|^2.
```

Near-field amplitude map:

```math
A(x,y) = \left|U_{\mathrm tot}^{(n)}(x,y)\right|.
```

Near-field phase map:

```math
\phi(x,y) = \arg U_{\mathrm tot}^{(n)}(x,y).
```

Amplitude in dB:

```math
A_{\mathrm dB}(x,y) = 20\log_{10}A(x,y).
```

Intensity in dB:

```math
I_{\mathrm dB}(x,y) = 10\log_{10}I(x,y).
```

Normalized intensity in dB:

```math
I_{\mathrm dB,norm}(x,y)
= 10\log_{10}\frac{I(x,y)}{\max_{\mathbf r \in \text{grid}} I(x,y)}.
```

Phase for plotting:

```math
\phi(x,y) = \operatorname{atan2}\!\left(\Im U_{\mathrm tot}^{(n)}(x,y),\Re U_{\mathrm tot}^{(n)}(x,y)\right).
```

Plotting restriction:

```math
\mathbf r \notin L.
```

Plotting rule:

```math
\text{mask grid points on the reflector contour and sufficiently close to the contour}.
```

## 13. Far-field evaluation

Observation direction:

```math
\widehat{\mathbf s}(\varphi) = (\cos\varphi,\sin\varphi).
```

Total pattern:

```math
\Phi(\varphi) = \Phi_0(\varphi) + \Phi_{\mathrm sc}(\varphi).
```

Feed pattern:

```math
\Phi_0(\varphi) = e^{kb\cos(\varphi-\beta)}.
```

Scattered pattern in paper form:

```math
\Phi_{\mathrm sc}(\varphi)
\approx
\frac{1}{n}\left(\frac{2\pi}{i}\right)^{1/2}
\sum_{q=1}^{Q}
\sum_{i=1}^{n}
\exp\!\left[-ik\,r_q^{\mathrm pol}(t_i)\cos\!\left(\theta_q(t_i)-\varphi\right)\right]
u_{q,i}.
```

Polar-coordinate node definitions:

```math
r_q^{\mathrm pol}(t_i) = \sqrt{x_q(t_i)^2 + y_q(t_i)^2}.
```

```math
\theta_q(t_i) = \operatorname{atan2}(y_q(t_i),x_q(t_i)).
```

Equivalent Cartesian code form:

```math
\Phi_{\mathrm sc}(\varphi)
\approx
\frac{1}{n}\left(\frac{2\pi}{i}\right)^{1/2}
\sum_{q=1}^{Q}
\sum_{i=1}^{n}
\exp\!\left[-ik\,\widehat{\mathbf s}(\varphi)\cdot \mathbf r_q(t_i)\right]
u_{q,i}.
```

## 14. Radiated power and directivity

```math
P = \int_{0}^{2\pi} |\Phi(\varphi)|^2\,d\varphi.
```

```math
\varphi_{\max} = \arg\max_{\varphi \in [0,2\pi)} |\Phi(\varphi)|.
```

```math
D = \frac{2\pi}{P}\,|\Phi(\varphi_{\max})|^2.
```

Discrete power integral:

```math
\varphi_m = \frac{2\pi m}{N_\varphi}.
```

```math
\Delta\varphi = \frac{2\pi}{N_\varphi}.
```

```math
P^{(N_\varphi)} \approx \Delta\varphi \sum_{m=0}^{N_\varphi-1} |\Phi(\varphi_m)|^2.
```

```math
D^{(N_\varphi)} \approx \frac{2\pi}{P^{(N_\varphi)}} \max_m |\Phi(\varphi_m)|^2.
```

## 15. Edge illumination

Reflector center point:

```math
\mathbf r_{\mathrm center} = \mathbf r_q(0).
```

Edge points:

```math
\mathbf r_{\mathrm edge},1 = \mathbf r_q(-1).
```

```math
\mathbf r_{\mathrm edge},2 = \mathbf r_q(1).
```

Single-edge illumination:

```math
E_{\mathrm edge},1 = 20\log_{10}\left|\frac{U_0(\mathbf r_{\mathrm edge},1)}{U_0(\mathbf r_{\mathrm center})}\right|.
```

```math
E_{\mathrm edge},2 = 20\log_{10}\left|\frac{U_0(\mathbf r_{\mathrm edge},2)}{U_0(\mathbf r_{\mathrm center})}\right|.
```

Symmetric reflector case:

```math
E_{\mathrm edge} = E_{\mathrm edge},1 = E_{\mathrm edge},2.
```

## 16. Numerical sanity checks

Boundary-condition residual:

```math
\varepsilon_{\mathrm bc}
= \max_{p,j}\left|
\frac{1}{n}\sum_{q=1}^{Q}\sum_{i=1}^{n}
H_0^{(1)}\!\left(kR_{pq}(t_i,\tau_j)\right)u_{q,i}
+
U_{p,0}(\tau_j)
\right|.
```

Current-convergence check:

```math
\varepsilon_{\mathrm cur}(n,2n)
= \max_{q,i}\left|u_{q,i}^{(2n)} - u_{q,i}^{(n)}\right|.
```

Pattern-convergence check:

```math
\varepsilon_{\Phi}(n,2n)
= \max_{\varphi} \left|\Phi^{(2n)}(\varphi) - \Phi^{(n)}(\varphi)\right|.
```

Physical power check:

```math
P > 0.
```

Near-field plotting check:

```math
\min_{q,i} \left|\mathbf r_{\mathrm grid} - \mathbf r_q(t_i)\right| > 0.
```

## 17. Implementation order

```math
\text{Step 1: define } x_q(t), y_q(t), x_q'(t), y_q'(t).
```

```math
\text{Step 2: define } U_0(\mathbf r), U_{p,0}(\tau), f_p(\tau), c_p.
```

```math
\text{Step 3: build nodes } t_i, \tau_j.
```

```math
\text{Step 4: evaluate } K_{pq}(t_i,\tau_j) \text{ and } M_{pq}(t_i) \text{ with singularity subtraction for } p=q.
```

```math
\text{Step 5: assemble } A \text{ and } b.
```

```math
\text{Step 6: solve } A\mathbf u = \mathbf b.
```

```math
\text{Step 7: evaluate } U_{\mathrm tot}^{(n)}(x,y) \text{ on a grid for near-field plots}.
```

```math
\text{Step 8: evaluate } \Phi(\varphi) \text{ and } D.
```

```math
\text{Step 9: verify } \varepsilon_{\mathrm bc}, \varepsilon_{\mathrm cur}, \varepsilon_{\Phi}.
```

## 18. Notes for near-field plotting

```math
\text{Use a rectangular grid } (x_a,y_b).
```

```math
\text{For intensity plots, use } I(x_a,y_b).
```

```math
\text{For phase plots, use } \phi(x_a,y_b).
```

```math
\text{For normalized maps, subtract the maximum before converting to dB}.
```

```math
\text{Do not sample exactly on the reflector contour}.
```

```math
\text{Mask a thin tube around each } L_q \text{ if needed for clean plots}.
```
