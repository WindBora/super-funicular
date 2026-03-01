# mds_strip_scatter.py
# E-polarized plane-wave (or CSP-beam) scattering by a finite PEC sinusoidal strip
# Discrete-singularities (Nyström) scheme on Chebyshev nodes (I/II kinds)
# Based on the course model: Cauchy-singular IE + extra condition; kernels K, M; ν-unknown
# (Rusynyk coursework; Nosich–Gandel CSP option)
# Requires: numpy, scipy (special), optional matplotlib for plots

from dataclasses import dataclass
import numpy as np
from scipy.special import jv, yv
from scipy.integrate import quad, quad_vec
from numpy.linalg import solve
import matplotlib.pyplot as plt

# ---------------------------
# Geometry: sinusoidal strip S
# Parameter t in [-1,1] maps to x,y
# ---------------------------


@dataclass
class StripGeometry:
    L: float  # projection length along x (meters)
    A: float  # sinusoid amplitude (meters): y = A*sin(pi*x/L)
    b: float
    # parameterization: x(t)= (L/2)*t, y(t)= A*sin(pi*t)

    def r(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.L * t / 2
        y = self.A * np.sin(np.pi * t * self.b)
        return x, y

    def rp(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # derivative wrt t
        dx = self.L / 2
        dy = self.b * self.A * np.pi * np.cos(np.pi * t * self.b)
        return dx, dy


# ---------------------------
# Incident fields U0 and dU0/dt
# ---------------------------


@dataclass
class PlaneWave:
    k: float  # wavenumber 2π/λ
    phi: float  # incidence angle (rad), k-vector = (cos phi, sin phi)

    def U0(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(1j * self.k * (x * np.cos(self.phi) + y * np.sin(self.phi)))

    def U0_t(self, t: np.ndarray) -> np.ndarray:
        x_t, y_t = geom.r(t)
        return self.U0(x_t, y_t)

    def dU0_dt(
        self, x: np.ndarray, y: np.ndarray, dxdt: np.ndarray, dydt: np.ndarray
    ) -> np.ndarray:
        U = self.U0(x, y)
        return 1j * self.k * U * (np.cos(self.phi) * dxdt + np.sin(self.phi) * dydt)


@dataclass
class CSPBeam:
    k: float
    x0: complex
    y0: complex
    # Complex-source point: U0 = H0^(1)(k*|r - rc|) ; optional orientation via complex x0,y0

    def U0(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)
        return hankel1(0, self.k * R)

    def dU0_dt(
        self, x: np.ndarray, y: np.ndarray, dxdt: np.ndarray, dydt: np.ndarray
    ) -> np.ndarray:
        # d/dt H0(kR) = -k H1(kR) * dR/dt ; with R = sqrt(...)
        Rx = x - self.x0
        Ry = y - self.y0
        R = np.sqrt(Rx**2 + Ry**2)
        # avoid division by zero at the CSP location
        dRdt = (Rx * dxdt + Ry * dydt) / (R + 0j)
        return -self.k * hankel1(1, self.k * R) * dRdt


# ---------------------------
# Chebyshev nodes (Nyström)
# ---------------------------


def cheb_first_kind(n: int):
    # ti = cos((2i-1)π/(2n)), i=1..n
    i = np.arange(1, n + 1)
    return np.cos((2 * i - 1) * np.pi / (2 * n))


def cheb_second_kind_interior(n: int):
    # t0_j = cos(jπ/n), j=1..n-1  (exclude ±1)
    j = np.arange(1, n)
    return np.cos(j * np.pi / n)


def hankel1(v: np.ndarray, z: np.ndarray) -> np.ndarray:
    return jv(v, z) + yv(v, z) * 1j


# ---------------------------
# Kernels K(t, t0) and M(t)
# ---------------------------
# Course formulas (single PEC strip, E-pol):
# K(t, t0) = -H1^{(1)}(k R(t,t0)) - 1/(t - t0)
# M(t) = ∫_{-1}^1 H0^{(1)}(k R(t, τ)) dτ / sqrt(1-τ^2)
# Discretize M(t) with Gauss–Chebyshev: (π/n) sum_{i=1..n} H0(k R(t, ti))
# Right-hand: f(t0) = - dU0/dt0  ;  c = ∫ U0(τ) dτ / sqrt(1-τ^2) ≈ (π/n) Σ U0(ti)
# Unknown ν(t); j(t) = ν(t)*sqrt(x'(t)^2+y'(t)^2)/sqrt(1-t^2)


def R_of_t(geom: StripGeometry, t: np.ndarray, tp: np.ndarray) -> np.ndarray:
    x, y = geom.r(t)
    xp, yp = geom.r(tp)
    return np.sqrt((x - xp) ** 2 + (y - yp) ** 2)


def K_kernel(
    k: float, geom: StripGeometry, t: np.ndarray, t0: np.ndarray
) -> np.ndarray:
    # safe for vectorized inputs
    R = R_of_t(geom, t, t0)
    return -1 * hankel1(1, k * R) - 1.0 / (t - t0)


def M_kernel_discrete(k: float, geom: StripGeometry, ti: np.ndarray):
    # returns M(ti) approximated by Gauss–Chebyshev
    # ∫ f(τ)/sqrt(1-τ^2) dτ ≈ (π/n) Σ f(t_nodes)
    def x(q: np.ndarray) -> np.ndarray:
        return hankel1(0, k * R_of_t(geom, ti, q)) / np.sqrt(1 - q**2)

    M_t, err = quad(x, -1, 1, complex_func=True, limit=1000)
    # print("M_t err", err)
    return M_t


def sol(lam, k, geom, inc):
    n = 10

    # Geometry and frequency
    ti = cheb_first_kind(n)  # I-kind nodes for ν
    t0j = cheb_second_kind_interior(n)  # II-kind interior nodes for Cauchy rows

    x_t0, y_t0 = geom.r(t0j)
    dx_t0, dy_t0 = geom.rp(t0j)

    b = -inc.dU0_dt(x_t0, y_t0, dx_t0, dy_t0)
    c, err = quad(
        lambda t: inc.U0_t(t) / np.sqrt(1 - t**2), -1, 1, complex_func=True, limit=1000
    )
    b = np.concat([b, np.array([c])])
    # print("c_err", err)

    A = np.zeros((n, n), dtype=complex)

    for j in range(n - 1):
        for i in range(n):
            A[i, j] = (-1 / (ti[i] - t0j[j]) + K_kernel(k, geom, ti[i], t0j[j])) / n

    for i in range(n):
        tmp = M_kernel_discrete(k, geom, ti[i])
        A[n - 1, i] = tmp

    # print(A)

    # print(A.shape)
    # print(b.shape)
    v = np.matmul(np.linalg.inv(A), np.matrix.transpose(b))
    # print(v)

    Uq = [np.sin((2 * q - 1) * np.pi / 2) / np.sin((2 * q - 1) * np.pi / 2 / n) for q in range(1, n+1)]

    def v_func(t: np.ndarray) -> np.ndarray:
        s = 0
        # T = np.cos((2 * i - 1) * np.pi / (2 * n))

        T_prev_prev = 1
        T_prev = t
        for _ in range(1, n):
            T = 2 * t * T_prev - T_prev_prev
            T_prev, T_prev_prev = T, T_prev

        for q in range(1, n + 1):
            # cos(sigma) -> sin((n+1)sigma)/sin(sigma)
            # U = np.sin((2 * q - 1) * np.pi / 2) / np.sin((2 * q - 1) * np.pi / 2 / n)
            s += v[q - 1] / (t - ti[q - 1]) / Uq[q - 1]

        return s * T / n

    return v_func


if __name__ == "__main__":
    # Geometry and frequency
    lam = 0.01  # 1 cm wavelength
    k = 2 * np.pi / lam
    geom = StripGeometry(L=5 * lam, A=0.05 * lam, b=3)  # fairly smooth sinusoid
    inc = PlaneWave(k, np.deg2rad(30))
    v_fun = sol(lam, k, geom, inc)

    x, y = geom.r(np.linspace(-1, 1, 100))

    plt.plot(x, y)
    plt.show()

    def j_fun(t: np.ndarray):
        dx, dy = geom.rp(t)
        result = v_fun(t) / np.sqrt(1 - t**2) / np.sqrt(dx**2 + dy**2)
        return result

    def U_sc(x: np.ndarray, y: np.ndarray):
        def R_of_x_y_t(t: np.ndarray):
            Rx, Ry = geom.r(t)
            return np.sqrt((x - Rx) ** 2 + (y - Ry) ** 2)

        f = lambda t: hankel1(0, k * R_of_x_y_t(t)) * v_fun(t) / np.sqrt(1 - t**2)

        val, err = quad(f, -1, 1, complex_func=True, limit=50)
        # print("U_sc_error_val", err)
        result = 1.0j / 4.0 * val

        return result

    nx = 50
    ny = 50
    x, y = np.meshgrid(np.linspace(-0.03, 0.03, nx), np.linspace(-0.001, 0.001, ny))
    z = np.zeros((nx, ny), complex)
    for i in range(nx):
        for j in range(ny):
            usc = U_sc(x[j, i], y[i, j])
            u0 = inc.U0(x[j, i], y[i, j])
            z[j, i] = usc + u0

    # print(x)
    # print(y)
    # print(z)
    # z = np.array(z, dtype=complex)

    # plt.subplot(1, 2, 1)
    # plt.contourf(x, y, z.real, levels=50, cmap="viridis")

    # plt.subplot(1, 2, 2)
    # plt.contourf(x, y, z.imag, levels=50, cmap="plasma")

    # plt.subplot(2, 2, 3)
    plt.pcolormesh(x, y, np.abs(z))
    # plt.contourf(x, y, np.abs(z), levels=50, cmap="plasma")
    plt.show()

    # plt.pcolormesh(x, y, z.real)
    # plt.show()

    # plt.pcolormesh(x, y, z.imag)
    # plt.show()

    # Solve with plane wave at 30°
    # sol = solve_strip_e_wave(
    #     geom=geom,
    #     k=k,
    #     n=64,
    #     incident="plane",
    #     inc_params={"phi": np.deg2rad(0.0)}
    # )

    # print(sol.nu_at_t)
    # print(sol.j_at_t)

    # print("Solved Nystrom system with n =", sol.n)
    # print("||ν||_2 =", np.linalg.norm(sol.nu_at_t))
    # print("||j||_2  =", np.linalg.norm(sol.j_at_t))

    # # Optional quick plot of |j|
    # try:
    #     import matplotlib.pyplot as plt
    #     plt.plot(sol.t_nodes, np.abs(sol.j_at_t))
    #     plt.xlabel("t")
    #     plt.ylabel("|j(t)|")
    #     plt.title("Surface current magnitude on the strip")
    #     # plt.show()
    # except Exception:
    #     pass
