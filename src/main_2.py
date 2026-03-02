"""
Docstring for src.main_2

CSP: Complex Source Point

"""

import numpy as np

from scipy.integrate import quad
from scipy.differentiate import derivative

import matplotlib.pyplot as plt

from .models.parametrized_curve import ArcSegment, ParamCurve, LineSegment
from .utils.math_utils import hankel1, cheb_first_kind, cheb_second_kind_interior


class Solution:
    param_curve: ParamCurve
    """Parametrized curve"""

    k: int
    """Wave number. AKA k"""

    r_cs: np.ndarray
    """Complex source point (CSP) coordinates (complex)"""

    csp_feed_length: float
    """Complex source point (CSP) feed length. AKA wave length. AKA b"""

    csp_feed_length_half: float
    """Half of complex source point (CSP) feed length"""

    kb: float
    """k * csp_feed_length_half"""

    n: int
    """Number of points to calculate (higher number - higher precision and more computation power needed)"""

    csp_feed_angle: float
    """Angle in degrees of Complex source point (CSP)"""

    def __init__(
        self,
        param_curve: ParamCurve,
        csp_feed_length: float = 0.3,
        kb: float = 2.6,
        csp_feed_angle: float = -90.0,
        n: int = 50,
    ):
        self.param_curve = param_curve
        self.csp_feed_length = csp_feed_length
        self.csp_feed_length_half = csp_feed_length / 2
        self.kb = kb
        self.csp_feed_angle = csp_feed_angle
        self.n = n

        # wave_length = 1 / (np.pi) # lambda
        # wave_number = np.pi * 2 / wave_length # k

        wave_number = kb / self.csp_feed_length_half  # k
        self.k = wave_number
        wave_length = np.pi * 2 / wave_number  # lambda

        csp_feed_angle_radians = (
            csp_feed_angle * np.pi / 180
        )  # Beta, radians = degree * pi / 180
        csp_feed_direction_b = np.array(
            [
                self.csp_feed_length_half * np.cos(csp_feed_angle_radians),
                self.csp_feed_length_half * np.sin(csp_feed_angle_radians),
            ]
        )  # vector b

        r_0 = np.array([0.0, 0.0])
        self.r_cs = r_0 + 1j * csp_feed_direction_b

    def solve(
        self,
        x_min: float = -10.0,
        x_max: float = 10.0,
        y_min: float = -10.0,
        y_max: float = 10.0,
    ):
        xs = np.linspace(x_min, x_max, 200)
        ys = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(xs, ys)

        # Calculate incident field
        U0 = self.csp_incident_field(X, Y)
        U0_abs = np.abs(U0) ** 2

        U = self.csp_secondary_field(X, Y)
        U_abs = np.abs(U) ** 2

        # Z = U0_abs + U_abs  # Full solution
        Z = np.abs(U0 + U) ** 2  # Full solution
        # Z = np.abs(U0 + U)  # Full solution
        # Z = U0_abs  # Incident
        # Z = U_abs # Dispersion

        q = np.abs(U0 + U)
        # q = np.abs(U0)

        Z = 20 * np.log10(q / np.max(q))

        return Z

    def csp_incident_field(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        y = np.asarray(y, dtype=np.complex128)

        R = np.sqrt((x - self.r_cs[0]) ** 2 + (y - self.r_cs[1]) ** 2)
        return hankel1(0, self.k * R)

    def csp_incident_field_from_t(self, t: np.ndarray) -> np.ndarray:
        x = np.asarray(self.param_curve.x(t), dtype=np.complex128)
        y = np.asarray(self.param_curve.y(t), dtype=np.complex128)

        R = np.sqrt((x - self.r_cs[0]) ** 2 + (y - self.r_cs[1]) ** 2)
        return hankel1(0, self.k * R)

    # Sure?
    def csp_incident_field_parametrized_t_arr_derivative(
        self, t: np.ndarray
    ) -> np.ndarray:
        x = self.param_curve.x(t)
        y = self.param_curve.y(t)
        R = np.sqrt((x - self.r_cs[0]) ** 2 + (y - self.r_cs[1]) ** 2)

        result = (
            -self.k
            * hankel1(1, self.k * R)
            * self.param_curve.R_t0_der_coord(t, self.r_cs)
        )

        return result

    # def constant_c(self) -> float:
    #     result, err = quad(
    #         lambda t: self.csp_incident_field_from_t(t) / np.sqrt(1 - t**2),
    #         -1,
    #         1,
    #         complex_func=True,
    #         limit=200,
    #     )
    #     return -result

    def constant_c(self, t: np.ndarray) -> float:
        result = np.pi * np.sum(self.csp_incident_field_from_t(t)) / self.n
        return -result

    # def M_t(self, t: float) -> float:
    #     result, err = quad(
    #         lambda t_0: hankel1(0, self.k * self.param_curve.R(t, t_0))
    #         / np.sqrt(1 - t_0**2),
    #         -1,
    #         1,
    #         complex_func=True,
    #         limit=200,
    #     )
    #     return result
    
    def M_t(self, t: np.ndarray, ti: float) -> float:
        result = 0
        for i in range(t.shape[0]):
            if ti == t[i]:
                continue
            result += hankel1(0, self.k * self.param_curve.R(ti, t[i]))
        return np.pi * result / self.n

    def K_t_t0(
        self, t: float | np.ndarray, t_0: float | np.ndarray
    ) -> float | np.ndarray:
        result = -1 / (t - t_0) - self.k * hankel1(
            1, self.k * self.param_curve.R(t, t_0)
        ) * self.param_curve.R_t0_der(t, t_0)
        return result

    def f_t_0_arr(self, t_0: np.ndarray):
        result = -self.csp_incident_field_parametrized_t_arr_derivative(t_0)
        return result

    def csp_secondary_field(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        y = np.asarray(y, dtype=np.complex128)

        t = cheb_first_kind(self.n)
        self.t = t
        t_0j = cheb_second_kind_interior(self.n)
        self.t_0j = t_0j

        # Ax = b
        A = np.zeros((self.n, self.n), dtype=np.complex128)
        b = np.zeros((self.n,), dtype=np.complex128)
        f_vector = self.f_t_0_arr(t_0j)

        for j in range(self.n - 1):
            for i in range(self.n):
                A[j, i] = (1 / (t[i] - t_0j[j]) + self.K_t_t0(t[i], t_0j[j])) / self.n
            b[j] = f_vector[j]

        for i in range(self.n):
            tmp = self.M_t(t, t[i]) / self.n
            A[self.n - 1, i] = tmp

        b[self.n - 1] = self.constant_c(t)

        v = np.linalg.solve(A, b)
        self.v = v

        x_t = self.param_curve.x(t).astype(np.complex128, copy=False)
        y_t = self.param_curve.y(t).astype(np.complex128, copy=False)
        dx = x_t[:, None, None] - x[None, :, :]
        dy = y_t[:, None, None] - y[None, :, :]
        R = np.sqrt(dx**2 + dy**2)
        H = hankel1(0, self.k * R)
        U = (np.pi / self.n) * np.sum(H * v[:, None, None], axis=0)

        # return U
        return -1j / 4 * U 
        # return U


# Visualize intensity

if __name__ == "__main__":
    x_min, x_max, y_min, y_max = -100.0, 100.0, -100.0, 100.0

    sol = Solution(LineSegment(-100, 0, 0, -100), csp_feed_angle=180, n=100)
    # sol = Solution(ArcSegment(-75, -75, -50, 50), csp_feed_angle=180)
    Z = sol.solve()

    plt.figure(figsize=(7, 5.5))
    plt.imshow(
        Z,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="equal",
        cmap="jet",
    )
    plt.clim(-30, 0)
    plt.colorbar(label=r"$|U(x,y)|^2$")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("CSP Incident Field Intensity Heatmap")
    plt.tight_layout()
    plt.show()
