import numpy as np
from scipy import special

class Bessel:
    @staticmethod
    def Jn_integral_complex(n: int, x, num_points: int = 4001):
        """
        Numerical Bessel J_n(x) using complex integral:

            J_n(x) = (1/pi) * Re ∫_0^pi exp(i*(n*t - x*sin(t))) dt

        Composite Simpson's rule on [0, pi].

        Parameters
        ----------
        n : int
        x : float, complex, or array-like
        num_points : int
            Number of grid points (odd required for Simpson)

        Returns
        -------
        J_n(x) as float (if x real scalar) or complex/ndarray
        """
        if not isinstance(n, (int, np.integer)):
            raise TypeError("n must be an integer.")
        if num_points < 3:
            raise ValueError("num_points must be >= 3.")
        if num_points % 2 == 0:
            num_points += 1

        x_arr = np.asarray(x, dtype=np.complex128)

        t = np.linspace(0.0, np.pi, num_points)
        h = t[1] - t[0]

        phase = n * t - x_arr[..., None] * np.sin(t)
        f = np.exp(1j * phase)

        # Simpson's rule
        s = f[..., 0] + f[..., -1]
        s += 4.0 * np.sum(f[..., 1:-1:2], axis=-1)
        s += 2.0 * np.sum(f[..., 2:-1:2], axis=-1)

        integral = (h / 3.0) * s
        J = integral / np.pi

        # By definition J_n(x) is the real part for real x,
        # but we return the complex value in general:
        if np.isscalar(x):
            return J.item()
        return J


def compare_accuracy_complex(ns=(0, 1, 2, 5), x_min=1e-6, x_max=40.0, num_x=600,
                             num_points=16001):
    """
    Compare complex-integral Jn vs SciPy:
      - scipy.special.jv (true J_n)
      - scipy.special.jv + 1j*scipy.special.yv = H_n^(1)

    For real x:
      Re(H_n^(1)(x)) = J_n(x)

    yv is singular at x=0, so x_min must be > 0.
    """

    xs = np.linspace(x_min, x_max, num_x)

    results = []
    for n in ns:
        J_int = Bessel.Jn_integral_complex(n, xs, num_points=num_points)
        J_int_real = np.real(J_int)

        J_scipy = special.jv(n, xs)
        H1 = special.jv(n, xs) + 1j * special.yv(n, xs)
        ReH1 = np.real(H1)

        abs_err_jv = np.max(np.abs(J_int_real - J_scipy))
        rel_err_jv = np.max(np.abs(J_int_real - J_scipy) / (np.abs(J_scipy) + 1e-300))

        abs_err_ReH1 = np.max(np.abs(J_int_real - ReH1))
        rel_err_ReH1 = np.max(np.abs(J_int_real - ReH1) / (np.abs(ReH1) + 1e-300))

        diff_ReH1_jv = np.max(np.abs(ReH1 - J_scipy))

        results.append({
            "n": int(n),
            "num_points": int(num_points),
            "x_range": (float(x_min), float(x_max)),
            "max_abs_err_vs_jv": float(abs_err_jv),
            "max_rel_err_vs_jv": float(rel_err_jv),
            "max_abs_err_vs_Re(H1)": float(abs_err_ReH1),
            "max_rel_err_vs_Re(H1)": float(rel_err_ReH1),
            "max_abs(Re(H1)-jv)": float(diff_ReH1_jv),
        })

    return results


if __name__ == "__main__":
    res = compare_accuracy_complex(
        ns=(0, 1, 2, 5, 10),
        x_min=1e-6,
        x_max=40.0,
        num_x=800,
        num_points=20001
    )

    for r in res:
        print(
            f"n={r['n']:2d} | "
            f"max|int-jv|={r['max_abs_err_vs_jv']:.3e}, "
            f"max rel={r['max_rel_err_vs_jv']:.3e} | "
            f"max|int-Re(H1)|={r['max_abs_err_vs_Re(H1)']:.3e}, "
            f"max rel={r['max_rel_err_vs_Re(H1)']:.3e} | "
            f"max|Re(H1)-jv|={r['max_abs(Re(H1)-jv)']:.3e}"
        )