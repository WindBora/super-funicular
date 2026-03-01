import numpy as np
from scipy.special import jv, yv


def hankel1(v: np.ndarray, z: np.ndarray) -> np.ndarray:
    return jv(v, z) + yv(v, z) * 1j


def cheb_first_kind(n: int):
    # ti = cos((2i-1)π/(2n)), i=1..n
    i = np.arange(1, n + 1)
    return np.cos((2 * i - 1) * np.pi / (2 * n))


def cheb_second_kind_interior(n: int):
    # t0_j = cos(jπ/n), j=1..n-1  (including +-1)
    j = np.arange(1, n)
    return np.cos(j * np.pi / n)
