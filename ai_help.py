import numpy as np
from scipy.special import hankel1

def csp_incident_field(x, y, k, x0, y0, b, beta):
    """
    Complex-Source-Point (CSP) incident field:
        U0(r) = H_0^(1)( k * |r - rc| )
    where
        rc = r0 + i*bvec,
        r0 = (x0, y0),
        bvec = (b*cos(beta), b*sin(beta)).

    Parameters
    ----------
    x, y : array_like
        Observation coordinates (can be scalars or numpy arrays / meshgrids).
    k : float
        Wavenumber (2*pi / lambda).
    x0, y0 : float
        Real part of CSP location (feed center).
    b : float
        Imaginary shift magnitude (controls beam width).
    beta : float
        Direction of imaginary shift in radians.

    Returns
    -------
    U0 : complex ndarray
        Incident field evaluated at (x, y).
    """
    # Convert inputs to arrays for broadcasting
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)

    # Complex source point rc = (x0, y0) + i*(bcosβ, bsinβ)
    rcx = x0 + 1j * (b * np.cos(beta))
    rcy = y0 + 1j * (b * np.sin(beta))

    # Complex distance R = sqrt((x-rcx)^2 + (y-rcy)^2)
    R = np.sqrt((x - rcx)**2 + (y - rcy)**2)

    # U0 = Hankel1(order=0, k*R)
    U0 = hankel1(0, k * R)
    return U0

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Physical parameters
    lam = 0.03               # wavelength [m] (example)
    k = 2 * np.pi / lam

    # CSP parameters (example)
    x0, y0 = 0.0, 0.0         # real CSP location
    b = 0.5 * lam             # imaginary shift magnitude
    beta = 0.0                # shift direction (radians)

    # Observation grid
    xs = np.linspace(-0.3, 0.3, 401)
    ys = np.linspace(-0.3, 0.3, 401)
    X, Y = np.meshgrid(xs, ys)

    U0 = csp_incident_field(X, Y, k, x0, y0, b, beta)

    # Example: intensity map (near-field magnitude squared)
    I0 = np.abs(U0)**2
    print(U0.shape, I0.min(), I0.max())