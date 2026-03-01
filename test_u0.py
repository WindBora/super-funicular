import numpy as np
from scipy.special import jv, yv

import matplotlib.pyplot as plt


def csp_incident_field(x, y, k, rcs):
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)

    R = np.sqrt((x - rcs[0])**2 + (y - rcs[1])**2)
    return hankel1(1, k * R)

def hankel1(v: np.ndarray, z: np.ndarray) -> np.ndarray:
    return jv(v, z) + yv(v, z) * 1j

# def hankel1(z: np.ndarray) -> np.ndarray:
#     return j0(z) + y0(z) * 1j


csp_feed_length = 0.001 # wave_lenght # b, CSP
csp_feed_length_half = csp_feed_length / 2


kb = 9

# wave_length = 1 / (np.pi) # lambda
# wave_number = np.pi * 2 / wave_length # k

wave_number = kb / csp_feed_length_half # k
wave_length = np.pi * 2 / wave_number # lambda

print(wave_number, wave_length)

csp_feed_angle_radians = 30 * np.pi / 180 # Beta, radians = degree * pi / 180
csp_feed_direction_b = np.array([csp_feed_length_half * np.cos(csp_feed_angle_radians), csp_feed_length_half * np.sin(csp_feed_angle_radians)]) # vector b

r0 = np.array([-3., 0.])
rcs = r0 + 1j * csp_feed_direction_b

xs = np.linspace(-100., 100., 500)
ys = np.linspace(-100., 100., 500)
X, Y = np.meshgrid(xs, ys)


U0 = csp_incident_field(X, Y, wave_number, rcs)

# Visualize intensity
Z = np.abs(U0)**2

plt.figure(figsize=(7, 5.5))
plt.imshow(
    Z,
    extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    origin="lower",
    aspect="equal",
    cmap="jet"
)
plt.colorbar(label=r"$|U_0(x,y)|^2$")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("CSP Incident Field Intensity Heatmap")
plt.tight_layout()
plt.show()