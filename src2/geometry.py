"""Curve primitives and geometry helpers for reflector configurations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .numerics import as_float_array, broadcast_constant_like


class ParamCurve(ABC):
    """Abstract parametric curve used by the boundary-integral solver.

    Every reflector is parameterized by ``t in [-1, 1]``. Implementations must
    provide curve coordinates and first derivatives with respect to ``t``.
    """

    @abstractmethod
    def x(self, t: float | np.ndarray) -> np.ndarray:
        """Return the x-coordinate at parameter values ``t``."""

    @abstractmethod
    def y(self, t: float | np.ndarray) -> np.ndarray:
        """Return the y-coordinate at parameter values ``t``."""

    @abstractmethod
    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        """Return ``dx/dt`` evaluated at parameter values ``t``."""

    @abstractmethod
    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        """Return ``dy/dt`` evaluated at parameter values ``t``."""

    def coords(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x(t), y(t))`` for the given parameter values."""

        return self.x(t), self.y(t)

    def derivatives(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(dx/dt, dy/dt)`` for the given parameter values."""

        return self.x_der(t), self.y_der(t)

    def speed(self, t: float | np.ndarray) -> np.ndarray:
        """Return the metric factor ``sqrt((dx/dt)^2 + (dy/dt)^2)``."""

        dx, dy = self.derivatives(t)
        return np.sqrt(dx * dx + dy * dy)


@dataclass(frozen=True)
class LineSegment(ParamCurve):
    """Straight segment between two endpoints.

    Parameters
    ----------
    x1, y1:
        Coordinates of the endpoint corresponding to ``t=-1``.
    x2, y2:
        Coordinates of the endpoint corresponding to ``t=1``.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def x(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = as_float_array(t)
        s = 0.5 * (t_arr + 1.0)
        return self.x1 + s * (self.x2 - self.x1)

    def y(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = as_float_array(t)
        s = 0.5 * (t_arr + 1.0)
        return self.y1 + s * (self.y2 - self.y1)

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        return broadcast_constant_like(t, 0.5 * (self.x2 - self.x1))

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return broadcast_constant_like(t, 0.5 * (self.y2 - self.y1))


@dataclass(frozen=True)
class SinusoidalStrip(ParamCurve):
    """Open sinusoidal strip parameterized along the global x-axis.

    The strip is defined by

    ``x(t) = x_center + (length / 2) * t``

    ``y(t) = y_base + amplitude * sin(2π * frequency * (x(t) - x_center) + phase)``

    where ``t in [-1, 1]``.

    Parameters
    ----------
    x_center:
        Midpoint x-coordinate of the strip.
    y_base:
        Baseline y-coordinate about which the strip oscillates.
    length:
        Total horizontal span of the strip.
    amplitude:
        Vertical oscillation amplitude.
    frequency:
        Spatial frequency in cycles per wavelength unit. The total number of
        periods along the strip equals ``frequency * length``.
    phase_rad:
        Phase offset of the sinusoid in radians.
    """

    x_center: float
    y_base: float
    length: float
    amplitude: float
    frequency: float
    phase_rad: float = 0.0

    def x(self, t: float | np.ndarray) -> np.ndarray:
        """Return the strip x-coordinate for parameter values ``t``."""

        t_arr = as_float_array(t)
        return self.x_center + 0.5 * self.length * t_arr

    def y(self, t: float | np.ndarray) -> np.ndarray:
        """Return the strip y-coordinate for parameter values ``t``."""

        x = self.x(t)
        arg = 2.0 * np.pi * self.frequency * (x - self.x_center) + self.phase_rad
        return self.y_base + self.amplitude * np.sin(arg)

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        """Return the derivative ``dx/dt`` for parameter values ``t``."""

        return broadcast_constant_like(t, 0.5 * self.length)

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        """Return the derivative ``dy/dt`` for parameter values ``t``."""

        x = self.x(t)
        dx_dt = self.x_der(t)
        arg = 2.0 * np.pi * self.frequency * (x - self.x_center) + self.phase_rad
        return self.amplitude * np.cos(arg) * (2.0 * np.pi * self.frequency) * dx_dt


@dataclass(frozen=True)
class ParabolaArc(ParamCurve):
    """Parabolic reflector arc of the form ``x = y^2 / (4f)``.

    Parameters
    ----------
    f:
        Focal length of the underlying parabola.
    y1, y2:
        Lower and upper truncation values of the ``y`` coordinate.
    """

    f: float
    y1: float
    y2: float

    @classmethod
    def symmetric(cls, f: float, aperture: float) -> "ParabolaArc":
        """Build a parabola centered about ``y=0`` with the given aperture."""

        half = 0.5 * aperture
        return cls(f=f, y1=-half, y2=half)

    def y(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = as_float_array(t)
        mid = 0.5 * (self.y1 + self.y2)
        half_span = 0.5 * (self.y2 - self.y1)
        return mid + half_span * t_arr

    def x(self, t: float | np.ndarray) -> np.ndarray:
        y = self.y(t)
        return (y * y) / (4.0 * self.f)

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return broadcast_constant_like(t, 0.5 * (self.y2 - self.y1))

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        y = self.y(t)
        return y * self.y_der(t) / (2.0 * self.f)


@dataclass(frozen=True)
class HyperbolaArc(ParamCurve):
    """Hyperbolic reflector arc parameterized by ``(x/a)^2 - (y/b)^2 = 1``.

    Parameters
    ----------
    a, b:
        Semi-axis lengths of the hyperbola.
    center_x:
        x-coordinate of the hyperbola center.
    y1, y2:
        Lower and upper truncation values of the ``y`` coordinate.
    branch:
        Hyperbola branch to use, either ``"left"`` or ``"right"``.
    """

    a: float
    b: float
    center_x: float
    y1: float
    y2: float
    branch: str = "left"

    @classmethod
    def confocal(
        cls,
        focus_left_x: float,
        focus_right_x: float,
        vertex_x: float,
        y1: float,
        y2: float,
        branch: str = "left",
    ) -> "HyperbolaArc":
        """Build a confocal hyperbola from the two focus positions.

        Parameters
        ----------
        focus_left_x, focus_right_x:
            x-coordinates of the two foci on the optical axis.
        vertex_x:
            x-coordinate of the chosen branch vertex.
        y1, y2:
            Lower and upper truncation values along ``y``.
        branch:
            Branch to keep, either ``"left"`` or ``"right"``.
        """

        if focus_right_x <= focus_left_x:
            raise ValueError("focus_right_x must be greater than focus_left_x")
        center_x = 0.5 * (focus_left_x + focus_right_x)
        c = 0.5 * (focus_right_x - focus_left_x)
        if branch == "left":
            a = center_x - vertex_x
        elif branch == "right":
            a = vertex_x - center_x
        else:
            raise ValueError("branch must be 'left' or 'right'")
        if a <= 0.0 or a >= c:
            raise ValueError("vertex_x must define 0 < a < c for a confocal hyperbola")
        b = np.sqrt(c * c - a * a)
        return cls(a=a, b=float(b), center_x=center_x, y1=y1, y2=y2, branch=branch)

    def _branch_sign(self) -> float:
        return -1.0 if self.branch == "left" else 1.0

    def y(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = as_float_array(t)
        mid = 0.5 * (self.y1 + self.y2)
        half_span = 0.5 * (self.y2 - self.y1)
        return mid + half_span * t_arr

    def x(self, t: float | np.ndarray) -> np.ndarray:
        y = self.y(t)
        root = np.sqrt(1.0 + (y / self.b) ** 2)
        return self.center_x + self._branch_sign() * self.a * root

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return broadcast_constant_like(t, 0.5 * (self.y2 - self.y1))

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        y = self.y(t)
        dy_dt = self.y_der(t)
        root = np.sqrt(1.0 + (y / self.b) ** 2)
        dx_dy = self._branch_sign() * self.a * y / (self.b * self.b * root)
        return dx_dy * dy_dt


@dataclass(frozen=True)
class EllipseArc(ParamCurve):
    """General ellipse arc with optional in-plane rotation.

    Parameters
    ----------
    center_x, center_y:
        Center of the ellipse.
    a, b:
        Semi-axis lengths in the local ellipse coordinates.
    theta1_rad, theta2_rad:
        Start and end parameter angles of the retained arc.
    rotation_rad:
        Rotation angle of the ellipse local frame relative to the global axes.
    """

    center_x: float
    center_y: float
    a: float
    b: float
    theta1_rad: float
    theta2_rad: float
    rotation_rad: float = 0.0

    @classmethod
    def from_foci(
        cls,
        focus_1: tuple[float, float],
        focus_2: tuple[float, float],
        semi_major: float,
        theta1_deg: float,
        theta2_deg: float,
    ) -> "EllipseArc":
        """Build an ellipse arc from two foci and the semi-major axis.

        Parameters
        ----------
        focus_1, focus_2:
            Coordinates of the two ellipse foci.
        semi_major:
            Semi-major axis length ``a``.
        theta1_deg, theta2_deg:
            Start and end angles of the retained arc in the local ellipse frame.
        """

        x1, y1 = focus_1
        x2, y2 = focus_2
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        dx = x2 - x1
        dy = y2 - y1
        c = 0.5 * float(np.hypot(dx, dy))
        if semi_major <= c:
            raise ValueError("semi_major must be larger than half the focal separation")
        b = float(np.sqrt(semi_major * semi_major - c * c))
        rotation_rad = float(np.arctan2(dy, dx))
        return cls(
            center_x=float(center_x),
            center_y=float(center_y),
            a=float(semi_major),
            b=b,
            theta1_rad=float(np.deg2rad(theta1_deg)),
            theta2_rad=float(np.deg2rad(theta2_deg)),
            rotation_rad=rotation_rad,
        )

    def _theta(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = as_float_array(t)
        mid = 0.5 * (self.theta1_rad + self.theta2_rad)
        half_span = 0.5 * (self.theta2_rad - self.theta1_rad)
        return mid + half_span * t_arr

    def _dtheta_dt(self) -> float:
        return 0.5 * (self.theta2_rad - self.theta1_rad)

    def _local_coords(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = self._theta(t)
        return self.a * np.cos(theta), self.b * np.sin(theta)

    def _local_derivatives(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = self._theta(t)
        dtheta_dt = self._dtheta_dt()
        return -self.a * np.sin(theta) * dtheta_dt, self.b * np.cos(theta) * dtheta_dt

    def _rotate(self, x_local: np.ndarray, y_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.cos(self.rotation_rad)
        s = np.sin(self.rotation_rad)
        x_world = self.center_x + c * x_local - s * y_local
        y_world = self.center_y + s * x_local + c * y_local
        return x_world, y_world

    def _rotate_derivative(
        self, dx_local: np.ndarray, dy_local: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        c = np.cos(self.rotation_rad)
        s = np.sin(self.rotation_rad)
        dx_world = c * dx_local - s * dy_local
        dy_world = s * dx_local + c * dy_local
        return dx_world, dy_world

    def x(self, t: float | np.ndarray) -> np.ndarray:
        x_local, y_local = self._local_coords(t)
        x_world, _ = self._rotate(x_local, y_local)
        return x_world

    def y(self, t: float | np.ndarray) -> np.ndarray:
        x_local, y_local = self._local_coords(t)
        _, y_world = self._rotate(x_local, y_local)
        return y_world

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        dx_local, dy_local = self._local_derivatives(t)
        dx_world, _ = self._rotate_derivative(dx_local, dy_local)
        return dx_world

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        dx_local, dy_local = self._local_derivatives(t)
        _, dy_world = self._rotate_derivative(dx_local, dy_local)
        return dy_world


@dataclass(frozen=True)
class TranslatedCurve(ParamCurve):
    """Translated copy of another curve.

    Parameters
    ----------
    base:
        Underlying curve before translation.
    dx, dy:
        Translation offsets applied to the curve coordinates.
    """

    base: ParamCurve
    dx: float = 0.0
    dy: float = 0.0

    def x(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.x(t) + self.dx

    def y(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.y(t) + self.dy

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.x_der(t)

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.y_der(t)


@dataclass(frozen=True)
class ReflectedCurveX(ParamCurve):
    """Mirror a base curve about the vertical line ``x = mirror_x``.

    Parameters
    ----------
    base:
        Curve to mirror.
    mirror_x:
        x-coordinate of the vertical reflection line.
    """

    base: ParamCurve
    mirror_x: float

    def x(self, t: float | np.ndarray) -> np.ndarray:
        return 2.0 * self.mirror_x - self.base.x(t)

    def y(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.y(t)

    def x_der(self, t: float | np.ndarray) -> np.ndarray:
        return -self.base.x_der(t)

    def y_der(self, t: float | np.ndarray) -> np.ndarray:
        return self.base.y_der(t)


def build_focus_centered_parabola(
    aperture: float,
    focal_ratio: float,
    x_focus: float = 0.0,
    y_focus: float = 0.0,
) -> ParamCurve:
    """Build a parabola whose focus is placed at ``(x_focus, y_focus)``."""

    f = focal_ratio * aperture
    base = ParabolaArc.symmetric(f=f, aperture=aperture)
    return TranslatedCurve(base=base, dx=x_focus - f, dy=y_focus)


def build_shifted_parabola_from_bounds(
    focal_length: float,
    y1: float,
    y2: float,
    x_shift: float = 0.0,
    y_shift: float = 0.0,
) -> ParamCurve:
    """Build a parabola and translate it by ``(x_shift, y_shift)``."""

    base = ParabolaArc(f=focal_length, y1=y1, y2=y2)
    return TranslatedCurve(base=base, dx=x_shift, dy=y_shift)


def build_shifted_parabola(
    aperture: float,
    focal_ratio: float,
    x_shift: float,
    y_shift: float = 0.0,
) -> ParamCurve:
    """Build a symmetric parabola and translate it by ``(x_shift, y_shift)``."""

    f = focal_ratio * aperture
    base = ParabolaArc.symmetric(f=f, aperture=aperture)
    return TranslatedCurve(base=base, dx=x_shift, dy=y_shift)


def symmetric_bounds(aperture: float) -> tuple[float, float]:
    """Return the symmetric truncation bounds ``(-aperture/2, +aperture/2)``."""

    half = 0.5 * aperture
    return -half, half


def build_axis_aligned_ellipse_side(
    center_x: float,
    center_y: float,
    semi_x: float,
    semi_y: float,
    half_angle_deg: float,
    side: str,
) -> EllipseArc:
    """Build a left or right bracket-like side of an axis-aligned ellipse.

    Parameters
    ----------
    center_x, center_y:
        Center of the ellipse.
    semi_x, semi_y:
        Horizontal and vertical semi-axis lengths.
    half_angle_deg:
        Half of the retained angular span around the chosen side.
        Symmetric angles guarantee that both arc endpoints have the same ``x``.
    side:
        Which side of the ellipse to keep, either ``"left"`` or ``"right"``.
    """

    if side == "right":
        theta1_deg = -half_angle_deg
        theta2_deg = half_angle_deg
    elif side == "left":
        theta1_deg = 180.0 - half_angle_deg
        theta2_deg = 180.0 + half_angle_deg
    else:
        raise ValueError("side must be 'left' or 'right'")
    return EllipseArc(
        center_x=float(center_x),
        center_y=float(center_y),
        a=float(semi_x),
        b=float(semi_y),
        theta1_rad=float(np.deg2rad(theta1_deg)),
        theta2_rad=float(np.deg2rad(theta2_deg)),
        rotation_rad=0.0,
    )


def ellipse_foci(
    center_x: float,
    center_y: float,
    a1: float,
    a2: float,
    rotation_deg: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the geometric foci of the ellipse ``(x'/a1)^2 + (y'/a2)^2 = 1``.

    Parameters
    ----------
    center_x, center_y:
        Center of the ellipse in global coordinates.
    a1, a2:
        Semi-axes of the ellipse in its local frame. ``a1`` is the semi-axis
        aligned with the local x-axis before rotation, matching the notation in
        the paper figure.
    rotation_deg:
        Rotation of the local ellipse frame relative to the global x-axis.
    """

    if a1 <= 0.0 or a2 <= 0.0:
        raise ValueError("a1 and a2 must be positive")
    if a1 < a2:
        raise ValueError("a1 must be greater than or equal to a2 for a horizontal-focus ellipse")
    c = float(np.sqrt(a1 * a1 - a2 * a2))
    angle = float(np.deg2rad(rotation_deg))
    dx = c * np.cos(angle)
    dy = c * np.sin(angle)
    return (center_x - dx, center_y - dy), (center_x + dx, center_y + dy)


def build_ellipse_arc_from_equation(
    center_x: float,
    center_y: float,
    a1: float,
    a2: float,
    half_angle_deg: float,
    side: str,
    rotation_deg: float = 0.0,
) -> EllipseArc:
    """Build an open ellipse arc from the paper-style equation definition.

    The underlying full curve is

    ``((x') / a1)^2 + ((y') / a2)^2 = 1``

    where ``(x', y')`` are coordinates in the reflector's local frame. The
    retained arc is selected by ``side`` and then rotated into the global frame.

    Parameters
    ----------
    center_x, center_y:
        Ellipse center in global coordinates.
    a1, a2:
        Semi-axes of the ellipse in its local frame.
    half_angle_deg:
        Half of the retained angular span around the chosen side.
        Values below ``90`` degrees produce clearly open arcs.
    side:
        Which side of the ellipse to keep, either ``"left"`` or ``"right"``.
    rotation_deg:
        Rotation of the ellipse local frame in global coordinates.
    """

    if side == "right":
        theta1_deg = -half_angle_deg
        theta2_deg = half_angle_deg
    elif side == "left":
        theta1_deg = 180.0 - half_angle_deg
        theta2_deg = 180.0 + half_angle_deg
    else:
        raise ValueError("side must be 'left' or 'right'")
    return EllipseArc(
        center_x=float(center_x),
        center_y=float(center_y),
        a=float(a1),
        b=float(a2),
        theta1_rad=float(np.deg2rad(theta1_deg)),
        theta2_rad=float(np.deg2rad(theta2_deg)),
        rotation_rad=float(np.deg2rad(rotation_deg)),
    )
