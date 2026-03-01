from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ParamCurve(ABC):
    """
    Parametrized curve in the plane: (x(t), y(t)), typically with t in [-1, 1].
    Implementations must be vectorized (accept numpy arrays).
    """

    @abstractmethod
    def x(self, t: FloatArray) -> FloatArray: ...

    @abstractmethod
    def y(self, t: FloatArray) -> FloatArray: ...

    @abstractmethod
    def x_der(self, t: FloatArray) -> FloatArray: ...

    @abstractmethod
    def y_der(self, t: FloatArray) -> FloatArray: ...

    # Correct
    def R(self, t: FloatArray, t0: FloatArray) -> FloatArray:
        """
        Pairwise distance R(t, t0) = sqrt( (x(t)-x(t0))^2 + (y(t)-y(t0))^2 )

        Supports broadcasting: t and t0 can be (n,1) and (1,m).
        """
        xt = self.x(t)
        yt = self.y(t)
        xt0 = self.x(t0)
        yt0 = self.y(t0)
        return np.sqrt((xt - xt0) ** 2 + (yt - yt0) ** 2)

    # Correct
    def R_t0_der_coord(self, t: FloatArray, coords0: FloatArray) -> FloatArray:
        """Mathematica provided"""
        xt = self.x(t)
        yt = self.y(t)
        xt0 = coords0[0]
        yt0 = coords0[1]
        result = ((xt - xt0) * self.x_der(t) + (yt - yt0) * self.y_der(t)) / np.sqrt(
            (xt - xt0) ** 2 + (yt - yt0) ** 2
        )
        return result

    # Correct
    def R_t0_der(self, t: FloatArray, t0: FloatArray) -> FloatArray:
        """Mathematica provided"""
        xt = self.x(t)
        yt = self.y(t)
        xt0 = self.x(t0)
        yt0 = self.y(t0)
        result = ((xt0 - xt) * self.x_der(t0) + (yt0 - yt) * self.y_der(t0)) / np.sqrt(
            (xt - xt0) ** 2 + (yt - yt0) ** 2
        )
        return result


@dataclass(frozen=True)
class LineSegment(ParamCurve):
    """
    Straight line from P0=(x0,y0) to P1=(x1,y1) with t in [-1,1].

    Mapping: s=(t+1)/2 in [0,1], then P(t)=P0 + s*(P1-P0).
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def x(self, t: FloatArray) -> FloatArray:
        s = (t + 1.0) / 2
        return self.x1 + s * (self.x2 - self.x1)

    def y(self, t: FloatArray) -> FloatArray:
        s = (t + 1.0) / 2
        return self.y1 + s * (self.y2 - self.y1)

    def x_der(self, t: FloatArray) -> FloatArray:
        return (self.x2 - self.x1) / 2

    def y_der(self, t: FloatArray) -> FloatArray:
        return (self.y2 - self.y1) / 2


@dataclass(frozen=True)
class ArcSegment(ParamCurve):
    x1: float
    x2: float
    y1: float
    y2: float
    _a: float = field(init=False, repr=False)
    _b: float = field(init=False, repr=False)
    _dydt: float = field(init=False, repr=False)
    _xder_factor: float = field(init=False, repr=False)
    _use_linear_x_of_y: bool = field(init=False, repr=False)
    _mx: float = field(init=False, repr=False)
    _bx: float = field(init=False, repr=False)
    _xder_linear: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        denom = self.y2 * self.y2 - self.x2 * self.x2
        dydt = (self.y2 - self.x2) / 2.0
        use_linear = np.isclose(denom, 0.0)

        if use_linear:
            if np.isclose(self.y2, self.x2):
                mx = 0.0
                bx = self.x1
            else:
                mx = (self.y1 - self.x1) / (self.y2 - self.x2)
                bx = self.x1 - mx * self.x2
            a = 0.0
            b = 0.0
            xder_factor = 0.0
            xder_linear = mx * dydt
        else:
            a = (self.y1 - self.x1) / denom
            b = self.x1 - a * self.x2 * self.x2
            mx = 0.0
            bx = 0.0
            xder_factor = a * (self.y2 - self.x2)
            xder_linear = 0.0

        object.__setattr__(self, "_a", a)
        object.__setattr__(self, "_b", b)
        object.__setattr__(self, "_dydt", dydt)
        object.__setattr__(self, "_xder_factor", xder_factor)
        object.__setattr__(self, "_use_linear_x_of_y", use_linear)
        object.__setattr__(self, "_mx", mx)
        object.__setattr__(self, "_bx", bx)
        object.__setattr__(self, "_xder_linear", xder_linear)

    def y(self, t: FloatArray) -> FloatArray:
        return self.x2 + (t + 1.0) * self._dydt

    def x(self, t: FloatArray) -> FloatArray:
        yv = self.y(t)
        if self._use_linear_x_of_y:
            return self._mx * yv + self._bx
        return self._a * yv * yv + self._b

    def y_der(self, t: FloatArray) -> FloatArray:
        return self._dydt

    def x_der(self, t: FloatArray) -> FloatArray:
        if self._use_linear_x_of_y:
            return self._xder_linear
        return self._xder_factor * self.y(t)
