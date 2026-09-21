"""Named search intervals and fixed values for phase-spiral components.

The fitter accepts one ParameterBounds object to broadcast constraints, or an
ordered sequence for component-specific constraints. These classes do not perform
unit conversion; coordinates and parameter values must use consistent units.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import numpy as np
import optype as op

type _ToBounds = Interval | Fixed | Sequence[op.CanFloat] | op.CanFloat


@dataclass(frozen=True)
class Interval:
    """A finite, closed search interval with lower <= upper.

    Equal endpoints are treated as a fixed parameter when constructing the
    optimization layout. Endpoints use the units of the constrained parameter.

    Attributes
    ----------
    lower : float
        The lower bound of the parameter.
    upper : float
        The upper bound of the parameter.

    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate interval bounds."""
        lower, upper = self.lower, self.upper

        if not np.isfinite(lower) or not np.isfinite(upper):
            msg = "Interval bounds must be finite."
            raise ValueError(msg)

        if lower > upper:
            msg = "Lower bound must not exceed upper bound."
            raise ValueError(msg)


@dataclass(frozen=True)
class Fixed:
    """A finite parameter value excluded from the optimization vector.

    The value uses the units of the constrained parameter.

    Attributes
    ----------
    value : float
        The fixed value.

    """

    value: float

    def __post_init__(self) -> None:
        """Validate interval bounds."""
        value = self.value

        if not np.isfinite(value):
            msg = "Fixed parameters must be finite."
            raise ValueError(msg)


class ParameterBounds:
    """Named parameter constraints for one phase-spiral component.

    Each keyword accepts an Interval, a two-value sequence, a Fixed, or a scalar.
    Scalars specify fixed values; omitted keywords retain their default intervals.
    Equal interval endpoints also become fixed during layout construction.

    Units below assume z in kpc and vz in km/s. More generally, r has the units
    of z, scale_factor has units of vz/z, and b and c have units of z/rad and
    z/rad**2, from r = b*phi_s + c*phi_s**2. No unit conversion is performed.
    Default intervals are search choices, not upper limits on valid parameters.

    Attributes
    ----------
    alpha : Interval | Fixed
        The bounds on the amplitude of the spiral perturbation `alpha`.
        The amplitude must always be non-negative and the default range is [0.0, 1.0].
        The units are dimensionless.
    b : Interval | Fixed
        The bounds on the linear winding parameter `b`.
        The winding parameter is always positive and the default range is [0.005, 0.1].
        The units are kpc/rad.
    c : Interval | Fixed
        The bounds on the quadratic winding parameter `c`.
        The winding parameter is always non-negative and the default range is [0.0, 0.004].
        The units are kpc/rad^2.
    theta0 : Interval | Fixed
        Angular offset in radians, default [-pi, pi]. Any finite ordered interval
        is allowed. Angles remain unwrapped: (pi - 0.2, pi + 0.2) searches around
        pi without crossing a discontinuity in the parameter representation.
    scale_factor : Interval | Fixed
        The velocity-to-position scale S in r = sqrt(z**2 + (vz/S)**2).
        The scale factor is always positive and the default range is [30.0, 70.0].
        The units are km/s/kpc.
    rho : Interval | Fixed
        The bounds on `rho`.
        The flattening function distance is always non-negative and the default range is [0.0, 0.18].
        The units are kpc.

    Notes
    -----
    All endpoints must be finite and lower <= upper. alpha, c, and rho must
    be nonnegative; b and scale_factor must be strictly positive. The bounds
    describe coefficient magnitudes; use the fitter's winding argument to choose
    direction. Intervals wider than 2*pi are allowed but search redundant angles.

    ParameterBounds fields remain mutable. Avoid modifying supplied constraints
    while a fit or its lazy event stream is in progress.

    Examples
    --------
    >>> bounds = ParameterBounds(c=Fixed(0.0), scale_factor=(35.0, 60.0))
    >>> bounds.c
    Fixed(value=0.0)

    """

    def __init__(
        self,
        *,
        alpha: _ToBounds = (0.0, 1.0),
        b: _ToBounds = (0.005, 0.1),
        c: _ToBounds = (0.0, 0.004),
        theta0: _ToBounds = (-np.pi, np.pi),
        scale_factor: _ToBounds = (30.0, 70.0),
        rho: _ToBounds = (0.0, 0.18),
    ) -> None:
        """Create parameter bounds.

        Parameters
        ----------
        alpha : _ToBounds
            The bounds on the spiral amplitude alpha.
        b : _ToBounds
            The bounds on the linear winding parameter b.
        c : _ToBounds
            The bounds on the quadratic winding parameter c.
        theta0 : _ToBounds
            The bounds on the angle theta0.
        scale_factor : _ToBounds
            The bounds on the scale factor S.
        rho : _ToBounds
            The bounds on the flattening function distance rho.

        """

        self.alpha: Interval | Fixed = ParameterBounds._parse_bounds(alpha)
        self.b: Interval | Fixed = ParameterBounds._parse_bounds(b)
        self.c: Interval | Fixed = ParameterBounds._parse_bounds(c)
        self.theta0: Interval | Fixed = ParameterBounds._parse_bounds(theta0)
        self.scale_factor: Interval | Fixed = ParameterBounds._parse_bounds(scale_factor)
        self.rho: Interval | Fixed = ParameterBounds._parse_bounds(rho)

        ParameterBounds._validate_nonnegative("alpha", self.alpha)
        ParameterBounds._validate_positive("b", self.b)
        ParameterBounds._validate_nonnegative("c", self.c)
        ParameterBounds._validate_positive("scale_factor", self.scale_factor)
        ParameterBounds._validate_nonnegative("rho", self.rho)

    @override
    def __str__(self) -> str:
        buffer = f"{type(self).__name__}("
        for idx, (name, bound) in enumerate(
            (
                ("alpha", self.alpha),
                ("b", self.b),
                ("c", self.c),
                ("theta0", self.theta0),
                ("scale_factor", self.scale_factor),
                ("rho", self.rho),
            )
        ):
            if idx > 0:
                buffer += ", "
            buffer += f"{name}={bound}"
        buffer += ")"
        return buffer

    @staticmethod
    def _parse_bounds(value: object) -> Interval | Fixed:
        """Parse an arbitrary object into either `Interval` or `Fixed`.

        Parameters
        ----------
        value : object
            An arbitrary object to be parsed.

        Returns
        -------
        Interval | Fixed
            The parameter bounds as an interval or a fixed parameter.

        """
        # Already parsed
        if isinstance(value, Interval | Fixed):
            return value

        # Parse Interval
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if len(value) != 2:
                msg = "Parameter bounds must contain exactly two values."
                raise TypeError(msg)

            lower: object = value[0]
            upper: object = value[1]

            if not isinstance(lower, op.CanFloat) or not isinstance(upper, op.CanFloat):
                msg = "Parameter bounds must be floats."
                raise TypeError(msg)

            return Interval(lower=float(lower), upper=float(upper))

        # Parse Fixed
        if isinstance(value, op.CanFloat):
            return Fixed(value=float(value))

        # Invalid type
        msg = "Parameter bounds must be `Interval`, `Fixed`, asequence of two floats or a single float."
        raise TypeError(msg)

    @staticmethod
    def _validate_positive(name: str, bound: Interval | Fixed) -> None:
        """Validate the bound is strictly positive.

        Parameters
        ----------
        name : str
            The name of the parameter bounds.
        bound : Interval | Fixed
            The bound as an interval or a fixed parameter.

        """
        lower = bound.lower if isinstance(bound, Interval) else bound.value

        if lower <= 0.0:
            msg = f"{name} must be positive."
            raise ValueError(msg)

    @staticmethod
    def _validate_nonnegative(name: str, bound: Interval | Fixed) -> None:
        """Validate the bound is non-negative.

        Parameters
        ----------
        name : str
            The name of the parameter bounds.
        bound : Interval | Fixed
            The bound as an interval or a fixed parameter.

        """
        lower = bound.lower if isinstance(bound, Interval) else bound.value

        if lower < 0.0:
            msg = f"{name} must be non-negative."
            raise ValueError(msg)
