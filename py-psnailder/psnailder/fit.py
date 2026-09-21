"""The spiral fitting algorithm."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, override

import numpy as np
import optype as op
from scipy import ndimage, optimize, special

from ._background_utils import generate_initial_background
from ._likelihood_utils import ln_likelihood
from .model import PSpiralModel

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from optype import numpy as onp


type _ObjectiveFunc = Callable[[onp.Array1D[np.float64]], onp.ToFloat]
type _SmoothingFunc = Callable[[onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _MaskFunc = Callable[[onp.Array2D[np.float64], onp.Array2D[np.float64]], onp.Array2D[np.float64]]

_DEFAULT_PARAM_LO: Final[onp.Array1D[np.float64]] = np.array([0.0, 0.005, 0.0, -np.pi, 30.0, 0.0])
_DEFAULT_PARAM_HI: Final[onp.Array1D[np.float64]] = np.array([1.0, 0.1, 0.004, +np.pi, 70.0, 0.18])

log: Final[logging.Logger] = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "FitEvent",
    "FitOutcome",
    "FitProgress",
    "FitSuccess",
    "FitFailure",
    "FitFailureReason",
    "FitTerminationReason",
    "PSpiralFitResult",
    "PSpiralFitter",
    "create_gaussian_smoother",
    "create_sigmoid_mask",
]


class FitTerminationReason(StrEnum):
    """The reason background refinement terminated.

    Variants
    --------
    NO_IMPROVEMENT
        Refinement stopped when background no longer improves i.e. background has converged.
    ITERATION_LIMIT
        The refinement iteration limit was reached.
    FIXED_BACKGROUND
        The user has asked for no background refinement.
    INVALID_BACKGROUND_UPDATE
        The updated background is invalid in some way and refinement can no longer proceed.
    FAILED_REOPTIMIZATION
        Re-optimization failed; the previously accepted fit was retained.

    """

    NO_IMPROVEMENT = "no_improvement"
    ITERATION_LIMIT = "iteration_limit"
    FIXED_BACKGROUND = "fixed_background"
    INVALID_BACKGROUND_UPDATE = "invalid_background_update"
    FAILED_REOPTIMIZATION = "failed_reoptimization"


class FitFailureReason(StrEnum):
    """The reason model optimisation failed.

    Variants
    --------
    NO_VALID_CANDIDATE
        There was no valid candidate found when optimizing.
    OPTIMIZER_FAILED
        The optimizer failed on its input.

    """

    NO_VALID_CANDIDATE = "no_valid_candidate"
    OPTIMIZER_FAILED = "optimizer_failed"


@dataclass(frozen=True)
class FitSuccess:
    """A successful result of fitting.

    Attributes
    ----------
    result : PSpiralFitResult
        The successful result.

    """

    result: PSpiralFitResult


@dataclass(frozen=True)
class FitFailure:
    """The diagnostic when fitting has failed.

    Attributes
    ----------
    reason : FitFailureReason
        The reason for failure.
    message : str
        An accompanying message.

    """

    reason: FitFailureReason
    message: str


@dataclass(frozen=True)
class FitProgress:
    """A valid intermediate fit.

    Attributes
    ----------
    model : PSpiralModel
        The current valid model.
    iteration : int
        The refinement step; zero denotes the initial fit.
    lnl : float
        The log-likelihood.

    """

    model: PSpiralModel
    iteration: int
    lnl: float


@dataclass
class PSpiralFitResult:
    """A result of the spiral fitting process.

    Attributes
    ----------
    initial_model : PSpiralModel
        The initial model.
    final_model : PSpiralModel
        The best fit model.
    data : Array2D[f64]
        The data.
    num_iterations : int
        The number of background refinement attempts, excluding the initial fit.
    max_iterations : int | None
        The maximum number of background refinement attempts, or None for no limit.
    lnl : float
        The log-likelihood.
    reason : FitTerminationReason
        The reason fitting stopped.

    """

    initial_model: PSpiralModel
    final_model: PSpiralModel
    data: onp.Array2D[np.float64]
    num_iterations: int
    max_iterations: int | None
    lnl: float
    reason: FitTerminationReason


type FitOutcome = FitSuccess | FitFailure
type FitEvent = FitProgress | FitSuccess | FitFailure


def create_gaussian_smoother(sigma: float) -> _SmoothingFunc:
    """Return a function that applies Gaussian smoothing to a 2D array.

    Parameters
    ----------
    sigma : float
        The smoothing width in units of pixels (for each axis).

    Returns
    -------
    func : Callable[[Array2D[f64]], Array2D[f64]]
        The Gaussian smoothing function.

    """

    def _func(arr: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        return ndimage.gaussian_filter(arr, sigma=sigma)

    return _func


def create_sigmoid_mask(z_scale: float, vz_scale: float) -> _MaskFunc:
    """Return a function that creates a sigmoid mask given a mesh over z and vz.

    The mask is parameterised by the scale length and scale velocity.

    Parameters
    ----------
    z_scale : float
        The scale length.
    vz_scale : float
        The scale velocity.

    Returns
    -------
    func : Callable[[Array2D[f64], Array2D[f64]], Array2D[f64]]
        The sigmoid mask function.

    """

    def _func(z_mesh: onp.Array2D[np.float64], vz_mesh: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        return -special.expit(np.square(z_mesh / z_scale) + np.square(vz_mesh / vz_scale) - 1.0) + 1.0

    return _func


type _ToBounds = Interval | Fixed | Sequence[op.CanFloat] | op.CanFloat


@dataclass(frozen=True)
class Interval:
    """Parameter bounds expressed as an interval.

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
    """A fixed parameter constraint.

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
    """Parameter bounds describing a component.

    Attributes
    ----------
    alpha : Interval | Fixed
        The bounds on alpha.
    b : Interval | Fixed
        The bounds on b.
    c : Interval | Fixed
        The bounds on c.
    theta0 : Interval | Fixed
        The bounds on theta0.
    scale_factor : Interval | Fixed
        The bounds on scale_factor.
    rho : Interval | Fixed
        The bounds on rho.

    """

    def __init__(
        self,
        *,
        alpha: _ToBounds,
        b: _ToBounds,
        c: _ToBounds,
        theta0: _ToBounds,
        scale_factor: _ToBounds,
        rho: _ToBounds,
    ) -> None:
        """Create parameter bounds.

        Parameters
        ----------
        alpha : _ToBounds
            The bounds on alpha.
        b : _ToBounds
            The bounds on b.
        c : _ToBounds
            The bounds on c.
        theta0 : _ToBounds
            The bounds on theta0.
        scale_factor : _ToBounds
            The bounds on scale_factor.
        rho : _ToBounds
            The bounds on rho.

        """

        self.alpha: Interval | Fixed = ParameterBounds._parse_bounds(alpha)
        self.b: Interval | Fixed = ParameterBounds._parse_bounds(b)
        self.c: Interval | Fixed = ParameterBounds._parse_bounds(c)
        self.theta0: Interval | Fixed = ParameterBounds._normalize_angle("theta0", ParameterBounds._parse_bounds(theta0))
        self.scale_factor: Interval | Fixed = ParameterBounds._parse_bounds(scale_factor)
        self.rho: Interval | Fixed = ParameterBounds._parse_bounds(rho)

        ParameterBounds._validate_nonnegative("alpha", self.alpha)
        ParameterBounds._validate_positive("b", self.b)
        ParameterBounds._validate_nonnegative("c", self.c)
        ParameterBounds._validate_positive("scale_factor", self.scale_factor)
        ParameterBounds._validate_nonnegative("rho", self.rho)

    @staticmethod
    def default() -> ParameterBounds:
        """Construct default bounds.

        Returns
        -------
        ParameterBounds
            The default bounds.

        """

        return ParameterBounds(
            alpha=(0.0, 1.0),
            b=(0.005, 0.1),
            c=(0.0, 0.004),
            theta0=(-np.pi, np.pi),
            scale_factor=(30.0, 70.0),
            rho=(0.0, 0.18),
        )

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

    @staticmethod
    def _normalize_angle(
        name: str,
        bound: Interval | Fixed,
    ) -> Interval | Fixed:
        """Normalize angles in a bound to be within [-pi, pi].

        Parameters
        ----------
        name : str
            The parameter's name.
        bound : Interval | Fixed
            The bound.

        Returns
        -------
        bound : Interval | Fixed
            The normalized bound.

        """
        period = 2.0 * np.pi

        if isinstance(bound, Fixed):
            return Fixed((bound.value + np.pi) % period - np.pi)

        width = bound.upper - bound.lower

        # Any interval covering a full revolution permits every angle.
        if width >= period:
            return Interval(-np.pi, np.pi)

        lower = (bound.lower + np.pi) % period - np.pi
        upper = lower + width

        if upper > np.pi:
            msg = f"{name} interval crosses the -pi/+pi boundary after normalization: [{lower}, {upper}]."
            raise ValueError(msg)

        if width == 0.0:
            return Fixed(lower)

        return Interval(lower, upper)


@dataclass(frozen=True)
class _ParameterLayout:
    """The parameter layout.

    Mostly a helper to convert between arrays of free parameters and the full parameter array.

    Attributes
    ----------
    template : Array1D[f64]
        An array of the full parameter set.
        Fixed values are set in their respective slot while non-fixed values will be overwritten.
    free_indices : Array1D[intp]
        The indices of the free parameters.
    lower : Array1D[f64]
        The lower bounds of the free parameters.
    upper  : Array1D[f64]
        The upper bounds of the free parameters.

    """

    template: onp.Array1D[np.float64]
    free_indices: onp.Array1D[np.intp]
    lower: onp.Array1D[np.float64]
    upper: onp.Array1D[np.float64]

    def __post_init__(self) -> None:
        """Validate the bounds and free indices."""

        num_total_parameters = len(self.template)
        num_free_parameters = len(self.free_indices)
        if num_free_parameters > num_total_parameters:
            msg = "The number of free parameters must not exceed the number of total parameters."
            raise ValueError(msg)

        if np.max(self.free_indices) >= len(self.template):
            msg = "The free parameter indices must not point past the number of total parameters."
            raise ValueError(msg)

        if np.min(self.free_indices) < 0:
            msg = "The free parameter indices must not be negative."
            raise ValueError(msg)

        lower, upper = self.lower, self.upper

        num_lower_bounds = len(lower)
        num_upper_bounds = len(upper)
        if num_lower_bounds != num_upper_bounds or num_lower_bounds != num_free_parameters:
            msg = "The number of free parameters must be equal to the number of lower and upper bounds."
            raise ValueError(msg)

        has_invalid_bounds = np.any(lower > upper)
        if has_invalid_bounds:
            msg = "Lower bounds must not exceed upper bounds."
            raise ValueError(msg)

    @property
    def num_free(self) -> int:
        """int: The number of free parameters."""
        return len(self.free_indices)

    def pack(self, full_parameters: onp.Array1D[np.float64], *, eps: float = 1e-12) -> onp.Array1D[np.float64]:
        """Pack an array of the full parameter set into an array of just the free parameters.

        The values are also clamped to be within the bounds +- a small epsilon.

        Parameters
        ----------
        full_parameters : Array1D[f64]
            The full parameter array.
        eps : float
            The ratio of the parameter bounds' width to be used to calculate the small epsilon.

        Returns
        -------
        free_parameters : Array1D[f64]
            The full parameter array stripped of any fixed values.

        Notes
        -----
        The values at indices where the parameter is fixed are not validated in anyway and are
        simply just discarded.

        """

        if len(full_parameters) != len(self.template):
            msg = f"Expected the number of full parameters to be {len(self.template)}."
            raise ValueError(msg)

        values: onp.Array1D[np.float64] = full_parameters[self.free_indices]
        width: onp.Array1D[np.float64] = self.upper - self.lower
        clamped = np.clip(values, self.lower + eps * width, self.upper - eps * width)
        return clamped

    def unpack(self, free_parameters: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]:
        """Unpack an array of the free parameter set into an array of the full parameter set.

        Parameters
        ----------
        free_parameters : Array1D[f64]
            The free parameter array.

        Returns
        -------
        full_parameters : Array1D[f64]
            An expanded array of the full parameters.

        """

        num_free_parameters = self.num_free
        if len(free_parameters) != num_free_parameters:
            msg = f"Expected the number of free parameters to be {num_free_parameters}"
            raise ValueError(msg)

        full_parameters = np.copy(self.template)
        full_parameters[self.free_indices] = free_parameters
        return full_parameters


class PSpiralFitter:
    """A configuration of the spiral fitting algorithm."""

    def __init__(
        self,
        *,
        max_iterations: int | None = 50,
        smoothing_func: _SmoothingFunc | None = None,
        mask_func: _MaskFunc | None = None,
        bounds: ParameterBounds | None = None,
    ) -> None:
        if max_iterations is not None and max_iterations < 0:
            raise ValueError("max_iterations must be nonnegative or None.")
        self._max_iterations: int | None = max_iterations

        self._smoothing_func: _SmoothingFunc = create_gaussian_smoother(2.0) if smoothing_func is None else smoothing_func
        self._mask_func: _MaskFunc = create_sigmoid_mask(1.0, 40.0) if mask_func is None else mask_func
        self._bounds: ParameterBounds = bounds if bounds is not None else ParameterBounds.default()

    def fit_spiral(
        self,
        z: onp.Array1D[np.float64],
        vz: onp.Array1D[np.float64],
        z_bins: onp.Array1D[np.float64],
        vz_bins: onp.Array1D[np.float64],
        *,
        winding: Literal[-1, 1] | None = None,
        warm_start: onp.Array1D[np.float64] | None = None,
        rng: np.random.Generator | None = None,
        num_components: int | None = None,
        improve_background: bool = True,
    ) -> FitOutcome:
        val = _get_value_from_gen(
            self.fit_spiral_gen(
                z,
                vz,
                z_bins,
                vz_bins,
                winding=winding,
                warm_start=warm_start,
                rng=rng,
                num_components=num_components,
                improve_background=improve_background,
            )
        )
        assert val is not None
        assert isinstance(val, FitSuccess | FitFailure)
        return val

    def fit_spiral_gen(
        self,
        z: onp.Array1D[np.float64],
        vz: onp.Array1D[np.float64],
        z_bins: onp.Array1D[np.float64],
        vz_bins: onp.Array1D[np.float64],
        *,
        winding: Literal[-1, 1] | None = None,
        warm_start: onp.Array1D[np.float64] | None = None,
        rng: np.random.Generator | None = None,
        num_components: int | None = None,
        improve_background: bool = True,
    ) -> Generator[FitEvent]:
        z_centres = 0.5 * (z_bins[:-1] + z_bins[1:])
        vz_centres = 0.5 * (vz_bins[:-1] + vz_bins[1:])
        z_mesh, vz_mesh = np.meshgrid(z_centres, vz_centres)
        density, _, _ = np.histogram2d(z, vz, bins=(z_bins, vz_bins), density=False)
        density = density.T
        background = generate_initial_background(z, vz, z_mesh, vz_mesh)
        # Normalize initial background so it has the same total counts as the density
        # This prevents an unnormalized KDE from being ignored by the likelihood
        # and ensures the initial background is on the same scale as the data.
        if np.sum(background) > 0:
            background = background / np.sum(background) * np.sum(density)
        yield from self.fit_spiral_with_background_gen(
            density,
            background,
            z_mesh,
            vz_mesh,
            winding=winding,
            warm_start=warm_start,
            rng=rng,
            num_components=num_components,
            improve_background=improve_background,
        )

    def fit_spiral_with_background(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        *,
        winding: Literal[-1, 1] | None = None,
        warm_start: onp.Array1D[np.float64] | None = None,
        rng: np.random.Generator | None = None,
        num_components: int | None = None,
        improve_background: bool = True,
    ) -> FitOutcome:
        val = _get_value_from_gen(
            self.fit_spiral_with_background_gen(
                initial_density,
                initial_background,
                z_mesh,
                vz_mesh,
                winding=winding,
                warm_start=warm_start,
                rng=rng,
                num_components=num_components,
                improve_background=improve_background,
            )
        )
        assert val is not None
        assert isinstance(val, FitSuccess | FitFailure)
        return val

    def fit_spiral_with_background_gen(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        *,
        winding: Literal[-1, 1] | None = None,
        warm_start: onp.Array1D[np.float64] | None = None,
        rng: np.random.Generator | None = None,
        num_components: int | None = None,
        improve_background: bool = True,
    ) -> Generator[FitEvent]:
        """Fit a phase spiral to the given vertical phase space map and background.

        Parameters
        ----------
        initial_density : Array2D[f64]
            The initial density.
        initial_background : Array2D[f64]
            The initial background.
        z_mesh : Array2D[f64]
            The z values for each cell.
        vz_mesh : Array2D[f64]
            The Vz values for each cell.
        winding : Literal[-1, 1] | None
            The winding direction to force if given otherwise it will be automatically determined.
        warm_start : Array1D[f64] | None
            The warm start parameters if given.
        rng : np.random.Generator | None
            Random generator shared by selection and refinement. If None, a new
            generator is created. Pass np.random.default_rng(seed) to reproduce a fit.
        improve_background : bool
            Whether to iteratively improve the background. If ``False``, the
            background is fixed to ``initial_background``.


        Yields
        ------
        event : FitEvent
            Valid progress snapshots followed by one terminal success or failure.

        """
        # Yield progress snapshots, then exactly one terminal success or failure.
        # If given a warm start, the number of components must be explicitly set.
        if warm_start is not None and num_components is None:
            msg = "Can not use warm start if the number of component is not set."
            raise ValueError(msg)

        if rng is None:
            rng = np.random.default_rng()

        mask: Final[onp.Array2D[np.float64]] = self._mask_func(z_mesh, vz_mesh)

        initial_fit = self._establish_initial_fit(
            initial_density,
            initial_background,
            mask,
            z_mesh,
            vz_mesh,
            rng=rng,
            num_components=num_components,
            guess=warm_start,
            winding=winding,
        )

        # Failed
        if isinstance(initial_fit, FitFailure):
            yield initial_fit
            return

        # Succeeded but no background refinement
        if not improve_background:
            yield initial_fit
            return

        yield from self._refine_background_gen(
            initial_fit,
            mask,
            rng,
        )

    def _optimize_parameters(
        self,
        objective_func: _ObjectiveFunc,
        *,
        rng: np.random.Generator,
        guess: onp.Array1D[np.float64] | None,
        num_components: int = 1,
    ) -> optimize.OptimizeResult:
        # `guess` may be None or a flat vector of length 6 * `num_components`
        assert guess is None or (guess.ndim == 1 and len(guess) == 6 * num_components)
        lo = np.tile(self._param_lo, num_components)
        hi = np.tile(self._param_hi, num_components)
        bounds = optimize.Bounds(lo, hi)

        if guess is not None:
            eps = 1e-12 * (hi - lo)
            clamped = np.clip(guess, lo + eps, hi - eps)
            out_of_bounds = int(np.count_nonzero(clamped != guess))
            if out_of_bounds:
                log.warning(
                    "Initial guess had %d of %d parameters outside bounds; clamping.",
                    out_of_bounds,
                    guess.size,
                )
            guess = clamped

        def objective(parameters: onp.Array1D[np.float64]) -> float:
            return float(objective_func(parameters))

        res = optimize.differential_evolution(objective, bounds=bounds, x0=guess, rng=rng)
        if not res.success:
            log.warning("Failed to find maximum likelihood: %s", res.message)
        return res

    def _establish_initial_fit(
        self,
        density: onp.Array2D[np.float64],
        background: onp.Array2D[np.float64],
        mask: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        *,
        rng: np.random.Generator,
        num_components: int | None = None,
        guess: onp.Array1D[np.float64] | None = None,
        winding: Literal[-1, 1] | None = None,
    ) -> FitOutcome:
        if num_components is not None:
            return self._fit_with_fixed_background(
                density,
                background,
                mask,
                z_mesh,
                vz_mesh,
                num_components=num_components,
                rng=rng,
                guess=guess,
                winding=winding,
            )

        res1 = self._fit_with_fixed_background(
            density,
            background,
            mask,
            z_mesh,
            vz_mesh,
            num_components=1,
            rng=rng,
            guess=guess,
            winding=winding,
        )
        res2 = self._fit_with_fixed_background(
            density,
            background,
            mask,
            z_mesh,
            vz_mesh,
            num_components=2,
            rng=rng,
            guess=guess,
            winding=winding,
        )

        # Both fits failed
        if isinstance(res1, FitFailure) and isinstance(res2, FitFailure):
            return FitFailure(
                FitFailureReason.NO_VALID_CANDIDATE, "No valid candidate found for either 1-component and 2-component fits."
            )
        # 1-component fit succeeded while 2-component fit failed
        elif isinstance(res1, FitSuccess) and isinstance(res2, FitFailure):
            selected = res1.result
        # 1-component fit failed while 2-component fit succeeded
        elif isinstance(res1, FitFailure) and isinstance(res2, FitSuccess):
            selected = res2.result
        # Both succeeded
        else:
            assert isinstance(res1, FitSuccess), "just checked in above branches."
            assert isinstance(res2, FitSuccess), "just checked in above branches."
            q1 = ln_likelihood(density, res1.result.final_model.prediction(), mask)
            q2 = ln_likelihood(density, res2.result.final_model.prediction(), mask)
            # BIC penalizes the larger model's additional parameters.
            k1 = 6
            k2 = 12
            num_particles = np.sum(density)
            b1 = k1 * np.log(num_particles) - 2.0 * q1
            b2 = k2 * np.log(num_particles) - 2.0 * q2
            selected = res2.result if b2 < b1 else res1.result

        return FitSuccess(selected)

    def _fit_with_fixed_background(
        self,
        density: onp.Array2D[np.float64],
        background: onp.Array2D[np.float64],
        mask: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        *,
        num_components: int,
        rng: np.random.Generator,
        guess: onp.Array1D[np.float64] | None = None,
        winding: Literal[-1, 1] | None = None,
    ) -> FitOutcome:
        def wrap_winding_objective(current_winding: Literal[-1, 1]) -> _ObjectiveFunc:
            def _objective(parameters: onp.Array1D[np.float64]) -> float:
                params = np.array(parameters, dtype=np.float64).reshape((num_components, 6))
                model = PSpiralModel(params, z_mesh, vz_mesh, background, winding=current_winding)
                return -ln_likelihood(density, model.prediction(), mask)

            return _objective

        res: optimize.OptimizeResult
        chosen_winding: Literal[-1, 1]
        if winding is None:
            pos_res = self._optimize_parameters(wrap_winding_objective(1), rng=rng, guess=guess, num_components=num_components)
            neg_res = self._optimize_parameters(wrap_winding_objective(-1), rng=rng, guess=guess, num_components=num_components)
            if np.isfinite(pos_res.fun) and (not np.isfinite(neg_res.fun) or pos_res.fun <= neg_res.fun):
                chosen_winding = 1
                res = pos_res
            else:
                chosen_winding = -1

                res = neg_res
        else:
            # Optimize for chosen winding.
            chosen_winding = winding
            res = self._optimize_parameters(wrap_winding_objective(winding), rng=rng, guess=guess, num_components=num_components)
        if not np.isfinite(res.fun):
            return FitFailure(reason=FitFailureReason.NO_VALID_CANDIDATE, message="No valid candidate model was found.")
        params: onp.Array2D[np.float64] = np.array(res.x, dtype=np.float64).reshape((num_components, 6))
        model = PSpiralModel(params, z_mesh, vz_mesh, background, winding=chosen_winding)
        return FitSuccess(
            PSpiralFitResult(
                initial_model=model,
                final_model=model,
                data=density,
                num_iterations=0,
                max_iterations=self._max_iterations,
                lnl=-float(res.fun),
                reason=FitTerminationReason.FIXED_BACKGROUND,
            )
        )

    def _refine_background_gen(
        self, initial_fit: FitSuccess, mask: onp.Array2D[np.float64], rng: np.random.Generator
    ) -> Generator[FitSuccess | FitProgress]:
        # Yield the initial fit
        initial_model: PSpiralModel = initial_fit.result.final_model
        initial_lnl: float = initial_fit.result.lnl
        z_mesh: onp.Array2D[np.float64] = initial_model.z_mesh
        vz_mesh: onp.Array2D[np.float64] = initial_model.vz_mesh
        num_components: int = initial_fit.result.final_model.num_components
        accepted: tuple[PSpiralModel, float] = (initial_model, initial_lnl)
        yield FitProgress(model=initial_model, lnl=initial_lnl, iteration=0)

        num_iterations: int = 0
        initial_density: Final[onp.Array2D[np.float64]] = initial_fit.result.data
        initial_density_norm = np.sum(initial_density)
        while self._max_iterations is None or (num_iterations < self._max_iterations):
            num_iterations += 1

            current_model, current_lnl = accepted
            current_perturbation = current_model.signal()
            new_background = self._smoothing_func(initial_density / current_perturbation)
            new_background: onp.Array2D[np.float64] = new_background / np.sum(new_background) * initial_density_norm

            if np.any(~np.isfinite(new_background)):
                yield FitSuccess(
                    result=PSpiralFitResult(
                        initial_model=initial_model,
                        final_model=current_model,
                        lnl=current_lnl,
                        data=initial_density,
                        num_iterations=num_iterations,
                        max_iterations=self._max_iterations,
                        reason=FitTerminationReason.INVALID_BACKGROUND_UPDATE,
                    )
                )
                return

            candidate = self._fit_with_fixed_background(
                initial_density,
                new_background,
                mask,
                z_mesh,
                vz_mesh,
                num_components=num_components,
                rng=rng,
                guess=current_model.parameters.flatten(),
                winding=initial_model.winding,
            )

            if isinstance(candidate, FitFailure):
                yield FitSuccess(
                    PSpiralFitResult(
                        initial_model=initial_model,
                        final_model=current_model,
                        lnl=current_lnl,
                        data=initial_density,
                        num_iterations=num_iterations,
                        max_iterations=self._max_iterations,
                        reason=FitTerminationReason.FAILED_REOPTIMIZATION,
                    )
                )
                return

            # New lnl worse or equal to currently accepted lnl
            if candidate.result.lnl <= current_lnl:
                yield FitSuccess(
                    PSpiralFitResult(
                        initial_model=initial_model,
                        final_model=current_model,
                        lnl=current_lnl,
                        data=initial_density,
                        num_iterations=num_iterations,
                        max_iterations=self._max_iterations,
                        reason=FitTerminationReason.NO_IMPROVEMENT,
                    )
                )
                return

            accepted = (candidate.result.final_model, candidate.result.lnl)
            yield FitProgress(
                model=candidate.result.final_model,
                iteration=num_iterations,
                lnl=candidate.result.lnl,
            )
        yield FitSuccess(
            PSpiralFitResult(
                initial_model=initial_model,
                final_model=accepted[0],
                lnl=accepted[1],
                data=initial_density,
                num_iterations=num_iterations,
                max_iterations=self._max_iterations,
                reason=FitTerminationReason.ITERATION_LIMIT,
            )
        )


def _get_value_from_gen[T](gen: Generator[T]) -> T | None:
    """Unwrap last yield value from generator.

    Parameters
    ----------
    gen : Generator[T]
        The generator.

    Returns
    -------
    val : T | None
        The last yielded value or ``None`` if the generator is empty.

    """
    val: T | None = None
    for inner in gen:
        val = inner
    return val
