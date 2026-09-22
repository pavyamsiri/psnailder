"""The spiral fitting algorithm."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from scipy import ndimage, optimize, special

from ._background_utils import generate_initial_background
from ._likelihood_utils import ln_likelihood
from .bounds import Fixed, Interval, ParameterBounds
from .model import PSpiralModel
from .param_layout import ParameterLayout

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from optype import numpy as onp


type _ObjectiveFunc = Callable[[onp.Array1D[np.float64]], onp.ToFloat]
type _SmoothingFunc = Callable[[onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _MaskFunc = Callable[[onp.Array2D[np.float64], onp.Array2D[np.float64]], onp.Array2D[np.float64]]

log: Final[logging.Logger] = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "Interval",
    "Fixed",
    "ParameterBounds",
    "FitEvent",
    "FitOutcome",
    "FitProgress",
    "FitSuccess",
    "FitFailure",
    "FitFailureReason",
    "FitTerminationReason",
    "OptimizationDiagnostics",
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
    CONVERGED
        A positive improvement was accepted but did not exceed the refinement tolerance.
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
    CONVERGED = "converged"
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
    """An outcome containing a valid fit, not a guarantee of convergence.

    Inspect result.reason to distinguish an iteration limit, lack of improvement,
    or retention of a valid fit after a failed refinement attempt.

    Attributes
    ----------
    result : PSpiralFitResult
        The successful result.
    diagnostics : OptimizationDiagnostics
        The overall diagnostics of all optimization attempts, including rejected fits.

    """

    result: PSpiralFitResult
    diagnostics: OptimizationDiagnostics


@dataclass(frozen=True)
class FitFailure:
    """A terminal outcome indicating that no valid fit was obtained.

    This outcome has no model or result payload. Invalid caller configuration
    can still raise exceptions; FitFailure represents an unsuccessful fit.

    Attributes
    ----------
    reason : FitFailureReason
        The reason for failure.
    message : str
        An accompanying message.
    diagnostics : OptimizationDiagnostics
        The overall diagnostics of all optimizations up to failure.

    """

    reason: FitFailureReason
    message: str
    diagnostics: OptimizationDiagnostics


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
        The score of model.prediction() against the data and fitting mask.
    diagnostics : OptimizationDiagnostics
        Work for this step. At iteration zero this includes component and winding selection.

    Notes
    -----
    Only accepted fits are emitted. A rejected attempt may increase the final
    result's iteration count without producing another progress event. Treat
    models and their arrays as read-only while consuming the stream.

    """

    model: PSpiralModel
    iteration: int
    lnl: float
    diagnostics: OptimizationDiagnostics


@dataclass
class PSpiralFitResult:
    """A result of the spiral fitting process.

    Attributes
    ----------
    initial_model : PSpiralModel
        The selected initial fixed-background fit, after component-count and
        winding selection. This is not the caller's warm-start guess.
    final_model : PSpiralModel
        The best fit model.
    data : Array2D[f64]
        The data.
    num_iterations : int
        The number of background refinement attempts, excluding the initial fit.
    max_iterations : int | None
        The maximum number of background refinement attempts, or None for no limit.
    lnl : float
        The score of final_model.prediction() against data and the fitting mask.
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


@dataclass(frozen=True)
class _OptimizationResult:
    """The parameter optimization result.

    Attributes
    ----------
    parameters : Array1D[f64]
        The optimized (free) parameters.
    cost : float
        The minimized cost.
    success : bool
        Whether optimization was successful.
    nfev : int
        The number of function evaluations done during optimization.
    nit : int
        The number of optimization iterations.

    """

    parameters: onp.Array1D[np.float64]
    cost: float
    success: bool
    nfev: int
    nit: int
    message: str

    @property
    def diagnostics(self) -> OptimizationDiagnostics:
        return OptimizationDiagnostics(message=self.message, success=self.success, nfev=self.nfev, nit=self.nit)


@dataclass(frozen=True)
class OptimizationDiagnostics:
    """The parameter optimization's diagnostics.

    Attributes
    ----------
    message : str
        The optimizer's diagnostic message.
    success : bool
        Whether every included optimizer converged. A valid fit can still have
        success=False, for example after an optimizer iteration limit.
    nfev : int
        The number of function evaluations done during optimization.
    nit : int
        The sum of optimizer iterations, not background refinement attempts.

    """

    message: str
    success: bool
    nfev: int
    nit: int


def _combine_diagnostics(*items: OptimizationDiagnostics) -> OptimizationDiagnostics:
    """Combine disjoint attempts into an immutable snapshot."""
    return OptimizationDiagnostics(
        message="; ".join(item.message for item in items),
        success=all(item.success for item in items),
        nfev=sum(item.nfev for item in items),
        nit=sum(item.nit for item in items),
    )


class PSpiralFitter:
    """A configuration of the spiral fitting algorithm."""

    def __init__(
        self,
        *,
        max_iterations: int | None = 50,
        atol: float = 0.0,
        rtol: float = 0.0,
        smoothing_func: _SmoothingFunc | None = None,
        mask_func: _MaskFunc | None = None,
        bounds: ParameterBounds | Sequence[ParameterBounds] | None = None,
    ) -> None:
        """Initialize the fitter given the configuration.

        Parameters
        ----------
        max_iterations : int | None
            Maximum number of refinement attempts, excluding the initial fit.
            Default 50. Zero retains the initial fit; None imposes no iteration
            limit, though other termination conditions still apply.
        smoothing_func : _SmoothingFunc | None
            Callable mapping a 2D background proposal to a same-shaped array.
            None uses a Gaussian smoother with sigma=2.0 grid cells.
        atol, rtol : float
            Finite, nonnegative background-refinement tolerances. Accept a positive
            log-likelihood improvement and stop when it is at most
            atol + rtol * abs(previous_lnl). Both default to zero, preserving
            strict improvement. These do not change optimizer tolerances.
        mask_func : _MaskFunc | None
            Callable taking (z_mesh, vz_mesh) and returning same-shaped residual
            weights. None uses a sigmoid mask with scales 1.0 and 40.0 in the
            respective coordinate units. Weights multiply residuals before squaring.
        bounds : ParameterBounds | Sequence[ParameterBounds] | None
            None uses ParameterBounds(). A single object broadcasts its constraints
            to each component, whose free values are optimized independently.
            A sequence uses its first N entries for an N-component candidate.
            Automatic one/two-component selection requires at least two sequence
            entries; a one-entry sequence does not broadcast. Order therefore
            affects the one-component candidate in automatic selection.

        """
        if max_iterations is not None and max_iterations < 0:
            raise ValueError("max_iterations must be nonnegative or None.")
        self._max_iterations: int | None = max_iterations
        for name, tolerance in (("atol", atol), ("rtol", rtol)):
            if not np.isfinite(tolerance) or tolerance < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        self._atol = float(atol)
        self._rtol = float(rtol)

        self._smoothing_func: _SmoothingFunc = create_gaussian_smoother(2.0) if smoothing_func is None else smoothing_func
        self._mask_func: _MaskFunc = create_sigmoid_mask(1.0, 40.0) if mask_func is None else mask_func

        # A single object broadcasts; explicit sequences select an ordered prefix.
        self._bounds: ParameterBounds | Sequence[ParameterBounds] = bounds if bounds is not None else ParameterBounds()

    def _component_bounds(self, num_components: int) -> Sequence[ParameterBounds]:
        if num_components < 1:
            raise ValueError("Component bounds require a positive component count.")
        if isinstance(self._bounds, ParameterBounds):
            return (self._bounds,) * num_components
        if num_components > len(self._bounds):
            raise ValueError("Not enough bounds for the requested component count.")
        return self._bounds[:num_components]

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
        """Bin samples, estimate their background, and return a terminal outcome.

        Parameters
        ----------
        z, vz : Array1D[f64]
            Paired position and velocity samples in consistent units, conventionally
            kpc and km/s.
        z_bins, vz_bins : Array1D[f64]
            Increasing histogram bin edges in the corresponding coordinate units.
            Binned maps have shape (len(vz_bins)-1, len(z_bins)-1).
        winding : {-1, 1} or None
            Fixed direction, or None to compare both directions during initialization.
        warm_start : Array1D[f64] or None
            Full, component-major parameter vector; requires num_components.
            See fit_spiral_with_background_gen for ordering and bounds handling.
        rng : numpy.random.Generator or None
            Shared optimization RNG. Supply a fresh np.random.default_rng(seed)
            for reproducible calls; reusing a generator advances its state.
        num_components : int or None
            Positive component count, or None to compare one and two using BIC.
        improve_background : bool
            Whether to refine after obtaining the initial fit. Default True.

        Returns
        -------
        FitSuccess or FitFailure
            Check the outcome type before accessing FitSuccess.result. A valid
            result can be returned even when refinement did not converge.

        See Also
        --------
        fit_spiral_gen : Lazy version yielding progress and a terminal outcome.
        fit_spiral_with_background_gen : Common fitting-option semantics.
        """
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
        """Lazily bin samples and yield fit events.

        Inputs and keyword options are the same as fit_spiral. Binning and
        background estimation start on iteration, not generator construction.

        Yields
        ------
        FitProgress or FitSuccess or FitFailure
            When refinement is enabled and initialization succeeds, progress
            starts at step zero. Accepted updates emit further progress, followed
            by exactly one terminal outcome. With refinement disabled, only the
            terminal outcome is emitted. Input errors can still raise exceptions.

        See Also
        --------
        fit_spiral : Sample arrays, bin edges, and fitting options.
        fit_spiral_with_background_gen : Event and warm-start semantics.
        """
        for name, edges in (("z_bins", z_bins), ("vz_bins", vz_bins)):
            if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
                raise ValueError(f"{name} must contain finite, strictly increasing bin edges.")

        z_centres = 0.5 * (z_bins[:-1] + z_bins[1:])
        vz_centres = 0.5 * (vz_bins[:-1] + vz_bins[1:])
        z_mesh, vz_mesh = np.meshgrid(z_centres, vz_centres)
        density, _, _ = np.histogram2d(z, vz, bins=(z_bins, vz_bins), density=False)
        density = density.T
        background = generate_initial_background(z, vz, z_mesh, vz_mesh)
        # Midpoint density times bin area approximates probability mass.
        # Rows follow vz and columns follow z, matching the transposed counts.
        background = background * np.diff(vz_bins)[:, None] * np.diff(z_bins)[None, :]
        # Normalize over the fitting region to match its observed count total.
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
        """Fit an existing count map and background, returning the terminal outcome.

        The four positional arrays must share a 2D shape, with rows along vz and
        columns along z. Keyword options and normalization behavior are documented
        in fit_spiral_with_background_gen.

        Returns
        -------
        FitSuccess or FitFailure
            FitSuccess contains a valid result and termination reason. FitFailure
            contains diagnostics without a result. Invalid inputs may still raise.

        See Also
        --------
        fit_spiral_with_background_gen : Full parameter and event documentation.
        """
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
            Observed counts per cell, with positive finite total. Rows correspond
            to vz and columns to z; all four input maps must have identical shapes.
        initial_background : Array2D[f64]
            Initial background shape. Every fitted prediction is normalized to
            the observed total, and that scale is stored in the returned model's
            background. With refinement disabled the shape is fixed, but its
            overall normalization may change. The supplied array is not rescaled in place.
        z_mesh : Array2D[f64]
            The z values for each cell.
        vz_mesh : Array2D[f64]
            The Vz values for each cell.
        winding : Literal[-1, 1] | None
            Fixed direction, or None to compare both during initial fitting.
            The selected direction remains fixed throughout refinement.
        warm_start : Array1D[f64] | None
            Full vector of length 6*num_components, ordered as alpha, b, c,
            theta0, scale_factor, rho for each component in bounds order.
            Requires an explicit num_components. Fixed entries are replaced by
            configured values; free entries are clipped slightly inside their
            intervals and seed one member of the DE population. This does not
            resume optimizer state. Later refinements reuse the accepted model.
        rng : np.random.Generator | None
            Random generator shared by selection and refinement. If None, a new
            generator is created. Pass np.random.default_rng(seed) to reproduce a fit.
        num_components : int | None
            Positive component count, or None to compare one and two components
            on the initial background using BIC and their free-parameter counts.
            Explicit bounds sequences use the first N entries for each candidate;
            automatic selection requires at least two entries. The chosen count
            remains fixed during refinement.
        improve_background : bool
            Whether to iteratively update the background shape. False returns
            the initial normalized fit with reason FIXED_BACKGROUND.


        Yields
        ------
        event : FitEvent
            With refinement enabled, the initial valid fit is FitProgress at
            iteration zero, followed by progress for each accepted update and
            exactly one FitSuccess or FitFailure. Rejected attempts do not emit
            progress. Without refinement, only the terminal outcome is emitted.

        Notes
        -----
        Each refinement attempt proposes a background, re-optimizes the selected
        model, and accepts only a strictly higher score. Equal or lower scores
        terminate with NO_IMPROVEMENT. Invalid proposals or failed re-optimization
        retain the last valid fit with their corresponding termination reason.
        max_iterations counts attempts, including rejected attempts, and excludes
        initialization. Zero returns the initial fit with ITERATION_LIMIT.
        All-fixed parameter configurations bypass DE and are evaluated directly.
        Invalid caller configuration can raise; an inability to obtain a valid
        initial fit is represented by FitFailure. Supplied models, bounds, and
        arrays should not be mutated while consuming the stream.

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
        bounds: optimize.Bounds,
        rng: np.random.Generator,
        guess: onp.Array1D[np.float64] | None,
    ) -> _OptimizationResult:
        """Minimize the cost of the objective.

        Parameters
        ----------
        objective_func : ObjectiveFunc
            The objective to minimize.
        bounds : optimize.Bounds
            The bounds object.
        rng : np.random.Generator
            The rng.
        guess : Array1D[f64] | None
            A guess of the optimal parameters, must be trimmed to only free parameters.

        Returns
        -------
        result : OptimizationResult
            The optimization result.

        """

        def objective(parameters: onp.Array1D[np.float64]) -> float:
            return float(objective_func(parameters))

        if bounds.lb.size == 0:
            parameters = np.empty(0, dtype=np.float64)
            cost = objective(parameters)
            return _OptimizationResult(
                parameters=parameters,
                cost=cost,
                success=bool(np.isfinite(cost)),
                nfev=1,
                nit=0,
                message="All parameters fixed; evaluated objective once.",
            )

        res = optimize.differential_evolution(objective, bounds=bounds, x0=guess, rng=rng)
        if not res.success:
            log.warning("Optimization did not converge: %s", res.message)
        return _OptimizationResult(
            parameters=res.x,
            cost=res.fun,
            success=res.success,
            nfev=res.nfev,
            nit=res.nit,
            message=str(res.message),
        )

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

        if not isinstance(self._bounds, ParameterBounds) and len(self._bounds) < 2:
            msg = "Automatic component selection requires at least two sets of bounds."
            raise ValueError(msg)

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
                FitFailureReason.NO_VALID_CANDIDATE,
                "No valid candidate found for either 1-component and 2-component fits.",
                _combine_diagnostics(res1.diagnostics, res2.diagnostics),
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
            k1 = ParameterLayout.from_bounds(self._component_bounds(1)).num_free
            k2 = ParameterLayout.from_bounds(self._component_bounds(2)).num_free
            num_particles = np.sum(density)
            b1 = k1 * np.log(num_particles) - 2.0 * q1
            b2 = k2 * np.log(num_particles) - 2.0 * q2
            selected = res2.result if b2 < b1 else res1.result

        return FitSuccess(
            selected,
            diagnostics=_combine_diagnostics(res1.diagnostics, res2.diagnostics),
        )

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
        layout = ParameterLayout.from_bounds(self._component_bounds(num_components))
        bounds = optimize.Bounds(lb=layout.lower, ub=layout.upper)
        free_guess: onp.Array1D[np.float64] | None = layout.pack(guess) if guess is not None else None

        density_total = np.sum(density)
        if not np.isfinite(density_total) or density_total <= 0:
            return FitFailure(
                reason=FitFailureReason.NO_VALID_CANDIDATE,
                message="Total observed count must be positive and finite.",
                diagnostics=OptimizationDiagnostics(message="No optimization took place.", success=False, nfev=0, nit=0),
            )

        def wrap_winding_objective(current_winding: Literal[-1, 1]) -> _ObjectiveFunc:
            def _objective(free_parameters: onp.Array1D[np.float64]) -> float:
                params = layout.unpack(free_parameters).reshape((num_components, 6))
                model = PSpiralModel(params, z_mesh, vz_mesh, background, winding=current_winding)
                prediction = model.prediction()
                prediction_total = np.sum(prediction)
                if not np.isfinite(prediction_total) or prediction_total <= 0:
                    return float("inf")
                prediction *= density_total / prediction_total
                return -ln_likelihood(density, prediction, mask)

            return _objective

        res: _OptimizationResult
        chosen_winding: Literal[-1, 1]
        if winding is None:
            pos_res = self._optimize_parameters(wrap_winding_objective(1), rng=rng, guess=free_guess, bounds=bounds)
            neg_res = self._optimize_parameters(wrap_winding_objective(-1), rng=rng, guess=free_guess, bounds=bounds)
            diagnostics = _combine_diagnostics(pos_res.diagnostics, neg_res.diagnostics)
            if np.isfinite(pos_res.cost) and (not np.isfinite(neg_res.cost) or pos_res.cost <= neg_res.cost):
                chosen_winding = 1
                res = pos_res
            else:
                chosen_winding = -1

                res = neg_res
        else:
            # Optimize for chosen winding.
            chosen_winding = winding
            res = self._optimize_parameters(wrap_winding_objective(winding), rng=rng, guess=free_guess, bounds=bounds)
            diagnostics = res.diagnostics

        if not np.isfinite(res.cost):
            return FitFailure(
                reason=FitFailureReason.NO_VALID_CANDIDATE, message="No valid candidate model was found.", diagnostics=diagnostics
            )

        params: onp.Array2D[np.float64] = layout.unpack(res.parameters).reshape((num_components, 6))
        model = PSpiralModel(params, z_mesh, vz_mesh, background, winding=chosen_winding)
        prediction_total = np.sum(model.prediction())
        if not np.isfinite(prediction_total) or prediction_total <= 0:
            return FitFailure(
                reason=FitFailureReason.NO_VALID_CANDIDATE,
                message="Total predicted count must be positive and finite.",
                diagnostics=diagnostics,
            )
        # Store the winning scale without mutating the caller's background.
        model.background = background * (density_total / prediction_total)
        final_lnl = ln_likelihood(density, model.prediction(), mask)
        if not np.isfinite(final_lnl):
            return FitFailure(
                reason=FitFailureReason.NO_VALID_CANDIDATE, message="Normalized prediction is invalid.", diagnostics=diagnostics
            )
        return FitSuccess(
            PSpiralFitResult(
                initial_model=model,
                final_model=model,
                data=density,
                num_iterations=0,
                max_iterations=self._max_iterations,
                lnl=final_lnl,
                reason=FitTerminationReason.FIXED_BACKGROUND,
            ),
            diagnostics=diagnostics,
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
        diagnostics = initial_fit.diagnostics
        yield FitProgress(model=initial_model, lnl=initial_lnl, iteration=0, diagnostics=diagnostics)

        num_iterations: int = 0
        termination_reason = FitTerminationReason.ITERATION_LIMIT
        initial_density: Final[onp.Array2D[np.float64]] = initial_fit.result.data
        while self._max_iterations is None or (num_iterations < self._max_iterations):
            num_iterations += 1

            current_model, current_lnl = accepted
            current_perturbation = current_model.signal()
            new_background = self._smoothing_func(initial_density / current_perturbation)

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
                    ),
                    diagnostics=diagnostics,
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
            # Count rejected and failed attempts too; progress only reports accepted steps.
            diagnostics = _combine_diagnostics(diagnostics, candidate.diagnostics)

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
                    ),
                    diagnostics=diagnostics,
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
                    ),
                    diagnostics=diagnostics,
                )
                return

            accepted = (candidate.result.final_model, candidate.result.lnl)
            converged = candidate.result.lnl - current_lnl <= self._atol + self._rtol * abs(current_lnl)
            yield FitProgress(
                model=candidate.result.final_model,
                iteration=num_iterations,
                lnl=candidate.result.lnl,
                diagnostics=candidate.diagnostics,
            )
            # Improvement was too marginal and so we have converged
            if converged:
                termination_reason = FitTerminationReason.CONVERGED
                break
        yield FitSuccess(
            PSpiralFitResult(
                initial_model=initial_model,
                final_model=accepted[0],
                lnl=accepted[1],
                data=initial_density,
                num_iterations=num_iterations,
                max_iterations=self._max_iterations,
                reason=termination_reason,
            ),
            diagnostics=diagnostics,
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
