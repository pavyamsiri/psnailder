"""The spiral fitting algorithm."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from scipy import ndimage, optimize, special

from ._background_utils import generate_initial_background
from ._likelihood_utils import ln_likelihood
from .model import PSpiralModel

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import Final

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
        The current iteration.
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
        The number of iterations taken.
    max_iterations : int | None
        The maximum number of iterations.
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
    def _func(arr: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        return ndimage.gaussian_filter(arr, sigma=sigma)

    return _func


def create_sigmoid_mask(z_scale: float, vz_scale: float) -> _MaskFunc:
    def _func(z_mesh: onp.Array2D[np.float64], vz_mesh: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        return -special.expit(np.square(z_mesh / z_scale) + np.square(vz_mesh / vz_scale) - 1.0) + 1.0

    return _func


class PSpiralFitter:
    """A configuration of the spiral fitting algorithm."""

    def __init__(
        self,
        *,
        max_iterations: int | None = 50,
        smoothing_func: _SmoothingFunc | None = None,
        mask_func: _MaskFunc | None = None,
        param_lo: onp.Array1D[np.float64] | None = None,
        param_hi: onp.Array1D[np.float64] | None = None,
    ) -> None:
        self._max_iterations: int | None = max_iterations

        self._smoothing_func: _SmoothingFunc = create_gaussian_smoother(2.0) if smoothing_func is None else smoothing_func
        self._mask_func: _MaskFunc = create_sigmoid_mask(1.0, 40.0) if mask_func is None else mask_func

        self._param_lo: onp.Array1D[np.float64] = np.copy(param_lo if param_lo is not None else _DEFAULT_PARAM_LO).astype(
            np.float64
        )
        self._param_hi: onp.Array1D[np.float64] = np.copy(param_hi if param_hi is not None else _DEFAULT_PARAM_HI).astype(
            np.float64
        )

        # Check shapes
        if self._param_lo.shape != (6,) or self._param_hi.shape != (6,):
            msg = "Parameter bounds must each have shape (6,)."
            raise ValueError(msg)

        # Replace nans with default values
        self._param_lo[np.isnan(self._param_lo)] = _DEFAULT_PARAM_LO[np.isnan(self._param_lo)]
        self._param_hi[np.isnan(self._param_hi)] = _DEFAULT_PARAM_HI[np.isnan(self._param_hi)]

        # Validate bounds
        if not (np.all(np.isfinite(self._param_lo)) and np.all(np.isfinite(self._param_hi))):
            msg = "Parameter bounds must be finite."
            raise ValueError(msg)
        if np.any(self._param_lo > self._param_hi):
            msg = "Lower bounds must not exceed upper bounds."
            raise ValueError(msg)

    def fit_spiral(
        self,
        z: onp.Array1D[np.float64],
        vz: onp.Array1D[np.float64],
        z_bins: onp.Array1D[np.float64],
        vz_bins: onp.Array1D[np.float64],
        *,
        winding: Literal[-1, 1] | None = None,
        warm_start: onp.Array1D[np.float64] | None = None,
        seed: int | None = None,
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
                seed=seed,
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
        seed: int | None = None,
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
            seed=seed,
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
        seed: int | None = None,
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
                seed=seed,
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
        seed: int | None = None,
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
        seed : int | None
            The random seed for the multi-start draws, or ``None`` for no seed.
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

        rng = np.random.default_rng(seed)

        mask: Final[onp.Array2D[np.float64]] = self._mask_func(z_mesh, vz_mesh)

        best_background: onp.Array2D[np.float64] = initial_background
        # Initialize best_quality to -inf so that after the first model is found
        # we compare the model prediction likelihood rather than the background-only likelihood.
        best_quality: float = float("-inf")

        initial_model: PSpiralModel | None = None
        current_model: PSpiralModel | None = None
        best_model: PSpiralModel | None = None

        num_iterations: int = 0
        reason: FitTerminationReason = FitTerminationReason.ITERATION_LIMIT
        best_winding: Literal[-1, 1] | None = winding

        # Initialize warm start from caller-provided warm_start so we can reuse it.
        current_warm_start: onp.Array1D[np.float64] | None = warm_start

        # If num_components is None, compare 1- and 2-component fits using the
        # initial background only (improve_background=False) and pick the better
        # model. Continue the rest of the algorithm with that fixed choice.
        if num_components is None:
            res1 = self.fit_spiral_with_background(
                initial_density,
                initial_background,
                z_mesh,
                vz_mesh,
                winding=winding,
                warm_start=current_warm_start,
                seed=seed,
                num_components=1,
                improve_background=False,
            )
            res2 = self.fit_spiral_with_background(
                initial_density,
                initial_background,
                z_mesh,
                vz_mesh,
                winding=winding,
                warm_start=current_warm_start,
                seed=seed,
                num_components=2,
                improve_background=False,
            )

            # Both fits failed
            if isinstance(res1, FitFailure) and isinstance(res2, FitFailure):
                yield FitFailure(
                    FitFailureReason.NO_VALID_CANDIDATE, "No valid candidate found for either 1-component and 2-component fits."
                )
                return
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
                q1 = ln_likelihood(initial_density, res1.result.final_model.prediction(), mask)
                q2 = ln_likelihood(initial_density, res2.result.final_model.prediction(), mask)
                # BIC penalizes the larger model's additional parameters.
                k1 = 6
                k2 = 12
                num_particles = np.sum(initial_density)
                b1 = k1 * np.log(num_particles) - 2.0 * q1
                b2 = k2 * np.log(num_particles) - 2.0 * q2
                selected = res2.result if b2 < b1 else res1.result

            # Selection already found a valid fit; preserve it if refinement fails.
            initial_model = selected.final_model
            best_model = selected.final_model
            best_background = best_model.background
            best_quality = selected.lnl
            best_winding = best_model.winding
            num_components = best_model.parameters.shape[0]
            current_warm_start = best_model.to_array()
            if not improve_background:
                yield FitSuccess(selected)
                return

        while self._max_iterations is None or (num_iterations < self._max_iterations):
            num_iterations += 1

            # Number of 6-parameter components to fit
            param_count: int = num_components

            def wrap_winding_objective(current_winding: Literal[-1, 1]) -> _ObjectiveFunc:
                def _objective(parameters: onp.Array1D[np.float64]) -> float:
                    params = np.array(parameters, dtype=np.float64).reshape((param_count, 6))
                    model = PSpiralModel(params, z_mesh, vz_mesh, best_background, winding=current_winding)
                    return -ln_likelihood(initial_density, model.prediction(), mask)

                return _objective

            # Auto-select winding on first iteration if unset, then optimize for it.
            res: optimize.OptimizeResult
            if best_winding is None:
                pos_res = self._optimize_parameters(
                    wrap_winding_objective(1), rng=rng, warm_start=current_warm_start, param_count=param_count
                )
                neg_res = self._optimize_parameters(
                    wrap_winding_objective(-1), rng=rng, warm_start=current_warm_start, param_count=param_count
                )
                if np.isfinite(pos_res.fun) and (not np.isfinite(neg_res.fun) or pos_res.fun <= neg_res.fun):
                    best_winding = 1
                    res = pos_res
                else:
                    best_winding = -1
                    res = neg_res
            else:
                # Optimize for chosen winding.
                res = self._optimize_parameters(
                    wrap_winding_objective(best_winding), rng=rng, warm_start=current_warm_start, param_count=param_count
                )

            if not np.isfinite(res.fun):
                if best_model is None:
                    yield FitFailure(reason=FitFailureReason.NO_VALID_CANDIDATE, message="No valid candidate model was found.")
                    return
                else:
                    reason = FitTerminationReason.FAILED_REOPTIMIZATION
                    break

            best_params: onp.Array1D[np.float64] = np.array(res.x, dtype=np.float64)
            params = best_params.reshape((param_count, 6))
            current_model = PSpiralModel(params, z_mesh, vz_mesh, best_background, winding=best_winding)

            # Set the first model
            if initial_model is None:
                initial_model = current_model
                best_model = current_model
                best_quality = ln_likelihood(initial_density, initial_model.prediction(), mask)

            if not improve_background:
                best_model = current_model
                best_quality = -float(res.fun)
                reason = FitTerminationReason.FIXED_BACKGROUND
                break

            yield FitProgress(
                model=current_model,
                iteration=num_iterations,
                lnl=-float(res.fun),
            )

            # Update background
            current_perturbation = current_model.signal()
            new_background = self._smoothing_func(initial_density / current_perturbation)
            new_background: onp.Array2D[np.float64] = new_background / np.sum(new_background) * np.sum(initial_density)

            candidate_model = PSpiralModel(
                parameters=current_model.parameters,
                z_mesh=z_mesh,
                vz_mesh=vz_mesh,
                background=new_background,
                winding=best_winding,
                flattening_strength=current_model.flattening_strength,
            )
            candidate_quality = ln_likelihood(
                initial_density,
                candidate_model.prediction(),
                mask,
            )

            if not np.isfinite(candidate_quality):
                log.warning("Background refinement produced an invalid prediction.")
                reason = FitTerminationReason.INVALID_BACKGROUND_UPDATE
                break

            # Quality has degraded => we have converged
            if candidate_quality < best_quality:
                reason = FitTerminationReason.NO_IMPROVEMENT
                break

            # Update best parameters
            best_quality = candidate_quality
            best_background = candidate_model.background
            best_model = candidate_model
            current_warm_start = candidate_model.to_array()

        if initial_model is None or best_model is None:
            yield FitFailure(reason=FitFailureReason.NO_VALID_CANDIDATE, message="No valid model was ever found.")
            return
        yield FitSuccess(
            result=PSpiralFitResult(
                initial_model=initial_model,
                final_model=best_model,
                data=initial_density,
                num_iterations=num_iterations,
                max_iterations=self._max_iterations,
                lnl=best_quality,
                reason=reason,
            )
        )

    def _optimize_parameters(
        self,
        objective_func: _ObjectiveFunc,
        *,
        rng: np.random.Generator,
        warm_start: onp.Array1D[np.float64] | None,
        param_count: int = 1,
    ) -> optimize.OptimizeResult:
        # warm_start may be None or a flat vector of length 6 * param_count
        assert warm_start is None or (warm_start.ndim == 1 and len(warm_start) == 6 * param_count)
        base_bounds = list(zip(self._param_lo.tolist(), self._param_hi.tolist(), strict=True))
        bounds = base_bounds * param_count

        nfev = 0

        def counted_objective(parameters: onp.Array1D[np.float64]) -> float:
            nonlocal nfev
            nfev += 1
            return float(objective_func(parameters))

        res = optimize.differential_evolution(counted_objective, bounds=bounds, x0=warm_start, seed=rng)
        if not res.success:
            log.warning("Failed to find maximum likelihood: %s", res.message)
        return res


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
