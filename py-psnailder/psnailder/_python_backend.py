"""The python fitting backend."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Final, Literal, override

import numpy as np
from optype import numpy as onp
from scipy import ndimage, optimize, special

from psnailder._likelihood_utils import ln_likelihood

from ._backends import (
    BackendEvent,
    BackendResult,
    FitBackend,
    FitFailure,
    FitFailureReason,
    FitProgress,
    FitRequest,
    FitSuccess,
    FitTerminationReason,
    GaussianSmoothConfig,
    MaskConfig,
    OptimizationDiagnostics,
    OptimizationResult,
    PSpiralFitResult,
    SigmoidMaskConfig,
    SmoothConfig,
    combine_diagnostics,
)
from .bounds import ParameterBounds
from .model import PSpiralModel
from .param_layout import ParameterLayout

type _ObjectiveFunc = Callable[[onp.Array1D[np.float64]], onp.ToFloat]
type _SmoothingFunc = Callable[[onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _MaskFunc = Callable[[onp.Array2D[np.float64], onp.Array2D[np.float64]], onp.Array2D[np.float64]]

log: Final[logging.Logger] = logging.getLogger(__name__)


def create_gaussian_smoother(z_scale: float, vz_scale: float) -> _SmoothingFunc:
    """Return a function that applies Gaussian smoothing to a 2D array.

    Parameters
    ----------
    z_scale : float
        The smoothing width in units of pixels for the z-axis (second axis/columns).
    vz_scale : float
        The smoothing width in units of pixels for the Vz-axis (first axis/rows).

    Returns
    -------
    func : Callable[[Array2D[f64]], Array2D[f64]]
        The Gaussian smoothing function.

    """

    def _func(arr: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        return ndimage.gaussian_filter(arr, sigma=(vz_scale, z_scale))

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


class PythonFitBackend(FitBackend):
    def __init__(
        self,
        max_iterations: int | None = 50,
        atol: float = 0.0,
        rtol: float = 0.0,
        smoothing_func: _SmoothingFunc | SmoothConfig | None = None,
        mask_func: _MaskFunc | MaskConfig | None = None,
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
        if max_iterations is not None and (
            isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0  # pyright: ignore[reportUnnecessaryIsInstance]
        ):
            msg = "max_iterations must be a nonnegative integer or None."
            raise ValueError(msg)
        self._max_iterations: int | None = max_iterations

        for name, tolerance in (("atol", atol), ("rtol", rtol)):
            if not np.isfinite(tolerance) or tolerance < 0:
                msg = f"{name} must be finite and nonnegative."
                raise ValueError(msg)

        self._atol: float = float(atol)
        self._rtol: float = float(rtol)

        self._smoothing_func: _SmoothingFunc = PythonFitBackend._parse_smooth_func(smoothing_func)
        self._mask_func: _MaskFunc = PythonFitBackend._parse_mask_func(mask_func)

        # A single object broadcasts; explicit sequences select an ordered prefix.
        self._bounds: ParameterBounds | Sequence[ParameterBounds] = bounds if bounds is not None else ParameterBounds()

    @staticmethod
    def _parse_smooth_func(config: _SmoothingFunc | SmoothConfig | None) -> _SmoothingFunc:
        if isinstance(config, Callable):
            return config

        if isinstance(config, SmoothConfig):
            if isinstance(config, GaussianSmoothConfig):
                return create_gaussian_smoother(config.z_scale, config.vz_scale)
            msg = f"Unsupported smoothing config: {config}"
            raise ValueError(msg)

        return create_gaussian_smoother(2.0, 2.0)

    @staticmethod
    def _parse_mask_func(config: _MaskFunc | MaskConfig | None) -> _MaskFunc:
        if isinstance(config, Callable):
            return config

        if isinstance(config, MaskConfig):
            if isinstance(config, SigmoidMaskConfig):
                return create_sigmoid_mask(config.z_scale, config.vz_scale)
            msg = f"Unsupported masking config: {config}"
            raise ValueError(msg)

        return create_sigmoid_mask(1.0, 40.0)

    def _component_bounds(self, num_components: int) -> Sequence[ParameterBounds]:
        if num_components < 1:
            msg = "Component bounds require a positive component count."
            raise ValueError(msg)
        if isinstance(self._bounds, ParameterBounds):
            return (self._bounds,) * num_components
        if num_components > len(self._bounds):
            msg = "Not enough bounds for the requested component count."
            raise ValueError(msg)
        return self._bounds[:num_components]

    @override
    def fit_batch(self, requests: Sequence[FitRequest], *, workers: int | None = None) -> list[BackendResult]:
        """Fit independently in threads and collect terminal outcomes in input order.

        Each request owns its fitting state. Custom mask and smoothing callbacks
        must support concurrent calls; use workers=1 for serial callbacks.
        Exceptions propagate as they do for a single fit.
        """
        if workers is not None and (type(workers) is not int or workers < 1):
            msg = "workers must be a positive integer or None."
            raise ValueError(msg)
        if not requests:
            return []
        if workers == 1:
            return [self.fit(request) for request in requests]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self.fit, requests))

    @override
    def fit(
        self,
        request: FitRequest,
    ) -> BackendResult:
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
        val = _get_value_from_iter(self.fit_events(request))
        assert val is not None
        assert isinstance(val, FitSuccess | FitFailure)
        return val

    @override
    def fit_events(self, request: FitRequest) -> Iterator[BackendEvent]:
        initial_density: onp.Array2D[np.float64] = request.initial_density
        initial_background: onp.Array2D[np.float64] = request.initial_background
        z_mesh: onp.Array2D[np.float64] = request.z_mesh
        vz_mesh: onp.Array2D[np.float64] = request.vz_mesh
        winding: Literal[-1, 1] | None = request.winding
        warm_start: onp.Array1D[np.float64] | None = request.warm_start
        rng: np.random.Generator | None = request.rng
        num_components: int | None = request.num_components
        improve_background: bool = request.improve_background

        # Yield progress snapshots, then exactly one terminal success or failure.
        # If given a warm start, the number of components must be explicitly set.
        self._validate_fit_options(num_components, winding, warm_start)

        if rng is None:
            rng = np.random.default_rng()

        self._validate_array_inputs(initial_density, initial_background, z_mesh, vz_mesh)

        mask: Final[onp.Array2D[np.float64]] = self._mask_func(z_mesh, vz_mesh)
        self._validate_callback_shape("mask_func", mask, initial_density.shape)
        if not np.all(np.isfinite(mask)) or np.any(mask < 0) or not np.any(mask > 0):
            msg = "mask_func must return finite, nonnegative weights with at least one positive weight."
            raise ValueError(msg)
        if not self._valid_background(initial_density) or not self._valid_background(initial_background):
            yield self._unusable_data("Counts and background must have positive finite totals.")
            return

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

    @staticmethod
    def _unusable_data(message: str) -> FitFailure:
        return FitFailure(
            FitFailureReason.NO_VALID_CANDIDATE,
            message,
            OptimizationDiagnostics(message="No optimization took place.", success=False, nfev=0, nit=0),
        )

    def _validate_fit_options(
        self,
        num_components: int | None,
        winding: int | None,
        warm_start: onp.Array1D[np.float64] | None,
    ) -> None:
        if num_components is not None and (
            isinstance(num_components, bool) or not isinstance(num_components, (int, np.integer)) or num_components not in (1, 2)  # pyright: ignore[reportUnnecessaryIsInstance]
        ):
            msg = "`num_components` must be 1 or 2, or `None`; check component bounds."
            raise ValueError(msg)
        if winding is not None and (
            isinstance(winding, bool) or not isinstance(winding, (int, np.integer)) or winding not in (-1, 1)  # pyright: ignore[reportUnnecessaryIsInstance]
        ):
            msg = "winding must be -1 or 1, or None."
            raise ValueError(msg)
        if warm_start is not None and num_components is None:
            msg = "Can not use warm start if the number of component is not set."
            raise ValueError(msg)
        selected_bounds = self._component_bounds(num_components if num_components is not None else 2)
        if warm_start is not None:
            warm_start = ParameterLayout.from_bounds(selected_bounds).pack(warm_start)

    @staticmethod
    def _validate_callback_shape(name: str, value: object, shape: tuple[int, int]) -> None:
        if not isinstance(value, np.ndarray) or value.shape != shape:
            msg = f"{name} must return a 2D array with shape {shape}."
            raise ValueError(msg)
        if value.dtype.kind not in "biuf":
            msg = f"{name} must return a real numeric array."
            raise ValueError(msg)

    @staticmethod
    def _valid_background(background: onp.Array2D[np.float64]) -> bool:
        with np.errstate(over="ignore", invalid="ignore"):
            total = np.sum(background)
        return bool(np.all(np.isfinite(background)) and np.all(background >= 0) and np.isfinite(total) and total > 0)

    @staticmethod
    def _validate_array_inputs(
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
    ) -> None:
        # Validate dimensionality of arrays
        if initial_density.size == 0:
            msg = "Input maps must be nonempty."
            raise ValueError(msg)
        for name, ndim in (
            ("initial_density", initial_density.ndim),
            ("initial_background", initial_background.ndim),
            ("z_mesh", z_mesh.ndim),
            ("vz_mesh", vz_mesh.ndim),
        ):
            if ndim != 2:
                msg = f"`{name}` must be a 2D array."
                raise ValueError(msg)
        # Validate that arrays are the same shape
        common_shape: tuple[int, int] = initial_density.shape
        for name, shape in (
            ("initial_density", initial_density.shape),
            ("initial_background", initial_background.shape),
            ("z_mesh", z_mesh.shape),
            ("vz_mesh", vz_mesh.shape),
        ):
            if shape != common_shape:
                msg = f"`{name}` was expected to have shape {common_shape} but was {shape}."
                raise ValueError(msg)
        # Validate coordinates are purely finite
        for name, arr in (
            ("z_mesh", z_mesh),
            ("vz_mesh", vz_mesh),
        ):
            if np.any(~np.isfinite(arr)):
                msg = f"{name} has non-finite coordinates which is not allowed."
                raise ValueError(msg)
        # Validate counts and background are purely finite and non-negative
        for name, arr in (
            ("initial_density", initial_density),
            ("initial_background", initial_background),
        ):
            if np.any(~np.isfinite(arr)):
                msg = f"{name} has non-finite counts which is not allowed."
                raise ValueError(msg)
            if np.any(arr < 0.0):
                msg = f"{name} has negative counts which is not allowed."
                raise ValueError(msg)

    def _optimize_parameters(
        self,
        objective_func: _ObjectiveFunc,
        *,
        bounds: optimize.Bounds,
        rng: np.random.Generator,
        guess: onp.Array1D[np.float64] | None,
    ) -> OptimizationResult:
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
            return OptimizationResult(
                parameters=parameters,
                cost=cost,
                success=bool(np.isfinite(cost)),  # pyright: ignore[reportAny]
                nfev=1,
                nit=0,
                message="All parameters fixed; evaluated objective once.",
            )

        res = optimize.differential_evolution(objective, bounds=bounds, x0=guess, rng=rng)
        if not res.success:
            log.warning("Optimization did not converge: %s", res.message)
        return OptimizationResult(
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
    ) -> BackendResult:
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
                combine_diagnostics(res1.diagnostics, res2.diagnostics),
            )
        # 1-component fit succeeded while 2-component fit failed
        if isinstance(res1, FitSuccess) and isinstance(res2, FitFailure):
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
            b1 = k1 * np.log(num_particles) - 2.0 * q1  # pyright: ignore[reportAny]
            b2 = k2 * np.log(num_particles) - 2.0 * q2  # pyright: ignore[reportAny]
            selected = res2.result if b2 < b1 else res1.result

        return FitSuccess(
            selected,
            diagnostics=combine_diagnostics(res1.diagnostics, res2.diagnostics),
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
    ) -> BackendResult:
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

        res: OptimizationResult
        chosen_winding: Literal[-1, 1]
        if winding is None:
            pos_res = self._optimize_parameters(wrap_winding_objective(1), rng=rng, guess=free_guess, bounds=bounds)
            neg_res = self._optimize_parameters(wrap_winding_objective(-1), rng=rng, guess=free_guess, bounds=bounds)
            diagnostics = combine_diagnostics(pos_res.diagnostics, neg_res.diagnostics)
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
            self._validate_callback_shape("smoothing_func", new_background, initial_density.shape)

            if not self._valid_background(new_background):
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
            diagnostics = combine_diagnostics(diagnostics, candidate.diagnostics)

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


def _get_value_from_iter[T](it: Iterator[T]) -> T | None:
    """Unwrap last yield value from iterator.

    Parameters
    ----------
    it : Iterator[T]
        The iterator.

    Returns
    -------
    val : T | None
        The last yielded value or ``None`` if the iterator is empty.

    """
    val: T | None = None
    for inner in it:
        val = inner
    return val
