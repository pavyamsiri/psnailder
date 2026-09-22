"""Adapter between the Python fitting protocol and the native Rust fitter.

The adapter owns Python-specific configuration and converts it to native inputs:
the mask becomes a numeric array, while unsupported callbacks and bounds are
reported before fitting. The native fitter is constructed eagerly so extension
availability and constructor configuration fail at backend creation time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, override

import numpy as np
from scipy import special

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
    PSpiralFitResult,
    SigmoidMaskConfig,
    SmoothConfig,
)
from ._internal import PSpiralFitter as RustPSpiralFitter
from .bounds import Fixed, Interval, ParameterBounds
from .model import PSpiralModel

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from optype import numpy as onp

    from ._internal import PSpiralFitResult as _RustFitResult
    from ._internal import PSpiralModel as _RustModel


type _SmoothingFunc = Callable[[onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _MaskFunc = Callable[[onp.Array2D[np.float64], onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _RustBound = tuple[float, float]

log: Final[logging.Logger] = logging.getLogger(__name__)


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


class RustFitBackend(FitBackend):
    """Adapt the existing batch PyO3 fitter to the Python backend protocol."""

    def __init__(
        self,
        *,
        max_iterations: int | None,
        atol: float,
        rtol: float,
        smoothing_func: _SmoothingFunc | SmoothConfig | None,
        mask_func: _MaskFunc | MaskConfig | None,
        bounds: ParameterBounds | Sequence[ParameterBounds] | None,
    ) -> None:
        smooth_config = RustFitBackend._parse_smooth_func(smoothing_func)

        self._mask_func: _MaskFunc = RustFitBackend._parse_mask_func(mask_func)
        self._bounds: ParameterBounds | Sequence[ParameterBounds] = bounds if bounds is not None else ParameterBounds()
        self._rust_fitter: RustPSpiralFitter = RustPSpiralFitter(
            max_iterations=max_iterations,
            atol=atol,
            rtol=rtol,
            sigma_z=smooth_config.z_scale,
            sigma_vz=smooth_config.vz_scale,
            bounds=self._rust_bounds_components(),
        )

    @staticmethod
    def _parse_smooth_func(config: _SmoothingFunc | SmoothConfig | None) -> GaussianSmoothConfig:
        if isinstance(config, Callable):
            msg = "Python callbacks are not allowed for the rust backend's smoother."
            raise TypeError(msg)

        if isinstance(config, SmoothConfig):
            if isinstance(config, GaussianSmoothConfig):
                if config.z_scale != config.vz_scale:
                    msg = "The Rust backend currently requires equal Gaussian smoothing scales."
                    raise ValueError(msg)
                return config
            msg = f"Unsupported smoothing config: {config}"
            raise ValueError(msg)

        return GaussianSmoothConfig()

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

    def _component_bounds(self, num_components: int) -> tuple[ParameterBounds, ...]:
        """Apply Python's broadcasting/prefix rules before native conversion."""
        if num_components < 1:
            msg = "Component bounds require a positive component count."
            raise ValueError(msg)
        if isinstance(self._bounds, ParameterBounds):
            return (self._bounds,) * num_components
        if num_components > len(self._bounds):
            msg = "Not enough bounds for the requested component count."
            raise ValueError(msg)
        return tuple(self._bounds[:num_components])

    def _rust_bounds(self, num_components: int) -> tuple[_RustBound, ...]:
        """Serialize named Python bounds in Rust's component-major order.

        Fixed values and zero-width intervals are represented by equal endpoints;
        the Rust layout treats those entries as fixed and removes them from the
        optimization vector.
        """
        serialized: list[_RustBound] = []
        for component in self._component_bounds(num_components):
            serialized.extend(RustFitBackend._rust_bounds_for_component(component))
        return tuple(serialized)

    def _rust_bounds_components(self) -> tuple[tuple[_RustBound, ...], ...]:
        """Serialize configured bounds for the native one-/two-component fitter."""
        if isinstance(self._bounds, ParameterBounds):
            return (self._rust_bounds(1),)
        if not self._bounds:
            msg = "At least one component bound set is required."
            raise ValueError(msg)
        if len(self._bounds) > 2:
            msg = "The Rust backend supports at most two component bound sets."
            raise ValueError(msg)
        return tuple(RustFitBackend._rust_bounds_for_component(component) for component in self._bounds)

    @staticmethod
    def _rust_bounds_for_component(bounds: ParameterBounds) -> tuple[_RustBound, ...]:
        serialized: list[_RustBound] = []
        for bound in (bounds.alpha, bounds.b, bounds.c, bounds.theta0, bounds.scale_factor, bounds.rho):
            if isinstance(bound, Fixed):
                serialized.append((bound.value, bound.value))
            else:
                assert isinstance(bound, Interval)
                serialized.append((bound.lower, bound.upper))
        return tuple(serialized)

    @override
    def fit_batch(self, requests: Sequence[FitRequest], *, workers: int | None = None) -> list[BackendResult]:
        """Prepare grids, then fit the supported items in one native batch."""
        if workers is not None and (type(workers) is not int or workers < 1):
            msg = "workers must be a positive integer or None."
            raise ValueError(msg)
        outcomes: dict[int, BackendResult] = {}
        indices: list[int] = []
        inputs: list[
            tuple[
                onp.Array1D[np.float64],
                onp.Array1D[np.float64],
                onp.Array1D[np.float64],
                onp.Array1D[np.float64],
                onp.Array1D[np.float64],
                tuple[int, int],
            ]
        ] = []
        for index, request in enumerate(requests):
            unsupported = self._unsupported_reason(request)
            if unsupported is not None:
                outcomes[index] = self._failure(unsupported)
                continue
            grids = (request.initial_density, request.initial_background, request.z_mesh, request.vz_mesh)
            shape = request.initial_density.shape
            if any(grid.ndim != 2 or grid.shape != shape for grid in grids):
                msg = "Batch grids must share a 2D shape within each input."
                raise ValueError(msg)
            mask = self._mask_func(request.z_mesh, request.vz_mesh)
            if mask.shape != shape:
                msg = "Batch mask must match the grid shape."
                raise ValueError(msg)
            if any(not np.all(np.isfinite(grid)) for grid in (*grids, mask)):
                msg = "Batch grids and masks must be finite."
                raise ValueError(msg)
            if np.any(mask < 0) or not np.any(mask > 0):
                msg = "Batch masks must have nonnegative weights with at least one positive weight."
                raise ValueError(msg)
            if any(np.any(grid < 0) or not np.isfinite(grid.sum()) or grid.sum() <= 0 for grid in grids[:2]):
                outcomes[index] = self._failure("Counts and background must have positive finite totals and nonnegative values.")
                continue
            inputs.append(
                (
                    request.initial_density.ravel(),
                    request.initial_background.ravel(),
                    mask.ravel(),
                    request.z_mesh.ravel(),
                    request.vz_mesh.ravel(),
                    shape,
                )
            )
            indices.append(index)
        if inputs:
            native_results = self._rust_fitter.fit_batch(inputs, workers=workers)
            for index, result in zip(indices, native_results, strict=True):
                outcomes[index] = FitSuccess(
                    result=self._convert_result(result, requests[index]),
                    diagnostics=self._rust_diagnostics(result),
                )
        return [outcomes[index] for index in range(len(requests))]

    @override
    def fit(self, request: FitRequest) -> BackendResult:
        unsupported = self._unsupported_reason(request)
        if unsupported is not None:
            return self._failure(unsupported)
        initial_density = request.initial_density
        initial_background = request.initial_background
        z_mesh = request.z_mesh
        vz_mesh = request.vz_mesh
        mask = self._mask_func(z_mesh, vz_mesh)
        initial_density = request.initial_density
        shape = initial_density.shape

        res = self._rust_fitter.fit_spiral_with_background(
            initial_density.flatten(),
            initial_background.flatten(),
            mask.flatten(),
            z_mesh.flatten(),
            vz_mesh.flatten(),
            shape=shape,
        )
        return FitSuccess(result=RustFitBackend._convert_result(res, request), diagnostics=RustFitBackend._rust_diagnostics(res))

    @override
    def fit_events(self, request: FitRequest) -> Iterator[BackendEvent]:
        """Yield each accepted Rust refinement checkpoint and the terminal result."""
        unsupported = self._unsupported_reason(request)
        if unsupported is not None:
            yield self._failure(unsupported)
            return
        initial_density = request.initial_density
        initial_background = request.initial_background
        z_mesh = request.z_mesh
        vz_mesh = request.vz_mesh
        mask = self._mask_func(z_mesh, vz_mesh)
        initial_density = request.initial_density
        shape = initial_density.shape

        checkpoints = self._rust_fitter.fit_spiral_with_background_events(
            initial_density.flatten(),
            initial_background.flatten(),
            mask.flatten(),
            z_mesh.flatten(),
            vz_mesh.flatten(),
            shape=shape,
        )
        for checkpoint in checkpoints:
            result = RustFitBackend._convert_result(checkpoint, request)
            diagnostics = RustFitBackend._rust_diagnostics(checkpoint)
            if checkpoint.terminal:
                yield FitSuccess(result=result, diagnostics=diagnostics)
            else:
                yield FitProgress(
                    model=result.final_model,
                    iteration=result.num_iterations,
                    lnl=result.lnl,
                    diagnostics=diagnostics,
                )

    def _unsupported_reason(self, request: FitRequest) -> str | None:
        checks = (
            (request.num_components is not None, "Rust backend currently performs automatic component selection only."),
            (request.winding is not None, "Rust backend currently selects winding automatically only."),
            (request.warm_start is not None, "Rust backend does not yet support warm starts."),
            (not request.improve_background, "Rust binding does not yet expose fixed-background fitting."),
        )
        return next((message for condition, message in checks if condition), None)

    @staticmethod
    def _convert_result(rust_result: _RustFitResult, request: FitRequest) -> PSpiralFitResult:
        shape = request.initial_density.shape
        initial_background = np.asarray(rust_result.initial_background, dtype=np.float64).reshape(shape)
        final_background = np.asarray(rust_result.final_background, dtype=np.float64).reshape(shape)
        initial_model = RustFitBackend._convert_model(rust_result.initial_model, request, initial_background)
        final_model = RustFitBackend._convert_model(rust_result.final_model, request, final_background)
        reason = FitTerminationReason.CONVERGED if rust_result.converged else FitTerminationReason.ITERATION_LIMIT
        return PSpiralFitResult(
            initial_model=initial_model,
            final_model=final_model,
            data=np.asarray(rust_result.data, dtype=np.float64).reshape(shape),
            num_iterations=rust_result.num_iterations,
            max_iterations=rust_result.max_iterations,
            lnl=rust_result.lnl,
            reason=reason,
        )

    @staticmethod
    def _convert_model(rust_model: _RustModel, request: FitRequest, background: onp.Array2D[np.float64]) -> PSpiralModel:
        components = rust_model.components
        parameters = np.array(
            [
                [component.alpha, component.b, component.c, component.theta0, component.scale_factor, component.rho]
                for component in components
            ],
            dtype=np.float64,
        )
        windings: list[int] = [int(component.winding) for component in components]
        assert len(windings) >= 1, "Should always be at least 1 component."
        assert all(winding == windings[0] for winding in windings), "windings should be the same across all components."
        winding = int(windings[0])
        assert winding in (-1, 1), "Winding should be either -1 or 1."
        return PSpiralModel(parameters, request.z_mesh, request.vz_mesh, background, winding=winding)

    @staticmethod
    def _rust_diagnostics(rust_result: _RustFitResult) -> OptimizationDiagnostics:
        return OptimizationDiagnostics(
            message="Rust TikTak/Nelder-Mead optimization completed.",
            success=True,
            nfev=rust_result.nfev,
            nit=rust_result.nit,
        )

    @staticmethod
    def _failure(message: str) -> FitFailure:
        return FitFailure(
            reason=FitFailureReason.NO_VALID_CANDIDATE,
            message=message,
            diagnostics=OptimizationDiagnostics(
                message="Rust backend did not produce optimizer diagnostics.", success=False, nfev=0, nit=0
            ),
        )
