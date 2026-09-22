"""Adapter for the current PyO3 Rust fitting binding.

This is intentionally a small compatibility backend. The Rust binding currently
accepts flattened, already-binned arrays and exposes a batch result only. Bounds,
warm starts, Python callbacks, and per-iteration events will be added when the
Rust core APIs support them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from ._backends import (
    BackendEvent,
    BackendResult,
    FitBackend,
    FitFailure,
    FitFailureReason,
    FitRequest,
    FitSuccess,
    FitTerminationReason,
    GaussianSmoothConfig,
    OptimizationDiagnostics,
    PSpiralFitResult,
    SigmoidMaskConfig,
)
from .model import PSpiralModel

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from optype import numpy as onp

    from ._internal import PSpiralFitResult as _RustFitResult
    from ._internal import PSpiralFitter as _RustFitter
    from ._internal import PSpiralModel as _RustModel
    from .bounds import ParameterBounds


class RustFitBackend(FitBackend):
    """Adapt the existing batch PyO3 fitter to the Python backend protocol."""

    def __init__(
        self,
        *,
        max_iterations: int | None,
        atol: float,
        rtol: float,
        smoothing_func: object,
        mask_func: object,
        bounds: ParameterBounds | Sequence[ParameterBounds] | None,
    ) -> None:
        self._max_iterations: int | None = max_iterations
        self._atol: float = atol
        self._rtol: float = rtol
        self._smoothing_func: object = smoothing_func
        self._mask_func: object = mask_func
        self._bounds: object = bounds
        self._rust_fitter: _RustFitter | None = None

    @override
    def fit(self, request: FitRequest) -> BackendResult:
        return self._fit_once(request)

    @override
    def fit_events(self, request: FitRequest) -> Iterator[BackendEvent]:
        # The current binding is batch-only. Preserve the event API by yielding
        # its converted terminal outcome; a native Rust iterator can replace
        # this method without changing PSpiralFitter.
        yield self._fit_once(request)

    def _fit_once(self, request: FitRequest) -> BackendResult:
        unsupported = self._unsupported_reason(request)
        if unsupported is not None:
            return self._failure(unsupported)

        try:
            rust_fitter = self._get_rust_fitter()
            mask = self._make_mask(request.z_mesh, request.vz_mesh)
            shape = request.initial_density.shape
            rust_result = rust_fitter.fit_spiral_with_background(
                np.asarray(request.initial_density, dtype=np.float64).ravel(),
                np.asarray(request.initial_background, dtype=np.float64).ravel(),
                mask.ravel(),
                np.asarray(request.z_mesh, dtype=np.float64).ravel(),
                np.asarray(request.vz_mesh, dtype=np.float64).ravel(),
                shape,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            return self._failure(f"Rust extension is unavailable: {exc}")
        except (TypeError, ValueError) as exc:
            return self._failure(f"Rust fitting failed: {exc}")

        return FitSuccess(self._convert_result(rust_result, request), self._rust_diagnostics())

    def _get_rust_fitter(self) -> _RustFitter:
        if self._rust_fitter is None:
            from . import _internal  # noqa: PLC0415 -- load the optional extension lazily.

            smoothing_sigma = self._smoothing_sigma()
            self._rust_fitter = _internal.PSpiralFitter(
                max_iterations=self._max_iterations,
                smoothing_sigma=smoothing_sigma,
            )
        return self._rust_fitter

    def _make_mask(self, z_mesh: onp.Array2D[np.float64], vz_mesh: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        config = self._mask_func
        if config is None:
            config = SigmoidMaskConfig()
        if not isinstance(config, SigmoidMaskConfig):
            msg = "The Rust backend requires SigmoidMaskConfig; Python mask callbacks are unsupported."
            raise TypeError(msg)
        z_scale, vz_scale = config.z_scale, config.vz_scale
        mask = -1.0 / (1.0 + np.exp(-(np.square(z_mesh / z_scale) + np.square(vz_mesh / vz_scale) - 1.0))) + 1.0
        return np.asarray(mask, dtype=np.float64)

    def _smoothing_sigma(self) -> float:
        config = self._smoothing_func
        if config is None:
            return 2.0
        if isinstance(config, GaussianSmoothConfig):
            if config.z_scale != config.vz_scale:
                msg = "The current Rust backend requires equal Gaussian smoothing scales."
                raise ValueError(msg)
            return config.z_scale
        msg = "The Rust backend requires GaussianSmoothConfig; Python smoothing callbacks are unsupported."
        raise TypeError(msg)

    def _unsupported_reason(self, request: FitRequest) -> str | None:
        checks = (
            (self._bounds is not None, "Rust backend does not yet support ParameterBounds."),
            (request.num_components is not None, "Rust backend currently performs automatic component selection only."),
            (request.winding is not None, "Rust backend currently selects winding automatically only."),
            (request.warm_start is not None, "Rust backend does not yet support warm starts."),
            (request.rng is not None, "Rust backend accepts no numpy.random.Generator; seed support is not wired yet."),
            (not request.improve_background, "Rust binding does not yet expose fixed-background fitting."),
            (self._max_iterations == 0, "Rust binding currently requires at least one refinement iteration."),
            (self._atol != 0.0 or self._rtol != 0.0, "Rust binding does not yet expose refinement tolerances."),
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
        winding = components[0].winding if components else 1
        return PSpiralModel(parameters, request.z_mesh, request.vz_mesh, background, winding=winding)

    @staticmethod
    def _rust_diagnostics() -> OptimizationDiagnostics:
        return OptimizationDiagnostics(
            message="Rust optimizer diagnostics are not exposed by the current binding.",
            success=True,
            nfev=0,
            nit=0,
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
