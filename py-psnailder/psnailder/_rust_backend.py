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
from .model import PSpiralModel

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from optype import numpy as onp

    from ._internal import PSpiralFitResult as _RustFitResult
    from ._internal import PSpiralModel as _RustModel
    from .bounds import ParameterBounds


type _SmoothingFunc = Callable[[onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _MaskFunc = Callable[[onp.Array2D[np.float64], onp.Array2D[np.float64]], onp.Array2D[np.float64]]

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
        self._max_iterations: int | None = max_iterations
        self._atol: float = atol
        self._rtol: float = rtol
        _smooth_config: SmoothConfig = RustFitBackend._parse_smooth_func(smoothing_func)

        self._mask_func: _MaskFunc = RustFitBackend._parse_mask_func(mask_func)

        self._bounds: object = bounds
        self._rust_fitter: RustPSpiralFitter = RustPSpiralFitter(
            max_iterations=max_iterations,
            atol=atol,
            rtol=rtol,
        )

    @staticmethod
    def _parse_smooth_func(config: _SmoothingFunc | SmoothConfig | None) -> SmoothConfig:
        if isinstance(config, Callable):
            msg = "Python callbacks are not allowed for the rust backend's smoother."
            raise TypeError(msg)

        if isinstance(config, SmoothConfig):
            if isinstance(config, GaussianSmoothConfig):
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

    @override
    def fit(self, request: FitRequest) -> BackendResult:
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
        return FitSuccess(result=RustFitBackend._convert_result(res, request), diagnostics=RustFitBackend._rust_diagnostics())

    @override
    def fit_events(self, request: FitRequest) -> Iterator[BackendEvent]:
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
        yield FitSuccess(result=RustFitBackend._convert_result(res, request), diagnostics=RustFitBackend._rust_diagnostics())

    def _unsupported_reason(self, request: FitRequest) -> str | None:
        checks = (
            (self._bounds is not None, "Rust backend does not yet support ParameterBounds."),
            (request.num_components is not None, "Rust backend currently performs automatic component selection only."),
            (request.winding is not None, "Rust backend currently selects winding automatically only."),
            (request.warm_start is not None, "Rust backend does not yet support warm starts."),
            (request.rng is not None, "Rust backend accepts no numpy.random.Generator; seed support is not wired yet."),
            (not request.improve_background, "Rust binding does not yet expose fixed-background fitting."),
            (self._max_iterations == 0, "Rust binding currently requires at least one refinement iteration."),
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
