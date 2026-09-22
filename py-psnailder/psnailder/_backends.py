"""The interface for the fitting backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, override

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal

    import numpy as np
    from optype import numpy as onp

    from .model import PSpiralModel


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


@dataclass(frozen=True)
class OptimizationResult:
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


def combine_diagnostics(*items: OptimizationDiagnostics) -> OptimizationDiagnostics:
    """Combine disjoint attempts into an immutable snapshot."""
    return OptimizationDiagnostics(
        message="; ".join(item.message for item in items),
        success=all(item.success for item in items),
        nfev=sum(item.nfev for item in items),
        nit=sum(item.nit for item in items),
    )


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


type BackendResult = FitSuccess | FitFailure
type BackendEvent = FitProgress | FitSuccess | FitFailure


@dataclass(frozen=True)
class FitRequest:
    initial_density: onp.Array2D[np.float64]
    initial_background: onp.Array2D[np.float64]
    z_mesh: onp.Array2D[np.float64]
    vz_mesh: onp.Array2D[np.float64]
    winding: Literal[-1, 1] | None = None
    warm_start: onp.Array1D[np.float64] | None = None
    rng: np.random.Generator | None = None
    num_components: int | None = None
    improve_background: bool = True


class SmoothConfig(ABC):
    @abstractmethod
    def kind(self) -> str: ...


@dataclass(frozen=True)
class GaussianSmoothConfig(SmoothConfig):
    z_scale: float = 2.0
    vz_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.z_scale <= 0.0:
            msg = "`z_scale` must be positive."
            raise ValueError(msg)

        if self.vz_scale <= 0.0:
            msg = "`vz_scale` must be positive."
            raise ValueError(msg)

    @override
    def kind(self) -> str:
        return "Gaussian"


class MaskConfig(ABC):
    @abstractmethod
    def kind(self) -> str: ...


@dataclass(frozen=True)
class SigmoidMaskConfig(MaskConfig):
    z_scale: float = 1.0
    vz_scale: float = 40.0

    def __post_init__(self) -> None:
        if self.z_scale <= 0.0:
            msg = "`z_scale` must be positive."
            raise ValueError(msg)

        if self.vz_scale <= 0.0:
            msg = "`vz_scale` must be positive."
            raise ValueError(msg)

    @override
    def kind(self) -> str:
        return "Sigmoid"


class FitBackend(Protocol):
    def fit(self, request: FitRequest) -> BackendResult: ...
    def fit_events(self, request: FitRequest) -> Iterator[BackendEvent]: ...
