"""Experimental Python bootstrap uncertainty with bounded local refits.

Results condition on the selected winding, component count, and local solution.
Use the original fitter configuration; fitted results do not retain its provenance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._backends import FitFailure, FitTerminationReason, OptimizationDiagnostics
from ._python_backend import PythonFitBackend
from .bounds import Interval, ParameterBounds
from .param_layout import ParameterLayout

if TYPE_CHECKING:
    from optype import numpy as onp

    from ._backends import BackendResult, PSpiralFitResult
    from .fit import PSpiralFitter

__all__ = ["BootstrapReplicate", "BootstrapResult", "BootstrapSamples", "bootstrap_uncertainty"]


@dataclass(frozen=True, kw_only=True)
class BootstrapSamples:
    """Original paired samples and bin edges, retained by reference.

    ``improve_background`` must match the original fit. Arrays must not change
    during execution. Include out-of-grid stars if they contributed to its KDE.
    """

    z: onp.Array1D[np.float64]
    vz: onp.Array1D[np.float64]
    z_bins: onp.Array1D[np.float64]
    vz_bins: onp.Array1D[np.float64]
    improve_background: bool = True


@dataclass(frozen=True)
class BootstrapReplicate:
    """One requested draw, including failed draws and optimizer diagnostics."""

    index: int
    parameters: onp.Array1D[np.float64] | None
    diagnostics: OptimizationDiagnostics
    reason: str
    count_total: int


@dataclass(frozen=True)
class BootstrapResult:
    """Conditional bootstrap summary in flattened component-major parameter order.

    ``parameters`` has one row per requested replicate; failures are NaN rows.
    Angles are unwrapped around ``reference`` and exchangeable components aligned.
    Covariance uses accepted replicates and ddof=1. Intervals are percentile
    intervals, not calibrated guarantees. Summaries are NaN with fewer than two
    accepted draws. Inspect ``warnings`` and ``replicates`` before using them.
    """

    reference: onp.Array1D[np.float64]
    parameters: onp.Array2D[np.float64]
    covariance: onp.Array2D[np.float64]
    standard_errors: onp.Array1D[np.float64]
    intervals: onp.Array2D[np.float64]
    bias: onp.Array1D[np.float64]
    replicates: tuple[BootstrapReplicate, ...]
    method: Literal["samples", "parametric_counts"]
    confidence_level: float
    seed: int
    maxiter: int
    warnings: tuple[str, ...]

    @property
    def n_successful(self) -> int:
        """Number of accepted local refits."""
        return sum(item.parameters is not None for item in self.replicates)


def _validate_samples(samples: BootstrapSamples, result: PSpiralFitResult) -> None:
    if (
        samples.z.ndim != 1
        or samples.vz.shape != samples.z.shape
        or samples.z.size < 2
        or not np.all(np.isfinite(samples.z))
        or not np.all(np.isfinite(samples.vz))
    ):
        msg = "Samples must be finite paired 1D arrays with at least two observations."
        raise ValueError(msg)
    for edges in (samples.z_bins, samples.vz_bins):
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            msg = "Bin edges must be finite and strictly increasing."
            raise ValueError(msg)
    counts, _, _ = np.histogram2d(samples.z, samples.vz, bins=(samples.z_bins, samples.vz_bins))
    z_grid, vz_grid = np.meshgrid(
        (samples.z_bins[:-1] + samples.z_bins[1:]) / 2,
        (samples.vz_bins[:-1] + samples.vz_bins[1:]) / 2,
    )
    model = result.final_model
    if not (
        np.array_equal(counts.T, result.data) and np.array_equal(z_grid, model.z_mesh) and np.array_equal(vz_grid, model.vz_mesh)
    ):
        msg = "Samples and bins must reproduce the original fit's count map and grid."
        raise ValueError(msg)


def _local_bounds(bounds: ParameterBounds, phase: float) -> ParameterBounds:
    """Center full-period angular intervals on the reference for local search."""
    theta = bounds.theta0
    if isinstance(theta, Interval) and theta.upper - theta.lower >= 2 * np.pi - 1e-12:
        theta = Interval(phase - np.pi, phase + np.pi)
    return ParameterBounds(
        alpha=bounds.alpha,
        b=bounds.b,
        c=bounds.c,
        theta0=theta,
        scale_factor=bounds.scale_factor,
        rho=bounds.rho,
    )


def _aligned(
    values: onp.Array1D[np.float64],
    reference: onp.Array1D[np.float64],
    scale: onp.Array1D[np.float64],
    *,
    exchangeable: bool,
) -> onp.Array1D[np.float64]:
    def unwrap(candidate: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]:
        candidate = candidate.copy()
        candidate[3::6] = reference[3::6] + (candidate[3::6] - reference[3::6] + np.pi) % (2 * np.pi) - np.pi
        return candidate

    direct = unwrap(values)
    if exchangeable:
        swapped = unwrap(values.reshape(2, 6)[::-1].flatten())
        if np.sum(np.square((swapped - reference) / scale)) < np.sum(np.square((direct - reference) / scale)):
            return swapped
    return direct


def bootstrap_uncertainty(
    fitter: PSpiralFitter,
    result: PSpiralFitResult,
    *,
    samples: BootstrapSamples | None = None,
    n_resamples: int = 200,
    confidence_level: float = 0.95,
    seed: int = 0,
    workers: int = 1,
    maxiter: int = 500,
) -> BootstrapResult:
    """Bootstrap an existing fit using its original Python fitter configuration.

    With ``samples``, resample paired stars and rebuild their KDE, then repeat
    the specified refinement policy. Without samples, simulate multinomial
    counts from the fitted prediction and hold its background shape fixed.
    Winding and component count remain fixed in both cases.

    Each replicate starts at the original estimate and uses scaled L-BFGS-B.
    No global searches or retries are performed in this experimental version.
    Failed/nonconverged draws remain in the result and are excluded from its
    explicitly qualified summaries. The original fitter and fit are unchanged.

    Custom callbacks must be thread-safe for ``workers > 1``. Reusing ``seed``
    preserves each draw when increasing ``n_resamples`` or changing worker count.
    Only the Python backend is supported. Callers must supply the same bounds,
    mask, smoothing, and refinement configuration used for the original fit.
    """
    backend = fitter._backend  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] -- package-internal integration.
    if not isinstance(backend, PythonFitBackend):
        msg = "Bootstrap uncertainty currently supports only the Python backend."
        raise NotImplementedError(msg)
    for name, value in (("n_resamples", n_resamples), ("workers", workers), ("maxiter", maxiter)):
        if type(value) is not int or value < 1:
            msg = f"{name} must be a positive integer."
            raise ValueError(msg)
    if type(seed) is not int or seed < 0:
        msg = "seed must be a nonnegative integer."
        raise ValueError(msg)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        msg = "confidence_level must be between zero and one."
        raise ValueError(msg)

    model = result.final_model
    reference = model.parameters.flatten().copy()
    original_bounds = backend._component_bounds(model.num_components)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] -- package-internal integration.
    layout = ParameterLayout.from_bounds(original_bounds)
    if not np.allclose(layout.unpack(layout.pack(reference)), reference, rtol=1e-10, atol=1e-10):
        msg = "The fitter bounds are incompatible with the reference parameters."
        raise ValueError(msg)
    local_backend = backend.with_local_optimizer(maxiter=maxiter)
    local_backend._bounds = tuple(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] -- isolated copy.
        _local_bounds(bound, float(reference[6 * i + 3])) for i, bound in enumerate(original_bounds)
    )
    local_fitter = copy(fitter)
    local_fitter._backend = local_backend  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] -- package-internal integration.
    scale = np.ones_like(reference)
    scale[layout.free_indices] = layout.upper - layout.lower
    exchangeable = model.num_components == 2 and str(original_bounds[0]) == str(original_bounds[1])

    data = result.data
    prediction = model.prediction()
    if (
        not np.all(np.isfinite(data))
        or np.any(data < 0)
        or not np.all(data == np.floor(data))
        or np.sum(data) <= 0
        or not np.all(np.isfinite(prediction))
        or np.any(prediction < 0)
        or np.sum(prediction) <= 0
    ):
        msg = "Bootstrap requires nonnegative integer counts and a finite nonnegative fitted prediction."
        raise ValueError(msg)
    total = int(np.sum(data))
    probabilities = prediction.ravel() / np.sum(prediction)
    if samples is not None:
        _validate_samples(samples, result)

    def run(index: int) -> BootstrapReplicate:
        draw_seed, optimizer_seed = np.random.SeedSequence(seed, spawn_key=(index,)).spawn(2)
        rng = np.random.default_rng(draw_seed)
        optimizer_rng = np.random.default_rng(optimizer_seed)
        outcome: BackendResult
        if samples is None:
            counts = rng.multinomial(total, probabilities).reshape(data.shape).astype(np.float64)
            count_total = total
            outcome = local_fitter.fit_spiral_with_background(
                counts,
                model.background,
                model.z_mesh,
                model.vz_mesh,
                num_components=model.num_components,
                winding=model.winding,
                warm_start=reference,
                rng=optimizer_rng,
                improve_background=False,
            )
        else:
            indices = rng.integers(samples.z.size, size=samples.z.size)
            z, vz = samples.z[indices], samples.vz[indices]
            counts, _, _ = np.histogram2d(z, vz, bins=(samples.z_bins, samples.vz_bins))
            count_total = int(np.sum(counts))
            outcome = local_fitter.fit_spiral(
                z,
                vz,
                samples.z_bins,
                samples.vz_bins,
                num_components=model.num_components,
                winding=model.winding,
                warm_start=reference,
                rng=optimizer_rng,
                improve_background=samples.improve_background,
            )
        parameters = None
        if isinstance(outcome, FitFailure):
            reason = outcome.message
        else:
            reason = str(outcome.result.reason)
            invalid = outcome.result.reason in (
                FitTerminationReason.FAILED_REOPTIMIZATION,
                FitTerminationReason.INVALID_BACKGROUND_UPDATE,
            )
            if outcome.diagnostics.success and not invalid:
                parameters = _aligned(
                    outcome.result.final_model.parameters.flatten(), reference, scale, exchangeable=exchangeable
                )
            elif not outcome.diagnostics.success:
                reason = f"Local optimizer did not converge: {outcome.diagnostics.message}"
        return BootstrapReplicate(index, parameters, outcome.diagnostics, reason, count_total)

    if workers == 1:
        replicates = tuple(run(i) for i in range(n_resamples))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            replicates = tuple(executor.map(run, range(n_resamples)))
    return _summarize(
        replicates,
        reference,
        layout,
        exchangeable=exchangeable,
        method="parametric_counts" if samples is None else "samples",
        confidence_level=confidence_level,
        seed=seed,
        maxiter=maxiter,
    )


def _summarize(
    replicates: tuple[BootstrapReplicate, ...],
    reference: onp.Array1D[np.float64],
    layout: ParameterLayout,
    *,
    exchangeable: bool,
    method: Literal["samples", "parametric_counts"],
    confidence_level: float,
    seed: int,
    maxiter: int,
) -> BootstrapResult:
    n_resamples = len(replicates)
    parameters = np.full((n_resamples, reference.size), np.nan)
    for item in replicates:
        if item.parameters is not None:
            parameters[item.index] = item.parameters
    accepted = parameters[np.all(np.isfinite(parameters), axis=1)]
    warnings = ["Experimental local-refit bootstrap; global-search adequacy and interval coverage have not been validated."]
    if method == "parametric_counts":
        warnings.append("Conditional on the fitted background, selected component count, and winding.")
    else:
        warnings.append("Conditional on selected component count, winding, and the original fitter configuration.")
    if len(accepted) < n_resamples:
        warnings.append(
            f"{n_resamples - len(accepted)} of {n_resamples} replicates failed; summaries exclude them and may be biased."
        )
    if any(item.reason == FitTerminationReason.ITERATION_LIMIT for item in replicates):
        warnings.append("Some replicates reached the background refinement limit.")
    covariance = np.full((reference.size, reference.size), np.nan)
    intervals = np.full((reference.size, 2), np.nan)
    bias = np.full(reference.size, np.nan)
    if len(accepted) >= 2:
        covariance = np.asarray(np.cov(accepted, rowvar=False, ddof=1))
        tail = (1 - confidence_level) / 2
        intervals = np.quantile(accepted, [tail, 1 - tail], axis=0).T
        bias = accepted.mean(axis=0) - reference
        fixed = np.ones(reference.size, dtype=np.bool_)
        fixed[layout.free_indices] = False
        covariance[fixed, :] = 0.0
        covariance[:, fixed] = 0.0
        bias[fixed] = 0.0
        intervals[fixed, :] = reference[fixed, None]
        free_values = accepted[:, layout.free_indices]
        if np.any(np.isclose(free_values, layout.lower)) or np.any(np.isclose(free_values, layout.upper)):
            warnings.append("Some free parameters touch bounds; percentile intervals may have poor coverage.")
        if np.any(np.ptp(accepted[:, 3::6], axis=0) > np.pi):
            warnings.append("Angular samples span more than half a period; linear summaries may be unreliable.")
    else:
        warnings.append("At least two accepted replicates are required; summaries are NaN.")
    if exchangeable:
        warnings.append("Exchangeable components were matched by scaled parameter distance; inspect for ambiguous assignments.")
    return BootstrapResult(
        reference=reference,
        parameters=parameters,
        covariance=covariance,
        standard_errors=np.sqrt(np.diag(covariance)),
        intervals=intervals,
        bias=bias,
        replicates=replicates,
        method=method,
        confidence_level=confidence_level,
        seed=seed,
        maxiter=maxiter,
        warnings=tuple(warnings),
    )
