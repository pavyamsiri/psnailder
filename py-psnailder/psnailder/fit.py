"""The spiral fitting algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import optimize  # noqa: F401 -- retained as a compatibility patch target for callers/tests.

from psnailder._backends import FitBackend, FitRequest
from psnailder._python_backend import PythonFitBackend
from psnailder._rust_backend import RustFitBackend

from ._backends import (
    BackendEvent,
    BackendResult,
    FitFailure,
    FitFailureReason,
    FitProgress,
    FitSuccess,
    FitTerminationReason,
    GaussianSmoothConfig,
    MaskConfig,
    OptimizationDiagnostics,
    OptimizationResult,
    PSpiralFitResult,
    SigmoidMaskConfig,
    SmoothConfig,
)
from ._background_utils import generate_initial_background
from ._python_backend import create_gaussian_smoother, create_sigmoid_mask
from .bounds import Fixed, Interval, ParameterBounds

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from optype import numpy as onp


type _SmoothingFunc = Callable[[onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _MaskFunc = Callable[[onp.Array2D[np.float64], onp.Array2D[np.float64]], onp.Array2D[np.float64]]

__all__: list[str] = [
    "FitEvent",
    "FitFailure",
    "FitFailureReason",
    "FitInput",
    "FitOutcome",
    "FitProgress",
    "FitSuccess",
    "FitTerminationReason",
    "Fixed",
    "GaussianSmoothConfig",
    "Interval",
    "MaskConfig",
    "OptimizationDiagnostics",
    "OptimizationResult",
    "PSpiralFitResult",
    "PSpiralFitter",
    "ParameterBounds",
    "SigmoidMaskConfig",
    "SmoothConfig",
    "create_gaussian_smoother",
    "create_sigmoid_mask",
]

type FitOutcome = BackendResult
type FitEvent = BackendEvent


@dataclass(frozen=True, kw_only=True)
class FitInput:
    """Prepared grids and options for one item in :meth:`PSpiralFitter.fit_batch`.

    ``density``, ``background``, ``z_mesh`` and ``vz_mesh`` must share a 2D
    shape, with rows along vz and columns along z. Different batch items may
    have different shapes. Arrays are retained by reference; do not mutate
    them during fitting. Options have the same meanings as in
    ``fit_spiral_with_background``. Supply a separate random generator for
    each item when reproducible Python optimization is required.
    """

    density: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]
    z_mesh: onp.Array2D[np.float64]
    vz_mesh: onp.Array2D[np.float64]
    winding: Literal[-1, 1] | None = None
    warm_start: onp.Array1D[np.float64] | None = None
    rng: np.random.Generator | None = None
    num_components: int | None = None
    improve_background: bool = True


class PSpiralFitter:
    """A configuration of the spiral fitting algorithm."""

    def __init__(
        self,
        *,
        backend: Literal["python", "rust"] = "python",
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
        backend : {"python", "rust"}
            Backend selected for this fitter instance. The Rust backend currently
            supports only the subset documented by `RustFitBackend`.
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
        self._backend: FitBackend
        if backend == "python":
            self._backend = PythonFitBackend(
                max_iterations=max_iterations,
                atol=atol,
                rtol=rtol,
                smoothing_func=smoothing_func,
                mask_func=mask_func,
                bounds=bounds,
            )
        elif backend == "rust":
            self._backend = RustFitBackend(
                max_iterations=max_iterations,
                atol=atol,
                rtol=rtol,
                smoothing_func=smoothing_func,
                mask_func=mask_func,
                bounds=bounds,
            )
        else:
            msg = "Only `python` and `rust` backends are currently supported."  # pyright: ignore[reportUnreachable]
            raise ValueError(msg)

    def fit_batch(self, inputs: Sequence[FitInput], *, workers: int | None = None) -> list[FitOutcome]:
        """Fit prepared grids as a batch using the configured backend.

        Parameters
        ----------
        inputs : Sequence[FitInput]
            Prepared grids and per-fit options. Fitter configuration is shared
            across all items.
        workers : int or None
            Positive worker limit for the batch. One requests serial execution;
            None lets the backend choose. Rust shares this limit between
            batch fitting and inner optimization. Python uses a thread pool;
            None uses ThreadPoolExecutor's default worker count. Python control
            flow remains subject to the GIL on GIL-enabled interpreters.

        Returns
        -------
        list[FitOutcome]
            One terminal success or failure per input, in input order.
            Individual fitting failures retain their position in the list.

        Raises
        ------
        ValueError
            If workers is not a positive integer or None.

        Notes
        -----
        Custom mask and smoothing callbacks must support concurrent calls when
        using Python threads. Use workers=1 for serial execution. Give each input
        its own random generator for reproducibility. Invalid inputs or callback
        exceptions propagate to the caller, as with single fits.

        """
        if workers is not None and (type(workers) is not int or workers < 1):
            msg = "workers must be a positive integer or None."
            raise ValueError(msg)
        requests = [
            FitRequest(
                initial_density=item.density,
                initial_background=item.background,
                z_mesh=item.z_mesh,
                vz_mesh=item.vz_mesh,
                winding=item.winding,
                warm_start=item.warm_start,
                rng=item.rng,
                num_components=item.num_components,
                improve_background=item.improve_background,
            )
            for item in inputs
        ]
        return self._backend.fit_batch(requests, workers=workers)

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
    ) -> BackendResult:
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
            One or two components, or None to compare both using BIC.
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
    ) -> Generator[BackendEvent]:
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
        # Validate bins
        self._validate_public_options(num_components, winding, warm_start)
        for name, edges in (("z_bins", z_bins), ("vz_bins", vz_bins)):
            if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
                msg = f"{name} must contain finite, strictly increasing bin edges."
                raise ValueError(msg)
        # Validate dimensionality of z and vz
        for name, ndim in (("z", z.ndim), ("vz", vz.ndim)):
            if ndim != 1:
                msg = f"`{name}` must be a 1D array."
                raise ValueError(msg)
        # Validate that arrays are the same shape
        common_shape: tuple[int] = z.shape
        for name, shape in (("z", z.shape), ("vz", vz.shape)):
            if shape != common_shape:
                msg = f"`{name}` was expected to have shape {common_shape} but was {shape}."
                raise ValueError(msg)

        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(vz)):
            msg = "z and vz samples must be finite."
            raise ValueError(msg)
        if z.size < 2:
            yield self._unusable_data("At least two samples are required for background estimation.")
            return

        z_centres = 0.5 * (z_bins[:-1] + z_bins[1:])
        vz_centres = 0.5 * (vz_bins[:-1] + vz_bins[1:])
        z_mesh, vz_mesh = np.meshgrid(z_centres, vz_centres)
        density, _, _ = np.histogram2d(z, vz, bins=(z_bins, vz_bins), density=False)
        density = density.T
        if not np.any(density > 0):
            yield self._unusable_data("No samples fall inside the fitting region.")
            return
        try:
            background = generate_initial_background(z, vz, z_mesh, vz_mesh)
        except np.linalg.LinAlgError as exc:
            yield self._unusable_data(f"KDE background estimation failed: {exc}")
            return
        # Midpoint density times bin area approximates probability mass.
        # Rows follow vz and columns follow z, matching the transposed counts.
        background = background * np.diff(vz_bins)[:, None] * np.diff(z_bins)[None, :]
        if not self._valid_background(background):
            yield self._unusable_data("KDE produced an invalid background.")
            return
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
        return self._backend.fit(
            FitRequest(
                initial_density=initial_density,
                initial_background=initial_background,
                z_mesh=z_mesh,
                vz_mesh=vz_mesh,
                winding=winding,
                warm_start=warm_start,
                rng=rng,
                num_components=num_components,
                improve_background=improve_background,
            )
        )

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
    ) -> Generator[BackendEvent]:
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
            One or two components, or None to compare one and two components
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
        yield from self._backend.fit_events(
            request=FitRequest(
                initial_density=initial_density,
                initial_background=initial_background,
                z_mesh=z_mesh,
                vz_mesh=vz_mesh,
                winding=winding,
                warm_start=warm_start,
                rng=rng,
                num_components=num_components,
                improve_background=improve_background,
            )
        )

    @staticmethod
    def _validate_public_options(
        num_components: int | None,
        winding: int | None,
        warm_start: onp.Array1D[np.float64] | None,
    ) -> None:
        if num_components is not None and (
            isinstance(num_components, bool) or not isinstance(num_components, (int, np.integer)) or num_components not in (1, 2)
        ):
            msg = "`num_components` must be 1 or 2, or `None`; check component bounds."
            raise ValueError(msg)
        if winding is not None and (
            isinstance(winding, bool) or not isinstance(winding, (int, np.integer)) or winding not in (-1, 1)
        ):
            msg = "winding must be -1 or 1, or None."
            raise ValueError(msg)
        if warm_start is not None and num_components is None:
            msg = "Can not use warm start if the number of component is not set."
            raise ValueError(msg)

    @staticmethod
    def _unusable_data(message: str) -> FitFailure:
        return FitFailure(
            FitFailureReason.NO_VALID_CANDIDATE,
            message,
            OptimizationDiagnostics(message="No optimization took place.", success=False, nfev=0, nit=0),
        )

    @staticmethod
    def _valid_background(background: onp.Array2D[np.float64]) -> bool:
        with np.errstate(over="ignore", invalid="ignore"):
            total = np.sum(background)
        return bool(np.all(np.isfinite(background)) and np.all(background >= 0) and np.isfinite(total) and total > 0)


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
