"""The spiral fitting algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage, special, optimize

from psnailder.component import PSpiralComponent

from .background_utils import generate_initial_background
from .likelihood_utils import ln_likelihood
from .model import PSpiralModel

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import Final

    from optype import numpy as onp


type _ObjectiveFunc = Callable[[onp.Array1D[np.float64]], onp.ToFloat]
type _SmoothingFunc = Callable[[onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _MaskFunc = Callable[[onp.Array2D[np.float64], onp.Array2D[np.float64]], onp.Array2D[np.float64]]

_DEFAULT_PARAM_LO: onp.Array1D[np.float64] = np.array([0.0, 0.005, 0.0, -np.pi, 30.0, 0.0])
_DEFAULT_PARAM_HI: onp.Array1D[np.float64] = np.array([1.0, 0.1, 0.004, +np.pi, 70.0, 0.18])

__all__: Final[list[str]] = [
    "PSpiralFitResult",
    "PSpiralFitter",
    "create_gaussian_smoother",
    "create_sigmoid_mask",
]


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
    converged : bool
        A flag signifying whether fitting converged before the fitting process stopped.

    """

    initial_model: PSpiralModel
    final_model: PSpiralModel
    data: onp.Array2D[np.float64]
    num_iterations: int
    max_iterations: int | None
    converged: bool


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
        num_starts: int = 20,
        max_iterations: int | None = 50,
        smoothing_func: _SmoothingFunc | None = None,
        mask_func: _MaskFunc | None = None,
        param_lo: onp.Array1D[np.float64] | None = None,
        param_hi: onp.Array1D[np.float64] | None = None,
    ) -> None:
        self._max_iterations: int | None = max_iterations
        self._num_starts: int = num_starts

        self._smoothing_func: _SmoothingFunc = create_gaussian_smoother(2.0) if smoothing_func is None else smoothing_func
        self._mask_func: _MaskFunc = create_sigmoid_mask(1.0, 40.0) if mask_func is None else mask_func

        self._param_lo: onp.Array1D[np.float64] = param_lo if param_lo is not None else _DEFAULT_PARAM_LO
        self._param_hi: onp.Array1D[np.float64] = param_hi if param_hi is not None else _DEFAULT_PARAM_HI

        # Replace nans with default values
        self._param_lo[np.isnan(self._param_lo)] = _DEFAULT_PARAM_LO[np.isnan(self._param_lo)]
        self._param_hi[np.isnan(self._param_hi)] = _DEFAULT_PARAM_HI[np.isnan(self._param_hi)]

    def fit_spiral(
        self,
        z: onp.Array1D[np.float64],
        vz: onp.Array1D[np.float64],
        z_bins: onp.Array1D[np.float64],
        vz_bins: onp.Array1D[np.float64],
        *,
        warm_start: onp.Array1D[np.float64] | None = None,
        seed: int | None = None,
        improve_background: bool = True,
    ) -> PSpiralFitResult:
        val = _get_value_from_gen(
            self.fit_spiral_gen(z, vz, z_bins, vz_bins, warm_start=warm_start, seed=seed, improve_background=improve_background)
        )
        assert val is not None
        return val

    def fit_spiral_gen(
        self,
        z: onp.Array1D[np.float64],
        vz: onp.Array1D[np.float64],
        z_bins: onp.Array1D[np.float64],
        vz_bins: onp.Array1D[np.float64],
        *,
        warm_start: onp.Array1D[np.float64] | None = None,
        seed: int | None = None,
        improve_background: bool = True,
    ) -> Generator[PSpiralFitResult]:
        z_centres = 0.5 * (z_bins[:-1] + z_bins[1:])
        vz_centres = 0.5 * (vz_bins[:-1] + vz_bins[1:])
        z_mesh, vz_mesh = np.meshgrid(z_centres, vz_centres)
        density, _, _ = np.histogram2d(z, vz, bins=(z_bins, vz_bins), density=False)
        density = density.T
        background = generate_initial_background(z, vz, z_mesh, vz_mesh)
        return self.fit_spiral_with_background_gen(
            density, background, z_mesh, vz_mesh, warm_start=warm_start, seed=seed, improve_background=improve_background
        )

    def fit_spiral_with_background(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        *,
        warm_start: onp.Array1D[np.float64] | None = None,
        seed: int | None = None,
        improve_background: bool = True,
    ) -> PSpiralFitResult:
        val = _get_value_from_gen(
            self.fit_spiral_with_background_gen(
                initial_density,
                initial_background,
                z_mesh,
                vz_mesh,
                warm_start=warm_start,
                seed=seed,
                improve_background=improve_background,
            )
        )
        assert val is not None
        return val

    def fit_spiral_with_background_gen(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        *,
        warm_start: onp.Array1D[np.float64] | None = None,
        seed: int | None = None,
        improve_background: bool = True,
    ) -> Generator[PSpiralFitResult]:
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
        warm_start : dict[ParamName, float]
            The warm start parameters.
        seed : int | None
            The random seed for the multi-start draws, or ``None`` for no seed.
        improve_background : bool
            Whether to iteratively improve the background. If ``False``, the
            background is fixed to ``initial_background``.


        Yields
        ------
        result : PSpiralFitResult
            The fit at each iteration.

        """
        rng = np.random.default_rng(seed)

        mask: Final[onp.Array2D[np.float64]] = self._mask_func(z_mesh, vz_mesh)

        best_background: onp.Array2D[np.float64] = initial_background
        best_quality: float = ln_likelihood(initial_density, best_background, mask)

        initial_model: PSpiralModel | None = None
        current_model: PSpiralModel | None = None
        best_model: PSpiralModel | None = None

        converged: bool = False

        num_iterations: int = 0

        while self._max_iterations is None or (num_iterations < self._max_iterations):
            num_iterations += 1

            def _objective(parameters: onp.Array1D[np.float64]) -> float:
                spiral_component = PSpiralComponent.from_array(parameters)
                model = PSpiralModel((spiral_component,), z_mesh, vz_mesh, best_background)
                return -ln_likelihood(initial_density, model.prediction(), mask)

            res = self._optimize_parameters(_objective, rng=rng, warm_start=warm_start)
            best_params: onp.Array1D[np.float64] = np.array(res.x, dtype=np.float64)
            current_model = PSpiralModel((PSpiralComponent.from_array(best_params),), z_mesh, vz_mesh, best_background)

            # Set the first model
            if initial_model is None:
                initial_model = current_model

            if not improve_background:
                best_model = current_model
                converged = True
                break

            # Yield this iteration's result
            yield PSpiralFitResult(
                initial_model=initial_model,
                final_model=current_model,
                data=initial_density,
                num_iterations=num_iterations,
                max_iterations=self._max_iterations,
                converged=converged,
            )

            # Update background
            current_perturbation = current_model.signal()
            new_background = self._smoothing_func(initial_density / current_perturbation)
            new_background = new_background / new_background.sum() * initial_density.sum()
            new_data = current_perturbation * new_background
            quality = ln_likelihood(initial_density, new_data, mask)

            # Quality has degraded => we have converged
            if best_quality >= quality:
                converged = best_model is not None
                if best_model is None:
                    best_model = initial_model
                break

            # Update best parameters
            best_quality = quality
            best_background = new_background
            best_model = current_model

        assert initial_model is not None
        assert best_model is not None
        yield PSpiralFitResult(
            initial_model=initial_model,
            final_model=best_model,
            data=initial_density,
            num_iterations=num_iterations,
            max_iterations=self._max_iterations,
            converged=converged,
        )

    def _optimize_parameters(
        self, objective_func: _ObjectiveFunc, *, rng: np.random.Generator, warm_start: onp.Array1D[np.float64] | None
    ) -> optimize.OptimizeResult:
        assert warm_start is None or (warm_start.ndim == 1 and len(warm_start) == 6)
        bounds = list(zip(self._param_lo.tolist(), self._param_hi.tolist(), strict=True))

        best_res: optimize.OptimizeResult | None = None
        for i in range(self._num_starts):
            x0: onp.Array1D[np.float64]
            if i == 0 and warm_start is not None:
                x0 = warm_start
            else:
                x0 = rng.uniform(self._param_lo, self._param_hi)
            res = optimize.minimize(objective_func, x0=x0, bounds=bounds)
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        assert best_res is not None, "failed to find a single minimum."
        return best_res


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
