"""The spiral fitting algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Callable

import numpy as np
import os
import concurrent.futures
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
        winding: Literal[-1, 1] | None = None,
        warm_start: onp.Array1D[np.float64] | None = None,
        seed: int | None = None,
        num_components: int | None = None,
        improve_background: bool = True,
    ) -> PSpiralFitResult:
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
    ) -> Generator[PSpiralFitResult]:
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
        return self.fit_spiral_with_background_gen(
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
    ) -> PSpiralFitResult:
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
        result : PSpiralFitResult
            The fit at each iteration.

        """
        rng = np.random.default_rng(seed)

        mask: Final[onp.Array2D[np.float64]] = self._mask_func(z_mesh, vz_mesh)

        best_background: onp.Array2D[np.float64] = initial_background
        # Initialize best_quality to -inf so that after the first model is found
        # we compare the model prediction likelihood rather than the background-only likelihood.
        best_quality: float = float("-inf")

        initial_model: PSpiralModel | None = None
        current_model: PSpiralModel | None = None
        best_model: PSpiralModel | None = None

        converged: bool = False
        num_iterations: int = 0

        # Initialize warm start from caller-provided warm_start so we can reuse it.
        current_warm_start: onp.Array1D[np.float64] | None = warm_start

        # If num_components is None, compare 1- and 2-component fits using the
        # initial background only (improve_background=False) and pick the better
        # model. Continue the rest of the algorithm with that fixed choice.
        if num_components is None:
            res1 = _get_value_from_gen(
                self.fit_spiral_with_background_gen(
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
            )
            # Prepare warm start for 2-component fit: duplicate the 1-component params and
            # initialise the second component's theta0 to be 180 degrees (pi) offset from the first.
            comp1_params = res1.final_model.components[0].to_array()
            comp2_params = comp1_params.copy()
            comp2_params[3] = comp1_params[3] + np.pi
            warm_start_for_res2 = np.concatenate([comp1_params, comp2_params]) if current_warm_start is None else current_warm_start
            res2 = _get_value_from_gen(
                self.fit_spiral_with_background_gen(
                    initial_density,
                    initial_background,
                    z_mesh,
                    vz_mesh,
                    winding=winding,
                    warm_start=warm_start_for_res2,
                    seed=seed,
                    num_components=2,
                    improve_background=False,
                )
            )
            q1 = ln_likelihood(initial_density, res1.final_model.prediction(), mask)
            q2 = ln_likelihood(initial_density, res2.final_model.prediction(), mask)
            # Penalize the larger model using AIC: AIC = 2k - 2 lnL, k = number of parameters
            k1 = 6
            k2 = 12
            a1 = 2 * k1 - 2.0 * q1
            a2 = 2 * k2 - 2.0 * q2
            if a2 < a1:
                num_components = 2
                current_warm_start = np.concatenate([c.to_array() for c in res2.final_model.components])
            else:
                num_components = 1
                current_warm_start = res1.final_model.components[0].to_array()

        # Winding: None => auto-select on first iteration; otherwise use provided winding.
        best_winding: Literal[-1, 1] | None = winding

        while self._max_iterations is None or (num_iterations < self._max_iterations):
            num_iterations += 1

            # Number of 6-parameter components to fit
            param_count: int = 1 if num_components is None else num_components

            def wrap_winding_objective(current_winding: Literal[-1, 1]):
                """Return objective and analytic jacobian for given winding."""

                def _objective(parameters: onp.Array1D[np.float64]) -> float:
                    comps = tuple(
                        PSpiralComponent.from_array(parameters[6 * i : 6 * (i + 1)], winding=current_winding)
                        for i in range(param_count)
                    )
                    model = PSpiralModel(comps, z_mesh, vz_mesh, best_background)
                    return -ln_likelihood(initial_density, model.prediction(), mask)

                def _jac(parameters: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]:
                    # Analytic Jacobian: d objective / d parameters
                    # objective = -lnL; so jac = - d lnL / dp
                    # Compute model signal per component and per-parameter derivatives
                    comps = [
                        PSpiralComponent.from_array(parameters[6 * i : 6 * (i + 1)], winding=current_winding)
                        for i in range(param_count)
                    ]
                    # For each component compute perturbation and its derivatives wrt its 6 params
                    # We'll vectorize across the grid
                    z_flat = z_mesh.ravel()
                    vz_flat = vz_mesh.ravel()
                    ncell = z_flat.size

                    # Stack perturbations
                    pert_stack = np.empty((param_count, ncell), dtype=np.float64)
                    deriv_stack = np.empty((param_count, 6, ncell), dtype=np.float64)

                    for idx, comp in enumerate(comps):
                        # extract params
                        alpha = float(comp.alpha)
                        b = float(comp.b)
                        c = float(comp.c)
                        theta0 = float(comp.theta0)
                        scale = float(comp.scale_factor)
                        rho = float(comp.rho)
                        winding_val = int(comp.winding)
                        f = float(comp.flattening_strength)

                        # scaled coordinates
                        scaled_z = z_flat * scale
                        scaled_vz = vz_flat / scale
                        r = np.hypot(z_flat, scaled_vz)
                        theta = np.arctan2(vz_flat, scaled_z)

                        # phase and derivatives wrt b,c and r
                        if c != 0.0:
                            half = 0.5 * b / c
                            A = half * half + r / c
                            sqrtA = np.sqrt(A)
                            phase = -half + sqrtA
                            # dphase/db
                            dhalf_db = 0.5 / c
                            dA_db = 2.0 * half * dhalf_db
                            dphase_db = -dhalf_db + 0.5 * dA_db / sqrtA
                            # dphase/dc
                            dhalf_dc = -0.5 * b / (c * c)
                            dA_dc = 2.0 * half * dhalf_dc - r / (c * c)
                            dphase_dc = -dhalf_dc + 0.5 * dA_dc / sqrtA
                            # dphase/dr
                            dphase_dr = 0.5 / sqrtA * (1.0 / c)
                        else:
                            # c == 0: phase = r / b
                            phase = r / b
                            dphase_db = -r / (b * b)
                            dphase_dc = 0.0
                            dphase_dr = 1.0 / b

                        # flattening and derivatives
                        ex = np.exp(-(r - rho) / f)
                        F = 1.0 / (1.0 + ex)
                        dF_dr = (ex / (f * (1.0 + ex) * (1.0 + ex)))  # = (1/f)*F*(1-F)
                        dF_dr = dF_dr
                        dF_drho = -dF_dr

                        # U = winding*theta - phase - theta0
                        U = winding_val * theta - phase - theta0
                        cosU = np.cos(U)
                        sinU = np.sin(U)

                        # perturbation
                        pert = 1.0 + alpha * F * cosU
                        pert_stack[idx, :] = pert

                        # derivatives
                        # dalpha
                        deriv_stack[idx, 0, :] = F * cosU
                        # db
                        deriv_stack[idx, 1, :] = alpha * F * sinU * dphase_db
                        # dc
                        deriv_stack[idx, 2, :] = alpha * F * sinU * dphase_dc
                        # dtheta0
                        deriv_stack[idx, 3, :] = alpha * F * sinU * 1.0
                        # dscale (scale_factor)
                        # dr/dscale = - vz^2 / (r * scale^3)
                        # dtheta/dscale = - vz_flat * z_flat / ( (z_flat * scale)**2 + vz_flat**2 )
                        with np.errstate(divide='ignore', invalid='ignore'):
                            dr_dscale = - (vz_flat * vz_flat) / (r * (scale ** 3))
                            denom_theta = (z_flat * scale) ** 2 + vz_flat ** 2
                            dtheta_dscale = - (vz_flat * z_flat) / denom_theta
                            dtheta_dscale = np.where(denom_theta == 0.0, 0.0, dtheta_dscale)
                            dr_dscale = np.where(r == 0.0, 0.0, dr_dscale)

                        # dphase/dscale = dphase/dr * dr_dscale
                        dphase_dscale = dphase_dr * dr_dscale
                        dF_dscale = dF_dr * dr_dscale
                        deriv_stack[idx, 4, :] = (
                            alpha * (dF_dscale * cosU + F * (-sinU) * (winding_val * dtheta_dscale - dphase_dscale))
                        )
                        # drho
                        deriv_stack[idx, 5, :] = alpha * cosU * dF_drho

                    # Now resolve max across components
                    pert_stack = np.where(np.isfinite(pert_stack), pert_stack, -np.inf)
                    argmax = np.argmax(pert_stack, axis=0)
                    s = np.take_along_axis(pert_stack, argmax[None, :], axis=0)[0]
                    # dy/dp = background * ds/dp; assemble gradient
                    background_flat = best_background.ravel()
                    data_flat = initial_density.ravel()
                    mask_flat = mask.ravel()
                    y = background_flat * s
                    rvec = data_flat - y
                    # d lnL / dy per cell
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dlnL_dy = 0.5 * (mask_flat ** 2) * rvec * (2.0 * y + rvec) / (y * y)
                        dlnL_dy = np.where(np.isfinite(dlnL_dy), dlnL_dy, 0.0)

                    grad = np.zeros(6 * param_count, dtype=np.float64)
                    for idx in range(param_count):
                        # select cells where this component contributes
                        sel = argmax == idx
                        if not np.any(sel):
                            continue
                        ds_dp = deriv_stack[idx, :, sel]  # shape (6, nsel)
                        # dy/dp = background_flat[sel] * ds_dp
                        dy_dp = background_flat[sel][None, :] * ds_dp
                        contrib = (dlnL_dy[sel][None, :] * dy_dp).sum(axis=1)
                        # place into grad
                        grad[6 * idx : 6 * (idx + 1)] = -contrib  # negative because objective = -lnL

                    return grad

                return _objective, _jac

            # Auto-select winding on first iteration if unset, then optimize for it.
            if best_winding is None:
                pos_obj, pos_jac = wrap_winding_objective(1)
                neg_obj, neg_jac = wrap_winding_objective(-1)
                pos_res = self._optimize_parameters(pos_obj, jac_func=pos_jac, rng=rng, warm_start=current_warm_start, param_count=param_count)
                neg_res = self._optimize_parameters(neg_obj, jac_func=neg_jac, rng=rng, warm_start=current_warm_start, param_count=param_count)
                best_winding = 1 if pos_res.fun <= neg_res.fun else -1

            # Optimize for chosen winding.
            obj_func, jac_func = wrap_winding_objective(best_winding)
            res = self._optimize_parameters(obj_func, jac_func=jac_func, rng=rng, warm_start=current_warm_start, param_count=param_count)

            best_params: onp.Array1D[np.float64] = np.array(res.x, dtype=np.float64)
            comps = tuple(
                PSpiralComponent.from_array(best_params[6 * i : 6 * (i + 1)], winding=best_winding) for i in range(param_count)
            )
            current_model = PSpiralModel(comps, z_mesh, vz_mesh, best_background)

            # Set the first model
            if initial_model is None:
                initial_model = current_model
                # After the first model is constructed, evaluate its likelihood so
                # comparisons are done against model predictions rather than the
                # background-only prediction. This prevents the algorithm from
                # always thinking the model is a large improvement over the
                # background-only case and unnecessarily updating the background.
                if best_quality == float("-inf"):
                    best_quality = ln_likelihood(initial_density, initial_model.prediction(), mask)

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
            if best_quality > quality:
                converged = best_model is not None
                if best_model is None:
                    best_model = initial_model
                break

            # Update best parameters
            best_quality = quality
            best_background = new_background
            best_model = current_model
            current_warm_start = best_params

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
        self,
        objective_func: _ObjectiveFunc,
        *,
        jac_func: Callable[[onp.Array1D[np.float64]], onp.Array1D[np.float64]] | None = None,
        rng: np.random.Generator,
        warm_start: onp.Array1D[np.float64] | None,
        param_count: int = 1,
    ) -> optimize.OptimizeResult:
        # warm_start may be None or a flat vector of length 6 * param_count
        assert warm_start is None or (warm_start.ndim == 1 and len(warm_start) == 6 * param_count)
        base_bounds = list(zip(self._param_lo.tolist(), self._param_hi.tolist(), strict=True))
        bounds = base_bounds * param_count

        best_res: optimize.OptimizeResult | None = None
        # Generate initial starting points for all multi-starts
        x0s: list[onp.Array1D[np.float64]] = []
        for i in range(self._num_starts):
            if i == 0 and warm_start is not None and len(warm_start) == 6 * param_count:
                x0s.append(warm_start)
            else:
                # Sample each component's 6 params independently
                x0s.append(rng.uniform(np.tile(self._param_lo, param_count), np.tile(self._param_hi, param_count)))

        # Run multi-start optimizer trials in parallel using threads. Threads avoid
        # pickling nested objective closures and can still provide concurrency since
        # heavy work happens in C (NumPy/SciPy) which releases the GIL.
        max_workers = min(self._num_starts, (os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            if jac_func is None:
                futures = [ex.submit(optimize.minimize, objective_func, x0=x0, bounds=bounds) for x0 in x0s]
            else:
                futures = [ex.submit(optimize.minimize, objective_func, x0=x0, bounds=bounds, jac=jac_func) for x0 in x0s]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
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
