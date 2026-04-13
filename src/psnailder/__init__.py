"""A module that implements  the phase spiral fitting algorithm described in Alinder et al. 2023."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import emcee  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from scipy import ndimage, special, stats, optimize

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from typing import Final, Literal

    from optype import numpy as onp


__all__: Final[Sequence[str]] = [
    "AlinderModel",
    "SpiralFitDiagnostics",
    "SpiralFitter",
    "SpiralFitterMCMC",
    "SpiralFitterMinimizer",
    "generate_initial_background",
]


@dataclass
class _ParamBounds:
    """A class to describe the bounds of a parameter.

    Attributes
    ----------
    name : ParamName
        The name of the parameter.
    lo : float
        The lower bound.
    hi : float
        The higher bound.

    """

    name: _ParamName
    lo: float
    hi: float


_DEFAULT_PARAM_BOUNDS: Final[tuple[_ParamBounds, ...]] = (
    _ParamBounds(name="alpha", lo=0.0, hi=1.0),
    _ParamBounds(name="b", lo=0.005, hi=0.1),
    _ParamBounds(name="c", lo=0.0, hi=0.004),
    _ParamBounds(name="theta0", lo=-np.pi, hi=np.pi),
    _ParamBounds(name="scale_factor", lo=30.0, hi=70.0),
    _ParamBounds(name="rho", lo=0.0, hi=0.18),
)
_NUM_PARAMETERS: Final[int] = 6
_ALPHA_INDEX: Final[int] = 0
_B_INDEX: Final[int] = 1
_C_INDEX: Final[int] = 2
_THETA0_INDEX: Final[int] = 3
_SCALE_FACTOR_INDEX: Final[int] = 4
_RHO_INDEX: Final[int] = 5

type _SmoothingFunc = Callable[[onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _ParamName = Literal["alpha", "b", "c", "theta0", "scale_factor", "rho"]


@dataclass
class SpiralFitDiagnostics:
    """A result of the spiral fitting process.

    Attributes
    ----------
    initial_model : AlinderModel
        The initial model.
    final_model : AlinderModel
        The best fit model.
    data : Array2D[f64]
        The data.
    z_mesh : Array2D[f64]
        The z values for each cell.
    vz_mesh : Array2D[f64]
        The Vz values for each cell.
    samples : Array2D[f64]
        The MCMC samples.
    log_probs : Array1D[f64]
        The log probabilities.
    num_iterations : int
        The number of iterations taken.
    max_iterations : int | None
        The maximum number of iterations.
    converged : bool
        A flag signifying whether fitting converged before the fitting process stopped.

    """

    initial_model: AlinderModel
    final_model: AlinderModel
    data: onp.Array2D[np.float64]
    z_mesh: onp.Array2D[np.float64]
    vz_mesh: onp.Array2D[np.float64]
    samples: onp.Array2D[np.float64]
    log_probs: onp.Array1D[np.float64]
    num_iterations: int
    max_iterations: int | None
    converged: bool


@dataclass
class AlinderModel:
    """A Alinder et al. 2023 phase spiral model.

    Attributes
    ----------
    alpha : float
        The spiral amplitude.
    b : float
        The linear winding amplitude.
    c : float
        The quadratic winding amplitude.
    theta0 : float
        The angle offset in radians.
    scale_factor : float
        The scale factor.
    rho : float
        The flattening function distance.
    background : Array2D[f64]
        The background.

    """

    alpha: float
    b: float
    c: float
    theta0: float
    scale_factor: float
    rho: float
    background: onp.Array2D[np.float64]

    def perturbation(self, z_mesh: onp.Array2D[np.float64], vz_mesh: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        """Calculate the perturbation.

        Parameters
        ----------
        z_mesh : Array2D[f64]
            The z values for each cell.
        vz_mesh : Array2D[f64]
            The Vz values for each cell.

        Returns
        -------
        perturbation : Array2D[f64]
            The perturbation.

        """
        scaled_z = np.multiply(z_mesh, self.scale_factor)
        scaled_vz = vz_mesh * np.reciprocal(self.scale_factor)
        r_mesh = np.hypot(z_mesh, scaled_vz)
        theta_mesh = np.arctan2(vz_mesh, scaled_z)

        abs_b = abs(self.b)
        abs_c = abs(self.c)
        sign = np.sign(self.b) if self.b != 0.0 else 1.0

        if abs_c != 0.0:
            half_b_over_c = 0.5 * abs_b / abs_c
            phase = sign * (-half_b_over_c + np.sqrt(np.square(half_b_over_c) + r_mesh / abs_c))
        else:
            phase = sign * (r_mesh / abs_b)

        flattening = special.expit((r_mesh - self.rho) / 0.1)
        return 1.0 + self.alpha * flattening * np.cos(theta_mesh - phase - self.theta0)

    def fit(self, z_mesh: onp.Array2D[np.float64], vz_mesh: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        """Calculate the predicted data/fit.

        Parameters
        ----------
        z_mesh : Array2D[f64]
            The z values for each cell.
        vz_mesh : Array2D[f64]
            The Vz values for each cell.

        Returns
        -------
        fit : Array2D[f64]
            The prediction.

        """
        pert = self.perturbation(z_mesh, vz_mesh)
        return np.multiply(pert, self.background)

    def phase_angle(self) -> float:
        """Calculate the phase angle.

        Returns
        -------
        phase_angle : float
            The phase angle in radians.

        """
        R_TEST: Final[float] = 0.5

        phase: float
        if self.c != 0.0:
            b_over_2c = 0.5 * self.b / self.c
            phase = -b_over_2c + np.sqrt(np.square(b_over_2c) + R_TEST / self.c)
        else:
            phase = R_TEST / self.b
        return phase + self.theta0

    def __repr__(self) -> str:
        fields: list[str] = []
        fields.append(f"alpha={self.alpha}")
        fields.append(f"b={self.b}")
        fields.append(f"c={self.c}")
        fields.append(f"theta0={np.rad2deg(self.theta0)} deg")
        fields.append(f"scale_factor={self.scale_factor}")
        fields.append(f"rho={self.rho}")
        return self.__class__.__qualname__ + "(" + ", ".join(fields) + ")"


@dataclass
class AlinderModelCollection:
    """A collection of Alinder et al. 2023 phase spiral models.

    Attributes
    ----------
    parameters : Array2D[f64]
        The model parameters in the shape (num_walkers, 6).
    background : Array2D[f64]
        The background.

    """

    parameters: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]

    def __post_init__(self) -> None:
        assert self.parameters.shape[1] == _NUM_PARAMETERS

    @property
    def num_walkers(self) -> int:
        """int: The number of walkers."""
        return self.parameters.shape[0]

    @property
    def alpha(self) -> onp.Array1D[np.float64]:
        """float: The spiral amplitude."""
        return self.parameters[:, _ALPHA_INDEX]

    @property
    def b(self) -> onp.Array1D[np.float64]:
        """float: The linear winding amplitude."""
        return self.parameters[:, _B_INDEX]

    @property
    def c(self) -> onp.Array1D[np.float64]:
        """float: The quadratic winding amplitude."""
        return self.parameters[:, _C_INDEX]

    @property
    def theta0(self) -> onp.Array1D[np.float64]:
        """float: The angle offset in radians."""
        return self.parameters[:, _THETA0_INDEX]

    @property
    def scale_factor(self) -> onp.Array1D[np.float64]:
        """float: The scale factor."""
        return self.parameters[:, _SCALE_FACTOR_INDEX]

    @property
    def rho(self) -> onp.Array1D[np.float64]:
        """float: The flattening function distance."""
        return self.parameters[:, _RHO_INDEX]

    def perturbation(
        self,
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        valid: onp.Array1D[np.bool_],
    ) -> onp.Array3D[np.float64]:
        """Calculate the perturbation.

        Parameters
        ----------
        z_mesh : Array2D[f64]
            The z values for each cell.
        vz_mesh : Array2D[f64]
            The Vz values for each cell.
        valid : Array2D[bool]
            A mask over the valid parameters.

        Returns
        -------
        perturbation : Array2D[f64]
            The perturbation.

        """
        alpha_arr = self.alpha
        b_arr = self.b
        c_arr = self.c
        theta0_arr = self.theta0
        scale_factor_arr = self.scale_factor
        rho_arr = self.rho

        abs_b_arr = np.abs(b_arr)
        abs_c_arr = np.abs(c_arr)
        sign_arr = np.sign(b_arr)
        sign_arr[sign_arr == 0.0] = 1.0

        z_mesh_broadcast = z_mesh[:, :, None]
        vz_mesh_broadcast = vz_mesh[:, :, None]
        r_mesh = np.hypot(z_mesh_broadcast, vz_mesh_broadcast / scale_factor_arr)
        theta_mesh = np.arctan2(vz_mesh_broadcast, scale_factor_arr * z_mesh_broadcast)

        phase: onp.Array3D[np.float64] = np.zeros_like(r_mesh)
        nonzero_mask = abs_c_arr != 0.0

        nonzero_value_mask = nonzero_mask & valid
        half_b_over_c = 0.5 * abs_b_arr[nonzero_value_mask] / abs_c_arr[nonzero_value_mask]
        discriminant = np.square(half_b_over_c) + r_mesh[:, :, nonzero_value_mask] / abs_c_arr[nonzero_value_mask]
        phase[:, :, nonzero_value_mask] = sign_arr[nonzero_value_mask] * (-half_b_over_c + np.sqrt(discriminant))

        zero_value_mask = ~nonzero_mask & valid
        phase[:, :, zero_value_mask] = sign_arr[zero_value_mask] * (r_mesh[:, :, zero_value_mask] / abs_b_arr[zero_value_mask])

        flattening = special.expit((r_mesh - rho_arr) / 0.1)
        return 1.0 + alpha_arr * flattening * np.cos(theta_mesh - phase - theta0_arr)


class SpiralFitter(ABC):
    """A configuration of the Alinder et al 2023 fitting algorithm."""

    def __init__(
        self,
        *,
        max_iterations: int | None = 50,
        smoothing_func: _SmoothingFunc | None = None,
        param_lo: dict[_ParamName, float] | None = None,
        param_hi: dict[_ParamName, float] | None = None,
        param_noise: float = 0.01,
        use_density: bool = True,
    ) -> None:
        self._max_iterations: int | None = max_iterations
        self._smoothing_func: _SmoothingFunc = SpiralFitter._default_smoothing if smoothing_func is None else smoothing_func
        self._param_lo: onp.Array1D[np.float64] = np.array([param.lo for param in _DEFAULT_PARAM_BOUNDS], dtype=np.float64)
        self._param_hi: onp.Array1D[np.float64] = np.array([param.hi for param in _DEFAULT_PARAM_BOUNDS], dtype=np.float64)
        self._use_density: bool = use_density
        self._param_noise: float = param_noise

        default_lo: dict[_ParamName, float] = {p.name: p.lo for p in _DEFAULT_PARAM_BOUNDS}
        default_hi: dict[_ParamName, float] = {p.name: p.hi for p in _DEFAULT_PARAM_BOUNDS}

        if param_lo is not None:
            if unknown := param_lo.keys() - default_lo.keys():
                msg = f"Unknown parameter names in `param_lo`: {unknown}"
                raise ValueError(msg)
            default_lo.update(param_lo)

        if param_hi is not None:
            if unknown := param_hi.keys() - default_hi.keys():
                msg = f"Unknown parameter names in `param_hi`: {unknown}"
                raise ValueError(msg)
            default_hi.update(param_hi)

        for name in default_lo:
            if default_lo[name] >= default_hi[name]:
                msg = f"param_lo[{name!r}] must be strictly less than param_hi[{name!r}]"
                raise ValueError(msg)

        self._param_lo = np.array([default_lo[p.name] for p in _DEFAULT_PARAM_BOUNDS], dtype=np.float64)
        self._param_hi = np.array([default_hi[p.name] for p in _DEFAULT_PARAM_BOUNDS], dtype=np.float64)

    @staticmethod
    def _default_smoothing(arr: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        return ndimage.gaussian_filter(arr, sigma=2)

    def _ln_prior(self, models: AlinderModelCollection) -> onp.Array1D[np.float64]:
        in_bounds: onp.Array1D[np.bool_] = np.ones(models.num_walkers, dtype=np.bool_)

        for index in range(_NUM_PARAMETERS):
            in_bounds &= np.logical_and(
                models.parameters[:, index] >= self._param_lo[index], models.parameters[:, index] <= self._param_hi[index]
            )

        prior = np.zeros_like(in_bounds, dtype=np.float64)
        prior[~in_bounds] = -np.inf

        return prior

    def ln_prob(
        self,
        models: AlinderModelCollection,
        counts: onp.Array2D[np.float64],
        background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
    ) -> onp.Array1D[np.float64]:
        ln_prior = self._ln_prior(models)
        pert = models.perturbation(z_mesh, vz_mesh, np.isfinite(ln_prior))
        mask = _mask(z_mesh, vz_mesh)
        valid = background > 0

        predicted = pert[valid, :] * background[valid, None]
        denom = np.copy(predicted)
        denom[denom == 0.0] = 1.0
        data = counts[valid, None]
        broadcast_mask = mask[valid, None]
        residuals = broadcast_mask * (data - predicted)
        ln_likelihood = -0.5 * np.sum(
            np.square(residuals) / denom,
            axis=0,
        )

        ln_likelihood[~np.isfinite(ln_likelihood)] = -np.inf

        return ln_likelihood + ln_prior

    def fit_spiral(
        self,
        z: onp.Array1D[np.float64],
        vz: onp.Array1D[np.float64],
        z_bins: onp.Array1D[np.float64],
        vz_bins: onp.Array1D[np.float64],
        *,
        use_median: bool,
        seed: int | None = None,
    ) -> SpiralFitDiagnostics:
        val = _get_value_from_gen(self.fit_spiral_gen(z, vz, z_bins, vz_bins, use_median=use_median, seed=seed))
        assert val is not None
        return val

    def fit_spiral_gen(
        self,
        z: onp.Array1D[np.float64],
        vz: onp.Array1D[np.float64],
        z_bins: onp.Array1D[np.float64],
        vz_bins: onp.Array1D[np.float64],
        *,
        use_median: bool,
        seed: int | None = None,
    ) -> Generator[SpiralFitDiagnostics]:
        z_centres = 0.5 * (z_bins[:-1] + z_bins[1:])
        vz_centres = 0.5 * (vz_bins[:-1] + vz_bins[1:])
        vz_mesh, z_mesh = np.meshgrid(vz_centres, z_centres)

        density, _, _ = np.histogram2d(z, vz, bins=(z_bins, vz_bins), density=self._use_density)
        background = generate_initial_background(z, vz, z_mesh, vz_mesh, density.sum())

        return self.fit_spiral_with_background_gen(density, background, z_mesh, vz_mesh, use_median=use_median, seed=seed)

    def fit_spiral_with_background(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        *,
        use_median: bool,
        seed: int | None = None,
    ) -> SpiralFitDiagnostics:
        val = _get_value_from_gen(
            self.fit_spiral_with_background_gen(
                initial_density, initial_background, z_mesh, vz_mesh, use_median=use_median, seed=seed
            )
        )
        assert val is not None
        return val

    @abstractmethod
    def fit_spiral_with_background_gen(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        use_median: bool,
        seed: int | None = None,
    ) -> Generator[SpiralFitDiagnostics]: ...


class SpiralFitterMCMC(SpiralFitter):
    """A configuration of the Alinder et al 2023 fitting algorithm.

    This implementation uses MCMC to fit the parameters.

    """

    def __init__(
        self,
        num_samples: int = 5_000,
        num_discard: int = 1_000,
        num_walkers: int = 32,
        max_iterations: int | None = 50,
        smoothing_func: _SmoothingFunc | None = None,
        param_lo: dict[_ParamName, float] | None = None,
        param_hi: dict[_ParamName, float] | None = None,
        param_noise: float = 0.01,
        use_density: bool = True,
    ) -> None:
        super().__init__(
            max_iterations=max_iterations,
            smoothing_func=smoothing_func,
            param_lo=param_lo,
            param_hi=param_hi,
            param_noise=param_noise,
            use_density=use_density,
        )
        self._num_samples: int = num_samples
        self._num_discard: int = num_discard
        self._num_walkers: int = num_walkers

    def fit_spiral_with_background_gen(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        use_median: bool,
        seed: int | None = None,
    ) -> Generator[SpiralFitDiagnostics]:
        rng = np.random.default_rng(seed=seed)
        mask = _mask(z_mesh, vz_mesh)
        background = initial_background
        best_quality: float = _calculate_rmse_with_mask(initial_density, initial_background, mask)
        initial_model: AlinderModel | None = None
        current_model: AlinderModel | None = None
        best_model: AlinderModel | None = None
        best_samples: onp.Array2D[np.float64] | None = None
        best_probs: onp.Array1D[np.float64] | None = None
        converged: bool = False
        num_iterations: int = 0
        while self._max_iterations is None or (num_iterations < self._max_iterations):
            num_iterations += 1
            np.random.seed(seed)
            sampler = emcee.EnsembleSampler(
                self._num_walkers,
                _NUM_PARAMETERS,
                ln_prob_mcmc,
                args=(self, initial_density, background, z_mesh, vz_mesh),
                vectorize=True,
            )
            np.random.seed(None)

            p0: onp.Array2D[np.float64]
            if best_model is None:
                p0 = rng.uniform(self._param_lo, self._param_hi, size=(self._num_walkers, _NUM_PARAMETERS))
            else:
                param_range = self._param_hi - self._param_lo
                noise = rng.normal(
                    loc=0.0,
                    scale=param_range * self._param_noise,
                    size=(self._num_walkers, _NUM_PARAMETERS),
                )
                old_params: onp.Array2D[np.float64] = np.tile(
                    np.array(
                        [
                            best_model.alpha,
                            best_model.b,
                            best_model.c,
                            best_model.theta0,
                            best_model.scale_factor,
                            best_model.rho,
                        ],
                        dtype=np.float64,
                    ),
                    (self._num_walkers, 1),
                )
                p0 = np.clip(
                    old_params + noise,
                    self._param_lo,
                    self._param_hi,
                )

            sampler.run_mcmc(p0, self._num_samples, progress=False)  # pyright: ignore[reportUnknownMemberType]

            flat_samples: onp.Array2D[np.float64] = sampler.get_chain(discard=self._num_discard, flat=True)  # pyright: ignore[reportUnknownVariableType, reportAssignmentType, reportUnknownMemberType]
            log_probs: onp.Array1D[np.float64] = sampler.get_log_prob(discard=self._num_discard, flat=True)  # pyright: ignore[reportUnknownVariableType, reportAssignmentType, reportUnknownMemberType]
            if use_median:
                best_params = np.median(flat_samples, axis=0)
            else:
                best_index = np.argmax(log_probs)
                best_params = flat_samples[best_index]

            current_model = AlinderModel(
                alpha=best_params[0],
                b=best_params[1],
                c=best_params[2],
                theta0=best_params[3],
                scale_factor=best_params[4],
                rho=best_params[5],
                background=background,
            )

            if initial_model is None:
                initial_model = current_model

            yield SpiralFitDiagnostics(
                initial_model=initial_model,
                final_model=current_model,
                data=initial_density,
                z_mesh=z_mesh,
                vz_mesh=vz_mesh,
                samples=flat_samples,
                log_probs=log_probs,
                num_iterations=num_iterations,
                max_iterations=self._max_iterations,
                converged=converged,
            )

            current_perturbation = current_model.perturbation(z_mesh, vz_mesh)
            new_background = self._smoothing_func(initial_density / current_perturbation)
            new_background = new_background / new_background.sum() * initial_density.sum()
            new_data = current_perturbation * new_background
            fit_ssr = _calculate_rmse_with_mask(initial_density, new_data, mask)
            quality = fit_ssr
            if best_quality <= quality:
                converged = best_model is not None
                if best_model is None:
                    best_model = initial_model
                    best_samples = flat_samples
                break
            best_quality = quality
            background = new_background
            best_model = current_model
            best_samples = flat_samples
            best_probs = log_probs

        assert initial_model is not None
        assert best_model is not None
        assert best_samples is not None
        assert best_probs is not None

        yield SpiralFitDiagnostics(
            initial_model=initial_model,
            final_model=best_model,
            data=initial_density,
            z_mesh=z_mesh,
            vz_mesh=vz_mesh,
            samples=best_samples,
            log_probs=best_probs,
            num_iterations=num_iterations,
            max_iterations=self._max_iterations,
            converged=converged,
        )


class SpiralFitterMinimizer(SpiralFitter):
    """A configuration of the Alinder et al 2023 fitting algorithm.

    This implementation uses scipy.optimize.minimize to fit the parameters.

    Unlike the MCMC fitter, there is no posterior distribution to sample from.
    The ``samples`` and ``log_probs`` fields of :class:`SpiralFitDiagnostics` will
    therefore each contain a single row/entry representing the optimizer solution.
    The ``use_median`` parameter has no effect and is accepted only for API
    compatibility with :class:`SpiralFitterMCMC`.

    """

    def __init__(
        self,
        objective: Literal["prob", "error"] = "prob",
        max_iterations: int | None = 50,
        smoothing_func: _SmoothingFunc | None = None,
        param_lo: dict[_ParamName, float] | None = None,
        param_hi: dict[_ParamName, float] | None = None,
        param_noise: float = 0.01,
        use_density: bool = True,
    ) -> None:
        super().__init__(
            max_iterations=max_iterations,
            smoothing_func=smoothing_func,
            param_lo=param_lo,
            param_hi=param_hi,
            param_noise=param_noise,
            use_density=use_density,
        )
        self._objective: Literal["prob", "error"] = objective

    def fit_spiral_with_background_gen(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        use_median: bool,
        seed: int | None = None,
    ) -> Generator[SpiralFitDiagnostics]:
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
        use_median : bool
            Unused; accepted for API compatibility with :class:`SpiralFitterMCMC`.
        seed : int | None
            The random seed to use or `None` if no seed.

        Returns
        -------
        result : SpiralFitDiagnostics
            The fitting result.

        """
        bounds: list[tuple[float, float]] = list(zip(self._param_lo.tolist(), self._param_hi.tolist(), strict=True))

        mask = _mask(z_mesh, vz_mesh)
        background = initial_background
        best_quality: float = _calculate_rmse_with_mask(initial_density, initial_background, mask)
        initial_model: AlinderModel | None = None
        current_model: AlinderModel | None = None
        best_model: AlinderModel | None = None
        best_samples: onp.Array2D[np.float64] | None = None
        best_probs: onp.Array1D[np.float64] | None = None
        converged: bool = False
        num_iterations: int = 0

        objective_function = ln_prob_opt if self._objective == "prob" else rmse_opt

        while self._max_iterations is None or (num_iterations < self._max_iterations):
            num_iterations += 1
            res = optimize.differential_evolution(
                objective_function,
                bounds=bounds,  # pyright: ignore[reportArgumentType]
                args=(self, initial_density, background, z_mesh, vz_mesh),
                seed=seed,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                tol=1e-5,
                polish=True,
            )

            best_params: onp.Array1D[np.float64] = np.array(res.x, dtype=np.float64)

            # Represent the single optimiser solution as a one-row "sample" array so
            # the rest of the pipeline (which expects MCMC-style arrays) works unchanged.
            flat_samples: onp.Array2D[np.float64] = best_params.reshape(1, _NUM_PARAMETERS)
            # res.fun is the *negated* log-prob (we minimise -ln_prob), so negate back.
            log_probs: onp.Array1D[np.float64] = np.array([-res.fun], dtype=np.float64)

            current_model = AlinderModel(
                alpha=best_params[_ALPHA_INDEX],
                b=best_params[_B_INDEX],
                c=best_params[_C_INDEX],
                theta0=best_params[_THETA0_INDEX],
                scale_factor=best_params[_SCALE_FACTOR_INDEX],
                rho=best_params[_RHO_INDEX],
                background=background,
            )

            if initial_model is None:
                initial_model = current_model

            yield SpiralFitDiagnostics(
                initial_model=initial_model,
                final_model=current_model,
                data=initial_density,
                z_mesh=z_mesh,
                vz_mesh=vz_mesh,
                samples=flat_samples,
                log_probs=log_probs,
                num_iterations=num_iterations,
                max_iterations=self._max_iterations,
                converged=converged,
            )

            current_perturbation = current_model.perturbation(z_mesh, vz_mesh)
            new_background = self._smoothing_func(initial_density / current_perturbation)
            new_background = new_background / new_background.sum() * initial_density.sum()
            new_data = current_perturbation * new_background
            quality = _calculate_rmse_with_mask(initial_density, new_data, mask)

            if best_quality <= quality:
                # Quality has stopped improving — we have converged (or never improved).
                converged = best_model is not None
                if best_model is None:
                    # First iteration was already the best; treat initial model as best.
                    best_model = initial_model
                    best_samples = flat_samples
                    best_probs = log_probs
                break

            best_quality = quality
            background = new_background
            best_model = current_model
            best_samples = flat_samples
            best_probs = log_probs

        assert initial_model is not None
        assert best_model is not None
        assert best_samples is not None
        assert best_probs is not None

        yield SpiralFitDiagnostics(
            initial_model=initial_model,
            final_model=best_model,
            data=initial_density,
            z_mesh=z_mesh,
            vz_mesh=vz_mesh,
            samples=best_samples,
            log_probs=best_probs,
            num_iterations=num_iterations,
            max_iterations=self._max_iterations,
            converged=converged,
        )


def _mask(z_mesh: onp.Array2D[np.float64], vz_mesh: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
    return -special.expit(np.square(z_mesh) + np.square(vz_mesh / 40) - 1.0) + 1.0


def calculate_rmse(
    data: onp.Array2D[np.float64],
    estimate: onp.Array2D[np.float64],
    z_mesh: onp.Array2D[np.float64],
    vz_mesh: onp.Array2D[np.float64],
) -> float:
    mask = _mask(z_mesh, vz_mesh)
    return _calculate_rmse_with_mask(data, estimate, mask)


def _calculate_rmse_with_mask(
    data: onp.Array2D[np.float64], estimate: onp.Array2D[np.float64], mask: onp.Array2D[np.float64]
) -> float:
    return np.sqrt(np.mean(np.square(mask * (data - estimate))))


def ln_prob_mcmc(
    parameters: onp.Array2D[np.float64],
    fitter: SpiralFitter,
    counts: onp.Array2D[np.float64],
    background: onp.Array2D[np.float64],
    z_mesh: onp.Array2D[np.float64],
    vz_mesh: onp.Array2D[np.float64],
) -> onp.Array1D[np.float64]:
    """Log likelihood probability for the MCMC sampler (vectorised over walkers).

    Parameters
    ----------
    parameters : Array2D[f64]
        Parameter matrix of shape ``(num_walkers, 6)``.
    fitter : SpiralFitter
        The fitting configuration.
    counts : Array2D[f64]
        The observed data in a grid.
    background : Array2D[f64]
        The background.
    z_mesh : Array2D[f64]
        The z values for each cell.
    vz_mesh : Array2D[f64]
        The Vz values for each cell.

    Returns
    -------
    log_likelihood : Array1D[f64]
        The log of the likelihood for each walker.

    """
    collection = AlinderModelCollection(parameters=parameters, background=background)
    return fitter.ln_prob(collection, counts, background, z_mesh, vz_mesh)


def ln_prob_opt(
    parameters: onp.Array1D[np.float64],
    fitter: SpiralFitter,
    counts: onp.Array2D[np.float64],
    background: onp.Array2D[np.float64],
    z_mesh: onp.Array2D[np.float64],
    vz_mesh: onp.Array2D[np.float64],
) -> float:
    """Negated log-probability for use with :func:`scipy.optimize.minimize`.

    ``scipy.optimize.minimize`` minimises, so we return ``-ln_prob`` so that
    minimising this objective is equivalent to maximising the log-probability.

    Parameters
    ----------
    parameters : Array1D[f64]
        Parameter vector ``[alpha, b, c, theta0, scale_factor, rho]`` of length 6.
    fitter : SpiralFitter
        The fitting configuration.
    counts : Array2D[f64]
        The observed data in a grid.
    background : Array2D[f64]
        The background.
    z_mesh : Array2D[f64]
        The z values for each cell.
    vz_mesh : Array2D[f64]
        The Vz values for each cell.

    Returns
    -------
    neg_log_likelihood : float
        The negated log-probability.

    """
    collection = AlinderModelCollection(parameters=parameters.reshape(1, _NUM_PARAMETERS), background=background)
    return float(-fitter.ln_prob(collection, counts, background, z_mesh, vz_mesh)[0])


def rmse_opt(
    parameters: onp.Array1D[np.float64],
    fitter: SpiralFitter,
    counts: onp.Array2D[np.float64],
    background: onp.Array2D[np.float64],
    z_mesh: onp.Array2D[np.float64],
    vz_mesh: onp.Array2D[np.float64],
) -> float:
    """Calculate RMSE for use with :func:`scipy.optimize.minimize`.

    Parameters
    ----------
    parameters : Array1D[f64]
        Parameter vector ``[alpha, b, c, theta0, scale_factor, rho]`` of length 6.
    fitter : SpiralFitter
        The fitting configuration.
    counts : Array2D[f64]
        The observed data in a grid.
    background : Array2D[f64]
        The background.
    z_mesh : Array2D[f64]
        The z values for each cell.
    vz_mesh : Array2D[f64]
        The Vz values for each cell.

    Returns
    -------
    rmse : float
        The RMSE between the model and the data.

    """
    _ = fitter
    model = AlinderModel(
        alpha=parameters[0],
        b=parameters[0],
        c=parameters[0],
        theta0=parameters[0],
        scale_factor=parameters[0],
        rho=parameters[0],
        background=background,
    )
    mask = _mask(z_mesh, vz_mesh)
    predict = model.fit(z_mesh, vz_mesh)
    return _calculate_rmse_with_mask(counts, predict, mask)


def generate_initial_background(
    z: onp.Array1D[np.float64],
    vz: onp.Array1D[np.float64],
    z_mesh: onp.Array2D[np.float64],
    vz_mesh: onp.Array2D[np.float64],
    density_scale: float,
) -> onp.Array2D[np.float64]:
    """Generate a symmetric KDE background as the initial background estimate.

    Parameters
    ----------
    z : Array1D[f64]
        The z coordinate of the stars.
    vz : Array1D[f64]
        The Vz velocity of the stars.
    z_mesh : Array2D[f64]
        The z values for each cell.
    vz_mesh : Array2D[f64]
        The Vz values for each cell.
    density_scale : float
        The density normalisation scale.

    Returns
    -------
    background : Array2D[f64]
        The symmetric initial background generated by a Gaussian KDE.

    """
    z_mirrored = np.concatenate([z, z])
    vz_mirrored = np.concatenate([vz, -vz])

    kde = stats.gaussian_kde(np.vstack([z_mirrored, vz_mirrored]), bw_method="scott")

    grid_points = np.vstack([z_mesh.ravel(), vz_mesh.ravel()])
    estimated_background = kde(grid_points).reshape(z_mesh.shape)

    return estimated_background / estimated_background.sum() * density_scale


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
