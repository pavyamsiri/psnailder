"""A module that implements  the phase spiral fitting algorithm described in Alinder et al. 2023."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import emcee  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from scipy import ndimage, special, stats

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from typing import Final, Literal

    from optype import numpy as onp


__all__: Final[Sequence[str]] = [
    "AlinderModel",
    "SpiralFitDiagnostics",
    "SpiralFitter",
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
        r_mesh = np.hypot(z_mesh, vz_mesh / self.scale_factor)
        theta_mesh = np.arctan2(vz_mesh, self.scale_factor * z_mesh)

        phase: onp.Array2D[np.float64]
        if self.c != 0.0:
            phase = -0.5 * self.b / self.c + np.sqrt(np.square(0.5 * self.b / self.c) + r_mesh / self.c)
        else:
            phase = r_mesh / self.b
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
        return pert * self.background

    def phase_angle(self) -> float:
        """Calculate the phase angle.

        Returns
        -------
        phase_angle : float
            The phase angle in radians.

        """
        R_TEST: Final[float] = 0.5

        b_over_2c = 0.5 * self.b / self.c
        phase = -b_over_2c + np.sqrt(np.square(b_over_2c) + R_TEST / self.c) if self.c != 0.0 else R_TEST / self.b
        return phase + self.theta0


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

        z_mesh_broadcast = z_mesh[:, :, None]
        vz_mesh_broadcast = vz_mesh[:, :, None]
        r_mesh = np.hypot(z_mesh_broadcast, vz_mesh_broadcast / scale_factor_arr)
        theta_mesh = np.arctan2(vz_mesh_broadcast, scale_factor_arr * z_mesh_broadcast)

        phase: onp.Array3D[np.float64] = np.zeros_like(r_mesh)
        nonzero_mask = c_arr != 0.0
        discriminant = (
            np.square(0.5 * b_arr[nonzero_mask & valid] / c_arr[nonzero_mask & valid])
            + r_mesh[:, :, nonzero_mask & valid] / c_arr[nonzero_mask & valid]
        )
        phase[:, :, nonzero_mask & valid] = -0.5 * b_arr[nonzero_mask & valid] / c_arr[nonzero_mask & valid] + np.sqrt(
            discriminant
        )
        phase[:, :, ~nonzero_mask & valid] = r_mesh[:, :, ~nonzero_mask & valid] / b_arr[~nonzero_mask & valid]
        flattening = special.expit((r_mesh - rho_arr) / 0.1)
        return 1.0 + alpha_arr * flattening * np.cos(theta_mesh - phase - theta0_arr)


class SpiralFitter:
    """A configuration of the Alinder et al 2023 fitting algorithm."""

    def __init__(
        self,
        *,
        num_samples: int = 5_000,
        num_discard: int = 1_000,
        num_walkers: int = 32,
        max_iterations: int | None = 50,
        smoothing_func: _SmoothingFunc | None = None,
        param_lo: dict[_ParamName, float] | None = None,
        param_hi: dict[_ParamName, float] | None = None,
    ) -> None:
        """Construct a fitter.

        Parameters
        ----------
        num_samples : int
            The number of MCMC posterior samples.
        num_discard : int
            The number of MCMC samples to discard/burn in.
        num_walkers : int
            The number of walkers.
        max_iterations : int | None
            The maximum number of background fitting iterations. Set this to `None` to allow
            not bound the number of iterations.
        smoothing_func : _SmoothingFunc | None
            The smoothing function to apply to obtain the background in each iteration. Set this to `None` to
            use the default smoothing.
        param_lo : dict[ParamName, float] | None
            The lower bounds of the parameters. Set this to `None` to use the default bounds.
        param_hi : dict[ParamName, float] | None
            The upper bounds of the parameters. Set this to `None` to use the default bounds.

        """
        self._num_samples: int = num_samples
        self._num_discard: int = num_discard
        self._num_walkers: int = num_walkers
        self._max_iterations: int | None = max_iterations
        self._smoothing_func: _SmoothingFunc = SpiralFitter._default_smoothing if smoothing_func is None else smoothing_func
        self._param_lo: onp.Array1D[np.float64] = np.array([param.lo for param in _DEFAULT_PARAM_BOUNDS], dtype=np.float64)
        self._param_hi: onp.Array1D[np.float64] = np.array([param.hi for param in _DEFAULT_PARAM_BOUNDS], dtype=np.float64)

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

        # Validate lo < hi after merging
        for name in default_lo:
            if default_lo[name] >= default_hi[name]:
                msg = f"param_lo[{name!r}] must be strictly less than param_hi[{name!r}]"
                raise ValueError(msg)

        # Preserve canonical ordering
        self._param_lo = np.array([default_lo[p.name] for p in _DEFAULT_PARAM_BOUNDS], dtype=np.float64)
        self._param_hi = np.array([default_hi[p.name] for p in _DEFAULT_PARAM_BOUNDS], dtype=np.float64)

    @staticmethod
    def _default_smoothing(arr: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        """The default smoothing function to get the background.

        Parameters
        ----------
        arr : Array2D[f64]
            The array to smooth.

        Returns
        -------
        smoothed : Array2D[f64]
            The smoothed array.

        """
        return ndimage.gaussian_filter(arr, sigma=2)

    def _ln_prior(self, models: AlinderModelCollection) -> onp.Array1D[np.float64]:
        """Log prior probability (uniform within bounds, -inf outside).

        Parameters
        ----------
        models : AlinderModelVectorised
            A collection of spiral models.

        Returns
        -------
        log_prior : Array1D[f64]
            The log of the prior in the shape (num_walkers,).

        """
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
        """Log likelihood probability.

        Parameters
        ----------
        models : AlinderModelVectorised
            A collection of spiral models.
        counts : Array2D[f64]
            The observed data in a grid.
        background : Array2D[f64]
            The background.
        z_mesh : Array2D[f64]
            The z value at each mesh point.
        vz_mesh : Array2D[f64]
            The vz value at each mesh point.

        Returns
        -------
        ln_likelihood : Array1D[f64]
            The log of the likelihood.

        """
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
        seed: int | None = None,
    ) -> SpiralFitDiagnostics:
        """Fit a phase spiral to the given vertical phase space distribution.

        Parameters
        ----------
        z : Array1D[f64]
            The z coordinate of the stars.
        vz : Array1D[f64]
            The Vz velocity of the stars.
        z_bins : Array1D[f64]
            The bin edges over z.
        vz_bins : Array1D[f64]
            The bin edges over vz.
        seed : int | None
            The random seed to use or `None` if no seed.

        Returns
        -------
        result : SpiralFitDiagnostics
            The fitting result.

        """
        val = _get_value_from_gen(self.fit_spiral_gen(z, vz, z_bins, vz_bins, seed=seed))
        assert val is not None
        return val

    def fit_spiral_gen(
        self,
        z: onp.Array1D[np.float64],
        vz: onp.Array1D[np.float64],
        z_bins: onp.Array1D[np.float64],
        vz_bins: onp.Array1D[np.float64],
        seed: int | None = None,
    ) -> Generator[SpiralFitDiagnostics]:
        """Fit a phase spiral to the given vertical phase space distribution.

        Parameters
        ----------
        z : Array1D[f64]
            The z coordinate of the stars.
        vz : Array1D[f64]
            The Vz velocity of the stars.
        z_bins : Array1D[f64]
            The bin edges over z.
        vz_bins : Array1D[f64]
            The bin edges over vz.
        seed : int | None
            The random seed to use or `None` if no seed.

        Returns
        -------
        result : SpiralFitDiagnostics
            The fitting result.

        """
        z_centres = 0.5 * (z_bins[:-1] + z_bins[1:])
        vz_centres = 0.5 * (vz_bins[:-1] + vz_bins[1:])
        vz_mesh, z_mesh = np.meshgrid(vz_centres, z_centres)

        density, _, _ = np.histogram2d(z, vz, bins=(z_bins, vz_bins), density=True)
        background = generate_initial_background(z, vz, z_mesh, vz_mesh, density.sum())

        return self.fit_spiral_with_background_gen(density, background, z_mesh, vz_mesh, seed=seed)

    def fit_spiral_with_background(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
        seed: int | None = None,
    ) -> SpiralFitDiagnostics:
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
        seed : int | None
            The random seed to use or `None` if no seed.

        Returns
        -------
        result : SpiralFitDiagnostics
            The fitting result.

        """
        val = _get_value_from_gen(self.fit_spiral_with_background_gen(initial_density, initial_background, z_mesh, vz_mesh, seed))
        assert val is not None
        return val

    def fit_spiral_with_background_gen(
        self,
        initial_density: onp.Array2D[np.float64],
        initial_background: onp.Array2D[np.float64],
        z_mesh: onp.Array2D[np.float64],
        vz_mesh: onp.Array2D[np.float64],
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
        seed : int | None
            The random seed to use or `None` if no seed.

        Returns
        -------
        result : SpiralFitDiagnostics
            The fitting result.

        """
        rng = np.random.default_rng(seed=seed)
        mask = _mask(z_mesh, vz_mesh)
        background = initial_background
        best_quality: float = _calculate_rmse_with_mask(initial_density, initial_background, mask)
        initial_model: AlinderModel | None = None
        current_model: AlinderModel | None = None
        best_model: AlinderModel | None = None
        best_samples: onp.Array2D[np.float64] | None = None
        converged: bool = False
        num_iterations: int = 0
        while self._max_iterations is None or (num_iterations < self._max_iterations):
            num_iterations += 1
            np.random.seed(seed)
            sampler = emcee.EnsembleSampler(
                self._num_walkers,
                _NUM_PARAMETERS,
                ln_prob,
                args=(self, initial_density, background, z_mesh, vz_mesh),
                vectorize=True,
            )
            np.random.seed(None)

            p0 = rng.uniform(self._param_lo, self._param_hi, size=(self._num_walkers, _NUM_PARAMETERS))
            sampler.run_mcmc(p0, self._num_samples, progress=False)  # pyright: ignore[reportUnknownMemberType]

            flat_samples: onp.Array2D[np.float64] = sampler.get_chain(discard=self._num_discard, flat=True)  # pyright: ignore[reportUnknownVariableType, reportAssignmentType, reportUnknownMemberType]
            log_probs: onp.Array1D[np.float64] = sampler.get_log_prob(discard=self._num_discard, flat=True)  # pyright: ignore[reportUnknownVariableType, reportAssignmentType, reportUnknownMemberType]
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
                num_iterations=num_iterations,
                max_iterations=self._max_iterations,
                converged=converged,
            )

            current_perturbation = current_model.perturbation(z_mesh, vz_mesh)
            new_background = self._smoothing_func(initial_density / current_perturbation)
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

        assert initial_model is not None
        assert best_model is not None
        assert best_samples is not None

        yield SpiralFitDiagnostics(
            initial_model=initial_model,
            final_model=best_model,
            data=initial_density,
            z_mesh=z_mesh,
            vz_mesh=vz_mesh,
            samples=best_samples,
            num_iterations=num_iterations,
            max_iterations=self._max_iterations,
            converged=converged,
        )


def _mask(z_mesh: onp.Array2D[np.float64], vz_mesh: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
    """Construct a mask used when evaluating quality.

    Parameters
    ----------
    z_mesh : Array2D[f64]
        The z values for each cell.
    vz_mesh : Array2D[f64]
        The Vz values for each cell.

    Returns
    -------
    mask : Array2D[f64]
        The mask.

    """
    return -special.expit(np.square(z_mesh) + np.square(vz_mesh / 40) - 1.0) + 1.0


def calculate_rmse(
    data: onp.Array2D[np.float64],
    estimate: onp.Array2D[np.float64],
    z_mesh: onp.Array2D[np.float64],
    vz_mesh: onp.Array2D[np.float64],
) -> float:
    """Calculate the root mean sum of square residuals.

    Parameters
    ----------
    data : Array2D[f64]
        The data to compare to.
    estimate : Array2D[f64]
        The estimate/prediction.
    z_mesh : Array2D[f64]
        The z values for each cell.
    vz_mesh : Array2D[f64]
        The Vz values for each cell.

    Returns
    -------
    rmse : float
        The root mean sum of square residuals.

    """
    mask = _mask(z_mesh, vz_mesh)
    return _calculate_rmse_with_mask(data, estimate, mask)


def _calculate_rmse_with_mask(
    data: onp.Array2D[np.float64], estimate: onp.Array2D[np.float64], mask: onp.Array2D[np.float64]
) -> float:
    """Calculate the root mean sum of square residuals.

    Parameters
    ----------
    data : Array2D[f64]
        The data to compare to.
    estimate : Array2D[f64]
        The estimate/prediction.
    mask : Array2D[f64]
        The mask.

    Returns
    -------
    rmse : float
        The root mean sum of square residuals.

    """
    return np.sqrt(np.mean(np.square(mask * (data - estimate))))


def ln_prob(
    parameters: onp.Array2D[np.float64],
    fitter: SpiralFitter,
    counts: onp.Array2D[np.float64],
    background: onp.Array2D[np.float64],
    z_mesh: onp.Array2D[np.float64],
    vz_mesh: onp.Array2D[np.float64],
) -> onp.Array1D[np.float64]:
    """Log likelihood probability (uniform within bounds, -inf outside).

    Parameters
    ----------
    parameters : Array2D[f64]
        Parameter vector [alpha, b, c, theta0, S, rho], in the shape (num_walkers, 6).
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
        The log of the likelihood.

    """
    collection = AlinderModelCollection(parameters=parameters, background=background)
    return fitter.ln_prob(collection, counts, background, z_mesh, vz_mesh)


def generate_initial_background(
    z: onp.Array1D[np.float64],
    vz: onp.Array1D[np.float64],
    z_mesh: onp.Array2D[np.float64],
    vz_mesh: onp.Array2D[np.float64],
    density_scale: float,
) -> onp.Array2D[np.float64]:
    """

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
        The last yielded value or `None` if the generator does not have any values.

    """
    val: T | None = None
    for inner in gen:
        val = inner
    return val
