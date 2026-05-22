"""A module that implements the phase spiral fitting algorithm described in Alinder et al. 2023."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from psnailder import component, fit, model

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Final
    from optype import numpy as onp


__all__: Final[list[str]] = ["component", "fit", "model"]


type _ObjectiveFunc = Callable[[onp.Array1D[np.float64]], onp.ToFloat]
type _SmoothingFunc = Callable[[onp.Array2D[np.float64]], onp.Array2D[np.float64]]
type _MaskFunc = Callable[[onp.Array2D[np.float64], onp.Array2D[np.float64]], onp.Array2D[np.float64]]


def _main() -> None:
    import time
    from collections.abc import Callable
    from typing import Any, Literal
    from optype import numpy as onp

    import numpy as np
    from phasmix.component import AlinderComponent, GaussianComponent
    from phasmix.mock import MockModel

    from psnailder._internal import ln_likelihood_f64 as ln_likelihood_rust_f64
    from psnailder._internal import PSpiralComponent as PSpiralComponentRust
    from psnailder._internal import PSpiralModel as PSpiralModelRust
    from psnailder._likelihood_utils import ln_likelihood

    def _run_benchmark(name: str, func: Callable[[], Any], *, num_trials: int = 100_000) -> None:  # pyright: ignore[reportExplicitAny]
        start_time = time.perf_counter()
        for _ in range(num_trials):
            _ = func()  # pyright: ignore[reportAny]
        elapsed = time.perf_counter() - start_time

        print(f"Took {elapsed:.2f}s to evaluate {num_trials} {name}: ~{(1000**2 * elapsed / num_trials):.2f} microseconds")

    true_signal = AlinderComponent(
        alpha=0.5,
        b=0.05,
        c=0.002,
        theta0=0.0,
        scale_factor=40.00,
        rho=0.09,
        winding=1,
    )
    background = GaussianComponent(x_scale=1, y_scale=40.0, amplitude=1, variance=0.25)

    mock_model = MockModel((true_signal,), (background,))

    dx: float = 0.05
    dy: float = 1
    x_bins = np.arange(-1.2, 1.2 + dx, dx)
    y_bins = np.arange(-60.0, 60.0 + dy, dy)

    x_centres = 0.5 * (x_bins[:-1] + x_bins[1:])
    y_centres = 0.5 * (y_bins[:-1] + y_bins[1:])

    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    num_particles: int = 500_000
    mock_data = mock_model.mock_grid(x_bins, y_bins)
    density = num_particles * mock_data.density
    background = num_particles * mock_data.background

    mask = fit.create_sigmoid_mask(1.0, 40.0)(x_mesh, y_mesh)

    parameters = np.array(
        [
            [true_signal.alpha, true_signal.b, true_signal.c, true_signal.theta0, true_signal.scale_factor, true_signal.rho],
        ],
    ).reshape((1, 6))

    def wrap_winding_objective(current_winding: Literal[-1, 1]) -> _ObjectiveFunc:
        def _objective(parameters: onp.Array1D[np.float64]) -> float:
            rust_model = PSpiralModelRust(
                [
                    PSpiralComponentRust(
                        alpha=parameters[0],
                        lnb=parameters[1],
                        c=parameters[2],
                        theta0=parameters[3],
                        scale_factor=parameters[4],
                        rho=parameters[5],
                        winding=current_winding,
                        flattening_strength=None,
                    )
                ]
            )
            prediction_flat = background.flatten() * rust_model.perturbation(x_mesh.flatten(), y_mesh.flatten())
            prediction = prediction_flat.reshape(density.shape)

            val = -ln_likelihood(
                density,
                prediction,
                mask,
            )
            return val

        return _objective

    from scipy import optimize

    param_lo: onp.Array1D[np.float64] = np.array([0.0, np.log(0.005), 0.0, -np.pi, np.log(30.0), 0.0])
    param_hi: onp.Array1D[np.float64] = np.array([1.0, np.log(0.1), 0.004, +np.pi, np.log(70.0), 0.18])

    print(f"low = {param_lo}")
    print(f"high = {param_hi}")

    bounds = list(zip(param_lo.tolist(), param_hi.tolist(), strict=True))

    log_parameters = np.copy(parameters.flatten())
    log_parameters[1] = np.log(log_parameters[1])
    log_parameters[4] = np.log(log_parameters[4])

    def _find_minimum_multi_local() -> None:
        for i in range(20):
            res = optimize.minimize(wrap_winding_objective(1), x0=log_parameters, bounds=bounds, method="L-BFGS-B")
            if not res.success:
                res = optimize.minimize(wrap_winding_objective(1), x0=log_parameters, bounds=bounds, method="Nelder-Mead")

    def _find_minimum_de() -> None:
        de_res = optimize.differential_evolution(wrap_winding_objective(1), bounds=bounds)
        _ = de_res

    def _find_minimum_basinhopping() -> None:
        de_res = optimize.basinhopping(wrap_winding_objective(1), x0=log_parameters * (1 + 1e-5))
        print(de_res)
        _ = de_res

    _run_benchmark("multiple local", _find_minimum_multi_local, num_trials=10)
    _run_benchmark("differential evolution", _find_minimum_de, num_trials=2)
    _run_benchmark("basinhopping ", _find_minimum_basinhopping, num_trials=2)


if __name__ == "__main__":
    _main()
