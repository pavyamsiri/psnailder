"""A module that implements the phase spiral fitting algorithm described in Alinder et al. 2023."""

from __future__ import annotations

from typing import TYPE_CHECKING

from psnailder import component, fit, model

if TYPE_CHECKING:
    from typing import Final

__all__: Final[list[str]] = ["component", "fit", "model"]


def _main() -> None:
    import time
    from collections.abc import Callable
    from typing import Any

    import numpy as np
    from phasmix.component import AlinderComponent, GaussianComponent
    from phasmix.mock import MockModel

    from psnailder._internal import ln_likelihood_f64 as ln_likelihood_rust_f64
    from psnailder._internal import PSpiralComponent as PSpiralComponentRust
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
            [
                true_signal.alpha,
                true_signal.b,
                true_signal.c,
                true_signal.theta0 + np.pi,
                true_signal.scale_factor,
                true_signal.rho,
            ],
        ],
    )
    true_model = model.PSpiralModel(parameters, x_mesh, y_mesh, background, winding=1)
    components = [
        component.PSpiralComponent.from_array(parameters[0, :], winding=1),
        component.PSpiralComponent.from_array(parameters[1, :], winding=1),
    ]
    prediction = true_model.prediction()

    rust_component = PSpiralComponentRust(
        alpha=parameters[0, 0],
        b=parameters[0, 1],
        c=parameters[0, 2],
        theta0=parameters[0, 3],
        scale_factor=parameters[0, 4],
        rho=parameters[0, 5],
        winding=1,
        flattening_strength=None,
    )

    def _scalar_prediction() -> None:
        signal = np.full_like(background, -np.inf)
        for comp in components:
            signal = np.maximum(signal, comp.perturbation(x_mesh, y_mesh))
        _ = background * signal

    # _run_benchmark("likelihoods", lambda: ln_likelihood(density, prediction, mask))
    _run_benchmark(
        "likelihoods (rust f64)", lambda: ln_likelihood_rust_f64(density.flatten(), prediction.flatten(), mask.flatten())
    )
    _run_benchmark("predictions (rust, 1 component)", lambda: rust_component.perturbation(x_mesh.flatten(), y_mesh.flatten()))
    _run_benchmark("predictions (numpy, 1 component)", lambda: components[0].perturbation(x_mesh, y_mesh))
    # _run_benchmark("scalar predictions", _scalar_prediction)


if __name__ == "__main__":
    _main()
