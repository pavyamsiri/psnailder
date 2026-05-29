"""A module that implements the phase spiral fitting algorithm described in Alinder et4 al. 2023."""

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

    from matplotlib import pyplot as plt
    import numpy as np
    from phasmix.component import AlinderComponent, GaussianComponent
    from phasmix.mock import MockModel

    from psnailder._internal import PSpiralFitter as PSpiralFitterRust
    from psnailder.fit import PSpiralFitter as PSpiralFitterPython
    from psnailder._background_utils import generate_initial_background

    signal1 = AlinderComponent(
        alpha=0.5,
        b=0.05,
        c=0.002,
        theta0=-np.pi / 2,
        scale_factor=40.00,
        rho=0.09,
        winding=1,
    )
    signal2 = AlinderComponent(
        alpha=0.5,
        b=0.05,
        c=0.002,
        theta0=np.pi / 2,
        scale_factor=40.00,
        rho=0.09,
        winding=1,
    )
    background_comp = GaussianComponent(x_scale=1, y_scale=40.0, amplitude=1, variance=0.25)

    mock_model = MockModel(
        (signal1,),
        (background_comp,),
    )

    print(f"{len(mock_model._signal)}-arm model")

    num_x_bins = 100
    num_y_bins = 100
    x_edges = np.linspace(-1.2, 1.2, num_x_bins + 1)
    y_edges = np.linspace(-60.0, 60.0, num_y_bins + 1)

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    num_particles: int = 100_000
    print(f"Sampling {num_particles} particles...")
    particles = mock_model.mock_particles(num_particles, x_edges, y_edges, seed=1)
    z_samples = particles.x
    vz_samples = particles.y

    density, _, _ = np.histogram2d(z_samples, vz_samples, bins=(x_edges, y_edges))
    density = density.T

    print("Generating initial background estimate via KDE...")
    initial_background = generate_initial_background(z_samples, vz_samples, x_mesh, y_mesh)
    # Normalize initial background
    initial_background = initial_background / np.sum(initial_background) * np.sum(density)

    mask = fit.create_sigmoid_mask(1.0, 40.0)(x_mesh, y_mesh)

    print("\n--- Rust Version ---")
    fitter_rust = PSpiralFitterRust(num_samples=256, max_iterations=10)
    start_time = time.perf_counter()
    res_rust = fitter_rust.fit_spiral_with_background(
        density.flatten(),
        initial_background.flatten(),
        mask.flatten(),
        x_mesh.flatten(),
        y_mesh.flatten(),
        (num_y_bins, num_x_bins),
    )
    elapsed_rust = time.perf_counter() - start_time
    print(f"Rust took {elapsed_rust:.3f} seconds")
    print(f"Rust iterations: {res_rust.num_iterations}")
    print(f"Rust converged: {res_rust.converged}")
    print(f"Rust final model: {res_rust.final_model}")

    print("\n--- Python Version ---")
    fitter_py = PSpiralFitterPython(num_starts=20, max_iterations=10)
    start_time = time.perf_counter()
    res_py = fitter_py.fit_spiral_with_background(
        density, initial_background, x_mesh, y_mesh, num_components=None, improve_background=True
    )
    elapsed_py = time.perf_counter() - start_time
    print(f"Python took {elapsed_py:.3f} seconds")
    print(f"Python iterations: {res_py.num_iterations}")
    print(f"Python converged: {res_py.converged}")
    print(f"Python final model: {res_py.final_model}")

    rs_background = res_rust.final_background.reshape(x_mesh.shape)
    rs_density = res_rust.final_model.perturbation(x_mesh.flatten(), y_mesh.flatten()).reshape(x_mesh.shape) * rs_background

    fig = plt.figure(figsize=(12, 8))
    # [true density, python density, rust density]
    # [true background, python background, rust background]
    true_density_axes = fig.add_subplot(231)
    py_density_axes = fig.add_subplot(232)
    rs_density_axes = fig.add_subplot(233)
    true_background_axes = fig.add_subplot(234)
    py_background_axes = fig.add_subplot(235)
    rs_background_axes = fig.add_subplot(236)

    true_density_axes.set_title("True density")
    py_density_axes.set_title(f"Python density: lnl = {res_py.lnl}")
    rs_density_axes.set_title(f"Rust density: lnl = {res_rust.lnl}")
    true_background_axes.set_title("True background")
    py_background_axes.set_title("Python background")
    rs_background_axes.set_title("Rust background")

    true_density_axes.pcolormesh(x_mesh, y_mesh, density)
    py_density_axes.pcolormesh(x_mesh, y_mesh, res_py.final_model.prediction())
    rs_density_axes.pcolormesh(x_mesh, y_mesh, rs_density)

    true_background_axes.pcolormesh(x_mesh, y_mesh, initial_background)
    py_background_axes.pcolormesh(x_mesh, y_mesh, res_py.final_model.background)
    rs_background_axes.pcolormesh(x_mesh, y_mesh, rs_background)

    fig.tight_layout()
    fig.savefig("./out.png")
    plt.close(fig)


if __name__ == "__main__":
    _main()
