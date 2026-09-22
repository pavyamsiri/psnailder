"""Sample script to test parity between python and rust."""

# The aim of this script to print out diagnostics regarding the test.
# ruff: noqa: T201

from __future__ import annotations

import time

import numpy as np
from matplotlib import pyplot as plt
from phasmix.component import AlinderComponent, GaussianComponent
from phasmix.mock import MockModel

from psnailder import fit
from psnailder._background_utils import generate_initial_background
from psnailder._likelihood_utils import ln_likelihood
from psnailder.fit import PSpiralFitter


def _main() -> None:
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
        (signal1, signal2),
        (background_comp,),
    )

    print(f"{len(mock_model.signal)}-arm model")

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

    print(f"ln likelihood (null) = {ln_likelihood(density, initial_background, mask)}")

    print("\n--- Rust Version ---")

    fitter_rust = PSpiralFitter(backend="rust", max_iterations=10)
    start_time = time.perf_counter()
    outcome_rust = fitter_rust.fit_spiral_with_background(
        density, initial_background, x_mesh, y_mesh, num_components=None, improve_background=True, rng=np.random.default_rng(1)
    )
    elapsed_rust = time.perf_counter() - start_time
    if isinstance(outcome_rust, fit.FitFailure):
        print(f"rustthon fit failed: {outcome_rust.reason}: {outcome_rust.message}")
        return
    res_rust = outcome_rust.result
    print(f"Rust took {elapsed_rust:.3f} seconds")
    print(f"Rust refinement attempts: {res_rust.num_iterations}")
    print(f"Rust termination: {res_rust.reason}")
    print(f"Rust final model: {res_rust.final_model}")
    print(f"Rust final lnl: {res_rust.lnl}")
    print(f"Rust pvalue: {res_rust.final_model.pvalue(density, mask)}")

    print("\n--- Python Version ---")
    fitter_py = PSpiralFitter(backend="python", max_iterations=10)
    start_time = time.perf_counter()
    outcome_py = fitter_py.fit_spiral_with_background(
        density, initial_background, x_mesh, y_mesh, num_components=None, improve_background=True, rng=np.random.default_rng(1)
    )
    elapsed_py = time.perf_counter() - start_time
    if isinstance(outcome_py, fit.FitFailure):
        print(f"Python fit failed: {outcome_py.reason}: {outcome_py.message}")
        return
    res_py = outcome_py.result
    print(f"Python took {elapsed_py:.3f} seconds")
    print(f"Python refinement attempts: {res_py.num_iterations}")
    print(f"Python termination: {res_py.reason}")
    print(f"Python final model: {res_py.final_model}")
    print(f"Python final lnl: {res_py.lnl}")
    print(f"Python pvalue: {res_py.final_model.pvalue(density, mask)}")

    rs_background = res_rust.final_model.background.reshape(x_mesh.shape)
    rs_density = res_rust.final_model.prediction()

    fig = plt.figure(figsize=(12, 8))  # pyright: ignore[reportUnknownMemberType]
    # [true density, python density, rust density]
    # [true background, python background, rust background]
    true_density_axes = fig.add_subplot(231)
    py_density_axes = fig.add_subplot(232)
    rs_density_axes = fig.add_subplot(233)
    true_background_axes = fig.add_subplot(234)
    py_background_axes = fig.add_subplot(235)
    rs_background_axes = fig.add_subplot(236)

    _ = true_density_axes.set_title("True density")  # pyright: ignore[reportUnknownMemberType]
    _ = py_density_axes.set_title(f"Python density: lnl = {res_py.lnl}")  # pyright: ignore[reportUnknownMemberType]
    _ = rs_density_axes.set_title(f"Rust density: lnl = {res_rust.lnl}")  # pyright: ignore[reportUnknownMemberType]
    _ = true_background_axes.set_title("True background")  # pyright: ignore[reportUnknownMemberType]
    _ = py_background_axes.set_title("Python background")  # pyright: ignore[reportUnknownMemberType]
    _ = rs_background_axes.set_title("Rust background")  # pyright: ignore[reportUnknownMemberType]

    _ = true_density_axes.pcolormesh(x_mesh, y_mesh, density)  # pyright: ignore[reportUnknownMemberType]
    _ = py_density_axes.pcolormesh(x_mesh, y_mesh, res_py.final_model.prediction())  # pyright: ignore[reportUnknownMemberType]
    _ = rs_density_axes.pcolormesh(x_mesh, y_mesh, rs_density)  # pyright: ignore[reportUnknownMemberType]

    _ = true_background_axes.pcolormesh(x_mesh, y_mesh, initial_background)  # pyright: ignore[reportUnknownMemberType]
    _ = py_background_axes.pcolormesh(x_mesh, y_mesh, res_py.final_model.background)  # pyright: ignore[reportUnknownMemberType]
    _ = rs_background_axes.pcolormesh(x_mesh, y_mesh, rs_background)  # pyright: ignore[reportUnknownMemberType]

    fig.tight_layout()
    fig.savefig("./out.png")  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


if __name__ == "__main__":
    _main()
