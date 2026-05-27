"""A module to study the reliability of global minimization."""

from __future__ import annotations
import time
import numpy as np
from phasmix.component import AlinderComponent, GaussianComponent
from phasmix.mock import MockModel
from psnailder._internal import PSpiralFitter as PSpiralFitterRust
from psnailder.fit import PSpiralFitter as PSpiralFitterPython
from psnailder.fit import create_sigmoid_mask
from psnailder._background_utils import generate_initial_background
from psnailder._likelihood_utils import ln_likelihood

def run_study():
    # Setup mock data (consistent with main benchmark)
    signal1 = AlinderComponent(alpha=0.5, b=0.05, c=0.002, theta0=-np.pi / 2, scale_factor=40.0, rho=0.09, winding=1)
    background_comp = GaussianComponent(x_scale=1, y_scale=40.0, amplitude=1, variance=0.25)
    mock_model = MockModel((signal1,), (background_comp,))
    
    num_x_bins, num_y_bins = 100, 100
    x_edges = np.linspace(-1.2, 1.2, num_x_bins + 1)
    y_edges = np.linspace(-60.0, 60.0, num_y_bins + 1)
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    particles = mock_model.mock_particles(1_000_000, x_edges, y_edges, seed=1)
    density, _, _ = np.histogram2d(particles.x, particles.y, bins=(x_edges, y_edges))
    density = density.T
    
    initial_background = generate_initial_background(particles.x, particles.y, x_mesh, y_mesh)
    initial_background = initial_background / np.sum(initial_background) * np.sum(density)
    mask = create_sigmoid_mask(1.0, 40.0)(x_mesh, y_mesh)
    
    bg_lnl = ln_likelihood(density, initial_background, mask)
    print(f"Initial Background lnl: {bg_lnl:.4f}")

    # True model lnl
    true_signal = signal1(x_mesh, y_mesh)
    true_prediction = initial_background * true_signal
    true_lnl = ln_likelihood(density, true_prediction, mask)
    print(f"True Model lnl: {true_lnl:.4f}")

    print(f"{'Method':<20} | {'lnl':<15} | {'Time (s)':<10} | {'Status'}")
    print("-" * 60)

    # 1. Python DE (Baseline)
    fitter_py = PSpiralFitterPython(num_starts=20, max_iterations=1) # Single iteration for pure global search check
    start = time.perf_counter()
    res_py = fitter_py.fit_spiral_with_background(density, initial_background, x_mesh, y_mesh, improve_background=False)
    elapsed = time.perf_counter() - start
    print(f"{'Python DE':<20} | {res_py.lnl:<15.4f} | {elapsed:<10.3f} | {'OK'}")

    # 2. Rust TikTak with various sample sizes
    sample_sizes = [256, 512, 1024, 2048, 4096]
    for n in sample_sizes:
        fitter_rs = PSpiralFitterRust(num_samples=n, max_iterations=1)
        start = time.perf_counter()
        res_rs = fitter_rs.fit_spiral_with_background(
            density.flatten(), initial_background.flatten(), mask.flatten(),
            x_mesh.flatten(), y_mesh.flatten(), (num_y_bins, num_x_bins)
        )
        elapsed = time.perf_counter() - start
        print(f"{f'Rust TikTak({n})':<20} | {res_rs.lnl:<15.4f} | {elapsed:<10.3f} | {'OK'}")

if __name__ == "__main__":
    run_study()
