from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from phasmix.component import AlinderComponent

# Add benchmarks to path to import TikTakOpt and helpers
sys.path.append(str(Path(__file__).parent.parent / "benchmarks"))
from benchmark_optimize import TikTakOpt, _create_objective


@given(
    alpha=st.floats(0.1, 0.9),  # Avoid extreme edges
    b=st.floats(0.01, 0.09),
    c=st.floats(0.0005, 0.0035),
    theta0=st.floats(-np.pi + 0.1, np.pi - 0.1),
    S=st.floats(35.0, 65.0),
    rho=st.floats(0.02, 0.16),
)
@settings(max_examples=10, deadline=None)  # Small number of examples as optimization is slow
def test_tiktak_recovery(alpha: float, b: float, c: float, theta0: float, S: float, rho: float) -> None:
    """Property based test to check if TikTakOpt can recover AlinderComponent parameters."""
    signal_comp = AlinderComponent(
        alpha=alpha,
        b=b,
        c=c,
        theta0=theta0,
        scale_factor=S,
        rho=rho,
        winding=1,
    )
    true_params = np.array([alpha, b, c, theta0, S, rho])
    objective = _create_objective(signal_comp)

    # Standard bounds used in benchmarks
    lb = np.array([0.0, 0.005, 0.0, -np.pi, 30.0, 0.0])
    ub = np.array([1.0, 0.1, 0.004, +np.pi, 70.0, 0.18])

    # We use more sobol points for testing to ensure robustness
    # 2**12 = 4096 points, which is the default for a thorough search
    # n_star=64 provides more local searches to escape local minima
    opt = TikTakOpt(objective, num_sobol=2**12, n_star=64, seed=42)

    # Use a random guess within bounds as the "good guess" counterpart
    # TikTak should be robust to this
    rng = np.random.default_rng(42)
    guess = rng.uniform(lb, ub)

    estimated, ll, nfev = opt.minimize(guess, lb, ub)

    # Check if estimated parameters are close to truth
    # Relaxing atol further as some parameters (like theta0 and rho) can be degenerate
    # and likelihood-based recovery is not always perfect.
    np.testing.assert_allclose(estimated, true_params, rtol=2e-2, atol=1e-2, err_msg=f"Failed to recover parameters.\nTruth: {true_params}\nEstimated: {estimated}\nLL: {ll}\nNFEV: {nfev}")
