from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from phasmix.component import AlinderComponent

# Add benchmarks to path to import OPTIMIZER_REGISTRY and helpers
sys.path.append(str(Path(__file__).parent.parent / "benchmarks"))
from benchmark_optimize import OPTIMIZER_REGISTRY, _create_objective

if TYPE_CHECKING:
    from benchmark_optimize import Optimizer


@pytest.mark.parametrize("opt_name", ["de", "multistart", "basinhopping", "tiktak"])
@given(
    alpha=st.floats(0.1, 0.9),
    b=st.floats(0.01, 0.09),
    c=st.floats(0.0005, 0.0035),
    theta0=st.floats(-np.pi + 0.1, np.pi - 0.1),
    S=st.floats(35.0, 65.0),
    rho=st.floats(0.02, 0.16),
)
@settings(max_examples=5, deadline=None)
def test_optimizer_recovery(opt_name: str, alpha: float, b: float, c: float, theta0: float, S: float, rho: float) -> None:
    """Check if the selected optimizer can recover AlinderComponent parameters."""
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

    optimizer: Optimizer = OPTIMIZER_REGISTRY[opt_name](objective)
    
    # Use a random guess within bounds
    rng = np.random.default_rng(42)
    guess = rng.uniform(lb, ub)

    estimated, ll, nfev = optimizer.minimize(guess, lb, ub)

    # We use the same relaxed tolerances as in the tiktak test
    # since these are common for likelihood-based recovery in this problem.
    np.testing.assert_allclose(
        estimated, 
        true_params, 
        rtol=2e-2, 
        atol=1e-2, 
        err_msg=f"Optimizer '{opt_name}' failed to recover parameters.\nTruth: {true_params}\nEstimated: {estimated}\nLL: {ll}\nNFEV: {nfev}"
    )
