"""Tests of the spiral fitting algorithm."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from optype import numpy as onp
from scipy.optimize import OptimizeResult

from psnailder._likelihood_utils import ln_likelihood
from psnailder.fit import PSpiralFitter


@pytest.mark.parametrize(
    ("updates", "improve_background", "expected_converged", "expected_iterations", "accepts_background"),
    [
        pytest.param((True,), True, False, 1, True, id="accepted-at-iteration-limit"),
        pytest.param((False,), True, False, 1, False, id="first-update-rejected"),
        pytest.param((True, False), True, True, 2, True, id="accepted-then-rejected"),
        pytest.param((), False, True, 1, False, id="fixed-background"),
    ],
)
def test_background_refinement_result_consistency(
    monkeypatch: pytest.MonkeyPatch,
    updates: tuple[bool, ...],
    improve_background: bool,
    expected_converged: bool,
    expected_iterations: int,
    accepts_background: bool,
) -> None:
    """Keep model, background, and score aligned on every termination path."""
    data = np.array([[1.0, 3.0], [2.0, 4.0]])
    initial_background = np.full_like(data, 2.5)
    degraded_background = np.array([[7.0, 1.0], [1.0, 1.0]])
    z_mesh = np.zeros_like(data)
    vz_mesh = np.zeros_like(data)
    mask = np.ones_like(data)
    # Zero amplitude makes the signal exactly one. The initial score is -1;
    # using data as the background gives score zero, while the other proposal
    # is worse. All backgrounds have the same total, so normalization is inert.
    parameters = np.array([0.0, 0.05, 0.002, 0.0, 40.0, 0.09])
    proposals = iter(updates)
    smoothing_calls = 0

    def optimize_parameters(
        self: PSpiralFitter,
        objective_func: Callable[[onp.Array1D[np.float64]], onp.ToFloat],
        *,
        rng: np.random.Generator,
        warm_start: onp.Array1D[np.float64] | None,
        param_count: int = 1,
    ) -> OptimizeResult:
        return OptimizeResult(x=parameters.copy(), fun=float(objective_func(parameters)), success=True)

    def smooth(arr: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        nonlocal smoothing_calls
        smoothing_calls += 1
        return (data if next(proposals) else degraded_background).copy()

    def make_mask(z: onp.Array2D[np.float64], vz: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        return mask

    monkeypatch.setattr(PSpiralFitter, "_optimize_parameters", optimize_parameters)
    fitter = PSpiralFitter(
        # Rejection cases stop before their budget; the acceptance-only case
        # uses its entire budget. Fixed-background fitting must skip smoothing.
        max_iterations=3 if False in updates else 1,
        smoothing_func=smooth,
        mask_func=make_mask,
    )
    results = list(
        fitter.fit_spiral_with_background_gen(
            data,
            initial_background,
            z_mesh,
            vz_mesh,
            num_components=1,
            winding=1,
            improve_background=improve_background,
        )
    )

    for result in results:
        recomputed = ln_likelihood(data, result.final_model.prediction(), mask)
        assert result.lnl == pytest.approx(recomputed)
        np.testing.assert_array_equal(result.initial_model.background, initial_background)

    final = results[-1]
    assert final.converged is expected_converged
    assert final.num_iterations == expected_iterations
    assert smoothing_calls == len(updates)
    assert final.lnl == pytest.approx(0.0 if accepts_background else -1.0)
    np.testing.assert_array_equal(final.final_model.background, data if accepts_background else initial_background)


@pytest.mark.parametrize("seed", range(3))
def test_gaussian_fit_improvement_opt_prob(seed: int) -> None:
    """ "Test that Gaussian distributed vertical phase space distributions have low improvement.

    Parameters
    ----------
    seed : int
        The random seed.

    """
    rng = np.random.default_rng(seed)

    z = rng.normal(loc=0.0, scale=0.2, size=100_000)
    vz = rng.normal(loc=0.0, scale=20.0, size=100_000)

    fitter = PSpiralFitter()
    dz: float = 0.025
    dvz: float = 2.0
    z_bins = np.arange(-1.2, 1.2 + dz, dz)
    vz_bins = np.arange(-60.0, 60.0 + dvz, dvz)
    res = fitter.fit_spiral(z, vz, z_bins, vz_bins)

    assert res.final_model.pvalue(res.data, fitter._mask_func(res.final_model.z_mesh, res.final_model.vz_mesh)) > 0.05


def smoke() -> None:
    assert False
