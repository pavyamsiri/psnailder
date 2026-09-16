"""Tests of the spiral fitting algorithm."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest
from optype import numpy as onp
from scipy.optimize import OptimizeResult

from psnailder._likelihood_utils import ln_likelihood
from psnailder.fit import (
    FitFailure,
    FitFailureReason,
    FitProgress,
    FitSuccess,
    FitTerminationReason,
    PSpiralFitter,
    _DEFAULT_PARAM_HI,
    _DEFAULT_PARAM_LO,
)


def test_bounds_preserve_caller_arrays_and_resolve_defaults() -> None:
    """NaN replacement preserves inputs and retains explicit endpoints."""
    lower = np.array([0.2, np.nan, np.nan, -1.0, np.nan, np.nan])
    upper = np.array([0.8, np.nan, np.nan, 1.0, np.nan, np.nan])
    original_lower = lower.copy()
    original_upper = upper.copy()

    fitter = PSpiralFitter(param_lo=lower, param_hi=upper)

    np.testing.assert_array_equal(lower, original_lower)
    np.testing.assert_array_equal(upper, original_upper)
    np.testing.assert_array_equal(fitter._param_lo, [0.2, 0.005, 0.0, -1.0, 30.0, 0.0])
    np.testing.assert_array_equal(fitter._param_hi, [0.8, 0.1, 0.004, 1.0, 70.0, 0.18])

    lower[:] = -100.0
    upper[:] = 100.0
    np.testing.assert_array_equal(fitter._param_lo, [0.2, 0.005, 0.0, -1.0, 30.0, 0.0])
    np.testing.assert_array_equal(fitter._param_hi, [0.8, 0.1, 0.004, 1.0, 70.0, 0.18])


def test_default_bounds_are_independent() -> None:
    """Each fitter owns bounds separate from other instances and defaults."""
    first = PSpiralFitter()
    second = PSpiralFitter()

    for first_bounds, second_bounds, defaults in (
        (first._param_lo, second._param_lo, _DEFAULT_PARAM_LO),
        (first._param_hi, second._param_hi, _DEFAULT_PARAM_HI),
    ):
        np.testing.assert_array_equal(first_bounds, defaults)
        np.testing.assert_array_equal(second_bounds, defaults)
        assert not np.shares_memory(first_bounds, second_bounds)
        assert not np.shares_memory(first_bounds, defaults)
        assert not np.shares_memory(second_bounds, defaults)


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("shape", [(), (0,), (5,), (7,), (1, 6), (6, 1)])
def test_bounds_reject_invalid_shapes(lower: bool, shape: tuple[int, ...]) -> None:
    # NaNs also ensure shape validation happens before default substitution.
    invalid = np.full(shape, np.nan)
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        PSpiralFitter(param_lo=invalid if lower else None, param_hi=None if lower else invalid)


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_bounds_reject_infinite_endpoints(lower: bool, value: float) -> None:
    invalid = np.full(6, np.nan)
    invalid[0] = value
    with pytest.raises(ValueError, match="must be finite"):
        PSpiralFitter(param_lo=invalid if lower else None, param_hi=None if lower else invalid)


@pytest.mark.parametrize("index", range(6))
def test_bounds_reject_reversed_endpoints(index: int) -> None:
    lower = _DEFAULT_PARAM_LO.copy()
    upper = _DEFAULT_PARAM_HI.copy()
    lower[index] = upper[index] + 1.0
    with pytest.raises(ValueError, match="Lower bounds must not exceed upper bounds"):
        PSpiralFitter(param_lo=lower, param_hi=upper)


def test_bounds_validate_order_after_resolving_defaults() -> None:
    lower = np.full(6, np.nan)
    upper = np.full(6, np.nan)
    upper[0] = -1.0  # Conflicts with the default lower alpha bound of zero.
    with pytest.raises(ValueError, match="Lower bounds must not exceed upper bounds"):
        PSpiralFitter(param_lo=lower, param_hi=upper)


def test_bounds_accept_equal_endpoints() -> None:
    parameters = np.array([0.5, 0.05, 0.002, 0.0, 40.0, 0.09])
    fitter = PSpiralFitter(param_lo=parameters, param_hi=parameters)
    np.testing.assert_array_equal(fitter._param_lo, parameters)
    np.testing.assert_array_equal(fitter._param_hi, parameters)
    assert not np.shares_memory(fitter._param_lo, fitter._param_hi)


@pytest.mark.parametrize("num_components", [1, 2])
def test_warm_start_forwarded_to_differential_evolution(num_components: int) -> None:
    """The public fit API passes the full initial guess to SciPy's x0."""
    parameters = np.tile([0.0, 0.05, 0.002, 0.0, 40.0, 0.09], num_components)
    data = np.ones((2, 2))
    mesh = np.zeros_like(data)
    fitter = PSpiralFitter(max_iterations=1)
    optimizer_result = OptimizeResult(x=parameters.copy(), fun=0.0, success=True)

    with patch("psnailder.fit.optimize.differential_evolution", return_value=optimizer_result) as optimizer:
        result = fitter.fit_spiral_with_background(
            data,
            data.copy(),
            mesh,
            mesh,
            warm_start=parameters,
            num_components=num_components,
            winding=1,
            improve_background=False,
        )

    optimizer.assert_called_once()
    assert optimizer.call_args is not None
    np.testing.assert_array_equal(optimizer.call_args.kwargs["x0"], parameters)
    assert isinstance(result, FitSuccess)
    np.testing.assert_array_equal(result.result.final_model.to_array(), parameters)


@pytest.mark.parametrize("num_components", [1, 2])
def test_warm_start_rejected_with_automatic_component_selection(num_components: int) -> None:
    """Neither a six- nor twelve-parameter guess can select a component count."""
    parameters = np.tile([0.0, 0.05, 0.002, 0.0, 40.0, 0.09], num_components)
    data = np.ones((2, 2))
    mesh = np.zeros_like(data)
    fitter = PSpiralFitter(max_iterations=1)

    with patch("psnailder.fit.optimize.differential_evolution") as optimizer:
        with pytest.raises(ValueError, match="warm start.*component"):
            fitter.fit_spiral_with_background(
                data,
                data.copy(),
                mesh,
                mesh,
                warm_start=parameters,
                num_components=None,
                winding=1,
                improve_background=False,
            )

    optimizer.assert_not_called()


@pytest.mark.parametrize(
    ("updates", "improve_background", "expected_reason", "expected_iterations", "accepts_background"),
    [
        pytest.param((True,), True, FitTerminationReason.ITERATION_LIMIT, 1, True, id="accepted-at-iteration-limit"),
        pytest.param((False,), True, FitTerminationReason.NO_IMPROVEMENT, 1, False, id="first-update-rejected"),
        pytest.param((True, False), True, FitTerminationReason.NO_IMPROVEMENT, 2, True, id="accepted-then-rejected"),
        pytest.param((), False, FitTerminationReason.FIXED_BACKGROUND, 1, False, id="fixed-background"),
    ],
)
def test_background_refinement_result_consistency(
    monkeypatch: pytest.MonkeyPatch,
    updates: tuple[bool, ...],
    improve_background: bool,
    expected_reason: FitTerminationReason,
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

    for event in results[:-1]:
        assert isinstance(event, FitProgress)
        recomputed = ln_likelihood(data, event.model.prediction(), mask)
        assert event.lnl == pytest.approx(recomputed)

    outcome = results[-1]
    assert isinstance(outcome, FitSuccess)
    final = outcome.result
    np.testing.assert_array_equal(final.initial_model.background, initial_background)
    assert final.lnl == pytest.approx(ln_likelihood(data, final.final_model.prediction(), mask))
    assert final.reason is expected_reason
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
    outcome = fitter.fit_spiral(z, vz, z_bins, vz_bins)
    assert isinstance(outcome, FitSuccess)
    res = outcome.result

    assert res.final_model.pvalue(res.data, fitter._mask_func(res.final_model.z_mesh, res.final_model.vz_mesh)) > 0.05


@pytest.mark.parametrize("positive_valid", [True, False])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_winding_selection_retains_finite_candidate(positive_valid: bool, invalid: float) -> None:
    parameters = np.array([0.0, 0.05, 0.002, 0.0, 40.0, 0.09])
    grid = np.ones((2, 2))
    valid = OptimizeResult(x=parameters, fun=0.0)
    failed = OptimizeResult(x=parameters, fun=invalid)
    with patch.object(PSpiralFitter, "_optimize_parameters", side_effect=[valid, failed] if positive_valid else [failed, valid]):
        outcome = PSpiralFitter().fit_spiral_with_background(
            grid,
            grid,
            grid,
            grid,
            num_components=1,
            improve_background=False,
        )
    assert isinstance(outcome, FitSuccess)
    assert outcome.result.final_model.winding == (1 if positive_valid else -1)


@pytest.mark.parametrize("successful_count", [None, 1, 2, 0], ids=["neither", "one-arm", "two-arms", "both"])
@pytest.mark.parametrize("improve", [False, True])
def test_component_selection_retains_success(successful_count: int | None, improve: bool) -> None:
    grid = np.ones((2, 2))
    fitter = PSpiralFitter(max_iterations=1)
    calls = 0

    def optimize(
        objective: Callable[[onp.Array1D[np.float64]], onp.ToFloat],
        *,
        rng: np.random.Generator,
        warm_start: onp.Array1D[np.float64] | None,
        param_count: int = 1,
    ) -> OptimizeResult:
        nonlocal calls
        calls += 1
        parameters = np.tile([0.0, 0.05, 0.002, 0.0, 40.0, 0.09], param_count)
        valid = calls <= 2 and (successful_count == 0 or param_count == successful_count)
        return OptimizeResult(x=parameters, fun=float(objective(parameters)) if valid else np.inf)

    with patch.object(fitter, "_optimize_parameters", side_effect=optimize):
        events = list(fitter.fit_spiral_with_background_gen(grid, grid, grid, grid, winding=-1, improve_background=improve))
    assert len(events) == 1
    outcome = events[0]
    if successful_count is None:
        assert isinstance(outcome, FitFailure)
        assert outcome.reason is FitFailureReason.NO_VALID_CANDIDATE
    else:
        assert isinstance(outcome, FitSuccess)
        result = outcome.result
        # With equal likelihoods, BIC chooses the one-component fit.
        assert result.final_model.parameters.shape == (successful_count or 1, 6)
        assert result.final_model.winding == -1
        assert result.lnl == pytest.approx(0.0)
        assert result.reason is (FitTerminationReason.FAILED_REOPTIMIZATION if improve else FitTerminationReason.FIXED_BACKGROUND)
        assert calls == (3 if improve else 2)


@pytest.mark.parametrize("accepted_first", [False, True])
def test_invalid_background_retains_valid_fit(accepted_first: bool) -> None:
    data = np.array([[1.0, 3.0], [2.0, 4.0]])
    background = np.full_like(data, 2.5)
    mesh = np.zeros_like(data)
    parameters = np.array([0.0, 0.05, 0.002, 0.0, 40.0, 0.09])
    proposals = iter(([data.copy()] if accepted_first else []) + [np.full_like(data, np.nan)])
    fitter = PSpiralFitter(max_iterations=3, smoothing_func=lambda arr: next(proposals))
    scores = [1.0, 0.0] if accepted_first else [1.0]
    with patch.object(fitter, "_optimize_parameters", side_effect=[OptimizeResult(x=parameters, fun=s) for s in scores]):
        events = list(fitter.fit_spiral_with_background_gen(data, background, mesh, mesh, winding=1, num_components=1))
    assert all(isinstance(event, FitProgress) for event in events[:-1])
    outcome = events[-1]
    assert isinstance(outcome, FitSuccess)
    assert outcome.result.reason is FitTerminationReason.INVALID_BACKGROUND_UPDATE
    np.testing.assert_array_equal(outcome.result.final_model.background, data if accepted_first else background)


def test_zero_iteration_budget_yields_failure() -> None:
    grid = np.ones((2, 2))
    events = list(PSpiralFitter(max_iterations=0).fit_spiral_with_background_gen(grid, grid, grid, grid, num_components=1))
    assert len(events) == 1
    assert isinstance(events[0], FitFailure)
