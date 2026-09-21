"""Tests of the spiral fitting algorithm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from optype import numpy as onp
from scipy.optimize import Bounds, OptimizeResult

from psnailder._likelihood_utils import ln_likelihood
from psnailder.fit import (
    FitFailure,
    FitFailureReason,
    FitProgress,
    FitSuccess,
    FitTerminationReason,
    PSpiralFitter,
    _OptimizationResult,
)


@pytest.mark.parametrize("num_components", [1, 2])
def test_warm_start_forwarded_to_differential_evolution(num_components: int) -> None:
    """The public fit API passes the full initial guess to SciPy's x0."""
    parameters = np.tile([0.0, 0.05, 0.002, 0.0, 40.0, 0.09], num_components)
    data = np.ones((2, 2))
    mesh = np.zeros_like(data)
    fitter = PSpiralFitter(max_iterations=1)
    rng = np.random.default_rng(42)
    optimizer_result = OptimizeResult(x=parameters.copy(), fun=0.0, success=True)

    with patch("psnailder.fit.optimize.differential_evolution", return_value=optimizer_result) as optimizer:
        result = fitter.fit_spiral_with_background(
            data,
            data.copy(),
            mesh,
            mesh,
            warm_start=parameters,
            rng=rng,
            num_components=num_components,
            winding=1,
            improve_background=False,
        )

    optimizer.assert_called_once()
    assert optimizer.call_args is not None
    expected_guess = parameters.copy()
    expected_guess[::6] = 1e-12  # pack moves endpoints slightly inside their intervals.
    np.testing.assert_array_equal(optimizer.call_args.kwargs["x0"], expected_guess)
    assert optimizer.call_args.kwargs["rng"] is rng
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
        pytest.param((), False, FitTerminationReason.FIXED_BACKGROUND, 0, False, id="fixed-background"),
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
        guess: onp.Array1D[np.float64] | None,
        bounds: Bounds,
    ) -> _OptimizationResult:
        return _OptimizationResult(parameters=parameters.copy(), cost=float(objective_func(parameters)), success=True)

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
    assert [event.iteration for event in results if isinstance(event, FitProgress)] == (
        [0, 1] if accepts_background else ([0] if improve_background else [])
    )

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
    outcome = fitter.fit_spiral(z, vz, z_bins, vz_bins, rng=rng)
    assert isinstance(outcome, FitSuccess)
    res = outcome.result

    assert res.final_model.pvalue(res.data, fitter._mask_func(res.final_model.z_mesh, res.final_model.vz_mesh)) > 0.05


@pytest.mark.parametrize("positive_valid", [True, False])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_winding_selection_retains_finite_candidate(positive_valid: bool, invalid: float) -> None:
    parameters = np.array([0.0, 0.05, 0.002, 0.0, 40.0, 0.09])
    grid = np.ones((2, 2))
    valid = _OptimizationResult(parameters=parameters, cost=0.0, success=True)
    failed = _OptimizationResult(parameters=parameters, cost=invalid, success=False)
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
        guess: onp.Array1D[np.float64] | None,
        bounds: Bounds,
    ) -> _OptimizationResult:
        nonlocal calls
        calls += 1
        param_count = bounds.lb.size // 6
        parameters = np.tile([0.0, 0.05, 0.002, 0.0, 40.0, 0.09], param_count)
        valid = calls <= 2 and (successful_count == 0 or param_count == successful_count)
        return _OptimizationResult(parameters=parameters, cost=float(objective(parameters)) if valid else np.inf, success=valid)

    with patch.object(fitter, "_optimize_parameters", side_effect=optimize):
        events = list(fitter.fit_spiral_with_background_gen(grid, grid, grid, grid, winding=-1, improve_background=improve))
    assert len(events) == (2 if improve and successful_count is not None else 1)
    assert all(isinstance(event, FitProgress) for event in events[:-1])
    outcome = events[-1]
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
    with patch.object(
        fitter,
        "_optimize_parameters",
        side_effect=[_OptimizationResult(parameters=parameters, cost=s, success=True) for s in scores],
    ):
        events = list(fitter.fit_spiral_with_background_gen(data, background, mesh, mesh, winding=1, num_components=1))
    assert all(isinstance(event, FitProgress) for event in events[:-1])
    outcome = events[-1]
    assert isinstance(outcome, FitSuccess)
    assert outcome.result.reason is FitTerminationReason.INVALID_BACKGROUND_UPDATE
    np.testing.assert_array_equal(outcome.result.final_model.background, data if accepted_first else background)


def test_zero_refinement_budget_retains_initial_fit() -> None:
    grid = np.ones((2, 2))
    parameters = np.array([0.0, 0.05, 0.002, 0.0, 40.0, 0.09])
    with patch.object(
        PSpiralFitter, "_optimize_parameters", return_value=_OptimizationResult(parameters=parameters, cost=0.0, success=True)
    ) as optimizer:
        events = list(
            PSpiralFitter(max_iterations=0).fit_spiral_with_background_gen(
                grid,
                grid,
                grid,
                grid,
                num_components=1,
                winding=1,
            )
        )
    optimizer.assert_called_once()
    assert len(events) == 2
    assert isinstance(events[0], FitProgress)
    assert events[0].iteration == 0
    assert isinstance(events[-1], FitSuccess)
    assert events[-1].result.reason is FitTerminationReason.ITERATION_LIMIT
    assert events[-1].result.num_iterations == 0


def test_negative_refinement_budget_is_invalid() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        PSpiralFitter(max_iterations=-1)


@pytest.mark.parametrize("num_components", [None, 1, 2])
@pytest.mark.parametrize("winding", [None, -1])
def test_selection_only_happens_once(
    num_components: int | None,
    winding: Literal[-1, 1] | None,
) -> None:
    data = np.array([[1.0, 3.0], [2.0, 4.0]])
    background = np.full_like(data, 2.5)
    mesh = np.zeros_like(data)
    parameters = np.array([0.0, 0.05, 0.002, 0.0, 40.0, 0.09])
    rng = np.random.default_rng(42)
    seen_rngs: list[np.random.Generator] = []
    guesses: list[onp.Array1D[np.float64] | None] = []
    scores: list[float] = []

    def optimize(
        objective: Callable[[onp.Array1D[np.float64]], onp.ToFloat],
        *,
        rng: np.random.Generator,
        guess: onp.Array1D[np.float64] | None,
        bounds: Bounds,
    ) -> _OptimizationResult:
        seen_rngs.append(rng)
        guesses.append(guess)
        params = np.tile(parameters, bounds.lb.size // 6)
        score = float(objective(params))
        scores.append(score)
        return _OptimizationResult(parameters=params, cost=score, success=True)

    def mask(z: onp.Array2D[np.float64], vz: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        return np.ones_like(z)

    fitter = PSpiralFitter(max_iterations=1, smoothing_func=lambda arr: arr.copy(), mask_func=mask)
    with patch.object(fitter, "_optimize_parameters", side_effect=optimize) as optimizer:
        with patch.object(fitter, "_mask_func", wraps=mask) as mask_factory:
            events = list(
                fitter.fit_spiral_with_background_gen(
                    data,
                    background,
                    mesh,
                    mesh,
                    num_components=num_components,
                    winding=winding,
                    rng=rng,
                )
            )
    initial_calls = (2 if num_components is None else 1) * (2 if winding is None else 1)
    assert optimizer.call_count == initial_calls + 1
    mask_factory.assert_called_once()
    assert all(seen is rng for seen in seen_rngs)
    assert all(guess is None for guess in guesses[:-1])
    expected_guess = np.tile(parameters, num_components or 1)
    expected_guess[::6] = 1e-12
    np.testing.assert_array_equal(guesses[-1], expected_guess)
    # Re-optimization evaluates the proposed background, not the initial one again.
    assert scores[:-1] == pytest.approx([1.0] * initial_calls)
    assert scores[-1] == pytest.approx(0.0)
    assert len(events) == 3
    for step, event in enumerate(events[:-1]):
        assert isinstance(event, FitProgress)
        assert event.iteration == step
        assert event.model.winding == (winding or 1)
        assert event.lnl == pytest.approx(ln_likelihood(data, event.model.prediction(), np.ones_like(data)))
    outcome = events[-1]
    assert isinstance(outcome, FitSuccess)
    assert outcome.result.num_iterations == 1
    assert outcome.result.reason is FitTerminationReason.ITERATION_LIMIT


def test_sample_wrapper_forwards_rng_and_outcome() -> None:
    fitter = PSpiralFitter(max_iterations=0)
    rng = np.random.default_rng(42)
    samples = np.array([-0.5, 0.5])
    bins = np.array([-1.0, 0.0, 1.0])
    failure = FitFailure(FitFailureReason.NO_VALID_CANDIDATE, "No candidate.")
    with patch("psnailder.fit.generate_initial_background", return_value=np.ones((2, 2))):
        with patch.object(fitter, "fit_spiral_with_background_gen", return_value=iter([failure])) as fit:
            outcome = fitter.fit_spiral(samples, samples, bins, bins, rng=rng)
    assert outcome is failure
    fit.assert_called_once()
    assert fit.call_args is not None
    assert fit.call_args.kwargs["rng"] is rng


@pytest.mark.parametrize("background_scale", [0.01, 1.0, 100.0])
def test_normalized_two_arm_amplitude_recovery(background_scale: float) -> None:
    """Recover amplitude independently of the supplied background normalization."""
    from psnailder.fit import ParameterBounds
    from psnailder.model import PSpiralModel

    z_mesh, vz_mesh = np.meshgrid(np.linspace(-1.2, 1.2, 40), np.linspace(-60.0, 60.0, 40))
    background = np.exp(-0.5 * (z_mesh**2 + (vz_mesh / 40.0) ** 2) / 0.25)
    parameters = np.array([[0.5, 0.05, 0.002, angle, 40.0, 0.09] for angle in (-np.pi / 2, np.pi / 2)])
    truth = PSpiralModel(parameters, z_mesh, vz_mesh, background)
    data = truth.prediction()
    data *= 100_000 / data.sum()
    supplied_background = background * background_scale
    original_background = supplied_background.copy()
    fitter = PSpiralFitter(
        bounds=[ParameterBounds(b=0.05, c=0.002, theta0=angle, scale_factor=40.0, rho=0.09) for angle in (-np.pi / 2, np.pi / 2)],
        mask_func=lambda z, vz: np.ones_like(z),
    )
    outcome = fitter.fit_spiral_with_background(
        data,
        supplied_background,
        z_mesh,
        vz_mesh,
        num_components=2,
        winding=1,
        improve_background=False,
        rng=np.random.default_rng(12),
    )
    assert isinstance(outcome, FitSuccess)
    result = outcome.result
    np.testing.assert_allclose(result.final_model.parameters[:, 0], 0.5, atol=1e-5)
    assert result.final_model.prediction().sum() == pytest.approx(data.sum())
    assert result.lnl == pytest.approx(ln_likelihood(data, result.final_model.prediction(), np.ones_like(data)))
    assert result.lnl == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_array_equal(supplied_background, original_background)
    assert result.final_model.background.sum() < data.sum()
