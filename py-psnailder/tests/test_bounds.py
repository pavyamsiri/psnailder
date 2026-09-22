"""Named bounds, reduced vectors, and component selection."""

# Test names describe the behavior under test; separate docstrings are optional.
# ruff: noqa: D103

from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from psnailder._likelihood_utils import ln_likelihood
from psnailder.fit import FitSuccess, Fixed, Interval, ParameterBounds, PSpiralFitter, create_sigmoid_mask
from psnailder.model import PSpiralModel
from psnailder.param_layout import ParameterLayout


def arm(alpha: Interval | Fixed | None = None, theta: float = 0.0) -> ParameterBounds:
    if alpha is None:
        alpha = Fixed(0.0)
    return ParameterBounds(alpha=alpha, b=0.05, c=(0.0, 0.0), theta0=theta, scale_factor=40.0, rho=0.09)


@pytest.mark.parametrize("count", [1, 2, None])
def test_single_bounds_broadcasts(count: int | None) -> None:
    grid = np.ones((2, 2))
    fitter = PSpiralFitter(bounds=arm())
    with patch("psnailder.fit.optimize.differential_evolution") as optimizer:
        outcome = fitter.fit_spiral_with_background(
            grid,
            grid,
            grid,
            grid,
            num_components=count,
            winding=1,
            improve_background=False,
        )
    optimizer.assert_not_called()
    assert isinstance(outcome, FitSuccess)
    expected = np.tile([0.0, 0.05, 0.0, 0.0, 40.0, 0.09], (count or 1, 1))
    np.testing.assert_array_equal(outcome.result.final_model.parameters, expected)


def test_broadcast_free_parameters_remain_independent() -> None:
    grid = np.ones((2, 2))
    fitter = PSpiralFitter(bounds=arm(Interval(0.1, 0.9)))
    guess = np.array([0.2, 0.05, 0.0, 0.0, 40.0, 0.09, 0.7, 0.05, 0.0, 0.0, 40.0, 0.09])
    with patch(
        "psnailder.fit.optimize.differential_evolution",
        return_value=OptimizeResult(x=np.array([0.2, 0.7]), fun=0.0, success=True, nfev=10, nit=2, message="Converged"),  # pyright: ignore[reportCallIssue]
    ) as optimizer:
        outcome = fitter.fit_spiral_with_background(
            grid,
            grid,
            grid,
            grid,
            num_components=2,
            winding=1,
            warm_start=guess,
            improve_background=False,
        )
    assert optimizer.call_args is not None
    np.testing.assert_array_equal(optimizer.call_args.kwargs["bounds"].lb, [0.1, 0.1])  # pyright: ignore[reportAny]
    np.testing.assert_array_equal(optimizer.call_args.kwargs["bounds"].ub, [0.9, 0.9])  # pyright: ignore[reportAny]
    np.testing.assert_array_equal(optimizer.call_args.kwargs["x0"], [0.2, 0.7])  # pyright: ignore[reportAny]
    assert isinstance(outcome, FitSuccess)
    np.testing.assert_array_equal(outcome.result.final_model.to_array(), guess)


def test_layout_preserves_component_order_and_fixed_values() -> None:
    layout = ParameterLayout.from_bounds([arm(Interval(0.1, 0.4)), arm(Interval(0.5, 0.9), np.pi)])
    assert layout.num_free == 2
    np.testing.assert_array_equal(layout.free_indices, [0, 6])
    full = layout.unpack(np.array([0.2, 0.7]))
    np.testing.assert_array_equal(full.reshape(2, 6), [[0.2, 0.05, 0.0, 0.0, 40.0, 0.09], [0.7, 0.05, 0.0, np.pi, 40.0, 0.09]])
    np.testing.assert_array_equal(layout.pack(full), [0.2, 0.7])
    full[:] = 0.0
    assert layout.unpack(np.array([0.2, 0.7]))[4] == 40.0


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("winding", [None, 1])
def test_all_fixed_skips_de(count: int, winding: Literal[-1, 1] | None) -> None:
    grid = np.ones((2, 2))
    fitter = PSpiralFitter(bounds=[arm(), arm(theta=np.pi)])
    with patch("psnailder.fit.optimize.differential_evolution") as de:
        outcome = fitter.fit_spiral_with_background(
            grid,
            grid,
            grid,
            grid,
            num_components=count,
            winding=winding,
            improve_background=False,
        )
    de.assert_not_called()
    assert isinstance(outcome, FitSuccess)
    assert outcome.result.final_model.parameters.shape == (count, 6)
    assert outcome.result.lnl == 0.0
    assert outcome.diagnostics.nfev == (2 if winding is None else 1)
    assert outcome.diagnostics.nit == 0
    assert outcome.diagnostics.success


def test_reduced_guess_and_expanded_result() -> None:
    fitter = PSpiralFitter(bounds=[arm(Interval(0.1, 0.4)), arm(Interval(0.5, 0.9), np.pi)])
    guess = np.array([0.2, 0.05, 0.0, 0.0, 40.0, 0.09, 0.7, 0.05, 0.0, np.pi, 40.0, 0.09])
    grid = np.ones((2, 2))
    with patch(
        "psnailder.fit.optimize.differential_evolution",
        return_value=OptimizeResult(x=np.array([0.2, 0.7]), fun=0.0, success=True, nfev=10, nit=2, message="Converged"),  # pyright: ignore[reportCallIssue]
    ) as de:
        outcome = fitter.fit_spiral_with_background(
            grid, grid, grid, grid, num_components=2, winding=1, warm_start=guess, improve_background=False
        )
    assert de.call_args is not None
    np.testing.assert_array_equal(de.call_args.kwargs["x0"], [0.2, 0.7])  # pyright: ignore[reportAny]
    np.testing.assert_array_equal(de.call_args.kwargs["bounds"].lb, [0.1, 0.5])  # pyright: ignore[reportAny]
    assert isinstance(outcome, FitSuccess)
    np.testing.assert_array_equal(outcome.result.final_model.to_array(), guess)


@pytest.mark.parametrize("count", [None, 2, 0])
def test_insufficient_bounds_fail_before_optimization(count: int | None) -> None:
    grid = np.ones((2, 2))
    with patch("psnailder.fit.optimize.differential_evolution") as de, pytest.raises(ValueError, match="bounds"):
        _ = PSpiralFitter(bounds=[arm()]).fit_spiral_with_background(grid, grid, grid, grid, num_components=count)
    de.assert_not_called()


def test_bic_counts_fixed_parameters_as_zero() -> None:
    # Use a shape difference: a constant amplitude difference disappears when
    # predictions are normalized to the observed total.
    fitter = PSpiralFitter(bounds=[arm(), arm(Fixed(0.5))])
    background = np.ones((2, 2))
    coordinates = np.array([[0.0, 0.1], [0.2, 0.3]])
    parameters = np.array([[0.0, 0.05, 0.0, 0.0, 40.0, 0.09], [0.5, 0.05, 0.0, 0.0, 40.0, 0.09]])
    data = PSpiralModel(parameters, coordinates, coordinates, background, winding=1).prediction()
    outcome = fitter.fit_spiral_with_background(data, background, coordinates, coordinates, winding=1, improve_background=False)
    assert isinstance(outcome, FitSuccess)
    assert outcome.result.final_model.num_components == 2
    mask = create_sigmoid_mask(1.0, 40.0)(coordinates, coordinates)
    assert outcome.result.lnl == pytest.approx(ln_likelihood(data, outcome.result.final_model.prediction(), mask))  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_bound_values_must_be_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _ = Fixed(value)
    with pytest.raises(ValueError, match="finite"):
        _ = Interval(0.0, value)


def test_reversed_interval_is_invalid() -> None:
    with pytest.raises(ValueError, match="Lower bound"):
        _ = Interval(1.0, 0.0)


def test_layout_rejects_duplicate_indices_and_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="unique"):
        _ = ParameterLayout(np.zeros(6), np.array([0, 0], dtype=np.intp), np.zeros(2), np.ones(2))
    layout = ParameterLayout.from_bounds([arm(Interval(0.0, 1.0))])
    with pytest.raises(ValueError, match="Expected the number of full parameters"):
        _ = layout.pack(np.zeros((6, 1)))  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="Expected the number of free parameters"):
        _ = layout.unpack(np.zeros((1, 1)))  # pyright: ignore[reportArgumentType]
