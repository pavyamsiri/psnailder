"""Behavioral tests for the optional Python local-refit bootstrap."""

# ruff: noqa: D103

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from scipy import optimize

from psnailder import BootstrapSamples, bootstrap_uncertainty
from psnailder.bounds import ParameterBounds
from psnailder.fit import FitSuccess, PSpiralFitter
from psnailder.model import PSpiralModel
from psnailder.uncertainty import _aligned  # pyright: ignore[reportPrivateUsage] -- test periodic component alignment.

if TYPE_CHECKING:
    from psnailder.fit import PSpiralFitResult


@pytest.fixture
def fitted() -> tuple[PSpiralFitter, PSpiralFitResult]:
    parameters = np.array([[0.4, 0.05, 0.002, 0.3, 40.0, 0.09]])
    z, vz = np.meshgrid(np.linspace(-0.8, 0.8, 14), np.linspace(-35.0, 35.0, 15))
    background = 100 * np.exp(-np.square(z) - np.square(vz / 40))
    model = PSpiralModel(parameters, z, vz, background, winding=1)
    counts = np.random.default_rng(8).poisson(model.prediction()).astype(np.float64)
    bounds = ParameterBounds(alpha=(0.0, 0.9), b=0.05, c=0.002, theta0=0.3, scale_factor=40.0, rho=0.09)
    fitter = PSpiralFitter(bounds=bounds)
    outcome = fitter.fit_spiral_with_background(
        counts, background, z, vz, num_components=1, winding=1, improve_background=False, rng=np.random.default_rng(9)
    )
    assert isinstance(outcome, FitSuccess)
    return fitter, outcome.result


def test_count_bootstrap_is_local_reproducible_and_nonmutating(fitted: tuple[PSpiralFitter, PSpiralFitResult]) -> None:
    fitter, result = fitted
    original = result.final_model.parameters.copy()
    background = result.final_model.background.copy()
    with patch("psnailder._python_backend.optimize.differential_evolution", side_effect=AssertionError("global search")):
        serial = bootstrap_uncertainty(fitter, result, n_resamples=8, seed=12)
        parallel = bootstrap_uncertainty(fitter, result, n_resamples=10, seed=12, workers=2)
    assert serial.n_successful == 8
    np.testing.assert_array_equal(serial.parameters, parallel.parameters[:8])
    np.testing.assert_array_equal(result.final_model.parameters, original)
    np.testing.assert_array_equal(result.final_model.background, background)
    assert serial.standard_errors[0] > 0
    np.testing.assert_allclose(serial.standard_errors[1:], 0.0, atol=1e-14)
    assert all(item.count_total == int(result.data.sum()) for item in serial.replicates)
    assert np.all(serial.parameters[:, 0] >= 0)
    assert np.all(serial.parameters[:, 0] <= 0.9)
    assert serial.intervals.shape == (6, 2)
    assert serial.covariance.shape == (6, 6)
    assert serial.method == "parametric_counts"


def test_failed_local_refits_are_retained(fitted: tuple[PSpiralFitter, PSpiralFitResult]) -> None:
    fitter, result = fitted
    uncertainty = bootstrap_uncertainty(fitter, result, n_resamples=3, maxiter=1)
    assert uncertainty.n_successful == 0
    assert len(uncertainty.replicates) == 3
    assert np.all(np.isnan(uncertainty.parameters))
    assert np.all(np.isnan(uncertainty.standard_errors))
    assert any("failed" in warning for warning in uncertainty.warnings)


def test_sample_bootstrap_rebuilds_background_and_replays_refinement() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(0, 0.4, 120)
    vz = rng.normal(0, 18, 120)
    z_bins, vz_bins = np.linspace(-1, 1, 8), np.linspace(-50, 50, 9)
    bounds = ParameterBounds(alpha=0.4, b=0.05, c=0.002, theta0=0.0, scale_factor=40.0, rho=0.09)
    fitter = PSpiralFitter(bounds=bounds, max_iterations=1)
    outcome = fitter.fit_spiral(z, vz, z_bins, vz_bins, num_components=1, winding=1)
    assert isinstance(outcome, FitSuccess)
    from psnailder import fit  # noqa: PLC0415 -- wrap the real preprocessing implementation.

    with (
        patch("psnailder.fit.generate_initial_background", wraps=fit.generate_initial_background) as kde,
        patch("psnailder._python_backend.optimize.minimize", side_effect=AssertionError("all parameters fixed")),
        patch("psnailder._python_backend.optimize.differential_evolution", side_effect=AssertionError("global search")),
    ):
        summary = bootstrap_uncertainty(
            fitter,
            outcome.result,
            samples=BootstrapSamples(z=z, vz=vz, z_bins=z_bins, vz_bins=vz_bins),
            n_resamples=3,
            seed=7,
        )
    assert kde.call_count == 3
    assert summary.n_successful == 3
    assert summary.method == "samples"
    assert all(item.reason != "fixed_background" for item in summary.replicates)
    np.testing.assert_allclose(summary.standard_errors, 0.0, atol=1e-14)
    for call in kde.call_args_list:
        sampled_z, sampled_vz = call.args[:2]
        # Each sampled position retains its original paired velocity.
        for zi, vi in zip(sampled_z, sampled_vz, strict=True):
            assert np.any((z == zi) & (vz == vi))


def test_alignment_handles_exchangeable_components_and_phase_wraps() -> None:
    reference = np.array([0.2, 0.04, 0.001, 3.1, 40, 0.08, 0.7, 0.08, 0.003, -1, 50, 0.1])
    candidate = reference.copy()
    candidate[3] -= 2 * np.pi
    candidate = candidate.reshape(2, 6)[::-1].flatten()
    aligned = _aligned(candidate, reference, np.ones(12), exchangeable=True)
    np.testing.assert_allclose(aligned, reference)


def test_one_draw_has_no_covariance(fitted: tuple[PSpiralFitter, PSpiralFitResult]) -> None:
    fitter, result = fitted
    summary = bootstrap_uncertainty(fitter, result, n_resamples=1)
    assert summary.n_successful == 1
    assert np.all(np.isnan(summary.covariance))


def test_rust_backend_is_rejected(fitted: tuple[PSpiralFitter, PSpiralFitResult]) -> None:
    _, result = fitted
    with pytest.raises(NotImplementedError, match="Python backend"):
        bootstrap_uncertainty(PSpiralFitter(backend="rust"), result, n_resamples=2)


@pytest.mark.parametrize("n_resamples", [0, -1])
def test_invalid_resample_count(fitted: tuple[PSpiralFitter, PSpiralFitResult], n_resamples: int) -> None:
    fitter, result = fitted
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_uncertainty(fitter, result, n_resamples=n_resamples)


def test_weighted_counts_are_rejected(fitted: tuple[PSpiralFitter, PSpiralFitResult]) -> None:
    fitter, result = fitted
    result.data = result.data + 0.5
    with pytest.raises(ValueError, match="integer counts"):
        bootstrap_uncertainty(fitter, result, n_resamples=2)


def test_full_period_bounds_are_centered_without_changing_restricted_bounds() -> None:
    from psnailder.uncertainty import _local_bounds  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]

    centered = _local_bounds(ParameterBounds(), 3.1)
    assert centered.theta0 == ParameterBounds(theta0=(3.1 - np.pi, 3.1 + np.pi)).theta0
    restricted = ParameterBounds(theta0=(-0.2, 0.2))
    assert _local_bounds(restricted, 0.1).theta0 == restricted.theta0


def test_singular_sample_draws_remain_failed_replicates() -> None:
    rng = np.random.default_rng(17)
    z, vz = rng.normal(size=30), rng.normal(size=30)
    edges = np.linspace(-3, 3, 7)
    fitter = PSpiralFitter(bounds=ParameterBounds(alpha=0.2, b=0.05, c=0.002, theta0=0, scale_factor=40, rho=0.09))
    outcome = fitter.fit_spiral(z, vz, edges, edges, num_components=1, winding=1, improve_background=False)
    assert isinstance(outcome, FitSuccess)
    with patch("psnailder.fit.generate_initial_background", side_effect=np.linalg.LinAlgError("singular draw")):
        summary = bootstrap_uncertainty(
            fitter,
            outcome.result,
            samples=BootstrapSamples(z=z, vz=vz, z_bins=edges, vz_bins=edges, improve_background=False),
            n_resamples=3,
        )
    assert summary.n_successful == 0
    assert all("KDE" in item.reason for item in summary.replicates)


def test_sample_refinement_uses_local_optimizer_for_every_update() -> None:
    rng = np.random.default_rng(23)
    z, vz = rng.normal(0, 0.4, 200), rng.normal(0, 18, 200)
    z_bins, vz_bins = np.linspace(-1, 1, 9), np.linspace(-50, 50, 10)
    bounds = ParameterBounds(alpha=(0, 0.8), b=0.05, c=0.002, theta0=0.1, scale_factor=40, rho=0.09)
    fitter = PSpiralFitter(bounds=bounds, max_iterations=1)
    outcome = fitter.fit_spiral(z, vz, z_bins, vz_bins, num_components=1, winding=1, rng=rng)
    assert isinstance(outcome, FitSuccess)
    with (
        patch("psnailder._python_backend.optimize.minimize", wraps=optimize.minimize) as local,
        patch("psnailder._python_backend.optimize.differential_evolution", side_effect=AssertionError("global search")),
    ):
        summary = bootstrap_uncertainty(
            fitter,
            outcome.result,
            samples=BootstrapSamples(z=z, vz=vz, z_bins=z_bins, vz_bins=vz_bins),
            n_resamples=3,
            seed=13,
        )
    assert summary.n_successful == 3
    assert local.call_count == 6  # One initial fit and one refinement per draw.
