"""Tests of the spiral fitting algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from psnailder import SpiralFitterMinimizer


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

    fitter = SpiralFitterMinimizer(objective="prob")
    dz: float = 0.025
    dvz: float = 2.0
    z_bins = np.arange(-1.2, 1.2 + dz, dz)
    vz_bins = np.arange(-60.0, 60.0 + dvz, dvz)
    res = fitter.fit_spiral(z, vz, z_bins, vz_bins)

    assert res.final_model.pvalue(res.data, res.z_mesh, res.vz_mesh) >= 0.05


@pytest.mark.parametrize("seed", range(3))
def test_gaussian_fit_improvement_opt_rmse(seed: int) -> None:
    """ "Test that Gaussian distributed vertical phase space distributions have low improvement.

    Parameters
    ----------
    seed : int
        The random seed.

    """
    rng = np.random.default_rng(seed)

    z = rng.normal(loc=0.0, scale=0.2, size=100_000)
    vz = rng.normal(loc=0.0, scale=20.0, size=100_000)

    fitter = SpiralFitterMinimizer(objective="error")
    dz: float = 0.025
    dvz: float = 2.0
    z_bins = np.arange(-1.2, 1.2 + dz, dz)
    vz_bins = np.arange(-60.0, 60.0 + dvz, dvz)
    res = fitter.fit_spiral(z, vz, z_bins, vz_bins)

    assert res.final_model.pvalue(res.data, res.z_mesh, res.vz_mesh) >= 0.05
