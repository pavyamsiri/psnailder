"""Tests of the spiral fitting algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from psnailder import SpiralFitter, calculate_rmse


@pytest.mark.parametrize("seed", range(3))
def test_gaussian_fit_improvement(seed: int) -> None:
    """ "Test that Gaussian distributed vertical phase space distributions have low improvement.

    Parameters
    ----------
    seed : int
        The random seed.

    """
    rng = np.random.default_rng(seed)

    z = rng.normal(loc=0.0, scale=0.2, size=1000)
    vz = rng.normal(loc=0.0, scale=20.0, size=1000)

    fitter = SpiralFitter()
    dz: float = 0.025
    dvz: float = 2.0
    z_bins = np.arange(-1.2, 1.2 + dz, dz)
    vz_bins = np.arange(-60.0, 60.0 + dvz, dvz)
    res = fitter.fit_spiral(z, vz, z_bins, vz_bins)
    init_bg_rmse = calculate_rmse(res.data, res.initial_model.background, res.z_mesh, res.vz_mesh)
    final_bg_rmse = calculate_rmse(res.data, res.final_model.background, res.z_mesh, res.vz_mesh)
    final_fit_rmse = calculate_rmse(res.data, res.final_model.fit(res.z_mesh, res.vz_mesh), res.z_mesh, res.vz_mesh)

    init_bg_improvement = (init_bg_rmse - final_fit_rmse) / init_bg_rmse
    final_bg_improvement = (final_bg_rmse - final_fit_rmse) / final_bg_rmse

    assert init_bg_improvement < 0.05
    assert final_bg_improvement < 0.03
