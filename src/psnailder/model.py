"""Module containing the phase spiral data model based off Alinder et. al. 2023 and Alinder et. al. 2024."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import special

from .component import PSpiralComponent
from .likelihood_utils import lrt_pvalue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optype import numpy as onp


@dataclass
class PSpiralModel:
    parameters: onp.Array2D[np.float64]
    z_mesh: onp.Array2D[np.float64]
    vz_mesh: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]
    winding: Literal[-1, 1] = 1
    flattening_strength: float = 0.1

    def prediction(self) -> onp.Array2D[np.float64]:
        assert self.z_mesh.ndim == 2
        assert self.z_mesh.shape == self.vz_mesh.shape
        assert self.z_mesh.shape == self.background.shape
        return self.background * self.signal()

    def signal(self) -> onp.Array2D[np.float64]:
        assert self.z_mesh.ndim == 2
        assert self.z_mesh.shape == self.vz_mesh.shape
        assert self.z_mesh.shape == self.background.shape

        if self.parameters.size == 0:
            return np.ones_like(self.z_mesh, dtype=np.float64)

        # parameters.shape == (n_components, 6)
        params = np.asarray(self.parameters, dtype=np.float64)
        assert params.ndim == 2 and params.shape[1] == 6
        alphas = params[:, 0][:, None, None]
        b = params[:, 1][:, None, None]
        c = params[:, 2][:, None, None]
        theta0 = params[:, 3][:, None, None]
        scale = params[:, 4][:, None, None]
        rho = params[:, 5][:, None, None]

        z = self.z_mesh[None, :, :]
        vz = self.vz_mesh[None, :, :]

        scaled_vz = vz / scale
        r = np.hypot(z, scaled_vz)
        theta = np.arctan2(vz, z * scale)

        # spiral phase: handle c != 0 and c == 0 (vectorised, avoid dividing by zero)
        phase = np.empty_like(r)
        c_mask = c[:, 0, 0] != 0.0

        # Compute for components where c != 0 using boolean indexing
        half = 0.5 * b[c_mask] / c[c_mask]
        phase[c_mask] = -half + np.sqrt(np.square(half) + r[c_mask] / c[c_mask])

        # For components where c == 0, use r / b
        phase[~c_mask] = r[~c_mask] / b[~c_mask]

        flattening = special.expit((r - rho) / self.flattening_strength)
        pert = 1.0 + alphas * flattening * np.cos(self.winding * theta - phase - theta0)

        # Combine components by taking the pixelwise maximum across components
        signal = np.max(pert, axis=0)
        signal[~np.isfinite(signal)] = 1.0
        return signal

    def pvalue(self, data: onp.Array2D[np.float64], mask: onp.Array2D[np.float64]) -> float:
        assert self.z_mesh.ndim == 2
        assert self.z_mesh.shape == self.vz_mesh.shape
        assert self.z_mesh.shape == self.background.shape
        assert self.z_mesh.shape == data.shape

        return lrt_pvalue(data, self.prediction(), self.background, mask, dof=6)

    @property
    def components(self) -> Sequence[PSpiralComponent]:
        """Compatibility: materialize components from parameters."""
        return tuple(
            PSpiralComponent.from_array(self.parameters[i], winding=self.winding) for i in range(self.parameters.shape[0])
        )

    def to_array(self) -> onp.Array1D[np.float64]:
        """Return flattened parameter vector (n_components * 6,)."""
        return np.asarray(self.parameters, dtype=np.float64).ravel()
