"""Module containing the phase spiral data model based off Alinder et. al. 2023 and Alinder et. al. 2024."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .component import PSpiralComponent
from .likelihood_utils import lrt_pvalue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optype import numpy as onp


@dataclass
class PSpiralModel:
    components: Sequence[PSpiralComponent]
    z_mesh: onp.Array2D[np.float64]
    vz_mesh: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]

    def prediction(self) -> onp.Array2D[np.float64]:
        assert self.z_mesh.ndim == 2
        assert self.z_mesh.shape == self.vz_mesh.shape
        assert self.z_mesh.shape == self.background.shape
        return self.background * self.signal()

    def signal(self) -> onp.Array2D[np.float64]:
        assert self.z_mesh.ndim == 2
        assert self.z_mesh.shape == self.vz_mesh.shape
        assert self.z_mesh.shape == self.background.shape

        signal: onp.Array2D[np.float64] = np.full_like(self.z_mesh, -np.inf, dtype=np.float64)

        for comp in self.components:
            signal = np.maximum(signal, comp.perturbation(self.z_mesh, self.vz_mesh))
        signal[~np.isfinite(signal)] = 1.0

        return signal

    def pvalue(self, data: onp.Array2D[np.float64], mask: onp.Array2D[np.float64]) -> float:
        assert self.z_mesh.ndim == 2
        assert self.z_mesh.shape == self.vz_mesh.shape
        assert self.z_mesh.shape == self.background.shape
        assert self.z_mesh.shape == data.shape

        return lrt_pvalue(data, self.prediction(), self.background, mask, dof=6)
