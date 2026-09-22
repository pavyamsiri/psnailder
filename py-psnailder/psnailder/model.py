"""Module containing the phase spiral data model based off Alinder et. al. 2023 and Alinder et. al. 2024."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import special

from ._likelihood_utils import lrt_pvalue
from .component import PSpiralComponent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optype import numpy as onp


@dataclass
class PSpiralModel:
    """A representation of the phase spiral's model.

    Attributes
    ----------
    parameters : Array2D[f64]
        The model's parameters shaped like (num_components, num_parameters = 6).
    z_mesh : Array2D[f64]
        The mesh of the z coordinates.
    vz_mesh : Array2D[f64]
        The mesh of the vz coordinates.
    background : Array2D[f64]
        The model's background.
    winding : -1 or 1
        The winding direction.
    flattening_strength : float
        The flattening strength shared among the components.

    """

    parameters: onp.Array2D[np.float64]
    z_mesh: onp.Array2D[np.float64]
    vz_mesh: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]
    winding: Literal[-1, 1] = 1
    flattening_strength: float = 0.1

    def prediction(self) -> onp.Array2D[np.float64]:
        """Return the model's prediction of the number count map: `background * signal`.

        Returns
        -------
        prediction : Array2D[f64]
            The model's prediction.

        """
        assert self.z_mesh.ndim == 2
        assert self.z_mesh.shape == self.vz_mesh.shape
        assert self.z_mesh.shape == self.background.shape
        return self.background * self.signal()

    def signal(self) -> onp.Array2D[np.float64]:
        """Return the model's fitted spiral signal.

        Returns
        -------
        signal : Array2D[f64]
            The model's fitted spiral signal.

        """
        assert self.z_mesh.ndim == 2
        assert self.z_mesh.shape == self.vz_mesh.shape
        assert self.z_mesh.shape == self.background.shape

        if self.parameters.size == 0:
            return np.ones_like(self.z_mesh, dtype=np.float64)

        # Asserting that parameters.shape == (n_components, 6)
        params = np.asarray(self.parameters, dtype=np.float64)
        assert params.ndim == 2, "`parameters` should be 2D."
        assert params.shape[1] == 6, "`parameters` second axis should have length 6 (the number of spiral parameters)."
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

        # Match Rust's magnitude convention and near-zero linear branch.
        phase = np.empty_like(r)
        abs_b = np.abs(b)
        abs_c = np.abs(c)
        c_mask = abs_c[:, 0, 0] > 1e-10

        # Quadratic branch for abs(c) > 1e-10.
        half = 0.5 * abs_b[c_mask] / abs_c[c_mask]
        phase[c_mask] = -half + np.sqrt(np.square(half) + r[c_mask] / abs_c[c_mask])

        # Linear branch for abs(c) <= 1e-10.
        phase[~c_mask] = r[~c_mask] / abs_b[~c_mask]

        flattening = special.expit((r - rho) / self.flattening_strength)
        pert = 1.0 + alphas * flattening * np.cos(self.winding * theta - phase - theta0)

        # Combine components by taking the pixelwise maximum across components
        return np.max(pert, axis=0)  # pyright: ignore[reportAny]

    def pvalue(self, data: onp.Array2D[np.float64], mask: onp.Array2D[np.float64]) -> float:
        """Calculate the pvalue of the model's prediction fit to the data over the background's fit.

        Parameters
        ----------
        data : Array2D[f64]
            The data to fit to.
        mask : Array2D[f64]
            The mask to use when evaluating likelihood.

        Returns
        -------
        pvalue : float
            The pvalue.

        """
        assert self.z_mesh.ndim == 2
        assert self.z_mesh.shape == self.vz_mesh.shape
        assert self.z_mesh.shape == self.background.shape
        assert self.z_mesh.shape == data.shape

        return lrt_pvalue(data, self.prediction(), self.background, mask, dof=6 * self.parameters.shape[0])

    @property
    def components(self) -> Sequence[PSpiralComponent]:
        """Compatibility: materialize components from parameters."""
        return tuple(
            PSpiralComponent.from_array(self.parameters[i], flattening_strength=self.flattening_strength, winding=self.winding)  # pyright: ignore[reportAny]
            for i in range(self.parameters.shape[0])
        )

    @property
    def num_components(self) -> int:
        """int: The number of components in the model."""
        return self.parameters.shape[0]

    def to_array(self) -> onp.Array1D[np.float64]:
        """Return flattened parameter vector (n_components * 6,)."""
        return np.asarray(self.parameters, dtype=np.float64).ravel()
