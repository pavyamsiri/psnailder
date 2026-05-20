"""Module containing functions to compute the likelihoods used."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optype import numpy as onp


def ln_likelihood(data: onp.Array2D[np.float64], prediction: onp.Array2D[np.float64], mask: onp.Array2D[np.float64]) -> float:
    """Compute the log-likelihood of the model prediction to the data.

    Parameters
    ----------
    data : Array2D[f64]
        The observations.
    prediction : Array2D[f64]
        The model prediction.
    mask : Array2D[f64]
        The mask to de-emphasise the less important parts of the phase plane.

    Returns
    -------
    float
        The log-likelihood.

    """
    assert data.ndim == 2
    assert data.shape == prediction.shape
    assert data.shape == mask.shape

    square_residuals = np.square(mask * (data - prediction))
    denom = prediction
    valid = denom > 0

    return -0.5 * np.sum(square_residuals[valid] / denom[valid])
