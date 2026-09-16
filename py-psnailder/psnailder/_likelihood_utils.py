"""Module containing functions to compute the likelihoods used."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

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


def lrt_pvalue(
    data: onp.Array2D[np.float64],
    alt: onp.Array2D[np.float64],
    null: onp.Array2D[np.float64],
    mask: onp.Array2D[np.float64],
    *,
    dof: int,
) -> float:
    """Calculate the likelihood ratio with respect to the null model and return its p-value.

    Parameters
    ----------
    data : Array2D[f64]
        The observed data.
    alt : Array2D[f64]
        The alternative model prediction.
    null : Array2D[f64]
        The null model prediction.
    mask : Array2D[f64]
        The mask to de-emphasise the less important parts of the phase plane.

    Returns
    -------
    pvalue : float
        The p-value representing the probability that the null model is likely over the alternative model.

    """
    ln_like_alt = ln_likelihood(
        data,
        alt,
        mask,
    )
    ln_like_null = ln_likelihood(
        data,
        null,
        mask,
    )
    lmbd = -2.0 * (ln_like_null - ln_like_alt)
    return stats.chi2.sf(lmbd, df=dof)
