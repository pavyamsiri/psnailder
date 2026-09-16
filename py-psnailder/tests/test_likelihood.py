"""Invalid predictions must penalize the whole candidate."""

import numpy as np
import pytest

from psnailder._likelihood_utils import ln_likelihood


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_nonfinite_prediction_penalizes_candidate(invalid: float, index: int, weight: float) -> None:
    data = np.full((2, 3), 2.0)
    prediction = np.ones_like(data)
    prediction.flat[index] = invalid
    mask = np.ones_like(data)
    mask.flat[index] = weight

    # Invalid arithmetic would raise if the guard failed to return early.
    with np.errstate(all="raise"):
        score = ln_likelihood(data, prediction, mask)
    assert score == -np.inf
    assert -score == np.inf


def test_finite_prediction_score_is_unchanged() -> None:
    data = np.array([[2.0, 3.0, 4.0]])
    prediction = np.array([[1.0, 0.0, 2.0]])
    mask = np.array([[1.0, 1.0, 0.5]])
    assert ln_likelihood(data, prediction, mask) == pytest.approx(-0.75)
