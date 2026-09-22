"""Utilities to consolidate type casting of shaped arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from optype import numpy as onp


def verify_array_shape[SCT: np.generic, ShapeT: tuple[Any, ...]](
    arr: onp.ArrayND[SCT, tuple[Any, ...]],  # pyright: ignore[reportExplicitAny]
    shape: ShapeT,
) -> onp.ArrayND[SCT, ShapeT]:
    """Verify that the given array has the expected shape, casting it into the proper shaped array type.

    Parameters
    ----------
    arr : ArrayND[SCT, Any]
        The unshaped array.
    shape : SCT
        The expected shape.

    Returns
    -------
    ArrayND[SCT, ShapeT]
        The shaped array.

    """
    if arr.shape != shape:
        msg = f"Shape mismatch: expected {shape}, got {arr.shape}"
        raise TypeError(msg)
    return cast("onp.ArrayND[SCT, ShapeT]", arr)
