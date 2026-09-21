"""Helper to convert parameter arrays from the full set to a free parameter only set and vice-versa."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .bounds import Fixed, Interval, ParameterBounds

if TYPE_CHECKING:
    from optype import numpy as onp


@dataclass(frozen=True)
class ParameterLayout:
    """The parameter layout.

    Mostly a helper to convert between arrays of free parameters and the full parameter array.

    Attributes
    ----------
    template : Array1D[f64]
        An array of the full parameter set.
        Fixed values are set in their respective slot while non-fixed values will be overwritten.
    free_indices : Array1D[intp]
        The indices of the free parameters.
    lower : Array1D[f64]
        The lower bounds of the free parameters.
    upper  : Array1D[f64]
        The upper bounds of the free parameters.

    """

    template: onp.Array1D[np.float64]
    free_indices: onp.Array1D[np.intp]
    lower: onp.Array1D[np.float64]
    upper: onp.Array1D[np.float64]

    def __post_init__(self) -> None:
        """Validate the bounds and free indices."""

        if any(array.ndim != 1 for array in (self.template, self.free_indices, self.lower, self.upper)):
            raise ValueError("Layout arrays must be one-dimensional.")
        if not np.issubdtype(self.free_indices.dtype, np.integer):
            raise ValueError("Free parameter indices must be integers.")
        if np.unique(self.free_indices).size != self.free_indices.size:
            raise ValueError("Free parameter indices must be unique.")
        if not all(np.all(np.isfinite(array)) for array in (self.template, self.lower, self.upper)):
            raise ValueError("Layout values must be finite.")

        num_total_parameters = len(self.template)
        num_free_parameters = len(self.free_indices)
        if num_free_parameters > num_total_parameters:
            msg = "The number of free parameters must not exceed the number of total parameters."
            raise ValueError(msg)

        if np.any(self.free_indices >= len(self.template)):
            msg = "The free parameter indices must not point past the number of total parameters."
            raise ValueError(msg)

        if np.any(self.free_indices < 0):
            msg = "The free parameter indices must not be negative."
            raise ValueError(msg)

        lower, upper = self.lower, self.upper

        num_lower_bounds = len(lower)
        num_upper_bounds = len(upper)
        if num_lower_bounds != num_upper_bounds or num_lower_bounds != num_free_parameters:
            msg = "The number of free parameters must be equal to the number of lower and upper bounds."
            raise ValueError(msg)

        has_invalid_bounds = np.any(lower > upper)
        if has_invalid_bounds:
            msg = "Lower bounds must not exceed upper bounds."
            raise ValueError(msg)

    @property
    def num_free(self) -> int:
        """int: The number of free parameters."""
        return len(self.free_indices)

    @staticmethod
    def from_bounds(bounds_list: Sequence[ParameterBounds]) -> ParameterLayout:
        """Create layout from bounds object.

        Parameters
        ----------
        bounds_list : Sequence[ParameterBounds]
            Bounds for each component, in model order.

        Returns
        -------
        layout : ParameterLayout
            The layout.

        """

        parameter_bounds_list: list[Interval | Fixed] = []
        for bounds in bounds_list:
            parameter_bounds_list.append(bounds.alpha)
            parameter_bounds_list.append(bounds.b)
            parameter_bounds_list.append(bounds.c)
            parameter_bounds_list.append(bounds.theta0)
            parameter_bounds_list.append(bounds.scale_factor)
            parameter_bounds_list.append(bounds.rho)

        template = np.zeros(len(parameter_bounds_list), dtype=np.float64)

        free_indices_list: list[int] = []
        lower_list: list[float] = []
        upper_list: list[float] = []

        for idx, current_bounds in enumerate(parameter_bounds_list):
            # Free
            if isinstance(current_bounds, Interval):
                if current_bounds.lower != current_bounds.upper:
                    free_indices_list.append(idx)
                    lower_list.append(current_bounds.lower)
                    upper_list.append(current_bounds.upper)
                else:
                    template[idx] = current_bounds.lower
            # Fixed
            else:
                assert isinstance(current_bounds, Fixed), f"Should be `Fixed` by type annotations: {current_bounds}"
                template[idx] = current_bounds.value

        free_indices = np.array(free_indices_list, dtype=np.intp)
        lower = np.array(lower_list, dtype=np.float64)
        upper = np.array(upper_list, dtype=np.float64)

        return ParameterLayout(
            template=template,
            free_indices=free_indices,
            lower=lower,
            upper=upper,
        )

    def pack(self, full_parameters: onp.Array1D[np.float64], *, eps: float = 1e-12) -> onp.Array1D[np.float64]:
        """Pack an array of the full parameter set into an array of just the free parameters.

        The values are also clamped to be within the bounds +- a small epsilon.

        Parameters
        ----------
        full_parameters : Array1D[f64]
            The full parameter array.
        eps : float
            The ratio of the parameter bounds' width to be used to calculate the small epsilon.

        Returns
        -------
        free_parameters : Array1D[f64]
            The full parameter array stripped of any fixed values.

        Notes
        -----
        The values at indices where the parameter is fixed are not validated in anyway and are
        simply just discarded.

        """

        if full_parameters.shape != self.template.shape:
            msg = f"Expected the number of full parameters to be {len(self.template)}."
            raise ValueError(msg)

        values: onp.Array1D[np.float64] = full_parameters[self.free_indices]
        if not np.all(np.isfinite(values)):
            raise ValueError("Free initial guesses must be finite.")
        width: onp.Array1D[np.float64] = self.upper - self.lower
        clamped = np.clip(values, self.lower + eps * width, self.upper - eps * width)
        return clamped

    def unpack(self, free_parameters: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]:
        """Unpack an array of the free parameter set into an array of the full parameter set.

        Parameters
        ----------
        free_parameters : Array1D[f64]
            The free parameter array.

        Returns
        -------
        full_parameters : Array1D[f64]
            An expanded array of the full parameters.

        """

        num_free_parameters = self.num_free
        if free_parameters.shape != (num_free_parameters,):
            msg = f"Expected the number of free parameters to be {num_free_parameters}"
            raise ValueError(msg)

        full_parameters = np.copy(self.template)
        full_parameters[self.free_indices] = free_parameters
        return full_parameters
