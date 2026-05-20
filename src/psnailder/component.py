"""Module containing the data representation of a phase spiral arm component."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from scipy import special
import numpy as np

from ._shape_utils import verify_array_shape

if TYPE_CHECKING:
    from optype import numpy as onp


@dataclass
class PSpiralComponent:
    """A phase spiral component.

    Attributes
    ----------
    alpha : float
        The spiral amplitude.
    b : float
        The linear winding amplitude.
    c : float
        The quadratic winding amplitude.
    theta0 : float
        The angle offset in radians.
    scale_factor : float
        The scale factor.
    rho : float
        The flattening function distance in phase radius units.
    flattening_strength : float
        The strength of the flattening in phase radius units.
    winding : -1 or +1
        The winding direction i.e., the sign of phi_s(r).

    """

    alpha: float
    b: float
    c: float
    theta0: float
    scale_factor: float
    rho: float
    winding: Literal[-1, 1]
    flattening_strength: float = 0.1

    def perturbation[ShapeT: tuple[Any, ...]](
        self, z: onp.ArrayND[np.float64, ShapeT], vz: onp.ArrayND[np.float64, ShapeT]
    ) -> onp.ArrayND[np.float64, ShapeT]:
        """Calculate the contribution to the perturbation from this component.

        The form being f(r, theta) = 1 + alpha * flattening(r, rho) * cos(theta - phi_s(r) - theta0).

        Parameters
        ----------
        z_mesh : ArrayND[f64, S]
            The z coordinates.
        vz_mesh : ArrayND[f64, S]
            The vz coordinates.

        Returns
        -------
        perturbation : ArrayND[f64, S]
            The perturbation.

        """
        assert z.shape == vz.shape

        scaled_z = np.multiply(z, self.scale_factor)
        scaled_vz = vz * np.reciprocal(self.scale_factor)
        r_mesh = np.hypot(z, scaled_vz)
        theta_mesh = np.arctan2(vz, scaled_z)

        phase = self.spiral_phase(r_mesh)

        flattening = special.expit((r_mesh - self.rho) / self.flattening_strength)
        pert = 1.0 + self.alpha * flattening * np.cos(self.winding * theta_mesh - phase - self.theta0)
        return verify_array_shape(pert, z.shape)

    def spiral_phase[ShapeT: tuple[Any, ...]](self, r: onp.ArrayND[np.float64, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]:
        """Compute the phase angle of the spiral phi_s(r).

        Parameters
        ----------
        r : ArrayND[f64, S]
            The phase distance.

        Returns
        -------
        phase : ArrayND[f64, S]
            The spiral phase in radians.

        """
        b_val: np.float64 = np.float64(self.b)
        c_val: np.float64 = np.float64(self.c)
        # phi_s(r) = (+/-) (-b/2c + sqrt((b/2c)^2 + r/c))
        if c_val != 0.0:
            half_b_over_c = 0.5 * b_val / c_val
            phase = -half_b_over_c + np.sqrt(np.square(half_b_over_c) + r / c_val)
        # phi_s(r) = (+/-) r / b
        else:
            phase = r / b_val
        return verify_array_shape(phase, r.shape)

    def model_phase(self, r_test: float = 0.5) -> float:
        """Calculate the model phase angle.

        Parameters
        ----------
        r_test : float
            The reference phase distance to calculate the angle at.

        Returns
        -------
        model_phase : float
            The model phase angle in radians.

        """
        phase = float(self.spiral_phase(np.array(r_test))[0])
        return phase + self.theta0

    @staticmethod
    def from_array(parameters: onp.Array1D[np.float64], *, winding: Literal[1, -1]) -> PSpiralComponent:
        """Convert an array of parameters into a spiral component.

        Parameters
        ----------
        parameters : Array1D[f64]
            The parameters array in the form [alpha, b, c, theta0, scale_factor, rho].

        Returns
        -------
        SpiralComponent
            The spiral component.

        """
        assert parameters.ndim == 1
        assert len(parameters) == 6

        return PSpiralComponent(
            alpha=parameters[0],
            b=parameters[1],
            c=parameters[2],
            theta0=parameters[3],
            scale_factor=parameters[4],
            rho=parameters[5],
            winding=winding,
        )

    def to_array(self) -> onp.Array1D[np.float64]:
        """Return the parameters as an array.

        Returns
        -------
        parameters : Array1D[f64]
            The parameters array in the form [alpha, b, c, theta0, scale_factor, rho].

        """
        return np.array([self.alpha, self.b, self.c, self.theta0, self.scale_factor, self.rho], dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"alpha={self.alpha!r}, "
            f"b={self.b!r}, "
            f"c={self.c!r}, "
            f"theta0={self.theta0!r}, "
            f"scale_factor={self.scale_factor!r}, "
            f"rho={self.rho!r}, "
            f"winding={self.winding!r}, "
            f"flattening_strength={self.flattening_strength!r})"
        )

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"α={self.alpha:.4g}, "
            f"b={self.b:.4g}, "
            f"c={self.c:.4g}, "
            f"θ₀={self.theta0:.4g} rad, "
            f"scale={self.scale_factor:.4g}, "
            f"ρ={self.rho:.4g}, "
            f"winding={self.winding:+d})"
        )
