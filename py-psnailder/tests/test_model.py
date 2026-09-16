"""Consistency of component and model evaluation."""

from typing import Literal

import numpy as np
import pytest

from psnailder._internal import PSpiralComponent as RustComponent
from psnailder.component import PSpiralComponent
from psnailder.model import PSpiralModel


@pytest.mark.parametrize("b", [-0.05, 0.05])
@pytest.mark.parametrize("c", [0.0, -1e-11, 1e-11, -1e-10, 1e-10, -2e-10, 2e-10, -0.002, 0.002])
@pytest.mark.parametrize("winding", [-1, 1])
def test_component_model_and_rust_agree(b: float, c: float, winding: Literal[-1, 1]) -> None:
    parameters = np.array([0.5, b, c, 0.3, 40.0, 0.09])
    component = PSpiralComponent.from_array(parameters, flattening_strength=0.1, winding=winding)
    rust_component = RustComponent(alpha=0.5, b=b, c=c, theta0=0.3, scale_factor=40.0, rho=0.09, winding=winding)
    z, vz = np.meshgrid(np.array([-0.3, 0.0, 0.2]), np.array([-12.0, 0.0, 8.0]))
    model = PSpiralModel(parameters[None, :], z, vz, np.ones_like(z), winding=winding)

    signal = component.perturbation(z, vz)
    np.testing.assert_allclose(model.signal(), signal, rtol=1e-12, atol=1e-12)
    # Near the cutoff the quadratic formula subtracts large, nearly equal
    # values; Rust's fused arithmetic can round differently from NumPy.
    np.testing.assert_allclose(
        rust_component.perturbation(z.ravel(), vz.ravel()).reshape(z.shape),
        signal,
        rtol=1e-7,
        atol=1e-7,
    )


@pytest.mark.parametrize("c", [0.0, -1e-10, 1e-10])
def test_model_phase_accepts_scalar_radius(c: float) -> None:
    component = PSpiralComponent(0.5, -0.05, c, 0.3, 40.0, 0.09, winding=1)
    assert component.model_phase() == pytest.approx(0.5 / 0.05 + 0.3)


def test_extracted_components_preserve_flattening_strength() -> None:
    parameters = np.array([[0.5, 0.05, 0.002, 0.3, 40.0, 0.09], [0.4, 0.04, 0.001, -0.2, 45.0, 0.1]])
    z, vz = np.meshgrid(np.array([-0.1, 0.0, 0.1]), np.array([-4.0, 0.0, 4.0]))
    model = PSpiralModel(parameters, z, vz, np.ones_like(z), winding=-1, flattening_strength=0.2)
    components = model.components

    for component, expected in zip(components, parameters, strict=True):
        assert component.flattening_strength == 0.2
        assert component.winding == -1
        np.testing.assert_array_equal(component.to_array(), expected)
    combined = np.maximum.reduce([component.perturbation(z, vz) for component in components])
    np.testing.assert_allclose(model.signal(), combined)


def test_invalid_phase_is_not_replaced_with_unit_signal() -> None:
    # Both winding coefficients zero makes the phase undefined.
    parameters = np.array([[0.5, 0.0, 0.0, 0.0, 40.0, 0.09]])
    z = np.array([[0.1, 0.2]])
    model = PSpiralModel(parameters, z, np.zeros_like(z), np.ones_like(z))

    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isnan(model.signal()).all()
        assert np.isnan(model.prediction()).all()
