"""A module that implements the phase spiral fitting algorithm described in Alinder et4 al. 2023."""

from __future__ import annotations

from typing import TYPE_CHECKING

from psnailder import component, fit, model
from psnailder.uncertainty import BootstrapReplicate, BootstrapResult, BootstrapSamples, bootstrap_uncertainty

if TYPE_CHECKING:
    from typing import Final


__all__: Final[list[str]] = [
    "BootstrapReplicate",
    "BootstrapResult",
    "BootstrapSamples",
    "bootstrap_uncertainty",
    "component",
    "fit",
    "model",
]
