"""A module that implements the phase spiral fitting algorithm described in Alinder et4 al. 2023."""

from __future__ import annotations

from typing import TYPE_CHECKING

from psnailder import component, fit, model

if TYPE_CHECKING:
    from typing import Final


__all__: Final[list[str]] = ["component", "fit", "model"]
