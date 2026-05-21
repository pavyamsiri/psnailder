"""A module that implements the phase spiral fitting algorithm described in Alinder et al. 2023."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import component, fit, model


if TYPE_CHECKING:
    from typing import Final

__all__: Final[list[str]] = ["component", "fit", "model"]

if __name__ == "__main__":
    pass
