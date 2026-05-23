"""Benchmark various optimisation routines with respect to both speed and reliability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

import numpy as np
from phasmix.component import AlinderComponent, GaussianComponent
from phasmix.mock import MockModel
from scipy import optimize

from psnailder import fit, model
from psnailder._likelihood_utils import ln_likelihood
import rich

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from optype import numpy as onp


type Objective = Callable[[onp.Array1D[np.float64]], float]


@runtime_checkable
class Optimizer(Protocol):
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float]: ...

    def name(self) -> str: ...


class ScipyLBFGSBOpt(Optimizer):
    def __init__(self, objective: Objective) -> None:
        self.objective: Objective = objective

    @override
    def name(self) -> str:
        return "scipy lbfgs optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float]:
        bounds = optimize.Bounds(lb=lb, ub=ub)

        res = optimize.minimize(self.objective, guess, bounds=bounds, method="L-BFGS-B")
        est_params = np.array(res.x)
        return (est_params, res.fun)


def _create_objective(signal_comp: AlinderComponent) -> Objective:
    background_comp = GaussianComponent(x_scale=1, y_scale=40.0, amplitude=1, variance=0.25)

    mock_model = MockModel((signal_comp,), (background_comp,))

    dx: float = 0.05
    dy: float = 1
    x_bins = np.arange(-1.2, 1.2 + dx, dx)
    y_bins = np.arange(-60.0, 60.0 + dy, dy)

    x_centres = 0.5 * (x_bins[:-1] + x_bins[1:])
    y_centres = 0.5 * (y_bins[:-1] + y_bins[1:])

    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    num_particles: int = 500_000
    mock_data = mock_model.mock_grid(x_bins, y_bins)
    density = num_particles * mock_data.density
    background = num_particles * mock_data.background

    mask = fit.create_sigmoid_mask(1.0, 40.0)(x_mesh, y_mesh)

    def _objective(parameters: onp.Array1D[np.float64]) -> float:
        nonlocal x_mesh, y_mesh, background
        prediction = model.PSpiralModel(parameters.reshape((1, 6)), x_mesh, y_mesh, background, winding=1).prediction()

        val = -ln_likelihood(
            density,
            prediction,
            mask,
        )
        return val

    return _objective


def _check_accuracy(
    names: Sequence[str],
    truth: onp.Array1D[np.float64],
    optimizer: Optimizer,
    *,
    guess: onp.Array1D[np.float64],
    bad_guess: onp.Array1D[np.float64],
    lb: onp.Array1D[np.float64],
    ub: onp.Array1D[np.float64],
) -> None:
    good_estimated, good_ll = optimizer.minimize(guess, lb, ub)
    bad_estimated, bad_ll = optimizer.minimize(bad_guess, lb, ub)
    len_str = len(optimizer.name())
    footer = "-" * (len_str + 13)
    rich.print(f"-- [yellow]{optimizer.name()}[/yellow] ([green]good[/green]) --")
    _report_result(names, truth, good_estimated, good_ll)
    rich.print(f"-- [yellow]{optimizer.name()}[/yellow] ([red]bad[/red])  --")
    _report_result(names, truth, bad_estimated, bad_ll)
    print(footer)


def _report_result(names: Sequence[str], truth: onp.Array1D[np.float64], estimated: onp.Array1D[np.float64], ll: float) -> None:
    assert len(names) == len(truth)
    assert len(names) == len(estimated)

    is_good = True
    for name, gt, est in zip(names, truth, estimated, strict=True):
        is_close = np.isclose(gt, est, rtol=1e-3, atol=5e-4)
        equality = "~" if is_close else "!="
        is_good &= is_close
        rich.print(f"[cyan]{name}[/cyan]: [green]{gt:.5f}[/green] {equality} [magenta]{est:.5f}[/magenta]")
    rich.print(f"Minimized neg likelihood = {ll:.5f}")
    if is_good:
        rich.print("Solution is [green]good[/green]!")
    else:
        rich.print("Solution is [red]bad[/red]!")


def main() -> None:
    names: list[str] = ["alpha", "b", "c", "theta0", "S", "rho"]
    signal_comp = AlinderComponent(
        alpha=0.5,
        b=0.05,
        c=0.002,
        theta0=0.0,
        scale_factor=40.00,
        rho=0.09,
        winding=1,
    )
    true_params = np.array(
        [signal_comp.alpha, signal_comp.b, signal_comp.c, signal_comp.theta0, signal_comp.scale_factor, signal_comp.rho]
    )

    objective = _create_objective(signal_comp)

    good_guess = np.array([0.5, 0.05, 0.002, 0.0, 42.0, 0.09])
    bad_guess = np.array([1.0, 0.001, 0.03, np.pi / 2, 60.0, 0.00])
    lb: onp.Array1D[np.float64] = np.array([0.0, 0.005, 0.0, -np.pi, 30.0, 0.0])
    ub: onp.Array1D[np.float64] = np.array([1.0, 0.1, 0.004, +np.pi, 70.0, 0.18])

    scipy_lbfgsb = ScipyLBFGSBOpt(objective)

    _check_accuracy(names, true_params, scipy_lbfgsb, guess=good_guess, bad_guess=bad_guess, lb=lb, ub=ub)


if __name__ == "__main__":
    main()
