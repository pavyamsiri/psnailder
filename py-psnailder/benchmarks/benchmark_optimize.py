"""Benchmark various optimisation routines with respect to both speed and reliability."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

import numpy as np
import rich
from phasmix.component import AlinderComponent, GaussianComponent
from phasmix.mock import MockModel
from scipy import optimize

from psnailder import fit, model
from psnailder._likelihood_utils import ln_likelihood

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from optype import numpy as onp


type Objective = Callable[[onp.Array1D[np.float64]], float]


@runtime_checkable
class Optimizer(Protocol):
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]: ...

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
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        bounds = optimize.Bounds(lb=lb, ub=ub)

        res = optimize.minimize(self.objective, guess, bounds=bounds, method="L-BFGS-B")
        est_params = np.array(res.x)
        return (est_params, res.fun, res.nfev)


class ScipyLBFGSBToNMOpt(Optimizer):
    def __init__(self, objective: Objective) -> None:
        self.objective: Objective = objective

    @override
    def name(self) -> str:
        return "scipy lbfgs to nm (two-stage) optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        bounds = optimize.Bounds(lb=lb, ub=ub)

        nfev: int = 0
        res = optimize.minimize(self.objective, guess, bounds=bounds, method="L-BFGS-B")
        first_stage = np.array(res.x)
        nfev += res.nfev

        res = optimize.minimize(self.objective, first_stage, bounds=bounds, method="Nelder-Mead")
        est_params = np.array(res.x)
        nfev += res.nfev

        return (est_params, res.fun, nfev)


class ScipyNaiveMultistartOpt(Optimizer):
    def __init__(self, objective: Objective, seed: int) -> None:
        self.objective: Objective = objective
        self.seed: int = seed

    @override
    def name(self) -> str:
        return "scipy naive multi-start lbfgs optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        bounds = optimize.Bounds(lb=lb, ub=ub)

        rng = np.random.default_rng()
        best_res = None
        nfev: int = 0
        for i in range(20):
            if i == 0:
                x0 = guess
            else:
                x0 = rng.uniform(lb, ub)
            res = optimize.minimize(self.objective, x0, bounds=bounds, method="L-BFGS-B")
            if best_res is None or res.fun <= best_res.fun:
                best_res = res
            nfev += res.nfev
        assert best_res is not None
        est_params = np.array(best_res.x)
        return (est_params, best_res.fun, nfev)


class ScipyDEOpt(Optimizer):
    def __init__(self, objective: Objective) -> None:
        self.objective: Objective = objective

    @override
    def name(self) -> str:
        return "scipy differential evolution optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        bounds = optimize.Bounds(lb=lb, ub=ub)

        res = optimize.differential_evolution(self.objective, bounds, maxiter=200, popsize=10, tol=1e-4)
        est_params = np.array(res.x)
        return (est_params, res.fun, res.nfev)


class ScipyDualAnnealingOpt(Optimizer):
    def __init__(self, objective: Objective) -> None:
        self.objective: Objective = objective

    @override
    def name(self) -> str:
        return "scipy dual annealing optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        bounds = optimize.Bounds(lb=lb, ub=ub)
        res = optimize.dual_annealing(self.objective, bounds, x0=guess)
        return (np.array(res.x), res.fun, res.nfev)


class ScipySHGOOpt(Optimizer):
    def __init__(self, objective: Objective) -> None:
        self.objective: Objective = objective

    @override
    def name(self) -> str:
        return "scipy shgo optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        bounds = optimize.Bounds(lb=lb, ub=ub)
        res = optimize.shgo(self.objective, bounds)
        return (np.array(res.x), res.fun, res.nfev)


class ScipyDIRECTOpt(Optimizer):
    def __init__(self, objective: Objective) -> None:
        self.objective: Objective = objective

    @override
    def name(self) -> str:
        return "scipy direct optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        bounds = optimize.Bounds(lb=lb, ub=ub)
        res = optimize.direct(self.objective, bounds)
        return (np.array(res.x), res.fun, res.nfev)


class ScipyBasinHoppingOpt(Optimizer):
    def __init__(self, objective: Objective) -> None:
        self.objective: Objective = objective

    @override
    def name(self) -> str:
        return "scipy basin hopping optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        res = optimize.basinhopping(
            self.objective,
            guess,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": optimize.Bounds(lb=lb, ub=ub),
            },
        )
        return (np.array(res.x), res.fun, res.nfev)


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
    lb: onp.Array1D[np.float64],
    ub: onp.Array1D[np.float64],
    n_random: int = 10,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    rich.print(f"-- [yellow]{optimizer.name()}[/yellow] ([green]good guess[/green]) --")
    t0 = time.perf_counter()
    good_estimated, good_ll, good_nfev = optimizer.minimize(guess, lb, ub)
    elapsed = time.perf_counter() - t0
    _report_result(names, truth, good_estimated, good_ll)
    rich.print(f"Time: {elapsed:.3f} s")
    rich.print(f"# of FEs: {good_nfev}")

    successes = 0
    total_time = 0.0
    best_ll: float = np.inf
    total_nfev: int = 0
    for _ in range(n_random):
        bad_guess = rng.uniform(lb, ub)
        t0 = time.perf_counter()
        estimated, ll, nfev = optimizer.minimize(bad_guess, lb, ub)
        total_time += time.perf_counter() - t0
        successes += all(np.isclose(t, e, rtol=1e-3, atol=5e-4) for t, e in zip(truth, estimated))
        best_ll = min(best_ll, ll)
        total_nfev += nfev

    rating = "perfect" if successes == n_random else "imperfect"
    rating_color = "green" if successes == n_random else "red"

    rich.print(f"-- [yellow]{optimizer.name()}[/yellow] ([red]random guesses[/red]) --")
    rich.print(f"Success rate: {successes}/{n_random} [{rating_color}]({rating})[/{rating_color}]")
    rich.print(f"Total time: {total_time:.3f} s  Mean: {total_time / n_random:.3f} s  Best: {best_ll}")
    rich.print(f"Mean # of FEs: {total_nfev / n_random}")


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
    lb: onp.Array1D[np.float64] = np.array([0.0, 0.005, 0.0, -np.pi, 30.0, 0.0])
    ub: onp.Array1D[np.float64] = np.array([1.0, 0.1, 0.004, +np.pi, 70.0, 0.18])

    _check_accuracy(names, true_params, ScipyLBFGSBOpt(objective), guess=good_guess, lb=lb, ub=ub)
    _check_accuracy(names, true_params, ScipyLBFGSBToNMOpt(objective), guess=good_guess, lb=lb, ub=ub)
    _check_accuracy(names, true_params, ScipyDEOpt(objective), guess=good_guess, lb=lb, ub=ub)
    _check_accuracy(names, true_params, ScipyNaiveMultistartOpt(objective, seed=42), guess=good_guess, lb=lb, ub=ub)
    _check_accuracy(names, true_params, ScipyDualAnnealingOpt(objective), guess=good_guess, lb=lb, ub=ub)
    _check_accuracy(names, true_params, ScipyDIRECTOpt(objective), guess=good_guess, lb=lb, ub=ub)
    _check_accuracy(names, true_params, ScipySHGOOpt(objective), guess=good_guess, lb=lb, ub=ub)
    _check_accuracy(names, true_params, ScipyBasinHoppingOpt(objective), guess=good_guess, lb=lb, ub=ub)


if __name__ == "__main__":
    main()
