"""Benchmark various optimisation routines with respect to both speed and reliability."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

import numpy as np
import rich
from phasmix.component import AlinderComponent, GaussianComponent
from phasmix.mock import MockModel
from rich.table import Table
from scipy import optimize, stats

from psnailder import fit, model, _internal
from psnailder._likelihood_utils import ln_likelihood

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from optype import numpy as onp


type Objective = Callable[[onp.Array1D[np.float64]], float]


@dataclass
class BenchmarkReport:
    optimizer_name: str
    good_guess_time: float
    good_guess_nfev: int
    good_guess_success: bool
    success_rate: float  # fraction of random restarts that succeeded
    mean_time: float
    mean_nfev: float
    best_ll: float
    n_random: int

    @property
    def succeeded(self) -> bool:
        """Optimizer is considered viable if it solves the problem reliably."""
        return self.success_rate > 0.0

    @property
    def perfect(self) -> bool:
        return self.success_rate == 1.0


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


class TikTakOpt(Optimizer):
    def __init__(self, objective: Objective, num_sobol: int = 2**12, n_star: int | None = None, seed: int | None = None) -> None:
        self.objective: Objective = objective
        self.n_sobol: int = num_sobol
        self.n_star: int | None = n_star
        self.seed: int | None = seed

    @override
    def name(self) -> str:
        return "tiktak optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        return TikTakOpt.tiktak(self.objective, lb, ub, n_sobol=self.n_sobol, n_star=self.n_star, seed=self.seed)

    @staticmethod
    def _tiktak_weight(iteration: int, n_sobol: int, w_min: float = 0.1, w_max: float = 0.995) -> float:
        """Weight that increases from w_min to w_max as iterations progress."""
        return max(w_min, min(w_max, (iteration / n_sobol) ** 0.5))

    @staticmethod
    def tiktak(
        objective: Objective,
        lb: onp.Array1D[np.float64],
        ub: onp.Array1D[np.float64],
        n_sobol: int = 2**12,
        n_star: int | None = None,
        seed: int | None = None,
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        if n_star is None:
            n_star = n_sobol // 2**7

        # --- Phase 1: Sobol evaluation ---
        rng = np.random.default_rng(seed)
        sobol = stats.qmc.Sobol(d=len(lb), scramble=True, rng=rng)
        unit_points = sobol.random_base2(int(np.log2(n_sobol)))
        points = stats.qmc.scale(unit_points, lb, ub)
        nfev: int = n_sobol

        values = np.array([objective(p) for p in points])

        # seed the local phase with the best n_star points
        top_indices = np.argsort(values)[:n_star]
        top_points = points[top_indices]
        top_values = values[top_indices]

        # --- Phase 2: iterated local search ---
        bounds = optimize.Bounds(lb=lb, ub=ub)

        best_point = top_points[0]
        best_value = top_values[0]
        for i, (candidate, value) in enumerate(zip(top_points, top_values)):
            best_point = candidate
            best_value = value
            w = TikTakOpt._tiktak_weight(i + 1, n_star)
            start = (1 - w) * candidate + w * best_point
            res = optimize.minimize(objective, start, method="Nelder-Mead", bounds=bounds)
            nfev += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_point = np.array(res.x)
        return (
            best_point,
            best_value,
            nfev,
        )


class RustTikTakOpt(Optimizer):
    def __init__(self, data_ctx: dict[str, onp.ArrayND[np.float64, Any]]) -> None:
        self.data_ctx = data_ctx

    @override
    def name(self) -> str:
        return "rust tiktak optimizer"

    @override
    def minimize(
        self, guess: onp.Array1D[np.float64], lb: onp.Array1D[np.float64], ub: onp.Array1D[np.float64]
    ) -> tuple[onp.Array1D[np.float64], float, int]:
        bounds = list(zip(lb.tolist(), ub.tolist(), strict=True))
        params, cost, nfev = _internal.fit_spiral_rust(
            self.data_ctx["density"].ravel(),
            self.data_ctx["background"].ravel(),
            self.data_ctx["mask"].ravel(),
            self.data_ctx["z"].ravel(),
            self.data_ctx["vz"].ravel(),
            bounds,
        )
        params = np.array(params)
        params[3] %= np.pi
        return (params, cost, int(nfev))


OPTIMIZER_REGISTRY: dict[str, Callable[[Objective, dict[str, Any]], Optimizer]] = {
    "lbfgsb": lambda obj, ctx: ScipyLBFGSBOpt(obj),
    "lbfgsb-nm": lambda obj, ctx: ScipyLBFGSBToNMOpt(obj),
    "de": lambda obj, ctx: ScipyDEOpt(obj),
    "multistart": lambda obj, ctx: ScipyNaiveMultistartOpt(obj, seed=42),
    "dual-annealing": lambda obj, ctx: ScipyDualAnnealingOpt(obj),
    "direct": lambda obj, ctx: ScipyDIRECTOpt(obj),
    "shgo": lambda obj, ctx: ScipySHGOOpt(obj),
    "basinhopping": lambda obj, ctx: ScipyBasinHoppingOpt(obj),
    "tiktak": lambda obj, ctx: TikTakOpt(obj, num_sobol=2**10),
    "rust-tiktak": lambda obj, ctx: RustTikTakOpt(ctx),
}


def _create_objective(signal_comp: AlinderComponent, *, use_rust: bool = False) -> tuple[Objective, dict[str, Any]]:
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

    data_ctx = {
        "density": density,
        "background": background,
        "mask": mask,
        "z": x_mesh,
        "vz": y_mesh,
    }

    if use_rust:
        z_flat = x_mesh.ravel()
        vz_flat = y_mesh.ravel()
        density_flat = density.ravel()
        background_flat = background.ravel()
        mask_flat = mask.ravel()

        def _objective_rust(parameters: onp.Array1D[np.float64]) -> float:
            # Convert Python parameters to Rust format
            # Python: [alpha, b, c, theta0, scale_factor, rho]
            # Rust:   [alpha, lnb, c, theta0, scale_factor (log), rho, winding, flattening_strength]
            alpha, b, c, theta0, scale_factor, rho = parameters
            rust_comp = _internal.PSpiralComponent(
                alpha=alpha,
                lnb=np.log(b),
                c=c,
                theta0=theta0,
                scale_factor=np.log(scale_factor),
                rho=rho,
                winding=1,
                flattening_strength=0.1,
            )
            rust_model = _internal.PSpiralModel([rust_comp])
            return -rust_model.evaluate_likelihood(density_flat, background_flat, mask_flat, z_flat, vz_flat)

        return _objective_rust, data_ctx
    else:

        def _objective(parameters: onp.Array1D[np.float64]) -> float:
            nonlocal x_mesh, y_mesh, background
            prediction = model.PSpiralModel(parameters.reshape((1, 6)), x_mesh, y_mesh, background, winding=1).prediction()

            val = -ln_likelihood(
                density,
                prediction,
                mask,
            )
            return val

        return _objective, data_ctx


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
        raise ValueError(f"{name}")
        is_close = np.isclose(gt, est, rtol=1e-3, atol=5e-4)
        equality = "~" if is_close else "!="
        is_good &= is_close
        rich.print(f"[cyan]{name}[/cyan]: [green]{gt:.5f}[/green] {equality} [magenta]{est:.5f}[/magenta]")
    rich.print(f"Minimized neg likelihood = {ll:.5f}")
    if is_good:
        rich.print("Solution is [green]good[/green]!")
    else:
        rich.print("Solution is [red]bad[/red]!")


def _check_accuracy(
    names: Sequence[str],
    truth: onp.Array1D[np.float64],
    optimizer: Optimizer,
    *,
    guess: onp.Array1D[np.float64],
    lb: onp.Array1D[np.float64],
    ub: onp.Array1D[np.float64],
    num_random: int = 10,
    seed: int = 42,
) -> BenchmarkReport:
    rng = np.random.default_rng(seed)

    rich.print(f"-- [yellow]{optimizer.name()}[/yellow] ([green]good guess[/green]) --")
    t0 = time.perf_counter()
    good_estimated, good_ll, good_nfev = optimizer.minimize(guess, lb, ub)
    good_time = time.perf_counter() - t0
    good_success = _is_success(truth, good_estimated)
    _report_result(names, truth, good_estimated, good_ll)
    rich.print(f"Time: {good_time:.3f} s  # of FEs: {good_nfev}")

    successes = 0
    total_time = 0.0
    total_nfev = 0
    best_ll: float = np.inf

    for _ in range(num_random):
        bad_guess = rng.uniform(lb, ub)
        t0 = time.perf_counter()
        estimated, ll, nfev = optimizer.minimize(bad_guess, lb, ub)
        total_time += time.perf_counter() - t0
        successes += _is_success(truth, estimated)
        best_ll = min(best_ll, ll)
        total_nfev += nfev

    success_rate = successes / num_random
    rating_color = "green" if success_rate == 1.0 else ("yellow" if success_rate > 0.0 else "red")
    rich.print(f"-- [yellow]{optimizer.name()}[/yellow] ([red]random guesses[/red]) --")
    rich.print(f"Success rate: {successes}/{num_random} [{rating_color}]({success_rate:.0%})[/{rating_color}]")
    rich.print(f"Total time: {total_time:.3f} s  Mean: {total_time / num_random:.3f} s  Best LL: {best_ll:.5f}")
    rich.print(f"Mean # of FEs: {total_nfev / num_random:.1f}")

    return BenchmarkReport(
        optimizer_name=optimizer.name(),
        good_guess_time=good_time,
        good_guess_nfev=good_nfev,
        good_guess_success=good_success,
        success_rate=success_rate,
        mean_time=total_time / num_random,
        mean_nfev=total_nfev / num_random,
        best_ll=best_ll,
        n_random=num_random,
    )


def _is_success(truth: onp.Array1D[np.float64], estimated: onp.Array1D[np.float64]) -> bool:
    return all(np.isclose(t, e, rtol=1e-3, atol=5e-4) for t, e in zip(truth, estimated))


def _print_rankings(reports: Sequence[BenchmarkReport]) -> None:
    viable = [r for r in reports if r.succeeded]
    unranked = [r for r in reports if not r.succeeded]

    # rank by success rate (descending) then mean time (ascending)
    ranked = sorted(viable, key=lambda r: (-r.success_rate, r.mean_time))

    table = Table(title="Optimizer Rankings", show_lines=True)
    table.add_column("Rank", justify="right")
    table.add_column("Optimizer")
    table.add_column("Success Rate", justify="right")
    table.add_column("Mean Time (s)", justify="right")
    table.add_column("Mean FEs", justify="right")
    table.add_column("Good Guess Time (s)", justify="right")

    for i, r in enumerate(ranked, start=1):
        color = "green" if r.perfect else "yellow"
        table.add_row(
            str(i),
            f"[{color}]{r.optimizer_name}[/{color}]",
            f"{r.success_rate:.0%}",
            f"{r.mean_time:.3f}",
            f"{r.mean_nfev:.0f}",
            f"{r.good_guess_time:.3f}",
        )

    for r in unranked:
        table.add_row("N/A", f"[red]{r.optimizer_name}[/red]", "0%", "N/A", "N/A", "N/A")

    rich.print(table)


def _report_result(names, truth, estimated, ll):
    is_good = True
    for name, gt, est in zip(names, truth, estimated, strict=True):
        is_close = np.isclose(gt, est, rtol=1e-3, atol=5e-4)
        equality = "~" if is_close else "!="
        is_good &= is_close
        rich.print(f"[cyan]{name}[/cyan]: [green]{gt:.5f}[/green] {equality} [magenta]{est:.5f}[/magenta]")
    rich.print(f"Minimized neg likelihood = {ll:.5f}")
    rich.print("Solution is " + ("[green]good[/green]!" if is_good else "[red]bad[/red]!"))


def main(raw_args: Sequence[str]) -> None:
    args = _parse_args(raw_args)
    pseed: int | None = int(args.pseed) if args.pseed is not None else None
    names: list[str] = ["alpha", "b", "c", "theta0", "S", "rho"]
    lb: onp.Array1D[np.float64] = np.array([0.0, 0.005, 0.0, -np.pi, 30.0, 0.0])
    ub: onp.Array1D[np.float64] = np.array([1.0, 0.1, 0.004, +np.pi, 70.0, 0.18])

    if pseed is None:
        signal_comp = AlinderComponent(
            alpha=0.5,
            b=0.05,
            c=0.002,
            theta0=0.0,
            scale_factor=40.00,
            rho=0.09,
            winding=1,
        )
    else:
        rng = np.random.default_rng(pseed)
        signal_comp = AlinderComponent(
            alpha=rng.uniform(low=lb[0], high=ub[0]),
            b=rng.uniform(low=lb[1], high=ub[1]),
            c=rng.uniform(low=lb[2], high=ub[2]),
            theta0=rng.uniform(low=lb[3], high=ub[3]),
            scale_factor=rng.uniform(low=lb[4], high=ub[4]),
            rho=rng.uniform(low=lb[5], high=ub[5]),
        )
    true_params = np.array(
        [signal_comp.alpha, signal_comp.b, signal_comp.c, signal_comp.theta0, signal_comp.scale_factor, signal_comp.rho]
    )

    objective, data_ctx = _create_objective(signal_comp, use_rust=args.use_rust)
    good_guess = np.array([0.5, 0.05, 0.002, 0.0, 42.0, 0.09])

    optimizer_selection: list[str] = list(args.optimizers)
    if "all" in optimizer_selection:
        selected = list(OPTIMIZER_REGISTRY.keys())
    elif "default" in optimizer_selection:
        selected = [
            "de",
            "multistart",
            "basinhopping",
            "tiktak",
            "rust-tiktak",
        ]
    else:
        selected = optimizer_selection
    optimizers = [OPTIMIZER_REGISTRY[name](objective, data_ctx) for name in selected]

    num_random: int = int(args.num_random)
    seed: int = int(args.seed)

    if args.use_rust:
        rich.print("[cyan]Using Rust PSpiralModel for predictions[/cyan]")
    else:
        rich.print("[cyan]Using Python PSpiralModel for predictions[/cyan]")

    reports = [
        _check_accuracy(names, true_params, opt, guess=good_guess, lb=lb, ub=ub, num_random=num_random, seed=seed)
        for opt in optimizers
    ]

    _print_rankings(reports)


def _parse_args(raw_args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark optimizers against the psnailder objective.")
    _ = parser.add_argument(
        "optimizers",
        nargs="*",
        default=["default"],
        metavar="OPTIMIZER",
        help=f"Optimizers to benchmark. Choices: {', '.join(OPTIMIZER_REGISTRY)}. Defaults to all.",
    )
    _ = parser.add_argument(
        "-nrandom",
        "--nrandom",
        dest="num_random",
        type=int,
        default=10,
        help="Number of random restarts per optimizer (default: 10).",
    )
    _ = parser.add_argument("--seed", type=int, default=42, help="RNG seed for random guesses (default: 42).")
    _ = parser.add_argument("--pseed", type=int, dest="pseed", help="RNG seed for parameters. (default: 53).")
    _ = parser.add_argument(
        "--use-rust",
        action="store_true",
        help="Use the Rust PSpiralModel for predictions instead of the Python implementation.",
    )
    return parser.parse_args(raw_args)


if __name__ == "__main__":
    main(sys.argv[1:])
