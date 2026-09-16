"""Tuning script to find the optimal parameters for each optimizer on this machine."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import rich
from phasmix.component import AlinderComponent
from rich.table import Table

from benchmark_optimize import (
    ScipyBasinHoppingOpt,
    ScipyDEOpt,
    ScipyNaiveMultistartOpt,
    TikTakOpt,
    _create_objective,
)

if TYPE_CHECKING:
    from benchmark_optimize import Optimizer


def run_trials(
    opt_factory: callable[[], Optimizer],
    truth: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    num_trials: int = 10,
    seed: int = 42,
) -> tuple[float, float]:
    """Run multiple trials and return (success_rate, mean_time)."""
    rng = np.random.default_rng(seed)
    successes = 0
    total_time = 0.0

    for _ in range(num_trials):
        # Generate a new random problem for each trial to avoid overfitting to one scenario
        # but keep it reproducible with seed if needed. Actually we want it to be robust.
        # Let's keep the target parameters fixed but randomize the starting guess.
        guess = rng.uniform(lb, ub)
        
        # We need a fresh objective for each trial if we were randomizing truth, 
        # but for tuning speed vs robustness we can use a fixed truth.
        signal_comp = AlinderComponent(
            alpha=truth[0], b=truth[1], c=truth[2], 
            theta0=truth[3], scale_factor=truth[4], rho=truth[5],
            winding=1
        )
        objective = _create_objective(signal_comp)
        opt = opt_factory(objective)

        t0 = time.perf_counter()
        estimated, _, _ = opt.minimize(guess, lb, ub)
        total_time += time.perf_counter() - t0
        
        is_success = all(np.isclose(t, e, rtol=2e-2, atol=1e-2) for t, e in zip(truth, estimated))
        successes += 1 if is_success else 0

    return successes / num_trials, total_time / num_trials


def main():
    rich.print("[bold blue]Tuning Optimizers for Reliability and Speed (Multi-Target)[/bold blue]")
    
    # Multiple target scenarios to ensure robustness
    targets = [
        np.array([0.5, 0.05, 0.002, 0.0, 40.0, 0.09]),      # Central
        np.array([0.2, 0.02, 0.001, 1.5, 35.0, 0.05]),     # Small/Low
        np.array([0.8, 0.08, 0.003, -1.5, 65.0, 0.15]),    # Large/High
    ]
    
    lb = np.array([0.0, 0.005, 0.0, -np.pi, 30.0, 0.0])
    ub = np.array([1.0, 0.1, 0.004, +np.pi, 70.0, 0.18])
    
    num_trials_per_target = 5 

    results = []

    def run_multi_target_trials(opt_factory):
        all_successes = 0
        total_trials = len(targets) * num_trials_per_target
        total_time = 0.0
        for truth in targets:
            rate, mtime = run_trials(opt_factory, truth, lb, ub, num_trials_per_target)
            all_successes += int(rate * num_trials_per_target)
            total_time += mtime * num_trials_per_target
        return all_successes / total_trials, total_time / total_trials

    # 1. Tuning TikTak
    rich.print("\n[yellow]Tuning TikTak...[/yellow]")
    for n_sobol in [512, 1024, 2048, 4096]:
        for n_star in [16, 32, 64, 128]:
            if n_star > n_sobol: continue
            rate, mtime = run_multi_target_trials(lambda obj: TikTakOpt(obj, num_sobol=n_sobol, n_star=n_star, seed=42))
            results.append(("TikTak", f"sobol={n_sobol}, star={n_star}", rate, mtime))
            rich.print(f"  sobol={n_sobol:4}, star={n_star:2} -> Success: {rate:4.0%}, Time: {mtime:.3f}s")
            if rate >= 1.0: break # Found a truly viable set

    # 2. Tuning Multistart
    rich.print("\n[yellow]Tuning Multistart...[/yellow]")
    for n_restarts in [20, 40, 60]:
        rate, mtime = run_multi_target_trials(lambda obj: ScipyNaiveMultistartOpt(obj, seed=42, n_restarts=n_restarts))
        results.append(("Multistart", f"starts={n_restarts}", rate, mtime))
        rich.print(f"  starts={n_restarts:3} -> Success: {rate:4.0%}, Time: {mtime:.3f}s")
        if rate >= 0.95: break

    # 3. Tuning DE
    rich.print("\n[yellow]Tuning Differential Evolution...[/yellow]")
    for popsize in [10, 15]:
        for maxiter in [50, 100, 200]:
            rate, mtime = run_multi_target_trials(lambda obj: ScipyDEOpt(obj, popsize=popsize, maxiter=maxiter))
            results.append(("DE", f"pop={popsize}, iter={maxiter}", rate, mtime))
            rich.print(f"  pop={popsize:2}, iter={maxiter:3} -> Success: {rate:4.0%}, Time: {mtime:.3f}s")
            if rate >= 0.95: break

    # 4. Tuning Basinhopping
    rich.print("\n[yellow]Tuning Basinhopping...[/yellow]")
    for niter in [50, 100, 200]:
        rate, mtime = run_trials(lambda obj: ScipyBasinHoppingOpt(obj, niter=niter), targets[0], lb, ub, num_trials_per_target)
        results.append(("Basinhopping", f"niter={niter}", rate, mtime))
        rich.print(f"  niter={niter:3} -> Success: {rate:4.0%}, Time: {mtime:.3f}s")
        if rate == 1.0: break

    # Summary table
    table = Table(title="Tuning Summary (Sorted by Speed)")
    table.add_column("Optimizer")
    table.add_column("Params")
    table.add_column("Success Rate")
    table.add_column("Mean Time (s)")
    
    # Filter for 100% success and sort by time
    viable = [r for r in results if r[2] == 1.0]
    for r in sorted(viable, key=lambda x: x[3]):
        table.add_row(r[0], r[1], f"[green]{r[2]:.0%}[/green]", f"{r[3]:.3f}")
    
    rich.print("\n")
    rich.print(table)

if __name__ == "__main__":
    main()
