# psnailder: Python fitting API

Fit phase-spiral models to position/velocity samples or an existing count map.
The examples below describe the Python fitter; the private Rust binding does
not yet expose the same bounds or outcome API.

## Bounds and units

```python
from psnailder.bounds import Fixed, Interval, ParameterBounds
from psnailder.fit import PSpiralFitter

shared_bounds = ParameterBounds(
    alpha=(0.0, 0.8),
    c=Fixed(0.0),
    scale_factor=Interval(35.0, 60.0),
)
fitter = PSpiralFitter(bounds=shared_bounds, max_iterations=20)
```

Each keyword accepts an `Interval`, a two-value sequence, a `Fixed`, or a scalar.
A scalar or equal interval endpoints fixes that parameter. Omitted keywords use
defaults. Fixed parameters are excluded from optimization; an all-fixed model is
evaluated without running differential evolution.

The fitter does not convert units. The defaults assume `z` in kpc and `vz` in
km/s. From `r = sqrt(z**2 + (vz/S)**2)` and `r = b*phi_s + c*phi_s**2`:

| Parameter | Meaning | Units with these coordinates | Default interval | Allowed domain |
| --- | --- | --- | --- | --- |
| `alpha` | Perturbation amplitude | Dimensionless | `[0, 1]` | Nonnegative |
| `b` | Linear winding coefficient | kpc/rad | `[0.005, 0.1]` | Positive |
| `c` | Quadratic winding coefficient | kpc/rad² | `[0, 0.004]` | Nonnegative |
| `theta0` | Angular offset | rad | `[-π, π]` | Any finite value |
| `scale_factor` | Velocity-to-position scale S | km/s/kpc | `[30, 70]` | Positive |
| `rho` | Central flattening transition radius | kpc | `[0, 0.18]` | Nonnegative |

All interval endpoints must be finite and ordered. Default intervals are search
choices, not the full valid domains. Radians are dimensionless; retaining them
in the coefficient units makes the phase convention explicit.

Angles remain unwrapped. For example, `(np.pi - 0.2, np.pi + 0.2)` is a valid
interval around π. Wider-than-2π intervals are allowed but search redundant angles.

One bounds object broadcasts constraints to every component; free parameter
values remain independent. For separate arm constraints, supply a sequence:

```python
import numpy as np

two_arm_fitter = PSpiralFitter(bounds=(
    ParameterBounds(theta0=(-0.2, 0.2)),
    ParameterBounds(theta0=(np.pi - 0.2, np.pi + 0.2)),
))
```

An N-component fit uses the first N bounds entries. Automatic selection compares
one and two components, so an explicit sequence needs at least two entries. Its
one-component candidate uses only the first entry: ordering matters. A one-entry
sequence does not broadcast. These independent angle intervals do not impose an
exact relative separation between the arms.

## Fit samples and handle the outcome

Given paired one-dimensional sample arrays `z`, `vz`, and increasing bin edges:

```python
from psnailder.fit import FitFailure

outcome = fitter.fit_spiral(
    z, vz, z_bins, vz_bins,
    num_components=None,
    winding=None,
    rng=np.random.default_rng(42),
)

if isinstance(outcome, FitFailure):
    print(outcome.reason, outcome.message)
else:
    result = outcome.result
    prediction = result.final_model.prediction()
    print(result.reason, result.num_iterations, result.lnl)
```

`num_components=None` compares one- and two-component initial fits using BIC,
counting free parameters. `winding=None` tries both directions. The selected count
and winding stay fixed during background refinement.

`FitSuccess` means a valid fit exists, not that optimization or refinement
necessarily converged. Inspect `result.reason`: `fixed_background`,
`iteration_limit`, `converged`, `no_improvement`, `invalid_background_update`, or
`failed_reoptimization`. If refinement fails after a valid fit exists, that fit
is retained. Invalid configuration may still raise exceptions; the outcome API
does not catch every possible input or numerical-library error.

`warm_start` is a full, flat parameter vector, ordered
`[alpha, b, c, theta0, scale_factor, rho]` for each component. It requires an
explicit `num_components`. Fixed coordinates are replaced by their configured
values; free coordinates are clipped slightly inside their bounds. The vector
seeds one DE population member rather than resuming optimizer state.

For reproducible calls, construct a fresh `np.random.default_rng(seed)` each time.
Reusing the same generator advances its state. Avoid changing supplied bounds or
arrays while fitting, especially while consuming a lazy generator.

## Existing count maps

Use `fit_spiral_with_background(density, background, z_mesh, vz_mesh, ...)` when
the data are already binned. All four arrays must have the same 2D shape, with
rows along `vz` and columns along `z`. For sample binning, this is
`(len(vz_bins) - 1, len(z_bins) - 1)`. `density` contains counts per cell.

Sample fitting multiplies the KDE density at each bin centre by that bin's area
before normalizing to the observed count within the fitting region. This supports
unequal widths using a midpoint approximation, not an exact integral over each
bin. Already-binned backgrounds should represent counts per cell, not density.

The default refinement smoother operates in grid-cell units. On unequal bins,
it is not a uniform physical-width kernel; supply a custom `smoothing_func` or
disable refinement when that distinction matters.

Every prediction is normalized to the observed total, and the winning scale is
stored in the model's background. `improve_background=False` keeps the supplied
background's shape, but its overall scale can change. The caller's background
array is not rescaled in place.

## Progress and iteration budgets

```python
from psnailder.fit import FitProgress

for event in fitter.fit_spiral_gen(
    z, vz, z_bins, vz_bins, rng=np.random.default_rng(42),
):
    if isinstance(event, FitProgress):
        print(event.iteration, event.lnl)
    elif isinstance(event, FitFailure):
        print(event.reason, event.message)
    else:
        result = event.result
        print(result.reason)
```

Both `*_gen` methods are lazy. With refinement enabled, a successful initial fit
is progress step **0**. Each accepted update emits another progress event.
Exactly one terminal success or failure follows normal execution; with refinement
disabled, only that terminal outcome is emitted.

`max_iterations` counts refinement attempts, excluding the initial fit. Rejected
attempts count even though they emit no progress snapshot. Zero keeps the initial
fit; `None` removes the iteration cap. Refinement stops on an equal or worse score.

Configure refinement tolerances with `PSpiralFitter(atol=1e-6, rtol=1e-4)`.
A positive score improvement at most `atol + rtol * abs(previous_lnl)` is
accepted and emitted as progress, then refinement stops with reason `converged`.
An equal or worse score is never accepted and stops with `no_improvement`.
Convergence takes precedence if it occurs on the last allowed iteration.
Both tolerances must be finite and nonnegative and default to zero, preserving
the previous behavior. They do not configure the underlying optimizer.

`result.initial_model` is the selected initial fixed-background fit, not the warm
start. Treat progress models and their arrays as read-only; they are not deeply
immutable snapshots.
