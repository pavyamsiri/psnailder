"""Terminal Python batch fitting behavior."""

from __future__ import annotations

import numpy as np
import pytest

from psnailder.fit import FitFailure, FitInput, FitSuccess, ParameterBounds, PSpiralFitter


@pytest.mark.parametrize("workers", [1, 2, None])
def test_batch_preserves_results_and_failures(workers: int | None) -> None:
    """Independent real fits retain input order and failures retain their slot."""
    fitter = PSpiralFitter(bounds=ParameterBounds(alpha=0.0, b=0.05, c=0.0, theta0=0.0, scale_factor=40.0, rho=0.09))
    background = np.ones((2, 3))
    mesh = np.zeros_like(background)
    inputs = [
        FitInput(
            density=background * count,
            background=background,
            z_mesh=mesh,
            vz_mesh=mesh,
            num_components=1,
            winding=1,
            improve_background=False,
        )
        for count in (2, 0, 3)
    ]
    results = fitter.fit_batch(inputs, workers=workers)
    assert len(results) == 3
    assert isinstance(results[1], FitFailure)
    for index in (0, 2):
        result = results[index]
        assert isinstance(result, FitSuccess)
        np.testing.assert_array_equal(result.result.data, inputs[index].density)
        assert result.result.lnl == 0.0
        assert result.diagnostics.nfev == 1
    np.testing.assert_array_equal(background, np.ones((2, 3)))


def test_empty_batch() -> None:
    """An empty batch has no outcomes."""
    assert PSpiralFitter().fit_batch([]) == []


def test_batch_propagates_invalid_input() -> None:
    """Invalid caller inputs raise through the thread pool."""
    grid = np.ones((2, 3))
    item = FitInput(density=grid, background=np.ones((3, 2)), z_mesh=grid, vz_mesh=grid)
    with pytest.raises(ValueError, match="shape"):
        PSpiralFitter().fit_batch([item], workers=2)


@pytest.mark.parametrize("workers", [1, 2])
def test_rust_batch_matches_single_fits(workers: int) -> None:
    """Native batches preserve order and match the existing single-fit path."""
    fitter = PSpiralFitter(
        backend="rust",
        max_iterations=1,
        bounds=ParameterBounds(alpha=0.0, b=0.05, c=0.0, theta0=0.0, scale_factor=40.0, rho=0.09),
    )
    inputs = []
    for shape, count in [((2, 3), 2), ((3, 2), 3)]:
        grid = np.full(shape, float(count))
        inputs.append(FitInput(density=grid, background=grid, z_mesh=np.zeros(shape), vz_mesh=np.zeros(shape)))
    unsupported = FitInput(density=grid, background=grid, z_mesh=grid, vz_mesh=grid, num_components=1)
    results = fitter.fit_batch([inputs[0], unsupported, inputs[1]], workers=workers)
    assert isinstance(results[1], FitFailure)
    for item, result in zip(inputs, (results[0], results[2]), strict=True):
        single = fitter.fit_spiral_with_background(item.density, item.background, item.z_mesh, item.vz_mesh)
        assert isinstance(single, FitSuccess)
        assert isinstance(result, FitSuccess)
        np.testing.assert_array_equal(result.result.data, item.density)
        np.testing.assert_array_equal(result.result.final_model.parameters, single.result.final_model.parameters)
        assert result.result.lnl == single.result.lnl
        assert result.diagnostics.nfev == single.diagnostics.nfev
    assert fitter.fit_batch([], workers=workers) == []


def test_native_batch_releases_gil_after_copying() -> None:
    """A Python thread can run during fitting without changing the copied input."""
    import sys  # noqa: PLC0415
    from threading import Event, Timer  # noqa: PLC0415

    from psnailder import _internal  # noqa: PLC0415

    grid = np.ones(128 * 128)
    mesh = np.zeros_like(grid)
    fitter = _internal.PSpiralFitter(
        max_iterations=1,
        bounds=[[(0.0, 0.0), (0.05, 0.05), (0.0, 0.0), (0.0, 0.0), (40.0, 40.0), (0.09, 0.09)]],
    )
    changed = Event()

    def mutate_source() -> None:
        grid.fill(7.0)
        changed.set()

    timer = Timer(0.05, mutate_source)
    previous_interval = sys.getswitchinterval()
    try:
        # Prevent a normal bytecode scheduling switch immediately after return
        # from making a GIL-holding native call look like it released the GIL.
        sys.setswitchinterval(10.0)
        timer.start()
        results = fitter.fit_batch([(grid, grid, np.ones_like(grid), mesh, mesh, (128, 128))], workers=1)
        ran_during_fit = changed.is_set()
    finally:
        sys.setswitchinterval(previous_interval)
        timer.join()
    assert ran_during_fit
    np.testing.assert_array_equal(results[0].data, np.ones_like(grid))
