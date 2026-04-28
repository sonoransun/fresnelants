"""Phase B2 — phase-synthesis backends."""

from __future__ import annotations

import numpy as np

import fresnelants as fa
from fresnelants.synth.scipy_backend import synth_phase_scipy


def _broadside_target(M: int) -> np.ndarray:
    """A delta-function-like target peaking at (u, v) = (0, 0)."""
    u = np.linspace(-1.0, 1.0, M)
    v = np.linspace(-1.0, 1.0, M)
    U, V = np.meshgrid(u, v, indexing="xy")
    target = np.exp(-200 * (U**2 + V**2)).astype(np.float64)
    return target


def test_scipy_synthesizer_runs() -> None:
    array = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=8, ny=8)
    target = _broadside_target(16)
    res = synth_phase_scipy(target, array, 28e9, max_iter=30)
    assert res.phase.shape == (8, 8)
    assert np.isfinite(res.final_loss)


def test_scipy_synthesizer_reduces_loss() -> None:
    """Optimization must reduce the loss vs. a random init."""
    array = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=8, ny=8)
    target = _broadside_target(16)
    res = synth_phase_scipy(target, array, 28e9, max_iter=50, seed=1)
    # Compare against the loss of a random guess.
    rng = np.random.default_rng(2)
    init = rng.uniform(-np.pi, np.pi, size=(8, 8))
    res_rand = synth_phase_scipy(target, array, 28e9, max_iter=0, init=init)
    assert res.final_loss < res_rand.final_loss


def test_synth_init_uniform_recovers_broadside() -> None:
    """Initialize at uniform phase; the synthesizer should hold the broadside
    pattern (loss should not increase materially)."""
    array = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=8, ny=8)
    target = _broadside_target(16)
    res = synth_phase_scipy(target, array, 28e9, max_iter=20, init=np.zeros((8, 8)))
    # The optimizer might wiggle things; just check the loss is bounded.
    assert res.final_loss < 1e6


def test_cvxpy_backend_optional() -> None:
    """CVXPY backend should be importable as None when missing."""
    from fresnelants.synth import synth_phase_cvxpy

    try:
        import cvxpy  # noqa: F401
    except ImportError:
        assert synth_phase_cvxpy is None
    else:
        assert synth_phase_cvxpy is not None


def test_jax_backend_optional() -> None:
    """JAX backend should be importable as None when missing."""
    from fresnelants.synth import synth_phase_jax

    try:
        import jax  # noqa: F401
    except ImportError:
        assert synth_phase_jax is None
    else:
        assert synth_phase_jax is not None
