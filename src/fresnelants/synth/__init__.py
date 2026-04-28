"""Optimization-based phase synthesis for reconfigurable arrays.

Given a target far-field magnitude pattern on a (u, v) grid, the synthesizer
finds per-cell phases (or states) that approximately produce that pattern.
Three backends:

* ``scipy_backend`` — `scipy.optimize.minimize` (always available).
* ``cvxpy_backend`` — convex SOCP relaxation (requires ``cvxpy``).
* ``jax_backend`` — autodiff via JAX (requires ``jax``).

The convex and JAX paths fall back to the scipy backend if their dependencies
are missing.
"""

from .scipy_backend import synth_phase_scipy

try:
    from .cvxpy_backend import synth_phase_cvxpy
except ImportError:  # pragma: no cover
    synth_phase_cvxpy = None  # type: ignore[assignment]

try:
    from .jax_backend import synth_phase_jax
except ImportError:  # pragma: no cover
    synth_phase_jax = None  # type: ignore[assignment]

__all__ = ["synth_phase_cvxpy", "synth_phase_jax", "synth_phase_scipy"]
