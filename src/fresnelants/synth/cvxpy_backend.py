"""CVXPY convex-relaxation backend (optional).

Replaces the unit-amplitude constraint |c|=1 (non-convex) with |c|≤1, turning
the synthesis problem into an SOCP. The solution gives a magnitude-relaxed
optimum that's typically within a few dB of the non-convex optimum, and very
fast for moderate array sizes (≤ 64×64).
"""

from __future__ import annotations

import cvxpy as cp  # type: ignore[import-untyped]  # raises ImportError if missing
import numpy as np
from numpy.typing import NDArray

from ..units import k0
from .scipy_backend import SynthResult


def synth_phase_cvxpy(
    target_magnitude: NDArray[np.float64],
    array: object,
    freq: float,
    *,
    solver: str = "SCS",
) -> SynthResult:
    """Convex-relaxed phase synthesis. Returns a `SynthResult` with magnitudes
    embedded into the phase output (use ``np.angle`` if you only want phase).
    """
    if not hasattr(array, "cell_centers"):
        raise TypeError("array must expose cell_centers().")
    nx, ny = array.nx, array.ny
    x, y = array.cell_centers()
    X, Y = np.meshgrid(x, y, indexing="xy")
    k = k0(freq)

    M = target_magnitude.shape[0]
    u = np.linspace(-1.0, 1.0, M)
    v = np.linspace(-1.0, 1.0, M)
    U, V = np.meshgrid(u, v, indexing="xy")
    visible = U**2 + V**2 <= 1.0
    target = np.asarray(target_magnitude, dtype=np.float64)

    # Build the (M*M, ny*nx) basis matrix.
    A = np.exp(
        1j
        * k
        * (
            U.flatten()[:, None] * X.flatten()[None, :]
            + V.flatten()[:, None] * Y.flatten()[None, :]
        )
    )

    c = cp.Variable(ny * nx, complex=True)
    expr = A @ c
    # Magnitude target — use abs (CVXPY supports |.| through SOC constraints).
    constraints = [cp.abs(c) <= 1.0]
    visible_flat = visible.flatten()
    target_flat = target.flatten()
    # Squared-magnitude target on visible region only.
    t = cp.Variable(M * M, nonneg=True)
    objective = cp.Minimize(cp.sum(cp.multiply(visible_flat.astype(float), t)))
    # |Ax - target|² fitting via slack:
    constraints.append(cp.abs(expr) - target_flat <= cp.sqrt(t))
    constraints.append(target_flat - cp.abs(expr) <= cp.sqrt(t))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False)
    if c.value is None:
        raise RuntimeError(f"CVXPY solver {solver} failed: status={prob.status}")
    phase = np.angle(c.value).reshape(ny, nx)
    return SynthResult(
        phase=phase,
        final_loss=float(prob.value),
        n_iterations=0,
        converged=prob.status == "optimal",
    )
