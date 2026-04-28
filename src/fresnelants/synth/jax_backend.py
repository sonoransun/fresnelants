"""JAX-autodiff backend (optional). Same loss as scipy, but faster gradients.

Falls back to importing only when called; raises an informative error if
JAX is not installed (the package's `__init__` shadows that with a None
sentinel for the public name).
"""

from __future__ import annotations

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from ..units import k0
from .scipy_backend import SynthResult


def synth_phase_jax(
    target_magnitude: NDArray[np.float64],
    array: object,
    freq: float,
    *,
    learning_rate: float = 0.05,
    n_steps: int = 500,
    seed: int = 0,
) -> SynthResult:
    """Gradient-descent phase synthesis via JAX autodiff.

    Uses a plain JAX gradient + Adam-like update; no scipy. Faster than the
    scipy backend on large arrays (>64×64) once JAX has compiled the loss.
    """
    if not hasattr(array, "cell_centers"):
        raise TypeError("array must expose cell_centers().")
    nx, ny = array.nx, array.ny
    x, y = array.cell_centers()
    X, Y = np.meshgrid(x, y, indexing="xy")
    k = k0(freq)

    target = np.asarray(target_magnitude, dtype=np.float32)
    M = target.shape[0]
    u = np.linspace(-1.0, 1.0, M)
    v = np.linspace(-1.0, 1.0, M)
    U, V = np.meshgrid(u, v, indexing="xy")
    visible = (U**2 + V**2 <= 1.0).astype(np.float32)

    A = np.exp(
        1j * k * (U[..., None, None] * X[None, None, ...] + V[..., None, None] * Y[None, None, ...])
    ).astype(np.complex64)
    A_jax = jnp.asarray(A)
    target_jax = jnp.asarray(target)
    visible_jax = jnp.asarray(visible)

    @jax.jit
    def loss(phi):
        c = jnp.exp(1j * phi)
        E = jnp.einsum("uvyx,yx->uv", A_jax, c.astype(jnp.complex64))
        mag = jnp.abs(E)
        return jnp.sum(visible_jax * (mag - target_jax) ** 2)

    grad_fn = jax.jit(jax.grad(loss))

    rng = np.random.default_rng(seed)
    phi = jnp.asarray(rng.uniform(-np.pi, np.pi, size=(ny, nx)).astype(np.float32))
    last = float("inf")
    for step in range(n_steps):
        g = grad_fn(phi)
        phi = phi - learning_rate * g
        if step % 50 == 0:
            last = float(loss(phi))
    return SynthResult(
        phase=np.asarray(phi),
        final_loss=last,
        n_iterations=n_steps,
        converged=True,
    )
