"""scipy.optimize-based phase synthesis (always available).

Loss formulation:

    minimize_{phase ∈ R^{ny·nx}}  Σ_{(u,v)} w(u, v) · (|E_synth(u, v)| − |E_target(u, v)|)²

E_synth is computed from the array's per-cell phases via PO (cell amplitudes
are taken as 1 for unit-amplitude reflectarrays). Initialization uses the
analytical beam-steering phase if a ``beam_direction`` is supplied; otherwise
random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]

from ..units import k0


@dataclass(frozen=True, slots=True)
class SynthResult:
    """Result of an optimization-based phase synthesis."""

    phase: NDArray[np.float64]
    """Per-cell phase matrix [rad]."""
    final_loss: float
    n_iterations: int
    converged: bool


def synth_phase_scipy(
    target_magnitude: NDArray[np.float64],
    array: object,
    freq: float,
    *,
    weights: NDArray[np.float64] | None = None,
    method: str = "L-BFGS-B",
    max_iter: int = 200,
    init: NDArray[np.float64] | None = None,
    seed: int = 0,
) -> SynthResult:
    """Synthesize per-cell phases that approximate *target_magnitude*.

    Parameters
    ----------
    target_magnitude
        Desired far-field magnitude on the array's natural (u, v) grid.
        Shape must be ``(M, M)`` where ``M`` is the FFT pad size used in
        synthesis (default M = nx).
    array
        A `Reflectarray`-like object exposing ``cell_centers()`` and
        ``aperture_size``.
    freq
        Operating frequency [Hz].
    weights
        Per-direction weight grid (same shape as target_magnitude). Use to
        emphasize main-beam directions or null regions.
    method
        scipy.optimize method (default ``L-BFGS-B``, which uses analytical
        gradients we provide).
    max_iter
        Maximum iterations.
    init
        Initial phase guess (else random).
    seed
        RNG seed for the random init.
    """
    nx, ny = array.nx, array.ny
    if not hasattr(array, "cell_centers"):
        raise TypeError("array must expose cell_centers().")
    x, y = array.cell_centers()
    X, Y = np.meshgrid(x, y, indexing="xy")
    k = k0(freq)

    target = np.asarray(target_magnitude, dtype=np.float64)
    M = target.shape[0]
    if target.shape != (M, M):
        raise ValueError("target_magnitude must be square (M, M).")
    if weights is None:
        weights = np.ones_like(target)

    # (u, v) grid matching the target.
    u = np.linspace(-1.0, 1.0, M)
    v = np.linspace(-1.0, 1.0, M)
    U, V = np.meshgrid(u, v, indexing="xy")
    visible = U**2 + V**2 <= 1.0

    # Pre-compute the spatial-frequency basis once.
    # E_synth(u, v) = Σ_{cells} exp(jφ_cell + jk(u·x + v·y))
    # = sum over cells of exp(j·k(u·X + v·Y)) · exp(jφ).
    phase_kernel = np.exp(
        1j * k * (U[..., None, None] * X[None, None, ...] + V[..., None, None] * Y[None, None, ...])
    )  # shape (M, M, ny, nx)

    def loss_and_grad(phi_flat: np.ndarray) -> tuple[float, np.ndarray]:
        phi = phi_flat.reshape(ny, nx)
        c = np.exp(1j * phi)
        E = np.einsum("uvyx,yx->uv", phase_kernel, c)  # (M, M)
        mag = np.abs(E)
        diff = (mag - target) * visible
        L = float(np.sum(weights * diff**2))
        # gradient of L wrt φ:
        # ∂|E|/∂φ_cell = Re( (E* / |E|) · ∂E/∂φ ) = Re( (E* / |E|) · j·c · phase_kernel ).
        with np.errstate(divide="ignore", invalid="ignore"):
            E_unit = np.where(mag > 1e-30, E / mag, 0)
        # term = 2·w·(mag − target) · ∂|E|/∂φ
        d_mag_dphi = np.real(
            np.conj(E_unit)[..., None, None] * 1j * phase_kernel * c[None, None, ...]
        )
        grad = 2 * np.sum(
            weights[..., None, None] * (diff * visible)[..., None, None] * d_mag_dphi, axis=(0, 1)
        )
        return L, grad.flatten()

    if init is None:
        rng = np.random.default_rng(seed)
        init = rng.uniform(-np.pi, np.pi, size=(ny, nx))
    init_flat = np.asarray(init).flatten()

    res = minimize(
        loss_and_grad,
        init_flat,
        jac=True,
        method=method,
        options={"maxiter": max_iter},
    )
    phase = res.x.reshape(ny, nx)
    return SynthResult(
        phase=phase,
        final_loss=float(res.fun),
        n_iterations=int(res.nit) if hasattr(res, "nit") else max_iter,
        converged=bool(res.success),
    )
