"""Array-factor helpers for macro arrays of Fresnel-antenna elements.

The array factor for an N-element array with complex weights `w_i` and
positions `(x_i, y_i)` evaluated on a (u, v) = (sin θ cos φ, sin θ sin φ)
grid is

    AF(u, v) = Σ_{i=0}^{N-1}  w_i · exp[+j · k · (u·x_i + v·y_i)]

following the e^{jωt} time convention (Mailloux, *Phased Array Antenna
Handbook*). The total array far-field is the elementwise product of the
single-element pattern with this scalar AF, valid when all elements are
identical and mutual coupling is negligible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..units import k0


def steering_vector(
    theta_deg: float, phi_deg: float, freq: float, positions: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Forward steering vector ``exp(+j·k·(u·x_i + v·y_i))`` for a target.

    The conjugate ``np.conj(steering_vector(...))`` is the **conjugate-matched
    receive weight vector** that lands a beam at (θ, φ).
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must be (N, 2); got {positions.shape}")
    u_b = float(np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg)))
    v_b = float(np.sin(np.deg2rad(theta_deg)) * np.sin(np.deg2rad(phi_deg)))
    k = k0(freq)
    return np.exp(1j * k * (u_b * positions[:, 0] + v_b * positions[:, 1])).astype(np.complex128)


def array_factor(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    freq: float,
    positions: NDArray[np.float64],
    weights: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Evaluate ``AF(u, v) = Σ w_i · exp(+j·k·(u·x_i + v·y_i))`` on a grid.

    *u* and *v* are 1-D direction-cosine vectors (matching ``FarField.u``,
    ``FarField.v``); the result has shape ``(len(v), len(u))``. *positions*
    is ``(N, 2)`` and *weights* is ``(N,)`` complex.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must be (N, 2); got {positions.shape}")
    if weights.shape != (positions.shape[0],):
        raise ValueError(f"weights shape {weights.shape} does not match N={positions.shape[0]}")
    k = k0(freq)
    U, V = np.meshgrid(u, v, indexing="xy")
    # Vectorize over elements: phase[i, y, x] = k·(U[y,x]·x_i + V[y,x]·y_i)
    # Sum_i w_i · exp(+1j·phase_i) — done via einsum for memory efficiency.
    Ux = np.einsum("yx,n->nyx", U, positions[:, 0])
    Vy = np.einsum("yx,n->nyx", V, positions[:, 1])
    phase = k * (Ux + Vy)
    af = np.einsum("n,nyx->yx", weights, np.exp(1j * phase))
    return af.astype(np.complex128)


def quantize_weights(weights: NDArray[np.complex128], bits: int) -> NDArray[np.complex128]:
    """Quantize complex weights to an N-bit phase shifter (unit magnitude).

    Each weight is replaced by ``|w|·exp(j·θ_q)`` where θ_q is the nearest
    of ``2^bits`` equally-spaced phase steps. ``bits=0`` is a no-op
    (continuous phase). ``bits=1`` is a 1-bit array (180° steps); ``bits=2``
    is a 2-bit array (90° steps); typical RFICs ship 4–6 bits.
    """
    if bits <= 0:
        return weights.astype(np.complex128)
    n_steps = 2**bits
    step = 2.0 * np.pi / n_steps
    mag = np.abs(weights)
    phase = np.angle(weights)
    quantized_phase = np.round(phase / step) * step
    return (mag * np.exp(1j * quantized_phase)).astype(np.complex128)
