"""Metasurface unit cells for Phase 3 (Pancharatnam–Berry, anisotropic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import TunableCell


@dataclass(frozen=True, slots=True)
class PancharatnamBerryCell(TunableCell):
    """Geometric-phase cell — output phase = ±2 × rotation angle.

    Under circular polarization, rotating the cell by α imparts a far-field
    phase of ±2α (sign = handedness of the incident polarization). The cell
    *converts* polarization handedness on transmission (LCP ↔ RCP).

    *state* is the rotation angle in radians.
    """

    sense: str = "LCP_to_RCP"
    """Handedness mapping: 'LCP_to_RCP' or 'RCP_to_LCP'."""
    name: str = "PancharatnamBerryCell"

    def phase(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        alpha = np.asarray(state, dtype=np.float64)
        sign = +2.0 if self.sense == "LCP_to_RCP" else -2.0
        return sign * alpha

    def loss(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        # Idealized: loss-less PB cell. Real cells lose ~0.5 dB.
        s = np.asarray(state, dtype=np.float64)
        return np.full_like(s, 10 ** (-0.5 / 20.0))

    def state_set(self, freq: float) -> tuple[NDArray[np.float64], str]:
        return np.linspace(0.0, np.pi, 64), "continuous"


@dataclass(frozen=True, slots=True)
class AnisotropicEllipseCell(TunableCell):
    """Anisotropic ellipse resonator with a Jones matrix.

    Encodes a (rotation, semi-major / semi-minor) per cell. The Jones matrix
    in the cell's local frame is diagonal with phase delays φ_a, φ_b on the
    two principal axes; the global Jones matrix follows by rotation.

    *state* is a (rotation, axis_ratio) tuple per cell — exposed via the
    `jones_matrix(state, freq)` method since it doesn't fit the scalar
    phase / loss API exactly.
    """

    eps_par: float = 3.55  # along long axis
    eps_perp: float = 2.20  # along short axis
    thickness: float = 100e-6
    name: str = "AnisotropicEllipseCell"

    def _phi(self, eps: float, freq: float) -> float:
        n = float(np.sqrt(eps))
        lam = 3e8 / freq
        return -2.0 * np.pi * (n - 1.0) * self.thickness / lam

    def jones_matrix(self, rotation: ArrayLike, freq: float) -> NDArray[np.complex128]:
        """Return per-cell 2x2 Jones matrix; output shape (..., 2, 2)."""
        alpha = np.asarray(rotation, dtype=np.float64)
        phi_a = self._phi(self.eps_par, freq)
        phi_b = self._phi(self.eps_perp, freq)
        # Diagonal Jones in the cell frame.
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        Ea = np.exp(1j * phi_a)
        Eb = np.exp(1j * phi_b)
        # Global Jones = R(α) · diag(Ea, Eb) · R(−α).
        J = np.empty((*alpha.shape, 2, 2), dtype=np.complex128)
        J[..., 0, 0] = ca**2 * Ea + sa**2 * Eb
        J[..., 0, 1] = ca * sa * (Ea - Eb)
        J[..., 1, 0] = ca * sa * (Ea - Eb)
        J[..., 1, 1] = sa**2 * Ea + ca**2 * Eb
        return J

    # The scalar-cell API still works for x-polarized input only.
    def phase(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        J = self.jones_matrix(state, freq)
        return np.angle(J[..., 0, 0])

    def loss(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        J = self.jones_matrix(state, freq)
        return np.abs(J[..., 0, 0])

    def state_set(self, freq: float) -> tuple[NDArray[np.float64], str]:
        return np.linspace(0.0, np.pi, 32), "continuous"
