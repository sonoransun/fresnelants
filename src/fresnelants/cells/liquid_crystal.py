"""Liquid-crystal phase-shifting cell (sub-THz / THz-friendly).

The cell is modelled as a thin LC layer between two electrodes; applying a
DC bias rotates the LC director, changing the effective relative permittivity
seen by the RF wave. Phase shift scales with cell thickness × Δε / λ.

Defaults reflect industrial LCs from the LCD-substrate supply chain:

* **Merck GT3** — broadband mmW / sub-THz LC; Δε ≈ 0.81 around 100 GHz.
* **Merck E7** — workhorse LC, lower birefringence (Δε ≈ 0.45 at 100 GHz),
  cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import TunableCell


@dataclass(frozen=True, slots=True)
class LiquidCrystalCell(TunableCell):
    eps_par: float
    """Dielectric constant parallel to the director."""
    eps_perp: float
    """Dielectric constant perpendicular to the director."""
    cell_thickness: float
    """LC layer thickness [m]."""
    response_time_ms: float = 20.0
    name: str = "LiquidCrystalCell"

    def _eps_eff(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Effective ε at director angle θ (0 = perpendicular, π/2 = parallel)."""
        eps_par = self.eps_par
        eps_perp = self.eps_perp
        return (eps_par * eps_perp) / (eps_par * np.cos(theta) ** 2 + eps_perp * np.sin(theta) ** 2)

    multipass: int = 2
    """Number of round-trips the wave makes through the LC layer (≥ 2 for a
    conventional reflective cell; higher for Fabry–Perot LC cells)."""

    def phase(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        """Optical path difference accumulated through the LC layer.

        Phase = −multipass · k · (n_eff(θ) − √ε_perp) · cell_thickness, with
        the offset chosen so that θ=0 (LC perpendicular) gives zero phase.
        """
        theta = np.asarray(state, dtype=np.float64)
        eps_eff = self._eps_eff(theta)
        n_eff = np.sqrt(eps_eff)
        lam = 3e8 / freq
        return (
            -self.multipass
            * (2.0 * np.pi / lam)
            * (n_eff - np.sqrt(self.eps_perp))
            * self.cell_thickness
        )

    def loss(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        s = np.asarray(state, dtype=np.float64)
        return np.full_like(s, 10 ** (-0.5 / 20.0))

    def state_set(self, freq: float) -> tuple[NDArray[np.float64], str]:
        return np.linspace(0.0, np.pi / 2, 64), "continuous"


def Merck_GT3(cell_thickness: float = 1.5e-3, multipass: int = 8) -> LiquidCrystalCell:
    """Factory — Merck GT3 LC (broadband mmW / sub-THz).

    Defaults model a Fabry–Perot LC cell (8 internal bounces in a 1.5 mm
    cavity), reproducing the > 300° phase swing reported for sub-THz LC RIS
    at 100 GHz. Pass `multipass=2` for a simple reflective single-pass model
    (then bump cell_thickness to ~6 mm to recover the swing).
    """
    return LiquidCrystalCell(
        eps_par=3.20,
        eps_perp=2.39,
        cell_thickness=cell_thickness,
        response_time_ms=15.0,
        multipass=multipass,
        name="Merck GT3",
    )


def Merck_E7(cell_thickness: float = 2.5e-3, multipass: int = 8) -> LiquidCrystalCell:
    """Factory — Merck E7 LC (lower birefringence; thicker cell or higher multipass)."""
    return LiquidCrystalCell(
        eps_par=2.80,
        eps_perp=2.35,
        cell_thickness=cell_thickness,
        response_time_ms=30.0,
        multipass=multipass,
        name="Merck E7",
    )
