"""Varactor-diode unit cell — patch-resonator model.

A varactor-loaded patch behaves as an LC resonator whose resonance frequency
shifts with the diode's voltage-dependent capacitance. The reflection phase
of a resonant cell of quality factor Q is

    φ(f) = π − 2·arctan( (f − f_res(V)) / (f_res / (2Q)) )

where f_res(V) = 1/(2π·√(L_patch · C(V))). As V sweeps the diode capacitance
through its full range, f_res moves across the operating frequency, taking
the reflection phase through ~360°. Loss = 1 − 1/Q (canonical resonant
absorption at the centre, dropping off the wings).

Default factories use publicly published C(V) curves for two industry-standard
varactors:

* **Skyworks SMV1232** — silicon abrupt-junction, popular at 28 GHz
  (C ≈ 1.32 pF at 0 V → 0.45 pF at −15 V).
* **MACOM MAVR-011020** — GaAs hyperabrupt for mmW (C ≈ 1.10 pF → 0.20 pF).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import TunableCell


@dataclass(frozen=True, slots=True)
class VaractorCell(TunableCell):
    C0: float
    """Junction capacitance at zero bias [F]."""
    Cmin: float
    """Capacitance at maximum reverse bias [F]."""
    V_breakdown: float
    """Breakdown / max reverse voltage [V] (positive number, applied as −V)."""
    L_patch: float = 0.064e-9
    """Effective patch inductance [H]; default places resonance at ~ 28 GHz."""
    Q: float = 30.0
    """Loaded quality factor (controls loss and phase slope at resonance)."""
    name: str = "VaractorCell"

    def _capacitance(self, voltage: NDArray[np.float64]) -> NDArray[np.float64]:
        if abs(self.C0 - self.Cmin) < 1e-21:
            return np.full_like(voltage, self.C0)
        ratio = (self.C0 / self.Cmin) ** 2
        V_phi = self.V_breakdown / max(ratio - 1.0, 1e-9)
        return self.C0 / np.sqrt(1.0 + np.clip(voltage, 0, self.V_breakdown) / V_phi)

    def _f_res(self, C: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 / (2.0 * np.pi * np.sqrt(self.L_patch * np.maximum(C, 1e-21)))

    def phase(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        v = np.abs(np.asarray(state, dtype=np.float64))
        C = self._capacitance(v)
        f_res = self._f_res(C)
        bandwidth = f_res / (2.0 * self.Q)
        # Lorentzian phase: passes through 0 at f = f_res, ±π at the wings.
        return -2.0 * np.arctan2(freq - f_res, bandwidth)

    def loss(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        v = np.abs(np.asarray(state, dtype=np.float64))
        C = self._capacitance(v)
        f_res = self._f_res(C)
        bandwidth = f_res / (2.0 * self.Q)
        # |Γ| = 1 − 1/Q · 1/(1 + ((f − f_res) / bw)²) — real reflectarray cells
        # absorb most strongly at resonance and pass with low loss off-resonance.
        abs_dip = 1.0 / self.Q
        rel = (freq - f_res) / bandwidth
        return 1.0 - abs_dip / (1.0 + rel**2)

    def state_set(self, freq: float) -> tuple[NDArray[np.float64], str]:
        # Sample voltage range with finer resolution near the edges.
        return np.linspace(0.0, self.V_breakdown, 96), "continuous"


def Skyworks_SMV1232() -> VaractorCell:
    """Factory — Skyworks SMV1232 (datasheet typical values).

    Patch inductance tuned so f_res sweeps through 28 GHz across the diode's
    voltage range — gives ≥ 300° usable phase coverage at 28 GHz.
    """
    return VaractorCell(
        C0=1.32e-12,
        Cmin=0.45e-12,
        V_breakdown=15.0,
        L_patch=0.05e-9,
        Q=30.0,
        name="Skyworks SMV1232",
    )


def MACOM_MAVR011020() -> VaractorCell:
    """Factory — MACOM MAVR-011020 (GaAs hyperabrupt, mmW)."""
    return VaractorCell(
        C0=1.10e-12,
        Cmin=0.20e-12,
        V_breakdown=15.0,
        L_patch=0.020e-9,
        Q=50.0,
        name="MACOM MAVR-011020",
    )
