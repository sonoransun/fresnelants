"""Dielectric materials and conductor data used by phase-correcting designs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Dielectric:
    """Lossy dielectric with relative permittivity εr and loss tangent tanδ."""

    name: str
    eps_r: float
    tan_delta: float = 0.0

    @property
    def n(self) -> float:
        """Refractive index, n = √εr (non-magnetic)."""
        return float(np.sqrt(self.eps_r))

    def groove_depth_for_2pi(self, wavelength: float) -> float:
        """Groove depth that imparts 2π phase delay relative to free space.

        d = λ / (n − 1). Used when designing N-level phase-correcting plates.
        """
        if self.n <= 1.0:
            raise ValueError("Refractive index must be > 1 for a phase-correcting groove.")
        return wavelength / (self.n - 1.0)


# A small library of common low-loss dielectrics relevant at microwave / mmW.
PTFE = Dielectric("PTFE (Teflon)", eps_r=2.08, tan_delta=2e-4)
HDPE = Dielectric("HDPE", eps_r=2.34, tan_delta=2e-4)
POLYSTYRENE = Dielectric("Polystyrene", eps_r=2.55, tan_delta=3e-4)
REXOLITE = Dielectric("Rexolite 1422", eps_r=2.53, tan_delta=6.6e-4)
ROGERS_RO4003 = Dielectric("Rogers RO4003C", eps_r=3.55, tan_delta=2.7e-3)
ROGERS_RT5880 = Dielectric("Rogers RT/duroid 5880", eps_r=2.20, tan_delta=9e-4)
ALUMINA = Dielectric("Alumina (Al2O3)", eps_r=9.8, tan_delta=2e-4)
FUSED_SILICA = Dielectric("Fused silica", eps_r=3.78, tan_delta=1e-4)

LIBRARY: dict[str, Dielectric] = {
    d.name: d
    for d in (
        PTFE,
        HDPE,
        POLYSTYRENE,
        REXOLITE,
        ROGERS_RO4003,
        ROGERS_RT5880,
        ALUMINA,
        FUSED_SILICA,
    )
}
