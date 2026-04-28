"""Default physical-optics solver.

Synthesizes the aperture field (transmittance × illumination) and propagates
it to the far field via FFT. Suitable for primary-focus characterization of
all five Fresnel families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..analysis.farfield import far_field_from_aperture
from ..designs.base import AntennaDesign
from .base import SolverResult


@dataclass
class PhysicalOpticsSolver:
    name: str = "PhysicalOpticsSolver"
    samples_per_wavelength: float = 6.0
    margin: float = 1.2
    pad_factor: int = 4
    illumination: NDArray[np.complex128] | float | None = None

    def solve(
        self, design: AntennaDesign, freq: float, state: object | None = None
    ) -> SolverResult:
        aperture = design.aperture_field(
            freq,
            samples_per_wavelength=self.samples_per_wavelength,
            margin=self.margin,
            illumination=self.illumination,
            state=state,
        )
        ff = far_field_from_aperture(aperture, pad_factor=self.pad_factor)
        return SolverResult(
            aperture=aperture,
            far_field=ff,
            metadata={"solver": self.name, "freq": freq},
        )
