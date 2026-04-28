"""N-level (or continuous) dielectric phase-correcting Fresnel plate.

The plate consists of stepped grooves in a dielectric of refractive index n,
each groove producing a phase delay 2π/N relative to the next. A perfectly
continuous (N → ∞) profile gives a Fresnel lens with theoretical efficiency
≈ 100 %; quarter-wave (N = 4) plates give ≈ 81 %, eighth-wave (N = 8) plates
≈ 95 %.

References: Hristov §6, Wiltse "Recent Developments in Fresnel Zone Plate
Antennas at Microwave/Millimeter Wave" *Proc. SPIE* 3464 (1998).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.geometry import ApertureGrid
from ..core.materials import HDPE, Dielectric
from ..units import freq_to_wavelength
from .base import AntennaDesign


@dataclass
class PhaseCorrectingPlate(AntennaDesign):
    """N-level dielectric phase-correcting plate."""

    focal_length: float
    design_freq: float
    aperture_radius_m: float = 0.10
    levels: int = 4  # 1 = Soret, 2 = Wood, 4 = quarter-wave, ∞ = continuous
    dielectric: Dielectric = field(default=HDPE)
    name: str = "PhaseCorrectingPlate"

    @property
    def wavelength(self) -> float:
        return float(freq_to_wavelength(self.design_freq))

    @property
    def aperture_radius(self) -> float:
        return float(self.aperture_radius_m)

    def _required_phase(self, grid: ApertureGrid, freq: float) -> NDArray[np.float64]:
        """Phase the plate must introduce to collimate a feed at (0, 0, −F).

        Spherical-wave illumination has phase −k·d(r) on the aperture; the
        plate adds +k·(d − F) so the emerging wavefront is uniform-phase
        (with a global delay of −k·F that's absorbed into the reference).
        """
        lam = float(freq_to_wavelength(freq))
        d = np.sqrt(grid.X**2 + grid.Y**2 + self.focal_length**2)
        return 2.0 * np.pi * (d - self.focal_length) / lam

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        phi = self._required_phase(grid, freq)
        # Wrap into [0, 2π).
        phi_wrapped = np.mod(phi, 2.0 * np.pi)
        if self.levels >= 1024:  # treat as continuous
            phi_q = phi_wrapped
        else:
            step = 2.0 * np.pi / self.levels
            phi_q = np.floor(phi_wrapped / step) * step
        T = np.exp(1j * phi_q).astype(np.complex128)
        T[self.aperture_radius < grid.R] = 0.0
        # Apply dielectric loss as an amplitude factor proportional to local
        # groove depth (depth = phi_q · λ / (2π · (n − 1))).
        if self.dielectric.tan_delta > 0.0:
            depth = phi_q * self.wavelength / (2.0 * np.pi * (self.dielectric.n - 1.0))
            alpha = (
                2.0 * np.pi / self.wavelength * self.dielectric.n * 0.5 * self.dielectric.tan_delta
            )
            T *= np.exp(-alpha * depth)
        return T

    def groove_depths(self, grid: ApertureGrid) -> NDArray[np.float64]:
        """Physical groove depth (m) at each aperture sample (for STL export)."""
        phi = self._required_phase(grid, self.design_freq)
        phi_wrapped = np.mod(phi, 2.0 * np.pi)
        if self.levels >= 1024:
            phi_q = phi_wrapped
        else:
            step = 2.0 * np.pi / self.levels
            phi_q = np.floor(phi_wrapped / step) * step
        depth = phi_q * self.wavelength / (2.0 * np.pi * (self.dielectric.n - 1.0))
        depth[self.aperture_radius < grid.R] = 0.0
        return depth
