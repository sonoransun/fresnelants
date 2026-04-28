"""Soret (binary amplitude) and Wood (phase-reversal) Fresnel zone plates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.geometry import ApertureGrid, fresnel_zone_radii
from ..units import freq_to_wavelength
from .base import AntennaDesign


@dataclass
class SoretZonePlate(AntennaDesign):
    """Classical opaque/transparent zone plate (alternating zones blocked).

    Even (n = 2, 4, …) zones are opaque; odd zones transmit with unit
    amplitude. Theoretical primary-focus efficiency η ≈ 1/π² ≈ 10.1 %.
    """

    focal_length: float
    design_freq: float
    num_zones: int = 12
    odd_transparent: bool = True
    name: str = "SoretZonePlate"

    @property
    def wavelength(self) -> float:
        return float(freq_to_wavelength(self.design_freq))

    @property
    def zone_radii(self) -> NDArray[np.float64]:
        return fresnel_zone_radii(self.num_zones, self.focal_length, self.wavelength)

    @property
    def aperture_radius(self) -> float:
        return float(self.zone_radii[-1])

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        # Use design wavelength for zone boundaries (chromatic by design).
        radii = self.zone_radii
        T = np.zeros(grid.shape, dtype=np.complex128)
        # zone n (1-indexed) is the annulus r_{n-1} < r ≤ r_n with r_0 = 0.
        prev = 0.0
        for n, r in enumerate(radii, start=1):
            mask = (prev < grid.R) & (r >= grid.R)
            transparent = (n % 2 == 1) if self.odd_transparent else (n % 2 == 0)
            if transparent:
                T[mask] = 1.0 + 0.0j
            prev = r
        return T


@dataclass
class WoodZonePlate(AntennaDesign):
    """Phase-reversal (Wood) zone plate.

    Even zones are inverted (×−1) instead of opaqued; theoretical
    primary-focus efficiency η ≈ 4/π² ≈ 40.5 % — a 6 dB improvement over Soret.
    """

    focal_length: float
    design_freq: float
    num_zones: int = 12
    name: str = "WoodZonePlate"

    @property
    def wavelength(self) -> float:
        return float(freq_to_wavelength(self.design_freq))

    @property
    def zone_radii(self) -> NDArray[np.float64]:
        return fresnel_zone_radii(self.num_zones, self.focal_length, self.wavelength)

    @property
    def aperture_radius(self) -> float:
        return float(self.zone_radii[-1])

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        radii = self.zone_radii
        T = np.zeros(grid.shape, dtype=np.complex128)
        prev = 0.0
        for n, r in enumerate(radii, start=1):
            mask = (prev < grid.R) & (r >= grid.R)
            T[mask] = 1.0 + 0.0j if (n % 2 == 1) else -1.0 + 0.0j
            prev = r
        return T
