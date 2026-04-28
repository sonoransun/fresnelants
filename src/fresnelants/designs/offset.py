"""Offset (off-axis) Fresnel zone plate.

Zones are designed for a focal point at (F·tan α, 0, F) instead of (0, 0, F).
The boundaries are loci of constant path-length difference of n·λ/2 between
the source (assumed plane-wave normally incident) and the offset focal point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.geometry import ApertureGrid
from ..core.wavefront import SphericalWave
from ..units import freq_to_wavelength
from .base import AntennaDesign


@dataclass
class OffsetZonePlate(AntennaDesign):
    """Wood-type phase-reversal plate with the focus offset by *tilt_angle*."""

    focal_length: float
    design_freq: float
    tilt_angle: float = np.deg2rad(20.0)
    aperture_radius_m: float = 0.10
    name: str = "OffsetZonePlate"

    @property
    def wavelength(self) -> float:
        return float(freq_to_wavelength(self.design_freq))

    @property
    def aperture_radius(self) -> float:
        return float(self.aperture_radius_m)

    def _path_difference(self, grid: ApertureGrid) -> NDArray[np.float64]:
        """Phase delay [radians] needed at each aperture point.

        The offset focal point is at (F tan α, 0, F). Path length from the
        aperture point (x, y, 0) is √((x − F tan α)² + y² + F²). With a
        normally incident plane wave the on-aperture phase is uniform, so the
        path difference Δ = √(...) − F·secα is what the plate must compensate.
        """
        fx = self.focal_length * np.tan(self.tilt_angle)
        d = np.sqrt((grid.X - fx) ** 2 + grid.Y**2 + self.focal_length**2)
        return d - self.focal_length / np.cos(self.tilt_angle)

    def default_illumination(self, grid: ApertureGrid, freq: float) -> NDArray[np.complex128]:
        """Spherical-wave feed at the offset focal point behind the plate."""
        feed = SphericalWave(
            x0=self.focal_length * np.tan(self.tilt_angle),
            y0=0.0,
            z0=-self.focal_length,
        )
        return feed.field_on(grid, freq)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        delta = self._path_difference(grid)
        # Bin the path-difference into half-wavelength zones; alternate signs.
        zone_index = np.floor(2.0 * delta / self.wavelength).astype(np.int64)
        T = np.where((zone_index % 2) == 0, 1.0 + 0.0j, -1.0 + 0.0j)
        # Mask outside the physical aperture.
        T[self.aperture_radius < grid.R] = 0.0
        return T.astype(np.complex128)
