"""Incident wavefronts illuminating a Fresnel aperture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..units import k0
from .geometry import ApertureGrid


@dataclass(frozen=True, slots=True)
class PlaneWave:
    """Uniform plane wave with optional oblique incidence.

    Direction is given by polar angle θ from +z and azimuth φ. With θ=0 the
    wave is normally incident along +z (E in the aperture plane).
    """

    theta: float = 0.0
    phi: float = 0.0
    amplitude: complex = 1.0 + 0.0j

    def field_on(self, grid: ApertureGrid, freq: float) -> NDArray[np.complex128]:
        kx = k0(freq) * np.sin(self.theta) * np.cos(self.phi)
        ky = k0(freq) * np.sin(self.theta) * np.sin(self.phi)
        return self.amplitude * np.exp(1j * (kx * grid.X + ky * grid.Y))


@dataclass(frozen=True, slots=True)
class SphericalWave:
    """Spherical wave from a point feed at (x0, y0, z0).

    z0 should be negative (feed behind the aperture) or positive (focus point).
    The 1/r geometric attenuation is included.
    """

    x0: float = 0.0
    y0: float = 0.0
    z0: float = -1.0
    amplitude: complex = 1.0 + 0.0j

    def field_on(self, grid: ApertureGrid, freq: float) -> NDArray[np.complex128]:
        r = np.sqrt((grid.X - self.x0) ** 2 + (grid.Y - self.y0) ** 2 + self.z0**2)
        return self.amplitude * np.exp(-1j * k0(freq) * r) / r


@dataclass(frozen=True, slots=True)
class CosineFeed:
    """Cosine-q feed pattern (broadside feed model).

    Common reflectarray illumination model: amplitude ∝ cos^q(θ_feed)/r where
    θ_feed is the angle from the feed boresight to the aperture sample.
    """

    x0: float = 0.0
    y0: float = 0.0
    z0: float = -1.0
    q: float = 6.0
    amplitude: complex = 1.0 + 0.0j

    def field_on(self, grid: ApertureGrid, freq: float) -> NDArray[np.complex128]:
        dx = grid.X - self.x0
        dy = grid.Y - self.y0
        # The feed boresight points from the feed toward the aperture plane.
        # Its inner product with the feed-to-aperture vector has magnitude |z0|.
        r = np.sqrt(dx * dx + dy * dy + self.z0**2)
        cos_theta = np.abs(self.z0) / r
        return self.amplitude * (cos_theta**self.q) * np.exp(-1j * k0(freq) * r) / r
