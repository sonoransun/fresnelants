"""3D curvilinear / constructed Fresnel surface.

Models a Fresnel surface where the *physical* depth profile of a dielectric
(or the *normal* of a metallic reflector) follows a continuous parametric
function. Supports two construction styles:

* ``"hyperbolic"`` — a hyperbolic groove profile (the locally aspherical lens
  Fresnelization), wrapped to discrete groove heights modulo λ/(n−1).
* ``"axicon"`` — conical-ring grooves producing a Bessel-like focal line.
* ``"freeform"`` — accepts a user-supplied callable f(R) → depth [m].

The 3D mesh used for STL export is generated from the depth profile with
zone discontinuities preserved (vertical groove walls).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..core.geometry import ApertureGrid
from ..core.materials import HDPE, Dielectric
from ..units import freq_to_wavelength
from .base import AntennaDesign

ProfileKind = Literal["hyperbolic", "axicon", "freeform"]


@dataclass
class CurvilinearFresnel(AntennaDesign):
    """3D curvilinear/constructed Fresnel surface."""

    focal_length: float
    design_freq: float
    aperture_radius_m: float = 0.10
    profile: ProfileKind = "hyperbolic"
    dielectric: Dielectric = field(default=HDPE)
    axicon_angle: float = np.deg2rad(5.0)
    freeform: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
    name: str = "CurvilinearFresnel"

    @property
    def wavelength(self) -> float:
        return float(freq_to_wavelength(self.design_freq))

    @property
    def aperture_radius(self) -> float:
        return float(self.aperture_radius_m)

    def _continuous_depth(self, R: NDArray[np.float64]) -> NDArray[np.float64]:
        """Continuous (un-Fresnelized) dielectric thickness [m].

        Hyperbolic singlet: t(r) = (√(F² + R_max²) − √(F² + r²)) / (n − 1).
        Center-thickest, edge-zero, by Fermat's principle for a planar
        wavefront emerging on the front side under spherical illumination
        from the focal point.
        """
        n = self.dielectric.n
        F = self.focal_length
        Rmax = self.aperture_radius
        if self.profile == "hyperbolic":
            return (np.sqrt(F**2 + Rmax**2) - np.sqrt(F**2 + R**2)) / (n - 1.0)
        if self.profile == "axicon":
            # Conical lens: max thickness on axis, zero at the rim.
            return (Rmax - R) * np.tan(self.axicon_angle) / (n - 1.0)
        if self.profile == "freeform":
            if self.freeform is None:
                raise ValueError("profile='freeform' requires a `freeform` callable.")
            return np.asarray(self.freeform(R), dtype=np.float64)
        raise ValueError(f"Unknown profile {self.profile!r}")

    def fresnel_depth(self, R: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fresnelized (zone-folded) thickness profile [m]."""
        cont = self._continuous_depth(R)
        d_2pi = self.wavelength / (self.dielectric.n - 1.0)
        return np.mod(cont, d_2pi)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        depth = self.fresnel_depth(grid.R)
        lam = float(freq_to_wavelength(freq))
        # Phase delay relative to free-space passage: (n − 1) · depth · 2π/λ.
        phase = -(self.dielectric.n - 1.0) * depth * 2.0 * np.pi / lam
        T = np.exp(1j * phase).astype(np.complex128)
        T[self.aperture_radius < grid.R] = 0.0
        # Dielectric loss.
        if self.dielectric.tan_delta > 0.0:
            alpha = 2.0 * np.pi / lam * self.dielectric.n * 0.5 * self.dielectric.tan_delta
            T *= np.exp(-alpha * depth)
        return T
