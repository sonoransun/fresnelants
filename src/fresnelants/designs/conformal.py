"""Conformal Fresnel designs — cylindrical, spherical, freeform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.conformal import ConformalAperture
from ..core.geometry import ApertureGrid
from ..units import k0
from .base import AntennaDesign


@dataclass
class CylindricalFresnelLens(AntennaDesign):
    """Cylindrical Fresnel lens — zones unwrapped onto a cylinder.

    Models a metallic-or-dielectric Fresnel pattern wrapped on a circular
    cylinder (e.g. an automotive bumper). The feed sits at (0, 0, 0) inside;
    the aperture's tangential field is set so that the emitted wave focuses
    along the cylinder axis (broadside in the +y direction).
    """

    radius: float = 0.10
    height: float = 0.05
    design_freq: float = 77e9
    nu: int = 80
    nv: int = 60
    name: str = "CylindricalFresnelLens"
    focal_length: float = 0.0
    """Cylindrical lenses focus along their length, not at a point — this
    field is conventional and reflects the radial distance from axis."""

    def __post_init__(self) -> None:
        # Use radius as the conventional focal_length for downstream compat.
        if self.focal_length == 0.0:
            self.focal_length = self.radius

    @property
    def aperture_radius(self) -> float:
        # Half the cylinder height for "aperture extent" semantics.
        return float(self.height / 2.0)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        """Cylindrical lenses don't expose a flat-aperture transmittance.

        Use :meth:`conformal_aperture` with the conformal solver instead.
        """
        raise NotImplementedError(
            "CylindricalFresnelLens uses the conformal pipeline; call "
            "conformal_aperture() and pass to ConformalPOSolver."
        )

    def conformal_aperture(self, freq: float, state: object | None = None) -> ConformalAperture:
        """Build the cylindrical mesh with focusing tangential field."""
        ap = ConformalAperture.from_cylinder(
            radius=self.radius,
            height=self.height,
            nu=self.nu,
            nv=self.nv,
            freq=freq,
        )
        # Field on each facet: feed at origin radiating outward; PO target
        # is a plane wave traveling +y. For an outgoing plane wave with
        # e^{jωt − jk·r} convention, the aperture phase is −k·y.
        d_feed = np.linalg.norm(ap.points, axis=1)
        target_phase = -k0(freq) * ap.points[:, 1]
        cell_phase = -k0(freq) * d_feed + target_phase
        # Mask: keep only the front half of the cylinder (visible from +y).
        front = ap.points[:, 1] > 0
        Et = np.where(front, np.exp(1j * cell_phase), 0).astype(np.complex128)
        return ap.with_field(Et)


@dataclass
class SphericalFresnelLens(AntennaDesign):
    """Spherical-cap Fresnel lens (Luneburg-like).

    A hemispherical cap with an embedded Fresnel phase pattern; emits a
    near-omnidirectional pattern from a single feed.
    """

    radius: float = 0.05
    cap_angle_deg: float = 90.0
    design_freq: float = 28e9
    nu: int = 80
    nv: int = 40
    name: str = "SphericalFresnelLens"
    focal_length: float = 0.0

    def __post_init__(self) -> None:
        if self.focal_length == 0.0:
            self.focal_length = self.radius

    @property
    def aperture_radius(self) -> float:
        return float(self.radius)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        raise NotImplementedError(
            "SphericalFresnelLens uses the conformal pipeline; call conformal_aperture()."
        )

    def conformal_aperture(self, freq: float, state: object | None = None) -> ConformalAperture:
        ap = ConformalAperture.from_sphere(
            radius=self.radius,
            cap_angle=np.deg2rad(self.cap_angle_deg),
            nu=self.nu,
            nv=self.nv,
            freq=freq,
        )
        # Feed at sphere center; target is a plane wave traveling +z. Aperture
        # phase for an outgoing +z plane wave (e^{jωt − jk·r} convention) is −k·z.
        d_feed = np.linalg.norm(ap.points, axis=1)  # = radius (constant)
        target_phase = -k0(freq) * ap.points[:, 2]
        cell_phase = -k0(freq) * d_feed + target_phase
        Et = np.exp(1j * cell_phase).astype(np.complex128)
        return ap.with_field(Et)
