"""Conformal (curved-surface) aperture: triangulated mesh of (point, normal, area)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ConformalAperture:
    """Triangulated curved aperture with a complex tangential field per facet.

    Attributes
    ----------
    points : (N, 3) float
        Facet centroid positions [m].
    normals : (N, 3) float
        Outward unit normals at each facet.
    areas : (N,) float
        Facet areas [m²].
    Et : (N,) complex
        Tangential E-field magnitude at each facet (single-pol).
    freq : float
        Frequency [Hz].
    """

    points: NDArray[np.float64]
    normals: NDArray[np.float64]
    areas: NDArray[np.float64]
    Et: NDArray[np.complex128]
    freq: float

    def __post_init__(self) -> None:
        n = self.points.shape[0]
        if self.normals.shape != (n, 3):
            raise ValueError("normals must be (N, 3)")
        if self.areas.shape != (n,):
            raise ValueError("areas must be (N,)")
        if self.Et.shape != (n,):
            raise ValueError("Et must be (N,)")

    def power(self) -> float:
        return float(np.sum(np.abs(self.Et) ** 2 * self.areas))

    @classmethod
    def from_cylinder(
        cls,
        radius: float,
        height: float,
        nu: int = 80,
        nv: int = 60,
        freq: float = 28e9,
    ) -> ConformalAperture:
        """Right-circular-cylinder mesh with apex up the +y axis. Facets are
        the cell centers of the (φ, z) grid; normals point radially outward.
        Et is initialized to zero — the user fills it via a design's
        ``conformal_field()`` method.
        """
        phi = np.linspace(-np.pi / 2, np.pi / 2, nu)
        z = np.linspace(-height / 2, height / 2, nv)
        Phi, Z = np.meshgrid(phi, z, indexing="xy")
        X = radius * np.sin(Phi)
        Y = radius * np.cos(Phi)
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        normals = np.stack([np.sin(Phi).flatten(), np.cos(Phi).flatten(), np.zeros(Z.size)], axis=1)
        d_phi = (phi[1] - phi[0]) if nu > 1 else 1.0
        d_z = (z[1] - z[0]) if nv > 1 else 1.0
        areas = np.full(points.shape[0], radius * d_phi * d_z)
        Et = np.zeros(points.shape[0], dtype=np.complex128)
        return cls(points=points, normals=normals, areas=areas, Et=Et, freq=freq)

    @classmethod
    def from_sphere(
        cls,
        radius: float,
        cap_angle: float = np.pi / 2,
        nu: int = 80,
        nv: int = 40,
        freq: float = 28e9,
    ) -> ConformalAperture:
        """Hemispherical / spherical-cap mesh up to *cap_angle* polar angle.

        Cap angle π/2 → upper hemisphere; smaller → narrower cap.
        """
        theta = np.linspace(0, cap_angle, nv)
        phi = np.linspace(0, 2 * np.pi, nu, endpoint=False)
        Th, Ph = np.meshgrid(theta, phi, indexing="xy")
        X = radius * np.sin(Th) * np.cos(Ph)
        Y = radius * np.sin(Th) * np.sin(Ph)
        Z = radius * np.cos(Th)
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        # Outward normals are radial.
        norm_mag = np.linalg.norm(points, axis=1, keepdims=True) + 1e-30
        normals = points / norm_mag
        d_theta = (theta[1] - theta[0]) if nv > 1 else 1.0
        d_phi = (phi[1] - phi[0]) if nu > 1 else 1.0
        areas = (radius**2 * np.sin(Th) * d_theta * d_phi).flatten()
        Et = np.zeros(points.shape[0], dtype=np.complex128)
        return cls(points=points, normals=normals, areas=areas, Et=Et, freq=freq)

    @classmethod
    def from_cone(
        cls,
        half_angle: float,
        height: float,
        nu: int = 80,
        nv: int = 60,
        freq: float = 28e9,
    ) -> ConformalAperture:
        """Right-circular-cone mesh with apex at origin, opening toward +z.

        Half-angle ``α`` is measured from the +z axis; base radius = ``h·tan α``;
        slant length = ``h/cos α``. Outward normals tilt down-and-out:
        ``n = (cos α · cos φ, cos α · sin φ, −sin α)``.

        The mesh skips the degenerate apex point (z = 0); the innermost
        annulus starts at z = h / nv.
        """
        if not 0.0 < half_angle < np.pi / 2:
            raise ValueError("half_angle must be in (0, π/2).")
        if height <= 0.0:
            raise ValueError("height must be positive.")
        z = np.linspace(height / nv, height, nv) if nv > 1 else np.array([height])
        phi = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
        Phi, Z = np.meshgrid(phi, z, indexing="xy")
        rho = Z * np.tan(half_angle)
        X = rho * np.cos(Phi)
        Y = rho * np.sin(Phi)
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        nx = np.cos(half_angle) * np.cos(Phi)
        ny = np.cos(half_angle) * np.sin(Phi)
        nz = -np.sin(half_angle) * np.ones_like(Phi)
        normals = np.stack([nx.flatten(), ny.flatten(), nz.flatten()], axis=1)
        d_z = (z[1] - z[0]) if nv > 1 else height
        d_phi = (phi[1] - phi[0]) if nu > 1 else 2.0 * np.pi
        # dA = z · tan α · sec α · dφ · dz
        areas = (Z * np.tan(half_angle) / np.cos(half_angle) * d_phi * d_z).flatten()
        Et = np.zeros(points.shape[0], dtype=np.complex128)
        return cls(points=points, normals=normals, areas=areas, Et=Et, freq=freq)

    def with_field(self, Et: NDArray[np.complex128]) -> ConformalAperture:
        return ConformalAperture(
            points=self.points,
            normals=self.normals,
            areas=self.areas,
            Et=Et,
            freq=self.freq,
        )
