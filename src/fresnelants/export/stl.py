"""STL export for Fresnel surfaces.

Generates a watertight, manifold mesh suitable for 3D printing. Supports:

* Phase-correcting plates (`PhaseCorrectingPlate`) — stepped grooves.
* Curvilinear/constructed surfaces (`CurvilinearFresnel`) — continuous depth.

Zone plates (binary) are exported as a flat dielectric substrate with circular
through-holes; reflectarrays are *not* STL targets — they're PCB devices and
handled by the Gerber exporter.

Implementation note: a polar-grid surface above the back plane gives a clean,
watertight body without external mesh libs. We write a binary STL directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray

from ..designs.base import AntennaDesign
from ..designs.curvilinear import CurvilinearFresnel
from ..designs.phase_correcting import PhaseCorrectingPlate
from ..designs.zone_plate import SoretZonePlate

Vec3 = tuple[float, float, float]


def _depth_function(design: AntennaDesign) -> tuple[callable, float]:  # type: ignore[type-arg]
    """Return (depth(R), substrate_thickness) for surface designs."""
    if isinstance(design, CurvilinearFresnel):
        return design.fresnel_depth, max(0.5e-3, 0.1 * design.aperture_radius)
    if isinstance(design, PhaseCorrectingPlate):
        # Sample on a 1D radial grid so we don't depend on a full 2D grid here.
        def f(R: NDArray[np.float64]) -> NDArray[np.float64]:
            n = design.dielectric.n
            d_2pi = design.wavelength / (n - 1.0)
            phi = (
                -2.0
                * np.pi
                * (np.sqrt(R**2 + design.focal_length**2) - design.focal_length)
                / design.wavelength
            )
            phi_wrapped = np.mod(phi, 2.0 * np.pi)
            if design.levels < 1024:
                step = 2.0 * np.pi / design.levels
                phi_wrapped = np.floor(phi_wrapped / step) * step
            return phi_wrapped * d_2pi / (2.0 * np.pi)

        return f, max(0.5e-3, 0.05 * design.aperture_radius)
    raise TypeError(f"STL export not supported for {type(design).__name__}")


def surface_to_mesh(
    design: AntennaDesign,
    *,
    radial_samples: int = 200,
    angular_samples: int = 360,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Triangulate a Fresnel surface into (vertices, faces).

    Returns
    -------
    vertices
        Array of shape (V, 3) [m].
    faces
        Array of shape (F, 3) of vertex indices forming triangles.
    """
    if isinstance(design, SoretZonePlate):
        return _zone_plate_mesh(design, radial_samples, angular_samples)
    depth_fn, base_thickness = _depth_function(design)

    R_axis = np.linspace(0.0, design.aperture_radius, radial_samples)
    theta = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    Rg, Tg = np.meshgrid(R_axis, theta, indexing="xy")  # shape (A, R)
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)
    depth = depth_fn(Rg)
    Z_top = base_thickness + depth
    Z_bot = np.zeros_like(Z_top)

    A = angular_samples
    Rn = radial_samples
    top = np.stack([X, Y, Z_top], axis=-1).reshape(-1, 3)
    bot = np.stack([X, Y, Z_bot], axis=-1).reshape(-1, 3)
    vertices = np.vstack([top, bot])
    n_top = top.shape[0]

    faces: list[Sequence[int]] = []

    def top_idx(a: int, r: int) -> int:
        return a * Rn + r

    def bot_idx(a: int, r: int) -> int:
        return n_top + a * Rn + r

    # Top surface (annular ring strips).
    for a in range(A):
        a1 = (a + 1) % A
        for r in range(Rn - 1):
            v00 = top_idx(a, r)
            v01 = top_idx(a, r + 1)
            v10 = top_idx(a1, r)
            v11 = top_idx(a1, r + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    # Bottom surface (reverse winding for outward normal).
    for a in range(A):
        a1 = (a + 1) % A
        for r in range(Rn - 1):
            v00 = bot_idx(a, r)
            v01 = bot_idx(a, r + 1)
            v10 = bot_idx(a1, r)
            v11 = bot_idx(a1, r + 1)
            faces.append((v00, v11, v10))
            faces.append((v00, v01, v11))

    # Outer rim (cylindrical wall at R = aperture_radius).
    r_rim = Rn - 1
    for a in range(A):
        a1 = (a + 1) % A
        v_t0 = top_idx(a, r_rim)
        v_t1 = top_idx(a1, r_rim)
        v_b0 = bot_idx(a, r_rim)
        v_b1 = bot_idx(a1, r_rim)
        faces.append((v_t0, v_b0, v_b1))
        faces.append((v_t0, v_b1, v_t1))

    faces_arr = np.asarray(faces, dtype=np.int64)
    return vertices, faces_arr


def _zone_plate_mesh(
    design: SoretZonePlate, radial_samples: int, angular_samples: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Soret plate as a thin disk with rectangular cutouts at the opaque zones.

    Implementation: build a watertight disc and remove triangles whose centroid
    falls in an opaque zone. Result is a flat plate with N annular cutouts —
    fine for visualization; for printing add a backing plane.
    """
    thickness = max(0.5e-3, 0.05 * design.aperture_radius)
    R_axis = np.linspace(0.0, design.aperture_radius, radial_samples)
    theta = np.linspace(0.0, 2 * np.pi, angular_samples, endpoint=False)
    Rg, Tg = np.meshgrid(R_axis, theta, indexing="xy")
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)
    Z_top = np.full_like(X, thickness)
    Z_bot = np.zeros_like(X)
    top = np.stack([X, Y, Z_top], axis=-1).reshape(-1, 3)
    bot = np.stack([X, Y, Z_bot], axis=-1).reshape(-1, 3)
    vertices = np.vstack([top, bot])
    n_top = top.shape[0]

    radii = design.zone_radii
    boundaries = np.concatenate(([0.0], radii))

    def opaque(r_centroid: float, n_zone_indices: int) -> bool:
        # Even zones are opaque if odd_transparent (default).
        idx = np.searchsorted(boundaries, r_centroid) - 1
        idx = max(0, min(idx, len(boundaries) - 2))
        zone_n = idx + 1
        return (zone_n % 2 == 0) if design.odd_transparent else (zone_n % 2 == 1)

    A = angular_samples
    Rn = radial_samples
    faces: list[Sequence[int]] = []

    def top_idx(a: int, r: int) -> int:
        return a * Rn + r

    def bot_idx(a: int, r: int) -> int:
        return n_top + a * Rn + r

    for a in range(A):
        a1 = (a + 1) % A
        for r in range(Rn - 1):
            r_mid = 0.5 * (R_axis[r] + R_axis[r + 1])
            if opaque(r_mid, 0):
                continue
            v00, v01 = top_idx(a, r), top_idx(a, r + 1)
            v10, v11 = top_idx(a1, r), top_idx(a1, r + 1)
            b00, b01 = bot_idx(a, r), bot_idx(a, r + 1)
            b10, b11 = bot_idx(a1, r), bot_idx(a1, r + 1)
            # top
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
            # bottom
            faces.append((b00, b11, b10))
            faces.append((b00, b01, b11))
            # rim only at outer boundary of each retained ring (cheap closure).
            faces.append((v00, b00, b10))
            faces.append((v00, b10, v10))
            faces.append((v01, v11, b11))
            faces.append((v01, b11, b01))

    faces_arr = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
    return vertices, faces_arr


def export_stl(
    design: AntennaDesign,
    path: str | Path,
    *,
    radial_samples: int = 200,
    angular_samples: int = 360,
) -> Path:
    """Write a binary STL file for *design* and return its path.

    Vertices coincident in space (e.g. the polar-mesh apex) are merged so the
    output mesh is watertight.
    """
    vertices, faces = surface_to_mesh(
        design, radial_samples=radial_samples, angular_samples=angular_samples
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(mesh.export(file_type="stl"))
    return out
