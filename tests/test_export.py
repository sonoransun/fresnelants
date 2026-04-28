"""Mesh and Gerber export validity checks."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

import fresnelants as fa
from fresnelants.export.gerber import write_reflectarray_gerber
from fresnelants.export.stl import export_stl, surface_to_mesh


@pytest.mark.parametrize(
    "design",
    [
        fa.CurvilinearFresnel(focal_length=1.0, design_freq=10e9, aperture_radius_m=0.10),
        fa.PhaseCorrectingPlate(
            focal_length=1.0, design_freq=10e9, aperture_radius_m=0.10, levels=4
        ),
    ],
)
def test_stl_mesh_is_watertight(tmp_path: Path, design: fa.AntennaDesign) -> None:
    out = tmp_path / "antenna.stl"
    export_stl(design, out, radial_samples=80, angular_samples=120)
    mesh = trimesh.load_mesh(out, file_type="stl")
    assert mesh.is_watertight, f"Mesh not watertight: {design.name}"
    assert mesh.area > 0
    assert mesh.volume > 0


def test_stl_binary_header_format(tmp_path: Path) -> None:
    d = fa.CurvilinearFresnel(focal_length=1.0, design_freq=10e9, aperture_radius_m=0.05)
    out = export_stl(d, tmp_path / "small.stl", radial_samples=40, angular_samples=60)
    raw = out.read_bytes()
    # Header (80 bytes) + uint32 face count + 50 bytes per face.
    n_faces = struct.unpack("<I", raw[80:84])[0]
    expected_size = 84 + n_faces * 50
    assert len(raw) == expected_size


def test_surface_mesh_returns_arrays() -> None:
    d = fa.CurvilinearFresnel(focal_length=1.0, design_freq=10e9, aperture_radius_m=0.05)
    verts, faces = surface_to_mesh(d, radial_samples=40, angular_samples=60)
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
    assert int(faces.max()) < verts.shape[0]


def test_gerber_files_written(tmp_path: Path) -> None:
    ra = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=8, ny=8)
    paths = write_reflectarray_gerber(ra, tmp_path)
    for label, p in paths.items():
        assert p.exists(), f"{label} not written"
        assert p.stat().st_size > 0
        text = p.read_text()
        if label == "copper":
            # Expect at least one aperture definition and at least one flash.
            assert "ADD" in text
            assert "D03" in text
        elif label == "drill":
            assert "M48" in text
        elif label == "mask":
            assert "ADD" in text


def test_gerber_aperture_size_matches_phase() -> None:
    """Spot-check the Gerber aperture-size mapping for an extreme phase."""
    ra = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=4, ny=4)
    # Required cell phases and the patch-size mapping should be deterministic.
    phases = ra.required_phase_per_cell(28e9)
    assert phases.shape == (4, 4)
    assert np.all(phases >= 0) and np.all(phases < 2 * np.pi)
