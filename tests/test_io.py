"""IO: pydantic specs and YAML/JSON loaders."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import fresnelants as fa
from fresnelants.io.loaders import load_design
from fresnelants.io.specs import spec_from_dict


def test_spec_dispatch_roundtrip() -> None:
    designs = [
        {"kind": "soret", "focal_length": 1.0, "design_freq": 10e9, "num_zones": 8},
        {"kind": "wood", "focal_length": 1.0, "design_freq": 10e9, "num_zones": 8},
        {
            "kind": "offset",
            "focal_length": 1.0,
            "design_freq": 10e9,
            "tilt_angle_deg": 15.0,
            "aperture_radius": 0.20,
        },
        {
            "kind": "phase_correcting",
            "focal_length": 1.0,
            "design_freq": 10e9,
            "aperture_radius": 0.20,
            "levels": 4,
            "dielectric": "HDPE",
        },
        {
            "kind": "reflectarray",
            "focal_length": 0.20,
            "design_freq": 28e9,
            "nx": 16,
            "ny": 16,
            "beam_theta_deg": 10.0,
        },
        {
            "kind": "curvilinear",
            "focal_length": 1.0,
            "design_freq": 10e9,
            "aperture_radius": 0.20,
            "profile": "hyperbolic",
        },
    ]
    expected = {
        "soret": fa.SoretZonePlate,
        "wood": fa.WoodZonePlate,
        "offset": fa.OffsetZonePlate,
        "phase_correcting": fa.PhaseCorrectingPlate,
        "reflectarray": fa.Reflectarray,
        "curvilinear": fa.CurvilinearFresnel,
    }
    for spec in designs:
        d = spec_from_dict(spec)
        assert isinstance(d, expected[spec["kind"]])


def test_load_design_yaml(tmp_path: Path) -> None:
    spec = {"kind": "wood", "focal_length": 1.0, "design_freq": 10e9, "num_zones": 6}
    path = tmp_path / "design.yaml"
    path.write_text(yaml.safe_dump(spec))
    d = load_design(path)
    assert isinstance(d, fa.WoodZonePlate)
    assert d.num_zones == 6


def test_load_design_json(tmp_path: Path) -> None:
    spec = {
        "kind": "reflectarray",
        "focal_length": 0.20,
        "design_freq": 28e9,
        "nx": 8,
        "ny": 8,
    }
    path = tmp_path / "design.json"
    path.write_text(json.dumps(spec))
    d = load_design(path)
    assert isinstance(d, fa.Reflectarray)
    assert d.nx == 8
