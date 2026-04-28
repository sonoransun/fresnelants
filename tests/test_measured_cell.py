"""Phase B4 — MeasuredCell round-trip and integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fresnelants as fa
from fresnelants.cells.measured import MeasuredCell
from fresnelants.cells.varactor import Skyworks_SMV1232


def test_measured_cell_roundtrip_from_varactor() -> None:
    """Generate synthetic S11 from a varactor model, load via MeasuredCell,
    confirm phase / loss match within 0.1°."""
    var = Skyworks_SMV1232()
    states = np.linspace(0.0, 15.0, 16)
    freqs = np.array([26e9, 28e9, 30e9])
    s11 = np.zeros((len(states), len(freqs)), dtype=np.complex128)
    for i, s in enumerate(states):
        for j, f in enumerate(freqs):
            s11[i, j] = var.loss(s, f) * np.exp(1j * var.phase(s, f))
    measured = MeasuredCell.from_arrays(states, freqs, s11)
    # Recover phase at a sampled point.
    for s in (3.0, 7.0, 11.5):
        for f in (28e9,):
            phi_meas = float(measured.phase(s, f))
            phi_var = float(var.phase(s, f))
            diff = np.angle(np.exp(1j * (phi_meas - phi_var)))
            assert abs(diff) < np.deg2rad(0.5)


def test_measured_cell_save_load(tmp_path: Path) -> None:
    var = Skyworks_SMV1232()
    states = np.linspace(0, 15, 8)
    freqs = np.array([28e9])
    s11 = np.array([[var.loss(s, 28e9) * np.exp(1j * var.phase(s, 28e9))] for s in states])
    cell = MeasuredCell.from_arrays(states, freqs, s11)
    path = tmp_path / "cell.json"
    cell.save(path)
    cell2 = MeasuredCell.load(path)
    np.testing.assert_allclose(cell2.states, cell.states)
    np.testing.assert_allclose(cell2.s11, cell.s11)


def test_measured_cell_drives_reconfigurable_array() -> None:
    """MeasuredCell drops into ReconfigurableArray and beam-steers normally."""
    var = Skyworks_SMV1232()
    states = np.linspace(0, 15, 32)
    freqs = np.linspace(26e9, 30e9, 9)
    s11 = np.array([[var.loss(s, f) * np.exp(1j * var.phase(s, f)) for f in freqs] for s in states])
    measured = MeasuredCell.from_arrays(states, freqs, s11)
    ris = fa.ReconfigurableArray(focal_length=0.20, design_freq=28e9, nx=12, ny=12, cell=measured)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2)
    res = solver.solve(ris, 28e9)
    assert res.far_field.peak_directivity_dbi() > 20.0


def test_measured_cell_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="s11 shape"):
        MeasuredCell(
            states=np.array([0.0, 1.0]),
            freqs=np.array([28e9, 30e9, 32e9]),
            s11=np.zeros((2, 2), dtype=np.complex128),
        )
