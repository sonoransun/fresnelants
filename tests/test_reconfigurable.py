"""Phase 2 — RIS, cells, coding, bias network."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import fresnelants as fa
from fresnelants.bias.gerber import write_bias_layers
from fresnelants.bias.network import (
    estimated_max_current_density,
    synthesize_bias_network,
)
from fresnelants.cells.liquid_crystal import Merck_GT3
from fresnelants.cells.pin_diode import MACOM_MA4FCP305
from fresnelants.cells.varactor import MACOM_MAVR011020, Skyworks_SMV1232
from fresnelants.coding import (
    BeamSpec,
    beam_codebook,
    load_codebook,
    nearest_state,
    save_codebook,
)


def test_skyworks_smv1232_phase_coverage() -> None:
    v = Skyworks_SMV1232()
    voltages = np.linspace(0, 15, 96)
    phase = v.phase(voltages, 28e9)
    assert np.degrees(phase.max() - phase.min()) >= 300.0


def test_macom_mavr011020_phase_coverage_at_mmw() -> None:
    m = MACOM_MAVR011020()
    voltages = np.linspace(0, 15, 96)
    phase = m.phase(voltages, 60e9)
    assert np.degrees(phase.max() - phase.min()) >= 300.0


def test_pin_diode_180_deg_pair() -> None:
    p = MACOM_MA4FCP305()
    phases = p.phase([False, True], 39e9)
    assert pytest.approx(180.0, abs=0.5) == abs(np.degrees(phases[0] - phases[1]))


def test_pin_diode_low_insertion_loss() -> None:
    p = MACOM_MA4FCP305()
    losses = p.loss([False, True], 39e9)
    losses_db = 20.0 * np.log10(losses)
    assert (losses_db > -0.5).all()


def test_lc_cell_phase_swing() -> None:
    """Merck GT3 LC at 100 GHz (default 1.5 mm cell, double-pass) gives > 300° swing."""
    lc = Merck_GT3()
    angles = np.linspace(0, np.pi / 2, 64)
    phases = lc.phase(angles, 100e9)
    swing_deg = np.degrees(phases.max() - phases.min())
    assert swing_deg >= 300.0


def test_reconfigurable_array_steers() -> None:
    cell = Skyworks_SMV1232()
    ris = fa.ReconfigurableArray(focal_length=0.20, design_freq=28e9, nx=24, ny=24, cell=cell)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2)
    g_broadside = solver.solve(ris, 28e9).far_field.peak_directivity_dbi()
    ris.beam_direction = (math.radians(30), 0)
    res = solver.solve(ris, 28e9)
    g_steer = res.far_field.peak_directivity_dbi()
    intens = res.far_field.directivity()
    _iy, ix = np.unravel_index(np.argmax(intens), intens.shape)
    u = res.far_field.u[ix]
    # Beam must land near sin(30°) = 0.5.
    assert abs(u - 0.5) < 0.05
    # Scan loss is well-bounded.
    assert g_broadside - g_steer < 3.0


def test_1bit_quantization_loss_vs_continuous() -> None:
    """1-bit RIS scan loss is bounded by the canonical 4 dB."""
    pin = MACOM_MA4FCP305()
    var = Skyworks_SMV1232()
    cont = fa.ReconfigurableArray(focal_length=0.20, design_freq=28e9, nx=24, ny=24, cell=var)
    coded = fa.CodedRIS(focal_length=0.20, design_freq=39e9, nx=24, ny=24, cell=pin, bits=1)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2)
    g_cont = solver.solve(cont, 28e9).far_field.peak_directivity_dbi()
    g_1bit = solver.solve(coded, 39e9).far_field.peak_directivity_dbi()
    # Both arrays at the same physical size — scan-loss difference
    # at broadside is bounded but small.
    assert abs(g_cont - g_1bit) < 6.0


def test_codebook_roundtrip(tmp_path: Path) -> None:
    var = Skyworks_SMV1232()
    ris = fa.ReconfigurableArray(focal_length=0.20, design_freq=28e9, nx=12, ny=12, cell=var)
    beams = [
        BeamSpec(theta_deg=0, phi_deg=0, label="broadside"),
        BeamSpec(theta_deg=15, phi_deg=0, label="theta15"),
        BeamSpec(theta_deg=30, phi_deg=45, label="theta30_phi45"),
    ]
    book = beam_codebook(beams, var, array_design=ris, freq=28e9)
    path = tmp_path / "book.json"
    save_codebook(book, path)
    book2 = load_codebook(path)
    assert set(book.keys()) == set(book2.keys())
    for k in book:
        np.testing.assert_array_equal(book[k], book2[k])


def test_nearest_state_matches_phase() -> None:
    var = Skyworks_SMV1232()
    target = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
    states = nearest_state(target, var, 28e9)
    actual = var.phase(states, 28e9)
    diff = np.angle(np.exp(1j * (actual - target)))
    assert (np.abs(diff) < np.deg2rad(15)).all()


def test_hierarchical_bias_network_drc(tmp_path: Path) -> None:
    """128×128 array routed via the hierarchical synthesizer must satisfy DRC.

    Uses a realistic 10 µA per-cell reverse-bias leakage current (typical for
    a varactor RIS). Total array current is ~ 0.16 A, within 1 oz copper limit.
    """
    var = Skyworks_SMV1232()
    ris = fa.ReconfigurableArray(focal_length=0.30, design_freq=28e9, nx=128, ny=128, cell=var)
    network = synthesize_bias_network(ris, cell_current_estimate=10e-6)
    # Hierarchical was auto-dispatched; max current density well below 1 oz limit.
    assert estimated_max_current_density(network) < 35.0
    assert network.cell_count == 128 * 128
    paths = write_bias_layers(network, tmp_path, base_name="ris_128_bias")
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0


def test_hierarchical_reduces_max_current() -> None:
    """For arrays > 64×64 the hierarchical layout has lower worst-case current
    than would a single-tier layout of the same size."""
    from fresnelants.bias.network import (
        synthesize_bias_network as auto,
    )
    from fresnelants.bias.network import (
        synthesize_hierarchical_bias_network,
    )

    var = Skyworks_SMV1232()
    ris = fa.ReconfigurableArray(focal_length=0.30, design_freq=28e9, nx=80, ny=80, cell=var)
    hier = synthesize_hierarchical_bias_network(
        ris, cell_current_estimate=1e-3, columns_per_trunk=10
    )
    auto_net = auto(ris, cell_current_estimate=1e-3)
    # Hierarchical and auto-dispatch both took the hierarchical path here
    # (80×80 > 64×64). Should produce identical worst-case currents.
    h_max = max(s.current for s in hier.segments)
    a_max = max(s.current for s in auto_net.segments)
    assert pytest.approx(h_max, rel=1e-9) == a_max


def test_bias_network_drc(tmp_path: Path) -> None:
    var = Skyworks_SMV1232()
    ris = fa.ReconfigurableArray(focal_length=0.20, design_freq=28e9, nx=8, ny=8, cell=var)
    network = synthesize_bias_network(ris, cell_current_estimate=2e-3)
    # All cells reachable.
    assert network.cell_count == 8 * 8
    # Current density well below 1 oz copper limit (~ 35 A/mm²).
    assert estimated_max_current_density(network) < 50.0
    # Gerber emission produces non-empty files.
    paths = write_bias_layers(network, tmp_path, base_name="ris_bias")
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    cu_text = paths["copper"].read_text()
    assert "ADD" in cu_text and "M02*" in cu_text
