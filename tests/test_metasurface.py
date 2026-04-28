"""Phase 3 — metasurface (PB lens, dual-pol shared aperture)."""

from __future__ import annotations

import numpy as np

import fresnelants as fa
from fresnelants.analysis.dualpol_metrics import (
    polarization_purity,
)
from fresnelants.analysis.farfield import far_field_from_jones_aperture
from fresnelants.cells.metasurface import AnisotropicEllipseCell, PancharatnamBerryCell


def test_pb_cell_geometric_phase_law() -> None:
    """Rotating cell by α produces 2α phase under LCP→RCP conversion."""
    cell = PancharatnamBerryCell(sense="LCP_to_RCP")
    alphas = np.linspace(0, np.pi, 32)
    phases = cell.phase(alphas, 60e9)
    np.testing.assert_allclose(phases, 2.0 * alphas, atol=1e-9)


def test_pb_cell_handedness_sign() -> None:
    cell_lcp = PancharatnamBerryCell(sense="LCP_to_RCP")
    cell_rcp = PancharatnamBerryCell(sense="RCP_to_LCP")
    assert cell_lcp.phase(0.5, 60e9) > 0
    assert cell_rcp.phase(0.5, 60e9) < 0


def test_anisotropic_jones_returns_2x2() -> None:
    cell = AnisotropicEllipseCell()
    rot = np.array([0.0, np.pi / 4, np.pi / 2])
    J = cell.jones_matrix(rot, 60e9)
    assert J.shape == (3, 2, 2)


def test_metasurface_lens_focuses_at_design_freq() -> None:
    lens = fa.MetasurfaceLens(focal_length=0.05, design_freq=60e9, aperture_radius_m=0.025)
    ap = lens.jones_aperture_field(60e9, samples_per_wavelength=4.0)
    ff = far_field_from_jones_aperture(ap, pad_factor=2)
    # Peak directivity is finite & reasonable for a 25 mm aperture at 60 GHz.
    g = ff.peak_directivity_dbi()
    assert g > 25.0


def test_metasurface_lens_polarization_conversion() -> None:
    """An LCP-fed PB lens produces predominantly RCP output."""
    lens = fa.MetasurfaceLens(focal_length=0.05, design_freq=60e9, aperture_radius_m=0.025)
    ap = lens.jones_aperture_field(60e9, samples_per_wavelength=4.0)
    ff = far_field_from_jones_aperture(ap, pad_factor=2)
    # Polarization purity in RCP should be high (PB lens converts L → R).
    pp_rcp = polarization_purity(ff, "rcp")
    assert pp_rcp > 0.5  # most of the radiated power is RCP


def test_dualpol_shared_aperture_v_band() -> None:
    """V-pol focusing at f_v gives a real beam with finite directivity."""
    sa = fa.DualPolSharedAperture(
        f_v=28e9, f_h=39e9, focal_v=0.10, focal_h=0.10, aperture_radius_m=0.05
    )
    ap = sa.jones_aperture_field(28e9, samples_per_wavelength=4.0)
    ff = far_field_from_jones_aperture(ap, pad_factor=2)
    assert ff.peak_directivity_dbi() > 20.0


def test_dualpol_shared_aperture_h_band() -> None:
    sa = fa.DualPolSharedAperture(
        f_v=28e9, f_h=39e9, focal_v=0.10, focal_h=0.10, aperture_radius_m=0.05
    )
    ap = sa.jones_aperture_field(39e9, samples_per_wavelength=4.0)
    ff = far_field_from_jones_aperture(ap, pad_factor=2)
    assert ff.peak_directivity_dbi() > 20.0
