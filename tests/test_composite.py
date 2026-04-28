"""Phase 1 — composite multi-stage designs."""

from __future__ import annotations

import pytest

import fresnelants as fa


@pytest.fixture
def cascade_solver() -> fa.CascadePOSolver:
    return fa.CascadePOSolver(samples_per_wavelength=4.0, pad_factor=2)


def test_composite_requires_layers() -> None:
    with pytest.raises(ValueError):
        fa.CompositeAntenna(layers=[], feed_distance=0.10)


def test_composite_layered_field(cascade_solver: fa.CascadePOSolver) -> None:
    """A composite of one phase plate must reproduce the standalone plate's gain."""
    plate = fa.PhaseCorrectingPlate(
        focal_length=0.10, design_freq=30e9, aperture_radius_m=0.05, levels=8
    )
    composite = fa.CompositeAntenna(
        layers=[fa.Layer(plate, 0.0)],
        feed_distance=0.10,
        design_freq=30e9,
    )
    g_solo = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2).solve(plate, 30e9)
    g_comp = cascade_solver.solve(composite, 30e9)
    assert (
        abs(g_solo.far_field.peak_directivity_dbi() - g_comp.far_field.peak_directivity_dbi()) < 0.5
    )


def test_bifocal_two_beams(cascade_solver: fa.CascadePOSolver) -> None:
    """A BifocalLens emits non-trivial gain at both design frequencies."""
    bif = fa.BifocalLens(
        focal_a=0.10, freq_a=24e9, focal_b=0.10, freq_b=77e9, aperture_radius_m=0.05, levels=8
    )
    g24 = cascade_solver.solve(bif, 24e9).far_field.peak_directivity_dbi()
    g77 = cascade_solver.solve(bif, 77e9).far_field.peak_directivity_dbi()
    assert g24 > 20.0
    assert g77 > 30.0


def test_achromatic_doublet_band_edges(cascade_solver: fa.CascadePOSolver) -> None:
    """Doublet must be physically realizable across its design band."""
    doublet = fa.AchromaticDoublet(
        f_low=28e9, f_high=32e9, aperture_radius_m=0.05, levels=8, feed_distance=0.10
    )
    gains = [
        cascade_solver.solve(doublet, f).far_field.peak_directivity_dbi()
        for f in (28e9, 30e9, 32e9)
    ]
    # All gains finite and > 10 dBi (real beam, not noise).
    assert all(g > 10.0 for g in gains)


def test_folded_reflectarray_emits(cascade_solver: fa.CascadePOSolver) -> None:
    folded = fa.FoldedReflectarray(
        reflectarray_focal=0.20,
        design_freq_ra=28e9,
        nx=16,
        ny=16,
        polarizer_gap=0.04,
    )
    g = cascade_solver.solve(folded, 28e9).far_field.peak_directivity_dbi()
    # Gain reduced vs unfolded because of the two-pass model, but still > 15 dBi.
    assert g > 15.0


def test_composite_cli_compatible() -> None:
    """A composite refuses transmittance() (uses aperture_field directly)."""
    plate = fa.PhaseCorrectingPlate(focal_length=0.10, design_freq=30e9, aperture_radius_m=0.05)
    composite = fa.CompositeAntenna(layers=[fa.Layer(plate, 0.0)], feed_distance=0.10)
    from fresnelants.core.geometry import make_aperture_grid

    grid = make_aperture_grid(0.10, 4.0, 0.01)
    with pytest.raises(NotImplementedError):
        composite.transmittance(grid, 30e9)


def test_composite_backward_propagation_changes_output() -> None:
    """A non-zero reflection_coefficient adds a measurable multi-bounce contribution."""
    import numpy as np

    plate = fa.PhaseCorrectingPlate(focal_length=0.10, design_freq=30e9, aperture_radius_m=0.05)
    composite_no_bounce = fa.CompositeAntenna(
        layers=[fa.Layer(plate, 0.05, reflection_coefficient=0.5)],
        feed_distance=0.10,
        design_freq=30e9,
    )
    ap_forward = composite_no_bounce.aperture_field(30e9, samples_per_wavelength=4.0, bounces=1)
    ap_3bounce = composite_no_bounce.aperture_field(30e9, samples_per_wavelength=4.0, bounces=3)
    # Multi-bounce should produce a different field magnitude (non-zero contribution).
    assert not np.allclose(np.abs(ap_forward.Ez), np.abs(ap_3bounce.Ez), rtol=1e-6)


def test_composite_zero_reflection_unchanged_by_bounces() -> None:
    """Layers with reflection_coefficient=0 must be invariant to `bounces`."""
    import numpy as np

    plate = fa.PhaseCorrectingPlate(focal_length=0.10, design_freq=30e9, aperture_radius_m=0.05)
    composite = fa.CompositeAntenna(
        layers=[fa.Layer(plate, 0.05)], feed_distance=0.10, design_freq=30e9
    )
    a = composite.aperture_field(30e9, samples_per_wavelength=4.0, bounces=1)
    b = composite.aperture_field(30e9, samples_per_wavelength=4.0, bounces=3)
    np.testing.assert_allclose(a.Ez, b.Ez, atol=1e-12)
