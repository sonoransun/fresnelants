"""Phase 0 — JonesApertureField backwards-compat and dual-pol metrics."""

from __future__ import annotations

import numpy as np

from fresnelants.analysis.aperture import ApertureField
from fresnelants.analysis.dualpol_metrics import (
    axial_ratio_db,
    cross_polarization_db,
    polarization_purity,
)
from fresnelants.analysis.farfield import (
    far_field_from_aperture,
    far_field_from_jones_aperture,
)
from fresnelants.core.geometry import make_aperture_grid


def _uniform_aperture(R: float, freq: float) -> ApertureField:
    lam = 3e8 / freq
    grid = make_aperture_grid(2 * R * 1.2, 6.0, lam)
    Ez = np.where(grid.R <= R, 1.0 + 0j, 0 + 0j)
    return ApertureField(grid=grid, Ez=Ez, freq=freq)


def test_phase0_dualpol_scalar_matches_jones() -> None:
    """An x-polarized JonesApertureField must reproduce the scalar far-field."""
    ap = _uniform_aperture(0.10, 28e9)
    ff_scalar = far_field_from_aperture(ap, pad_factor=4)
    ff_jones = far_field_from_jones_aperture(ap.to_jones("x"), pad_factor=4)
    # Scalar Ez maps to Ex; magnitudes should match.
    np.testing.assert_allclose(np.abs(ff_jones.Ex), np.abs(ff_scalar.E), rtol=1e-9, atol=1e-9)
    # Total directivity should match within 0.05 dB.
    assert abs(ff_jones.peak_directivity_dbi() - ff_scalar.peak_directivity_dbi()) < 0.05


def test_phase0_pure_linear_has_low_cross_pol() -> None:
    """A purely x-polarized aperture has near-zero cross-pol."""
    ap = _uniform_aperture(0.10, 28e9)
    ff = far_field_from_jones_aperture(ap.to_jones("x"), pad_factor=4)
    cross = cross_polarization_db(ff, polarization="x")
    cross_visible = cross[ff.visible_mask]
    # Numerical floor — Ey is constructed as zeros so cross-pol should be -inf.
    assert np.nanmax(cross_visible) < -100.0


def test_phase0_circular_polarization_axial_ratio() -> None:
    """Equal-amplitude Ex and Ey with 90° phase = circular pol → AR ≈ 0 dB at peak."""
    R, freq = 0.10, 28e9
    lam = 3e8 / freq
    grid = make_aperture_grid(2 * R * 1.2, 6.0, lam)
    mask = grid.R <= R
    Ex = np.where(mask, 1.0 + 0j, 0)
    Ey = np.where(mask, 1j, 0)  # 90° lag → circular polarization
    from fresnelants.analysis.aperture import JonesApertureField

    ap = JonesApertureField(grid=grid, Ex=Ex, Ey=Ey, freq=freq)
    ff = far_field_from_jones_aperture(ap, pad_factor=4)
    ar = axial_ratio_db(ff)
    iy, ix = np.unravel_index(np.argmax(ff.total_intensity()), ff.total_intensity().shape)
    assert ar[iy, ix] < 1.0  # near-circular at the main beam


def test_phase0_polarization_purity() -> None:
    ap = _uniform_aperture(0.10, 28e9)
    ff = far_field_from_jones_aperture(ap.to_jones("x"), pad_factor=4)
    pp = polarization_purity(ff, "x")
    assert pp > 0.999  # ~all power in x


def test_phase0_state_kwarg_accepted_by_static_designs() -> None:
    """Static designs must silently ignore the new state kwarg."""
    import fresnelants as fa

    plate = fa.WoodZonePlate(focal_length=1.0, design_freq=10e9, num_zones=8)
    grid = make_aperture_grid(2 * plate.aperture_radius * 1.05, 4.0, 0.03)
    T1 = plate.transmittance(grid, 10e9)
    T2 = plate.transmittance(grid, 10e9, state={"voltage": 3.3})
    np.testing.assert_array_equal(T1, T2)


def test_phase0_solver_state_kwarg() -> None:
    """PhysicalOpticsSolver must thread state into design.aperture_field."""
    import fresnelants as fa

    plate = fa.WoodZonePlate(focal_length=1.0, design_freq=10e9, num_zones=8)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2)
    res_a = solver.solve(plate, 10e9)
    res_b = solver.solve(plate, 10e9, state="ignored")
    assert (
        abs(res_a.far_field.peak_directivity_dbi() - res_b.far_field.peak_directivity_dbi()) < 1e-6
    )
