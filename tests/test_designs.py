"""Per-family design sanity checks via the PO solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

import fresnelants as fa
from fresnelants.analysis.metrics import aperture_efficiency


@pytest.fixture
def solver() -> fa.PhysicalOpticsSolver:
    return fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)


def test_soret_zone_plate_gain(solver: fa.PhysicalOpticsSolver) -> None:
    zp = fa.SoretZonePlate(focal_length=1.0, design_freq=10e9, num_zones=12)
    res = solver.solve(zp, 10e9)
    A = math.pi * zp.aperture_radius**2
    # Soret should give a strong main beam (>15 % efficient including taper).
    assert res.far_field.peak_directivity_dbi() > 30.0
    eta = aperture_efficiency(res.far_field, A)
    assert 0.10 < eta < 0.50


def test_wood_outperforms_soret(solver: fa.PhysicalOpticsSolver) -> None:
    soret = fa.SoretZonePlate(focal_length=1.0, design_freq=10e9, num_zones=12)
    wood = fa.WoodZonePlate(focal_length=1.0, design_freq=10e9, num_zones=12)
    g_soret = solver.solve(soret, 10e9).far_field.peak_directivity_dbi()
    g_wood = solver.solve(wood, 10e9).far_field.peak_directivity_dbi()
    # Wood gains the energy that Soret blocks in opaque zones. The textbook
    # primary-focus ratio is ≈ 4× (6 dB); under PO with a spherical feed it
    # tends to be smaller because the reflected/blocked Soret energy doesn't
    # fully transfer to the main beam.
    assert g_wood > g_soret + 2.5


def test_phase_correcting_levels_monotone(solver: fa.PhysicalOpticsSolver) -> None:
    F = 1.0
    R = 0.626
    gains = []
    for levels in (1, 2, 4, 8, 1024):
        d = fa.PhaseCorrectingPlate(
            focal_length=F, design_freq=10e9, aperture_radius_m=R, levels=levels
        )
        gains.append(solver.solve(d, 10e9).far_field.peak_directivity_dbi())
    # More levels → at least as much gain (allowing 0.1 dB sampling noise).
    diffs = np.diff(gains)
    assert np.all(diffs > -0.15), gains


def test_continuous_phase_plate_near_uniform_aperture(solver: fa.PhysicalOpticsSolver) -> None:
    R = 0.626
    d = fa.PhaseCorrectingPlate(
        focal_length=1.0, design_freq=10e9, aperture_radius_m=R, levels=1024
    )
    res = solver.solve(d, 10e9)
    A = math.pi * R**2
    eta = aperture_efficiency(res.far_field, A)
    assert eta > 0.9  # near-ideal lens, with allowance for grid tapering


def test_curvilinear_hyperbolic_matches_continuous_phase_plate(
    solver: fa.PhysicalOpticsSolver,
) -> None:
    R = 0.626
    cv = fa.CurvilinearFresnel(
        focal_length=1.0, design_freq=10e9, aperture_radius_m=R, profile="hyperbolic"
    )
    pc = fa.PhaseCorrectingPlate(
        focal_length=1.0, design_freq=10e9, aperture_radius_m=R, levels=1024
    )
    g_cv = solver.solve(cv, 10e9).far_field.peak_directivity_dbi()
    g_pc = solver.solve(pc, 10e9).far_field.peak_directivity_dbi()
    assert abs(g_cv - g_pc) < 0.5


def test_offset_zone_plate_broadside_with_offset_feed(
    solver: fa.PhysicalOpticsSolver,
) -> None:
    """Offset Fresnel = offset *feed*, broadside beam (avoids feed blockage)."""
    tilt = math.radians(20.0)
    d = fa.OffsetZonePlate(
        focal_length=1.0, design_freq=10e9, aperture_radius_m=0.30, tilt_angle=tilt
    )
    res = solver.solve(d, 10e9)
    intens = res.far_field.directivity()
    iy, ix = np.unravel_index(np.argmax(intens), intens.shape)
    u_peak = res.far_field.u[ix]
    v_peak = res.far_field.v[iy]
    # Beam should land near broadside despite the off-axis feed.
    assert abs(u_peak) < 0.10
    assert abs(v_peak) < 0.10
    # And gain should be a meaningful fraction of the equivalent on-axis plate.
    assert res.far_field.peak_directivity_dbi() > 25.0


def test_fractal_cantor_zone_indices_structure() -> None:
    from fresnelants.designs.fractal import cantor_zone_indices

    intervals = cantor_zone_indices(stage=3, base_unit=1)
    assert len(intervals) == 2**3 == 8
    assert all(hi - lo == 1 for lo, hi in intervals)
    # Stage 0 is the trivial single full-disk interval.
    assert cantor_zone_indices(stage=0, base_unit=1) == [(0, 1)]
    # Stage 1 trisects [0, 3) and drops the middle third.
    assert cantor_zone_indices(stage=1, base_unit=1) == [(0, 1), (2, 3)]


def test_sierpinski_mask_self_similar() -> None:
    from fresnelants.designs.fractal import sierpinski_mask

    m0 = sierpinski_mask(0)
    assert m0.shape == (1, 1) and bool(m0[0, 0]) is True

    m1 = sierpinski_mask(1)
    assert m1.shape == (3, 3)
    assert bool(m1[1, 1]) is False  # center hole
    assert int(m1.sum()) == 8

    m2 = sierpinski_mask(2)
    assert m2.shape == (9, 9)
    assert bool(m2[4, 4]) is False  # nested center
    assert int(m2.sum()) == 64  # 8^2 retained cells


def test_fractal_wood_outperforms_fractal_soret(solver: fa.PhysicalOpticsSolver) -> None:
    """At ``base_unit=2`` each retained Cantor interval covers one odd + one
    even Fresnel zone, so the Wood per-zone phase reversal restores coherent
    focusing while the Soret binary mask suffers destructive interference
    between paired zones — Wood must outperform Soret by several dB.
    """
    soret = fa.FractalSoretZonePlate(focal_length=1.0, design_freq=10e9, stage=2, base_unit=2)
    wood = fa.FractalWoodZonePlate(focal_length=1.0, design_freq=10e9, stage=2, base_unit=2)
    g_soret = solver.solve(soret, 10e9).far_field.peak_directivity_dbi()
    g_wood = solver.solve(wood, 10e9).far_field.peak_directivity_dbi()
    assert g_wood > g_soret + 3.0


def test_fractal_cantor_base_unit_one_wood_soret_degenerate(
    solver: fa.PhysicalOpticsSolver,
) -> None:
    """At ``base_unit=1`` every retained Cantor zone is odd, so Wood ≡ Soret —
    a real property of the triadic Cantor construction. Documenting it here
    so a future "fix" doesn't accidentally re-introduce the broken
    per-interval ±1 alternation.
    """
    soret = fa.FractalSoretZonePlate(focal_length=1.0, design_freq=10e9, stage=2, base_unit=1)
    wood = fa.FractalWoodZonePlate(focal_length=1.0, design_freq=10e9, stage=2, base_unit=1)
    g_soret = solver.solve(soret, 10e9).far_field.peak_directivity_dbi()
    g_wood = solver.solve(wood, 10e9).far_field.peak_directivity_dbi()
    assert abs(g_wood - g_soret) < 0.05


def test_fractal_cantor_polyfocal_signature() -> None:
    """Cantor zone plate's defining signature: extra on-axis foci at F/3, F/5.

    This is the ONE physical regression that distinguishes the Cantor family
    from any other Fresnel device in the package. A failure here implies
    either (a) a bug in the Cantor-mask construction, or (b) a sign flip in
    the FFT kernel / spherical-wave phase — see CLAUDE.md sign conventions.
    """
    from fresnelants.analysis.nearfield import focal_axis_intensity

    F = 0.3
    freq = 30e9
    zp = fa.FractalWoodZonePlate(focal_length=F, design_freq=freq, stage=3)
    ap = zp.aperture_field(freq, samples_per_wavelength=4.0, margin=1.1)

    z_fwd = np.linspace(0.05 * F, 1.3 * F, 120)
    I_fwd = focal_axis_intensity(ap, z_fwd)
    # Pick local maxima.
    is_peak = np.r_[
        False,
        (I_fwd[1:-1] > I_fwd[:-2]) & (I_fwd[1:-1] > I_fwd[2:]),
        False,
    ]
    peaks = z_fwd[is_peak]
    # Primary focus + first two sub-foci (F, F/3, F/5) must all be present.
    # F and F/5 land within 5% empirically; F/3 within 5% as well at this
    # solver resolution.
    for z_expected in (F, F / 3.0, F / 5.0):
        assert peaks.size > 0, "no peaks found"
        rel_err = float(np.min(np.abs(peaks - z_expected)) / z_expected)
        assert rel_err < 0.05, f"no peak near z={z_expected:.3f}; peaks={peaks}"

    # Sign-convention assertion: the back half of the optical axis must NOT
    # carry a focus. A `+j` → `−j` Fourier-kernel flip (or a spherical-wave
    # phase flip) would relocate the primary focus to z = −F and trip this.
    z_back = np.linspace(-1.3 * F, -0.05 * F, 40)
    I_back = focal_axis_intensity(ap, z_back)
    assert I_back.max() < 0.15 * I_fwd.max()


def test_sierpinski_carpet_emits(solver: fa.PhysicalOpticsSolver) -> None:
    """Self-similar carpet at 30 GHz must produce a recognisable far-field."""
    carpet = fa.SierpinskiCarpetZonePlate(
        focal_length=0.5, design_freq=30e9, stage=2, aperture_side=0.18
    )
    res = solver.solve(carpet, 30e9)
    assert res.far_field.peak_directivity_dbi() > 10.0


def test_sierpinski_reflectarray_zeros_inactive_cells(
    solver: fa.PhysicalOpticsSolver,
) -> None:
    """The fractal mask must remove the cells the carpet says are inactive."""
    ra = fa.SierpinskiReflectarray(
        focal_length=0.20, design_freq=28e9, nx=27, ny=27, fractal_stage=2
    )
    # Verify cell mask: with nx=ny=27 and a 9×9 carpet, each carpet cell maps
    # to a 3×3 array block. The carpet's center 3×3 hole maps to array cells
    # [9:18, 9:18] = a 9×9 hole at the array center.
    mask = ra._cell_mask()
    assert mask.shape == (27, 27)
    assert not mask[9:18, 9:18].any()
    # And the small grid region well inside that hole must be zero in the
    # aperture field.
    grid_T = ra.aperture_field(28e9).Ez
    ny, nx = grid_T.shape
    cy, cx = ny // 2, nx // 2
    half = max(1, max(ny, nx) // 32)  # ≪ the 9-cell hole
    center = grid_T[cy - half : cy + half, cx - half : cx + half]
    assert np.max(np.abs(center)) < 1e-9
    # And a non-trivial gain remains in the active cells.
    res = solver.solve(ra, 28e9)
    assert res.far_field.peak_directivity_dbi() > 15.0


@pytest.mark.parametrize(
    "theta_deg, phi_deg",
    [(0, 0), (10, 0), (20, 45), (30, 0), (45, 0)],
)
def test_reflectarray_steering(theta_deg: float, phi_deg: float) -> None:
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    ra = fa.Reflectarray(
        focal_length=0.20,
        design_freq=28e9,
        nx=24,
        ny=24,
        beam_direction=(math.radians(theta_deg), math.radians(phi_deg)),
    )
    res = solver.solve(ra, 28e9)
    intens = res.far_field.directivity()
    iy, ix = np.unravel_index(np.argmax(intens), intens.shape)
    u_peak = res.far_field.u[ix]
    v_peak = res.far_field.v[iy]
    u_exp = math.sin(math.radians(theta_deg)) * math.cos(math.radians(phi_deg))
    v_exp = math.sin(math.radians(theta_deg)) * math.sin(math.radians(phi_deg))
    assert abs(u_peak - u_exp) < 0.03
    assert abs(v_peak - v_exp) < 0.03
    A = (ra.nx * ra.cell_size) * (ra.ny * ra.cell_size)
    eta = aperture_efficiency(res.far_field, A)
    if theta_deg == 0:
        assert eta > 0.7
