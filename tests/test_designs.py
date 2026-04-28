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
