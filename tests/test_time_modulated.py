"""Phase B1 — time-modulated arrays + harmonic far-field + DOA estimator."""

from __future__ import annotations

import numpy as np

import fresnelants as fa
from fresnelants.analysis.harmonics import direction_of_arrival, harmonic_far_field
from fresnelants.cells.varactor import Skyworks_SMV1232


def test_square_wave_zero_harmonic_equals_duty() -> None:
    """The DC (n=0) Fourier coefficient of a square wave equals its duty cycle."""
    array = fa.TimeModulatedArray(
        focal_length=0.20, design_freq=28e9, nx=4, ny=4, cell=Skyworks_SMV1232()
    )
    array.linear_progressive_schedule(time_per_cell=0.5, duty=0.5)
    c0 = array.harmonic_coefficient(0)
    np.testing.assert_allclose(c0, 0.5 + 0j, atol=1e-9)


def test_square_wave_first_harmonic_magnitude() -> None:
    """For a 50% duty schedule the |c_1| coefficient is 1/π ≈ 0.318."""
    array = fa.TimeModulatedArray(
        focal_length=0.20, design_freq=28e9, nx=4, ny=4, cell=Skyworks_SMV1232()
    )
    array.linear_progressive_schedule(time_per_cell=0.5, duty=0.5)
    c1 = array.harmonic_coefficient(1)
    expected_mag = 1.0 / np.pi
    np.testing.assert_allclose(np.abs(c1), expected_mag, atol=1e-9)


def test_square_wave_no_even_harmonics() -> None:
    """Symmetric 50% duty squares produce zero on even harmonics > 0."""
    array = fa.TimeModulatedArray(
        focal_length=0.20, design_freq=28e9, nx=4, ny=4, cell=Skyworks_SMV1232()
    )
    # Symmetric: t_on=0, duty=0.5 — same for every cell.
    sched = np.zeros((array.ny, array.nx, 2))
    sched[..., 0] = 0.0
    sched[..., 1] = 0.5
    array.schedule = sched
    c2 = array.harmonic_coefficient(2)
    c4 = array.harmonic_coefficient(4)
    assert np.allclose(np.abs(c2), 0.0, atol=1e-9)
    assert np.allclose(np.abs(c4), 0.0, atol=1e-9)


def test_harmonic_far_field_runs() -> None:
    array = fa.TimeModulatedArray(
        focal_length=0.20, design_freq=28e9, nx=12, ny=12, cell=Skyworks_SMV1232()
    )
    array.linear_progressive_schedule(time_per_cell=0.5, duty=0.5)
    ff_0 = harmonic_far_field(array, 28e9, 0, samples_per_wavelength=4.0, pad_factor=2)
    ff_1 = harmonic_far_field(array, 28e9, 1, samples_per_wavelength=4.0, pad_factor=2)
    assert ff_0.peak_directivity_dbi() > ff_1.peak_directivity_dbi()  # DC carries more power


def test_harmonic_solver() -> None:
    array = fa.TimeModulatedArray(
        focal_length=0.20, design_freq=28e9, nx=12, ny=12, cell=Skyworks_SMV1232()
    )
    array.linear_progressive_schedule(time_per_cell=0.5, duty=0.5)
    solver = fa.HarmonicPOSolver(samples_per_wavelength=4.0, pad_factor=2)
    res = solver.solve(array, 28e9, harmonics=[-1, 0, 1])
    assert res.harmonics == [-1, 0, 1]
    assert all(ff.peak_directivity_dbi() > -50 for ff in res.far_fields)


def test_doa_estimator_smoke() -> None:
    """DOA estimator returns finite (theta, phi) for sensible inputs."""
    array = fa.TimeModulatedArray(
        focal_length=0.20, design_freq=28e9, nx=12, ny=12, cell=Skyworks_SMV1232()
    )
    array.linear_progressive_schedule(time_per_cell=0.5, duty=0.5)
    ff_0 = harmonic_far_field(array, 28e9, 0, samples_per_wavelength=4.0, pad_factor=2)
    ff_1 = harmonic_far_field(array, 28e9, 1, samples_per_wavelength=4.0, pad_factor=2)
    theta, phi = direction_of_arrival(ff_0, ff_1)
    assert not np.isnan(theta)
    assert -90 <= theta <= 90
    assert -180 <= phi <= 180
