"""PO far-field validation against analytical identities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fresnelants.analysis.aperture import ApertureField
from fresnelants.analysis.farfield import far_field_from_aperture
from fresnelants.core.geometry import make_aperture_grid


@pytest.mark.parametrize("R_mm,freq", [(150, 10e9), (50, 30e9), (20, 60e9)])
def test_uniform_circular_aperture_directivity(R_mm: float, freq: float) -> None:
    """A uniform circular aperture has D = 4πA/λ². Verify within 0.5 dB."""
    R = R_mm * 1e-3
    lam = 3e8 / freq
    grid = make_aperture_grid(2 * R * 1.2, 6.0, lam)
    Ez = np.where(grid.R <= R, 1.0 + 0j, 0 + 0j)
    ap = ApertureField(grid=grid, Ez=Ez, freq=freq)
    ff = far_field_from_aperture(ap, pad_factor=4)
    expected_dbi = 10.0 * math.log10(4 * math.pi**2 * R**2 / lam**2)
    assert ff.peak_directivity_dbi() == pytest.approx(expected_dbi, abs=0.5)


def test_phase_ramp_steers_beam():
    """exp(-j k u₀ x) on the aperture peaks at u = +u₀."""
    freq = 28e9
    lam = 3e8 / freq
    k = 2 * math.pi / lam
    R = 0.05
    u0 = 0.4
    grid = make_aperture_grid(2 * R * 1.05, 8.0, lam)
    mask = (np.abs(grid.X) <= R) & (np.abs(grid.Y) <= R)
    Ez = np.where(mask, np.exp(-1j * k * u0 * grid.X), 0 + 0j)
    ap = ApertureField(grid=grid, Ez=Ez, freq=freq)
    ff = far_field_from_aperture(ap, pad_factor=4)
    d = ff.directivity()
    iy, ix = np.unravel_index(np.argmax(d), d.shape)
    assert ff.u[ix] == pytest.approx(u0, abs=0.01)
    assert ff.v[iy] == pytest.approx(0.0, abs=0.01)


def test_no_radiation_for_zero_field():
    freq = 10e9
    grid = make_aperture_grid(0.05, 4.0, 0.03)
    Ez = np.zeros(grid.shape, dtype=np.complex128)
    ap = ApertureField(grid=grid, Ez=Ez, freq=freq)
    ff = far_field_from_aperture(ap)
    assert ff.aperture_power == 0
    assert np.all(np.abs(ff.E) == 0)
