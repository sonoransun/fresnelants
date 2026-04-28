"""Closed-form identities for Fresnel zone geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fresnelants.core.geometry import (
    fresnel_zone_radii,
    make_aperture_grid,
    path_length_phase,
    zone_radius,
)


def test_zone_radius_first_zone_paraxial():
    # r1 ≈ √(λF) for λ ≪ F. Exact form has +(λ/2)² correction.
    F = 1.0
    lam = 0.01
    r1 = float(zone_radius(1, F, lam))
    expected = math.sqrt(lam * F)
    assert r1 == pytest.approx(expected, rel=2e-3)
    # And exact identity holds tightly.
    assert r1**2 == pytest.approx(lam * F + (lam / 2) ** 2, rel=1e-12)


def test_zone_radius_monotonic():
    radii = fresnel_zone_radii(20, 0.5, 0.03)
    assert np.all(np.diff(radii) > 0)


@given(
    n=st.integers(min_value=1, max_value=50),
    F=st.floats(min_value=0.05, max_value=5.0, allow_nan=False),
    lam=st.floats(min_value=1e-3, max_value=1.0, allow_nan=False),
)
def test_zone_radius_exact_form(n: int, F: float, lam: float) -> None:
    r = float(zone_radius(n, F, lam))
    assert r**2 == pytest.approx(n * lam * F + (n * lam / 2) ** 2)


def test_zone_radius_rejects_invalid():
    with pytest.raises(ValueError):
        zone_radius(-1, 1.0, 0.01)
    with pytest.raises(ValueError):
        zone_radius(1, -1.0, 0.01)
    with pytest.raises(ValueError):
        zone_radius(1, 1.0, -0.01)


def test_aperture_grid_shape():
    g = make_aperture_grid(0.10, 6.0, 0.01)
    assert g.shape == (g.ny, g.nx)
    assert g.X.shape == g.shape
    assert abs(g.dx - g.Lx / g.nx) < 1e-15
    assert g.R.min() >= 0


def test_path_length_phase_on_axis():
    g = make_aperture_grid(0.10, 4.0, 0.03)
    F = 1.0
    phi = path_length_phase(g, F, 0.03)
    # On-axis phase corresponds to distance F.
    cy, cx = g.ny // 2, g.nx // 2
    assert phi[cy, cx] == pytest.approx(2 * math.pi * F / 0.03, rel=1e-3)
