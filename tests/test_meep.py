"""Phase A4 — Meep adapter cross-validation (gated on the [fullwave] extra)."""

from __future__ import annotations

import pytest

import fresnelants as fa
from fresnelants.solvers.meep_adapter import MeepAdapter

pytestmark = pytest.mark.fullwave


def _have_meep() -> bool:
    try:
        import meep  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _have_meep(), reason="meep not installed (conda-only)")
def test_meep_adapter_runs_zone_plate() -> None:
    """A tiny zone plate must run through Meep and produce a finite focal field."""
    design = fa.SoretZonePlate(focal_length=0.10, design_freq=10e9, num_zones=4)
    adapter = MeepAdapter(resolution_per_wavelength=10, runtime_periods=20.0)
    result = adapter.solve(design, 10e9)
    assert result.aperture.power() > 0
    assert result.far_field.peak_directivity_dbi() > -50.0


def test_meep_adapter_raises_without_meep() -> None:
    """When Meep is missing, calling solve() must raise a clear runtime error."""
    if _have_meep():
        pytest.skip("meep IS installed; cannot test the missing-import branch here.")
    design = fa.SoretZonePlate(focal_length=0.10, design_freq=10e9, num_zones=4)
    adapter = MeepAdapter()
    with pytest.raises(RuntimeError, match="Meep is not installed"):
        adapter.solve(design, 10e9)


def test_meep_adapter_rejects_non_axisymmetric() -> None:
    """The adapter is currently axisymmetric-zone-plate only."""
    if not _have_meep():
        pytest.skip("meep not installed; cannot exercise the type guard.")
    ra = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=8, ny=8)
    adapter = MeepAdapter()
    with pytest.raises(TypeError, match="axisymmetric zone plates"):
        adapter.solve(ra, 28e9)
