"""Phase 4 — conformal designs."""

from __future__ import annotations

import numpy as np

import fresnelants as fa
from fresnelants.analysis.conformal_farfield import far_field_from_conformal
from fresnelants.core.conformal import ConformalAperture


def test_cylinder_mesh_shapes() -> None:
    ap = ConformalAperture.from_cylinder(radius=0.10, height=0.05, nu=40, nv=20, freq=77e9)
    assert ap.points.shape == (40 * 20, 3)
    assert ap.normals.shape == (40 * 20, 3)
    assert ap.areas.shape == (40 * 20,)
    # Normals should be unit length (approximately).
    np.testing.assert_allclose(np.linalg.norm(ap.normals, axis=1), 1.0, atol=1e-9)


def test_sphere_mesh_areas_sum_to_cap() -> None:
    R = 0.05
    cap = np.pi / 2  # hemisphere
    ap = ConformalAperture.from_sphere(radius=R, cap_angle=cap, nu=120, nv=60, freq=28e9)
    expected = 2 * np.pi * R**2 * (1 - np.cos(cap))  # spherical-cap area
    assert abs(ap.areas.sum() - expected) / expected < 0.05


def test_cylindrical_lens_emits_broadside() -> None:
    """Cylindrical Fresnel lens with feed at the axis emits in +y direction."""
    lens = fa.CylindricalFresnelLens(radius=0.05, height=0.04, design_freq=77e9, nu=60, nv=30)
    ap = lens.conformal_aperture(77e9)
    ff = far_field_from_conformal(ap, n_samples=64, chunk=4096)
    intens = ff.intensity()
    _iy, ix = np.unravel_index(np.argmax(intens), intens.shape)
    u_peak = ff.u[ix]
    # +y direction in far-field is (u, v) = (0, +1). The mesh is on the +y
    # side and emits broadside; peak should land near (u, v) ≈ (0, ±1).
    assert abs(u_peak) < 0.3
    # Peak directivity is non-trivial for a 4 cm × 10 cm cylindrical aperture
    # at 77 GHz.
    assert ff.peak_directivity_dbi() > 5.0


def test_spherical_lens_emits() -> None:
    lens = fa.SphericalFresnelLens(radius=0.05, cap_angle_deg=60, design_freq=28e9, nu=60, nv=30)
    ap = lens.conformal_aperture(28e9)
    ff = far_field_from_conformal(ap, n_samples=64, chunk=4096)
    assert ff.peak_directivity_dbi() > 5.0


def test_conformal_solver_smoke() -> None:
    lens = fa.CylindricalFresnelLens(radius=0.05, height=0.04, design_freq=77e9, nu=40, nv=20)
    solver = fa.ConformalPOSolver(n_samples=32, chunk=2048)
    res = solver.solve(lens, 77e9)
    assert res.far_field.peak_directivity_dbi() > 0.0


def test_jit_matches_numpy() -> None:
    """If Numba is installed, JIT and numpy paths must agree to 1e-9."""
    pytest = __import__("pytest")
    try:
        import numba  # noqa: F401
    except ImportError:
        pytest.skip("numba not installed")
    lens = fa.CylindricalFresnelLens(radius=0.05, height=0.04, design_freq=77e9, nu=40, nv=20)
    ap = lens.conformal_aperture(77e9)
    ff_np = far_field_from_conformal(ap, n_samples=32, chunk=2048, use_jit=False)
    ff_jit = far_field_from_conformal(ap, n_samples=32, chunk=2048, use_jit=True)
    np.testing.assert_allclose(ff_np.E, ff_jit.E, atol=1e-9, rtol=1e-9)
