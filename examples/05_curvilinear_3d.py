"""Example 5 — 3D curvilinear (hyperbolic singlet) Fresnel lens at 30 GHz.

Generates the lens, evaluates its far-field, plots a 3D surface profile,
renders the focal-region intensity, and exports an STL ready to 3D print.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

import fresnelants as fa
from fresnelants.analysis.metrics import aperture_efficiency
from fresnelants.export.stl import export_stl
from fresnelants.viz.plots2d import plot_farfield_2d, plot_principal_cuts
from fresnelants.viz.plots3d import (
    plot_3d_radiation_pattern,
    plot_3d_surface_profile,
    plot_focal_region,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
STL_OUT = Path(__file__).resolve().parent.parent / "docs" / "stl"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STL_OUT.mkdir(parents=True, exist_ok=True)
    freq = 30e9
    F = 0.10
    R = 0.05
    lens = fa.CurvilinearFresnel(
        focal_length=F, design_freq=freq, aperture_radius_m=R, profile="hyperbolic"
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    res = solver.solve(lens, freq)

    A = math.pi * lens.aperture_radius**2
    print(
        f"Curvilinear hyperbolic lens @ {freq / 1e9:.0f} GHz: "
        f"D = {res.far_field.peak_directivity_dbi():.2f} dBi  "
        f"η = {aperture_efficiency(res.far_field, A) * 100:.1f}%"
    )

    fig = plot_3d_surface_profile(lens, samples=160)
    fig.savefig(OUT / "curvilinear_surface_3d.png")
    plt.close(fig)

    fig = plot_farfield_2d(res.far_field)
    fig.savefig(OUT / "curvilinear_farfield.png")
    plt.close(fig)

    fig = plot_principal_cuts(res.far_field, label=f"Hyperbolic singlet, {freq / 1e9:.0f} GHz")
    fig.savefig(OUT / "curvilinear_cuts.png")
    plt.close(fig)

    fig = plot_3d_radiation_pattern(res.far_field, dynamic_range_db=30)
    fig.savefig(OUT / "curvilinear_3d_pattern.png")
    plt.close(fig)

    fig = plot_focal_region(res.aperture, z_focal=F, z_span=0.6 * F, samples=41)
    fig.savefig(OUT / "curvilinear_focal_region.png")
    plt.close(fig)

    stl_path = export_stl(
        lens, STL_OUT / "hyperbolic_lens.stl", radial_samples=200, angular_samples=240
    )
    print(f"  STL: {stl_path.relative_to(Path(__file__).resolve().parent.parent)}")


if __name__ == "__main__":
    main()
