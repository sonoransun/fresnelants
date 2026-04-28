"""Example 2 — Offset Fresnel zone plate at Ku-band (12 GHz).

Off-axis feed, broadside main beam — the antenna analogue of an offset-fed
parabolic dish. Avoids feed-blockage losses on conventional zone plates.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

import fresnelants as fa
from fresnelants.analysis.metrics import aperture_efficiency
from fresnelants.viz.plots2d import plot_farfield_2d, plot_principal_cuts, plot_zone_layout

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    F = 0.50
    freq = 12e9
    tilt_deg = 25.0
    design = fa.OffsetZonePlate(
        focal_length=F,
        design_freq=freq,
        aperture_radius_m=0.30,
        tilt_angle=math.radians(tilt_deg),
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    res = solver.solve(design, freq)

    A = math.pi * design.aperture_radius**2
    print(
        f"Offset Wood (tilt {tilt_deg}°) @ {freq / 1e9:.0f} GHz: "
        f"D = {res.far_field.peak_directivity_dbi():.2f} dBi  "
        f"η = {aperture_efficiency(res.far_field, A) * 100:.1f}%"
    )

    fig = plot_zone_layout(design)
    fig.savefig(OUT / "offset_layout.png")
    plt.close(fig)

    fig = plot_farfield_2d(res.far_field)
    fig.savefig(OUT / "offset_farfield.png")
    plt.close(fig)

    fig = plot_principal_cuts(res.far_field, label=f"Offset {tilt_deg}°, 12 GHz")
    fig.savefig(OUT / "offset_cuts.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
