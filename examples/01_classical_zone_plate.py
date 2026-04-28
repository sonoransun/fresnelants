"""Example 1 — Classical Soret and Wood zone plates at 10 GHz."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

import fresnelants as fa
from fresnelants.analysis.metrics import aperture_efficiency, hpbw
from fresnelants.viz.plots2d import plot_farfield_2d, plot_principal_cuts, plot_zone_layout

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    F = 1.0
    freq = 10e9
    soret = fa.SoretZonePlate(focal_length=F, design_freq=freq, num_zones=12)
    wood = fa.WoodZonePlate(focal_length=F, design_freq=freq, num_zones=12)

    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    res_s = solver.solve(soret, freq)
    res_w = solver.solve(wood, freq)

    A = math.pi * wood.aperture_radius**2
    print(
        f"Soret 12-zone @ {freq / 1e9:.0f} GHz: D = {res_s.far_field.peak_directivity_dbi():.2f} dBi, "
        f"η = {aperture_efficiency(res_s.far_field, A) * 100:.1f}%, "
        f"HPBW = {hpbw(res_s.far_field):.2f}°"
    )
    print(
        f"Wood  12-zone @ {freq / 1e9:.0f} GHz: D = {res_w.far_field.peak_directivity_dbi():.2f} dBi, "
        f"η = {aperture_efficiency(res_w.far_field, A) * 100:.1f}%, "
        f"HPBW = {hpbw(res_w.far_field):.2f}°"
    )

    fig = plot_zone_layout(soret)
    fig.savefig(OUT / "zone_plate_soret_layout.png")
    plt.close(fig)

    fig = plot_zone_layout(wood)
    fig.savefig(OUT / "zone_plate_wood_layout.png")
    plt.close(fig)

    fig = plot_farfield_2d(res_w.far_field)
    fig.savefig(OUT / "zone_plate_wood_farfield.png")
    plt.close(fig)

    fig = plot_principal_cuts(res_w.far_field, label="Wood 12 zones, 10 GHz")
    fig.savefig(OUT / "zone_plate_wood_cuts.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
