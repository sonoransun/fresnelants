"""Example 4 — 28 GHz mmW reflectarray with electronic beam steering."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fresnelants as fa
from fresnelants.analysis.metrics import aperture_efficiency
from fresnelants.export.gerber import write_reflectarray_gerber
from fresnelants.viz.plots2d import plot_farfield_2d, plot_principal_cuts, plot_zone_layout

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
GERBER_OUT = Path(__file__).resolve().parent.parent / "docs" / "gerber"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GERBER_OUT.mkdir(parents=True, exist_ok=True)

    freq = 28e9
    F = 0.20
    nx = ny = 32
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)

    ra = fa.Reflectarray(focal_length=F, design_freq=freq, nx=nx, ny=ny)
    res = solver.solve(ra, freq)

    A = (ra.nx * ra.cell_size) * (ra.ny * ra.cell_size)
    print(
        f"Broadside RA {nx}x{ny} @ {freq / 1e9:.0f} GHz: "
        f"D = {res.far_field.peak_directivity_dbi():.2f} dBi  "
        f"η = {aperture_efficiency(res.far_field, A) * 100:.1f}%"
    )

    fig = plot_zone_layout(ra)
    fig.savefig(OUT / "reflectarray_layout.png")
    plt.close(fig)

    fig = plot_farfield_2d(res.far_field)
    fig.savefig(OUT / "reflectarray_farfield.png")
    plt.close(fig)

    fig = plot_principal_cuts(res.far_field, label=f"Broadside RA, {freq / 1e9:.0f} GHz")
    fig.savefig(OUT / "reflectarray_cuts.png")
    plt.close(fig)

    # Beam-steering sweep.
    angles = [0, 10, 20, 30, 45]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for theta in angles:
        ra_s = fa.Reflectarray(
            focal_length=F,
            design_freq=freq,
            nx=nx,
            ny=ny,
            beam_direction=(math.radians(theta), 0.0),
        )
        ff = solver.solve(ra_s, freq).far_field
        theta_deg, db = ff.cut("E")
        valid = ~np.isnan(theta_deg)
        ax.plot(theta_deg[valid], db[valid], label=f"θ_b = {theta}°")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-40, 2)
    ax.set_xlabel("θ [deg]")
    ax.set_ylabel("Normalized gain [dB]")
    ax.set_title(f"Reflectarray beam steering at {freq / 1e9:.0f} GHz")
    ax.legend(loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "reflectarray_steering.png")
    plt.close(fig)

    # Gerber export.
    paths = write_reflectarray_gerber(ra, GERBER_OUT, base_name="reflectarray_28ghz")
    for label, p in paths.items():
        print(f"  Gerber {label}: {p.relative_to(Path(__file__).resolve().parent.parent)}")


if __name__ == "__main__":
    main()
