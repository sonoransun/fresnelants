"""Example 3 — N-level phase-correcting Fresnel plate at 94 GHz.

Demonstrates how primary-focus efficiency rises as the number of phase levels
increases: 1 level (Soret) → 2 (Wood) → 4 (quarter-wave) → continuous.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fresnelants as fa
from fresnelants.analysis.metrics import aperture_efficiency
from fresnelants.viz.plots2d import plot_farfield_2d, plot_principal_cuts, plot_zone_layout

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    F = 0.05
    freq = 94e9
    R = 0.025  # 25 mm aperture
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)

    levels_seq = [1, 2, 4, 8, 1024]
    gains: list[float] = []
    for levels in levels_seq:
        if levels == 1:
            d = fa.SoretZonePlate(
                focal_length=F,
                design_freq=freq,
                num_zones=int(R**2 / (3e8 / freq * F)),
            )
        elif levels == 2:
            d = fa.WoodZonePlate(  # type: ignore[assignment]
                focal_length=F,
                design_freq=freq,
                num_zones=int(R**2 / (3e8 / freq * F)),
            )
        else:
            d = fa.PhaseCorrectingPlate(  # type: ignore[assignment]
                focal_length=F,
                design_freq=freq,
                aperture_radius_m=R,
                levels=levels,
            )
        res = solver.solve(d, freq)
        A = math.pi * d.aperture_radius**2
        gains.append(res.far_field.peak_directivity_dbi())
        print(
            f"  levels={levels:>4}: D = {gains[-1]:5.2f} dBi  η = {aperture_efficiency(res.far_field, A) * 100:5.1f}%"
        )

    # Headline plate: quarter-wave (4-level) at 94 GHz.
    plate = fa.PhaseCorrectingPlate(focal_length=F, design_freq=freq, aperture_radius_m=R, levels=4)
    res = solver.solve(plate, freq)

    fig = plot_zone_layout(plate)
    fig.savefig(OUT / "phase_correcting_layout.png")
    plt.close(fig)

    fig = plot_farfield_2d(res.far_field)
    fig.savefig(OUT / "phase_correcting_farfield.png")
    plt.close(fig)

    fig = plot_principal_cuts(res.far_field, label="4-level phase plate, 94 GHz")
    fig.savefig(OUT / "phase_correcting_cuts.png")
    plt.close(fig)

    # Levels-vs-gain bar chart.
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    labels = [f"{n}" if n < 1024 else "cont." for n in levels_seq]
    ax.bar(labels, gains, color="tab:blue")
    ax.axhline(
        10 * np.log10(4 * np.pi * (np.pi * R**2) / (3e8 / freq) ** 2),
        color="red",
        linestyle="--",
        label="uniform-aperture max",
    )
    ax.set_xlabel("phase levels")
    ax.set_ylabel("Peak directivity [dBi]")
    ax.set_title(f"Levels vs gain @ {freq / 1e9:.0f} GHz, R = {R * 1e3:.0f} mm")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "phase_correcting_levels.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
