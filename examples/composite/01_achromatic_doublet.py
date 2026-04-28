"""Composite #1 — achromatic doublet bandwidth comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fresnelants as fa

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "img"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f_low, f_high = 28e9, 32e9
    R = 0.05
    F = 0.10

    doublet = fa.AchromaticDoublet(
        f_low=f_low, f_high=f_high, aperture_radius_m=R, levels=8, feed_distance=F
    )
    single = fa.PhaseCorrectingPlate(
        focal_length=F, design_freq=30e9, aperture_radius_m=R, levels=8
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2)
    cas = fa.CascadePOSolver(samples_per_wavelength=4.0, pad_factor=2)

    freqs = np.linspace(24e9, 36e9, 9)
    g_doublet, g_single = [], []
    for f in freqs:
        g_doublet.append(cas.solve(doublet, f).far_field.peak_directivity_dbi())
        g_single.append(solver.solve(single, f).far_field.peak_directivity_dbi())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs / 1e9, g_doublet, "o-", color="tab:blue", label="Achromatic doublet")
    ax.plot(freqs / 1e9, g_single, "s--", color="tab:orange", label="Single 8-level plate")
    ax.axvspan(f_low / 1e9, f_high / 1e9, alpha=0.10, color="tab:blue", label="design band")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Peak directivity [dBi]")
    ax.set_title("Achromatic doublet bandwidth")
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(OUT / "composite_doublet_bandwidth.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
