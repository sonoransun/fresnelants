"""Metasurface #1 — Pancharatnam-Berry circular-polarization lens at 60 GHz."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fresnelants as fa
from fresnelants.analysis.dualpol_metrics import (
    cross_polarization_db,
    polarization_purity,
)
from fresnelants.analysis.farfield import far_field_from_jones_aperture

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "img"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lens = fa.MetasurfaceLens(focal_length=0.05, design_freq=60e9, aperture_radius_m=0.025)
    ap = lens.jones_aperture_field(60e9, samples_per_wavelength=6.0)
    ff = far_field_from_jones_aperture(ap, pad_factor=4)

    print(f"Peak directivity: {ff.peak_directivity_dbi():.2f} dBi")
    print(f"RCP polarization purity: {polarization_purity(ff, 'rcp') * 100:.1f}%")
    print(f"LCP polarization purity: {polarization_purity(ff, 'lcp') * 100:.1f}%")

    # Co/cross views.
    co, cross = ff.co_cross("rcp")
    co_intens = np.abs(co) ** 2
    cross_intens = np.abs(cross) ** 2
    co_intens[~ff.visible_mask] = np.nan
    cross_intens[~ff.visible_mask] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    extent = (ff.u[0], ff.u[-1], ff.v[0], ff.v[-1])
    im0 = axes[0].imshow(
        10 * np.log10(np.maximum(co_intens, 1e-30)),
        extent=extent,
        origin="lower",
        cmap="inferno",
    )
    axes[0].set_title("Co-pol (RCP) far-field [dB]")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(
        cross_polarization_db(ff, "rcp"),
        extent=extent,
        origin="lower",
        cmap="inferno",
    )
    axes[1].set_title("Cross-pol level [dB below co peak]")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    for ax in axes:
        ax.set_xlabel(r"$u$")
        ax.set_ylabel(r"$v$")
        ax.set_aspect("equal")
    fig.suptitle("PB metasurface lens @ 60 GHz — LCP in / RCP focused")
    fig.tight_layout()
    fig.savefig(OUT / "metasurface_pb_pol.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
