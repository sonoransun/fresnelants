"""Conformal #1 — cylindrical Fresnel lens at 77 GHz (automotive radar)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fresnelants as fa
from fresnelants.analysis.conformal_farfield import far_field_from_conformal

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "img"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lens = fa.CylindricalFresnelLens(radius=0.05, height=0.04, design_freq=77e9, nu=80, nv=40)
    ap = lens.conformal_aperture(77e9)
    ff = far_field_from_conformal(ap, n_samples=64, chunk=4096)
    print(f"Cylindrical lens @ 77 GHz: D = {ff.peak_directivity_dbi():.2f} dBi")

    # 3D mesh of facet positions, color-coded by tangential field magnitude.
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    pts = ap.points
    mag = np.abs(ap.Et)
    sc = ax.scatter(pts[:, 0] * 1e3, pts[:, 1] * 1e3, pts[:, 2] * 1e3, c=mag, cmap="magma", s=4)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("Cylindrical aperture (radius 50 mm, height 40 mm)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="|Et|")

    # Far-field map.
    ax2 = fig.add_subplot(1, 2, 2)
    d = ff.directivity()
    d_db = 10 * np.log10(np.maximum(d, 1e-30))
    extent = (ff.u[0], ff.u[-1], ff.v[0], ff.v[-1])
    im = ax2.imshow(
        d_db, extent=extent, origin="lower", cmap="inferno", vmin=d_db.max() - 30, vmax=d_db.max()
    )
    ax2.set_title(f"Far-field directivity [dBi] — peak {d_db.max():.1f}")
    ax2.set_xlabel(r"$u$")
    ax2.set_ylabel(r"$v$")
    ax2.set_aspect("equal")
    fig.colorbar(im, ax=ax2, shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "conformal_cylindrical_77ghz.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
