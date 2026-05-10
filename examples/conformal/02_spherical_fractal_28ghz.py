"""Example: spherical fractal Fresnel lens at 77 GHz.

A 5 cm-radius hemispherical cap with a stage-2 Cantor activation pattern
in projected radial coordinate plus the standard +z-focusing phase ramp.
"""

from __future__ import annotations

from pathlib import Path

import fresnelants as fa
from fresnelants.analysis.conformal_farfield import far_field_from_conformal


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    lens = fa.SphericalFractalFresnelLens(
        radius=0.05, cap_angle_deg=90.0, design_freq=77e9, stage=2
    )
    ap = lens.conformal_aperture(77e9)
    ff = far_field_from_conformal(ap, n_samples=64, chunk=4096)
    print(
        f"Spherical fractal lens @ 77 GHz: peak directivity = {ff.peak_directivity_dbi():.2f} dBi"
    )
    print(f"  Mesh facets: {len(ap.points)}, aperture power: {ap.power():.3e}")


if __name__ == "__main__":
    main()
