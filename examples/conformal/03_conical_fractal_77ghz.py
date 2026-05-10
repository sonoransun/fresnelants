"""Example: conical fractal Fresnel lens at 77 GHz (axicon-style ring beam).

A 45° half-angle cone (5 cm height) with a stage-2 Cantor activation
pattern in projected radial coordinate. The cone naturally produces a
tilted ring beam (axicon / Bessel-beam style) rather than a single
broadside lobe — useful for non-diffracting wavefront generation.
"""

from __future__ import annotations

from pathlib import Path

import fresnelants as fa
from fresnelants.analysis.conformal_farfield import far_field_from_conformal


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    lens = fa.ConicalFractalFresnelLens(half_angle_deg=45.0, height=0.05, design_freq=77e9, stage=2)
    ap = lens.conformal_aperture(77e9)
    ff = far_field_from_conformal(ap, n_samples=64, chunk=4096)
    print(f"Conical fractal lens @ 77 GHz: peak directivity = {ff.peak_directivity_dbi():.2f} dBi")
    print(f"  Mesh facets: {len(ap.points)}, aperture power: {ap.power():.3e}")


if __name__ == "__main__":
    main()
