"""Example 7 — Sierpinski-carpet zone plate at 30 GHz.

A square aperture tiled with a stage-3 Sierpinski carpet. The mask's
self-similar structure produces a 4-fold-symmetric far-field with
self-similar diffraction lobes.
"""

from __future__ import annotations

import math

import fresnelants as fa
from fresnelants.analysis.metrics import aperture_efficiency, hpbw


def main() -> None:
    carpet = fa.SierpinskiCarpetZonePlate(
        focal_length=0.5, design_freq=30e9, stage=3, aperture_side=0.18
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=4)
    res = solver.solve(carpet, 30e9)

    A = math.pi * carpet.aperture_radius**2  # circumscribed disk area
    print(
        f"Sierpinski stage 3 (180 mm side): "
        f"D = {res.far_field.peak_directivity_dbi():.2f} dBi, "
        f"η = {aperture_efficiency(res.far_field, A) * 100:.1f}%, "
        f"HPBW = {hpbw(res.far_field):.2f}°"
    )


if __name__ == "__main__":
    main()
