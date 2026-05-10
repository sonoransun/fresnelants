"""Example 6 — Cantor fractal zone plate with polyfocal axial signature.

Synthesizes a stage-3 Cantor binary zone plate at 30 GHz, F = 0.30 m,
runs the PO solver, and plots the on-axis intensity sweep showing peaks
at z = F, F/3, F/5, F/7 — the defining multifocal property of fractal
zone plates (Saavedra/Furlan/Monsoriu 2003).
"""

from __future__ import annotations

import math

import numpy as np

import fresnelants as fa
from fresnelants.analysis.metrics import aperture_efficiency, hpbw
from fresnelants.analysis.nearfield import focal_axis_intensity


def main() -> None:
    F = 0.30
    freq = 30e9

    cantor = fa.FractalSoretZonePlate(focal_length=F, design_freq=freq, stage=3)
    devil = fa.FractalWoodZonePlate(focal_length=F, design_freq=freq, stage=2, base_unit=2)

    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=4)
    res_c = solver.solve(cantor, freq)
    res_d = solver.solve(devil, freq)

    A_c = math.pi * cantor.aperture_radius**2
    A_d = math.pi * devil.aperture_radius**2
    print(
        f"Cantor stage 3 ({cantor.num_zones} zones, base 1): "
        f"D = {res_c.far_field.peak_directivity_dbi():.2f} dBi, "
        f"η = {aperture_efficiency(res_c.far_field, A_c) * 100:.1f}%, "
        f"HPBW = {hpbw(res_c.far_field):.2f}°"
    )
    print(
        f"Devil's lens stage 2, base 2 ({devil.num_zones} zones): "
        f"D = {res_d.far_field.peak_directivity_dbi():.2f} dBi, "
        f"η = {aperture_efficiency(res_d.far_field, A_d) * 100:.1f}%, "
        f"HPBW = {hpbw(res_d.far_field):.2f}°"
    )

    # The signature: dump axial intensity at z = F, F/3, F/5, F/7 to confirm
    # the polyfocal property numerically.
    ap = cantor.aperture_field(freq, samples_per_wavelength=4.0, margin=1.1)
    z_grid = np.linspace(0.05 * F, 1.3 * F, 120)
    I_axial = focal_axis_intensity(ap, z_grid)
    print("\nOn-axis intensity (Cantor polyfocal foci):")
    for k in (1, 3, 5, 7):
        z_target = F / k
        idx = int(np.argmin(np.abs(z_grid - z_target)))
        print(
            f"  z = F/{k:>1d} = {z_target:.3f} m  →  |E|² = {I_axial[idx] / I_axial.max():.3f} (normalized)"
        )


if __name__ == "__main__":
    main()
