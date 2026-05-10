"""Example 8 — Sierpinski-tiled microstrip reflectarray at 28 GHz.

A 27 × 27 reflectarray whose active cells follow a stage-2 Sierpinski
carpet mask. Demonstrates that fractal sparsification preserves a
recognisable focused main beam at ~80 % the on-aperture cell count of a
fully populated array.
"""

from __future__ import annotations

import fresnelants as fa
from fresnelants.analysis.metrics import aperture_efficiency, hpbw


def main() -> None:
    full = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=27, ny=27)
    sra = fa.SierpinskiReflectarray(
        focal_length=0.20, design_freq=28e9, nx=27, ny=27, fractal_stage=2
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=4)
    res_full = solver.solve(full, 28e9)
    res_sra = solver.solve(sra, 28e9)

    A = (full.nx * full.cell_size) * (full.ny * full.cell_size)
    n_active = int(sra._cell_mask().sum())
    print(
        f"Full   27×27 RA: D = {res_full.far_field.peak_directivity_dbi():.2f} dBi, "
        f"η = {aperture_efficiency(res_full.far_field, A) * 100:.1f}%, "
        f"HPBW = {hpbw(res_full.far_field):.2f}°"
    )
    print(
        f"Sierpinski-tiled ({n_active}/{27 * 27} active cells): "
        f"D = {res_sra.far_field.peak_directivity_dbi():.2f} dBi, "
        f"η = {aperture_efficiency(res_sra.far_field, A) * 100:.1f}%, "
        f"HPBW = {hpbw(res_sra.far_field):.2f}°"
    )


if __name__ == "__main__":
    main()
