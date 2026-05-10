"""Example 11 — 128-element hexagonal macro array of Soret zone plates at 30 GHz.

Demonstrates the high-N limit: 128 small Soret elements in a close-packed
hexagonal grid form a sparse aperture array similar to the SKA-Low /
millimetre-wave radio-astronomy receiver architecture. Grating lobes are
present in visible space because element spacing exceeds λ.
"""

from __future__ import annotations

import fresnelants as fa


def main() -> None:
    elem = fa.SoretZonePlate(focal_length=0.05, design_freq=30e9, num_zones=2)
    wavelength = 3e8 / 30e9
    arr = fa.MacroFresnelArray.from_lattice(
        elem, n_elements=128, spacing_m=1.5 * wavelength, lattice="hex"
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=4)
    g_elem = solver.solve(elem, 30e9).far_field.peak_directivity_dbi()
    res = arr.solve(solver, 30e9)
    g_arr = res.far_field.peak_directivity_dbi()
    print(
        f"128× Soret-ZP hex-close-packed @ 30 GHz, spacing 1.5λ\n"
        f"  Element peak directivity: {g_elem:.2f} dBi\n"
        f"  Array   peak directivity: {g_arr:.2f} dBi  (Δ = {g_arr - g_elem:.2f} dB)\n"
        f"  Array footprint: {arr.array_extent[0] * 100:.1f} × {arr.array_extent[1] * 100:.1f} cm"
    )
    print(f"  Grating lobes (theory): u_g = ±λ/d = ±{wavelength / arr.min_neighbour_spacing:.3f}")


if __name__ == "__main__":
    main()
