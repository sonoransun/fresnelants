"""Example 10 — 16-element 4×4 rectangular macro array at 28 GHz.

Each element is a small 8×8 reflectarray (broad pattern, scannable). The
full array forms a 4-beam codebook for downlink reception across the
5G mm-wave Ku/Ka cells.
"""

from __future__ import annotations

import fresnelants as fa


def main() -> None:
    elem = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=8, ny=8)
    arr = fa.MacroFresnelArray.from_lattice(
        elem, n_elements=16, spacing_m=0.05, lattice="rect", rows=4
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=4)
    g_elem = solver.solve(elem, 28e9).far_field.peak_directivity_dbi()
    res = arr.solve(solver, 28e9)
    g_arr = res.far_field.peak_directivity_dbi()
    print(
        f"16× 8×8-RA macro 4×4 grid @ 28 GHz, spacing 5 cm\n"
        f"  Element peak directivity: {g_elem:.2f} dBi\n"
        f"  Array   peak directivity: {g_arr:.2f} dBi  (Δ = {g_arr - g_elem:.2f} dB)\n"
        f"  Array footprint: {arr.array_extent[0] * 100:.1f} × {arr.array_extent[1] * 100:.1f} cm"
    )

    # 4-beam codebook for receive multi-beam operation.
    book = arr.beam_codebook(
        directions=[(-15.0, 0.0), (-5.0, 0.0), (5.0, 0.0), (15.0, 0.0)],
        labels=["beam_W", "beam_C-", "beam_C+", "beam_E"],
    )
    print("\n4-beam codebook:")
    for label, w in book.items():
        res_b = arr.solve(solver, 28e9, weights=w)
        print(f"  {label}: peak D = {res_b.far_field.peak_directivity_dbi():.2f} dBi")


if __name__ == "__main__":
    main()
