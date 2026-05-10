"""Example 9 — 4-element linear macro array of Wood zone plates at 10 GHz.

The simplest phased-array receiver: four 8-zone Wood zone plates in a row,
spaced at 1.2 m. Demonstrates the array-factor × element-pattern formulation
and the textbook ~6 dB gain enhancement at broadside.
"""

from __future__ import annotations

import numpy as np

import fresnelants as fa


def main() -> None:
    elem = fa.WoodZonePlate(focal_length=1.0, design_freq=10e9, num_zones=8)
    arr = fa.MacroFresnelArray.from_lattice(elem, n_elements=4, spacing_m=1.2, lattice="linear")
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=4)
    g_elem = solver.solve(elem, 10e9).far_field.peak_directivity_dbi()
    res = arr.solve(solver, 10e9)
    g_arr = res.far_field.peak_directivity_dbi()
    print(
        f"4× Wood-ZP linear array @ 10 GHz, spacing 1.2 m\n"
        f"  Element peak directivity: {g_elem:.2f} dBi\n"
        f"  Array   peak directivity: {g_arr:.2f} dBi  (Δ = {g_arr - g_elem:.2f} dB)\n"
        f"  N elements: {arr.n_elements}, footprint extent: "
        f"{arr.array_extent[0] * 1e3:.0f} × {arr.array_extent[1] * 1e3:.0f} mm"
    )

    # Steering sweep within the element beam.
    print("\nSteering sweep:")
    for theta_b in (0, 1, 2, 3):
        w = arr.weights_for_beam(theta_b, 0.0)
        res_s = arr.solve(solver, 10e9, weights=w)
        intens = res_s.far_field.directivity()
        _iy, ix = np.unravel_index(int(np.argmax(intens)), intens.shape)
        u_peak = float(res_s.far_field.u[ix])
        u_target = float(np.sin(np.deg2rad(theta_b)))
        print(
            f"  θ_b = {theta_b}°: peak D = {res_s.far_field.peak_directivity_dbi():.2f} dBi, "
            f"u_peak = {u_peak:+.3f} (target {u_target:+.3f})"
        )


if __name__ == "__main__":
    main()
