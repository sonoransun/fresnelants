"""RIS #2 — 1-bit MACOM PIN diode array at 39 GHz, scan-loss curve."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fresnelants as fa
from fresnelants.cells.pin_diode import MACOM_MA4FCP305
from fresnelants.cells.varactor import Skyworks_SMV1232

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "img"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pin = MACOM_MA4FCP305()
    var = Skyworks_SMV1232()
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2)

    angles = np.arange(-60, 65, 5, dtype=float)
    g_1bit, g_cont = [], []
    for theta in angles:
        ris1 = fa.CodedRIS(
            focal_length=0.20,
            design_freq=39e9,
            nx=24,
            ny=24,
            cell=pin,
            bits=1,
            beam_direction=(math.radians(theta), 0),
        )
        rcont = fa.ReconfigurableArray(
            focal_length=0.20,
            design_freq=28e9,
            nx=24,
            ny=24,
            cell=var,
            beam_direction=(math.radians(theta), 0),
        )
        g_1bit.append(solver.solve(ris1, 39e9).far_field.peak_directivity_dbi())
        g_cont.append(solver.solve(rcont, 28e9).far_field.peak_directivity_dbi())

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(angles, g_1bit, "o-", color="tab:red", label="1-bit PIN @ 39 GHz")
    ax.plot(angles, g_cont, "s--", color="tab:blue", label="Continuous varactor @ 28 GHz")
    ax.set_xlabel("Steering angle θ_b [deg]")
    ax.set_ylabel("Peak directivity [dBi]")
    ax.set_title("RIS scan-loss vs steering — 1-bit PIN vs continuous varactor")
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(OUT / "ris_1bit_scanloss.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
