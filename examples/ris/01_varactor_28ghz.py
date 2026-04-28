"""RIS #1 — Skyworks SMV1232 varactor RIS at 28 GHz, beam steering sweep."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fresnelants as fa
from fresnelants.cells.varactor import Skyworks_SMV1232

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "img"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cell = Skyworks_SMV1232()
    voltages = np.linspace(0, 15, 64)
    phases = cell.phase(voltages, 28e9)
    losses = 20.0 * np.log10(cell.loss(voltages, 28e9))

    # 1) C-V → phase coverage panel
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(voltages, np.degrees(np.unwrap(phases)), "tab:blue")
    axes[0].set_xlabel("Reverse bias [V]")
    axes[0].set_ylabel("Reflection phase [deg]")
    axes[0].set_title("Skyworks SMV1232 phase vs bias @ 28 GHz")
    axes[1].plot(voltages, losses, "tab:red")
    axes[1].set_xlabel("Reverse bias [V]")
    axes[1].set_ylabel("Reflection loss [dB]")
    axes[1].set_title("Loss vs bias @ 28 GHz")
    fig.tight_layout()
    fig.savefig(OUT / "ris_varactor_cv.png")
    plt.close(fig)

    # 2) Beam-steering sweep at 28 GHz
    ris = fa.ReconfigurableArray(focal_length=0.20, design_freq=28e9, nx=24, ny=24, cell=cell)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2)
    angles = [0, 10, 20, 30, 45]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for theta in angles:
        ris.beam_direction = (math.radians(theta), 0)
        ff = solver.solve(ris, 28e9).far_field
        theta_deg, db = ff.cut("E")
        valid = ~np.isnan(theta_deg)
        ax.plot(theta_deg[valid], db[valid], label=f"θ_b = {theta}°")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-40, 2)
    ax.set_xlabel("θ [deg]")
    ax.set_ylabel("Normalized gain [dB]")
    ax.set_title("Skyworks-SMV1232 RIS 24×24 beam steering at 28 GHz")
    ax.legend(loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "ris_varactor_steering.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
