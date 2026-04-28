"""RIS #3 — Bias-network synthesis + Gerber export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import fresnelants as fa
from fresnelants.bias.gerber import write_bias_layers
from fresnelants.bias.network import (
    estimated_max_current_density,
    synthesize_bias_network,
)
from fresnelants.cells.varactor import Skyworks_SMV1232

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "img"
GERBER_OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "gerber"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GERBER_OUT.mkdir(parents=True, exist_ok=True)
    cell = Skyworks_SMV1232()
    ris = fa.ReconfigurableArray(focal_length=0.20, design_freq=28e9, nx=16, ny=16, cell=cell)
    network = synthesize_bias_network(ris, cell_current_estimate=2e-3)
    print(f"Total trace length: {network.total_length * 1e3:.1f} mm")
    print(f"Cell count: {network.cell_count}")
    print(f"Max current density: {estimated_max_current_density(network):.2f} A/mm²")

    paths = write_bias_layers(network, GERBER_OUT, base_name="ris_28ghz_bias")
    for label, p in paths.items():
        print(f"  {label}: {p.relative_to(Path(__file__).resolve().parent.parent.parent)}")

    # Visualize bias network on top of the cell grid.
    fig, ax = plt.subplots(figsize=(7, 7))
    for s in network.segments:
        ax.plot(
            [s.x0 * 1e3, s.x1 * 1e3],
            [s.y0 * 1e3, s.y1 * 1e3],
            color="tab:orange",
            linewidth=s.width * 5e3,
        )
    for p in network.cell_pads:
        ax.add_patch(plt.Circle((p.x * 1e3, p.y * 1e3), 0.5, color="tab:blue"))
    ax.add_patch(
        plt.Circle((network.root.x * 1e3, network.root.y * 1e3), 1.0, color="tab:red", label="VCC")
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("RIS bias network — 16×16 SMV1232 array")
    fig.tight_layout()
    fig.savefig(OUT / "ris_bias_network.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
