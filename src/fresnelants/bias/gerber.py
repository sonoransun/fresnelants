"""Emit a `BiasNetwork` to RS-274X Gerber + Excellon files."""

from __future__ import annotations

from pathlib import Path

from .network import BiasNetwork


def write_bias_layers(
    network: BiasNetwork,
    out_dir: str | Path,
    *,
    base_name: str = "bias",
) -> dict[str, Path]:
    """Write back-side copper, mask, and drill files for the bias network.

    Files produced:

    * ``<base>-B.Cu.gbr`` — back-side copper (traces + pads)
    * ``<base>-B.Mask.gbr`` — back-side soldermask cutouts on pads
    * ``<base>-B.Drill.drl`` — placeholder drill file
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cu = out / f"{base_name}-B.Cu.gbr"
    mask = out / f"{base_name}-B.Mask.gbr"
    drill = out / f"{base_name}-B.Drill.drl"

    _write_copper(cu, network)
    _write_mask(mask, network)
    _write_drill(drill, network)
    return {"copper": cu, "mask": mask, "drill": drill}


def _write_copper(path: Path, network: BiasNetwork) -> None:
    with path.open("w") as f:
        f.write("G04 FresnelAnts bias network — back-side copper*\n")
        f.write("%FSLAX36Y36*%\n")
        f.write("%MOMM*%\n")
        f.write("%LPD*%\n")

        # Trace apertures (one per unique width).
        widths = sorted({round(s.width * 1000.0, 4) for s in network.segments})
        trace_codes: dict[float, int] = {w: 10 + i for i, w in enumerate(widths)}
        for w_mm, code in trace_codes.items():
            f.write(f"%ADD{code}C,{w_mm:.4f}*%\n")

        # Pad apertures.
        pad_widths = sorted(
            {round(p.diameter * 1000.0, 4) for p in (*network.cell_pads, *network.decoupling_pads)}
        )
        pad_codes: dict[float, int] = {w: 100 + i for i, w in enumerate(pad_widths)}
        for w_mm, code in pad_codes.items():
            f.write(f"%ADD{code}C,{w_mm:.4f}*%\n")

        # Root pad.
        root_code = 200
        f.write(f"%ADD{root_code}C,{network.root.diameter * 1000.0:.4f}*%\n")

        # Trace draws.
        for s in network.segments:
            w_mm = round(s.width * 1000.0, 4)
            code = trace_codes[w_mm]
            f.write(f"D{code}*\n")
            f.write(_xy(s.x0, s.y0) + "D02*\n")
            f.write(_xy(s.x1, s.y1) + "D01*\n")

        # Cell pads.
        for p in network.cell_pads:
            w_mm = round(p.diameter * 1000.0, 4)
            f.write(f"D{pad_codes[w_mm]}*\n")
            f.write(_xy(p.x, p.y) + "D03*\n")

        # Decoupling pads.
        for p in network.decoupling_pads:
            w_mm = round(p.diameter * 1000.0, 4)
            f.write(f"D{pad_codes[w_mm]}*\n")
            f.write(_xy(p.x, p.y) + "D03*\n")

        # Root.
        f.write(f"D{root_code}*\n")
        f.write(_xy(network.root.x, network.root.y) + "D03*\n")
        f.write("M02*\n")


def _write_mask(path: Path, network: BiasNetwork) -> None:
    with path.open("w") as f:
        f.write("G04 FresnelAnts bias network — back-side soldermask cutouts*\n")
        f.write("%FSLAX36Y36*%\n")
        f.write("%MOMM*%\n")
        f.write("%LPD*%\n")
        # Open mask = pads grown by 0.10 mm.
        clearance = 0.10
        diameters = sorted(
            {
                round(p.diameter * 1000.0 + 2 * clearance, 4)
                for p in (*network.cell_pads, *network.decoupling_pads, network.root)
            }
        )
        codes = {d: 10 + i for i, d in enumerate(diameters)}
        for d, code in codes.items():
            f.write(f"%ADD{code}C,{d:.4f}*%\n")
        for p in (*network.cell_pads, *network.decoupling_pads, network.root):
            d = round(p.diameter * 1000.0 + 2 * clearance, 4)
            f.write(f"D{codes[d]}*\n")
            f.write(_xy(p.x, p.y) + "D03*\n")
        f.write("M02*\n")


def _write_drill(path: Path, network: BiasNetwork) -> None:
    with path.open("w") as f:
        f.write("M48\n")
        f.write("FMAT,2\nMETRIC,LZ,000.000\n")
        f.write("T1C0.500\n%\n")
        f.write("T1\n")
        # Drill at the root pad as the VCC entry point.
        f.write(f"X{round(network.root.x * 1_000_000)}Y{round(network.root.y * 1_000_000)}\n")
        f.write("M30\n")


def _xy(x_m: float, y_m: float) -> str:
    """Format X/Y coordinates in 3.6 fixed-point mm."""
    xi = round(x_m * 1_000_000_000.0 / 1000.0)
    yi = round(y_m * 1_000_000_000.0 / 1000.0)
    return f"X{xi}Y{yi}"
