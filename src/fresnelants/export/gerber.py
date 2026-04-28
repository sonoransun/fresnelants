"""Gerber RS-274X export for reflectarrays.

Each cell is rendered as a square copper patch whose side length encodes the
required reflection phase via a parametric size–phase curve. The mapping used
here is the canonical *square-patch* model: phase grows roughly linearly with
patch size between λ/4 and λ/2, with monotonic regions before and after the
resonance. We pick the patch length s(φ) ∈ [s_min, s_max] that best
approximates the required phase from a precomputed lookup.

Outputs:
* `<name>-F.Cu.gbr` — copper patches
* `<name>-F.Mask.gbr` — soldermask cutouts
* `<name>.drl` — drill file (empty placeholder)

The format used is the most-portable RS-274X subset (G04 comments, %FSLA, %MOMM,
flash with rectangle aperture). Most board houses (JLCPCB, OSH Park, etc.)
accept this directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..designs.reflectarray import Reflectarray
from ..units import freq_to_wavelength

# Empirically reasonable square-patch size ↔ phase curve. Real designs need to
# fit to full-wave element data, but this default gives a sensible Gerber that
# can be regenerated with a tuned curve later.
_PATCH_PHASES_RAD = np.array([0.0, 0.6, 1.5, 2.6, 3.7, 4.5, 5.2, 5.7, 2 * np.pi])
_PATCH_SIZES_NORM = np.array([0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62])


def _phase_to_size(phase: NDArray[np.float64], wavelength: float) -> NDArray[np.float64]:
    """Map required phase [0, 2π) to a patch side length [m]."""
    phase = np.mod(phase, 2.0 * np.pi)
    return np.interp(phase, _PATCH_PHASES_RAD, _PATCH_SIZES_NORM) * wavelength


def write_reflectarray_gerber(
    design: Reflectarray,
    out_dir: str | Path,
    *,
    base_name: str = "reflectarray",
    freq: float | None = None,
) -> dict[str, Path]:
    """Write copper / soldermask / drill files. Returns a dict of paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = freq if freq is not None else design.design_freq
    lam = float(freq_to_wavelength(f))

    phase = design.required_phase_per_cell(f)
    size = _phase_to_size(phase, lam)  # m
    x_centers, y_centers = design.cell_centers()

    files = {
        "copper": out / f"{base_name}-F.Cu.gbr",
        "mask": out / f"{base_name}-F.Mask.gbr",
        "drill": out / f"{base_name}.drl",
    }

    _write_gerber_layer(files["copper"], x_centers, y_centers, size, layer="Copper")
    # Soldermask: 0.1 mm clearance around each patch.
    _write_gerber_layer(
        files["mask"],
        x_centers,
        y_centers,
        size + 0.2e-3,
        layer="SolderMask",
    )
    _write_excellon_drill(files["drill"])
    return files


def _write_gerber_layer(
    path: Path,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    sizes_2d: NDArray[np.float64],
    *,
    layer: str,
) -> None:
    # Group cells by size to reuse one aperture per unique size (rounded to 1 µm).
    flat_sizes = sizes_2d.flatten() * 1000.0  # mm
    flat_x_idx, flat_y_idx = np.meshgrid(np.arange(xs.size), np.arange(ys.size), indexing="xy")
    flat_x = xs[flat_x_idx.flatten()] * 1000.0
    flat_y = ys[flat_y_idx.flatten()] * 1000.0

    rounded = np.round(flat_sizes, 3)
    unique = np.unique(rounded)
    aperture_codes: dict[float, int] = {s: 10 + i for i, s in enumerate(unique)}

    with path.open("w") as f:
        f.write("G04 FresnelAnts reflectarray Gerber*\n")
        f.write(f"G04 layer={layer}*\n")
        f.write("%FSLAX36Y36*%\n")  # 3 integer / 6 fractional digits
        f.write("%MOMM*%\n")  # millimeters
        f.write("%LPD*%\n")  # dark polarity
        for size, code in aperture_codes.items():
            f.write(f"%ADD{code}R,{size:.4f}X{size:.4f}*%\n")
        last_code = -1
        for x_mm, y_mm, s_mm in zip(flat_x, flat_y, rounded, strict=True):
            code = aperture_codes[s_mm]
            if code != last_code:
                f.write(f"D{code}*\n")
                last_code = code
            xi = round(x_mm * 1_000_000)
            yi = round(y_mm * 1_000_000)
            f.write(f"X{xi}Y{yi}D03*\n")
        f.write("M02*\n")


def _write_excellon_drill(path: Path) -> None:
    with path.open("w") as f:
        f.write("M48\n")  # header
        f.write("FMAT,2\nMETRIC,LZ,000.000\n")
        f.write("%\n")
        f.write("M30\n")
