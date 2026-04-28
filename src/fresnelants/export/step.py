"""STEP export — gated on the optional `cadquery` extra."""

from __future__ import annotations

from pathlib import Path

from ..designs.base import AntennaDesign
from ..designs.curvilinear import CurvilinearFresnel
from ..designs.phase_correcting import PhaseCorrectingPlate

try:  # pragma: no cover
    import cadquery as cq  # type: ignore[import-not-found]

    _CQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    cq = None  # type: ignore[assignment]
    _CQ_AVAILABLE = False


def export_step(design: AntennaDesign, path: str | Path) -> Path:
    """Write a STEP file for *design*. Requires `pip install fresnelants[cad]`."""
    if not _CQ_AVAILABLE:
        raise RuntimeError(
            "cadquery is not installed. Install the [cad] extra: `pip install 'fresnelants[cad]'`."
        )
    if not isinstance(design, (PhaseCorrectingPlate, CurvilinearFresnel)):
        raise TypeError(
            f"STEP export not implemented for {type(design).__name__}. "
            "Use `export_stl` for zone plates / reflectarrays."
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Build a revolved solid from the radial depth profile.
    import numpy as np

    R = np.linspace(0.0, design.aperture_radius, 256)
    if isinstance(design, CurvilinearFresnel):
        depth = design.fresnel_depth(R)
    else:
        from ..core.geometry import make_aperture_grid
        from ..units import freq_to_wavelength

        lam = float(freq_to_wavelength(design.design_freq))
        grid = make_aperture_grid(2.05 * design.aperture_radius, 4.0, lam)
        depth_grid = design.groove_depths(grid)
        # sample on x-axis
        ix = np.argmin(np.abs(grid.y))
        depth = np.interp(R, grid.x[grid.x >= 0], depth_grid[ix, grid.x >= 0])

    base_thickness = max(0.5e-3, 0.05 * design.aperture_radius)
    pts = [(0.0, 0.0)]
    for r, d in zip(R, depth, strict=True):
        pts.append((float(r), float(base_thickness + d)))
    pts.append((float(design.aperture_radius), 0.0))

    profile = cq.Workplane("XZ").polyline(pts).close()
    solid = profile.revolve(360, (0, 0, 0), (0, 0, 1))
    cq.exporters.export(solid, str(out))
    return out
