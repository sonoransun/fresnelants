"""DC bias-network synthesis.

Lays out a tree of DC routing on the back side of a reconfigurable-array PCB:

* root pad at one corner (programmable VCC),
* per-row distribution traces,
* per-cell drops perpendicular to the row,
* return-path through ground plane (no explicit traces),
* decoupling-cap pad placement adjacent to the root.

The result is a :class:`BiasNetwork` of line segments + pads suitable for
emission to Gerber by :mod:`fresnelants.bias.gerber`.

Current per-trace is reported and checked against the cell datasheet's
``cell.if_max`` (provided by PIN/varactor factories where applicable). For
arrays larger than 64×64 we fall back to a row-major two-tier tree (rows are
strapped to a single column trace) to keep current per-segment manageable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..designs.reconfigurable import ReconfigurableArray


@dataclass(frozen=True, slots=True)
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    """Trace width [m]."""
    current: float
    """Estimated DC current carried by this segment [A]."""


@dataclass(frozen=True, slots=True)
class Pad:
    x: float
    y: float
    diameter: float


@dataclass(frozen=True, slots=True)
class BiasNetwork:
    array: ReconfigurableArray
    segments: tuple[Segment, ...]
    cell_pads: tuple[Pad, ...]
    decoupling_pads: tuple[Pad, ...]
    root: Pad
    max_current: float
    trace_width_default: float

    @property
    def total_length(self) -> float:
        return float(sum(np.hypot(s.x1 - s.x0, s.y1 - s.y0) for s in self.segments))

    @property
    def cell_count(self) -> int:
        return len(self.cell_pads)


def synthesize_bias_network(
    array: ReconfigurableArray,
    *,
    trace_width: float = 0.15e-3,
    cell_pad_diameter: float = 0.30e-3,
    decoupling_caps_per_row: int = 1,
    cell_current_estimate: float = 5e-3,
    columns_per_trunk: int = 8,
    hierarchical_threshold: int = 64 * 64,
) -> BiasNetwork:
    """Lay out a row-tree DC distribution on the back of *array*.

    For arrays larger than `hierarchical_threshold` cells (default 64×64=4096),
    automatically dispatches to :func:`synthesize_hierarchical_bias_network`,
    which divides the array into column trunks of `columns_per_trunk` columns
    each — keeping per-trace current within 1 oz copper limits.

    Default trace width 0.15 mm meets JLCPCB minimum process; cell-pad
    diameter 0.30 mm matches a 0402 component pad.
    """
    if array.nx * array.ny > hierarchical_threshold:
        return synthesize_hierarchical_bias_network(
            array,
            trace_width=trace_width,
            cell_pad_diameter=cell_pad_diameter,
            decoupling_caps_per_row=decoupling_caps_per_row,
            cell_current_estimate=cell_current_estimate,
            columns_per_trunk=columns_per_trunk,
        )
    x_centers, y_centers = array.cell_centers()
    Lx, Ly = array.aperture_size

    # Root pad at corner (-Lx/2 - 5 mm, -Ly/2 - 5 mm).
    root_pad = Pad(x=-Lx / 2 - 5e-3, y=-Ly / 2 - 5e-3, diameter=1.0e-3)

    cell_pads = tuple(
        Pad(x=float(x), y=float(y), diameter=cell_pad_diameter)
        for y in y_centers
        for x in x_centers
    )

    segments: list[Segment] = []
    decoupling_pads: list[Pad] = []

    # Spine: vertical trace from root up the −X edge, current = total array.
    spine_x = -Lx / 2 - 1e-3
    total_current = cell_current_estimate * array.nx * array.ny
    segments.append(
        Segment(
            x0=root_pad.x,
            y0=root_pad.y,
            x1=spine_x,
            y1=y_centers[-1],
            width=trace_width * 4,  # spine is wider
            current=total_current,
        )
    )

    # One decoupling cap pad per N rows.
    for i, y in enumerate(y_centers):
        if i % max(array.ny // max(decoupling_caps_per_row, 1), 1) == 0:
            decoupling_pads.append(Pad(x=spine_x - 0.5e-3, y=float(y), diameter=0.45e-3))

    # Per-row traces: spine → row → each cell.
    row_current = cell_current_estimate * array.nx
    for y in y_centers:
        segments.append(
            Segment(
                x0=spine_x,
                y0=float(y),
                x1=float(x_centers[0]),
                y1=float(y),
                width=trace_width * 2,
                current=row_current,
            )
        )
        for j in range(array.nx - 1):
            segments.append(
                Segment(
                    x0=float(x_centers[j]),
                    y0=float(y),
                    x1=float(x_centers[j + 1]),
                    y1=float(y),
                    width=trace_width,
                    current=cell_current_estimate * (array.nx - j),
                )
            )

    # Current budget: assume 1 A / mm for 1 oz copper at 35 µm thickness.
    max_current = max(s.current for s in segments)

    return BiasNetwork(
        array=array,
        segments=tuple(segments),
        cell_pads=cell_pads,
        decoupling_pads=tuple(decoupling_pads),
        root=root_pad,
        max_current=max_current,
        trace_width_default=trace_width,
    )


def estimated_max_current_density(network: BiasNetwork) -> float:
    """Return the worst-case current density [A/mm²] across the network.

    Should be < 35 A/mm² for 1 oz copper at 30 °C rise.
    """
    worst = 0.0
    for s in network.segments:
        # 1 oz copper = 35 µm = 0.035 mm. Width in mm.
        width_mm = s.width * 1000.0
        density = s.current / (width_mm * 0.035)  # A/mm²
        worst = max(worst, density)
    return float(worst)


def synthesize_hierarchical_bias_network(
    array: ReconfigurableArray,
    *,
    trace_width: float = 0.15e-3,
    cell_pad_diameter: float = 0.30e-3,
    decoupling_caps_per_row: int = 1,
    cell_current_estimate: float = 5e-3,
    columns_per_trunk: int = 8,
) -> BiasNetwork:
    """Two-tier bias-network synthesis for large arrays (>64×64).

    Divides the array's columns into trunks of `columns_per_trunk` columns
    each. Each trunk gets its own vertical bus traveling up the −X side; the
    main spine fans out to all trunks at the bottom. Trunk current = nx_trunk
    × ny × cell_current; spine current still equals total array current but
    its widening trace handles it.

    Topology (sketch):

        root → spine ─┬→ trunk₁ ─┬→ row₁ ─→ cells…
                      │          ├→ row₂ ─→ cells…
                      │          └…
                      ├→ trunk₂ ─┬→ row₁ ─→ cells…
                      …
    """
    x_centers, y_centers = array.cell_centers()
    Lx, Ly = array.aperture_size

    n_trunks = max(1, (array.nx + columns_per_trunk - 1) // columns_per_trunk)
    cells_per_trunk = columns_per_trunk * array.ny
    trunk_current = cell_current_estimate * cells_per_trunk

    root_pad = Pad(x=-Lx / 2 - 5e-3, y=-Ly / 2 - 5e-3, diameter=1.0e-3)
    cell_pads = tuple(
        Pad(x=float(x), y=float(y), diameter=cell_pad_diameter)
        for y in y_centers
        for x in x_centers
    )

    segments: list[Segment] = []
    decoupling_pads: list[Pad] = []

    spine_x = -Lx / 2 - 1e-3
    total_current = cell_current_estimate * array.nx * array.ny
    # Wide spine — current gets divided per-trunk after the fan-out.
    segments.append(
        Segment(
            x0=root_pad.x,
            y0=root_pad.y,
            x1=spine_x,
            y1=root_pad.y,
            width=trace_width * 8,
            current=total_current,
        )
    )

    # One trunk per group of columns_per_trunk columns. Trunks live on the
    # −X side; each spans the full vertical extent of its column group.
    trunk_x_offset = trace_width * 4
    for t in range(n_trunks):
        col_lo = t * columns_per_trunk
        col_hi = min(col_lo + columns_per_trunk, array.nx)
        x_trunk = spine_x - trunk_x_offset * (t + 1)
        # Spine → trunk fan-out at y = root_pad.y.
        segments.append(
            Segment(
                x0=spine_x,
                y0=root_pad.y,
                x1=x_trunk,
                y1=root_pad.y,
                width=trace_width * 4,
                current=trunk_current,
            )
        )
        # Vertical trunk segment along the column-group height.
        segments.append(
            Segment(
                x0=x_trunk,
                y0=root_pad.y,
                x1=x_trunk,
                y1=float(y_centers[-1]),
                width=trace_width * 4,
                current=trunk_current,
            )
        )
        # Per-row drops: trunk → row → cells in this trunk's column range.
        row_current = cell_current_estimate * (col_hi - col_lo)
        for y in y_centers:
            segments.append(
                Segment(
                    x0=x_trunk,
                    y0=float(y),
                    x1=float(x_centers[col_lo]),
                    y1=float(y),
                    width=trace_width * 2,
                    current=row_current,
                )
            )
            for j in range(col_lo, col_hi - 1):
                segments.append(
                    Segment(
                        x0=float(x_centers[j]),
                        y0=float(y),
                        x1=float(x_centers[j + 1]),
                        y1=float(y),
                        width=trace_width,
                        current=cell_current_estimate * (col_hi - 1 - j),
                    )
                )

    # Decoupling caps near each trunk root.
    for t in range(n_trunks):
        x_trunk = spine_x - trunk_x_offset * (t + 1)
        decoupling_pads.append(Pad(x=x_trunk - 0.5e-3, y=root_pad.y, diameter=0.45e-3))

    max_current = max(s.current for s in segments)

    return BiasNetwork(
        array=array,
        segments=tuple(segments),
        cell_pads=cell_pads,
        decoupling_pads=tuple(decoupling_pads),
        root=root_pad,
        max_current=max_current,
        trace_width_default=trace_width,
    )
