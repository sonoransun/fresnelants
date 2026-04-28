"""Cascade-PO solver for :class:`fresnelants.designs.composite.CompositeAntenna`.

Generates the emergent aperture by chaining forward-propagation through each
layer (the composite class itself does the work), then runs the standard
2-D-FFT far-field on the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.farfield import far_field_from_aperture
from ..designs.base import AntennaDesign
from ..designs.composite import CompositeAntenna
from .base import SolverResult


@dataclass
class CascadePOSolver:
    name: str = "CascadePOSolver"
    samples_per_wavelength: float = 6.0
    margin: float = 1.2
    pad_factor: int = 4

    def solve(
        self, design: AntennaDesign, freq: float, state: object | None = None
    ) -> SolverResult:
        if not isinstance(design, CompositeAntenna):
            raise TypeError(
                f"CascadePOSolver requires a CompositeAntenna, got {type(design).__name__}"
            )
        aperture = design.aperture_field(
            freq,
            samples_per_wavelength=self.samples_per_wavelength,
            margin=self.margin,
            state=state,
        )
        ff = far_field_from_aperture(aperture, pad_factor=self.pad_factor)
        return SolverResult(
            aperture=aperture, far_field=ff, metadata={"solver": self.name, "freq": freq}
        )
