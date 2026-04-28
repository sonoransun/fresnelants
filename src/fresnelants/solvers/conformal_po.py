"""Conformal-PO solver: direct integration over a curved aperture mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..analysis.conformal_farfield import ConformalFarField, far_field_from_conformal
from ..core.conformal import ConformalAperture
from ..designs.base import AntennaDesign


@dataclass
class ConformalSolverResult:
    aperture: ConformalAperture
    far_field: ConformalFarField
    metadata: dict[str, Any] | None = None


@dataclass
class ConformalPOSolver:
    name: str = "ConformalPOSolver"
    n_samples: int = 128
    chunk: int = 4096

    def solve(
        self, design: AntennaDesign, freq: float, state: object | None = None
    ) -> ConformalSolverResult:
        if not hasattr(design, "conformal_aperture"):
            raise TypeError(
                f"ConformalPOSolver requires a design with conformal_aperture(), "
                f"got {type(design).__name__}"
            )
        aperture: ConformalAperture = design.conformal_aperture(freq, state=state)
        ff = far_field_from_conformal(aperture, n_samples=self.n_samples, chunk=self.chunk)
        return ConformalSolverResult(
            aperture=aperture, far_field=ff, metadata={"solver": self.name, "freq": freq}
        )
