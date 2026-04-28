"""Solver protocol shared by PO, MoM, and FDTD adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..analysis.aperture import ApertureField
from ..analysis.farfield import FarField
from ..designs.base import AntennaDesign


@dataclass
class SolverResult:
    """Bundle of fields produced by a solver."""

    aperture: ApertureField
    far_field: FarField
    metadata: dict[str, float | str] | None = None


@runtime_checkable
class Solver(Protocol):
    """Common interface: design + frequency → fields."""

    name: str

    def solve(
        self, design: AntennaDesign, freq: float, state: object | None = None
    ) -> SolverResult: ...
