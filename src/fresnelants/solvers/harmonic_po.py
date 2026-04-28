"""Harmonic-PO solver — runs the PO engine across a list of harmonics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..analysis.farfield import FarField
from ..analysis.harmonics import harmonic_far_field
from ..designs.base import AntennaDesign


@dataclass
class HarmonicSolverResult:
    """Result of a HarmonicPOSolver — one FarField per requested harmonic."""

    harmonics: list[int]
    far_fields: list[FarField]


@dataclass
class HarmonicPOSolver:
    name: str = "HarmonicPOSolver"
    samples_per_wavelength: float = 6.0
    pad_factor: int = 4

    def solve(
        self,
        design: AntennaDesign,
        freq: float,
        *,
        harmonics: Sequence[int] = (-1, 0, 1),
    ) -> HarmonicSolverResult:
        """Compute far-field at each requested harmonic of *design*."""
        if not hasattr(design, "harmonic_coefficient"):
            raise TypeError(
                f"HarmonicPOSolver requires a TimeModulatedArray; got {type(design).__name__}"
            )
        ffs = [
            harmonic_far_field(
                design,
                freq,
                n,
                samples_per_wavelength=self.samples_per_wavelength,
                pad_factor=self.pad_factor,
            )
            for n in harmonics
        ]
        return HarmonicSolverResult(harmonics=list(harmonics), far_fields=ffs)
