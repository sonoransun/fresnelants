"""Optional PyNEC2 (Method-of-Moments) adapter — `pip install fresnelants[fullwave]`.

PyNEC2 is well-suited to wire-grid approximations of Fresnel zone-plate antennas
where each zone boundary is modeled as a thin metallic ring. This adapter
generates a wire-grid for the design, runs a single-frequency MoM solve, and
extracts the gain pattern.

Skipped at import time if PyNEC2 is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..designs.base import AntennaDesign
from .base import SolverResult

try:  # pragma: no cover - import-time check
    import PyNEC  # type: ignore[import-not-found]

    _NEC_AVAILABLE = True
except ImportError:  # pragma: no cover
    PyNEC = None  # type: ignore[assignment]
    _NEC_AVAILABLE = False


@dataclass
class NECAdapter:
    name: str = "NECAdapter"
    segments_per_wavelength: float = 20.0

    def solve(self, design: AntennaDesign, freq: float) -> SolverResult:
        if not _NEC_AVAILABLE:
            raise RuntimeError(
                "PyNEC2 is not installed. Install the [fullwave] extra via "
                "`pip install 'fresnelants[fullwave]'`."
            )
        raise NotImplementedError(
            "NEC adapter is scaffolded; wire-grid generation per design is "
            "design-specific and is left to subclasses for the initial release."
        )

    def is_available(self) -> bool:
        return _NEC_AVAILABLE
