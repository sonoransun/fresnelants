"""FresnelAnts — Fresnel antenna design system.

Public API re-exports the most common symbols. Submodules expose the full set.
"""

from __future__ import annotations

from ._version import __version__
from .core.geometry import fresnel_zone_radii, zone_radius
from .core.wavefront import PlaneWave, SphericalWave
from .designs.base import AntennaDesign, DesignResult
from .designs.composite import (
    AchromaticDoublet,
    BifocalLens,
    CompositeAntenna,
    FoldedReflectarray,
    Layer,
)
from .designs.conformal import CylindricalFresnelLens, SphericalFresnelLens
from .designs.curvilinear import CurvilinearFresnel
from .designs.metasurface import DualPolSharedAperture, MetasurfaceLens
from .designs.offset import OffsetZonePlate
from .designs.phase_correcting import PhaseCorrectingPlate
from .designs.reconfigurable import CodedRIS, ReconfigurableArray
from .designs.reflectarray import Reflectarray
from .designs.time_modulated import TimeModulatedArray
from .designs.zone_plate import SoretZonePlate, WoodZonePlate
from .solvers.cascade_po import CascadePOSolver
from .solvers.conformal_po import ConformalPOSolver
from .solvers.harmonic_po import HarmonicPOSolver
from .solvers.physical_optics import PhysicalOpticsSolver
from .units import c0, freq_to_wavelength, k0, wavelength_to_freq

__all__ = [
    "AchromaticDoublet",
    "AntennaDesign",
    "BifocalLens",
    "CascadePOSolver",
    "CodedRIS",
    "CompositeAntenna",
    "ConformalPOSolver",
    "CurvilinearFresnel",
    "CylindricalFresnelLens",
    "DesignResult",
    "DualPolSharedAperture",
    "FoldedReflectarray",
    "HarmonicPOSolver",
    "Layer",
    "MetasurfaceLens",
    "OffsetZonePlate",
    "PhaseCorrectingPlate",
    "PhysicalOpticsSolver",
    "PlaneWave",
    "ReconfigurableArray",
    "Reflectarray",
    "SoretZonePlate",
    "SphericalFresnelLens",
    "SphericalWave",
    "TimeModulatedArray",
    "WoodZonePlate",
    "__version__",
    "c0",
    "freq_to_wavelength",
    "fresnel_zone_radii",
    "k0",
    "wavelength_to_freq",
    "zone_radius",
]
