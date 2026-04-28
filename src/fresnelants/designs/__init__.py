"""Fresnel antenna families."""

from .base import AntennaDesign, DesignResult
from .composite import (
    AchromaticDoublet,
    BifocalLens,
    CompositeAntenna,
    FoldedReflectarray,
    Layer,
)
from .conformal import CylindricalFresnelLens, SphericalFresnelLens
from .curvilinear import CurvilinearFresnel
from .metasurface import DualPolSharedAperture, MetasurfaceLens
from .offset import OffsetZonePlate
from .phase_correcting import PhaseCorrectingPlate
from .reconfigurable import CodedRIS, ReconfigurableArray
from .reflectarray import Reflectarray
from .zone_plate import SoretZonePlate, WoodZonePlate

__all__ = [
    "AchromaticDoublet",
    "AntennaDesign",
    "BifocalLens",
    "CodedRIS",
    "CompositeAntenna",
    "CurvilinearFresnel",
    "CylindricalFresnelLens",
    "DesignResult",
    "DualPolSharedAperture",
    "FoldedReflectarray",
    "Layer",
    "MetasurfaceLens",
    "OffsetZonePlate",
    "PhaseCorrectingPlate",
    "ReconfigurableArray",
    "Reflectarray",
    "SoretZonePlate",
    "SphericalFresnelLens",
    "WoodZonePlate",
]
