"""Pydantic v2 specs for declarative antenna definition (YAML / JSON)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..core.materials import HDPE, LIBRARY, Dielectric
from ..designs.base import AntennaDesign
from ..designs.curvilinear import CurvilinearFresnel, ProfileKind
from ..designs.fractal import (
    ConicalFractalFresnelLens,
    FractalSoretZonePlate,
    FractalWoodZonePlate,
    SierpinskiCarpetZonePlate,
    SierpinskiReflectarray,
    SphericalFractalFresnelLens,
)
from ..designs.macro_array import MacroFresnelArray
from ..designs.offset import OffsetZonePlate
from ..designs.phase_correcting import PhaseCorrectingPlate
from ..designs.reflectarray import Reflectarray
from ..designs.zone_plate import SoretZonePlate, WoodZonePlate


class _Base(BaseModel):
    model_config = {"extra": "forbid"}

    focal_length: float = Field(..., gt=0, description="Focal length [m].")
    design_freq: float = Field(..., gt=0, description="Design frequency [Hz].")


class SoretSpec(_Base):
    kind: Literal["soret"] = "soret"
    num_zones: int = Field(12, ge=1)
    odd_transparent: bool = True

    def build(self) -> SoretZonePlate:
        return SoretZonePlate(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            num_zones=self.num_zones,
            odd_transparent=self.odd_transparent,
        )


class WoodSpec(_Base):
    kind: Literal["wood"] = "wood"
    num_zones: int = Field(12, ge=1)

    def build(self) -> WoodZonePlate:
        return WoodZonePlate(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            num_zones=self.num_zones,
        )


class OffsetSpec(_Base):
    kind: Literal["offset"] = "offset"
    tilt_angle_deg: float = 20.0
    aperture_radius: float = Field(0.10, gt=0)

    def build(self) -> OffsetZonePlate:
        import math

        return OffsetZonePlate(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            tilt_angle=math.radians(self.tilt_angle_deg),
            aperture_radius_m=self.aperture_radius,
        )


class PhaseCorrectingSpec(_Base):
    kind: Literal["phase_correcting"] = "phase_correcting"
    aperture_radius: float = Field(0.10, gt=0)
    levels: int = Field(4, ge=1)
    dielectric: str = "HDPE"

    def build(self) -> PhaseCorrectingPlate:
        return PhaseCorrectingPlate(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            aperture_radius_m=self.aperture_radius,
            levels=self.levels,
            dielectric=_resolve_dielectric(self.dielectric),
        )


class ReflectarraySpec(_Base):
    kind: Literal["reflectarray"] = "reflectarray"
    nx: int = Field(32, ge=1)
    ny: int = Field(32, ge=1)
    cell_size: float = Field(0.0, ge=0)
    feed_offset: tuple[float, float] = (0.0, 0.0)
    beam_theta_deg: float = 0.0
    beam_phi_deg: float = 0.0
    feed_q: float = 6.0
    phase_levels: int | None = None

    def build(self) -> Reflectarray:
        import math

        return Reflectarray(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            nx=self.nx,
            ny=self.ny,
            cell_size=self.cell_size,
            feed_offset=self.feed_offset,
            beam_direction=(math.radians(self.beam_theta_deg), math.radians(self.beam_phi_deg)),
            feed_q=self.feed_q,
            phase_levels=self.phase_levels,
        )


class CurvilinearSpec(_Base):
    kind: Literal["curvilinear"] = "curvilinear"
    aperture_radius: float = Field(0.10, gt=0)
    profile: ProfileKind = "hyperbolic"
    dielectric: str = "HDPE"
    axicon_angle_deg: float = 5.0

    @field_validator("profile")
    @classmethod
    def _check_profile(cls, v: str) -> str:
        if v not in ("hyperbolic", "axicon", "freeform"):
            raise ValueError(f"unknown profile {v!r}")
        return v

    def build(self) -> CurvilinearFresnel:
        import math

        if self.profile == "freeform":
            raise ValueError("Freeform profiles must be built in code, not YAML.")
        return CurvilinearFresnel(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            aperture_radius_m=self.aperture_radius,
            profile=self.profile,
            dielectric=_resolve_dielectric(self.dielectric),
            axicon_angle=math.radians(self.axicon_angle_deg),
        )


class FractalSoretSpec(_Base):
    kind: Literal["fractal_soret"] = "fractal_soret"
    stage: int = Field(3, ge=0, le=6)
    base_unit: int = Field(1, ge=1)

    def build(self) -> FractalSoretZonePlate:
        return FractalSoretZonePlate(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            stage=self.stage,
            base_unit=self.base_unit,
        )


class FractalWoodSpec(_Base):
    kind: Literal["fractal_wood"] = "fractal_wood"
    stage: int = Field(3, ge=0, le=6)
    base_unit: int = Field(1, ge=1)

    def build(self) -> FractalWoodZonePlate:
        return FractalWoodZonePlate(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            stage=self.stage,
            base_unit=self.base_unit,
        )


class SierpinskiSpec(_Base):
    kind: Literal["sierpinski"] = "sierpinski"
    stage: int = Field(3, ge=0, le=5)
    aperture_side: float = Field(0.20, gt=0)

    def build(self) -> SierpinskiCarpetZonePlate:
        return SierpinskiCarpetZonePlate(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            stage=self.stage,
            aperture_side=self.aperture_side,
        )


class SierpinskiReflectarraySpec(_Base):
    kind: Literal["sierpinski_reflectarray"] = "sierpinski_reflectarray"
    nx: int = Field(27, ge=1)
    ny: int = Field(27, ge=1)
    cell_size: float = Field(0.0, ge=0)
    fractal_stage: int = Field(2, ge=0, le=4)
    feed_offset: tuple[float, float] = (0.0, 0.0)
    beam_theta_deg: float = 0.0
    beam_phi_deg: float = 0.0
    feed_q: float = 6.0

    def build(self) -> SierpinskiReflectarray:
        import math

        return SierpinskiReflectarray(
            focal_length=self.focal_length,
            design_freq=self.design_freq,
            nx=self.nx,
            ny=self.ny,
            cell_size=self.cell_size,
            feed_offset=self.feed_offset,
            beam_direction=(math.radians(self.beam_theta_deg), math.radians(self.beam_phi_deg)),
            feed_q=self.feed_q,
            fractal_stage=self.fractal_stage,
        )


class SphericalFractalSpec(BaseModel):
    model_config = {"extra": "forbid"}
    kind: Literal["spherical_fractal"] = "spherical_fractal"
    design_freq: float = Field(..., gt=0)
    radius: float = Field(0.05, gt=0)
    cap_angle_deg: float = Field(90.0, gt=0, le=180)
    stage: int = Field(2, ge=0, le=4)
    base_unit: int = Field(1, ge=1)
    phase_reversal: bool = True
    nu: int = Field(128, ge=4)
    nv: int = Field(40, ge=4)

    def build(self) -> SphericalFractalFresnelLens:
        return SphericalFractalFresnelLens(
            radius=self.radius,
            cap_angle_deg=self.cap_angle_deg,
            design_freq=self.design_freq,
            stage=self.stage,
            base_unit=self.base_unit,
            phase_reversal=self.phase_reversal,
            nu=self.nu,
            nv=self.nv,
        )


class ConicalFractalSpec(BaseModel):
    model_config = {"extra": "forbid"}
    kind: Literal["conical_fractal"] = "conical_fractal"
    design_freq: float = Field(..., gt=0)
    half_angle_deg: float = Field(45.0, gt=0, lt=90)
    height: float = Field(0.05, gt=0)
    stage: int = Field(2, ge=0, le=4)
    base_unit: int = Field(1, ge=1)
    phase_reversal: bool = True
    nu: int = Field(128, ge=4)
    nv: int = Field(60, ge=4)

    def build(self) -> ConicalFractalFresnelLens:
        return ConicalFractalFresnelLens(
            half_angle_deg=self.half_angle_deg,
            height=self.height,
            design_freq=self.design_freq,
            stage=self.stage,
            base_unit=self.base_unit,
            phase_reversal=self.phase_reversal,
            nu=self.nu,
            nv=self.nv,
        )


class MacroArraySpec(BaseModel):
    model_config = {"extra": "forbid"}
    kind: Literal["macro_array"] = "macro_array"
    element: dict[str, Any] = Field(
        ..., description="Nested design spec dict (with its own 'kind') for the prototype element."
    )
    lattice: Literal["linear", "rect", "hex", "ring"] = "linear"
    n_elements: int = Field(..., ge=1)
    spacing_m: float = Field(..., gt=0)
    rows: int | None = Field(None, ge=1, description="Row count for 'rect' lattice; auto if None.")
    bits: int = Field(
        0, ge=0, le=8, description="0=continuous; otherwise quantize to 2^bits phases."
    )
    coupling_q: float = Field(0.0, ge=0.0)
    beam_theta_deg: float = 0.0
    beam_phi_deg: float = 0.0

    def build(self) -> MacroFresnelArray:
        from ..core.geometry import element_lattice_positions

        # Build the prototype element via spec_from_dict (recursive dispatch).
        element = spec_from_dict(self.element)
        positions = element_lattice_positions(
            self.n_elements,
            self.spacing_m,
            self.lattice,
            rows=self.rows,
        )
        macro = MacroFresnelArray(
            element=element,
            element_positions=positions,
            coupling_q=self.coupling_q,
        )
        # Pre-compute weights for the requested beam direction.
        macro.weights = macro.weights_for_beam(
            self.beam_theta_deg,
            self.beam_phi_deg,
            bits=self.bits,
        )
        return macro


DesignSpec = (
    SoretSpec
    | WoodSpec
    | OffsetSpec
    | PhaseCorrectingSpec
    | ReflectarraySpec
    | CurvilinearSpec
    | FractalSoretSpec
    | FractalWoodSpec
    | SierpinskiSpec
    | SierpinskiReflectarraySpec
    | SphericalFractalSpec
    | ConicalFractalSpec
    | MacroArraySpec
)


def _resolve_dielectric(name: str) -> Dielectric:
    if name in LIBRARY:
        return LIBRARY[name]
    # Allow short aliases ("HDPE", "PTFE", …).
    for full, mat in LIBRARY.items():
        if full.split()[0].upper() == name.upper():
            return mat
    return HDPE


_REGISTRY: dict[str, type[BaseModel]] = {
    "soret": SoretSpec,
    "wood": WoodSpec,
    "offset": OffsetSpec,
    "phase_correcting": PhaseCorrectingSpec,
    "reflectarray": ReflectarraySpec,
    "curvilinear": CurvilinearSpec,
    "fractal_soret": FractalSoretSpec,
    "fractal_wood": FractalWoodSpec,
    "sierpinski": SierpinskiSpec,
    "sierpinski_reflectarray": SierpinskiReflectarraySpec,
    "spherical_fractal": SphericalFractalSpec,
    "conical_fractal": ConicalFractalSpec,
    "macro_array": MacroArraySpec,
}


def spec_from_dict(data: dict[str, Any]) -> AntennaDesign:
    """Dispatch a spec dict (`{"kind": "...", ...}`) to the right design."""
    if "kind" not in data:
        raise ValueError("Spec dict must include a 'kind' field.")
    kind = data["kind"]
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown design kind {kind!r}")
    spec_cls = _REGISTRY[kind]
    spec = spec_cls.model_validate(data)
    return spec.build()  # type: ignore[attr-defined,no-any-return]
