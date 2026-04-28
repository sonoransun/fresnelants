"""Pydantic v2 specs for declarative antenna definition (YAML / JSON)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..core.materials import HDPE, LIBRARY, Dielectric
from ..designs.base import AntennaDesign
from ..designs.curvilinear import CurvilinearFresnel, ProfileKind
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


DesignSpec = (
    SoretSpec | WoodSpec | OffsetSpec | PhaseCorrectingSpec | ReflectarraySpec | CurvilinearSpec
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
    return spec.build()  # type: ignore[no-any-return]
