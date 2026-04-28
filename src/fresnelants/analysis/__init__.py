"""Physical-optics analysis: aperture, near-field, far-field, metrics."""

from .aperture import ApertureField, JonesApertureField
from .dualpol_metrics import (
    axial_ratio_db,
    cross_polarization_db,
    polarization_purity,
)
from .farfield import (
    FarField,
    JonesFarField,
    far_field_from_aperture,
    far_field_from_jones_aperture,
)
from .metrics import aperture_efficiency, directivity_dbi, hpbw, sidelobe_level_db
from .nearfield import propagate_angular_spectrum

__all__ = [
    "ApertureField",
    "FarField",
    "JonesApertureField",
    "JonesFarField",
    "aperture_efficiency",
    "axial_ratio_db",
    "cross_polarization_db",
    "directivity_dbi",
    "far_field_from_aperture",
    "far_field_from_jones_aperture",
    "hpbw",
    "polarization_purity",
    "propagate_angular_spectrum",
    "sidelobe_level_db",
]
