"""Forward angular-spectrum cascade through stacked aperture-plane devices.

Used by :class:`fresnelants.designs.composite.CompositeAntenna` to chain the
fields of two or more designs separated by free-space propagation gaps. Only
forward propagation is modeled (no in-stack multi-bounce); good enough for
low-reflection plate stacks and the doublet / bifocal / folded designs.
"""

from __future__ import annotations

import numpy as np

from .aperture import ApertureField
from .nearfield import propagate_angular_spectrum


def propagate(field: ApertureField, dz: float) -> ApertureField:
    """Propagate *field* forward by *dz* metres (free space)."""
    if abs(dz) < 1e-15:
        return field
    new_Ez = propagate_angular_spectrum(field, dz)
    return field.with_field(new_Ez)


def apply_transmittance(field: ApertureField, transmittance: np.ndarray) -> ApertureField:
    """Multiply the aperture field by a (matching-shape) complex transmittance."""
    if transmittance.shape != field.grid.shape:
        raise ValueError(
            f"transmittance shape {transmittance.shape} does not match grid {field.grid.shape}"
        )
    return field.with_field(field.Ez * transmittance.astype(np.complex128))
