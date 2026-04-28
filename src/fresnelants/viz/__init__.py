"""Visualization (matplotlib for 2D + 3D; pyvista/plotly optional)."""

from .plots2d import (
    plot_aperture_amplitude,
    plot_aperture_phase,
    plot_farfield_2d,
    plot_principal_cuts,
    plot_zone_layout,
)
from .plots3d import (
    plot_3d_radiation_pattern,
    plot_3d_surface_profile,
    plot_focal_region,
)

__all__ = [
    "plot_3d_radiation_pattern",
    "plot_3d_surface_profile",
    "plot_aperture_amplitude",
    "plot_aperture_phase",
    "plot_farfield_2d",
    "plot_focal_region",
    "plot_principal_cuts",
    "plot_zone_layout",
]
