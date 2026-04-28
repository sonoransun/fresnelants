"""3D visualization using matplotlib's mplot3d (no external GPU deps).

PyVista / Plotly hooks live here too but degrade gracefully when those
optional packages are missing.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from ..analysis.aperture import ApertureField
from ..analysis.farfield import FarField
from ..analysis.nearfield import propagate_angular_spectrum
from ..designs.base import AntennaDesign
from . import style  # noqa: F401


def plot_3d_surface_profile(
    design: AntennaDesign,
    *,
    samples: int = 200,
    elevation: float = 30.0,
    azimuth: float = -55.0,
) -> Figure:
    """Render the physical depth profile of a Fresnel surface in 3D.

    Falls back gracefully for designs without depth (zone plates, reflectarrays):
    plots the binary/phase mask as a height field instead.
    """
    import matplotlib.pyplot as plt

    R = np.linspace(0, design.aperture_radius, samples // 2)
    theta = np.linspace(0, 2 * np.pi, samples)
    Rg, Tg = np.meshgrid(R, theta, indexing="xy")
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)

    # Try the design's own depth profile (curvilinear, phase-correcting).
    if hasattr(design, "fresnel_depth"):
        Z = design.fresnel_depth(Rg) * 1e3  # mm
        zlabel = "depth [mm]"
    elif hasattr(design, "groove_depths"):
        from ..core.geometry import make_aperture_grid
        from ..units import freq_to_wavelength

        lam = float(freq_to_wavelength(design.design_freq))
        spw = max(2.0, samples * lam / (2.0 * design.aperture_radius * 1.05))
        grid = make_aperture_grid(2.0 * design.aperture_radius * 1.05, spw, lam)
        depth = design.groove_depths(grid) * 1e3  # mm
        Z_field = depth
        # Resample onto polar grid via nearest neighbor.
        ix = np.clip(((X - grid.x[0]) / grid.dx).astype(int), 0, grid.nx - 1)
        iy = np.clip(((Y - grid.y[0]) / grid.dy).astype(int), 0, grid.ny - 1)
        Z = Z_field[iy, ix]
        zlabel = "groove depth [mm]"
    else:
        # Zone plate / reflectarray — height encodes phase or 0/1 transmittance.
        from ..core.geometry import make_aperture_grid
        from ..units import freq_to_wavelength

        lam = float(freq_to_wavelength(design.design_freq))
        spw = max(2.0, samples * lam / (2.0 * design.aperture_radius * 1.05))
        grid = make_aperture_grid(2.0 * design.aperture_radius * 1.05, spw, lam)
        T = design.transmittance(grid, design.design_freq)
        ix = np.clip(((X - grid.x[0]) / grid.dx).astype(int), 0, grid.nx - 1)
        iy = np.clip(((Y - grid.y[0]) / grid.dy).astype(int), 0, grid.ny - 1)
        Tij = T[iy, ix]
        if np.allclose(np.unique(np.abs(Tij)), [0, 1]) or np.unique(Tij).size <= 4:
            Z = np.abs(Tij).astype(float)
            zlabel = "transmittance"
        else:
            Z = (np.angle(Tij) + np.pi) / (2 * np.pi)
            zlabel = "phase / 2π"

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X * 1e3,
        Y * 1e3,
        Z,
        cmap="viridis",
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_title(f"{design.name} — surface profile")
    fig.colorbar(surf, ax=ax, shrink=0.6, label=zlabel)
    fig.tight_layout()
    return fig


def plot_3d_radiation_pattern(
    ff: FarField,
    *,
    dynamic_range_db: float = 30.0,
    elevation: float = 30.0,
    azimuth: float = -55.0,
) -> Figure:
    """3D rendering of the upper-hemisphere radiation pattern.

    Each (θ, φ) sample is plotted at radius = max(0, gain_dBi − (peak − DR)),
    producing a recognisable directivity solid.
    """
    import matplotlib.pyplot as plt

    d = ff.directivity()
    d_db = 10.0 * np.log10(np.maximum(d, 1e-30))
    peak = float(np.nanmax(d_db))
    floor = peak - dynamic_range_db
    radius = np.clip(d_db - floor, 0.0, None)

    # Restrict to visible region.
    mask = ff.U**2 + ff.V**2 <= 1.0
    U = ff.U.copy()
    V = ff.V.copy()
    R = radius.copy()
    R[~mask] = np.nan
    cos_theta = np.sqrt(np.clip(1 - U**2 - V**2, 0, 1))

    X = R * U
    Y = R * V
    Z = R * cos_theta

    # Decimate dense FFT grid for plotting performance.
    step = max(1, R.shape[0] // 80)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Zs = Z[::step, ::step]
    Cs = R[::step, ::step]

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        Xs,
        Ys,
        Zs,
        facecolors=plt.cm.inferno(Cs / max(np.nanmax(Cs), 1e-9)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    surf.set_clim(0, np.nanmax(Cs))
    ax.set_box_aspect((1, 1, 0.7))
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_title(f"3D radiation pattern (peak {peak:.1f} dBi, DR {dynamic_range_db:.0f} dB)")
    ax.set_xlabel("u·DR")
    ax.set_ylabel("v·DR")
    ax.set_zlabel("cos θ·DR")
    mappable = plt.cm.ScalarMappable(cmap="inferno")
    mappable.set_array(Cs)
    mappable.set_clim(floor, peak)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label="dBi")
    fig.tight_layout()
    return fig


def plot_focal_region(
    aperture: ApertureField,
    *,
    z_focal: float,
    z_span: float | None = None,
    samples: int = 41,
) -> Figure:
    """Plot |E(x, 0, z)|² across the focal region (axial slice)."""
    import matplotlib.pyplot as plt

    if z_span is None:
        z_span = z_focal
    z_values = np.linspace(z_focal - z_span / 2, z_focal + z_span / 2, samples)
    grid = aperture.grid
    intensity = np.empty((samples, grid.nx), dtype=np.float64)
    cy = grid.ny // 2
    for i, z in enumerate(z_values):
        field = propagate_angular_spectrum(aperture, float(z))
        intensity[i, :] = np.abs(field[cy, :]) ** 2

    fig, ax = plt.subplots(figsize=(7, 4))
    extent = (
        grid.x[0] * 1e3,
        grid.x[-1] * 1e3,
        z_values[0] * 1e3,
        z_values[-1] * 1e3,
    )
    im = ax.imshow(
        intensity,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    ax.axhline(z_focal * 1e3, color="cyan", linewidth=0.8, linestyle="--", label="design F")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Focal-region intensity |E|²")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, shrink=0.85, label="|E|²")
    fig.tight_layout()
    return fig
