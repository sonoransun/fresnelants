"""2D matplotlib plots: zone layouts, aperture maps, far-field cuts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from ..analysis.aperture import ApertureField
from ..analysis.farfield import FarField
from ..designs.base import AntennaDesign
from . import style  # noqa: F401  (apply rcParams)


def plot_zone_layout(design: AntennaDesign, *, samples: int = 800) -> Figure:
    """Plot the transmittance/reflectance pattern of *design* on a fine grid.

    Renders amplitude (for binary plates) or phase (for phase-correcting,
    reflectarray, curvilinear).
    """
    from ..core.geometry import make_aperture_grid
    from ..units import freq_to_wavelength

    lam = float(freq_to_wavelength(design.design_freq))
    extent = 2.0 * design.aperture_radius * 1.05
    spw = max(2.0, samples * lam / extent)
    grid = make_aperture_grid(extent, spw, lam)
    T = design.transmittance(grid, design.design_freq)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    amp = np.abs(T)
    pha = np.angle(T)
    pha[amp == 0] = np.nan

    axes[0].imshow(
        amp,
        extent=(grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3),
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=max(1.0, float(amp.max())),
    )
    axes[0].set_title(f"{design.name} — amplitude")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")

    im1 = axes[1].imshow(
        pha,
        extent=(grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3),
        origin="lower",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1].set_title(f"{design.name} — phase")
    axes[1].set_xlabel("x [mm]")
    cbar = fig.colorbar(im1, ax=axes[1], shrink=0.85)
    cbar.set_label("phase [rad]")

    for ax in axes:
        ax.add_patch(
            Circle(
                (0, 0),
                design.aperture_radius * 1e3,
                fill=False,
                edgecolor="red",
                linewidth=0.8,
                linestyle="--",
            )
        )
        ax.set_aspect("equal")
    fig.suptitle(
        f"{design.name}: F={design.focal_length * 1e3:.0f} mm, "
        f"f={design.design_freq / 1e9:.1f} GHz, "
        f"D={2 * design.aperture_radius * 1e3:.0f} mm"
    )
    fig.tight_layout()
    return fig


def plot_aperture_amplitude(field: ApertureField) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    g = field.grid
    im = ax.imshow(
        np.abs(field.Ez),
        extent=(g.x[0] * 1e3, g.x[-1] * 1e3, g.y[0] * 1e3, g.y[-1] * 1e3),
        origin="lower",
        cmap="magma",
    )
    ax.set_title("Aperture |E|")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.85, label="|E| (a.u.)")
    fig.tight_layout()
    return fig


def plot_aperture_phase(field: ApertureField) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    g = field.grid
    pha = np.angle(field.Ez)
    pha[np.abs(field.Ez) < 1e-12] = np.nan
    im = ax.imshow(
        pha,
        extent=(g.x[0] * 1e3, g.x[-1] * 1e3, g.y[0] * 1e3, g.y[-1] * 1e3),
        origin="lower",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax.set_title("Aperture phase")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.85, label="phase [rad]")
    fig.tight_layout()
    return fig


def plot_farfield_2d(ff: FarField, *, dynamic_range_db: float = 40.0) -> Figure:
    """U–V map of directivity in dBi (clipped to peak−DR)."""
    d = ff.directivity()
    d_db = 10.0 * np.log10(np.maximum(d, 1e-30))
    vmax = float(np.nanmax(d_db))
    vmin = vmax - dynamic_range_db

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    extent = (ff.u[0], ff.u[-1], ff.v[0], ff.v[-1])
    im = ax.imshow(d_db, extent=extent, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    # Visible-region circle.
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="white", linewidth=0.6, linestyle=":")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Far-field directivity [dBi]")
    ax.set_xlabel(r"$u = \sin\theta\cos\phi$")
    ax.set_ylabel(r"$v = \sin\theta\sin\phi$")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.85, label="dBi")
    fig.tight_layout()
    return fig


def plot_principal_cuts(ff: FarField, *, label: str | None = None) -> Figure:
    """E-plane and H-plane gain cuts, normalized to peak."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for plane, color in (("E", "tab:blue"), ("H", "tab:orange")):
        theta_deg, db = ff.cut(plane)
        valid = ~np.isnan(theta_deg)
        ax.plot(theta_deg[valid], db[valid], label=f"{plane}-plane", color=color)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-50, 2)
    ax.set_xlabel("θ [deg]")
    ax.set_ylabel("Normalized gain [dB]")
    title = "Principal-plane cuts"
    if label:
        title += f" — {label}"
    ax.set_title(title)
    ax.legend(loc="lower center")
    fig.tight_layout()
    return fig
