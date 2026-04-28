"""Planar reflectarray.

A reflectarray is a flat array of unit cells whose individual reflection
phases are tuned to reproduce, in the far field, the radiation of a parabolic
reflector — typically with a feed at (xf, yf, zf). Each cell's required phase
is the round-trip path-length difference between the feed and the target
beam direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.geometry import ApertureGrid
from ..core.wavefront import CosineFeed
from ..units import freq_to_wavelength, k0
from .base import AntennaDesign


@dataclass
class Reflectarray(AntennaDesign):
    """Square-grid microstrip reflectarray with arbitrary beam direction.

    Parameters
    ----------
    focal_length
        Distance from the aperture plane (z=0) to the feed phase center [m].
    design_freq
        Synthesis frequency [Hz].
    nx, ny
        Number of cells per axis.
    cell_size
        Center-to-center cell spacing [m]. Common: 0.5λ.
    feed_offset
        (x, y) offset of the feed from the array center [m].
    beam_direction
        (theta, phi) main-beam direction in radians (theta from +z).
    feed_q
        Cosine-q exponent for the feed amplitude pattern (illumination taper).
    phase_levels
        If finite, quantize the cell phase to this many discrete levels (e.g.
        2 for a 1-bit array, 4 for a 2-bit array). Default: continuous.
    """

    focal_length: float
    design_freq: float
    nx: int = 32
    ny: int = 32
    cell_size: float = 0.0
    feed_offset: tuple[float, float] = (0.0, 0.0)
    beam_direction: tuple[float, float] = (0.0, 0.0)
    feed_q: float = 6.0
    phase_levels: int | None = None
    name: str = "Reflectarray"

    def __post_init__(self) -> None:
        if self.cell_size <= 0.0:
            self.cell_size = 0.5 * float(freq_to_wavelength(self.design_freq))

    @property
    def aperture_radius(self) -> float:
        # Use circumscribed radius; physical aperture is square but we report
        # the bounding circle.
        return 0.5 * np.hypot(self.nx * self.cell_size, self.ny * self.cell_size)

    @property
    def aperture_size(self) -> tuple[float, float]:
        return self.nx * self.cell_size, self.ny * self.cell_size

    def cell_centers(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = (np.arange(self.nx) - (self.nx - 1) / 2.0) * self.cell_size
        y = (np.arange(self.ny) - (self.ny - 1) / 2.0) * self.cell_size
        return x, y

    def required_phase_per_cell(self, freq: float) -> NDArray[np.float64]:
        """Per-cell reflection phase (radians) for the configured beam."""
        x, y = self.cell_centers()
        X, Y = np.meshgrid(x, y, indexing="xy")
        k = k0(freq)
        # Path from feed at (xf, yf, F) to cell (X, Y, 0).
        xf, yf = self.feed_offset
        d_feed = np.sqrt((X - xf) ** 2 + (Y - yf) ** 2 + self.focal_length**2)
        # Outgoing plane-wave phase along beam direction.
        theta_b, phi_b = self.beam_direction
        u = np.sin(theta_b) * np.cos(phi_b)
        v = np.sin(theta_b) * np.sin(phi_b)
        # Required reflection phase: cancel the (negative) feed-path phase and
        # add the (positive) plane-wave phase ramp toward (u, v).
        # Incoming exp(−j k d_feed); outgoing exp(j (k(u·X + v·Y) + φ_R)).
        # Demand a constant total phase ⇒ φ_R = k·d_feed − k(u·X + v·Y).
        phase = k * d_feed - k * (u * X + v * Y)
        phase = np.mod(phase, 2.0 * np.pi)
        if self.phase_levels is not None and self.phase_levels >= 2:
            step = 2.0 * np.pi / self.phase_levels
            phase = np.floor(phase / step) * step
        return phase

    def default_illumination(self, grid: ApertureGrid, freq: float) -> NDArray[np.complex128]:
        """Cosine-q feed at (xf, yf, +F) illuminating the array."""
        xf, yf = self.feed_offset
        feed = CosineFeed(x0=xf, y0=yf, z0=self.focal_length, q=self.feed_q)
        return feed.field_on(grid, freq)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        """Per-cell reflection coefficient (unit amplitude, phase-only)."""
        x, y = self.cell_centers()
        phase = self.required_phase_per_cell(freq)
        ix = np.clip(np.round((grid.X - x[0]) / self.cell_size).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(np.round((grid.Y - y[0]) / self.cell_size).astype(np.int64), 0, self.ny - 1)
        T = np.exp(1j * phase[iy, ix])
        Lx, Ly = self.aperture_size
        outside = (np.abs(grid.X) > Lx / 2) | (np.abs(grid.Y) > Ly / 2)
        T[outside] = 0.0
        return T.astype(np.complex128)
