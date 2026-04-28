"""Sub-wavelength metasurface designs (dual-polarization).

`MetasurfaceLens` is a flat metasurface whose per-cell rotation imparts a
geometric (Pancharatnam–Berry) phase: rotating cell *n* by α(n) produces a
far-field phase ±2α(n) on the cross-polarization-converted output.

`DualPolSharedAperture` interleaves two metasurfaces — one for each
polarization — into a single physical aperture, useful for shared-aperture
mmW comms (Tx/Rx on orthogonal polarizations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..analysis.aperture import JonesApertureField
from ..cells.metasurface import AnisotropicEllipseCell, PancharatnamBerryCell
from ..core.geometry import ApertureGrid, make_aperture_grid
from ..units import freq_to_wavelength
from .base import AntennaDesign


@dataclass
class MetasurfaceLens(AntennaDesign):
    """Pancharatnam–Berry metasurface lens (dual-pol output).

    Rotation pattern α(x, y) follows the standard hyperbolic profile so that
    the LCP-input → RCP-output beam focuses to (0, 0, F).
    """

    focal_length: float = 0.05
    design_freq: float = 60e9
    aperture_radius_m: float = 0.025
    sense: Literal["LCP_to_RCP", "RCP_to_LCP"] = "LCP_to_RCP"
    cell: PancharatnamBerryCell | None = None
    name: str = "MetasurfaceLens"

    def __post_init__(self) -> None:
        if self.cell is None:
            self.cell = PancharatnamBerryCell(sense=self.sense)

    @property
    def aperture_radius(self) -> float:
        return float(self.aperture_radius_m)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        """Scalar transmittance — co-pol channel (cross-pol projection of
        the Jones output). For the full dual-pol response use
        :meth:`jones_aperture_field`."""
        rotation = self._rotation_map(grid, freq)
        return np.exp(1j * 2.0 * rotation).astype(np.complex128)

    def _rotation_map(self, grid: ApertureGrid, freq: float) -> NDArray[np.float64]:
        """Cell rotation angle α(r) so that the geometric phase −2α cancels
        the spherical-feed propagation phase −k·d at the aperture."""
        lam = float(freq_to_wavelength(freq))
        d = np.sqrt(grid.X**2 + grid.Y**2 + self.focal_length**2)
        # HWP imparts phase −2α; choose α = −k(d − F)/2 so the total at the
        # output is uniform (= −k·F).
        alpha = -np.pi * (d - self.focal_length) / lam
        out_of_aperture = self.aperture_radius < grid.R
        alpha = np.where(out_of_aperture, 0.0, alpha)
        return alpha

    def jones_aperture_field(self, freq: float, **kwargs) -> JonesApertureField:
        """Synthesize a dual-pol aperture: incident LCP → emergent RCP focused."""
        from ..core.wavefront import SphericalWave

        samples_per_wavelength = kwargs.get("samples_per_wavelength", 6.0)
        margin = kwargs.get("margin", 1.2)
        wavelength = float(freq_to_wavelength(freq))
        extent = 2.0 * self.aperture_radius * margin
        grid = make_aperture_grid(extent, samples_per_wavelength, wavelength)

        feed = SphericalWave(z0=-self.focal_length)
        E_in = feed.field_on(grid, freq)  # scalar, treat as LCP component magnitude

        # Incident LCP at aperture using the convention adopted in
        # JonesApertureField.to_circular() (e_l = (Ex + jEy)/√2): LCP → (1, -j).
        Ex_in = E_in / np.sqrt(2.0)
        Ey_in = -1j * E_in / np.sqrt(2.0)

        # Cell rotation map. PB cell at angle α acts as a half-wave plate
        # at α; its Jones matrix is [[cos 2α, sin 2α], [sin 2α, -cos 2α]].
        # For LCP input the HWP itself imparts the geometric phase exp(±j·2α)
        # along with the polarization handedness flip. No extra phase factor
        # is needed.
        alpha = self._rotation_map(grid, freq)
        if self.sense == "RCP_to_LCP":
            alpha = -alpha
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        Ex_out = (ca**2 - sa**2) * Ex_in + 2 * ca * sa * Ey_in
        Ey_out = 2 * ca * sa * Ex_in - (ca**2 - sa**2) * Ey_in
        # Mask outside aperture
        mask = self.aperture_radius >= grid.R
        Ex_out = np.where(mask, Ex_out, 0)
        Ey_out = np.where(mask, Ey_out, 0)
        return JonesApertureField(
            grid=grid, Ex=Ex_out.astype(np.complex128), Ey=Ey_out.astype(np.complex128), freq=freq
        )


@dataclass
class DualPolSharedAperture(AntennaDesign):
    """Two interleaved metasurfaces — one per polarization.

    Cells are split into a checkerboard: one set focuses Ex (vertical pol)
    at f_v with focal F_v; the other focuses Ey (horizontal pol) at f_h with
    focal F_h. Useful for V/H or low/high-band shared apertures.
    """

    f_v: float = 28e9
    f_h: float = 39e9
    focal_v: float = 0.10
    focal_h: float = 0.10
    aperture_radius_m: float = 0.05
    cell_anisotropic: AnisotropicEllipseCell | None = None
    name: str = "DualPolSharedAperture"

    def __post_init__(self) -> None:
        if self.cell_anisotropic is None:
            self.cell_anisotropic = AnisotropicEllipseCell()
        self.focal_length = 0.5 * (self.focal_v + self.focal_h)
        self.design_freq = 0.5 * (self.f_v + self.f_h)

    @property
    def aperture_radius(self) -> float:
        return float(self.aperture_radius_m)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        """Scalar transmittance — undefined; use :meth:`jones_aperture_field`."""
        raise NotImplementedError(
            "DualPolSharedAperture is dual-pol only; use jones_aperture_field()."
        )

    def jones_aperture_field(self, freq: float, **kwargs) -> JonesApertureField:
        """Build a Jones aperture using a checkerboard cell assignment."""
        from ..core.wavefront import SphericalWave

        samples_per_wavelength = kwargs.get("samples_per_wavelength", 6.0)
        margin = kwargs.get("margin", 1.2)
        wavelength = float(freq_to_wavelength(freq))
        extent = 2.0 * self.aperture_radius * margin
        grid = make_aperture_grid(extent, samples_per_wavelength, wavelength)

        feed = SphericalWave(z0=-self.focal_length)
        E_in = feed.field_on(grid, freq)

        # Checkerboard mask (1 → V cells, 0 → H cells).
        ix = ((grid.X / (wavelength * 0.5)).astype(int) + 100) % 2
        iy = ((grid.Y / (wavelength * 0.5)).astype(int) + 100) % 2
        v_mask = (ix ^ iy).astype(bool)
        h_mask = ~v_mask

        # V-pol focusing phase (operates at f_v).
        d_v = np.sqrt(grid.X**2 + grid.Y**2 + self.focal_v**2)
        phi_v = 2.0 * np.pi * (d_v - self.focal_v) / wavelength
        # H-pol focusing phase (at f_h).
        d_h = np.sqrt(grid.X**2 + grid.Y**2 + self.focal_h**2)
        phi_h = 2.0 * np.pi * (d_h - self.focal_h) / wavelength

        # Build aperture: V cells respond to f_v with phase phi_v, etc.
        Ex = np.zeros(grid.shape, dtype=np.complex128)
        Ey = np.zeros(grid.shape, dtype=np.complex128)
        # If freq matches f_v (within band), V cells are active.
        if abs(freq - self.f_v) < abs(freq - self.f_h):
            Ex = np.where(v_mask, E_in * np.exp(1j * phi_v), 0)
        else:
            Ey = np.where(h_mask, E_in * np.exp(1j * phi_h), 0)
        # Clip outside aperture
        outside = self.aperture_radius < grid.R
        Ex[outside] = 0
        Ey[outside] = 0
        return JonesApertureField(grid=grid, Ex=Ex, Ey=Ey, freq=freq)
