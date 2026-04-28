"""Tangential aperture-field representation (single-pol and dual-pol)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.geometry import ApertureGrid


@dataclass(frozen=True, slots=True)
class ApertureField:
    """Scalar tangential E-field on a flat aperture.

    Single-polarization representation, sufficient for the gain estimates of
    standard Fresnel families. Use :class:`JonesApertureField` for dual-pol /
    metasurface designs.
    """

    grid: ApertureGrid
    Ez: NDArray[np.complex128]
    freq: float

    def __post_init__(self) -> None:
        if self.Ez.shape != self.grid.shape:
            raise ValueError(f"Field shape {self.Ez.shape} does not match grid {self.grid.shape}.")

    def power(self) -> float:
        """Total radiated power in the aperture (∫|E|² dA), arbitrary units."""
        return float(np.sum(np.abs(self.Ez) ** 2) * self.grid.dx * self.grid.dy)

    def with_field(self, new_Ez: NDArray[np.complex128]) -> ApertureField:
        """Return a copy with a replaced field."""
        return ApertureField(grid=self.grid, Ez=new_Ez, freq=self.freq)

    def to_jones(self, axis: str = "x") -> JonesApertureField:
        """Lift a scalar field to a single-polarization Jones field along *axis*."""
        zero = np.zeros_like(self.Ez)
        if axis == "x":
            return JonesApertureField(grid=self.grid, Ex=self.Ez, Ey=zero, freq=self.freq)
        if axis == "y":
            return JonesApertureField(grid=self.grid, Ex=zero, Ey=self.Ez, freq=self.freq)
        raise ValueError("axis must be 'x' or 'y'")


@dataclass(frozen=True, slots=True)
class JonesApertureField:
    """Dual-polarization tangential E-field on a flat aperture.

    Stores `(Ex, Ey)` complex components. Used by metasurface designs (PB,
    anisotropic) and by polarization-aware analysis (axial ratio, cross-pol).
    """

    grid: ApertureGrid
    Ex: NDArray[np.complex128]
    Ey: NDArray[np.complex128]
    freq: float

    def __post_init__(self) -> None:
        if self.Ex.shape != self.grid.shape:
            raise ValueError(f"Ex shape {self.Ex.shape} does not match grid {self.grid.shape}.")
        if self.Ey.shape != self.grid.shape:
            raise ValueError(f"Ey shape {self.Ey.shape} does not match grid {self.grid.shape}.")

    def power(self) -> float:
        """Total radiated power, |Ex|² + |Ey|², integrated over the aperture."""
        density = np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2
        return float(np.sum(density) * self.grid.dx * self.grid.dy)

    def to_circular(self) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Right-circular and left-circular components (E_R, E_L)."""
        e_r = (self.Ex - 1j * self.Ey) / np.sqrt(2.0)
        e_l = (self.Ex + 1j * self.Ey) / np.sqrt(2.0)
        return e_r, e_l

    def with_components(
        self, Ex: NDArray[np.complex128], Ey: NDArray[np.complex128]
    ) -> JonesApertureField:
        return JonesApertureField(grid=self.grid, Ex=Ex, Ey=Ey, freq=self.freq)
