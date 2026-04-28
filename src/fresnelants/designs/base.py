"""`AntennaDesign` abstract base class.

Every Fresnel design exposes a uniform contract:

* `aperture_radius` — outer extent of the device.
* `aperture_field(grid, freq, illumination)` — synthesize the tangential E-field
  on a flat aperture grid.
* `geometry_layers(...)` — used by the visualization and export modules to
  draw / mesh the physical antenna.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..analysis.aperture import ApertureField
from ..core.geometry import ApertureGrid, make_aperture_grid
from ..core.wavefront import SphericalWave
from ..units import freq_to_wavelength


@dataclass
class DesignResult:
    """Container for a synthesized aperture and associated diagnostics."""

    design: AntennaDesign
    aperture: ApertureField
    transmittance: NDArray[np.complex128]


class AntennaDesign(ABC):
    """Abstract base for all Fresnel antenna families."""

    name: str = "AntennaDesign"
    focal_length: float
    """Designed focal length [m]."""
    design_freq: float
    """Frequency at which zones / phase profile were synthesized [Hz]."""

    @property
    @abstractmethod
    def aperture_radius(self) -> float:
        """Outer physical radius of the device [m]."""

    @abstractmethod
    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        """Complex transmission coefficient on the aperture grid.

        For binary zone plates this is 0/1; for phase-correcting plates it is a
        unit-modulus phase factor; for reflectarrays it is the reflection
        coefficient of each cell. *state* is an optional design-specific
        object (e.g., per-cell voltages for a reconfigurable surface); static
        designs ignore it.
        """

    def default_illumination(self, grid: ApertureGrid, freq: float) -> NDArray[np.complex128]:
        """Natural feed illumination for this design.

        Plates / lenses default to a spherical-wave feed at (0, 0, −F)
        (representing the antenna emitting through the device). Subclasses
        whose transmittance already bakes in the feed (e.g. reflectarrays)
        should override to return ``np.ones(grid.shape, complex)``.
        """
        feed = SphericalWave(z0=-self.focal_length)
        return feed.field_on(grid, freq)

    def aperture_field(
        self,
        freq: float,
        *,
        grid: ApertureGrid | None = None,
        samples_per_wavelength: float = 6.0,
        margin: float = 1.2,
        illumination: NDArray[np.complex128] | float | None = None,
        state: object | None = None,
    ) -> ApertureField:
        """Synthesize the aperture field at *freq*.

        If *grid* is None, build a square grid at `samples_per_wavelength`
        sampling, sized to `margin × 2 × aperture_radius`.

        *illumination* defaults to a unit-amplitude plane wave (1.0 across the
        aperture). It can be a scalar or a complex array of shape (ny, nx) —
        useful for spherical-wave / cosine-q feed illuminations on
        reflectarrays.
        """
        if grid is None:
            wavelength = float(freq_to_wavelength(freq))
            extent = 2.0 * self.aperture_radius * margin
            grid = make_aperture_grid(extent, samples_per_wavelength, wavelength)
        T = self.transmittance(grid, freq, state)
        if illumination is None:
            inc = self.default_illumination(grid, freq)
        elif np.isscalar(illumination):
            inc = np.full(T.shape, complex(illumination), dtype=np.complex128)  # type: ignore[arg-type]
        else:
            inc = np.asarray(illumination, dtype=np.complex128)
            if inc.shape != T.shape:
                raise ValueError(
                    f"illumination shape {inc.shape} does not match aperture {T.shape}."
                )
        Ez = T * inc
        return ApertureField(grid=grid, Ez=Ez, freq=freq)

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("_") and not callable(v)}
        kv = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{type(self).__name__}({kv})"
