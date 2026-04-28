"""Fresnel zone geometry — radii, focal-plane transforms, aperture grids.

Reference: Hristov, *Fresnel Zones in Wireless Links, Zone Plate Lenses and
Antennas* (Artech House, 2000); Wiltse, "The Fresnel Zone-Plate Lens", *Proc.
SPIE* 544 (1985).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..units import freq_to_wavelength


def zone_radius(
    n: int | NDArray[np.int_], focal_length: float, wavelength: float
) -> NDArray[np.float64]:
    """Radius of the *n*-th Fresnel zone boundary on a flat aperture.

    Uses the exact form

        rₙ = √( n·λ·F + (n·λ/2)² )

    which collapses to the paraxial rₙ ≈ √(n·λ·F) for n·λ ≪ F. *n* may be an
    array; the result has the same shape.
    """
    n_arr = np.asarray(n, dtype=np.float64)
    if np.any(n_arr < 0):
        raise ValueError("Zone index n must be non-negative.")
    if focal_length <= 0:
        raise ValueError("Focal length must be positive.")
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")
    return np.sqrt(n_arr * wavelength * focal_length + (n_arr * wavelength / 2.0) ** 2)


def fresnel_zone_radii(
    num_zones: int, focal_length: float, wavelength: float
) -> NDArray[np.float64]:
    """Return [r₁, r₂, …, r_N] for *num_zones* zones."""
    if num_zones <= 0:
        raise ValueError("num_zones must be positive.")
    return zone_radius(np.arange(1, num_zones + 1), focal_length, wavelength)


def aperture_radius(num_zones: int, focal_length: float, wavelength: float) -> float:
    """Outer radius of an *num_zones*-zone Fresnel aperture."""
    return float(zone_radius(num_zones, focal_length, wavelength))


def zone_radius_from_freq(n: int, focal_length: float, freq: float) -> float:
    """Convenience: zone radius given frequency rather than wavelength."""
    return float(zone_radius(n, focal_length, float(freq_to_wavelength(freq))))


@dataclass(frozen=True, slots=True)
class ApertureGrid:
    """Cartesian aperture grid centered at the origin.

    Attributes
    ----------
    nx, ny : int
        Number of samples along each axis (must be even for FFT efficiency).
    dx, dy : float
        Sample spacing [m].
    x, y : NDArray
        1-D coordinate vectors (length nx, ny).
    X, Y : NDArray
        2-D meshgrids (shape (ny, nx)) using `indexing='xy'`.
    R : NDArray
        Radial distance √(X² + Y²).
    PHI : NDArray
        Azimuth atan2(Y, X) ∈ (-π, π].
    """

    nx: int
    ny: int
    dx: float
    dy: float
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    R: NDArray[np.float64]
    PHI: NDArray[np.float64]

    @property
    def Lx(self) -> float:
        """Aperture extent along x [m]."""
        return self.nx * self.dx

    @property
    def Ly(self) -> float:
        """Aperture extent along y [m]."""
        return self.ny * self.dy

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)


def make_aperture_grid(
    extent: float, samples_per_wavelength: float, wavelength: float
) -> ApertureGrid:
    """Build a square ApertureGrid spanning [-extent/2, +extent/2] on each axis.

    Parameters
    ----------
    extent
        Total side length of the aperture sampling window [m].
    samples_per_wavelength
        Sampling density (typical 4–10).
    wavelength
        Free-space wavelength [m].
    """
    if extent <= 0 or samples_per_wavelength <= 0 or wavelength <= 0:
        raise ValueError("extent, samples_per_wavelength, wavelength must all be positive.")
    n = int(np.ceil(extent / wavelength * samples_per_wavelength))
    if n % 2:
        n += 1
    dx = extent / n
    coords = (np.arange(n) - n / 2 + 0.5) * dx
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    return ApertureGrid(
        nx=n,
        ny=n,
        dx=dx,
        dy=dx,
        x=coords,
        y=coords,
        X=X,
        Y=Y,
        R=np.hypot(X, Y),
        PHI=np.arctan2(Y, X),
    )


def offset_zone_centers(
    num_zones: int,
    focal_length: float,
    wavelength: float,
    tilt_angle: float,
) -> NDArray[np.float64]:
    """Center positions (x_n) of zones for an offset Fresnel plate.

    For a focal point displaced from the geometric axis by tan(α)·F, the
    stationary-phase point of zone *n* shifts by approximately

        Δx_n ≈ (n·λ/2)·sin α      (paraxial, small α)

    See Hristov §5.3. Returns an array of length *num_zones* giving the offset
    for each zone center; designs that need full off-axis path-length matching
    should use `path_length_phase` instead.
    """
    if num_zones <= 0:
        raise ValueError("num_zones must be positive.")
    n = np.arange(1, num_zones + 1, dtype=np.float64)
    return (n * wavelength / 2.0) * np.sin(tilt_angle)


def path_length_phase(
    grid: ApertureGrid,
    focal_length: float,
    wavelength: float,
    tilt: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    """Phase delay (in radians) imposed by a point-focus geometry.

    Computes the path difference between (X, Y, 0) on the aperture and the
    focal point at (F·tan θx, F·tan θy, F). Useful as the reference phase a
    Fresnel device must compensate.
    """
    fx = focal_length * np.tan(tilt[0])
    fy = focal_length * np.tan(tilt[1])
    distance = np.sqrt((grid.X - fx) ** 2 + (grid.Y - fy) ** 2 + focal_length**2)
    return 2.0 * np.pi * distance / wavelength
