"""Harmonic far-field analysis for time-modulated arrays."""

from __future__ import annotations

import math

import numpy as np

from ..core.geometry import make_aperture_grid
from ..units import freq_to_wavelength
from .aperture import ApertureField
from .farfield import FarField, far_field_from_aperture


def harmonic_far_field(
    array: object,  # TimeModulatedArray, type-imported lazily to avoid cycles
    freq: float,
    n: int,
    *,
    samples_per_wavelength: float = 6.0,
    pad_factor: int = 4,
) -> FarField:
    """Far-field at the n-th harmonic of a `TimeModulatedArray`.

    The carrier frequency is *freq*; the radiation at f_carrier ± n·f_modulation
    has aperture amplitude given by the n-th Fourier coefficient of each cell's
    schedule, multiplied by the cell's reflection coefficient at *freq*.
    """
    if not hasattr(array, "harmonic_coefficient"):
        raise TypeError("array must expose harmonic_coefficient(n).")
    coeff = array.harmonic_coefficient(n)  # (ny, nx) complex

    # Build the aperture-plane field. Cell-grid positions:
    x_centers, y_centers = array.cell_centers()
    Lx, Ly = array.aperture_size
    lam = float(freq_to_wavelength(freq))
    extent = max(Lx, Ly) * 1.2
    grid = make_aperture_grid(extent, samples_per_wavelength, lam)

    # Combine harmonic coefficient with the array's static beam-steering
    # phase (so the harmonic still respects the configured beam direction).
    static_phase = array.required_phase_per_cell(freq)
    cell_field = coeff * np.exp(1j * static_phase)  # (ny, nx)

    ix = np.clip(
        np.round((grid.X - x_centers[0]) / array.cell_size).astype(np.int64), 0, array.nx - 1
    )
    iy = np.clip(
        np.round((grid.Y - y_centers[0]) / array.cell_size).astype(np.int64), 0, array.ny - 1
    )
    Ez = cell_field[iy, ix]
    outside = (np.abs(grid.X) > Lx / 2) | (np.abs(grid.Y) > Ly / 2)
    Ez[outside] = 0.0

    # Apply the configured feed illumination, just like Reflectarray.
    inc = array.default_illumination(grid, freq)
    aperture = ApertureField(grid=grid, Ez=(Ez * inc).astype(np.complex128), freq=freq)
    return far_field_from_aperture(aperture, pad_factor=pad_factor)


def direction_of_arrival(
    far_field_at_zero: FarField, far_field_at_one: FarField
) -> tuple[float, float]:
    """Estimate (θ, φ) of an incident plane wave from harmonic phase.

    Compares the peak directions of the 0-th and 1-st harmonics: their
    spatial offset corresponds to the incident-wave direction (Yang &
    Tennant 2014). Returns angles in degrees.
    """
    d0 = far_field_at_zero.directivity()
    d1 = far_field_at_one.directivity()
    iy0, ix0 = np.unravel_index(np.argmax(d0), d0.shape)
    iy1, ix1 = np.unravel_index(np.argmax(d1), d1.shape)
    u_doa = far_field_at_one.u[ix1] - far_field_at_zero.u[ix0]
    v_doa = far_field_at_one.v[iy1] - far_field_at_zero.v[iy0]
    sin_theta = math.hypot(u_doa, v_doa)
    theta = math.degrees(math.asin(min(1.0, sin_theta)))
    phi = math.degrees(math.atan2(v_doa, u_doa))
    return theta, phi
