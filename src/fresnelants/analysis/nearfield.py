"""Angular-spectrum near-field propagator.

Propagates a tangential aperture field to a parallel plane at distance z by

    E(x, y, z) = F⁻¹{ F{E(x, y, 0)} · exp[ j √(k² − kx² − ky²) · z ] }

Evanescent components (kx² + ky² > k²) are damped to zero. Useful for plotting
the focal-region |E| of a Fresnel zone plate or lens.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..units import k0
from .aperture import ApertureField


def propagate_angular_spectrum(
    aperture: ApertureField,
    z: float,
) -> NDArray[np.complex128]:
    """Return E(x, y, z) on the same (x, y) grid as the aperture."""
    grid = aperture.grid
    k = k0(aperture.freq)
    fx = np.fft.fftfreq(grid.nx, d=grid.dx)
    fy = np.fft.fftfreq(grid.ny, d=grid.dy)
    KX, KY = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fy, indexing="xy")
    kz_sq = k**2 - KX**2 - KY**2
    kz = np.where(kz_sq >= 0, np.sqrt(np.maximum(kz_sq, 0)), 0.0)
    propagator = np.where(kz_sq >= 0, np.exp(1j * kz * z), 0.0)
    A = np.fft.fft2(aperture.Ez)
    return np.fft.ifft2(A * propagator).astype(np.complex128)


def focal_axis_intensity(
    aperture: ApertureField,
    z_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """|E(0, 0, z)|² along the optical axis. Used to verify focal length."""
    out = np.empty_like(z_values, dtype=np.float64)
    grid = aperture.grid
    cx, cy = grid.nx // 2, grid.ny // 2
    for i, z in enumerate(z_values):
        field = propagate_angular_spectrum(aperture, float(z))
        out[i] = float(np.abs(field[cy, cx]) ** 2)
    return out
