"""SI unit helpers and electromagnetic constants."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

c0: float = 299_792_458.0
"""Speed of light in vacuum [m/s]."""

eta0: float = 376.730_313_668
"""Impedance of free space [ohm]."""

eps0: float = 8.854_187_8128e-12
"""Vacuum permittivity [F/m]."""

mu0: float = 1.256_637_062_12e-6
"""Vacuum permeability [H/m]."""


def freq_to_wavelength(freq: ArrayLike) -> NDArray[np.float64]:
    """Free-space wavelength λ = c/f [m]."""
    f = np.asarray(freq, dtype=np.float64)
    return c0 / f


def wavelength_to_freq(wavelength: ArrayLike) -> NDArray[np.float64]:
    """Frequency f = c/λ [Hz]."""
    lam = np.asarray(wavelength, dtype=np.float64)
    return c0 / lam


def k0(freq: ArrayLike) -> NDArray[np.float64]:
    """Free-space wavenumber k = 2π/λ = 2πf/c [rad/m]."""
    f = np.asarray(freq, dtype=np.float64)
    return 2.0 * np.pi * f / c0


def db10(x: ArrayLike) -> NDArray[np.float64]:
    """Power-ratio dB. Floors at -300 dB to avoid log(0)."""
    arr = np.asarray(x, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(arr, 1e-30))


def db20(x: ArrayLike) -> NDArray[np.float64]:
    """Field-amplitude dB. Floors at -300 dB."""
    arr = np.abs(np.asarray(x))
    return 20.0 * np.log10(np.maximum(arr, 1e-30))
