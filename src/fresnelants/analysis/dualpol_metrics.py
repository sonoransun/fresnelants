"""Dual-polarization metrics: axial ratio, cross-pol level, polarization purity."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .farfield import JonesFarField


def axial_ratio_db(ff: JonesFarField) -> NDArray[np.float64]:
    """Pixel-wise axial ratio [dB]. AR = 0 dB → circular; AR → ∞ → linear."""
    er, el = ff.co_cross("rcp")
    a_r = np.abs(er)
    a_l = np.abs(el)
    num = a_r + a_l
    den = np.maximum(np.abs(a_r - a_l), 1e-30)
    ar_lin = num / den
    out = 20.0 * np.log10(np.maximum(ar_lin, 1e-30))
    out[~ff.visible_mask] = np.nan
    return out


def cross_polarization_db(ff: JonesFarField, polarization: str = "x") -> NDArray[np.float64]:
    """Cross-pol level [dB] relative to co-pol peak.

    For polarization='x', co = Ex, cross = Ey. Negative values = good rejection.
    """
    co, cross = ff.co_cross(polarization)
    co_peak = float(np.max(np.abs(co) ** 2))
    if co_peak <= 0:
        return np.full(co.shape, np.nan, dtype=np.float64)
    out = 10.0 * np.log10(np.maximum(np.abs(cross) ** 2 / co_peak, 1e-30))
    out[~ff.visible_mask] = np.nan
    return out


def polarization_purity(ff: JonesFarField, polarization: str = "x") -> float:
    """Fraction of total radiated power in the desired polarization (0..1)."""
    co, cross = ff.co_cross(polarization)
    p_co = float(np.sum(np.abs(co) ** 2 * ff.visible_mask))
    p_cross = float(np.sum(np.abs(cross) ** 2 * ff.visible_mask))
    total = p_co + p_cross
    return p_co / total if total > 0 else 0.0
