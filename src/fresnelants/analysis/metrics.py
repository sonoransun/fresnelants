"""Standard antenna performance metrics computed from FarField / ApertureField."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .farfield import FarField


def directivity_dbi(ff: FarField) -> float:
    """Peak directivity in dBi."""
    return ff.peak_directivity_dbi()


def hpbw(ff: FarField, plane: str = "E") -> float:
    """Half-power beamwidth (deg) in the requested principal plane."""
    theta_deg, db = ff.cut(plane)
    valid = ~np.isnan(theta_deg)
    th, dbv = theta_deg[valid], db[valid]
    if dbv.size == 0:
        return float("nan")
    peak_idx = int(np.argmax(dbv))

    # Walk outward from the peak in each direction until crossing −3 dB.
    def cross(direction: int) -> float:
        i = peak_idx
        last = th[peak_idx]
        while 0 <= i + direction < dbv.size:
            i += direction
            if dbv[i] <= -3.0:
                # Linear interpolation between (i, i-direction).
                d2, d1 = dbv[i], dbv[i - direction]
                t2, t1 = th[i], th[i - direction]
                if d1 == d2:
                    return t1
                frac = (-3.0 - d1) / (d2 - d1)
                return t1 + frac * (t2 - t1)
            last = th[i]
        return last

    left = cross(-1)
    right = cross(+1)
    return float(abs(right - left))


def sidelobe_level_db(ff: FarField, plane: str = "E") -> float:
    """First sidelobe level (dB below peak) in the requested principal plane."""
    theta_deg, db = ff.cut(plane)
    valid = ~np.isnan(theta_deg)
    db_valid = db[valid]
    n = db_valid.size
    if n < 5:
        return float("nan")
    peak_idx = int(np.argmax(db_valid))

    # Walk right from peak until db starts increasing again (first null), then
    # find the next local max — that's the first sidelobe.
    def first_sll(start: int, direction: int) -> float:
        i = start
        # Descend until local minimum (null).
        while 0 < i + direction < n - 1 and db_valid[i + direction] < db_valid[i]:
            i += direction
        # Now ascending — find next local maximum.
        max_val = -np.inf
        while 0 < i + direction < n - 1 and db_valid[i + direction] > db_valid[i]:
            i += direction
            max_val = max(max_val, db_valid[i])
        return float(max_val) if np.isfinite(max_val) else float("nan")

    sll_r = first_sll(peak_idx, +1)
    sll_l = first_sll(peak_idx, -1)
    candidates = [s for s in (sll_l, sll_r) if np.isfinite(s)]
    return max(candidates) if candidates else float("nan")


def aperture_efficiency(ff: FarField, physical_area: float) -> float:
    """η = D · λ² / (4π·A_phys). Returns linear efficiency in [0, 1+]."""
    if physical_area <= 0:
        raise ValueError("physical_area must be positive.")
    from ..units import freq_to_wavelength

    lam = float(freq_to_wavelength(ff.freq))
    D = 10.0 ** (ff.peak_directivity_dbi() / 10.0)
    return float(D * lam**2 / (4.0 * np.pi * physical_area))


def axial_ratio_db(ff: FarField) -> NDArray[np.float64]:
    """Placeholder — single-polarization solver does not yet expose AR.

    Reflectarrays / polarizers will populate this once dual-pol support lands.
    """
    raise NotImplementedError("Axial ratio requires dual-polarization fields.")
