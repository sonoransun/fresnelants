"""Far-field via the 2-D Fourier transform of the aperture distribution.

For a flat aperture with tangential field E_a(x, y) at z = 0 the far-field
radiation pattern in the forward hemisphere can be written (see Balanis §12-10)

    E(θ, φ) ≈ jk·cosθ / (2π·r) · ∬ E_a(x, y) · exp[+jk(x·u + y·v)] dx dy

with u = sinθ cosφ, v = sinθ sinφ. The integral is the 2-D Fourier transform
evaluated at spatial frequencies (kx, ky) = (k·u, k·v). We sample (u, v) on a
uniform grid and compute via a single FFT, zero-padding for fine angular
resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..units import k0
from .aperture import ApertureField, JonesApertureField


@dataclass(frozen=True, slots=True)
class FarField:
    """Far-field on a (u, v) = (sinθ cosφ, sinθ sinφ) grid.

    Attributes
    ----------
    u, v : NDArray
        Direction-cosine sampling vectors.
    U, V : NDArray
        Direction-cosine meshgrids.
    E : NDArray
        Complex far-field amplitude (proportional to true field; constant
        scaling cancels in directivity / efficiency calculations).
    freq : float
        Frequency [Hz].
    aperture_power : float
        ∫|E_a|² dA used for gain normalization.
    """

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    U: NDArray[np.float64]
    V: NDArray[np.float64]
    E: NDArray[np.complex128]
    freq: float
    aperture_power: float

    @property
    def visible_mask(self) -> NDArray[np.bool_]:
        """True where u² + v² ≤ 1 (real propagation directions)."""
        return self.U**2 + self.V**2 <= 1.0

    def intensity(self) -> NDArray[np.float64]:
        """|E|² with non-propagating directions zeroed."""
        out = np.abs(self.E) ** 2
        out[~self.visible_mask] = 0.0
        return out

    def directivity(self) -> NDArray[np.float64]:
        """Directivity D(u, v) [linear], peak gives peak directivity."""
        intensity = self.intensity()
        # Solid-angle integral on the (u, v) plane with element dΩ = du dv / cosθ.
        cos_theta = np.sqrt(np.clip(1.0 - self.U**2 - self.V**2, 0.0, 1.0))
        du = float(self.u[1] - self.u[0])
        dv = float(self.v[1] - self.v[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            dOmega = np.where(cos_theta > 1e-9, du * dv / cos_theta, 0.0)
        prad = float(np.sum(intensity * dOmega))
        if prad <= 0.0:
            return np.zeros_like(intensity)
        return 4.0 * np.pi * intensity / prad

    def peak_directivity_dbi(self) -> float:
        d = self.directivity()
        peak = float(np.max(d))
        return float(10.0 * np.log10(max(peak, 1e-30)))

    def cut(self, plane: str = "E") -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Principal-plane cut.

        Returns (theta_deg, gain_dB) with gain normalized to peak.
        """
        d = self.directivity()
        ny, nx = d.shape
        if plane.upper() == "E":
            line = d[ny // 2, :]
            sin_theta = self.u
        elif plane.upper() == "H":
            line = d[:, nx // 2]
            sin_theta = self.v
        else:
            raise ValueError("plane must be 'E' or 'H'")
        peak = float(np.max(line))
        line_db = 10.0 * np.log10(np.maximum(line / max(peak, 1e-30), 1e-12))
        # Map sinθ → θ; clip out-of-cone samples.
        theta = np.where(np.abs(sin_theta) <= 1.0, np.arcsin(np.clip(sin_theta, -1, 1)), np.nan)
        return np.degrees(theta), line_db


def far_field_from_aperture(
    aperture: ApertureField,
    pad_factor: int = 4,
) -> FarField:
    """FFT-based far-field of a tangential aperture distribution.

    Parameters
    ----------
    aperture
        Aperture field (assumed at z = 0).
    pad_factor
        Zero-pad multiplier for FFT (4 ≈ smooth cuts; 8 for clean SLL).
    """
    if pad_factor < 1:
        raise ValueError("pad_factor must be ≥ 1")
    grid = aperture.grid
    freq = aperture.freq
    Ez = aperture.Ez

    nx_pad = grid.nx * pad_factor
    ny_pad = grid.ny * pad_factor
    if nx_pad % 2:
        nx_pad += 1
    if ny_pad % 2:
        ny_pad += 1

    padded = np.zeros((ny_pad, nx_pad), dtype=np.complex128)
    y0 = (ny_pad - grid.ny) // 2
    x0 = (nx_pad - grid.nx) // 2
    padded[y0 : y0 + grid.ny, x0 : x0 + grid.nx] = Ez

    # Antenna far-field uses the +j Fourier kernel: ∫ E_a exp(+j(kx·x+ky·y)) dxdy.
    # numpy's ifft2 carries that convention (with a 1/N normalization we undo).
    spectrum = (
        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded)))
        * grid.dx
        * grid.dy
        * nx_pad
        * ny_pad
    )

    k = k0(freq)
    fx = np.fft.fftshift(np.fft.fftfreq(nx_pad, d=grid.dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny_pad, d=grid.dy))
    u = 2.0 * np.pi * fx / k  # u = kx / k
    v = 2.0 * np.pi * fy / k
    U, V = np.meshgrid(u, v, indexing="xy")

    # cosθ obliquity factor (Balanis 12-10b) keeps the radiation pattern correct
    # at moderate angles. Beyond visible region we mask.
    cos_theta = np.sqrt(np.clip(1.0 - U**2 - V**2, 0.0, 1.0))
    E_far = 1j * k / (2.0 * np.pi) * cos_theta * spectrum

    return FarField(
        u=u,
        v=v,
        U=U,
        V=V,
        E=E_far,
        freq=freq,
        aperture_power=aperture.power(),
    )


@dataclass(frozen=True, slots=True)
class JonesFarField:
    """Dual-polarization far-field on a (u, v) grid.

    Stores both Cartesian components (Ex_far, Ey_far). Use :func:`co_cross`
    to project onto a co-/cross-polar basis, or the helpers in
    :mod:`fresnelants.analysis.dualpol_metrics`.
    """

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    U: NDArray[np.float64]
    V: NDArray[np.float64]
    Ex: NDArray[np.complex128]
    Ey: NDArray[np.complex128]
    freq: float
    aperture_power: float

    @property
    def visible_mask(self) -> NDArray[np.bool_]:
        return self.U**2 + self.V**2 <= 1.0

    def total_intensity(self) -> NDArray[np.float64]:
        out = np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2
        out[~self.visible_mask] = 0.0
        return out

    def directivity(self) -> NDArray[np.float64]:
        intens = self.total_intensity()
        cos_theta = np.sqrt(np.clip(1.0 - self.U**2 - self.V**2, 0.0, 1.0))
        du = float(self.u[1] - self.u[0])
        dv = float(self.v[1] - self.v[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            dOmega = np.where(cos_theta > 1e-9, du * dv / cos_theta, 0.0)
        prad = float(np.sum(intens * dOmega))
        if prad <= 0.0:
            return np.zeros_like(intens)
        return 4.0 * np.pi * intens / prad

    def peak_directivity_dbi(self) -> float:
        d = self.directivity()
        peak = float(np.max(d))
        return float(10.0 * np.log10(max(peak, 1e-30)))

    def co_cross(
        self, polarization: str = "x"
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Project onto a (co, cross) basis. polarization ∈ {'x','y','rcp','lcp'}."""
        if polarization == "x":
            return self.Ex, self.Ey
        if polarization == "y":
            return self.Ey, self.Ex
        if polarization == "rcp":
            er = (self.Ex - 1j * self.Ey) / np.sqrt(2.0)
            el = (self.Ex + 1j * self.Ey) / np.sqrt(2.0)
            return er, el
        if polarization == "lcp":
            er = (self.Ex - 1j * self.Ey) / np.sqrt(2.0)
            el = (self.Ex + 1j * self.Ey) / np.sqrt(2.0)
            return el, er
        raise ValueError(f"unknown polarization {polarization!r}")


def far_field_from_jones_aperture(
    aperture: JonesApertureField, pad_factor: int = 4
) -> JonesFarField:
    """Dual-pol far-field via component-wise FFT of (Ex, Ey).

    Each component is propagated independently using the same +j antenna-
    convention kernel as the scalar :func:`far_field_from_aperture`.
    """
    if pad_factor < 1:
        raise ValueError("pad_factor must be ≥ 1")
    grid = aperture.grid
    freq = aperture.freq

    nx_pad = grid.nx * pad_factor
    ny_pad = grid.ny * pad_factor
    if nx_pad % 2:
        nx_pad += 1
    if ny_pad % 2:
        ny_pad += 1

    def _spectrum(field: NDArray[np.complex128]) -> NDArray[np.complex128]:
        padded = np.zeros((ny_pad, nx_pad), dtype=np.complex128)
        y0 = (ny_pad - grid.ny) // 2
        x0 = (nx_pad - grid.nx) // 2
        padded[y0 : y0 + grid.ny, x0 : x0 + grid.nx] = field
        return (
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded)))
            * grid.dx
            * grid.dy
            * nx_pad
            * ny_pad
        )

    spec_x = _spectrum(aperture.Ex)
    spec_y = _spectrum(aperture.Ey)

    k = k0(freq)
    fx = np.fft.fftshift(np.fft.fftfreq(nx_pad, d=grid.dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny_pad, d=grid.dy))
    u = 2.0 * np.pi * fx / k
    v = 2.0 * np.pi * fy / k
    U, V = np.meshgrid(u, v, indexing="xy")
    cos_theta = np.sqrt(np.clip(1.0 - U**2 - V**2, 0.0, 1.0))
    factor = 1j * k / (2.0 * np.pi) * cos_theta

    return JonesFarField(
        u=u,
        v=v,
        U=U,
        V=V,
        Ex=factor * spec_x,
        Ey=factor * spec_y,
        freq=freq,
        aperture_power=aperture.power(),
    )
