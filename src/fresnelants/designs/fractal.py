"""Fractal Fresnel zone plate antennas.

Six related designs implemented in this module:

* :class:`FractalSoretZonePlate` — 2D triadic-Cantor binary mask (Soret-style).
* :class:`FractalWoodZonePlate`  — 2D triadic-Cantor phase-reversal (Devil's lens).
* :class:`SierpinskiCarpetZonePlate` — 2D Cartesian fractal mask on a square aperture.
* :class:`SierpinskiReflectarray` — fractal-tiled microstrip reflectarray.
* :class:`SphericalFractalFresnelLens` — 3D conformal Cantor-zoned spherical cap.
* :class:`ConicalFractalFresnelLens`   — 3D conformal Cantor-zoned cone.

The Cantor zone plate has a unique multifocal signature — equally spaced
on-axis foci at z = F, F/3, F/5, … — that no other family in the package
exhibits. It doubles as the load-bearing physical regression test (see
``tests/test_designs.py::test_fractal_cantor_polyfocal_signature``).

References
----------
* Saavedra, Furlan & Monsoriu, "Fractal zone plates", *Opt. Lett.* 28, 971
  (2003).
* Monsoriu, Saavedra & Furlan, "Fractal Devil's lenses", *J. Opt. Soc. Am. A*
  24, 3500 (2007).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.conformal import ConformalAperture
from ..core.geometry import ApertureGrid, fresnel_zone_radii
from ..units import freq_to_wavelength, k0
from .base import AntennaDesign
from .reflectarray import Reflectarray


def cantor_zone_indices(stage: int, base_unit: int = 1) -> list[tuple[int, int]]:
    """Retained ``[n_lo, n_hi)`` integer Fresnel-zone-index intervals after
    *stage* triadic-Cantor middle-thirds iterations.

    Starting from the single interval ``[0, base_unit · 3^stage)`` representing
    the full disk in Fresnel-zone units, each iteration trisects every retained
    interval and discards the middle third. Stage S yields ``2^S`` intervals,
    each of width ``base_unit``, total span ``base_unit · 3^S``.
    """
    if stage < 0:
        raise ValueError("stage must be ≥ 0")
    if base_unit < 1:
        raise ValueError("base_unit must be ≥ 1")
    intervals: list[tuple[int, int]] = [(0, base_unit * 3**stage)]
    for _ in range(stage):
        next_intervals: list[tuple[int, int]] = []
        for lo, hi in intervals:
            third = (hi - lo) // 3
            next_intervals.append((lo, lo + third))
            next_intervals.append((lo + 2 * third, hi))
        intervals = next_intervals
    return intervals


def sierpinski_mask(stage: int) -> NDArray[np.bool_]:
    """Stage-S Sierpinski-carpet boolean mask of shape ``(3^S, 3^S)``.

    True = retained cell (transparent / active); False = removed cell. Built
    via Kronecker product of the 3×3 base pattern with itself ``stage`` times.
    Stage 0 → ``(1, 1)`` all-True. Stage 1 → centered hole. Stage S retains
    ``8^S`` cells out of ``9^S``.
    """
    if stage < 0:
        raise ValueError("stage must be ≥ 0")
    base = np.array(
        [[True, True, True], [True, False, True], [True, True, True]],
        dtype=bool,
    )
    mask = np.ones((1, 1), dtype=bool)
    for _ in range(stage):
        mask = np.kron(mask, base)
    return mask.astype(np.bool_)


def _cantor_radial_signs(
    rho: NDArray[np.float64],
    focal_length: float,
    wavelength: float,
    stage: int,
    base_unit: int,
    phase_reversal: bool,
) -> NDArray[np.float64]:
    """Cantor-zone activation pattern in a radial coordinate ``rho``.

    Returns a real array the same shape as ``rho`` containing ``±1`` on
    retained Fresnel zones and ``0`` elsewhere. For ``phase_reversal=False``
    (Soret-like) all retained zones carry ``+1``. For ``phase_reversal=True``
    (Wood-like) the multiplier follows per-Fresnel-zone parity — odd zones
    keep ``+1``, even zones get ``-1`` — which restores coherent focusing
    when retained intervals contain both odd and even Fresnel zones (i.e.
    ``base_unit ≥ 2``). With ``base_unit = 1`` every retained Fresnel zone
    in the Cantor construction is odd, so the Wood and Soret variants
    degenerate to the same mask — this is a real property of the standard
    triadic Cantor zone plate, not a bug.
    """
    num_zones = base_unit * 3**stage
    radii = fresnel_zone_radii(num_zones, focal_length, wavelength)
    intervals = cantor_zone_indices(stage, base_unit)
    out = np.zeros_like(rho, dtype=np.float64)
    for n_lo, n_hi in intervals:
        # 1-indexed Fresnel zones contained in the interval [n_lo, n_hi):
        # zones n_lo+1, n_lo+2, …, n_hi.
        for n_zone in range(n_lo + 1, n_hi + 1):
            r_inner = 0.0 if n_zone == 1 else float(radii[n_zone - 2])
            r_outer = float(radii[n_zone - 1])
            in_zone = (rho > r_inner) & (rho <= r_outer)
            if phase_reversal:
                out[in_zone] = 1.0 if (n_zone % 2 == 1) else -1.0
            else:
                out[in_zone] = 1.0
    return out


@dataclass
class FractalSoretZonePlate(AntennaDesign):
    """Cantor-set zone plate, binary amplitude variant (Soret-like).

    Replaces the standard Fresnel-zone partition with the triadic Cantor set
    in the squared-radius coordinate ζ = r²/(λF). Retains ``2^stage`` annuli
    out of ``base_unit · 3^stage`` Fresnel zones; retained zones transmit with
    unit amplitude, all others are opaque. Exhibits multifocal on-axis
    intensity peaks at z = F, F/3, F/5, …
    """

    focal_length: float
    design_freq: float
    stage: int = 3
    base_unit: int = 1
    name: str = "FractalSoretZonePlate"

    @property
    def wavelength(self) -> float:
        return float(freq_to_wavelength(self.design_freq))

    @property
    def num_zones(self) -> int:
        return int(self.base_unit * 3**self.stage)

    @property
    def zone_radii(self) -> NDArray[np.float64]:
        return fresnel_zone_radii(self.num_zones, self.focal_length, self.wavelength)

    @property
    def aperture_radius(self) -> float:
        return float(self.zone_radii[-1])

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        signs = _cantor_radial_signs(
            grid.R,
            self.focal_length,
            self.wavelength,
            self.stage,
            self.base_unit,
            phase_reversal=False,
        )
        return signs.astype(np.complex128)


@dataclass
class FractalWoodZonePlate(AntennaDesign):
    """Cantor-set zone plate, phase-reversal variant ("Devil's lens").

    Same Cantor partition as :class:`FractalSoretZonePlate`, but alternating
    retained annuli carry ``+1`` / ``−1`` instead of being blanked. Yields the
    same multifocal axial signature with ~4× primary-focus efficiency.
    """

    focal_length: float
    design_freq: float
    stage: int = 3
    base_unit: int = 1
    name: str = "FractalWoodZonePlate"

    @property
    def wavelength(self) -> float:
        return float(freq_to_wavelength(self.design_freq))

    @property
    def num_zones(self) -> int:
        return int(self.base_unit * 3**self.stage)

    @property
    def zone_radii(self) -> NDArray[np.float64]:
        return fresnel_zone_radii(self.num_zones, self.focal_length, self.wavelength)

    @property
    def aperture_radius(self) -> float:
        return float(self.zone_radii[-1])

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        signs = _cantor_radial_signs(
            grid.R,
            self.focal_length,
            self.wavelength,
            self.stage,
            self.base_unit,
            phase_reversal=True,
        )
        return signs.astype(np.complex128)


@dataclass
class SierpinskiCarpetZonePlate(AntennaDesign):
    """Square-aperture Sierpinski-carpet binary mask.

    Demonstrates a Cartesian (rather than radial) fractal Fresnel mask. The
    aperture of side ``aperture_side`` is tiled with a stage-S carpet on a
    ``3^S × 3^S`` cell grid; retained cells transmit. The far-field carries
    the carpet's self-similar 4-fold-symmetric Fourier signature.
    """

    focal_length: float
    design_freq: float
    stage: int = 3
    aperture_side: float = 0.20
    name: str = "SierpinskiCarpetZonePlate"

    @property
    def wavelength(self) -> float:
        return float(freq_to_wavelength(self.design_freq))

    @property
    def aperture_radius(self) -> float:
        # Half-diagonal of the square aperture (used for grid-extent sizing).
        return float(self.aperture_side * np.sqrt(2.0) / 2.0)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        L = self.aperture_side
        N = 3**self.stage
        mask = sierpinski_mask(self.stage)
        ix = np.clip(((grid.X + L / 2) / L * N).astype(np.int64), 0, N - 1)
        iy = np.clip(((grid.Y + L / 2) / L * N).astype(np.int64), 0, N - 1)
        T = np.zeros(grid.shape, dtype=np.complex128)
        T[mask[iy, ix]] = 1.0 + 0.0j
        outside = (np.abs(grid.X) > L / 2) | (np.abs(grid.Y) > L / 2)
        T[outside] = 0.0 + 0.0j
        return T


@dataclass
class SierpinskiReflectarray(Reflectarray):
    """Reflectarray whose active cells follow a Sierpinski-carpet pattern.

    Subclasses :class:`Reflectarray` and AND-masks the per-cell focusing
    transmittance with a stage-S Sierpinski carpet sampled at the array's
    cell grid. Retains the parent's beam-steering / focusing math; removed
    cells contribute zero reflection (modeling absorbing or off-state cells).

    The mask is rescaled (via nearest-neighbor mapping) to match the
    underlying ``(ny, nx)`` array shape so it works for non-power-of-3 cell
    counts; pure ``nx = ny = 3^stage`` arrays are handled exactly.
    """

    fractal_stage: int = 2
    name: str = "SierpinskiReflectarray"

    def _cell_mask(self) -> NDArray[np.bool_]:
        """(ny, nx) boolean mask after nearest-neighbor rescale of the carpet."""
        carpet = sierpinski_mask(self.fractal_stage)  # (M, M)
        M = carpet.shape[0]
        iy = np.clip((np.arange(self.ny) * M / self.ny).astype(np.int64), 0, M - 1)
        ix = np.clip((np.arange(self.nx) * M / self.nx).astype(np.int64), 0, M - 1)
        return np.asarray(carpet[iy[:, None], ix[None, :]], dtype=np.bool_)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        T = super().transmittance(grid, freq, state)
        # AND-mask the per-cell phase with the Sierpinski activation.
        x, y = self.cell_centers()
        mask = self._cell_mask()
        ix = np.clip(np.round((grid.X - x[0]) / self.cell_size).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(np.round((grid.Y - y[0]) / self.cell_size).astype(np.int64), 0, self.ny - 1)
        active = mask[iy, ix]
        T = np.where(active, T, 0.0 + 0.0j)
        return T.astype(np.complex128)


@dataclass
class SphericalFractalFresnelLens(AntennaDesign):
    """Spherical-cap conformal Fresnel lens with Cantor-zoned aperture.

    Reuses the spherical-cap mesh of :class:`SphericalFresnelLens` plus the
    standard ``-k·d_feed + (-k·z)`` focusing phase, but multiplies the
    tangential field by a Cantor activation pattern in the projected radial
    coordinate ρ = √(x² + y²). With ``phase_reversal=True`` it implements a
    spherical Devil's lens.
    """

    radius: float = 0.05
    cap_angle_deg: float = 90.0
    design_freq: float = 77e9
    stage: int = 2
    base_unit: int = 1
    phase_reversal: bool = True
    nu: int = 128
    nv: int = 40
    name: str = "SphericalFractalFresnelLens"
    focal_length: float = 0.0

    def __post_init__(self) -> None:
        if self.focal_length == 0.0:
            self.focal_length = self.radius

    @property
    def aperture_radius(self) -> float:
        return float(self.radius * np.sin(np.deg2rad(self.cap_angle_deg)))

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        raise NotImplementedError(
            "SphericalFractalFresnelLens uses the conformal pipeline; call conformal_aperture()."
        )

    def conformal_aperture(self, freq: float, state: object | None = None) -> ConformalAperture:
        ap = ConformalAperture.from_sphere(
            radius=self.radius,
            cap_angle=np.deg2rad(self.cap_angle_deg),
            nu=self.nu,
            nv=self.nv,
            freq=freq,
        )
        d_feed = np.linalg.norm(ap.points, axis=1)
        target_phase = -k0(freq) * ap.points[:, 2]
        cell_phase = -k0(freq) * d_feed + target_phase

        rho = np.hypot(ap.points[:, 0], ap.points[:, 1])
        wavelength = float(freq_to_wavelength(self.design_freq))
        signs = _cantor_radial_signs(
            rho,
            self.focal_length,
            wavelength,
            self.stage,
            self.base_unit,
            self.phase_reversal,
        )
        Et = (signs * np.exp(1j * cell_phase)).astype(np.complex128)
        return ap.with_field(Et)


@dataclass
class ConicalFractalFresnelLens(AntennaDesign):
    """Conical conformal Fresnel lens with Cantor-zoned aperture.

    A right-circular cone (apex at origin, opening toward +z) carrying a
    ``-k·d_feed + (-k·z)`` focusing phase modulated by the Cantor activation
    pattern in the projected radial coordinate ρ = z·tan(half_angle). The
    feed sits at the apex; the design steers a +z plane wave.
    """

    half_angle_deg: float = 45.0
    height: float = 0.05
    design_freq: float = 77e9
    stage: int = 2
    base_unit: int = 1
    phase_reversal: bool = True
    nu: int = 128
    nv: int = 60
    name: str = "ConicalFractalFresnelLens"
    focal_length: float = 0.0

    def __post_init__(self) -> None:
        if self.focal_length == 0.0:
            self.focal_length = self.height

    @property
    def aperture_radius(self) -> float:
        return float(self.height * np.tan(np.deg2rad(self.half_angle_deg)))

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        raise NotImplementedError(
            "ConicalFractalFresnelLens uses the conformal pipeline; call conformal_aperture()."
        )

    def conformal_aperture(self, freq: float, state: object | None = None) -> ConformalAperture:
        ap = ConformalAperture.from_cone(
            half_angle=np.deg2rad(self.half_angle_deg),
            height=self.height,
            nu=self.nu,
            nv=self.nv,
            freq=freq,
        )
        d_feed = np.linalg.norm(ap.points, axis=1)
        target_phase = -k0(freq) * ap.points[:, 2]
        cell_phase = -k0(freq) * d_feed + target_phase

        rho = np.hypot(ap.points[:, 0], ap.points[:, 1])
        wavelength = float(freq_to_wavelength(self.design_freq))
        signs = _cantor_radial_signs(
            rho,
            self.focal_length,
            wavelength,
            self.stage,
            self.base_unit,
            self.phase_reversal,
        )
        Et = (signs * np.exp(1j * cell_phase)).astype(np.complex128)
        return ap.with_field(Et)
