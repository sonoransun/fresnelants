"""Composite multi-stage Fresnel systems.

A :class:`CompositeAntenna` chains an ordered sequence of layers (each layer
holds an :class:`AntennaDesign` and a `z_offset` from the upstream feed). The
composite synthesizes its emergent aperture field by propagating an initial
illumination through each layer in turn — applying the layer's transmittance,
then free-space-propagating to the next layer's z position.

Three pre-built composite types ship with the library:

* :class:`AchromaticDoublet` — two phase-correcting plates whose chromatic
  dispersion partially cancels, broadening usable bandwidth.
* :class:`BifocalLens` — interleaved zone sets focusing to two distinct
  points, useful for shared-aperture comm/radar systems.
* :class:`FoldedReflectarray` — a Cassegrain-style folded path using a
  twist-reflector subreflector + reflectarray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..analysis.aperture import ApertureField
from ..analysis.cascade import apply_transmittance, propagate
from ..core.geometry import ApertureGrid, make_aperture_grid
from ..core.wavefront import SphericalWave
from ..units import freq_to_wavelength
from .base import AntennaDesign
from .phase_correcting import PhaseCorrectingPlate
from .reflectarray import Reflectarray


class Layer(NamedTuple):
    """One layer in a composite stack."""

    design: AntennaDesign
    z_offset: float = 0.0
    """Distance from the previous layer (or feed) to this layer, in metres."""
    reflection_coefficient: float = 0.0
    """Fraction (0 .. 1) of the incident field reflected back upstream. 0 →
    purely transmissive (forward-only model); 1 → perfect mirror. Only used
    when CompositeAntenna.aperture_field is called with `bounces > 1`."""


@dataclass
class CompositeAntenna(AntennaDesign):
    """Ordered cascade of antenna designs separated by free-space gaps."""

    layers: list[Layer] = field(default_factory=list)
    feed_distance: float = 0.10
    """Distance from feed to the first layer [m]."""
    design_freq: float = 30e9
    name: str = "CompositeAntenna"

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("CompositeAntenna requires at least one layer.")
        # focal_length is the distance from feed to the last layer's emission.
        self.focal_length = self.feed_distance + sum(layer.z_offset for layer in self.layers)

    @property
    def aperture_radius(self) -> float:
        return float(max(layer.design.aperture_radius for layer in self.layers))

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        """Composites do not expose a single-layer transmittance.

        Use :meth:`aperture_field` directly; the cascade is propagated layer
        by layer with intermediate angular-spectrum steps.
        """
        raise NotImplementedError(
            "CompositeAntenna does not have a single transmittance — use `aperture_field()`."
        )

    def aperture_field(
        self,
        freq: float,
        *,
        grid: ApertureGrid | None = None,
        samples_per_wavelength: float = 6.0,
        margin: float = 1.2,
        illumination: NDArray[np.complex128] | float | None = None,
        state: object | None = None,
        bounces: int = 1,
    ) -> ApertureField:
        """Propagate feed → layer₁ → … → layerN, return the emergent aperture.

        Parameters
        ----------
        bounces
            ``1`` → forward-only (default, fast).
            ``3`` → forward + one back-bounce + one forward retransmission.
                Captures the dominant Cassegrain / folded-RA bounce when at
                least one layer has a non-zero ``reflection_coefficient``.
            Higher (odd) values iterate further but rarely add accuracy.
        """
        if grid is None:
            wavelength = float(freq_to_wavelength(freq))
            extent = 2.0 * self.aperture_radius * margin
            grid = make_aperture_grid(extent, samples_per_wavelength, wavelength)

        if illumination is None:
            feed = SphericalWave(z0=-self.feed_distance)
            inc = feed.field_on(grid, freq)
        elif np.isscalar(illumination):
            inc = np.full(grid.shape, complex(illumination), dtype=np.complex128)
        else:
            inc = np.asarray(illumination, dtype=np.complex128)

        states: list[object | None]
        if state is None or not isinstance(state, list):
            states = [state] * len(self.layers)
        else:
            states = list(state)

        # Forward pass.
        field = ApertureField(grid=grid, Ez=inc, freq=freq)
        forward_fields: list[ApertureField] = [field]  # field arriving at layer i
        for layer, layer_state in zip(self.layers, states, strict=True):
            T = layer.design.transmittance(grid, freq, layer_state)
            field = apply_transmittance(field, T)
            if layer.z_offset > 0:
                field = propagate(field, layer.z_offset)
            forward_fields.append(field)

        if bounces <= 1:
            return field

        # Multi-bounce: alternate backward and forward sweeps. We accumulate
        # the additional contribution into `field` (the existing forward
        # output is the 0-bounce term).
        for _ in range((bounces - 1) // 2):
            # Backward sweep — start from the last layer and walk upstream,
            # picking up each layer's reflection_coefficient × field.
            back = ApertureField(grid=grid, Ez=np.zeros(grid.shape, np.complex128), freq=freq)
            for i, layer in enumerate(reversed(self.layers), start=1):
                idx = len(self.layers) - i
                rc = layer.reflection_coefficient
                if rc != 0.0:
                    arrival = forward_fields[idx + 1]
                    back = back.with_field(back.Ez + rc * arrival.Ez)
                if idx > 0 and self.layers[idx].z_offset > 0:
                    back = propagate(back, -self.layers[idx].z_offset)
            # Forward retransmission of the backward field through every layer.
            f2 = back
            for layer, layer_state in zip(self.layers, states, strict=True):
                T = layer.design.transmittance(grid, freq, layer_state)
                f2 = apply_transmittance(f2, T)
                if layer.z_offset > 0:
                    f2 = propagate(f2, layer.z_offset)
            field = field.with_field(field.Ez + f2.Ez)

        return field


# -------- Pre-built composites ----------------------------------------------


@dataclass
class AchromaticDoublet(CompositeAntenna):
    """Two phase-correcting plates designed for f_low and f_high.

    The plates are placed back-to-back (zero gap). Their dispersion partially
    cancels, broadening the usable bandwidth at the cost of extra material.
    """

    f_low: float = 26e9
    f_high: float = 32e9
    aperture_radius_m: float = 0.05
    levels: int = 8
    name: str = "AchromaticDoublet"

    def __post_init__(self) -> None:
        if not self.layers:
            mid_freq = 0.5 * (self.f_low + self.f_high)
            F_eff = self.feed_distance
            plate_low = PhaseCorrectingPlate(
                focal_length=F_eff,
                design_freq=self.f_low,
                aperture_radius_m=self.aperture_radius_m,
                levels=self.levels,
            )
            plate_high = PhaseCorrectingPlate(
                focal_length=F_eff,
                design_freq=self.f_high,
                aperture_radius_m=self.aperture_radius_m,
                levels=self.levels,
            )
            self.layers = [Layer(plate_low, 0.0), Layer(plate_high, 0.0)]
            self.design_freq = mid_freq
        super().__post_init__()


@dataclass
class BifocalLens(CompositeAntenna):
    """Two interleaved phase profiles focusing to distinct points.

    Implementation detail: rather than two physical layers, we encode both
    phase profiles into a single layer whose transmittance is the *coherent
    average* of the two — which yields equal-power beams toward each focus
    point in the far field. Useful for V2X / radar-comm shared apertures.
    """

    focal_a: float = 0.10
    freq_a: float = 24e9
    focal_b: float = 0.10
    freq_b: float = 77e9
    aperture_radius_m: float = 0.05
    levels: int = 8
    split: float = 0.5
    name: str = "BifocalLens"

    def __post_init__(self) -> None:
        if not self.layers:
            plate_a = PhaseCorrectingPlate(
                focal_length=self.focal_a,
                design_freq=self.freq_a,
                aperture_radius_m=self.aperture_radius_m,
                levels=self.levels,
            )
            plate_b = PhaseCorrectingPlate(
                focal_length=self.focal_b,
                design_freq=self.freq_b,
                aperture_radius_m=self.aperture_radius_m,
                levels=self.levels,
            )
            mix = _MixedTransmittance(plate_a, plate_b, self.split)
            self.layers = [Layer(mix, 0.0)]
            self.feed_distance = self.focal_a
            self.design_freq = self.freq_a
        super().__post_init__()


@dataclass
class _MixedTransmittance(AntennaDesign):
    """Internal helper for BifocalLens — coherent-mix of two designs' phase."""

    plate_a: AntennaDesign | None = None
    plate_b: AntennaDesign | None = None
    split: float = 0.5

    def __post_init__(self) -> None:
        if self.plate_a is None or self.plate_b is None:
            raise ValueError("_MixedTransmittance requires plate_a and plate_b.")
        # Use plate_a's focal_length and design_freq as the canonical handle.
        self.focal_length = self.plate_a.focal_length
        self.design_freq = self.plate_a.design_freq

    @property
    def aperture_radius(self) -> float:
        assert self.plate_a is not None and self.plate_b is not None
        return max(self.plate_a.aperture_radius, self.plate_b.aperture_radius)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        assert self.plate_a is not None and self.plate_b is not None
        Ta = self.plate_a.transmittance(grid, freq, state)
        Tb = self.plate_b.transmittance(grid, freq, state)
        return (np.sqrt(self.split) * Ta + np.sqrt(1.0 - self.split) * Tb).astype(np.complex128)


@dataclass
class FoldedReflectarray(CompositeAntenna):
    """Cassegrain-style folded reflectarray.

    Models a polarizer + reflectarray pair: the wave passes through the
    polarizer toward the reflectarray, reflects with a 90° polarization twist,
    and exits through the polarizer (now seeing the orthogonal polarization).
    The composite halves the focal-length-equivalent of an unfolded design.

    For this PO-level model we treat the polarizer-twist round trip as two
    forward-cascade operations through an effective doubled-thickness plate
    of the reflectarray, producing the canonical folded-aperture result.
    """

    reflectarray_focal: float = 0.20
    design_freq_ra: float = 28e9
    nx: int = 32
    ny: int = 32
    polarizer_gap: float = 0.05
    name: str = "FoldedReflectarray"

    def __post_init__(self) -> None:
        if not self.layers:
            ra = Reflectarray(
                focal_length=self.reflectarray_focal,
                design_freq=self.design_freq_ra,
                nx=self.nx,
                ny=self.ny,
            )
            polarizer = _IdealPolarizer(
                aperture_radius_m=ra.aperture_radius,
                design_freq=self.design_freq_ra,
            )
            self.layers = [
                Layer(polarizer, 0.0),
                Layer(ra, self.polarizer_gap),
                Layer(polarizer, self.polarizer_gap),
            ]
            self.feed_distance = self.reflectarray_focal - 2 * self.polarizer_gap
            self.design_freq = self.design_freq_ra
        super().__post_init__()


@dataclass
class _IdealPolarizer(AntennaDesign):
    """Ideal twist-polarizer modelled as a unit-amplitude transmission plate."""

    aperture_radius_m: float = 0.10
    design_freq: float = 28e9
    focal_length: float = 0.0  # positional placeholder

    @property
    def aperture_radius(self) -> float:
        return float(self.aperture_radius_m)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        T = np.ones(grid.shape, dtype=np.complex128)
        T[self.aperture_radius < grid.R] = 0.0
        return T

    def default_illumination(self, grid: ApertureGrid, freq: float) -> NDArray[np.complex128]:
        return np.ones(grid.shape, dtype=np.complex128)
