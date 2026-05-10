"""Macro arrays of Fresnel-antenna elements (phased-array receivers).

A `MacroFresnelArray` composes N copies of any `AntennaDesign` (a Fresnel
zone plate, reflectarray, conformal lens, fractal — anything subclassing
the base class) into a phased array with complex per-element weights.
This is the architectural level *above* the per-cell phased arrays
already in the package (`Reflectarray`, `ReconfigurableArray`, …): each
"element" of a macro array is itself a complete Fresnel antenna with its
own aperture many wavelengths across.

The radiation pattern is computed via the textbook **array-factor times
element-pattern** decomposition:

    E_total(θ, φ) = E_element(θ, φ) · AF(θ, φ; w_i, positions)

valid when all elements are identical and mutual coupling is negligible
(typical for sparse aperture arrays at multi-wavelength spacing). The
formulation is O(N) per (u, v) sample and trivially scales to N=128+.

References
----------
* Mailloux, *Phased Array Antenna Handbook*, 3rd ed. (Artech House, 2018).
* Hansen, *Phased Array Antennas*, 2nd ed. (Wiley, 2009).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..analysis.array_factor import array_factor, quantize_weights, steering_vector
from ..analysis.farfield import FarField
from ..core.geometry import ApertureGrid, element_lattice_positions
from ..units import freq_to_wavelength
from .base import AntennaDesign


@dataclass
class MacroArrayResult:
    """Far-field + bookkeeping for a `MacroFresnelArray.solve(...)` call."""

    far_field: FarField
    weights: NDArray[np.complex128]
    element_far_field: FarField
    array_factor_grid: NDArray[np.complex128]


@dataclass
class MacroFresnelArray(AntennaDesign):
    """N-element phased array of identical Fresnel-antenna elements.

    Parameters
    ----------
    element
        Prototype `AntennaDesign` — every element is conceptually a copy of
        this design. Must expose `aperture_radius` and `design_freq`.
    element_positions
        ``(N, 2)`` array of (x, y) element-center positions [m]. Build via
        :func:`fresnelants.core.geometry.element_lattice_positions`.
    weights
        Optional ``(N,)`` complex weight vector. Default: uniform = 1.
    coupling_q
        Mutual-coupling first-order correction strength. Default 0
        (no correction). When > 0, the per-element pattern is scaled by
        ``sqrt(1 − Q²·exp(−2·d_min/λ))`` where ``d_min`` is the minimum
        element-edge separation. This is a **trend model** (Carver & Mink
        1981, Mailloux ch. 8), not a full-wave coupling computation.
    """

    element: AntennaDesign = field(default_factory=lambda: None)  # type: ignore[arg-type]
    element_positions: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((1, 2), dtype=np.float64)
    )
    weights: NDArray[np.complex128] | None = None
    coupling_q: float = 0.0
    name: str = "MacroFresnelArray"
    focal_length: float = 0.0
    design_freq: float = 0.0

    def __post_init__(self) -> None:
        if self.element is None:
            raise ValueError("MacroFresnelArray requires a prototype `element`.")
        # Inherit canonical handles from the prototype element.
        if self.focal_length == 0.0:
            self.focal_length = float(getattr(self.element, "focal_length", 0.0))
        if self.design_freq == 0.0:
            self.design_freq = float(self.element.design_freq)
        positions = np.asarray(self.element_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"element_positions must be (N, 2); got {positions.shape}")
        self.element_positions = positions

    @property
    def n_elements(self) -> int:
        return int(self.element_positions.shape[0])

    @property
    def aperture_radius(self) -> float:
        """Bounding-circle radius of the array footprint plus element extent."""
        if self.n_elements == 0:
            return float(self.element.aperture_radius)
        max_lattice = float(np.max(np.linalg.norm(self.element_positions, axis=1)))
        return max_lattice + float(self.element.aperture_radius)

    @property
    def array_extent(self) -> tuple[float, float]:
        """(x_extent, y_extent) bounding-box side lengths [m]."""
        x = self.element_positions[:, 0]
        y = self.element_positions[:, 1]
        return (
            float(x.max() - x.min()) + 2.0 * float(self.element.aperture_radius),
            float(y.max() - y.min()) + 2.0 * float(self.element.aperture_radius),
        )

    @property
    def min_neighbour_spacing(self) -> float:
        """Minimum centre-to-centre distance between any two elements [m]."""
        if self.n_elements < 2:
            return float("inf")
        # O(N²) but cheap for N ≤ 128 and avoids a scipy dependency.
        diffs = self.element_positions[:, None, :] - self.element_positions[None, :, :]
        d2 = np.sum(diffs**2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        return float(np.sqrt(d2.min()))

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        """Macro arrays use the array-factor pipeline — see :meth:`solve`.

        The per-element transmittance is exposed via ``self.element``; this
        method intentionally raises so callers don't accidentally compute the
        single-element pattern when they meant the array.
        """
        raise NotImplementedError(
            "MacroFresnelArray uses the array-factor pipeline; call "
            "`.solve(solver, freq, weights=...)` to get the array's FarField, "
            "or `solver.solve(macro.element, freq)` to get just the element pattern."
        )

    def weights_for_beam(
        self,
        theta_deg: float,
        phi_deg: float = 0.0,
        freq: float | None = None,
        *,
        bits: int = 0,
    ) -> NDArray[np.complex128]:
        """Conjugate-matched receive weights for the target direction.

        With ``bits > 0`` the result is quantized to a ``2^bits``-state
        phase shifter (1-bit, 2-bit, …) — useful for analog/RFIC receivers.
        """
        f = float(freq) if freq is not None else float(self.design_freq)
        sv = steering_vector(theta_deg, phi_deg, f, self.element_positions)
        w = np.conj(sv)
        if bits > 0:
            w = quantize_weights(w, bits)
        return w

    def beam_codebook(
        self,
        directions: list[tuple[float, float]],
        freq: float | None = None,
        *,
        bits: int = 0,
        labels: list[str] | None = None,
    ) -> dict[str, NDArray[np.complex128]]:
        """Pre-compute a ``{label: weights}`` codebook for many directions.

        ``directions`` is a list of ``(theta_deg, phi_deg)`` tuples.
        ``labels``, if supplied, must match in length; otherwise labels
        default to ``"theta{θ:.0f}_phi{φ:.0f}"``.
        """
        if labels is not None and len(labels) != len(directions):
            raise ValueError("labels must match the number of directions")
        out: dict[str, NDArray[np.complex128]] = {}
        for i, (t, p) in enumerate(directions):
            w = self.weights_for_beam(t, p, freq, bits=bits)
            label = labels[i] if labels else f"theta{t:.0f}_phi{p:.0f}"
            out[label] = w
        return out

    def _coupling_scale(self, freq: float) -> float:
        """Heuristic per-element pattern scaling factor from coupling."""
        if self.coupling_q <= 0.0:
            return 1.0
        wavelength = float(freq_to_wavelength(self.design_freq))
        # Edge-to-edge separation; never let it go below 1 % λ.
        d_min = max(
            self.min_neighbour_spacing - 2.0 * float(self.element.aperture_radius),
            0.01 * wavelength,
        )
        gamma_sq = (self.coupling_q * np.exp(-d_min / wavelength)) ** 2
        return float(np.sqrt(max(0.0, 1.0 - gamma_sq)))

    def solve(
        self,
        solver: Any,
        freq: float,
        *,
        weights: NDArray[np.complex128] | None = None,
        bits: int = 0,
    ) -> MacroArrayResult:
        """Solve the macro-array far-field for the given complex weights.

        Parameters
        ----------
        solver
            A `PhysicalOpticsSolver` (or any solver exposing
            ``.solve(design, freq) → result_with_far_field``) used once on
            the prototype element to obtain the element pattern.
        freq
            Operating frequency [Hz].
        weights
            Per-element complex weights ``(N,)``. Defaults to the array's
            stored ``weights`` field, or uniform = 1 if also unset.
        bits
            If > 0, quantize *weights* to a ``2^bits``-state phase shifter
            before applying. Useful for emulating RFIC weight resolution.
        """
        if weights is None:
            weights = (
                self.weights
                if self.weights is not None
                else np.ones(self.n_elements, dtype=np.complex128)
            )
        weights = np.asarray(weights, dtype=np.complex128)
        if weights.shape != (self.n_elements,):
            raise ValueError(f"weights shape {weights.shape} != ({self.n_elements},)")
        if bits > 0:
            weights = quantize_weights(weights, bits)

        elem_result = solver.solve(self.element, freq)
        ff_elem: FarField = elem_result.far_field

        af = array_factor(ff_elem.u, ff_elem.v, freq, self.element_positions, weights)
        coupling = self._coupling_scale(freq)
        E_total = ff_elem.E * af * coupling

        # Total aperture power for downstream aperture_efficiency calls.
        ap_power_total = float(np.sum(np.abs(weights) ** 2)) * float(ff_elem.aperture_power)

        ff_total = FarField(
            u=ff_elem.u,
            v=ff_elem.v,
            U=ff_elem.U,
            V=ff_elem.V,
            E=E_total.astype(np.complex128),
            freq=freq,
            aperture_power=ap_power_total,
        )
        return MacroArrayResult(
            far_field=ff_total,
            weights=weights,
            element_far_field=ff_elem,
            array_factor_grid=af,
        )

    @classmethod
    def from_lattice(
        cls,
        element: AntennaDesign,
        n_elements: int,
        spacing_m: float,
        lattice: str = "linear",
        *,
        rows: int | None = None,
        weights: NDArray[np.complex128] | None = None,
        coupling_q: float = 0.0,
    ) -> MacroFresnelArray:
        """Convenience: build a MacroFresnelArray from a lattice descriptor."""
        positions = element_lattice_positions(n_elements, spacing_m, lattice, rows=rows)
        return cls(
            element=element,
            element_positions=positions,
            weights=weights,
            coupling_q=coupling_q,
        )
