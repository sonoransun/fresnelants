"""Reconfigurable Intelligent Surface (RIS) design.

`ReconfigurableArray` is a `Reflectarray` whose per-cell phase is determined
not by a fixed lookup but by a runtime *state* combined with a `TunableCell`
model. `state` is broadcast against the (ny, nx) cell grid; pass:

* ``None`` → idealized continuous phase (cell model bypassed; use as a sanity
  baseline).
* a scalar → all cells at the same state.
* an (ny, nx) array → per-cell state.

`CodedRIS` enforces N-bit quantization to the cell's actual achievable states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..cells.base import TunableCell
from ..cells.varactor import Skyworks_SMV1232
from ..coding import nearest_state
from ..core.geometry import ApertureGrid
from .reflectarray import Reflectarray


@dataclass
class ReconfigurableArray(Reflectarray):
    """Reflectarray driven by a per-cell `TunableCell` + state."""

    cell: TunableCell = field(default_factory=Skyworks_SMV1232)
    """Tunable-cell model determining each cell's phase / loss vs state."""
    name: str = "ReconfigurableArray"

    def with_beam_direction(self, beam: tuple[float, float]) -> ReconfigurableArray:
        """Return a copy steering to the given (theta, phi) direction."""
        from copy import copy as shallow_copy

        new = shallow_copy(self)
        new.beam_direction = beam
        return new

    def state_for_beam(self, freq: float) -> NDArray:
        """Cell states that steer the array to its configured beam direction."""
        target_phase = self.required_phase_per_cell(freq)
        return nearest_state(target_phase, self.cell, freq)

    def transmittance(
        self, grid: ApertureGrid, freq: float, state: object | None = None
    ) -> NDArray[np.complex128]:
        """Per-cell reflection coefficient using cell model + state.

        If `state` is None, derives a beam-steering state via
        :meth:`state_for_beam` for the configured beam direction. If `state`
        is a scalar or array, uses it directly.
        """
        x, y = self.cell_centers()
        if state is None:
            state_arr = self.state_for_beam(freq)
        elif np.isscalar(state):
            state_arr = np.full((self.ny, self.nx), state)
        else:
            state_arr = np.asarray(state)
            if state_arr.shape != (self.ny, self.nx):
                raise ValueError(f"state shape {state_arr.shape} != cell grid {(self.ny, self.nx)}")

        # Compute reflection coefficient per cell, then map onto the grid.
        gamma = self.cell.reflection_coefficient(state_arr, freq).astype(np.complex128)
        ix = np.clip(np.round((grid.X - x[0]) / self.cell_size).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(np.round((grid.Y - y[0]) / self.cell_size).astype(np.int64), 0, self.ny - 1)
        T = gamma[iy, ix]
        Lx, Ly = self.aperture_size
        outside = (np.abs(grid.X) > Lx / 2) | (np.abs(grid.Y) > Ly / 2)
        T[outside] = 0.0
        return T


@dataclass
class CodedRIS(ReconfigurableArray):
    """N-bit RIS — cell states quantized to ``2 ** bits`` choices.

    For PIN-diode cells (which have only 2 native states) the bits parameter
    is overridden to 1; for varactor / LC cells the constraint is enforced by
    sampling the cell's achievable phase range at the requested resolution.
    """

    bits: int = 1
    name: str = "CodedRIS"

    def state_for_beam(self, freq: float) -> NDArray:
        """Quantize the continuous beam-steering states to 2^bits levels."""
        target_phase = self.required_phase_per_cell(freq)
        if self.bits <= 0:
            raise ValueError("bits must be ≥ 1")
        # Get the cell's full state set, then sub-sample to 2^bits points across
        # the achievable phase range.
        states, kind = self.cell.state_set(freq)
        states_arr = np.asarray(states)
        if kind == "discrete" and len(states_arr) <= 2**self.bits:
            sub_states = states_arr
        else:
            phases = self.cell.phase(states_arr, freq)
            wrapped = np.mod(phases, 2.0 * np.pi)
            order = np.argsort(wrapped)
            n_levels = 2**self.bits
            picks = np.linspace(0, len(wrapped) - 1, n_levels).astype(int)
            sub_states = states_arr[order][picks]

        # For each target phase, choose the closest sub_state.
        cell_phase_lookup = self.cell.phase(sub_states, freq)
        flat = target_phase.flatten()
        dist = np.minimum(
            np.abs(
                np.mod(flat[:, None], 2 * np.pi) - np.mod(cell_phase_lookup[None, :], 2 * np.pi)
            ),
            2 * np.pi
            - np.abs(
                np.mod(flat[:, None], 2 * np.pi) - np.mod(cell_phase_lookup[None, :], 2 * np.pi)
            ),
        )
        idx = np.argmin(dist, axis=1)
        return sub_states[idx].reshape(target_phase.shape)
