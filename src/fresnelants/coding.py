"""RIS coding — quantize required phase to cell-achievable states; codebooks."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .cells.base import TunableCell


@dataclass(frozen=True, slots=True)
class BeamSpec:
    """A target main-beam direction."""

    theta_deg: float
    phi_deg: float = 0.0
    label: str | None = None

    @property
    def uv(self) -> tuple[float, float]:
        t = math.radians(self.theta_deg)
        p = math.radians(self.phi_deg)
        return math.sin(t) * math.cos(p), math.sin(t) * math.sin(p)


def nearest_state(target_phase: NDArray[np.float64], cell: TunableCell, freq: float) -> NDArray:
    """For each entry in *target_phase* (mod 2π), pick the cell state that best
    matches in phase. Works for continuous and discrete cells.
    """
    states, _kind = cell.state_set(freq)
    states_arr = np.asarray(states)
    # Sample cell phase across all candidate states.
    cell_phases = np.unwrap(cell.phase(states_arr, freq))
    flat = np.mod(np.asarray(target_phase, dtype=np.float64), 2.0 * np.pi).flatten()
    # Bring cell_phases into [0, 2π) for closest-by-circular-distance matching.
    cell_phases_mod = np.mod(cell_phases, 2.0 * np.pi)
    dist = np.minimum(
        np.abs(flat[:, None] - cell_phases_mod[None, :]),
        2 * np.pi - np.abs(flat[:, None] - cell_phases_mod[None, :]),
    )
    idx = np.argmin(dist, axis=1)
    chosen = states_arr[idx].reshape(target_phase.shape)
    return chosen


def beam_codebook(
    beams: Iterable[BeamSpec],
    cell: TunableCell,
    *,
    array_design: object,  # Reflectarray or ReconfigurableArray
    freq: float,
) -> dict[str, NDArray]:
    """Pre-compute a codebook of cell-state matrices for a list of beams.

    Returns a dict keyed by beam label (or `θ_φ` if no label given) with the
    (ny, nx) state matrix that steers the array to that direction.
    """
    if not hasattr(array_design, "required_phase_per_cell"):
        raise TypeError("array_design must expose required_phase_per_cell(freq).")
    book: dict[str, NDArray] = {}
    base_beam = getattr(array_design, "beam_direction", None)
    for beam in beams:
        # Ask the array what phase it needs for this direction.
        u, v = beam.uv
        beam_dir = (
            math.asin(min(1.0, math.hypot(u, v))),
            math.atan2(v, u) if (u or v) else 0.0,
        )
        # Mutate-in-place is brittle; use a local copy.
        if hasattr(array_design, "with_beam_direction"):
            tmp = array_design.with_beam_direction(beam_dir)
        else:
            tmp = array_design
            tmp.beam_direction = beam_dir
        phases = tmp.required_phase_per_cell(freq)
        states = nearest_state(phases, cell, freq)
        label = beam.label or f"theta{beam.theta_deg:.0f}_phi{beam.phi_deg:.0f}"
        book[label] = states
    if base_beam is not None and hasattr(array_design, "beam_direction"):
        array_design.beam_direction = base_beam  # restore
    return book


def save_codebook(book: dict[str, NDArray], path: str | Path) -> None:
    """Serialize a codebook to JSON. Arrays become nested lists."""
    serial = {k: v.tolist() for k, v in book.items()}
    Path(path).write_text(json.dumps(serial), encoding="utf-8")


def load_codebook(path: str | Path) -> dict[str, NDArray]:
    """Inverse of :func:`save_codebook`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: np.asarray(v) for k, v in raw.items()}


def dithered_phase(
    target_phase: NDArray[np.float64], cell: TunableCell, freq: float, frames: int = 8
) -> NDArray[np.float64]:
    """Time-dither a quantized cell sequence to recover effective resolution.

    Returns an (frames, *target_phase.shape) array of states; averaging across
    `frames` reproduces *target_phase* to better than the cell's bit depth.
    """
    states, _ = cell.state_set(freq)
    states_arr = np.asarray(states)
    out = np.empty((frames, *target_phase.shape), dtype=states_arr.dtype)
    rng = np.random.default_rng(0)
    for i in range(frames):
        noise = rng.uniform(
            -np.pi / max(len(states_arr), 1),
            np.pi / max(len(states_arr), 1),
            size=target_phase.shape,
        )
        dithered = np.mod(target_phase + noise, 2.0 * np.pi)
        out[i] = nearest_state(dithered, cell, freq)
    return out
