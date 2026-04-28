"""Measured-cell wrapper: phase / loss interpolated from a sweep table.

`MeasuredCell` is a `TunableCell` whose response comes from a tabulated
``(state, freq) → S₁₁`` mapping — typically the output of a full-wave (HFSS,
CST, openEMS) parameter sweep, or measured S-parameters from a VNA.

Two ingestion paths:

* `from_arrays(states, freqs, s11)` — pass in the data directly.
* `from_touchstone_sweep({state: path})` — read each Touchstone (.s2p / .s4p)
  via scikit-rf and interpolate. Requires the optional `[measurement]` extra.

The cell exposes the same `phase / loss / state_set` API as other cells, so it
drops into `ReconfigurableArray(cell=measured_cell)` without further changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]

from .base import TunableCell


@dataclass
class MeasuredCell(TunableCell):
    """Interpolated cell from tabulated (state, freq) → S₁₁ data."""

    states: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    """1-D array of state values (e.g. bias voltages)."""
    freqs: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    """1-D array of frequencies [Hz]."""
    s11: NDArray[np.complex128] = field(default_factory=lambda: np.zeros((0, 0), np.complex128))
    """2-D array of complex reflection coefficients, shape ``(len(states), len(freqs))``."""
    name: str = "MeasuredCell"

    def __post_init__(self) -> None:
        if self.s11.shape != (len(self.states), len(self.freqs)):
            raise ValueError(
                f"s11 shape {self.s11.shape} does not match "
                f"(len(states)={len(self.states)}, len(freqs)={len(self.freqs)})"
            )
        if len(self.states) > 0 and len(self.freqs) > 0:
            self._interp_real = RegularGridInterpolator(
                (self.states, self.freqs), self.s11.real, bounds_error=False, fill_value=None
            )
            self._interp_imag = RegularGridInterpolator(
                (self.states, self.freqs), self.s11.imag, bounds_error=False, fill_value=None
            )
        else:
            self._interp_real = None
            self._interp_imag = None

    def _gamma(self, state: ArrayLike, freq: float) -> NDArray[np.complex128]:
        if self._interp_real is None:
            raise ValueError("MeasuredCell has no data — load via from_arrays/touchstone first.")
        s = np.asarray(state, dtype=np.float64)
        pts = np.stack([s.flatten(), np.full(s.size, float(freq))], axis=1)
        re = self._interp_real(pts).reshape(s.shape)
        im = self._interp_imag(pts).reshape(s.shape)
        return (re + 1j * im).astype(np.complex128)

    def phase(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        return np.angle(self._gamma(state, freq)).astype(np.float64)

    def loss(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        return np.abs(self._gamma(state, freq)).astype(np.float64)

    def state_set(self, freq: float) -> tuple[NDArray[np.float64], str]:
        return self.states, "continuous"

    @classmethod
    def from_arrays(
        cls,
        states: ArrayLike,
        freqs: ArrayLike,
        s11: ArrayLike,
        name: str = "MeasuredCell",
    ) -> MeasuredCell:
        """Build a MeasuredCell from raw arrays (e.g. simulator output)."""
        return cls(
            states=np.asarray(states, dtype=np.float64),
            freqs=np.asarray(freqs, dtype=np.float64),
            s11=np.asarray(s11, dtype=np.complex128),
            name=name,
        )

    @classmethod
    def from_touchstone_sweep(
        cls,
        sweep: dict[float, str | Path],
        port: int = 0,
        name: str = "MeasuredCell",
    ) -> MeasuredCell:
        """Load multiple Touchstone files (one per state) into a MeasuredCell.

        Requires `scikit-rf` (`pip install fresnelants[measurement]`).
        """
        try:
            import skrf  # type: ignore[import-untyped]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "scikit-rf is required; install via `pip install fresnelants[measurement]`."
            ) from e
        states = sorted(sweep.keys())
        freqs_ref: NDArray[np.float64] | None = None
        s11_rows: list[NDArray[np.complex128]] = []
        for state in states:
            net = skrf.Network(str(sweep[state]))
            f = np.asarray(net.f, dtype=np.float64)
            if freqs_ref is None:
                freqs_ref = f
            elif not np.allclose(f, freqs_ref):
                raise ValueError(f"Frequency grid mismatch in Touchstone for state={state}")
            s11_rows.append(net.s[:, port, port].astype(np.complex128))
        assert freqs_ref is not None
        return cls(
            states=np.asarray(states, dtype=np.float64),
            freqs=freqs_ref,
            s11=np.stack(s11_rows, axis=0),
            name=name,
        )

    def save(self, path: str | Path) -> None:
        """Persist as JSON for embedded controllers / round-trip tests."""
        data = {
            "name": self.name,
            "states": self.states.tolist(),
            "freqs": self.freqs.tolist(),
            "s11_real": self.s11.real.tolist(),
            "s11_imag": self.s11.imag.tolist(),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> MeasuredCell:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        s11 = np.asarray(d["s11_real"]) + 1j * np.asarray(d["s11_imag"])
        return cls.from_arrays(d["states"], d["freqs"], s11, name=d.get("name", "MeasuredCell"))
