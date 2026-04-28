"""Tunable-cell abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

CellState = float | int | bool | np.ndarray
"""A single cell's state.

Continuous (varactor voltage), discrete bit (PIN diode), or per-cell array.
"""


class TunableCell(ABC):
    """Per-cell phase / loss as a function of state and frequency."""

    name: str = "TunableCell"

    @abstractmethod
    def phase(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        """Reflection phase [rad] at *state* and *freq*.

        *state* is broadcast-compatible — pass a scalar for a single cell or
        a 2-D array for a whole RIS.
        """

    @abstractmethod
    def loss(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        """Insertion / reflection loss [linear amplitude factor in (0, 1]]."""

    @abstractmethod
    def state_set(self, freq: float) -> tuple[ArrayLike, str]:
        """Return (canonical state values, kind ∈ {'continuous','discrete'})."""

    def reflection_coefficient(self, state: ArrayLike, freq: float) -> NDArray[np.complex128]:
        """Complex reflection coefficient = loss · exp(j · phase)."""
        return self.loss(state, freq).astype(np.complex128) * np.exp(1j * self.phase(state, freq))
