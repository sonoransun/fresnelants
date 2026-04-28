"""Time-modulated arrays — periodic 1-bit cell switching for harmonic beam steering.

Each cell switches between an "on" and "off" state with a periodic schedule;
viewed at the n-th harmonic of the modulation frequency, the array's effective
amplitude per cell is the n-th Fourier coefficient of the schedule. Different
harmonics steer their beams to different directions, enabling time-multiplexed
multi-beam apertures and harmonic direction-of-arrival (DOA) estimation.

Reference: Yang & Tennant, "Direction of arrival estimation using time-
modulated array antennas", Electronics Letters 50.13 (2014).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .reconfigurable import ReconfigurableArray


@dataclass
class TimeModulatedArray(ReconfigurableArray):
    """1-bit reconfigurable array with a periodic per-cell switching schedule.

    Each cell is described by ``(t_on, duty)`` — the start time of its
    "on" interval (as a fraction of the modulation period [0, 1]) and the
    fraction of the period spent on. The harmonic Fourier coefficients of the
    resulting square wave determine the per-harmonic far-field.
    """

    schedule: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 2)))
    """``(ny, nx, 2)`` array of ``(t_on, duty)`` per cell. Both in [0, 1]."""

    f_modulation: float = 10e3
    """Modulation frequency [Hz]; harmonics appear at f_carrier ± n·f_modulation."""

    name: str = "TimeModulatedArray"

    def harmonic_coefficient(self, n: int) -> NDArray[np.complex128]:
        """Per-cell n-th Fourier coefficient.

        For a square wave with start ``t_on`` and duty ``D``,

            c_0 = D
            c_n = (1 / (j·n·π)) · (e^{−j2π·n·(t_on+D)} − e^{−j2π·n·t_on})  for n ≠ 0

        Returns an ``(ny, nx)`` complex array.
        """
        if self.schedule.size == 0:
            return np.full((self.ny, self.nx), 1.0 if n == 0 else 0.0, dtype=np.complex128)
        sched = np.asarray(self.schedule)
        if sched.shape != (self.ny, self.nx, 2):
            raise ValueError(f"schedule shape {sched.shape} != expected {(self.ny, self.nx, 2)}")
        t_on = sched[..., 0]
        duty = sched[..., 1]
        if n == 0:
            return duty.astype(np.complex128)
        a = np.exp(-1j * 2.0 * np.pi * n * t_on)
        b = np.exp(-1j * 2.0 * np.pi * n * (t_on + duty))
        return ((b - a) / (-1j * 2.0 * np.pi * n)).astype(np.complex128)

    def linear_progressive_schedule(
        self, time_per_cell: float = 0.5, duty: float = 0.5
    ) -> NDArray[np.float64]:
        """Convenience: build a linear-time-delay schedule across the array.

        Cell ``(i, j)`` turns on at time ``(j / nx) · time_per_cell``. This
        produces an n-th harmonic beam steered by an angle proportional to n.
        """
        sched = np.zeros((self.ny, self.nx, 2))
        for j in range(self.nx):
            sched[:, j, 0] = (j / self.nx) * time_per_cell
        sched[..., 1] = duty
        self.schedule = sched
        return sched
