"""PIN-diode 1-bit reflective cell.

A PIN diode in a reflectarray cell switches the cell between two reflection
phases ~ 180° apart, with low insertion loss in either state. We model the
two states by directly imposing a 0° / 180° phase pair, with loss reflecting
the diode's published forward and reverse insertion-loss specs.

Defaults reflect publicly-published values:

* **MACOM MA4FCP305** — 0.05 dB low-loss switching PIN, X-band → mmW.
* **Skyworks SMP1340** — silicon PIN, microwave-band staple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import TunableCell


@dataclass(frozen=True, slots=True)
class PINDiodeCell(TunableCell):
    insertion_loss_db_on: float
    """Forward-bias insertion loss [dB]."""
    insertion_loss_db_off: float
    """Reverse-bias insertion loss [dB]."""
    R_on: float = 2.5
    R_off: float = 4_000.0
    C_off: float = 40e-15
    L_package: float = 0.4e-9
    switching_time_ns: float = 5.0
    name: str = "PINDiodeCell"

    def phase(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        s = np.asarray(state).astype(bool)
        # State True → 0; state False → +π. (1-bit phase difference.)
        return np.where(s, 0.0, np.pi).astype(np.float64)

    def loss(self, state: ArrayLike, freq: float) -> NDArray[np.float64]:
        s = np.asarray(state).astype(bool)
        loss_on = 10.0 ** (-self.insertion_loss_db_on / 20.0)
        loss_off = 10.0 ** (-self.insertion_loss_db_off / 20.0)
        return np.where(s, loss_on, loss_off).astype(np.float64)

    def state_set(self, freq: float) -> tuple[NDArray[np.bool_], str]:
        return np.array([False, True]), "discrete"


def MACOM_MA4FCP305() -> PINDiodeCell:
    """Factory — MACOM MA4FCP305 (0.05 dB low-loss PIN)."""
    return PINDiodeCell(
        insertion_loss_db_on=0.05,
        insertion_loss_db_off=0.10,
        R_on=2.5,
        R_off=4_000.0,
        C_off=40e-15,
        L_package=0.45e-9,
        switching_time_ns=2.0,
        name="MACOM MA4FCP305",
    )


def Skyworks_SMP1340() -> PINDiodeCell:
    """Factory — Skyworks SMP1340 (silicon PIN, microwave staple)."""
    return PINDiodeCell(
        insertion_loss_db_on=0.30,
        insertion_loss_db_off=0.50,
        R_on=1.2,
        R_off=10_000.0,
        C_off=200e-15,
        L_package=0.6e-9,
        switching_time_ns=20.0,
        name="Skyworks SMP1340",
    )
