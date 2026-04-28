"""Direct PO summation over a triangulated conformal aperture.

Includes an optional Numba-JIT path (gated on the [fast] extra) that runs
~10–50× faster than pure numpy on multi-thousand-facet meshes. The pure-
numpy path is always available; results are identical to within numerical
precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.conformal import ConformalAperture
from ..units import k0

try:  # pragma: no cover - import-time check
    from numba import njit, prange  # type: ignore[import-not-found]

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]


def _jit_kernel_factory():
    """Construct the JIT-compiled inner-loop kernel.

    Lazily built so the import cost is paid only when the JIT path is first
    used. Returns ``None`` if Numba is unavailable.
    """
    if not _NUMBA_AVAILABLE:
        return None
    if not hasattr(_jit_kernel_factory, "_kernel"):

        @njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
        def kernel(
            U_flat: np.ndarray,
            V_flat: np.ndarray,
            W_flat: np.ndarray,
            pts: np.ndarray,
            nor: np.ndarray,
            ar: np.ndarray,
            Et: np.ndarray,
            k: float,
        ) -> np.ndarray:
            n_dir = U_flat.shape[0]
            n_facet = pts.shape[0]
            out = np.zeros(n_dir, dtype=np.complex128)
            for d in prange(n_dir):  # type: ignore[misc]
                u = U_flat[d]
                v = V_flat[d]
                w = W_flat[d]
                acc = 0.0 + 0.0j
                for i in range(n_facet):
                    cos_obl = u * nor[i, 0] + v * nor[i, 1] + w * nor[i, 2]
                    if cos_obl <= 0.0:
                        continue
                    phase = k * (u * pts[i, 0] + v * pts[i, 1] + w * pts[i, 2])
                    acc += Et[i] * ar[i] * cos_obl * (np.cos(phase) + 1j * np.sin(phase))
                out[d] = acc
            return out

        _jit_kernel_factory._kernel = kernel  # type: ignore[attr-defined]
    return _jit_kernel_factory._kernel  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class ConformalFarField:
    """Far-field on a (u, v) grid produced by direct PO summation."""

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    U: NDArray[np.float64]
    V: NDArray[np.float64]
    E: NDArray[np.complex128]
    freq: float
    aperture_power: float

    @property
    def visible_mask(self) -> NDArray[np.bool_]:
        return self.U**2 + self.V**2 <= 1.0

    def intensity(self) -> NDArray[np.float64]:
        out = np.abs(self.E) ** 2
        out[~self.visible_mask] = 0.0
        return out

    def directivity(self) -> NDArray[np.float64]:
        intens = self.intensity()
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
        return float(10.0 * np.log10(max(np.max(d), 1e-30)))


def far_field_from_conformal(
    aperture: ConformalAperture,
    *,
    n_samples: int = 128,
    chunk: int = 4096,
    use_jit: bool | None = None,
) -> ConformalFarField:
    """Compute the far-field on an n × n (u, v) grid via direct integration.

    For each direction (u, v, w = √(1 − u² − v²)), the far-field is

        E(u, v) = Σᵢ Et[i] · cos θᵢ · exp(+j k r̂·r_i) · A_i

    where cos θᵢ is the local obliquity of the i-th facet relative to (u, v).

    Chunked over `n_samples × n_samples` directions in batches of `chunk`
    facets to keep memory bounded.

    Parameters
    ----------
    use_jit
        ``None`` → auto (use JIT if Numba is installed).
        ``True``  → require JIT, error if Numba missing.
        ``False`` → force the pure-numpy path.
    """
    u = np.linspace(-1.0, 1.0, n_samples)
    v = np.linspace(-1.0, 1.0, n_samples)
    U, V = np.meshgrid(u, v, indexing="xy")
    W = np.sqrt(np.clip(1.0 - U**2 - V**2, 0.0, 1.0))
    visible = U**2 + V**2 <= 1.0
    k = k0(aperture.freq)

    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()

    if use_jit is None:
        use_jit = _NUMBA_AVAILABLE
    if use_jit and not _NUMBA_AVAILABLE:
        raise RuntimeError(
            "use_jit=True requires Numba; install via `pip install fresnelants[fast]`."
        )

    if use_jit:
        kernel = _jit_kernel_factory()
        E_far = kernel(
            U_flat,
            V_flat,
            W_flat,
            np.ascontiguousarray(aperture.points),
            np.ascontiguousarray(aperture.normals),
            np.ascontiguousarray(aperture.areas),
            np.ascontiguousarray(aperture.Et),
            float(k),
        )
    else:
        n_facet = aperture.points.shape[0]
        E_far = np.zeros(U_flat.size, dtype=np.complex128)
        facets = aperture.points
        normals = aperture.normals
        areas = aperture.areas
        Et = aperture.Et
        for start in range(0, n_facet, chunk):
            stop = min(start + chunk, n_facet)
            pts = facets[start:stop]
            nor = normals[start:stop]
            ar = areas[start:stop]
            Et_chunk = Et[start:stop]
            phase = (
                U_flat[:, None] * pts[:, 0][None, :]
                + V_flat[:, None] * pts[:, 1][None, :]
                + W_flat[:, None] * pts[:, 2][None, :]
            ) * k
            cos_obl = (
                U_flat[:, None] * nor[:, 0][None, :]
                + V_flat[:, None] * nor[:, 1][None, :]
                + W_flat[:, None] * nor[:, 2][None, :]
            )
            cos_obl = np.maximum(cos_obl, 0.0)
            contrib = (Et_chunk * ar)[None, :] * cos_obl * np.exp(1j * phase)
            E_far += contrib.sum(axis=1)

    E_far[~visible.flatten()] = 0.0
    E_far = E_far.reshape(U.shape)
    return ConformalFarField(
        u=u,
        v=v,
        U=U,
        V=V,
        E=E_far,
        freq=aperture.freq,
        aperture_power=aperture.power(),
    )
