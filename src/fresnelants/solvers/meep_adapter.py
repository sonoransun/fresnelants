"""Meep (FDTD) adapter — requires `pip install fresnelants[fullwave]` or conda Meep.

Wires a real 2D-cylindrical FDTD pipeline for axisymmetric Fresnel zone-plate
designs. Meep is conda-only on most platforms; the import is wrapped so the
adapter can be imported without Meep installed (`solve()` will raise then).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.aperture import ApertureField
from ..core.geometry import make_aperture_grid
from ..designs.base import AntennaDesign
from ..designs.zone_plate import SoretZonePlate, WoodZonePlate
from ..units import freq_to_wavelength
from .base import SolverResult

try:  # pragma: no cover - import-time check
    import meep as mp  # type: ignore[import-not-found]

    _MEEP_AVAILABLE = True
except ImportError:  # pragma: no cover
    mp = None  # type: ignore[assignment]
    _MEEP_AVAILABLE = False


@dataclass
class MeepAdapter:
    """2D-cylindrical FDTD wrapper for axisymmetric zone plates."""

    name: str = "MeepAdapter"
    resolution_per_wavelength: int = 20
    """FDTD cells per design wavelength."""
    pml_thickness_lambda: float = 1.0
    runtime_periods: float = 50.0
    eps_dielectric: float = 2.5
    """Default dielectric constant for transparent zones (substrate-like)."""

    def is_available(self) -> bool:
        return _MEEP_AVAILABLE

    def solve(
        self, design: AntennaDesign, freq: float, state: object | None = None
    ) -> SolverResult:
        if not _MEEP_AVAILABLE:
            raise RuntimeError(
                "Meep is not installed. Install via "
                "`conda install -c conda-forge pymeep` or "
                "`pip install fresnelants[fullwave]` (Linux only)."
            )
        if not isinstance(design, (SoretZonePlate, WoodZonePlate)):
            raise TypeError(
                f"MeepAdapter only supports axisymmetric zone plates so far; "
                f"got {type(design).__name__}. Subclass MeepAdapter for other "
                f"geometries."
            )
        return self._solve_zone_plate(design, freq)

    def _solve_zone_plate(
        self, design: SoretZonePlate | WoodZonePlate, freq: float
    ) -> SolverResult:
        """2D-cylindrical FDTD: r-z slice with axisymmetric BC."""
        lam = float(freq_to_wavelength(freq))
        # Meep length unit = 1 metre.
        cell_size_r = design.aperture_radius * 1.4
        cell_size_z = max(2.0 * design.focal_length, 6 * lam)
        pml = self.pml_thickness_lambda * lam
        resolution = self.resolution_per_wavelength / lam  # cells per metre

        cell = mp.Vector3(cell_size_r + pml, 0, cell_size_z + 2 * pml)
        boundary_layers = [mp.PML(pml)]

        # Build geometry: stack of annular slabs at z = 0 with thickness lam/8.
        slab_thickness = lam / 8.0
        radii = design.zone_radii
        prev = 0.0
        geometry = []
        for n, r in enumerate(radii, start=1):
            inner = prev
            outer = float(r)
            transparent = n % 2 == 1
            if isinstance(design, SoretZonePlate):
                # Soret: opaque (PEC) on even zones, transparent on odd.
                if not transparent:
                    geometry.append(
                        mp.Block(
                            size=mp.Vector3(outer - inner, mp.inf, slab_thickness),
                            center=mp.Vector3((inner + outer) / 2, 0, 0),
                            material=mp.metal,
                        )
                    )
            else:
                # Wood: alternate dielectric (high-index for π phase) on even zones.
                if not transparent:
                    geometry.append(
                        mp.Block(
                            size=mp.Vector3(outer - inner, mp.inf, slab_thickness),
                            center=mp.Vector3((inner + outer) / 2, 0, 0),
                            material=mp.Medium(epsilon=self.eps_dielectric),
                        )
                    )
            prev = outer

        # Plane-wave source on the −z face.
        source_z = -cell_size_z / 2 + pml * 1.2
        sources = [
            mp.Source(
                mp.GaussianSource(frequency=freq * 1e-8, fwidth=freq * 1e-8 * 0.2),
                component=mp.Er,
                center=mp.Vector3(cell_size_r / 2, 0, source_z),
                size=mp.Vector3(cell_size_r, 0, 0),
            )
        ]

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=boundary_layers,
            geometry=geometry,
            sources=sources,
            resolution=resolution,
            dimensions=2,
            m=0,  # axisymmetric mode 0
        )

        # DFT monitor at the focal plane.
        focal_z = design.focal_length
        dft = sim.add_dft_fields(
            [mp.Er, mp.Ez],
            freq * 1e-8,
            0,
            1,
            center=mp.Vector3(cell_size_r / 2, 0, focal_z),
            size=mp.Vector3(cell_size_r, 0, 0),
        )
        sim.run(until=self.runtime_periods / (freq * 1e-8))

        # Extract focal-plane field.
        Er = sim.get_dft_array(dft, mp.Er, 0)
        focal_mag = np.abs(Er)
        # Build a 2-D ApertureField with axisymmetric repetition.
        grid = make_aperture_grid(2 * design.aperture_radius * 1.2, 6.0, lam)
        Ez_field = np.zeros(grid.shape, dtype=np.complex128)
        # Map radial DFT samples onto the grid.
        r_samples = np.linspace(0, cell_size_r, focal_mag.size)
        Ez_field = np.interp(grid.R.flatten(), r_samples, focal_mag).reshape(grid.shape)
        ap = ApertureField(grid=grid, Ez=Ez_field.astype(np.complex128), freq=freq)

        from ..analysis.farfield import far_field_from_aperture

        ff = far_field_from_aperture(ap, pad_factor=2)
        return SolverResult(aperture=ap, far_field=ff, metadata={"solver": self.name, "freq": freq})
