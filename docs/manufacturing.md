# Manufacturing

| Family | STL | STEP (`[cad]` extra) | Gerber / Excellon |
|---|:-:|:-:|:-:|
| Soret zone plate | ✓ | – | – |
| Wood zone plate | ✓ | – | – |
| Phase-correcting plate | ✓ | ✓ | – |
| Reflectarray | – | – | ✓ |
| Curvilinear | ✓ | ✓ | – |

## STL

`fresnelants.export.stl.export_stl(design, path)` writes a binary STL.
Vertices are merged via `trimesh` so the output is watertight. For lenses,
the substrate thickness is auto-set to ~5 % of the aperture radius (≥ 0.5 mm).

```python
import fresnelants as fa
from fresnelants.export.stl import export_stl

lens = fa.CurvilinearFresnel(
    focal_length=0.10, design_freq=30e9, aperture_radius_m=0.05
)
export_stl(lens, "lens.stl", radial_samples=200, angular_samples=240)
```

## STEP

Requires `pip install 'fresnelants[cad]'` (cadquery). Produces a
revolved solid that imports cleanly into FreeCAD / Fusion / Rhino /
SolidWorks.

```python
from fresnelants.export.step import export_step

export_step(lens, "lens.step")
```

## Gerber

The reflectarray exporter writes RS-274X copper, soldermask, and a
placeholder Excellon drill file:

```python
from fresnelants.export.gerber import write_reflectarray_gerber

paths = write_reflectarray_gerber(ra, "./gerber/", base_name="ra_28ghz")
```

Most fabricators (JLCPCB, OSH Park, PCBWay) accept this directly. The
patch size ↔ phase mapping is a deliberately conservative default; for
production designs replace it with a curve fit to your unit cell from
HFSS / CST / openEMS.
