# Offset Fresnel zone plate

An offset-fed Fresnel plate has its feed displaced from the geometric axis,
producing a broadside main beam without the feed blockage that plagues
on-axis zone plates. The zone boundaries are loci of constant path-length
difference between the (now off-axis) feed and the aperture point.

| Layout | Far-field | E/H cuts |
|---|---|---|
| ![](../img/offset_layout.png) | ![](../img/offset_farfield.png) | ![](../img/offset_cuts.png) |

## API

```python
import math
import fresnelants as fa

design = fa.OffsetZonePlate(
    focal_length=0.50,
    design_freq=12e9,
    aperture_radius_m=0.30,
    tilt_angle=math.radians(25),
)
```

## CLI

```bash
fresnelants design offset --freq 12e9 -F 0.5 --radius 0.30 --tilt-deg 25 --out offset.json
fresnelants analyze offset.json
```
