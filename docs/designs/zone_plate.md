# Soret / Wood zone plate

The classical Fresnel zone plate. **Soret** plates use binary amplitude
modulation (alternating opaque / transparent zones); **Wood** plates phase-
reverse alternating zones (×−1) for ~6 dB more efficiency.

## Layouts

| Soret (amplitude) | Wood (phase-reversal) |
|---|---|
| ![Soret](../img/zone_plate_soret_layout.png) | ![Wood](../img/zone_plate_wood_layout.png) |

## Far-field

Wood plate, 12 zones, F = 1 m, 10 GHz:

![Wood far-field](../img/zone_plate_wood_farfield.png)
![Wood cuts](../img/zone_plate_wood_cuts.png)

## Theory

Zone radii are given by *rₙ = √(nλF + (nλ/2)²)* (see [theory](../theory.md)).
Soret efficiency saturates near 10 % (primary focus); Wood near 40 %.

## API

```python
import fresnelants as fa

soret = fa.SoretZonePlate(focal_length=1.0, design_freq=10e9, num_zones=12)
wood  = fa.WoodZonePlate(focal_length=1.0, design_freq=10e9, num_zones=12)
```

## CLI

```bash
fresnelants design zone-plate --freq 10e9 --focal-length 1.0 --zones 12 --kind wood --out wood.json
fresnelants analyze wood.json
```
