# Composite multi-stage systems

Cascade two or more antenna designs through a single PO pipeline.

```mermaid
graph LR
  Feed[Feed] --> L1[Layer 1<br/>Phase plate]
  L1 -->|angular spectrum| L2[Layer 2<br/>Phase plate]
  L2 -->|angular spectrum| OUT[Aperture]
  OUT --> FF[Far-field FFT]
```

## Bandwidth: achromatic doublet

![](../img/composite_doublet_bandwidth.png)

Two phase-correcting plates designed at the band edges (28 / 32 GHz) maintain
high directivity across the 28–32 GHz design band where a single mid-band
plate is more dispersive.

## Pre-built composites

| Class | Use case |
|---|---|
| `AchromaticDoublet` | Wide-band reflector / lens systems |
| `BifocalLens` | Shared-aperture comm + radar (24 / 77 GHz) |
| `FoldedReflectarray` | Cassegrain folded path, half the depth |

```python
import fresnelants as fa

doublet = fa.AchromaticDoublet(
    f_low=28e9, f_high=32e9, aperture_radius_m=0.05, levels=8, feed_distance=0.10
)
solver = fa.CascadePOSolver()
result = solver.solve(doublet, freq=30e9)
```

The `CompositeAntenna` class is the underlying primitive — wrap a list of
`Layer(design, z_offset)` tuples and the cascade-PO solver chains
forward-propagation through each layer via the angular-spectrum method.
