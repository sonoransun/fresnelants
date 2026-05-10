# Applications: which design when

This page maps deployment scenarios to FresnelAnts design families. Read it
the way an antenna engineer reads a parts catalog: scan the matrix for
candidates, then read the narrative for the scenario that matches your
problem.

For the historical reasoning behind why these families *exist* in the first
place, see the [Background & history](background/history.md) section. For the
formal references and citations, see the
[Bibliography](reference/bibliography.md).

## Capability matrix

The matrix below summarizes the eleven canonical design families. *Typical
band* and *aperture size* are rules of thumb — every family extends beyond
the listed range, but performance and manufacturability degrade outside it.
*Maturity* is library-internal: **stable** means covered by the v0.1 release
test suite; **research** means the implementation is correct but the design
trade-offs are still active research questions.

| Family | Typical band | Aperture size | Peak gain rule of thumb | Bandwidth | Steerable? | Polarization | Manufacturing | Export | Maturity |
|---|---|---|---|---|---|---|---|---|---|
| **Soret zone plate** (`SoretZonePlate`) | 5 – 100 GHz | 5 λ – 100 λ | $\eta \approx 10\%$; D ≈ aperture-area-limited | ~10–15 % fractional | No | Single-pol (transmittance) | Etched copper / printed binary mask | STL · Gerber | Stable |
| **Wood zone plate** (`WoodZonePlate`) | 5 – 100 GHz | 5 λ – 100 λ | $\eta \approx 40\%$ | ~10–15 % fractional | No | Single-pol | Etched copper / dielectric step | STL · Gerber | Stable |
| **Offset Fresnel** (`OffsetZonePlate`) | 5 – 60 GHz | 10 λ – 100 λ | $\eta \approx 35\%$ | ~10–15 % fractional | No (fixed feed offset) | Single-pol | Etched copper | STL · Gerber | Stable |
| **Phase-correcting plate** (`PhaseCorrectingPlate`) | 10 – 100 GHz | 5 λ – 50 λ | $\eta \approx 50\%$ at 4 levels; ≥ 70 % at 8 levels | ~15–25 % | No | Single-pol | Machined dielectric / 3D-printed | STL · STEP | Stable |
| **Reflectarray** (`Reflectarray`) | 5 – 100 GHz | 16 λ – 64 λ | $\eta \approx 45–60\%$ | ~5–15 % at design freq | Yes (via cell phase pattern) | Single-pol; dual-pol with rotated patches | PCB / microstrip | Gerber | Stable |
| **Curvilinear / 3D singlet** (`CurvilinearFresnel`) | 10 – 100 GHz | 5 λ – 30 λ | $\eta \approx 65–75\%$ | ~25–40 % | No | Single-pol | 3D-printed dielectric / machined | STL · STEP | Stable |
| **Composite cascade** (`CompositeAntenna`, `AchromaticDoublet`) | 10 – 100 GHz | 10 λ – 50 λ | $\eta \approx 50–65\%$ | **flat across 24 – 36 GHz with doublet** | No | Single-pol | Stacked plates with controlled spacing | STL · STEP per stage | Research |
| **Reconfigurable RIS** (`ReconfigurableArray`, `CodedRIS`) | 5 – 60 GHz | 16 λ – 64 λ | $\eta \approx 30–50\%$ | Limited by cell tunability | Yes (electronically) | Single-pol typical; dual with anisotropic cells | PCB + bias network | Gerber + bias-Gerber | Research |
| **Pancharatnam–Berry metasurface** (`MetasurfaceLens`, `DualPolSharedAperture`) | 20 – 110 GHz | 10 λ – 30 λ | $\eta \approx 50–70\%$ on cross-pol channel | Broadband (geometric phase is achromatic in $f$) | No (passive); yes via PIN-cell variant | Dual (LCP / RCP, co/cross) | Sub-wavelength PCB | Gerber | Research |
| **Conformal lens** (`CylindricalFresnelLens`, `SphericalFresnelLens`) | 24 – 110 GHz | 5 λ – 20 λ | $\eta \approx 55–70\%$ | Like underlying singlet | No | Single-pol | Wrapped or molded dielectric | STL · STEP (mesh) | Research |
| **Time-modulated array** (`TimeModulatedArray`) | 5 – 60 GHz | 8 λ – 32 λ | varies per harmonic | Carrier ± $n f_m$ harmonics | Yes (per-harmonic) | Single-pol | PCB + fast switches | Gerber | Research |
| **Fractal Fresnel** (`Fractal*`) | 10 – 100 GHz | 16 λ – 100 λ | Polyfocal (energy split across F, F/3, F/5) | ~10–15 % per focus | No (passive); polyfocal sweep | Single-pol | Etched fractal mask | Gerber · STL (3D) | Research |
| **MacroFresnelArray** (`MacroFresnelArray`) | 1 – 60 GHz | $N \times$ element size, scales to N=128+ | $\eta_{\text{element}} \cdot$ array factor | Per element, plus squint at scan | **Yes (codebook)** | Inherits element | Per-element fab + array structure | Mixed (per element) + codebook JSON | Research |

A few patterns to read off the matrix:

- The **stable / v0.1 row** is the lens lineage from
  [era 04](background/history/04-lens-and-zone-plates.md). Manufacturing
  paths are well understood (etched PCB or printed dielectric); the only
  exotic dependency is the `[cad]` extra for STEP.
- **Steerability is a sharp axis.** Only `Reflectarray`,
  `ReconfigurableArray`/`CodedRIS`, `TimeModulatedArray`, and
  `MacroFresnelArray` steer electronically. Everything else is fixed-beam.
- **Bandwidth is the other sharp axis.** Single-stage plates are
  fundamentally narrowband; the doublet, the curvilinear singlet, and
  the PB metasurface (geometric phase) are the wide-band options.
- **Polarization control is era-06's specialty.** Single-stage plates
  are single-pol; metasurface and rotated-patch reflectarrays open
  dual-pol shared-aperture designs.

## Application narratives

### Fixed Ka-band SATCOM terminal

**Recommendation:** [`PhaseCorrectingPlate`](designs/phase_correcting.md) at
8 or 16 phase levels, or a [`CurvilinearFresnel`](designs/curvilinear.md)
hyperbolic singlet for higher efficiency at the cost of weight.

The use case — receive-only, fixed pointing at a geosynchronous slot —
asks for high gain, modest steering (mechanical fine-tune is fine), and
very low cost. A 16-level phase-correcting plate at 30 GHz with a 20 cm
aperture exceeds 70 % efficiency and prints on a desktop SLA machine.
The curvilinear singlet pushes that to ~75 % at the cost of a thicker,
heavier piece.

```python
import fresnelants as fa

lens = fa.PhaseCorrectingPlate(
    focal_length=0.10, design_freq=30e9,
    aperture_radius_m=0.10, levels=16,
)
result = fa.PhysicalOpticsSolver().solve(lens, freq=30e9)
print(f"D = {result.far_field.peak_directivity_dbi():.1f} dBi")
```

See `docs/img/phase_correcting_levels.png` for the levels-vs-gain curve
that drives the `levels=16` choice.

### mmWave 5G base-station front-end

**Recommendation:** [`Reflectarray`](designs/reflectarray.md) at 28 GHz or
39 GHz with a 32×32 cell aperture, plus a `MacroFresnelArray` if the
deployment requires fast multi-beam codebook steering.

5G mmWave base stations need ±30° azimuth scan, sub-millisecond beam
switching, and dual-polarization. A passive reflectarray supplies the
electronic beam pattern at orders-of-magnitude lower cost than an AESA;
a `MacroFresnelArray` of 16 reflectarray elements brings codebook-based
multi-beam reception. The squint-budget conversation lives in
`extensions/composite.md` — for users who need wider instantaneous
bandwidth, the achromatic-doublet path is worth investigating.

```python
ra = fa.Reflectarray(
    nx=32, ny=32, cell_size=0.0054,  # ~λ/2 at 28 GHz
    focal_length=0.20, design_freq=28e9,
    steering_angle_deg=(0.0, 30.0),  # broadside → +30° elevation
)
result = fa.PhysicalOpticsSolver().solve(ra, freq=28e9)
```

The steering sweep `docs/img/reflectarray_steering.png` shows the gain-
versus-scan-angle envelope.

### 77 GHz automotive radar lens

**Recommendation:** [`CurvilinearFresnel`](designs/curvilinear.md)
(hyperbolic profile) for fixed forward-looking radar, or
[`CylindricalFresnelLens`](extensions/conformal.md) when the lens must
follow the bumper curvature.

A long-range automotive radar at 76–81 GHz needs ~30 dBi gain in a
package small enough to integrate behind a plastic bumper. A 5 cm
hyperbolic singlet hits the target with one moulded part. If the bumper
geometry refuses a flat lens, the cylindrical conformal variant projects
the same Fresnel zoning onto a mild-curvature mesh and the
`ConformalPOSolver` handles the direct PO integral.

```python
lens = fa.CurvilinearFresnel(
    profile="hyperbolic",
    focal_length=0.04, design_freq=77e9,
    aperture_radius_m=0.025,
    dielectric_eps_r=2.1,
)
```

For the conformal case, see `docs/img/conformal_cylindrical_77ghz.png`
for the reference far-field at 77 GHz.

### Radio-astronomy aperture array

**Recommendation:** [`MacroFresnelArray`](designs/macro_array.md) of
identical low-cost lens elements (zone plate or phase-correcting plate),
combined with the receive codebook for multi-beam survey work.

Modern radio astronomy is moving away from very-large reflectors toward
*aperture arrays* of inexpensive elements. The MacroFresnelArray is the
direct realization of this idea in the FresnelAnts taxonomy: each
element is a complete Fresnel antenna; the array factor combines them
with conjugate-matched receive weights. Scales to N=128+ without
FFT-cost growth because the per-element pattern is computed once and
the array factor is a closed-form $O(N)$ sum.

```python
element = fa.WoodZonePlate(focal_length=0.20, design_freq=10e9, num_zones=10)
array = fa.MacroFresnelArray.from_lattice(
    "hex", element=element, n_rings=4, spacing=0.30,
)
weights = array.weights_for_beam(theta_deg=0.0, phi_deg=0.0)
result = array.solve(fa.PhysicalOpticsSolver(), freq=10e9, weights=weights)
```

The 128-element hex pattern in `docs/img/macro_array_128x_hex_pattern.png`
illustrates the scale.

### Low-cost IoT / sub-6 GHz PCB Fresnel

**Recommendation:** [`WoodZonePlate`](designs/zone_plate.md) printed on
a single PCB layer, or [`SoretZonePlate`](designs/zone_plate.md) when
fabrication cannot afford the phase-reversal layer.

For sub-6 GHz IoT links — backhaul, ISM-band receivers, low-cost mesh
infrastructure — the trade-off is dominated by manufacturing cost. A
Soret plate is a single-layer copper-on-FR4 part. The Wood plate adds
either a quarter-wave dielectric step (one extra mill) or a metallic
phase-shift cell (one extra etching mask) to recover ~4× the gain.
Either fits in a hobbyist budget.

### Reconfigurable RIS for indoor coverage

**Recommendation:** [`CodedRIS`](extensions/ris.md) with a varactor or
PIN-diode cell, sized to the 28 GHz or 39 GHz coverage band and the
specific room geometry.

The "smart radio environment" use case — bouncing a mmWave signal
around a structural obstruction — is the highest-profile application of
era-06 reconfigurable surfaces. The cell choice depends on the channel
update rate: PIN diodes (microseconds, 1-bit) for fast switching; varactors
(continuous, ~270° swing) for accurate beam shaping; liquid crystal
(milliseconds, continuous) for slow tracking with no semiconductor
junction to fail.

The bias-network synthesis (back-side Gerber) lives in `bias/` and
emits the routing automatically; this is the single piece of
manufacturing detail RIS designs ship that other families do not need.

### Dual-polarization shared-aperture Tx/Rx

**Recommendation:**
[`DualPolSharedAperture`](extensions/metasurface.md) using a Pancharatnam–
Berry metasurface, or a rotated-patch `Reflectarray` if the application
tolerates a narrower bandwidth.

Sharing aperture between a co-pol transmit channel and a cross-pol
receive channel requires polarization isolation in the unit cell, which
a PB metasurface provides natively (the geometric phase is broadband and
acts only on cross-pol). A rotated-patch reflectarray offers the same
trick at lower fabrication complexity but narrower bandwidth.

### Conformal automotive bumper / spherical radome

**Recommendation:**
[`CylindricalFresnelLens`](extensions/conformal.md) for mild bumper
curvature; [`SphericalFresnelLens`](extensions/conformal.md) for radome
or aircraft-skin integration.

The conformal solver path is the only way to handle non-flat substrates
without falsely flattening the geometry. The `[viz3d]` extra is
recommended here for the 3D mesh visualization.

### Wideband / achromatic links

**Recommendation:** [`AchromaticDoublet`](extensions/composite.md) — two
phase-correcting plates with complementary dispersion stacked at
optimized spacing.

A single phase-correcting plate has ~25 % fractional bandwidth before
gain rolls off. Stacking two plates with opposite dispersion flattens
the gain curve across a 24–36 GHz band — exactly the figure shown in
`docs/img/composite_doublet_bandwidth.png`. The penalty is a thicker
package (two stages) and the cost of optimizing the inter-stage spacing.

### Harmonic beamforming and DOA estimation

**Recommendation:** [`TimeModulatedArray`](extensions/timemod.md)
combined with a `HarmonicPOSolver` sweep across the harmonic indices of
interest.

This is a single-RF-chain alternative to digital beamforming: instead of
$N$ ADCs and $N$ digital weight vectors, modulate the cell states
periodically and read the harmonic sidebands at $f_c \pm n f_m$ off a
single receiver. The directionality of each harmonic is the spatial
filter. `docs/img/timemod_harmonic_beams.png` shows the harmonic
patterns for a 28 GHz reference design.

### Polyfocal / multi-target operation

**Recommendation:** [`FractalSoretZonePlate`](designs/fractal.md) or
`FractalWoodZonePlate` if the application can use a fixed multi-focus
distribution (the canonical Cantor signature is foci at $z = F$, $F/3$,
$F/5$).

The polyfocal axial signature in
`docs/img/fractal_cantor_axial_intensity.png` shows the energy
distribution across the multiple foci. This is research-grade
territory; for a single-focus deployment, the conventional Wood plate
is more efficient.

## How this matrix was constructed

The performance numbers above come from the regression suite — peak-
directivity assertions in `tests/test_designs.py`, level-vs-gain sweeps
re-run on every commit by `docs/generate_figures.py`, and the
benchmarks in `tests/test_macro_array.py`. They are honest, but they
are also *idealized*: real hardware will exhibit additional losses
(dielectric, surface-roughness, feed-spillover, finite cell-edge
effects). Treat the matrix as a design-space map, not as a procurement
data sheet.

For end-to-end runnable examples per family, see
[`examples/`](https://github.com/example/fresnelants/tree/main/examples)
in the repository. For the historical reasoning behind these
recommendations, see [Background & history](background/history.md).
