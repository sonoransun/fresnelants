# 1963 – present — Reflectarrays, RIS, metasurfaces

## Period

From Berry, Malech, and Kennedy's 1963 reflectarray paper to today's
research-grade reconfigurable intelligent surfaces and time-modulated
metasurfaces. Frequencies span the full microwave and millimetre-wave
ranges, with active research pushing into the THz regime.

This era overlaps in time with [era 05 — phased arrays](05-phased-arrays.md)
but represents a *different* technological lineage: the reflectarray and
its descendants achieve phased-array-like steering *without* the per-element
amplifier chain, by replacing each active T/R module with a passive (or
reconfigurable) resonant cell.

## Driving applications

- **Cost-sensitive electronic steering.** A microstrip reflectarray
  achieves passive AESA-like phase control at orders-of-magnitude lower
  cost. Adopted from the 1990s onward in space, satellite, and remote-
  sensing applications where the AESA cost was prohibitive.
- **5G millimetre-wave coverage and RIS.** The 2018–present generation
  of "smart radio environments" research proposes
  reconfigurable-intelligent-surfaces (RIS) — large reflective panels of
  electronically-tunable cells — as a way to bend mmWave signals around
  obstructions in dense urban environments.
- **Dual-polarization shared-aperture radar and comms.** Pancharatnam–
  Berry metasurfaces enable a single aperture to handle co-polarized and
  cross-polarized channels simultaneously, shrinking the antenna
  footprint of dual-pol ground stations and aircraft radomes.
- **Harmonic beamforming and DOA estimation.** Time-modulated arrays
  dynamically switch element states to generate harmonic sidebands whose
  spatial filtering properties enable single-RF-chain direction-of-arrival
  estimation.
- **Conformal apertures.** Automotive radar at 77 GHz needs to fit
  inside a vehicle bumper that is not flat. Spherical and cylindrical
  conformal Fresnel lenses solve this by abandoning the flat-aperture
  assumption.

## Technological step

Several distinct innovations sit in this era. They are not a single
chronological progression — many of them developed in parallel.

### Microstrip reflectarrays (1963; commercial 1980s–present)

Berry, Malech, and Kennedy's original 1963 paper proposed a flat array of
short-circuited waveguide stubs as a replacement for a curved reflector.
The key idea: each cell adds a *fixed phase shift* tuned by its physical
geometry, so the aggregate aperture mimics a parabolic phase profile.

The microstrip-patch reflectarray (Munson 1974 patch antennas; Pozar
1990s reflectarray maturation) replaced the waveguide stub with a square
copper patch on a low-loss substrate. The patch's resonant frequency,
controlled by patch size, sets the phase response. A mid-2000s
state-of-the-art reflectarray approaches 50–60 % aperture efficiency at
fractional bandwidths of 5–15 %.

### Reconfigurable intelligent surfaces (RIS) — 2010s–present

A reflectarray cell becomes *reconfigurable* if its phase response can be
controlled electronically. Three technologies dominate:

- **Varactor diodes.** A reverse-biased semiconductor diode whose
  capacitance varies with bias voltage. Vendor: Skyworks SMV1232 is the
  textbook choice for sub-10 GHz. Continuous phase control over ~270°.
- **PIN diodes.** A two-state switch (high-Z OFF, low-Z ON) implementing
  a 1-bit cell. Cheap and fast (microseconds), but limited to discrete
  phase states. Vendor: MACOM MA4AGBLP912.
- **Liquid crystal.** A nematic-liquid-crystal layer whose refractive
  index varies with bias voltage. Slow (milliseconds), but offers
  continuous phase control with no discrete junctions. Used in mmWave
  beamforming research.

The `cells/` package in FresnelAnts encapsulates each of these as a
device model: bias-voltage → complex reflection coefficient as a
function of frequency. The `MeasuredCell` class lets users substitute
their own VNA-measured response.

### Pancharatnam–Berry metasurfaces — 2010s–present

A sub-wavelength array of *anisotropic* resonators rotated at distinct
angles imposes a *geometric* phase on the cross-polarized component of an
incident wave. The phase advance is twice the rotation angle, independent
of frequency over a broad range. This enables broadband *circularly-
polarized* lensing with cross-pol conversion built in. The
`MetasurfaceLens` and `DualPolSharedAperture` classes implement this with
a Jones-matrix unit-cell model.

### Time-modulated arrays — 1963 (Kummer); modern research 2010s–present

If the reflection coefficient of each cell is *time-varying* — switched
periodically between states — the aperture radiates not only at the
carrier but also at harmonic sidebands $f_c \pm n f_m$ for modulation
frequency $f_m$ and integer $n$. The harmonic sidebands carry information
about the spatial phase pattern and can be used for harmonic beamforming
or direction-of-arrival estimation with a single RF receive chain. The
`TimeModulatedArray` and `HarmonicPOSolver` classes implement this with
explicit harmonic decomposition.

### Conformal Fresnel lenses — 2000s–present

For applications where the substrate is not flat — automotive bumpers,
aircraft radomes, satellite-mounted spherical caps — the flat-aperture
FFT shortcut breaks. The `CylindricalFresnelLens` and
`SphericalFresnelLens` classes use a triangulated mesh aperture and a
direct PO integral over the surface (no flat FFT), implemented in
`ConformalPOSolver`.

### MacroFresnelArray — collapsing the lineages — 2024–present

The most recent addition (v0.5) treats each *complete Fresnel antenna* —
a zone plate, a curvilinear lens, a reflectarray, even a fractal — as
the radiating element of a phased array. The far-field is the textbook
array-factor × element-pattern product from [era 02](02-resonant-arrays.md),
with the per-element pattern computed by the appropriate Fresnel solver.
This combines the *aperture efficiency* of a Fresnel lens with the
*scanning agility* of a phased array; it is also the cheapest path to
very-large-aperture radio-astronomy arrays.

## What was lost / what was gained

- **Gained:** *electronic steering at lens-level cost*. A reflectarray
  costs 10–100× less than an AESA at the same gain and steering range.
- **Gained:** *bandwidth via composite stages*. The
  `AchromaticDoublet` family stacks two phase-correcting plates with
  complementary dispersion to flatten the gain across a 24–36 GHz band
  (`docs/img/composite_doublet_bandwidth.png`).
- **Gained:** *dual-polarization in a single aperture*. PB metasurfaces
  share aperture between co-pol and cross-pol channels.
- **Gained:** *conformal integration*. Curved-substrate lenses fit
  inside non-flat host structures with no flat-array compromise.
- **Lost:** *full hemispheric scan*. RIS cells have limited phase swing
  (varactor: ~270°; PIN: 180°), and large scan angles cost gain
  (`docs/img/ris_1bit_scanloss.png`).
- **Lost:** *unit-cell predictability*. Real-world cell behaviour
  diverges from the textbook unit-cell lookup. The `MeasuredCell` class
  exists precisely to substitute measured response for the analytical
  approximation when accuracy demands it
  (`docs/img/measured_roundtrip.png`).

## What survives — and what is still open

The era is still open. The reflectarray is now mature engineering;
microstrip RIS is at the technology-demonstrator stage in commercial
deployments (selected Asian and European 5G operators are running RIS
trials as of the early 2020s); time-modulated arrays and PB metasurfaces
are still primarily research instruments. The MacroFresnelArray is so
new it has no commercial deployment at all — it sits at the intersection
of radio-astronomy aperture-array roadmaps and 6G mmWave research.

What this means for the library: era-06 designs are explicitly marked as
research-grade in the `extensions/` documentation and in the
[Applications](../../applications.md) page. Their data-sheet performance
should be treated as an indication of capability, not as a manufacturing
guarantee.

## Where it shows up in FresnelAnts

- `src/fresnelants/designs/reflectarray.py` — `Reflectarray`, with the
  patch-size unit-cell lookup. The reference deployment is a 32×32 array
  at 28 GHz with ±30° steering.
- `src/fresnelants/designs/reconfigurable.py` — `ReconfigurableArray`
  and `CodedRIS`, the per-cell tunable variants. State vectors carry
  bias voltages or PIN states.
- `src/fresnelants/designs/metasurface.py` — `MetasurfaceLens` and
  `DualPolSharedAperture`. Jones-matrix unit-cell model with axial-ratio
  and cross-pol metrics.
- `src/fresnelants/designs/conformal.py` — `CylindricalFresnelLens`,
  `SphericalFresnelLens`. Direct PO over a triangulated mesh aperture.
- `src/fresnelants/designs/composite.py` — `CompositeAntenna`,
  `AchromaticDoublet`, `BifocalLens`, `FoldedReflectarray`. Cascade
  multiple plates through angular-spectrum propagation.
- `src/fresnelants/designs/time_modulated.py` — `TimeModulatedArray`,
  paired with `HarmonicPOSolver` for harmonic beamforming.
- `src/fresnelants/designs/fractal.py` — `FractalSoretZonePlate`,
  `FractalWoodZonePlate`, `SierpinskiCarpetZonePlate`,
  `SierpinskiReflectarray`, `SphericalFractalFresnelLens`,
  `ConicalFractalFresnelLens`. The polyfocal Cantor signature is the
  era-06 generalization of the era-04 zone plate.
- `src/fresnelants/designs/macro_array.py` — `MacroFresnelArray`. The
  v0.5 collapse of the lens and array lineages onto each other.
- `src/fresnelants/cells/` — vendor-grounded device models:
  Skyworks SMV1232 varactor (`cells/varactor.py`), MACOM MA4AGBLP912
  PIN diode (`cells/pin_diode.py`), Merck GT3/E7 nematic LC
  (`cells/liquid_crystal.py`), measured-cell interpolation
  (`cells/measured.py`), Pancharatnam–Berry / anisotropic
  (`cells/metasurface.py`).
- `src/fresnelants/bias/` — bias-network synthesis and back-side Gerber
  emission for RIS arrays.
- `src/fresnelants/analysis/cascade.py` — angular-spectrum propagation
  used by `CompositeAntenna`.
- `src/fresnelants/analysis/conformal_farfield.py` — direct PO integral
  over the conformal triangulated mesh.
- `src/fresnelants/analysis/harmonics.py` — harmonic decomposition for
  `TimeModulatedArray`.
- `src/fresnelants/analysis/dualpol_metrics.py` — Jones-matrix cross-pol
  and axial-ratio metrics for `MetasurfaceLens`.

## Further reading

- Berry, Malech, & Kennedy, "The reflectarray antenna", *IEEE Trans. AP*
  11, 1963 — the founding paper of the lineage.
- Huang & Encinar, *Reflectarray Antennas* (Wiley/IEEE, 2008) — modern
  microstrip-reflectarray engineering.
- Yu & Capasso, "Flat optics with designer metasurfaces", *Nature
  Materials* 13, 2014 — metasurface and PB-phase exposition.
- Cui *et al.*, "Coding metamaterials, digital metamaterials and
  programmable metamaterials", *Light: Sci. & Appl.* 3, 2014 — coded RIS
  origins.
- Di Renzo *et al.*, "Smart radio environments empowered by
  reconfigurable intelligent surfaces", *IEEE J. Sel. Areas Commun.* 38,
  2020 — RIS deployment and channel modelling.
- Kummer *et al.*, "Ultra-low sidelobes from time-modulated arrays",
  *IEEE Trans. AP* 11, 1963 — the original time-modulated paper, fifty
  years before the harmonic-beamforming revival.
- Josefsson & Persson, *Conformal Array Antenna Theory and Design*
  (Wiley/IEEE, 2006) — conformal-aperture canonical text.

For the full annotated reading list per design family — including the
synthesis backends and the measured-cell literature — see the
[Bibliography](../../reference/bibliography.md).

Return to the [history overview](../history.md), or jump to the
[applications guide](../../applications.md) to map a deployment scenario
to a recommended era-06 family.
