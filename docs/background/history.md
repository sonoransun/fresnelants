# A short history of radio-frequency reception

Fresnel-style antennas did not appear in a vacuum. Every design family in this
library — the Soret zone plate, the reflectarray, the conformal lens, the
reconfigurable intelligent surface — sits on top of a 130-year arc of
incremental improvements in how engineers capture energy from a passing
electromagnetic wave and steer it toward a detector.

This section walks that arc in six eras. Each era page answers the same five
questions:

- **What changed.** The technological step that defined the era.
- **Why it changed.** The driving application — radar, satellite downlink,
  cellular, automotive sensing — that made the previous answer inadequate.
- **What was traded.** Every step *added* a capability and *gave up* something
  else. The history is most useful when those trade-offs are explicit.
- **What survived.** Concepts that the next era kept and built on.
- **Where it shows up in FresnelAnts.** The specific module(s) and class(es)
  in this codebase that descend from that era's contribution.

The intent is not a textbook. The intent is to make the design space legible
so that, when you reach for `WoodZonePlate` instead of `Reflectarray`, or
`MetasurfaceLens` instead of `PhaseCorrectingPlate`, you understand the
hundred-year argument you are inheriting.

## Timeline at a glance

| Era | Period | Defining innovation | Dominant band | FresnelAnts descendants |
|---|---|---|---|---|
| [01 — Spark-gap](history/01-spark-gap-era.md) | 1888 – 1910 | Hertzian dipoles; resonant LC reception; coherer & crystal detectors | LF / MF (kHz to low MHz) | The dipole feed model in `core/wavefront.py` |
| [02 — Resonant arrays](history/02-resonant-arrays.md) | 1910 – 1940 | Tuned wire arrays; Yagi-Uda; Beverage; broadcast-era directional antennas | MF / HF / VHF | Array factor decomposition reused by `MacroFresnelArray` |
| [03 — Microwave reflectors](history/03-microwave-reflectors.md) | 1940 – 1970 | WWII radar; parabolic dish; horn feed; Cassegrain; satellite ground stations | UHF → X-band | The `SphericalWave` feed and the canonical reflector geometry that the offset Fresnel and reflectarray reproduce |
| [04 — Lenses & zone plates](history/04-lens-and-zone-plates.md) | 1875 – 1980 | Optical Fresnel zone plates (Soret, Wood) adapted to microwave; dielectric and Luneburg lenses | Optical → mmW | `zone_plate.py`, `phase_correcting.py`, `offset.py`, `curvilinear.py` — the v0.1 core |
| [05 — Phased arrays](history/05-phased-arrays.md) | 1960 – 2000 | Per-element phase control; AESA; digital beamforming; on-the-fly steering | L → Ka-band | `Reflectarray`, the array-factor maths in `MacroFresnelArray`, and the synthesis backends in `synth/` |
| [06 — Reflectarrays, RIS, metasurfaces](history/06-reflectarrays-ris-metasurfaces.md) | 1963 – present | Microstrip reflectarrays; reconfigurable intelligent surfaces; Pancharatnam–Berry metasurfaces; time-modulated apertures | C → THz | `reflectarray.py`, `reconfigurable.py`, `metasurface.py`, `time_modulated.py`, `composite.py`, `conformal.py`, `fractal.py`, `macro_array.py` |

## How to read the eras

The eras are roughly chronological, but not strictly. Two of them overlap on
purpose:

- **Era 04 (lenses & zone plates) starts in 1875** — before radio existed —
  because the optical zone plate was Augustin Soret's invention for visible
  light (1875) and Robert Wood's phase-reversal variant (1898) preceded any
  radio adaptation by nearly half a century. Treating the lens lineage as a
  child of the radio era would erase that. The microwave adaptation arrived
  in the mid-twentieth century, but the *idea* is older than Marconi's first
  transatlantic transmission.
- **Era 05 and Era 06 overlap by decades.** Berry, Malech, and Kennedy
  published the first reflectarray in 1963, while phased arrays were still
  in their analog-AESA infancy. The microstrip reflectarray boom in the
  1980s and the RIS / metasurface explosion in the 2010s are continuations
  of the *reflectarray* line, not of the *phased-array* line, even though
  the two superficially resemble each other.

## Why this matters for picking a design

A few conclusions fall out of the arc that are not obvious from any single
design's data sheet:

- **Fixed-beam Fresnel antennas (zone plates, phase-correcting plates,
  curvilinear lenses) are direct descendants of the optical lens lineage**,
  not of the radio-array lineage. Their efficiency, bandwidth, and aberration
  behaviour are best understood by analogy to optics — which is why
  `core/geometry.py` constructs the aperture in zones rather than in cells.
- **Reflectarrays and RIS are descendants of the array lineage**, not of
  the lens lineage, despite looking like flat plates. Their phase profile is
  *imposed* by per-cell tuning rather than emerging from a bulk geometry.
  This is why `Reflectarray` in this library exposes a `cell_phases(...)`
  attribute and not a zone radius schedule.
- **Conformal and 3D-curvilinear designs combine both lineages**: they
  recover the geometric phase advance of a curved dielectric (lens lineage)
  while sampling on a discrete mesh (array lineage). That is why
  `ConformalPOSolver` does not take an FFT shortcut.
- **MacroFresnelArray collapses the two lineages onto each other** by
  treating each lens-style element as a phased-array element. The textbook
  array-factor × element-pattern decomposition that makes this work was
  established in [Era 02](history/02-resonant-arrays.md).

If you want a single page that turns the historical reasoning into a
recommendation for *your* deployment scenario, jump to
[Applications](../applications.md). If you want the citations behind every
era, see the [Bibliography](../reference/bibliography.md).

## What this library does *not* do

It is worth being explicit about what is out of scope, because the history
makes it tempting to imagine otherwise:

- **Pre-resonant antennas and crystal detectors.** Fresnel-style apertures
  are inherently phase-coherent and high-frequency; the LF / MF era is here
  for context, not as a target band. The lowest sensible frequency for any
  design in this library is ~1 GHz (1 m wavelength) where the aperture
  begins to fit on a tabletop.
- **Active receivers.** The library models the *passive aperture* — what
  fraction of an incoming wave is captured and where it focuses. The LNA,
  mixer, IF chain, and digital back-end are downstream and are not the
  subject of this story.
- **Full-wave Maxwell solvers from scratch.** FresnelAnts builds on
  physical optics and the angular spectrum. NEC and Meep adapters exist
  for cases where a full-wave solver is required, but they are wrappers,
  not reimplementations.

With those scope notes in mind, start with
[Era 01 — the spark-gap era](history/01-spark-gap-era.md) and work forward,
or jump to whichever era describes the kind of antenna you actually plan to
build.
