# Bibliography

This is the consolidated reading list for the FresnelAnts library.
Citations are organized **by topic**, not by author, so a reader who
wants to understand reflectarrays can read straight through one section
without chasing alphabetical cross-references.

Each entry has a one- or two-line annotation explaining why it matters
and which FresnelAnts module reflects its content. Entries marked with
`†` are *primary sources* (the historically defining paper for a topic);
entries marked with `‡` are *modern engineering references* (the book or
review article a working engineer should keep on the desk).

## Foundational antenna texts

These are the canonical antenna textbooks that every other section
assumes the reader has access to. They are also the load-bearing
references for the dipole, array-factor, reflector, and aperture
formulae used throughout the library.

- **‡ Balanis, *Antenna Theory: Analysis and Design*, 4th ed. (Wiley,
  2016).** Chapter 4 (linear wire), chapter 6 (array factor), chapter 12
  (aperture and reflector antennas), chapter 14 (microstrip). The
  default reference for sign conventions, far-field formulae, and the
  array-factor decomposition. The library's e^{jωt} time convention,
  +j Fourier kernel, and aperture-integral form match Balanis.
- **‡ Stutzman & Thiele, *Antenna Theory and Design*, 3rd ed. (Wiley,
  2012).** Stronger on the practical side: feed-pattern modelling,
  edge-taper rules of thumb, link-budget worked examples. Used as a
  cross-check on Balanis when the engineering nuance differs.
- **‡ Collin, *Antennas and Radiowave Propagation* (McGraw-Hill, 1985).**
  Older but precise; the canonical reference for aperture-distribution
  synthesis and the Bucci–Migliore degree-of-freedom argument that
  underlies the `synth/` package.
- **‡ Kraus & Marhefka, *Antennas: For All Applications*, 3rd ed.
  (McGraw-Hill, 2002).** The intuitive companion. Useful for visualizing
  how feed patterns and aperture illuminations interact.

## Fresnel zone plates

The lens lineage from [era 04](../background/history/04-lens-and-zone-plates.md).
Implemented in `src/fresnelants/designs/zone_plate.py`,
`phase_correcting.py`, `offset.py`, `curvilinear.py`,
`core/geometry.py`.

- **† Soret, "Sur les phénomènes de diffraction produits par les réseaux
  circulaires", *Archives des Sciences Physiques et Naturelles* 52,
  1875.** The original Soret zone plate. Optical, not radio.
- **† Wood, "Phase-reversal zone-plates, and diffraction-telescopes",
  *Philosophical Magazine* 45, 1898.** The phase-reversal variant that
  doubles aperture efficiency. Implemented as `WoodZonePlate`.
- **‡ Hristov, *Fresnel Zones in Wireless Links, Zone Plate Lenses and
  Antennas* (Artech House, 2000).** The canonical engineering treatment
  of microwave Fresnel zone plates. Single most-important reference for
  this library.
- **Wiltse, "The Fresnel zone-plate lens", *Proc. SPIE* 544, 1985.** The
  paper that argued for zone plates as serious microwave engineering.
- **Black & Wiltse, "Millimeter-wave characteristics of phase-correcting
  Fresnel zone plates", *IEEE Trans. MTT* 35, 1987.** The performance
  baseline for `PhaseCorrectingPlate` at quarter-wavelength step
  resolution.
- **Petosa, *Dielectric Resonator Antenna Handbook* (Artech House,
  2007).** Adjacent literature: dielectric lens design rules used by
  `CurvilinearFresnel`.

## Reflectors and lens-style apertures

Era 03 + the continuous-lens limit of era 04. Implemented in the feed
abstractions in `src/fresnelants/core/wavefront.py` and the
illumination defaults in `src/fresnelants/designs/base.py`.

- **† Silver (ed.), *Microwave Antenna Theory and Design* (MIT Rad Lab
  vol. 12, 1949).** The wartime synthesis. Still the clearest derivation
  of the geometric-optics reflector treatment.
- **‡ Love (ed.), *Reflector Antennas* (IEEE Press, 1978).** Collected
  reprints, including the canonical offset-paraboloid analysis.
- **Hannan, "Microwave antennas derived from the Cassegrain telescope",
  *IRE Trans. AP* 9, 1961.** The Cassegrain-geometry primary reference.
- **‡ Rusch & Potter, *Analysis of Reflector Antennas* (Academic Press,
  1970).** The mathematical bridge from geometric optics to physical
  optics — the regime FresnelAnts operates in.

## Phased arrays

The array lineage from [era 02](../background/history/02-resonant-arrays.md)
and [era 05](../background/history/05-phased-arrays.md). Implemented in
`src/fresnelants/designs/macro_array.py`,
`src/fresnelants/designs/reflectarray.py`, and the synthesis backends.

- **† Yagi, "Beam transmission of ultra short waves", *Proc. IRE* 16,
  1928.** The Yagi-Uda paper. Foundational for the array-factor
  decomposition.
- **‡ Mailloux, *Phased Array Antenna Handbook*, 3rd ed. (Artech House,
  2017).** The canonical phased-array reference. Chapters 2 and 3 are
  the formal source for the array-factor mathematics that the
  `MacroFresnelArray` implements.
- **‡ Hansen, *Phased Array Antennas*, 2nd ed. (Wiley, 2009).**
  Complementary engineering reference; deeper on mutual coupling and
  scan-blindness phenomena. The `coupling_q` heuristic in
  `MacroFresnelArray` is a first-order trend model in that direction.
- **Skolnik, *Introduction to Radar Systems*, 3rd ed. (McGraw-Hill,
  2001).** Chapter 9 — the radar-engineer's view of the phased-array
  transition. Useful for system-level context.

## Reflectarrays

Era 06 reflectarray sub-lineage. Implemented in
`src/fresnelants/designs/reflectarray.py`,
`src/fresnelants/designs/reconfigurable.py`,
`src/fresnelants/cells/`.

- **† Berry, Malech, & Kennedy, "The reflectarray antenna", *IEEE Trans.
  AP* 11, 1963.** The founding paper. Short-circuited waveguide stubs;
  the first proposal of "phased-array steering with no phase shifters".
- **‡ Huang & Encinar, *Reflectarray Antennas* (Wiley/IEEE, 2008).** The
  canonical microstrip-reflectarray engineering text. The unit-cell
  patch-size lookup in `Reflectarray.cell_phase_lookup(...)` is
  parameterised after Huang's reference designs.
- **Pozar, Targonski, & Syrigos, "Design of millimeter-wave microstrip
  reflectarrays", *IEEE Trans. AP* 45, 1997.** The mid-1990s state of
  the art that established the modern fabrication baseline.
- **Munson, "Conformal microstrip antennas and microstrip phased
  arrays", *IEEE Trans. AP* 22, 1974.** Origin of the microstrip patch.
- **Encinar, "Design of two-layer printed reflectarrays using patches of
  variable size", *IEEE Trans. AP* 49, 2001.** Multi-layer reflectarray
  bandwidth-extension techniques. Adjacent to but not directly
  implemented by `CompositeAntenna`.

## Reconfigurable intelligent surfaces

The post-2010 RIS literature. Implemented in
`src/fresnelants/designs/reconfigurable.py`,
`src/fresnelants/cells/varactor.py`,
`src/fresnelants/cells/pin_diode.py`,
`src/fresnelants/cells/liquid_crystal.py`,
`src/fresnelants/bias/`.

- **† Cui, Qi, Wan, Zhao, & Cheng, "Coding metamaterials, digital
  metamaterials and programmable metamaterials", *Light: Sci. & Appl.*
  3, 2014.** The "coded RIS" formulation that names the modern
  research program.
- **‡ Di Renzo *et al.*, "Smart radio environments empowered by
  reconfigurable intelligent surfaces", *IEEE J. Sel. Areas Commun.*
  38, 2020.** The reference review article — system-level RIS
  motivation, channel modelling, deployment considerations.
- **Yang, Yang, Xu, & Long, "A programmable metasurface with dynamic
  polarization, scattering and focusing control", *Sci. Rep.* 6, 2016.**
  Multifunctional RIS proof-of-concept that motivated the
  `DualPolSharedAperture` line.
- **Skyworks Solutions, *SMV1232 Series: Hyperabrupt Junction Tuning
  Varactors* (data sheet, 2018).** The C-V curve modelled by
  `cells/varactor.py`. The `MeasuredCell` class lets you substitute
  your own VNA-measured response.
- **MACOM, *MA4AGBLP912 PIN Diode* (data sheet).** ON-state and
  OFF-state insertion loss modelled by `cells/pin_diode.py`.
- **Merck Performance Materials, *Liquid Crystal Mixtures for Microwave
  Applications* (technical bulletin, 2019).** The GT3 and E7 dielectric
  data underlying `cells/liquid_crystal.py`.

## Metasurfaces and Pancharatnam–Berry phase

The metasurface lineage from era 06. Implemented in
`src/fresnelants/designs/metasurface.py`,
`src/fresnelants/cells/metasurface.py`,
`src/fresnelants/analysis/dualpol_metrics.py`.

- **† Berry, "The adiabatic phase and Pancharatnam's phase for polarized
  light", *J. Mod. Opt.* 34, 1987.** The geometric-phase argument that
  underpins PB metasurfaces.
- **‡ Yu & Capasso, "Flat optics with designer metasurfaces", *Nature
  Materials* 13, 2014.** The reference review for metasurface design;
  the article that brought "flat optics" into engineering vocabulary.
- **Chen, Taylor, & Yu, "A review of metasurfaces: physics and
  applications", *Rep. Prog. Phys.* 79, 2016.** Companion review; deeper
  on the polarization conversion that `MetasurfaceLens` implements.
- **Yu, Genevet, *et al.*, "Light propagation with phase
  discontinuities: generalized laws of reflection and refraction",
  *Science* 334, 2011.** The Snell's-law generalization that justifies
  the phase-discontinuity unit-cell model.

## Time-modulated arrays

The time-modulated lineage. Implemented in
`src/fresnelants/designs/time_modulated.py`,
`src/fresnelants/analysis/harmonics.py`.

- **† Kummer, Villeneuve, Fong, & Terrio, "Ultra-low sidelobes from
  time-modulated arrays", *IEEE Trans. AP* 11, 1963.** The original
  time-modulated paper, fifty years before the harmonic-beamforming
  revival.
- **‡ Tennant & Chambers, "A two-element time-modulated array with
  direction-finding properties", *IEEE Antennas & Wireless Propag. Lett.*
  6, 2007.** The DOA-via-harmonic-sideband technique modelled by
  `HarmonicPOSolver`.
- **Yang, Gan, & Yang, "A new approach for synthesizing time-modulated
  array antennas", *IEEE Trans. AP* 51, 2003.** Modulation-schedule
  synthesis baseline.
- **Poli, Rocca, *et al.*, "Time-modulated array antennas — basic
  principles and recent applications", *IEEE Trans. AP* 60, 2012.**
  Comprehensive modern review.

## Conformal and curved-substrate antennas

Era 06 conformal sub-lineage. Implemented in
`src/fresnelants/designs/conformal.py`,
`src/fresnelants/core/conformal.py`,
`src/fresnelants/analysis/conformal_farfield.py`.

- **‡ Josefsson & Persson, *Conformal Array Antenna Theory and Design*
  (Wiley/IEEE, 2006).** The canonical reference. The triangulated-mesh
  abstraction and the direct-PO integral over the conformal aperture
  follow Josefsson's chapter 4 directly.
- **Knott, "Antenna analysis using a new 3-D physical optics method",
  *IEEE Trans. AP* 40, 1992.** The PO-on-a-mesh paper that motivates
  `ConformalPOSolver`.
- **Persson & Josefsson, "Calculating the mutual coupling between
  apertures on a convex cylinder using a hybrid UTD-MoM method", *IEEE
  Trans. AP* 47, 1999.** Adjacent: cylindrical-conformal coupling.
  FresnelAnts does not implement UTD-MoM but the reference is useful
  for cross-checking.

## Phase synthesis and inverse design

The synthesis backends. Implemented in `src/fresnelants/synth/` (scipy,
CVXPY, JAX backends).

- **† Woodward & Lawson, "The theoretical precision with which an
  arbitrary radiation pattern may be obtained from a source of finite
  size", *J. IEE* 95, 1948.** The original synthesis paper. Bucci–
  Migliore (below) generalize it.
- **‡ Bucci & Migliore, "Effective number of independent samples in
  array antennas", *IEEE Trans. AP* 56, 2008.** The
  degrees-of-freedom argument that bounds what synthesis can achieve.
- **‡ Boyd & Vandenberghe, *Convex Optimization* (Cambridge UP, 2004).**
  General reference; the CVXPY backend in `synth/cvxpy_backend.py` is
  the direct application of chapter 8 (geometric programming) to
  phase-only aperture synthesis.
- **Bertsekas, *Nonlinear Programming*, 3rd ed. (Athena Scientific,
  2016).** Reference for the projected-gradient methods used by the
  `synth/scipy_backend.py` backend.
- **Bradbury *et al.*, "JAX: composable transformations of Python+NumPy
  programs", 2018.** The JAX library underlying `synth/jax_backend.py`'s
  gradient-based synthesis path.

## Solvers, physical optics, and angular spectrum

Library internals. Implemented in `src/fresnelants/solvers/`,
`src/fresnelants/analysis/farfield.py`,
`src/fresnelants/analysis/cascade.py`.

- **‡ Goodman, *Introduction to Fourier Optics*, 4th ed. (W. H. Freeman,
  2017).** The reference for the angular-spectrum propagation that
  `CascadePOSolver` and `analysis/cascade.py` implement. Sign
  conventions match Goodman's choice (which matches Balanis for
  antenna engineers).
- **‡ Born & Wolf, *Principles of Optics*, 7th ed. (Cambridge UP,
  1999).** Companion reference; Section 8.3 is the canonical
  derivation of the Fresnel-Kirchhoff diffraction integral that the
  PO solver computes via FFT.
- **Booker & Clemmow, "The concept of an angular spectrum of plane
  waves, and its relation to that of polar diagram and aperture
  distribution", *Proc. IEE* 97, 1950.** The original angular-spectrum
  paper.

## Manufacturing and export

Library export paths. Implemented in `src/fresnelants/export/`.

- **‡ Coombs, *Printed Circuits Handbook*, 7th ed. (McGraw-Hill, 2016).**
  PCB fabrication reference; the Gerber RS-274X exporter assumes
  Coombs-class tolerances.
- **3MF Consortium, *3MF Core Specification*, current edition.** The
  successor to STL for 3D-printable export. The `[cad]` extra (cadquery)
  emits 3MF as well as STEP.
- **CadQuery documentation, current edition.** The library used by the
  optional `[cad]` extra for STEP export of curvilinear and 3D-printable
  designs.

## Cross-reference: which paper for which design family

| Design family | Primary reference | Engineering reference |
|---|---|---|
| `SoretZonePlate` | Soret 1875 † | Hristov 2000 ‡ |
| `WoodZonePlate` | Wood 1898 † | Hristov 2000 ‡ |
| `OffsetZonePlate` | Hristov 2000 §3.4 | Wiltse 1985 |
| `PhaseCorrectingPlate` | Black & Wiltse 1987 | Hristov 2000 ‡ |
| `Reflectarray` | Berry/Malech/Kennedy 1963 † | Huang & Encinar 2008 ‡ |
| `CurvilinearFresnel` | Petosa 2007 | Goodman 2017 ‡ |
| `CompositeAntenna` / `AchromaticDoublet` | Encinar 2001 | Goodman 2017 (cascade) |
| `ReconfigurableArray` / `CodedRIS` | Cui *et al.* 2014 † | Di Renzo 2020 ‡ |
| `MetasurfaceLens` | Yu *et al.* 2011 | Yu & Capasso 2014 ‡ |
| `Cylindrical/SphericalFresnelLens` | Knott 1992 | Josefsson & Persson 2006 ‡ |
| `TimeModulatedArray` | Kummer 1963 † | Poli & Rocca 2012 ‡ |
| `Fractal*` | Saavedra *et al.* "Fresnel zone plate with multiple foci" *Opt. Lett.* 28, 2003 | Hristov 2000 ‡ + Saavedra |
| `MacroFresnelArray` | Mailloux 2017 ‡ (array factor) | Hansen 2009 ‡ (mutual coupling) |

For the historical narrative that situates these references in time,
return to the [Background & history](../background/history.md) section.
For the formal symbol table that the references' formulae translate into
this codebase, see [Notation & symbols](notation.md).
