# 1875 – 1980 — Lenses & zone plates

## Period

This era is older than radio. The Fresnel zone plate as a focusing element
was published by Augustin Soret in 1875 and refined by Robert Wood in 1898 —
both for *visible-light* applications, decades before Hertz had even
demonstrated radio waves in the laboratory. Optical zone plates were used in
spectroscopy and X-ray imaging long before anyone proposed adapting the idea
to radio frequencies.

The radio adaptation happened slowly across the middle twentieth century.
Microwave Fresnel zone-plate antennas appeared in serious form in the 1960s,
matured into the millimetre-wave regime in the 1970s and 1980s, and reached
their canonical engineering treatment with Hristov's *Fresnel Zones in
Wireless Links, Zone Plate Lenses and Antennas* (Artech House, 2000).

In parallel, *dielectric lens antennas* — the radio analogue of an optical
glass lens — were developed for shipboard radar and short-haul terrestrial
links from the 1940s onward. The Luneburg lens (Luneburg 1944, Rinehart 1948)
and the Rotman lens (Rotman & Turner 1963) extended the lens lineage in
directions that have no zone-plate equivalent.

## Driving applications

- **Optics first.** Soret built his zone plate to demonstrate Fresnel
  diffraction in classroom and laboratory settings; Wood built the
  phase-reversal version to recover the energy that the binary Soret plate
  was throwing away.
- **Radar and shipboard antennas (1940s–1960s).** Dielectric lenses
  appeared in WWII airborne and shipboard radar where a lightweight
  fixed-beam aperture was preferable to a steered reflector.
- **Millimetre-wave links (1970s–1980s).** As radio engineers pushed into
  Ka-band and W-band, the zone plate's combination of light weight, flat
  profile, and ease of fabrication (etched copper or machined dielectric)
  became attractive. Wiltse's 1985 SPIE paper "The Fresnel Zone-Plate
  Lens" is the clearest statement of the modern microwave case.
- **CubeSat and low-cost terminals (post-2000).** Once 3D printing and
  PCB fabrication made flat dielectric and metallic plates trivially
  reproducible, the zone plate found a renaissance as a high-gain antenna
  for cost- and mass-constrained platforms.

## Technological step

The era's defining innovation is the recognition that **focusing does not
require a curved reflector or a thick dielectric lens** — it can be
accomplished by an aperture whose *transmittance* varies in a way that
imposes the right phase profile on the incident wave.

The Soret zone plate accomplishes this with the brutally simple device of
*blocking* every alternate Fresnel zone. The remaining zones contribute in
phase at the focal point; the blocked zones would have contributed out of
phase. A binary aperture mask — opaque rings on a transparent substrate —
focuses light. Aperture efficiency is poor (about 10 %), because half the
zones are wasted, but the device works.

Wood's 1898 phase-reversal zone plate doubles the efficiency to about 40 %
by *flipping the phase* of the zones the Soret plate was blocking, instead
of throwing them away. At microwave frequencies this is implemented either
as a quarter-wave dielectric step or as a metallic phase-shifting cell. The
Wood plate is the antenna engineer's most-direct descendant of Soret's
optical device.

The phase-correcting plate (also called a stepped or quantized lens) is the
natural generalization: instead of two phase levels (Soret) or two-with-a-
flip (Wood), use $N$ levels covering $0$ to $2\pi$ in $2\pi/N$ steps. The
gain rises monotonically with $N$: a 4-level plate is good for >50 %
efficiency, an 8-level plate exceeds 70 %, and a 16-level plate
asymptotes to the continuous-phase ideal. This is exactly the trade-off
visualised in `docs/img/phase_correcting_levels.png`.

The dielectric lens, the curvilinear singlet, and the 3D-printed lens
are the *continuous-phase* limit of the same idea: instead of stepping
the phase in $2\pi/N$ increments, vary the local thickness of a
dielectric so that the phase profile is exactly the focusing parabola.

## What was lost / what was gained

- **Gained:** *flatness*. A zone plate is a sheet; a phase-correcting
  plate is a few-millimetre slab; even a curvilinear lens is much
  thinner than its parabolic-reflector equivalent. Aperture-to-volume
  ratio improves by orders of magnitude.
- **Gained:** *manufacturability*. Etched copper on a PCB substrate, or
  3D-printed dielectric, costs orders of magnitude less than a precision
  parabolic mirror at the same gain.
- **Gained:** *integration*. A flat aperture can be laminated into a
  radome, embedded in a vehicle bumper, or mounted on a wall.
- **Lost:** *bandwidth*. Zone radii are wavelength-specific. A Soret
  plate designed for 28 GHz works only across maybe a 10–15 % fractional
  bandwidth before the focusing breaks down. Continuous dielectric
  lenses are wider-band but still narrower than a parabolic reflector.
- **Lost:** *peak efficiency*. The best zone plate (Wood, ~40 %) is
  dramatically below the best reflector (~70 %). The continuous lens
  recovers most of that loss but at the cost of thickness and dielectric
  loss tangent.

## What survived

The lens lineage gave FresnelAnts its v0.1 core. Every flat single-
polarization design family — Soret, Wood, offset Fresnel, phase-correcting
plate, curvilinear singlet — descends from this era's recognition that an
*aperture mask* can replace a reflector.

The mathematical machinery that survived:

- **Fresnel zone radii.** $r_n = \sqrt{n \lambda F + (n \lambda / 2)^2}$
  is the geometric formula that determines the zone boundaries on a flat
  aperture for a feed at distance $F$. Implemented in
  `src/fresnelants/core/geometry.py::fresnel_zone_radii(...)`.
- **Phase profile of an equivalent lens.** The phase advance required to
  collapse a spherical wave from a feed at $z = -F$ to a plane wave at
  the aperture is $\phi(r) = -k(\sqrt{r^2 + F^2} - F)$. A Wood plate
  approximates this with two levels per zone; a phase-correcting plate
  approximates it with $N$ levels; a curvilinear lens implements it
  continuously by varying dielectric thickness.
- **The aperture-mask abstraction.** `AntennaDesign.transmittance(grid,
  freq, state)` returns *the complex transmission coefficient at every
  point on the aperture*. This is the most general statement of the
  zone-plate idea: focusing reduces to the right choice of $T(x, y)$.

## Where it shows up in FresnelAnts

- `src/fresnelants/designs/zone_plate.py` — `SoretZonePlate` (binary
  amplitude mask) and `WoodZonePlate` (phase-reversal mask). The Wood
  plate is the textbook
  ~40 %-efficient single-frequency aperture.
- `src/fresnelants/designs/offset.py` — `OffsetZonePlate`, the flat
  analogue of the offset paraboloid from era 03.
- `src/fresnelants/designs/phase_correcting.py` —
  `PhaseCorrectingPlate(focal_length, design_freq, aperture_radius_m,
  levels)`. The `levels` parameter trades fabrication depth resolution
  against aperture efficiency; see
  `docs/img/phase_correcting_levels.png` for the swept curve.
- `src/fresnelants/designs/curvilinear.py` — `CurvilinearFresnel`, the
  continuous-phase 3D singlet. Hyperbolic, axicon (Bessel-like ring),
  and freeform profiles supported. STL export via `[base]`, STEP via
  the `[cad]` extra.
- `src/fresnelants/core/geometry.py::fresnel_zone_radii(...)` — the
  geometric primitive every plate-style design calls.
- `src/fresnelants/designs/fractal.py` — fractal Cantor and Sierpinski
  zone plates extend the lens lineage with self-similar masks; the
  Cantor polyfocal axial signature ($F$, $F/3$, $F/5$ foci) shown in
  `docs/img/fractal_cantor_axial_intensity.png` is a direct
  generalization of the Soret/Wood construction. The implementation
  detail that "Wood ≡ Soret at base_unit = 1" is documented in the
  design's docstring and in the polyfocal regression test
  `tests/test_designs.py::test_fractal_cantor_polyfocal_signature`.

## Further reading

- Soret, "Sur les phénomènes de diffraction produits par les réseaux
  circulaires", *Archives des Sciences Physiques et Naturelles* 52,
  1875 — the original optical paper.
- Wood, "Phase-reversal zone-plates, and diffraction-telescopes",
  *Philosophical Magazine* 45, 1898 — the phase-reversal variant.
- Wiltse, "The Fresnel zone-plate lens", *Proc. SPIE* 544, 1985 — the
  microwave engineering case.
- Hristov, *Fresnel Zones in Wireless Links, Zone Plate Lenses and
  Antennas* (Artech House, 2000) — the canonical engineering treatment.

See also the
[Bibliography → Fresnel zone plates](../../reference/bibliography.md#fresnel-zone-plates)
section for the full annotated reading list, including Black & Wiltse's
work on dielectric perforated zone plates and the more recent metalens
literature that bridges this era to era 06.

Continue to [Era 05 — phased arrays](05-phased-arrays.md).
