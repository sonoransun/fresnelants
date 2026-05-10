# 1960 – 2000 — Phased arrays

## Period

From the first operational electronically-scanned radar systems in the
early 1960s through the digital-beamforming revolution of the 1990s. The
frequency range covered the full microwave spectrum, L-band through
Ka-band, with the heaviest investment at S-, C-, and X-band where most
search and tracking radars operated.

## Driving applications

- **Air defence and missile tracking.** The MPQ-53 Patriot radar (S-band,
  passively-scanned), AN/SPY-1 Aegis (S-band, passively-scanned, four
  faces for hemispheric coverage), and AN/APG-77 (X-band, actively-
  scanned for the F-22) defined the phased-array procurement programs of
  the era. The military requirement was the same one a reflector could
  not satisfy: scan a search volume in milliseconds, then dwell on a
  target without mechanical motion.
- **Air-traffic control and weather radar.** Multifunction phased arrays
  began to displace mechanically-rotated reflectors at airports and
  weather-radar sites, though slowly — the cost trade-off favored
  reflectors at most installations until well into the 2000s.
- **Satellite uplink and earth-station diversity.** Phased arrays
  appeared at large earth stations for tracking LEO and MEO satellites
  where mechanical pointing could not keep up.

## Technological step

The era's defining innovation was **per-element phase control**. An array
of identical radiating elements — patches, dipoles, slots — each fed
through a controllable phase shifter could synthesize a beam in any
direction the array geometry supported. The mathematics had been
understood since [era 02](02-resonant-arrays.md), but only the maturation
of microwave ferrite and PIN-diode phase shifters in the 1950s and 1960s
made it economically realizable.

Three architectural waves defined the era:

1. **Passive electronically-scanned arrays (PESA).** A single transmit
   amplifier feeds a corporate or space-fed network of phase shifters,
   one per element. PESA dominated 1960s–1980s deployments and remains
   in service on legacy systems.
2. **Active electronically-scanned arrays (AESA).** Each element gets
   its own transmit/receive module — a small GaAs (later GaN) MMIC with
   integrated PA, LNA, phase shifter, and attenuator. The graceful
   degradation, lower-loss receive path, and ability to form multiple
   simultaneous beams transformed military radar in the 1990s.
3. **Digital beamforming (DBF).** Each element (or sub-array) digitizes
   its received signal, and beam formation happens in software. Any
   number of receive beams can be formed simultaneously, and adaptive
   nulling becomes a pure-DSP problem. DBF was the dominant research
   thread of the late 1990s; commercial systems (5G massive MIMO base
   stations, OneWeb terminals) became commonplace in the 2010s.

## What was lost / what was gained

- **Gained:** *electronic steering at radar speeds*. A modern AESA can
  point its beam in microseconds, an improvement of nine orders of
  magnitude over a hand-cranked dish. This single capability transformed
  search-while-track radar.
- **Gained:** *multiple simultaneous beams*. A digital array can receive
  $K$ beams in parallel by running $K$ weight vectors against the same
  raw element samples. The textbook receive codebook and adaptive
  nulling techniques became practical.
- **Gained:** *graceful failure*. An AESA whose $N$th element fails
  loses ~$1/N$ of its gain; a reflector whose feed fails is dead. This
  is the basis of the military procurement argument for AESA on
  high-value platforms.
- **Lost:** *cost*. A 1990s X-band AESA module cost thousands of
  dollars; a 4 m parabolic reflector cost a fraction of that per square
  metre of aperture. The phased array was a procurement decision, not a
  cost-optimisation.
- **Lost:** *bandwidth at large electrical sizes*. An array steered to
  off-broadside angles by phase shift (rather than by true time delay)
  exhibits beam squint as the frequency varies; wideband phased arrays
  require true-time-delay units, which add a further cost layer.

## What survived

Three things from the phased-array lineage survive in FresnelAnts:

1. **The reflectarray.** A flat array of passive resonant cells, each
   imposing a fixed phase shift on the reflected wave, is *exactly* a
   phased array with the phase shifters baked into the unit-cell
   geometry. Berry, Malech, and Kennedy's 1963 paper proposed this as a
   way to get phased-array steering without phase shifters; the modern
   microstrip variant (Huang & Encinar 2008) made it cheap. The
   `Reflectarray` class in `src/fresnelants/designs/reflectarray.py`
   inherits the phased-array array-factor mathematics directly.
2. **The receive-codebook abstraction.** A digital phased array forms
   beams by computing $\mathbf{w}^H \mathbf{x}$ for various weight
   vectors $\mathbf{w}$. The `MacroFresnelArray.beam_codebook(...)` API
   is the same idea, scaled up so each "element" is itself a complete
   Fresnel antenna.
3. **The phase-synthesis problem.** Once you have $N$ adjustable phases,
   the question becomes: given a target far-field pattern, what phases
   do you choose? The phased-array literature developed analytical
   answers (Chebyshev, Taylor, Villeneuve) and iterative answers
   (Woodward-Lawson, alternating projections, convex relaxations). The
   `synth/` package in this library — scipy, CVXPY, and JAX backends —
   inherits that lineage directly.

## Bandwidth, beam squint, and true-time delay

A subtlety that recurs in later eras: a phased array steered by *phase*
exhibits **beam squint** as the operating frequency changes. The beam
direction $\theta_0(f)$ moves with $f$ at off-broadside angles because
the phase $\phi = -k d \sin\theta_0$ depends on $k = 2\pi f / c$.

The fix is **true-time delay (TTD)**: implement the steering by physical
delay rather than by phase, and the steering becomes frequency-flat. TTD
units are bulky and expensive at large electrical sizes, so most
deployed phased arrays accept squint over their operating bandwidth.

In FresnelAnts, the reflectarray and reconfigurable-array families
inherit the squint behaviour by default — a `Reflectarray` whose cell
phases were optimised at 28 GHz will exhibit squint at 26 GHz and
30 GHz. The `CompositeAntenna` doublet path is one of the constructive
fixes: chain a phase-correcting plate with a squint-cancelling second
stage and recover broader instantaneous bandwidth, as illustrated in
`docs/img/composite_doublet_bandwidth.png`.

## Where it shows up in FresnelAnts

- `src/fresnelants/designs/reflectarray.py` — `Reflectarray`. Per-cell
  phase via a unit-cell phase-vs-patch-size lookup; default feed at
  $z = +F$ in front of the array (prime-focus geometry). The steering
  sweep regression test `tests/test_designs.py::test_reflectarray_steering`
  exercises off-broadside scan to ±30°.
- `src/fresnelants/designs/macro_array.py` — `MacroFresnelArray`. The
  array-factor × element-pattern decomposition is the phased-array
  abstraction; the elements happen to be lens-style apertures rather
  than patches.
- `src/fresnelants/synth/` — phase-synthesis backends with scipy,
  CVXPY (convex relaxation), and JAX (gradient-based) implementations.
  The motivating figures `docs/img/synth_broadside.png` show the
  inverse-design pipeline applied to a target far-field.
- `src/fresnelants/cells/` — vendor-grounded reconfigurable cell models
  that, while motivated by RIS (era 06), are used in `Reflectarray` to
  approximate the per-cell phase response of a real microstrip patch.

## Further reading

- Mailloux, *Phased Array Antenna Handbook*, 3rd ed. (Artech House,
  2017) — the canonical engineering reference.
- Hansen, *Phased Array Antennas*, 2nd ed. (Wiley, 2009).
- Kahrilas, *Electronic Scanning Radar Systems Design Handbook* (Artech
  House, 1976) — period reference for the PESA architecture.
- Skolnik, *Introduction to Radar Systems*, 3rd ed. (McGraw-Hill,
  2001) — chapter 9 covers the phased-array transition in radar.

For phased-array texts and the synthesis literature that descends from
them, see the
[Bibliography → phased arrays and synthesis](../../reference/bibliography.md#phased-arrays).

Continue to [Era 06 — reflectarrays, RIS, metasurfaces](06-reflectarrays-ris-metasurfaces.md).
