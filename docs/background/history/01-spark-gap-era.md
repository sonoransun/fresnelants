# 1888 – 1910 — The spark-gap era

## Period

Roughly 1888, when Heinrich Hertz first generated and detected radio waves in
his Karlsruhe laboratory, to about 1910, by which time continuous-wave
transmission (Fessenden, Alexanderson) was beginning to displace the noisy
spark-gap transmitter and the equally noisy coherer detector.

The dominant bands were LF and MF, kilohertz to a few megahertz. Wavelengths
ran from hundreds of metres to many kilometres. Antennas were correspondingly
large — Marconi's 1901 Newfoundland receiving "antenna" was a 152 m wire kite
flown above Signal Hill.

## Driving applications

- **Pure science** in Hertz's hands: confirming that Maxwell's equations
  predicted wave propagation through free space, and that electromagnetic
  energy could be radiated and re-detected.
- **Wireless telegraphy** in Marconi's hands: replacing the submarine cable
  with point-to-point Morse links. Ship-to-shore communication became the
  first commercial driver, with the *Titanic* disaster of 1912 cementing the
  technology's social importance.
- **Naval coordination**: by the late 1890s the British and Italian navies
  had operational shipboard installations.

## Technological step

The era's defining innovation was the recognition that a *resonant structure*
could couple efficiently to a propagating electromagnetic wave. Hertz's
loop-and-spark detector and Marconi's elevated wire each acted as an LC
circuit tuned to the transmitter's frequency, with the wire itself
contributing both inductance and radiation resistance. That insight — that an
antenna *is* a tuned circuit driven by the incident field — survived the era
intact and underlies every later design.

The spark-gap *transmitter* radiated a damped sinusoid burst with a wide
spectral footprint; the *receiver* was an open-circuited or coherer-tipped
wire that integrated incident energy and delivered a click to a Morse decoder.
The system was effectively monopulse, monochannel, and unsteerable.

## What was lost / what was gained

- **Gained:** the first quantitative link budget — Marconi could predict
  range from transmit power, antenna height, and ground conductivity, even
  if the underlying propagation theory (Sommerfeld surface wave, ionospheric
  reflection) would not be settled for another two decades.
- **Lost:** any concept of *direction*. Spark-gap antennas were essentially
  omnidirectional; the only directional control came from siting (kite vs.
  ground plane) and from polarization (vertical for ground-wave links).
- **Gained:** broadband energy collection — a long wire couples to almost
  any frequency below its cutoff.
- **Lost:** any concept of *spectrum sharing*. The era ended in part
  because the spark-gap occupied so much bandwidth that simultaneous
  operation of multiple stations became socially intolerable. The
  International Radio Telegraphic Convention of 1906 began the long
  process of band allocation that culminates in the modern 5G New Radio
  spectrum maps.

## What survived

Three concepts from the spark-gap era still appear in this codebase:

1. **The dipole as a primitive feed model.** The `SphericalWave` source in
   `core/wavefront.py` represents a point source whose phase and amplitude
   depend on distance from the feed; this is the Hertzian-dipole far-field
   approximation. Every transmitting plate in this library is illuminated by
   that abstraction by default.
2. **Reciprocity.** Hertz's experiments confirmed that the same antenna
   transmits and receives with the same pattern. FresnelAnts computes the
   transmit far-field via FFT of the aperture and treats the receive
   far-field as identical. There is no separate receive-mode solver.
3. **Aperture as integrator.** Marconi's long-wire receiver integrated
   energy along its length. The PO solver in `analysis/farfield.py` is the
   continuous-aperture version of that idea: integrate `T(x, y) · E_inc(x, y)`
   over the aperture, then transform.

## Where it shows up in FresnelAnts

- `src/fresnelants/core/wavefront.py` — `PlaneWave`, `SphericalWave`,
  `CosineFeed`. The `SphericalWave` class models a Hertzian feed at finite
  distance from a plate; `PlaneWave` models the far-field limit (a transmitter
  beyond the horizon, as in modern satellite reception).
- `src/fresnelants/designs/base.py` — the `default_illumination(grid, freq)`
  hook on `AntennaDesign` returns a `SphericalWave` at `z = -F` for plates
  whose feed sits behind the aperture, or `z = +F` for a reflectarray whose
  feed sits in front.
- `tests/test_farfield.py::test_uniform_circular_aperture_directivity` — a
  uniform-illumination sanity check that any era-01 antenna theorist would
  recognise: the far-field directivity of a fully-lit circular aperture is
  $4\pi A / \lambda^2$, and the test confirms the FFT path reproduces it.

## Further reading

- Hertz, *Electric Waves* (Macmillan, 1893; English translation 1900) — the
  primary source for the resonant-loop receiver.
- Marconi, "Wireless telegraphic communication", *Nobel Lecture*, 1909.
- Burns, *Communications: An International History of the Formative Years*
  (IET, 2004), chapters 1–4 — modern survey of the period.

For canonical antenna textbooks that cover the dipole / Hertzian formulae
that survived the era, see the
[Bibliography](../../reference/bibliography.md#foundational-antenna-texts).

Continue to [Era 02 — resonant arrays](02-resonant-arrays.md).
