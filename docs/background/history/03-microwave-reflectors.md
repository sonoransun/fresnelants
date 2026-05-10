# 1940 – 1970 — Microwave reflectors

## Period

From the WWII radar buildup, when the cavity magnetron made centimetric
wavelengths practical for the first time, through the early commercial
satellite era. Frequencies climbed from UHF (300 MHz) into S-, C-, and
X-band (~10 GHz), with experimental Ka-band and millimetre-wave work
beginning in the late 1960s. Wavelengths ran from a metre down to a few
centimetres — the era when antennas finally became *small relative to the
human body* and *large relative to the wavelength*, the regime in which
geometric and physical optics reasoning becomes accurate.

## Driving applications

- **Search and fire-control radar.** The Allied Chain Home network, the
  AI Mark IV airborne intercept radar, the SCR-584 anti-aircraft fire
  control set, and their successors created an enormous demand for
  high-gain, narrow-beam apertures in 200 MHz – 10 GHz.
- **Satellite ground stations.** Echo (1960), Telstar (1962), and the
  Intelsat geosynchronous series drove the construction of fixed
  parabolic dishes ranging from 10 m to 64 m (the Goldstone DSS-14
  antenna). The link budget was unforgiving: a passive Echo balloon
  required +60 dB of receive gain just to close a usable channel.
- **Radio astronomy.** Reber (1937), Jodrell Bank (1957), Arecibo (1963),
  and Effelsberg (1972) pushed reflector technology to mechanical limits
  — and proved that the parabolic dish was the cheapest path to large
  collecting area at centimetric wavelengths.

## Technological step

The era's defining innovation was the **shaped reflector** — a large
metallic surface curved so that incident plane waves converge to a focal
point (or, in transmit, a feed at the focus is collimated into a plane
wave). The parabolic reflector had been understood optically since
Archimedes, but only in this era did it become buildable and useful at
radio wavelengths.

Three reflector geometries dominated:

1. **Prime-focus paraboloid.** Feed at the focal point, dish concave
   toward the sky. Simple, but the feed and its support structure block
   part of the aperture (aperture blockage), and the long feed cable
   loses noise temperature at the LNA.
2. **Cassegrain.** A subreflector at the prime focus folds the focal
   path back through a hole in the main dish, allowing the LNA to sit
   *behind* the antenna. Standard for satellite uplink terminals.
3. **Offset paraboloid.** The aperture is a non-axisymmetric section of a
   parent paraboloid, with the feed displaced so that it does not block
   the aperture. This is the geometry of every modern Ku-band TV
   "dish" mounted on a residential wall.

Alongside the reflector, the **horn antenna** matured as a wideband,
low-sidelobe, calibrated feed and as a stand-alone primary aperture for
short-haul terrestrial links (the Bell System TD-2 4 GHz network).

## What was lost / what was gained

- **Gained:** *aperture efficiency at large electrical sizes*. A
  well-illuminated 10 m dish at 10 GHz exceeds 70 % aperture efficiency
  routinely. The same gain from an array would require ~10⁵ phased
  elements; in 1960, no such array was buildable.
- **Gained:** *broadband performance*. Reflector phase arises from
  geometry alone — a parabola is achromatic over decades of frequency.
  The bandwidth of a reflector system is limited only by its feed.
- **Lost:** *steerability*. Mechanical pointing is slow, expensive, and
  vulnerable to wind. A reflector pointed at a geosynchronous satellite
  is fine; a reflector tracking a low-Earth-orbit satellite or scanning
  a search volume is a different proposition.
- **Lost:** *flatness*. Reflectors are bulky, and at large diameters
  surface tolerances become a fabrication nightmare (the rule of thumb
  is $\lambda / 16$ RMS surface error).

## What survived

The reflector lineage gave FresnelAnts three durable concepts:

1. **The feed-aperture decomposition.** A reflector system is modelled as
   a *feed pattern* (the horn) illuminating a *passive aperture* (the
   reflecting surface, or its equivalent transmittance). This is exactly
   the structure of `AntennaDesign` in `src/fresnelants/designs/base.py`:
   `default_illumination(grid, freq)` returns the feed, `transmittance(grid,
   freq, state)` returns the aperture, and `aperture_field(...)` multiplies
   them. The reflectarray (era 06) is a flat reflector that emulates a
   parabolic phase profile with per-cell tuning, but it inherits the same
   decomposition unchanged.
2. **The offset geometry.** The offset paraboloid solves the blockage
   problem by displacing the feed off-axis. The
   [`OffsetZonePlate`](../../designs/offset.md) in this library is a flat
   aperture with the same structural goal: phase the zones for a feed at
   $z = -F$ tilted by $\theta_{\text{offset}}$, recovering a broadside
   beam without blocking the aperture.
3. **The link-budget mindset.** Reflector engineers think in terms of
   $G/T$, aperture efficiency $\eta$, and edge taper. The metrics module
   `analysis/metrics.py` exports peak directivity, HPBW, side-lobe level,
   and aperture efficiency — the same four numbers a Goldstone engineer
   would have demanded in 1965.

## What it could not do

The reflector era did not solve electronic beam steering. A radar that
needed to scan a 90° azimuth wedge in milliseconds had to either spin a
mechanical gimbal or — eventually — abandon the reflector and adopt the
phased array. The transition is the subject of
[Era 05](05-phased-arrays.md).

## Where it shows up in FresnelAnts

- `src/fresnelants/designs/base.py` — the feed/aperture/transmittance
  decomposition is exactly the reflector engineer's mental model. The
  *only* thing FresnelAnts changes is that the aperture is flat and the
  feed pattern is an explicit illumination function rather than a
  physical horn.
- `src/fresnelants/designs/offset.py` — `OffsetZonePlate` is the flat
  Fresnel analogue of the offset paraboloid.
- `src/fresnelants/designs/reflectarray.py` — `Reflectarray` defaults its
  feed to `z = +F` (in front of the array, like a prime-focus dish)
  rather than `z = -F` (behind the plate, like a transmissive lens). The
  override is documented in `default_illumination(...)`.
- `src/fresnelants/core/wavefront.py` — `CosineFeed(theta, n)` provides
  the cosine-tapered feed pattern that approximates a horn's principal
  cut, parameterised by the taper exponent $n$.
- `tests/test_designs.py::test_reflectarray_steering` — the load-bearing
  steering-sweep test. A reflector cannot steer; a reflectarray can; the
  test exists because the era-06 designs needed to demonstrate that a
  flat aperture with electronic phase control reproduces a steered
  reflector beam without mechanical pointing.

## Further reading

- Silver (ed.), *Microwave Antenna Theory and Design* (MIT Rad Lab vol.
  12, 1949) — the canonical wartime synthesis.
- Hannan, "Microwave antennas derived from the Cassegrain telescope",
  *IRE Trans. AP* 9, 1961.
- Love (ed.), *Reflector Antennas* (IEEE Press, 1978) — collected reprints.

For modern reflector treatments, see
[Bibliography → reflector and lens texts](../../reference/bibliography.md#foundational-antenna-texts).

Continue to [Era 04 — lenses & zone plates](04-lens-and-zone-plates.md).
