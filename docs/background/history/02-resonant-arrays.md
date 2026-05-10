# 1910 – 1940 — Resonant arrays

## Period

From the rise of continuous-wave (CW) transmission around 1910 through the
end of the broadcast era and the eve of the WWII radar buildup. The frequency
range expanded upward — MF AM broadcast (530 kHz – 1.7 MHz), HF
shortwave (3 – 30 MHz), and the first VHF experiments — pushing wavelengths
down from kilometres to metres.

## Driving applications

- **Commercial broadcast.** The 1920 inauguration of KDKA Pittsburgh and
  the rapid build-out of national AM networks demanded transmit antennas
  with predictable, *directional* coverage. Receive-side, every household
  needed an inexpensive, omnidirectional indoor antenna — but also,
  shortly after, *outdoor* antennas with enough gain to discriminate
  between adjacent stations.
- **Long-haul shortwave.** Trans-oceanic CW telegraphy and, later, voice
  broadcasting (BBC Empire Service 1932; Voice of America 1942) ran on
  large directive arrays beamed at specific continents.
- **Amateur experimentation.** The amateur community pushed into HF and
  VHF, and built much of the experimental knowledge — Yagi-Uda, log-periodic
  precursors, rhombic, Beverage — that the professional world adopted.

## Technological step

Three innovations defined the era:

1. **Tuned wire arrays.** A small number of resonant elements arranged on a
   wavelength-scale grid produce a directive pattern via constructive
   interference along one axis. The textbook *array factor* — the
   summation $\sum_n w_n \exp(j \mathbf{k} \cdot \mathbf{r}_n)$ that
   modulates a single-element pattern — was first analysed quantitatively
   in this era.
2. **The Yagi-Uda antenna** (Uda 1926; Yagi 1928). A driven dipole with a
   reflector and one or more directors achieves 8–15 dBi gain from a
   handful of metres of wire. It became the canonical
   "antenna-on-the-roof" of the broadcast and amateur eras and remains in
   production today.
3. **Travelling-wave receive antennas.** Beverage (1922) and the rhombic
   (Bruce 1931) exploited the fact that a long wire terminated in its
   characteristic impedance behaves as a directive travelling-wave
   structure. These were the first antennas designed *purely for
   reception* — high directivity, low efficiency, but enormous front-to-back
   ratio.

## What was lost / what was gained

- **Gained:** the *array factor*. The recognition that an array's far-field
  is the product of an element pattern and a geometry-dependent factor is
  the foundation of every steered-beam antenna built since. The
  decomposition is so robust that it underlies the
  [`MacroFresnelArray`](../../designs/macro_array.md) class added in v0.5
  — eighty-five years later — without modification.
- **Gained:** *steering by phasing*. Even before electronic phase shifters
  existed, wire arrays could be steered by mechanical relocation of taps
  on a transmission line. The principle that relative phase between
  elements determines beam direction was now part of the engineering
  vocabulary.
- **Lost:** the broadband response of the spark-gap dipole. Resonant
  arrays only work near their design frequency. AM broadcast antennas
  required separate engineering for each channel; shortwave antennas
  were carved into the band by physical geometry.
- **Lost:** simplicity. A Yagi-Uda is a precision instrument: element
  lengths and spacings within a few percent determine whether you have a
  10 dBi forward beam or a useless cardioid.

## What survived

The array-factor decomposition is the single most important survivor.
Stated in modern form for an array of identical elements at positions
$\mathbf{r}_n$ with complex weights $w_n$, illuminated by a wave with
direction cosines $(u, v)$:

$$
F(u, v) = E_{\text{element}}(u, v) \cdot \sum_n w_n \exp\bigl(j k (u x_n + v y_n)\bigr)
$$

This formula is the mathematical heart of the v0.5 `MacroFresnelArray`
class. It is also why the per-cell phase pattern of a `Reflectarray`
(v0.1) and the per-cell bias state of a `ReconfigurableArray` (v0.2) are
both expressible as *array weights*: the weight is the cell's complex
transmittance.

The era also contributed *receive-only design*. The Beverage antenna's
willingness to trade efficiency for directivity foreshadows the modern
receive-array codebook, where you optimize for SINR at the *output* of the
array rather than for the gain of any single element. The
`MacroFresnelArray.beam_codebook(...)` API descends from this idea.

## Where it shows up in FresnelAnts

- `src/fresnelants/designs/macro_array.py` — `MacroFresnelArray.solve(...)`
  computes its far-field by summing the array factor over element
  positions, exactly as the era's wire-array engineers did by hand. The
  per-element pattern is provided by any `AntennaDesign` subclass
  (zone plate, reflectarray, conformal lens, fractal).
- `src/fresnelants/designs/macro_array.py` — `from_lattice("linear" |
  "rect" | "hex" | "ring", ...)` enumerates the canonical wire-array
  geometries the era investigated.
- `src/fresnelants/designs/macro_array.py` — `weights_for_beam(theta, phi,
  bits=N)` and `beam_codebook(directions, ...)` operationalize the
  conjugate-matched and codebook receive concepts that descended from
  Beverage and rhombic practice.

## Further reading

- Yagi, "Beam transmission of ultra short waves", *Proc. IRE* 16, 1928.
- Beverage, Rice, & Kellogg, "The wave antenna: a new type of highly
  directive antenna", *Trans. AIEE* 42, 1923.
- Brown, *Radio and Radar Antennas* (RCA, 1947) — collected design rules
  from the era.

For canonical modern treatments of the array factor that descend directly
from this period, see the
[Bibliography](../../reference/bibliography.md#foundational-antenna-texts).

Continue to [Era 03 — microwave reflectors](03-microwave-reflectors.md).
