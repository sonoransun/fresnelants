# Notation & symbols

This page collects the symbols, units, and sign conventions used
throughout FresnelAnts. It is the formal companion to the
[Theory](../theory.md) page, and the spec the regression tests in
`tests/test_farfield.py`, `tests/test_designs.py`, and
`tests/test_conformal.py` enforce.

The conventions here match Balanis (*Antenna Theory: Analysis and
Design*, 4th ed.) and Goodman (*Introduction to Fourier Optics*, 4th
ed.). Every formula in the codebase has been chosen so these two
references are consistent — that is the entire reason this page exists.

## Geometric symbols

| Symbol | Meaning | Units | Where used |
|---|---|---|---|
| $F$ | Focal length (feed-to-aperture distance) | metres | every `AntennaDesign(focal_length=...)` constructor |
| $\lambda$ | Free-space wavelength | metres | `core/wavefront.py`; `λ = c / f` |
| $f$ | Operating frequency | Hz | every `solve(design, freq)` call |
| $k = 2\pi / \lambda$ | Free-space wavenumber | rad / metre | `core/wavefront.py`, `analysis/farfield.py` |
| $r_n$ | Radius of the $n$-th Fresnel zone on a flat aperture, for a feed at distance $F$ | metres | `core/geometry.py::fresnel_zone_radii(...)` |
| $a$ | Aperture radius (or half-side, for square apertures) | metres | every `aperture_radius_m=...` constructor |
| $(x, y)$ | Aperture-plane Cartesian coordinates | metres | `analysis/farfield.py` aperture grid |
| $(u, v)$ | Direction cosines: $u = \sin\theta \cos\phi$, $v = \sin\theta \sin\phi$ | dimensionless | far-field plots; `viz/plots2d.py` |
| $\theta$ | Polar angle from broadside (the $+z$ axis) | radians (degrees in the API) | `weights_for_beam(theta_deg=...)` |
| $\phi$ | Azimuth angle in the aperture plane | radians (degrees in the API) | far-field plots, steering directions |

The Fresnel zone radii satisfy
$$
r_n = \sqrt{n \lambda F + \left(\frac{n \lambda}{2}\right)^2},
$$
which reduces to $r_n \approx \sqrt{n \lambda F}$ in the paraxial limit
$F \gg n\lambda$.

The phase advance required to collapse a spherical wave from a feed at
$z = -F$ to a plane wave at the aperture is
$$
\phi(r) = -k\left(\sqrt{r^2 + F^2} - F\right).
$$
Wood, phase-correcting, and curvilinear designs all approximate this
profile with progressively finer phase resolution.

## Electromagnetic symbols

| Symbol | Meaning | Units | Where used |
|---|---|---|---|
| $E_z(x, y)$ | Aperture-plane electric field, $z$-component (single-pol scalar) | V / m | output of `aperture_field(...)` |
| $T(x, y; f, s)$ | Complex transmittance (or reflection coefficient) of the aperture at frequency $f$ in state $s$ | dimensionless complex | `transmittance(grid, freq, state)` |
| $E_{\text{inc}}(x, y; f)$ | Illumination field at the aperture | V / m | `default_illumination(grid, freq)` |
| $\Gamma$ | Reflection coefficient of a reflectarray cell | dimensionless complex | `cells/*.py`, `Reflectarray.cell_phases` |
| $\mathbf{J}(\theta, \phi)$ | Jones matrix of a metasurface unit cell | $2 \times 2$ complex | `cells/metasurface.py`, `analysis/dualpol_metrics.py` |
| $D$ | Peak directivity | dBi (or dimensionless linear) | `analysis/metrics.py::peak_directivity_dbi(...)` |
| $\eta$ | Aperture efficiency | dimensionless ($\in [0, 1]$) | `analysis/metrics.py::aperture_efficiency(...)` |
| HPBW | Half-power beamwidth | radians (degrees in the API) | `analysis/metrics.py::hpbw(...)` |
| SLL | First side-lobe level relative to the main beam | dB (negative) | `analysis/metrics.py::side_lobe_level(...)` |

The aperture field is the product of the illumination and the
transmittance:
$$
E_z(x, y; f, s) = T(x, y; f, s) \cdot E_{\text{inc}}(x, y; f).
$$

## Reconfigurable-cell symbols

| Symbol | Meaning | Units | Where used |
|---|---|---|---|
| $V$ | Bias voltage applied to a varactor | volts | `cells/varactor.py::reflection_coefficient(V, freq)` |
| $C(V)$ | Voltage-dependent varactor capacitance | farads | `cells/varactor.py::capacitance(V)` |
| $s_{\text{PIN}}$ | PIN diode state, ON or OFF | $\in \{0, 1\}$ | `cells/pin_diode.py` |
| $\epsilon_{\text{LC}}(V)$ | Voltage-dependent liquid-crystal permittivity | dimensionless | `cells/liquid_crystal.py` |
| $\mathbf{s}$ | Per-cell state vector for a reconfigurable array | mixed | `state` parameter on `transmittance(...)` |

## Time-modulated symbols

| Symbol | Meaning | Units | Where used |
|---|---|---|---|
| $T_m$ | Modulation period | seconds | `TimeModulatedArray(modulation_period=...)` |
| $f_m = 1 / T_m$ | Modulation frequency | Hz | `analysis/harmonics.py` |
| $n$ | Harmonic index | $\in \mathbb{Z}$ | `HarmonicPOSolver.solve(harmonics=...)` |
| $f_c$ | Carrier frequency | Hz | `solve(design, freq=f_c)` |

The time-modulated array radiates at frequencies $f_c + n f_m$ for
integer $n$. The harmonic patterns visible in
`docs/img/timemod_harmonic_beams.png` correspond to $n \in \{-2, -1, 0,
+1, +2\}$.

## Sign conventions

> **The library uses the e^{jωt} time convention throughout, consistent
> with Balanis and Stutzman & Thiele.** This is the convention that
> antenna engineers default to; physicists and some optics references
> use the opposite e^{-iωt} convention, and care is required when
> porting formulae across that boundary.

The consequences of e^{jωt}:

- **Outgoing spherical waves carry $\exp(-jkr) / r$.** Implemented by
  `SphericalWave` in `core/wavefront.py`.
- **The far-field integral uses a $+j$ Fourier kernel** —
  $\exp\bigl(+j(k_x x + k_y y)\bigr)$ — and is computed by
  `numpy.fft.ifft2(...)` inside
  `analysis/farfield.py::far_field_from_aperture(...)`.
  Switching to `numpy.fft.fft2(...)` without flipping the spectrum
  produces a beam pointing in the wrong half-space.
- **The phase advance through a focusing aperture is *negative*,**
  $\phi(r) = -k(\sqrt{r^2 + F^2} - F)$, because a converging wave's
  phase decreases toward the focal point with the convention chosen.
  The phase-correcting plate, the offset zone plate, the curvilinear
  surface, and the reflectarray *all had sign bugs in early
  development* that produced wrong-direction beam steering or aperture
  cancellation. The regression tests
  `tests/test_designs.py::test_reflectarray_steering` and
  `tests/test_farfield.py::test_uniform_circular_aperture_directivity`
  exist to catch reintroductions of those bugs.

> **When changing any phase formula, smoke-test with the steering sweep
> and the directivity sanity check.** The tests are fast (under a
> second) and they are load-bearing — a passing reflectarray steering
> sweep is the closest the library has to a "the conventions are right"
> certificate.

## Polarization conventions

| Symbol | Meaning |
|---|---|
| LCP | Left-handed circular polarization (IEEE convention: $E$ vector rotates *counter-clockwise* viewed from behind, in the direction of propagation) |
| RCP | Right-handed circular polarization (clockwise from behind) |
| co-pol | The polarization the antenna is designed to receive |
| cross-pol | The orthogonal polarization (typically a metric to *minimize* for single-pol designs and to *equalize* for dual-pol shared-aperture designs) |
| AR | Axial ratio (ratio of major-to-minor axis of the polarization ellipse, in dB; AR = 0 dB is perfect circular polarization, AR = ∞ is linear) |

The Jones-matrix dual-pol metrics in `analysis/dualpol_metrics.py`
follow the IEEE rotation convention. Researchers porting from physics
texts that use the opposite handedness convention should multiply the
off-diagonal Jones-matrix elements by $-1$ to match.

## Units in the API

The Python API is consistent about units: all lengths are **metres**,
all frequencies are **Hz**, all angles in user-facing constructors and
methods are **degrees** (with a `_deg` suffix). Internal calculations
convert to radians at the boundary. Examples:

```python
fa.PhaseCorrectingPlate(
    focal_length=0.10,        # 10 cm, in metres
    design_freq=30e9,         # 30 GHz, in Hz
    aperture_radius_m=0.05,   # 5 cm, in metres
    levels=8,                 # dimensionless
)

array.weights_for_beam(theta_deg=15.0, phi_deg=0.0)  # degrees, not radians
```

YAML and JSON specs follow the same convention. See
`src/fresnelants/io/specs.py` for the pydantic models.

## Cross-references

- The historical reasoning behind the conventions (why antenna
  engineering settled on e^{jωt} when physics did not) is in the
  [history overview](../background/history.md) and especially
  [era 03 — microwave reflectors](../background/history/03-microwave-reflectors.md).
- The full reading list, including the canonical sources for the
  conventions above, is in the
  [Bibliography](bibliography.md#foundational-antenna-texts).
- For the per-formula derivations, see [Theory](../theory.md).
