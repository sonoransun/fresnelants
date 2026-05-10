"""Single source of truth for every committed FresnelAnts figure.

Running ``python docs/generate_figures.py`` re-renders the entire ``docs/img``
gallery from the library code. CI invokes this and ``git diff --exit-code
docs/img/`` to ensure documentation never drifts from the implementation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fresnelants as fa  # noqa: E402
from fresnelants.viz.plots2d import (  # noqa: E402
    plot_farfield_2d,
    plot_principal_cuts,
    plot_zone_layout,
)
from fresnelants.viz.plots3d import (  # noqa: E402
    plot_3d_radiation_pattern,
    plot_3d_surface_profile,
    plot_focal_region,
)


def _save(fig: object, path: Path) -> None:
    fig.savefig(path)  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[arg-type]


def _hero_panel(out_dir: Path) -> None:
    """Two-up hero figure: 3D lens surface + 3D radiation pattern."""
    freq = 30e9
    F, R = 0.10, 0.05
    lens = fa.CurvilinearFresnel(
        focal_length=F, design_freq=freq, aperture_radius_m=R, profile="hyperbolic"
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    res = solver.solve(lens, freq)

    fig = plt.figure(figsize=(10, 4.6))
    from fresnelants.units import freq_to_wavelength

    lam = float(freq_to_wavelength(freq))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    R_ax = np.linspace(0, lens.aperture_radius, 80)
    theta = np.linspace(0, 2 * np.pi, 200)
    Rg, Tg = np.meshgrid(R_ax, theta, indexing="xy")
    X = Rg * np.cos(Tg) * 1e3
    Y = Rg * np.sin(Tg) * 1e3
    Z = lens.fresnel_depth(Rg) * 1e3
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("depth [mm]")
    ax.set_title("Hyperbolic Fresnel singlet")
    fig.colorbar(surf, ax=ax, shrink=0.55, label="depth [mm]")

    d = res.far_field.directivity()
    d_db = 10.0 * np.log10(np.maximum(d, 1e-30))
    peak = float(np.nanmax(d_db))
    floor = peak - 30
    radius = np.clip(d_db - floor, 0.0, None)
    mask = res.far_field.U**2 + res.far_field.V**2 <= 1.0
    R_pat = np.where(mask, radius, np.nan)
    cos_theta = np.sqrt(np.clip(1 - res.far_field.U**2 - res.far_field.V**2, 0, 1))
    Xs = R_pat * res.far_field.U
    Ys = R_pat * res.far_field.V
    Zs = R_pat * cos_theta
    step = max(1, R_pat.shape[0] // 80)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    cs = R_pat[::step, ::step]
    surf2 = ax2.plot_surface(
        Xs[::step, ::step],
        Ys[::step, ::step],
        Zs[::step, ::step],
        facecolors=plt.cm.inferno(cs / max(np.nanmax(cs), 1e-9)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    surf2.set_clim(0, np.nanmax(cs))
    ax2.set_box_aspect((1, 1, 0.7))
    ax2.set_title(f"3D radiation pattern (peak {peak:.1f} dBi)")
    ax2.set_xlabel("u·DR")
    ax2.set_ylabel("v·DR")
    ax2.set_zlabel("cos θ·DR")
    mappable = plt.cm.ScalarMappable(cmap="inferno")
    mappable.set_array(cs)
    mappable.set_clim(floor, peak)
    fig.colorbar(mappable, ax=ax2, shrink=0.55, label="dBi")
    fig.suptitle(
        f"FresnelAnts hero — {freq / 1e9:.0f} GHz hyperbolic singlet "
        f"({2 * R * 1e3:.0f} mm aperture, F/D = {F / (2 * R):.2f}, λ = {lam * 1e3:.1f} mm)"
    )
    fig.tight_layout()
    _save(fig, out_dir / "hero.png")


def _zone_plate_gallery(out_dir: Path) -> None:
    F, freq = 1.0, 10e9
    soret = fa.SoretZonePlate(focal_length=F, design_freq=freq, num_zones=12)
    wood = fa.WoodZonePlate(focal_length=F, design_freq=freq, num_zones=12)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    res_w = solver.solve(wood, freq)

    _save(plot_zone_layout(soret), out_dir / "zone_plate_soret_layout.png")
    _save(plot_zone_layout(wood), out_dir / "zone_plate_wood_layout.png")
    _save(plot_farfield_2d(res_w.far_field), out_dir / "zone_plate_wood_farfield.png")
    _save(
        plot_principal_cuts(res_w.far_field, label="Wood 12 zones, 10 GHz"),
        out_dir / "zone_plate_wood_cuts.png",
    )


def _offset_gallery(out_dir: Path) -> None:
    F, freq, tilt = 0.50, 12e9, math.radians(25.0)
    d = fa.OffsetZonePlate(
        focal_length=F, design_freq=freq, aperture_radius_m=0.30, tilt_angle=tilt
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    res = solver.solve(d, freq)
    _save(plot_zone_layout(d), out_dir / "offset_layout.png")
    _save(plot_farfield_2d(res.far_field), out_dir / "offset_farfield.png")
    _save(
        plot_principal_cuts(res.far_field, label="Offset 25°, 12 GHz"),
        out_dir / "offset_cuts.png",
    )


def _phase_correcting_gallery(out_dir: Path) -> None:
    F, freq, R = 0.05, 94e9, 0.025
    plate = fa.PhaseCorrectingPlate(focal_length=F, design_freq=freq, aperture_radius_m=R, levels=4)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    res = solver.solve(plate, freq)
    _save(plot_zone_layout(plate), out_dir / "phase_correcting_layout.png")
    _save(plot_farfield_2d(res.far_field), out_dir / "phase_correcting_farfield.png")
    _save(
        plot_principal_cuts(res.far_field, label="4-level phase plate, 94 GHz"),
        out_dir / "phase_correcting_cuts.png",
    )

    levels_seq = [1, 2, 4, 8, 1024]
    gains: list[float] = []
    for levels in levels_seq:
        if levels == 1:
            d = fa.SoretZonePlate(focal_length=F, design_freq=freq, num_zones=12)
        elif levels == 2:
            d = fa.WoodZonePlate(focal_length=F, design_freq=freq, num_zones=12)  # type: ignore[assignment]
        else:
            d = fa.PhaseCorrectingPlate(  # type: ignore[assignment]
                focal_length=F, design_freq=freq, aperture_radius_m=R, levels=levels
            )
        gains.append(solver.solve(d, freq).far_field.peak_directivity_dbi())

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    labels = [str(n) if n < 1024 else "cont." for n in levels_seq]
    ax.bar(labels, gains, color="tab:blue")
    ax.axhline(
        10 * np.log10(4 * np.pi * (np.pi * R**2) / (3e8 / freq) ** 2),
        color="red",
        linestyle="--",
        label="uniform-aperture max",
    )
    ax.set_xlabel("phase levels")
    ax.set_ylabel("Peak directivity [dBi]")
    ax.set_title(f"Levels vs gain @ {freq / 1e9:.0f} GHz")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir / "phase_correcting_levels.png")


def _reflectarray_gallery(out_dir: Path) -> None:
    F, freq = 0.20, 28e9
    nx = ny = 32
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    ra = fa.Reflectarray(focal_length=F, design_freq=freq, nx=nx, ny=ny)
    res = solver.solve(ra, freq)
    _save(plot_zone_layout(ra), out_dir / "reflectarray_layout.png")
    _save(plot_farfield_2d(res.far_field), out_dir / "reflectarray_farfield.png")
    _save(
        plot_principal_cuts(res.far_field, label="Broadside RA, 28 GHz"),
        out_dir / "reflectarray_cuts.png",
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for theta in (0, 10, 20, 30, 45):
        ra_s = fa.Reflectarray(
            focal_length=F,
            design_freq=freq,
            nx=nx,
            ny=ny,
            beam_direction=(math.radians(theta), 0.0),
        )
        ff = solver.solve(ra_s, freq).far_field
        theta_deg, db = ff.cut("E")
        valid = ~np.isnan(theta_deg)
        ax.plot(theta_deg[valid], db[valid], label=f"θ_b = {theta}°")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-40, 2)
    ax.set_xlabel("θ [deg]")
    ax.set_ylabel("Normalized gain [dB]")
    ax.set_title(f"Reflectarray beam steering at {freq / 1e9:.0f} GHz")
    ax.legend(loc="lower center", ncol=3)
    fig.tight_layout()
    _save(fig, out_dir / "reflectarray_steering.png")


def _curvilinear_gallery(out_dir: Path) -> None:
    F, freq, R = 0.10, 30e9, 0.05
    lens = fa.CurvilinearFresnel(
        focal_length=F, design_freq=freq, aperture_radius_m=R, profile="hyperbolic"
    )
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=6.0, pad_factor=4)
    res = solver.solve(lens, freq)
    _save(plot_3d_surface_profile(lens, samples=160), out_dir / "curvilinear_surface_3d.png")
    _save(plot_farfield_2d(res.far_field), out_dir / "curvilinear_farfield.png")
    _save(
        plot_principal_cuts(res.far_field, label="Hyperbolic singlet, 30 GHz"),
        out_dir / "curvilinear_cuts.png",
    )
    _save(
        plot_3d_radiation_pattern(res.far_field, dynamic_range_db=30),
        out_dir / "curvilinear_3d_pattern.png",
    )
    _save(
        plot_focal_region(res.aperture, z_focal=F, z_span=0.6 * F, samples=41),
        out_dir / "curvilinear_focal_region.png",
    )

    # Axicon variant for the curvilinear page.
    axicon = fa.CurvilinearFresnel(
        focal_length=F,
        design_freq=freq,
        aperture_radius_m=R,
        profile="axicon",
        axicon_angle=math.radians(8.0),
    )
    _save(plot_3d_surface_profile(axicon, samples=160), out_dir / "curvilinear_axicon_3d.png")


def _composite_gallery(out_dir: Path) -> None:
    f_low, f_high = 28e9, 32e9
    F, R = 0.10, 0.05
    doublet = fa.AchromaticDoublet(
        f_low=f_low, f_high=f_high, aperture_radius_m=R, levels=8, feed_distance=F
    )
    single = fa.PhaseCorrectingPlate(
        focal_length=F, design_freq=30e9, aperture_radius_m=R, levels=8
    )
    cas = fa.CascadePOSolver(samples_per_wavelength=4.0, pad_factor=2)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2)
    freqs = np.linspace(24e9, 36e9, 9)
    g_d = [cas.solve(doublet, f).far_field.peak_directivity_dbi() for f in freqs]
    g_s = [solver.solve(single, f).far_field.peak_directivity_dbi() for f in freqs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs / 1e9, g_d, "o-", color="tab:blue", label="Achromatic doublet")
    ax.plot(freqs / 1e9, g_s, "s--", color="tab:orange", label="Single 8-level plate")
    ax.axvspan(f_low / 1e9, f_high / 1e9, alpha=0.10, color="tab:blue", label="design band")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Peak directivity [dBi]")
    ax.set_title("Achromatic doublet bandwidth")
    ax.legend(loc="lower center")
    fig.tight_layout()
    _save(fig, out_dir / "composite_doublet_bandwidth.png")


def _ris_gallery(out_dir: Path) -> None:
    from fresnelants.cells.pin_diode import MACOM_MA4FCP305
    from fresnelants.cells.varactor import Skyworks_SMV1232

    cell = Skyworks_SMV1232()
    voltages = np.linspace(0, 15, 64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(voltages, np.degrees(np.unwrap(cell.phase(voltages, 28e9))), color="tab:blue")
    axes[0].set_xlabel("Reverse bias [V]")
    axes[0].set_ylabel("Reflection phase [deg]")
    axes[0].set_title("Skyworks SMV1232 phase vs bias @ 28 GHz")
    axes[1].plot(voltages, 20 * np.log10(cell.loss(voltages, 28e9)), color="tab:red")
    axes[1].set_xlabel("Reverse bias [V]")
    axes[1].set_ylabel("Loss [dB]")
    axes[1].set_title("Loss vs bias @ 28 GHz")
    fig.tight_layout()
    _save(fig, out_dir / "ris_varactor_cv.png")

    ris = fa.ReconfigurableArray(focal_length=0.20, design_freq=28e9, nx=24, ny=24, cell=cell)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=2)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for theta in (0, 10, 20, 30, 45):
        ris.beam_direction = (math.radians(theta), 0)
        ff = solver.solve(ris, 28e9).far_field
        td, db = ff.cut("E")
        ax.plot(td[~np.isnan(td)], db[~np.isnan(td)], label=f"θ_b = {theta}°")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-40, 2)
    ax.set_xlabel("θ [deg]")
    ax.set_ylabel("Normalized gain [dB]")
    ax.set_title("Skyworks-SMV1232 RIS 24×24 beam steering at 28 GHz")
    ax.legend(loc="lower center", ncol=3)
    fig.tight_layout()
    _save(fig, out_dir / "ris_varactor_steering.png")

    pin = MACOM_MA4FCP305()
    angles = np.arange(-60, 65, 5, dtype=float)
    g_1bit = []
    for theta in angles:
        ris1 = fa.CodedRIS(
            focal_length=0.20,
            design_freq=39e9,
            nx=20,
            ny=20,
            cell=pin,
            bits=1,
            beam_direction=(math.radians(theta), 0),
        )
        g_1bit.append(solver.solve(ris1, 39e9).far_field.peak_directivity_dbi())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(angles, g_1bit, "o-", color="tab:red")
    ax.set_xlabel("Steering angle θ_b [deg]")
    ax.set_ylabel("Peak directivity [dBi]")
    ax.set_title("1-bit MACOM PIN RIS @ 39 GHz — scan loss")
    fig.tight_layout()
    _save(fig, out_dir / "ris_1bit_scanloss.png")


def _metasurface_gallery(out_dir: Path) -> None:
    from fresnelants.analysis.dualpol_metrics import cross_polarization_db
    from fresnelants.analysis.farfield import far_field_from_jones_aperture

    lens = fa.MetasurfaceLens(focal_length=0.05, design_freq=60e9, aperture_radius_m=0.025)
    ap = lens.jones_aperture_field(60e9, samples_per_wavelength=4.0)
    ff = far_field_from_jones_aperture(ap, pad_factor=4)
    co, _cross = ff.co_cross("rcp")
    co_db = 10 * np.log10(np.maximum(np.abs(co) ** 2, 1e-30))
    co_db[~ff.visible_mask] = np.nan
    cross_db = cross_polarization_db(ff, "rcp")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    extent = (ff.u[0], ff.u[-1], ff.v[0], ff.v[-1])
    im0 = axes[0].imshow(co_db, extent=extent, origin="lower", cmap="inferno")
    axes[0].set_title("Co-pol (RCP) far-field [dB]")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(cross_db, extent=extent, origin="lower", cmap="inferno")
    axes[1].set_title("Cross-pol level [dB below co peak]")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    for ax in axes:
        ax.set_xlabel(r"$u$")
        ax.set_ylabel(r"$v$")
        ax.set_aspect("equal")
    fig.suptitle("PB metasurface lens @ 60 GHz — LCP in / RCP focused")
    fig.tight_layout()
    _save(fig, out_dir / "metasurface_pb_pol.png")


def _conformal_gallery(out_dir: Path) -> None:
    from fresnelants.analysis.conformal_farfield import far_field_from_conformal

    lens = fa.CylindricalFresnelLens(radius=0.05, height=0.04, design_freq=77e9, nu=60, nv=30)
    ap = lens.conformal_aperture(77e9)
    ff = far_field_from_conformal(ap, n_samples=48, chunk=4096)
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    pts = ap.points
    mag = np.abs(ap.Et)
    sc = ax.scatter(pts[:, 0] * 1e3, pts[:, 1] * 1e3, pts[:, 2] * 1e3, c=mag, cmap="magma", s=4)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("Cylindrical aperture (5 cm × 4 cm)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="|Et|")
    ax2 = fig.add_subplot(1, 2, 2)
    d = ff.directivity()
    d_db = 10 * np.log10(np.maximum(d, 1e-30))
    im = ax2.imshow(
        d_db,
        extent=(ff.u[0], ff.u[-1], ff.v[0], ff.v[-1]),
        origin="lower",
        cmap="inferno",
        vmin=d_db.max() - 30,
        vmax=d_db.max(),
    )
    ax2.set_title(f"Cylindrical lens far-field — peak {d_db.max():.1f} dBi")
    ax2.set_xlabel(r"$u$")
    ax2.set_ylabel(r"$v$")
    ax2.set_aspect("equal")
    fig.colorbar(im, ax=ax2, shrink=0.85)
    fig.tight_layout()
    _save(fig, out_dir / "conformal_cylindrical_77ghz.png")


def _timemod_gallery(out_dir: Path) -> None:
    from fresnelants.analysis.harmonics import harmonic_far_field
    from fresnelants.cells.varactor import Skyworks_SMV1232

    array = fa.TimeModulatedArray(
        focal_length=0.20, design_freq=28e9, nx=24, ny=24, cell=Skyworks_SMV1232()
    )
    array.linear_progressive_schedule(time_per_cell=0.5, duty=0.5)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n in (-2, -1, 0, 1, 2):
        ff = harmonic_far_field(array, 28e9, n, samples_per_wavelength=4.0, pad_factor=2)
        td, db = ff.cut("E")
        valid = ~np.isnan(td)
        ax.plot(td[valid], db[valid], label=f"n = {n:+d}")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-40, 2)
    ax.set_xlabel("θ [deg]")
    ax.set_ylabel("Normalized gain [dB]")
    ax.set_title("Time-modulated 24×24 array — harmonic beams (28 GHz carrier)")
    ax.legend(loc="lower center", ncol=5)
    fig.tight_layout()
    _save(fig, out_dir / "timemod_harmonic_beams.png")


def _synth_gallery(out_dir: Path) -> None:
    from fresnelants.synth.scipy_backend import synth_phase_scipy

    array = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=12, ny=12)
    M = 32
    u = np.linspace(-1.0, 1.0, M)
    v = np.linspace(-1.0, 1.0, M)
    U, V = np.meshgrid(u, v, indexing="xy")
    target = np.exp(-200 * (U**2 + V**2)).astype(np.float64)
    res = synth_phase_scipy(target, array, 28e9, max_iter=80)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    im0 = axes[0].imshow(target, extent=(-1, 1, -1, 1), origin="lower", cmap="inferno")
    axes[0].set_title("Target far-field magnitude")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    axes[0].set_xlabel(r"$u$")
    axes[0].set_ylabel(r"$v$")
    axes[0].set_aspect("equal")

    im1 = axes[1].imshow(
        res.phase,
        extent=(0, array.nx, 0, array.ny),
        origin="lower",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1].set_title("Synthesized cell phase [rad]")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    axes[1].set_xlabel("cell column")
    axes[1].set_ylabel("cell row")
    fig.suptitle(f"Phase synthesis — final loss = {res.final_loss:.2g}")
    fig.tight_layout()
    _save(fig, out_dir / "synth_broadside.png")


def _measured_gallery(out_dir: Path) -> None:
    from fresnelants.cells.measured import MeasuredCell
    from fresnelants.cells.varactor import Skyworks_SMV1232

    var = Skyworks_SMV1232()
    states = np.linspace(0, 15, 64)
    freqs = np.array([28e9])
    s11 = np.array([[var.loss(s, 28e9) * np.exp(1j * var.phase(s, 28e9))] for s in states])
    measured = MeasuredCell.from_arrays(states, freqs, s11)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(states, np.degrees(np.unwrap(var.phase(states, 28e9))), "tab:blue", label="model")
    ax.plot(
        states,
        np.degrees(np.unwrap(measured.phase(states, 28e9))),
        "k--",
        label="MeasuredCell roundtrip",
    )
    ax.set_xlabel("Reverse bias [V]")
    ax.set_ylabel("Reflection phase [deg]")
    ax.set_title("MeasuredCell roundtrip — Skyworks SMV1232 @ 28 GHz")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "measured_roundtrip.png")


def _annotate_axial_intensity(ax: object, F: float) -> None:
    """Add labeled F, F/3, F/5 markers to a polyfocal axial-intensity axis."""
    for k, label in ((1, "F"), (3, "F/3"), (5, "F/5")):
        ax.axvline(1.0 / k, color="tab:red", linestyle=":", alpha=0.6)  # type: ignore[attr-defined]
        ax.annotate(  # type: ignore[attr-defined]
            label,
            xy=(1.0 / k, 1.0),
            xytext=(0.0, -8),
            textcoords="offset points",
            color="tab:red",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )


def _annotate_zone_layout_fig(fig: object, design: object, max_circles: int = 30) -> None:
    """Overlay Fresnel zone radii as dashed circles on a plot_zone_layout figure."""
    from fresnelants.core.geometry import fresnel_zone_radii
    from fresnelants.units import freq_to_wavelength

    lam = float(freq_to_wavelength(design.design_freq))  # type: ignore[attr-defined]
    if not hasattr(design, "num_zones"):
        return
    n_zones = int(design.num_zones)  # type: ignore[attr-defined]
    radii = fresnel_zone_radii(n_zones, design.focal_length, lam)  # type: ignore[attr-defined]
    # Down-sample if there are too many zones to keep the figure readable.
    step = max(1, n_zones // max_circles)
    for i in range(0, n_zones, step):
        r_mm = float(radii[i]) * 1e3
        for ax in fig.axes:  # type: ignore[attr-defined]
            ax.add_patch(
                Circle(
                    (0, 0),
                    r_mm,
                    fill=False,
                    edgecolor="tab:cyan",
                    linestyle=":",
                    linewidth=0.4,
                    alpha=0.7,
                )
            )
    # Outer-zone label, in the corner of each subplot.
    for ax in fig.axes:  # type: ignore[attr-defined]
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.text(
                xlim[1] * 0.95,
                ylim[1] * 0.92,
                f"n_max = {n_zones}",
                ha="right",
                va="top",
                fontsize=8,
                color="tab:cyan",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "tab:cyan", "alpha": 0.85},
            )
        except Exception:
            pass


def _conformal_pattern_panel(
    fig: object,
    lens: object,
    freq: float,
    position: tuple[int, int, int],
    *,
    broadside_label: str = "main beam",
) -> tuple[float, float]:
    """Mesh + far-field 2-axis panel with feed marker + broadside arrow.

    Returns (peak_dbi, peak_off_axis_norm) so the caller can compose titles.
    """
    from fresnelants.analysis.conformal_farfield import far_field_from_conformal

    ap = lens.conformal_aperture(freq)  # type: ignore[attr-defined]
    ff = far_field_from_conformal(ap, n_samples=64, chunk=4096)
    pts = ap.points
    mag = np.abs(ap.Et)
    n_rows, n_cols, idx_3d = position

    ax = fig.add_subplot(n_rows, n_cols, idx_3d, projection="3d")  # type: ignore[attr-defined]
    sc = ax.scatter(
        pts[:, 0] * 1e3,
        pts[:, 1] * 1e3,
        pts[:, 2] * 1e3,
        c=mag,
        cmap="magma",
        s=4,
    )
    # Feed marker at the origin.
    ax.scatter(
        [0],
        [0],
        [0],
        c="red",
        s=80,
        marker="*",
        label="feed",
        edgecolors="white",
        linewidths=0.8,
        zorder=10,
    )
    # Broadside arrow along +z.
    z_max = float(np.max(pts[:, 2]) * 1e3)
    ax.quiver(
        0,
        0,
        z_max * 0.55,
        0,
        0,
        z_max * 0.55,
        color="lime",
        arrow_length_ratio=0.25,
        linewidth=2.0,
        zorder=11,
    )
    ax.text(
        0,
        0,
        z_max * 1.18,
        broadside_label,
        color="forestgreen",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(f"{lens.name} mesh @ {freq / 1e9:.0f} GHz")  # type: ignore[attr-defined]
    fig.colorbar(sc, ax=ax, shrink=0.6, label="|Et|")  # type: ignore[attr-defined]

    ax2 = fig.add_subplot(n_rows, n_cols, idx_3d + 1)  # type: ignore[attr-defined]
    d = ff.directivity()
    d_db = 10 * np.log10(np.maximum(d, 1e-30))
    im = ax2.imshow(
        d_db,
        extent=(ff.u[0], ff.u[-1], ff.v[0], ff.v[-1]),
        origin="lower",
        cmap="inferno",
        vmin=d_db.max() - 30,
        vmax=d_db.max(),
    )
    # Unit circle (visible-region boundary).
    ax2.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="white", linewidth=0.6, linestyle="--"))
    # Mark the on-axis (broadside) point and a θ=45° reference circle.
    ax2.plot(0, 0, "+", color="lime", markersize=12, markeredgewidth=2)
    ax2.add_patch(
        Circle(
            (0, 0),
            np.sin(np.deg2rad(45.0)),
            fill=False,
            edgecolor="white",
            linewidth=0.4,
            linestyle=":",
        )
    )
    ax2.text(
        0,
        np.sin(np.deg2rad(45.0)) + 0.03,
        "θ=45°",
        color="white",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax2.set_title(f"Far-field — peak {d_db.max():.1f} dBi")
    ax2.set_xlabel(r"$u = \sin\theta\cos\phi$")
    ax2.set_ylabel(r"$v = \sin\theta\sin\phi$")
    ax2.set_aspect("equal")
    fig.colorbar(im, ax=ax2, shrink=0.85, label="dBi")  # type: ignore[attr-defined]

    iy, ix = np.unravel_index(int(np.argmax(d_db)), d_db.shape)
    return float(d_db.max()), float(np.hypot(ff.u[ix], ff.v[iy]))


def _fractal_gallery(out_dir: Path) -> None:
    """Fractal Fresnel zone plate family — Cantor + Sierpinski + 3D variants."""
    from fresnelants.analysis.nearfield import focal_axis_intensity
    from fresnelants.core.geometry import make_aperture_grid
    from fresnelants.units import freq_to_wavelength as _f2w

    # 2D Cantor zone plates (binary + Devil's-lens phase). Use base_unit=2 so
    # the Wood phase reversal materially differs from the Soret binary mask.
    F, freq = 0.30, 30e9
    soret = fa.FractalSoretZonePlate(focal_length=F, design_freq=freq, stage=2, base_unit=2)
    wood = fa.FractalWoodZonePlate(focal_length=F, design_freq=freq, stage=2, base_unit=2)
    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=4)
    res_w = solver.solve(wood, freq)

    fig_s = plot_zone_layout(soret)
    _annotate_zone_layout_fig(fig_s, soret)
    _save(fig_s, out_dir / "fractal_cantor_soret_layout.png")
    fig_w = plot_zone_layout(wood)
    _annotate_zone_layout_fig(fig_w, wood)
    _save(fig_w, out_dir / "fractal_cantor_wood_layout.png")
    _save(plot_farfield_2d(res_w.far_field), out_dir / "fractal_cantor_farfield.png")
    _save(
        plot_principal_cuts(res_w.far_field, label="Cantor Wood stage 2 (base 2), 30 GHz"),
        out_dir / "fractal_cantor_cuts.png",
    )

    # Headline polyfocal-signature plot — labeled F, F/3, F/5 markers.
    F2, freq2 = 0.30, 30e9
    cantor = fa.FractalSoretZonePlate(focal_length=F2, design_freq=freq2, stage=3, base_unit=1)
    classical = fa.WoodZonePlate(focal_length=F2, design_freq=freq2, num_zones=27)
    ap_cantor = cantor.aperture_field(freq2, samples_per_wavelength=4.0, margin=1.1)
    ap_classical = classical.aperture_field(freq2, samples_per_wavelength=4.0, margin=1.1)
    z_grid = np.linspace(0.05 * F2, 1.3 * F2, 120)
    I_cantor = focal_axis_intensity(ap_cantor, z_grid)
    I_classical = focal_axis_intensity(ap_classical, z_grid)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(
        z_grid / F2,
        I_cantor / I_cantor.max(),
        color="tab:purple",
        linewidth=1.4,
        label=f"Cantor stage 3 ({cantor.num_zones} zones)",
    )
    ax.plot(
        z_grid / F2,
        I_classical / I_classical.max(),
        color="tab:gray",
        linestyle="--",
        linewidth=1.0,
        label=f"Classical Wood ({classical.num_zones} zones)",
    )
    _annotate_axial_intensity(ax, F2)
    ax.axhline(0.5, color="tab:gray", linestyle=":", alpha=0.4)
    ax.text(1.25, 0.51, "−3 dB", color="tab:gray", fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("z / F")
    ax.set_ylabel("|E(0,0,z)|²  (normalized)")
    ax.set_title("Polyfocal signature of the Cantor zone plate")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    _save(fig, out_dir / "fractal_cantor_axial_intensity.png")

    # Sierpinski carpet (Cartesian fractal mask).
    carpet = fa.SierpinskiCarpetZonePlate(
        focal_length=0.5, design_freq=30e9, stage=3, aperture_side=0.18
    )
    res_c = solver.solve(carpet, 30e9)
    _save(plot_zone_layout(carpet), out_dir / "fractal_sierpinski_layout.png")
    _save(plot_farfield_2d(res_c.far_field), out_dir / "fractal_sierpinski_farfield.png")

    # Sierpinski reflectarray.
    sra = fa.SierpinskiReflectarray(
        focal_length=0.20, design_freq=28e9, nx=27, ny=27, fractal_stage=2
    )
    res_sra = solver.solve(sra, 28e9)
    _save(plot_zone_layout(sra), out_dir / "fractal_sierpinski_reflectarray_layout.png")
    _save(
        plot_farfield_2d(res_sra.far_field),
        out_dir / "fractal_sierpinski_reflectarray_farfield.png",
    )

    # 3D conformal fractal lenses (use updated nu=128 default for clean Nyquist).
    sph_lens = fa.SphericalFractalFresnelLens(
        radius=0.05,
        cap_angle_deg=90.0,
        design_freq=77e9,
        stage=2,
    )
    cone_lens = fa.ConicalFractalFresnelLens(
        half_angle_deg=45.0,
        height=0.05,
        design_freq=77e9,
        stage=2,
    )
    fig = plt.figure(figsize=(11, 4.6))
    _conformal_pattern_panel(fig, sph_lens, 77e9, (1, 2, 1), broadside_label="main beam ↑")
    fig.tight_layout()
    _save(fig, out_dir / "fractal_3d_spherical.png")

    fig = plt.figure(figsize=(11, 4.6))
    _conformal_pattern_panel(fig, cone_lens, 77e9, (1, 2, 1), broadside_label="cone axis ↑")
    fig.tight_layout()
    _save(fig, out_dir / "fractal_3d_conical.png")

    # NEW: Cantor stage-evolution figure.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for col, stage in enumerate((1, 2, 3)):
        zp = fa.FractalSoretZonePlate(focal_length=F2, design_freq=freq2, stage=stage)
        # Mask panel.
        ext = 2.0 * zp.aperture_radius * 1.05
        lam = float(_f2w(freq2))
        grid_s = make_aperture_grid(ext, max(2.0, 500 * lam / ext), lam)
        T = zp.transmittance(grid_s, freq2)
        ax_m = axes[0, col]
        ax_m.imshow(
            np.abs(T),
            extent=(grid_s.x[0] * 1e3, grid_s.x[-1] * 1e3, grid_s.y[0] * 1e3, grid_s.y[-1] * 1e3),
            origin="lower",
            cmap="gray_r",
        )
        retained = 2**stage
        total = 3**stage
        ax_m.set_title(f"Stage {stage}  ({retained} retained / {total} zones)")
        ax_m.set_xlabel("x [mm]")
        ax_m.set_ylabel("y [mm]")
        # Axial-intensity panel below.
        ap_s = zp.aperture_field(freq2, samples_per_wavelength=4.0, margin=1.1)
        z_s = np.linspace(0.05 * F2, 1.3 * F2, 100)
        I_s = focal_axis_intensity(ap_s, z_s)
        ax_a = axes[1, col]
        ax_a.plot(z_s / F2, I_s / I_s.max(), color="tab:purple", linewidth=1.2)
        _annotate_axial_intensity(ax_a, F2)
        ax_a.set_xlabel("z / F")
        ax_a.set_ylabel("|E|²  (norm.)")
        ax_a.set_xlim(0, 1.3)
        ax_a.set_ylim(0, 1.08)
    fig.suptitle("Cantor zone plate — stage evolution and polyfocal structure")
    fig.tight_layout()
    _save(fig, out_dir / "fractal_cantor_stages.png")

    # NEW: base_unit comparison figure (4 panels: Soret/Wood × base 1/2).
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, base_unit in enumerate((1, 2)):
        s_zp = fa.FractalSoretZonePlate(
            focal_length=F2, design_freq=freq2, stage=2, base_unit=base_unit
        )
        w_zp = fa.FractalWoodZonePlate(
            focal_length=F2, design_freq=freq2, stage=2, base_unit=base_unit
        )
        for row, (name, zp) in enumerate((("Soret", s_zp), ("Wood", w_zp))):
            ext = 2.0 * zp.aperture_radius * 1.05
            lam = float(_f2w(freq2))
            grid_s = make_aperture_grid(ext, max(2.0, 400 * lam / ext), lam)
            T = zp.transmittance(grid_s, freq2)
            ax = axes[row, col]
            cmap = "gray_r" if name == "Soret" else "RdBu"
            vmin, vmax = (0.0, 1.0) if name == "Soret" else (-1.0, 1.0)
            ax.imshow(
                np.real(T),
                extent=(
                    grid_s.x[0] * 1e3,
                    grid_s.x[-1] * 1e3,
                    grid_s.y[0] * 1e3,
                    grid_s.y[-1] * 1e3,
                ),
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            d_db = solver.solve(zp, freq2).far_field.peak_directivity_dbi()
            ax.set_title(f"{name}  base_unit={base_unit}  →  D = {d_db:.1f} dBi")
            ax.set_xlabel("x [mm]")
            if col == 0:
                ax.set_ylabel("y [mm]")
    fig.suptitle(
        "Cantor base_unit comparison (stage 2)\n"
        "base_unit=1 → all retained zones odd → Soret ≡ Wood   |   "
        "base_unit=2 → Wood phase reversal recaptures even-zone energy"
    )
    fig.tight_layout()
    _save(fig, out_dir / "fractal_cantor_baseunit_comparison.png")

    # Hero composite: Cantor mask + polyfocal trace + spherical 3D pattern.
    fig = plt.figure(figsize=(13, 4.4))
    # 1) Cantor mask
    ax1 = fig.add_subplot(1, 3, 1)
    grid_extent = 2.0 * cantor.aperture_radius * 1.05
    lam = float(_f2w(freq2))
    grid = make_aperture_grid(grid_extent, max(2.0, 600 * lam / grid_extent), lam)
    T = cantor.transmittance(grid, freq2)
    ax1.imshow(
        np.abs(T),
        extent=(grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3),
        origin="lower",
        cmap="gray_r",
    )
    ax1.set_title(f"Cantor mask, stage 3 ({cantor.num_zones} zones)")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    # 2) Polyfocal axial trace
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(z_grid / F2, I_cantor / I_cantor.max(), color="tab:purple", linewidth=1.4)
    _annotate_axial_intensity(ax2, F2)
    ax2.set_xlabel("z / F")
    ax2.set_ylabel("|E(0,0,z)|²  (normalized)")
    ax2.set_title("Polyfocal axial signature")
    ax2.set_xlim(0, 1.3)
    ax2.set_ylim(0, 1.08)
    ax2.text(
        0.65,
        0.88,
        "Cantor zone plates focus\nat F, F/3, F/5, …",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="tab:purple",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "tab:purple", "alpha": 0.85},
    )
    # 3) Spherical 3D pattern
    sph = fa.SphericalFractalFresnelLens(
        radius=0.05,
        cap_angle_deg=90.0,
        design_freq=77e9,
        stage=2,
    )
    from fresnelants.analysis.conformal_farfield import far_field_from_conformal as _ffc

    ap_sph = sph.conformal_aperture(77e9)
    ff_sph = _ffc(ap_sph, n_samples=64, chunk=4096)
    d = ff_sph.directivity()
    d_db = 10 * np.log10(np.maximum(d, 1e-30))
    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.imshow(
        d_db,
        extent=(ff_sph.u[0], ff_sph.u[-1], ff_sph.v[0], ff_sph.v[-1]),
        origin="lower",
        cmap="inferno",
        vmin=d_db.max() - 30,
        vmax=d_db.max(),
    )
    ax3.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="white", linewidth=0.6, linestyle="--"))
    ax3.plot(0, 0, "+", color="lime", markersize=12, markeredgewidth=2)
    ax3.set_title(f"Spherical fractal lens — peak {d_db.max():.1f} dBi")
    ax3.set_xlabel(r"$u$")
    ax3.set_ylabel(r"$v$")
    ax3.set_aspect("equal")
    fig.colorbar(im, ax=ax3, shrink=0.85, label="dBi")
    fig.suptitle("FresnelAnts — fractal Fresnel zone plate family")
    fig.tight_layout()
    _save(fig, out_dir / "fractal_hero.png")


def _macro_array_gallery(out_dir: Path) -> None:
    """Macro arrays of Fresnel-antenna elements (phased-array receivers)."""
    from fresnelants.analysis.array_factor import (
        array_factor as _af,
    )

    solver = fa.PhysicalOpticsSolver(samples_per_wavelength=4.0, pad_factor=4)

    # ----- 4-element linear (Wood ZP @ 10 GHz, spacing 1.2 m) -----
    elem4 = fa.WoodZonePlate(focal_length=1.0, design_freq=10e9, num_zones=8)
    arr4 = fa.MacroFresnelArray.from_lattice(elem4, n_elements=4, spacing_m=1.2, lattice="linear")

    # Layout: top-down circles to scale.
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for x, y in arr4.element_positions:
        ax.add_patch(
            Circle(
                (x, y),
                elem4.aperture_radius,
                fill=True,
                facecolor="tab:blue",
                edgecolor="navy",
                alpha=0.5,
            )
        )
    ax.set_aspect("equal")
    extent = arr4.array_extent
    ax.set_xlim(-extent[0] / 2 * 1.1, extent[0] / 2 * 1.1)
    ax.set_ylim(-extent[1] / 2 * 1.5, extent[1] / 2 * 1.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"4× Wood-zone-plate linear macro array @ 10 GHz "
        f"(spacing 1.2 m, footprint {extent[0]:.1f}×{extent[1]:.1f} m)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "macro_array_4x_linear_layout.png")

    # Steering cuts (within element beam). Plot principal-cuts for 4 small
    # angles since the element pattern is quite narrow.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for theta_b in (0, 1, 2, 3):
        w = arr4.weights_for_beam(theta_b, 0.0)
        res = arr4.solve(solver, 10e9, weights=w)
        td, db = res.far_field.cut("E")
        valid = ~np.isnan(td)
        ax.plot(td[valid], db[valid], label=f"θ_b = {theta_b}°")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-30, 2)
    ax.set_xlabel("θ [deg]")
    ax.set_ylabel("Normalized gain [dB]")
    ax.set_title("4× linear macro array — beam steering within element pattern")
    ax.legend(loc="lower center", ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "macro_array_4x_steering_cuts.png")

    # ----- 16-element rectangular (8×8 reflectarray @ 28 GHz) -----
    elem16 = fa.Reflectarray(focal_length=0.20, design_freq=28e9, nx=8, ny=8)
    arr16 = fa.MacroFresnelArray.from_lattice(
        elem16, n_elements=16, spacing_m=0.05, lattice="rect", rows=4
    )
    res16 = arr16.solve(solver, 28e9)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # Layout panel.
    for x, y in arr16.element_positions:
        axes[0].add_patch(
            Circle(
                (x * 1e3, y * 1e3),
                elem16.aperture_radius * 1e3,
                fill=True,
                facecolor="tab:orange",
                edgecolor="darkred",
                alpha=0.5,
            )
        )
    axes[0].set_aspect("equal")
    extent16 = arr16.array_extent
    axes[0].set_xlim(-extent16[0] / 2 * 1.2 * 1e3, extent16[0] / 2 * 1.2 * 1e3)
    axes[0].set_ylim(-extent16[1] / 2 * 1.2 * 1e3, extent16[1] / 2 * 1.2 * 1e3)
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("16× 8x8-reflectarray, 4×4 macro grid")
    axes[0].grid(True, alpha=0.3)
    # Far-field panel.
    d_db = 10 * np.log10(np.maximum(res16.far_field.directivity(), 1e-30))
    im = axes[1].imshow(
        d_db,
        extent=(
            res16.far_field.u[0],
            res16.far_field.u[-1],
            res16.far_field.v[0],
            res16.far_field.v[-1],
        ),
        origin="lower",
        cmap="inferno",
        vmin=d_db.max() - 30,
        vmax=d_db.max(),
    )
    axes[1].add_patch(
        Circle((0, 0), 1.0, fill=False, edgecolor="white", linewidth=0.6, linestyle="--")
    )
    # Annotate predicted grating-lobe positions for spacing 0.05 m at 28 GHz.
    wavelength16 = 3e8 / 28e9
    u_g = wavelength16 / 0.05
    for du in (-u_g, u_g):
        for dv in (-u_g, 0, u_g):
            if du == 0 and dv == 0:
                continue
            if du**2 + dv**2 <= 1.0:
                axes[1].plot(du, dv, "x", color="cyan", markersize=10, markeredgewidth=2)
    axes[1].set_xlabel(r"$u$")
    axes[1].set_ylabel(r"$v$")
    axes[1].set_title(f"Far-field — peak {d_db.max():.1f} dBi (× = grating lobes)")
    axes[1].set_aspect("equal")
    fig.colorbar(im, ax=axes[1], shrink=0.85, label="dBi")
    fig.tight_layout()
    _save(fig, out_dir / "macro_array_16x_rect_pattern.png")

    # 4-beam codebook overlay (E-plane).
    fig, ax = plt.subplots(figsize=(8, 4.6))
    book = arr16.beam_codebook(
        directions=[(-15.0, 0.0), (-5.0, 0.0), (5.0, 0.0), (15.0, 0.0)],
        labels=["beam_W", "beam_C-", "beam_C+", "beam_E"],
    )
    for label, w in book.items():
        res_b = arr16.solve(solver, 28e9, weights=w)
        td, db = res_b.far_field.cut("E")
        valid = ~np.isnan(td)
        ax.plot(td[valid], db[valid], label=label)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 2)
    ax.set_xlabel("θ [deg]")
    ax.set_ylabel("Normalized gain [dB]")
    ax.set_title("16× rectangular macro array — 4-beam receive codebook @ 28 GHz")
    ax.legend(loc="lower center", ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "macro_array_16x_codebook.png")

    # ----- 128-element hex (Soret @ 30 GHz, 1.5λ spacing) -----
    elem128 = fa.SoretZonePlate(focal_length=0.05, design_freq=30e9, num_zones=2)
    wavelength128 = 3e8 / 30e9
    arr128 = fa.MacroFresnelArray.from_lattice(
        elem128, n_elements=128, spacing_m=1.5 * wavelength128, lattice="hex"
    )
    res128 = arr128.solve(solver, 30e9)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for x, y in arr128.element_positions:
        ax.add_patch(
            Circle(
                (x * 1e3, y * 1e3),
                elem128.aperture_radius * 1e3,
                fill=True,
                facecolor="tab:purple",
                edgecolor="indigo",
                alpha=0.4,
            )
        )
    ax.set_aspect("equal")
    extent128 = arr128.array_extent
    ax.set_xlim(-extent128[0] / 2 * 1.1 * 1e3, extent128[0] / 2 * 1.1 * 1e3)
    ax.set_ylim(-extent128[1] / 2 * 1.1 * 1e3, extent128[1] / 2 * 1.1 * 1e3)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"128× Soret-ZP hex-close-packed array @ 30 GHz "
        f"(spacing 1.5λ, peak D = {res128.far_field.peak_directivity_dbi():.1f} dBi)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "macro_array_128x_hex_layout.png")

    # 128-element 2D pattern with predicted grating lobes overlaid.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    d_db = 10 * np.log10(np.maximum(res128.far_field.directivity(), 1e-30))
    im = ax.imshow(
        d_db,
        extent=(
            res128.far_field.u[0],
            res128.far_field.u[-1],
            res128.far_field.v[0],
            res128.far_field.v[-1],
        ),
        origin="lower",
        cmap="inferno",
        vmin=d_db.max() - 30,
        vmax=d_db.max(),
    )
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="white", linewidth=0.6, linestyle="--"))
    # Predicted grating lobes for hex lattice at d = 1.5λ.
    u_g = wavelength128 / arr128.min_neighbour_spacing
    for ang_deg in range(0, 360, 60):
        a = np.deg2rad(ang_deg)
        du, dv = u_g * np.cos(a), u_g * np.sin(a)
        if du**2 + dv**2 <= 1.0:
            ax.plot(du, dv, "x", color="cyan", markersize=10, markeredgewidth=2)
    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$v$")
    ax.set_title(f"128× hex pattern — peak {d_db.max():.1f} dBi (× = predicted grating lobes)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.85, label="dBi")
    fig.tight_layout()
    _save(fig, out_dir / "macro_array_128x_hex_pattern.png")

    # ----- Quantization scan-loss study -----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    elem_q = fa.SoretZonePlate(focal_length=0.05, design_freq=30e9, num_zones=2)
    arr_q = fa.MacroFresnelArray.from_lattice(
        elem_q, n_elements=8, spacing_m=0.6 * wavelength128, lattice="linear"
    )
    u = np.linspace(-1.0, 1.0, 401)
    v = np.array([0.0])
    theta_b = 30.0
    for bits, label in ((0, "continuous"), (4, "4-bit"), (2, "2-bit"), (1, "1-bit")):
        w = arr_q.weights_for_beam(theta_b, 0.0, bits=bits)
        af = np.abs(_af(u, v, 30e9, arr_q.element_positions, w)[0])
        peak = float(af.max())
        ax.plot(
            u,
            20 * np.log10(np.maximum(af / peak, 1e-3)),
            label=f"{label} (peak {20 * np.log10(peak):.1f} dB)",
        )
    ax.axvline(np.sin(np.deg2rad(theta_b)), color="tab:red", linestyle=":", alpha=0.5)
    ax.text(
        np.sin(np.deg2rad(theta_b)) + 0.01,
        -2,
        f"target u={np.sin(np.deg2rad(theta_b)):.2f}",
        color="tab:red",
        fontsize=8,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-40, 2)
    ax.set_xlabel(r"$u = \sin θ$")
    ax.set_ylabel(r"$|AF|$ [dB normalized to peak]")
    ax.set_title(f"8-element AF — quantization scan-loss at θ_b = {theta_b}°")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "macro_array_quantization.png")

    # ----- Mutual-coupling correction trend -----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    spacings_lambdas = np.linspace(0.6, 5.0, 30)
    for q_val, color in (
        (0.0, "tab:gray"),
        (0.3, "tab:blue"),
        (0.5, "tab:orange"),
        (0.8, "tab:red"),
    ):
        scales = []
        for s_lam in spacings_lambdas:
            arr_c = fa.MacroFresnelArray.from_lattice(
                elem_q,
                n_elements=4,
                spacing_m=s_lam * wavelength128,
                lattice="linear",
                coupling_q=q_val,
            )
            scales.append(20 * np.log10(arr_c._coupling_scale(30e9)))
        ax.plot(spacings_lambdas, scales, "o-", color=color, label=f"Q = {q_val}", markersize=4)
    ax.set_xlabel("element spacing [λ]")
    ax.set_ylabel("per-element pattern scale [dB]")
    ax.set_title("Mutual-coupling first-order correction trend")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "macro_array_coupling.png")


def main(out_dir: Path | str | None = None) -> None:
    target = Path(out_dir) if out_dir else ROOT / "docs" / "img"
    target.mkdir(parents=True, exist_ok=True)
    print(f"Rendering gallery into {target}…")
    _hero_panel(target)
    _zone_plate_gallery(target)
    _offset_gallery(target)
    _phase_correcting_gallery(target)
    _reflectarray_gallery(target)
    _curvilinear_gallery(target)
    _composite_gallery(target)
    _ris_gallery(target)
    _metasurface_gallery(target)
    _conformal_gallery(target)
    _timemod_gallery(target)
    _synth_gallery(target)
    _measured_gallery(target)
    _fractal_gallery(target)
    _macro_array_gallery(target)
    print(f"Done. {len(list(target.glob('*.png')))} PNGs written.")


if __name__ == "__main__":
    main()
