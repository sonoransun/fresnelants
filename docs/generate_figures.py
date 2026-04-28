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
    print(f"Done. {len(list(target.glob('*.png')))} PNGs written.")


if __name__ == "__main__":
    main()
