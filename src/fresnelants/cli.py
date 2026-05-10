"""FresnelAnts CLI — `fresnelants <command> ...`.

Subcommands:

* ``design <kind> ...``     – synthesize a design and write its spec.
* ``analyze <spec>``        – run the PO solver and report directivity / HPBW / SLL.
* ``export stl <spec> <out>``     – write an STL of the physical surface.
* ``export gerber <spec> <dir>``  – write Gerber/Excellon files (reflectarrays only).
* ``gallery``               – regenerate the documentation figure gallery.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .analysis.metrics import (
    aperture_efficiency,
    directivity_dbi,
    hpbw,
    sidelobe_level_db,
)
from .designs.base import AntennaDesign
from .export.gerber import write_reflectarray_gerber
from .export.stl import export_stl
from .io.loaders import load_design
from .io.specs import (
    ConicalFractalSpec,
    CurvilinearSpec,
    FractalSoretSpec,
    FractalWoodSpec,
    MacroArraySpec,
    OffsetSpec,
    PhaseCorrectingSpec,
    ReflectarraySpec,
    SierpinskiReflectarraySpec,
    SierpinskiSpec,
    SoretSpec,
    SphericalFractalSpec,
    WoodSpec,
)
from .solvers.physical_optics import PhysicalOpticsSolver

app = typer.Typer(
    name="fresnelants",
    help="Fresnel antenna design system: zone plates, offset Fresnel, phase-correcting, reflectarrays, curvilinear.",
    no_args_is_help=True,
    add_completion=False,
)
design_app = typer.Typer(help="Synthesize an antenna design and write its spec.")
export_app = typer.Typer(help="Manufacturing exports (STL, Gerber).")
app.add_typer(design_app, name="design")
app.add_typer(export_app, name="export")

console = Console()


def _write_spec(spec_obj: object, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = spec_obj.model_dump(mode="json")  # type: ignore[attr-defined]
    if out.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {out}")


@design_app.command("zone-plate")
def design_zone_plate(
    freq: Annotated[float, typer.Option(help="Design frequency [Hz].")],
    focal_length: Annotated[float, typer.Option("--focal-length", "-F", help="Focal length [m].")],
    zones: Annotated[int, typer.Option(help="Number of zones.")] = 12,
    kind: Annotated[
        str, typer.Option(help="'soret' (binary) or 'wood' (phase-reversal).")
    ] = "wood",
    out: Annotated[Path, typer.Option(help="Output JSON/YAML path.")] = Path("zone_plate.json"),
) -> None:
    """Synthesize a Soret or Wood zone plate."""
    if kind == "soret":
        spec = SoretSpec(focal_length=focal_length, design_freq=freq, num_zones=zones)
    elif kind == "wood":
        spec = WoodSpec(focal_length=focal_length, design_freq=freq, num_zones=zones)
    else:
        raise typer.BadParameter("kind must be 'soret' or 'wood'.")
    _write_spec(spec, out)


@design_app.command("offset")
def design_offset(
    freq: Annotated[float, typer.Option()],
    focal_length: Annotated[float, typer.Option("--focal-length", "-F", help="Focal length [m].")],
    radius: Annotated[float, typer.Option(help="Aperture radius [m].")] = 0.10,
    tilt_deg: Annotated[float, typer.Option(help="Off-axis focal tilt [deg].")] = 20.0,
    out: Annotated[Path, typer.Option()] = Path("offset.json"),
) -> None:
    """Synthesize an offset Fresnel zone plate."""
    spec = OffsetSpec(
        focal_length=focal_length, design_freq=freq, aperture_radius=radius, tilt_angle_deg=tilt_deg
    )
    _write_spec(spec, out)


@design_app.command("phase-correcting")
def design_phase_correcting(
    freq: Annotated[float, typer.Option()],
    focal_length: Annotated[float, typer.Option("--focal-length", "-F", help="Focal length [m].")],
    radius: Annotated[float, typer.Option()] = 0.10,
    levels: Annotated[
        int,
        typer.Option(
            help="Number of phase levels (1=Soret, 2=Wood, 4=quarter-wave, 1024=continuous)."
        ),
    ] = 4,
    dielectric: Annotated[str, typer.Option()] = "HDPE",
    out: Annotated[Path, typer.Option()] = Path("phase_correcting.json"),
) -> None:
    """Synthesize an N-level phase-correcting Fresnel plate."""
    spec = PhaseCorrectingSpec(
        focal_length=focal_length,
        design_freq=freq,
        aperture_radius=radius,
        levels=levels,
        dielectric=dielectric,
    )
    _write_spec(spec, out)


@design_app.command("reflectarray")
def design_reflectarray(
    freq: Annotated[float, typer.Option()],
    focal_length: Annotated[float, typer.Option("--focal-length", "-F", help="Feed distance [m].")],
    nx: Annotated[int, typer.Option()] = 32,
    ny: Annotated[int, typer.Option()] = 32,
    cell: Annotated[float, typer.Option(help="Cell size [m]; 0 = 0.5λ.")] = 0.0,
    beam_theta_deg: Annotated[float, typer.Option(help="Main beam theta [deg].")] = 0.0,
    beam_phi_deg: Annotated[float, typer.Option(help="Main beam phi [deg].")] = 0.0,
    out: Annotated[Path, typer.Option()] = Path("reflectarray.json"),
) -> None:
    """Synthesize a planar reflectarray."""
    spec = ReflectarraySpec(
        focal_length=focal_length,
        design_freq=freq,
        nx=nx,
        ny=ny,
        cell_size=cell,
        beam_theta_deg=beam_theta_deg,
        beam_phi_deg=beam_phi_deg,
    )
    _write_spec(spec, out)


@design_app.command("curvilinear")
def design_curvilinear(
    freq: Annotated[float, typer.Option()],
    focal_length: Annotated[float, typer.Option("--focal-length", "-F", help="Focal length [m].")],
    radius: Annotated[float, typer.Option()] = 0.10,
    profile: Annotated[str, typer.Option(help="'hyperbolic' | 'axicon'.")] = "hyperbolic",
    dielectric: Annotated[str, typer.Option()] = "HDPE",
    axicon_deg: Annotated[
        float, typer.Option(help="Axicon half-angle [deg] (only for profile='axicon').")
    ] = 5.0,
    out: Annotated[Path, typer.Option()] = Path("curvilinear.json"),
) -> None:
    """Synthesize a 3D curvilinear/constructed Fresnel surface."""
    spec = CurvilinearSpec(
        focal_length=focal_length,
        design_freq=freq,
        aperture_radius=radius,
        profile=profile,  # type: ignore[arg-type]
        dielectric=dielectric,
        axicon_angle_deg=axicon_deg,
    )
    _write_spec(spec, out)


@design_app.command("fractal")
def design_fractal(
    freq: Annotated[float, typer.Option(help="Design frequency [Hz].")],
    focal_length: Annotated[float, typer.Option("--focal-length", "-F", help="Focal length [m].")],
    stage: Annotated[int, typer.Option(help="Cantor / Sierpinski iteration depth.")] = 3,
    kind: Annotated[
        str,
        typer.Option(
            help="'soret' (Cantor binary) | 'wood' (Cantor phase / Devil's lens) | "
            "'sierpinski' (square carpet) | 'sierpinski_reflectarray'."
        ),
    ] = "wood",
    aperture_side: Annotated[
        float, typer.Option(help="Side length [m] for Sierpinski-carpet variant.")
    ] = 0.20,
    nx: Annotated[int, typer.Option(help="Cell count for Sierpinski reflectarray.")] = 27,
    ny: Annotated[int, typer.Option(help="Cell count for Sierpinski reflectarray.")] = 27,
    out: Annotated[Path, typer.Option(help="Output JSON/YAML path.")] = Path("fractal.json"),
) -> None:
    """Synthesize a 2D fractal Fresnel design (Cantor or Sierpinski)."""
    if kind == "soret":
        spec: object = FractalSoretSpec(
            focal_length=focal_length, design_freq=freq, stage=stage, base_unit=1
        )
    elif kind == "wood":
        spec = FractalWoodSpec(
            focal_length=focal_length, design_freq=freq, stage=stage, base_unit=1
        )
    elif kind == "sierpinski":
        spec = SierpinskiSpec(
            focal_length=focal_length,
            design_freq=freq,
            stage=stage,
            aperture_side=aperture_side,
        )
    elif kind == "sierpinski_reflectarray":
        spec = SierpinskiReflectarraySpec(
            focal_length=focal_length,
            design_freq=freq,
            nx=nx,
            ny=ny,
            cell_size=0.0,
            fractal_stage=stage,
        )
    else:
        raise typer.BadParameter(
            "kind must be 'soret' | 'wood' | 'sierpinski' | 'sierpinski_reflectarray'."
        )
    _write_spec(spec, out)


@design_app.command("fractal-3d")
def design_fractal_3d(
    freq: Annotated[float, typer.Option(help="Design frequency [Hz].")],
    substrate: Annotated[
        str, typer.Option(help="'spherical' (cap) | 'conical' (cone).")
    ] = "spherical",
    radius: Annotated[float, typer.Option(help="Sphere radius [m].")] = 0.05,
    cap_angle_deg: Annotated[float, typer.Option(help="Spherical cap half-angle [deg].")] = 90.0,
    half_angle_deg: Annotated[float, typer.Option(help="Cone half-angle from axis [deg].")] = 45.0,
    height: Annotated[float, typer.Option(help="Cone axial height [m].")] = 0.05,
    stage: Annotated[int, typer.Option(help="Cantor iteration depth.")] = 2,
    phase_reversal: Annotated[
        bool, typer.Option(help="True for Devil's lens phase, False for Soret-binary.")
    ] = True,
    out: Annotated[Path, typer.Option()] = Path("fractal_3d.json"),
) -> None:
    """Synthesize a 3D conformal fractal Fresnel lens (spherical or conical)."""
    if substrate == "spherical":
        spec: object = SphericalFractalSpec(
            design_freq=freq,
            radius=radius,
            cap_angle_deg=cap_angle_deg,
            stage=stage,
            base_unit=1,
            phase_reversal=phase_reversal,
            nu=128,
            nv=40,
        )
    elif substrate == "conical":
        spec = ConicalFractalSpec(
            design_freq=freq,
            half_angle_deg=half_angle_deg,
            height=height,
            stage=stage,
            base_unit=1,
            phase_reversal=phase_reversal,
            nu=128,
            nv=60,
        )
    else:
        raise typer.BadParameter("substrate must be 'spherical' or 'conical'.")
    _write_spec(spec, out)


@design_app.command("macro-array")
def design_macro_array(
    freq: Annotated[float, typer.Option(help="Design frequency [Hz].")],
    element_spec: Annotated[
        Path, typer.Option(help="Path to a YAML/JSON spec for the prototype element.")
    ],
    lattice: Annotated[str, typer.Option(help="'linear' | 'rect' | 'hex' | 'ring'.")] = "linear",
    n_elements: Annotated[int, typer.Option(help="Number of elements (4, 8, 16, 128, …).")] = 8,
    spacing: Annotated[
        float, typer.Option(help="Centre-to-centre spacing in wavelengths (default 1.5).")
    ] = 1.5,
    rows: Annotated[
        int | None, typer.Option(help="Row count for 'rect' lattice (auto if omitted).")
    ] = None,
    bits: Annotated[
        int, typer.Option(help="0=continuous; otherwise quantize weights to 2^bits phases.")
    ] = 0,
    coupling_q: Annotated[
        float, typer.Option(help="Mutual-coupling correction strength (0=disabled).")
    ] = 0.0,
    beam_theta_deg: Annotated[float, typer.Option(help="Receive beam θ_b [deg].")] = 0.0,
    beam_phi_deg: Annotated[float, typer.Option(help="Receive beam φ_b [deg].")] = 0.0,
    out: Annotated[Path, typer.Option(help="Output JSON/YAML path.")] = Path("macro_array.json"),
) -> None:
    """Synthesize a macro array of Fresnel-antenna elements (phased-array receiver)."""
    import json

    import yaml

    raw = element_spec.read_text(encoding="utf-8")
    element_dict = (
        yaml.safe_load(raw) if element_spec.suffix.lower() in {".yaml", ".yml"} else json.loads(raw)
    )
    wavelength = 3.0e8 / freq
    spec = MacroArraySpec(
        element=element_dict,
        lattice=lattice,  # type: ignore[arg-type]
        n_elements=n_elements,
        spacing_m=spacing * wavelength,
        rows=rows,
        bits=bits,
        coupling_q=coupling_q,
        beam_theta_deg=beam_theta_deg,
        beam_phi_deg=beam_phi_deg,
    )
    _write_spec(spec, out)


@app.command("analyze")
def analyze(
    spec: Annotated[Path, typer.Argument(help="Path to a design spec (YAML/JSON).")],
    freq: Annotated[float | None, typer.Option(help="Override frequency [Hz].")] = None,
    samples_per_wavelength: Annotated[float, typer.Option()] = 6.0,
) -> None:
    """Run the physical-optics solver and print performance metrics."""
    from .designs.macro_array import MacroFresnelArray

    design = load_design(spec)
    f = freq if freq is not None else design.design_freq
    solver = PhysicalOpticsSolver(samples_per_wavelength=samples_per_wavelength)
    if isinstance(design, MacroFresnelArray):
        result = design.solve(solver, f)
    else:
        result = solver.solve(design, f)
    ff = result.far_field

    physical_area = math.pi * design.aperture_radius**2
    table = Table(title=f"{design.name} @ {f / 1e9:.2f} GHz")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Aperture diameter [mm]", f"{2 * design.aperture_radius * 1e3:.1f}")
    table.add_row("Focal length [mm]", f"{design.focal_length * 1e3:.1f}")
    table.add_row("Peak directivity [dBi]", f"{directivity_dbi(ff):.2f}")
    table.add_row("HPBW E-plane [deg]", f"{hpbw(ff, 'E'):.2f}")
    table.add_row("HPBW H-plane [deg]", f"{hpbw(ff, 'H'):.2f}")
    table.add_row("First SLL [dB]", f"{sidelobe_level_db(ff, 'E'):.2f}")
    table.add_row("Aperture efficiency", f"{aperture_efficiency(ff, physical_area) * 100:.1f} %")
    console.print(table)


@export_app.command("stl")
def export_stl_cmd(
    spec: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Argument()],
    radial: Annotated[int, typer.Option()] = 200,
    angular: Annotated[int, typer.Option()] = 360,
) -> None:
    """Export a Fresnel surface to binary STL."""
    design = load_design(spec)
    path = export_stl(design, out, radial_samples=radial, angular_samples=angular)
    console.print(f"[green]Wrote[/green] {path}")


@export_app.command("gerber")
def export_gerber_cmd(
    spec: Annotated[Path, typer.Argument()],
    out_dir: Annotated[Path, typer.Argument()],
    base_name: Annotated[str, typer.Option()] = "reflectarray",
) -> None:
    """Export a reflectarray to Gerber + Excellon."""
    design: AntennaDesign = load_design(spec)
    from .designs.reflectarray import Reflectarray

    if not isinstance(design, Reflectarray):
        raise typer.BadParameter("Gerber export only supports reflectarray designs.")
    paths = write_reflectarray_gerber(design, out_dir, base_name=base_name)
    for label, p in paths.items():
        console.print(f"[green]Wrote[/green] {label}: {p}")


@app.command("gallery")
def gallery(
    out_dir: Annotated[Path, typer.Option()] = Path("docs/img"),
) -> None:
    """Regenerate the documentation figure gallery."""
    from importlib import import_module

    mod = import_module("docs.generate_figures")
    mod.main(out_dir=out_dir)  # type: ignore[attr-defined]
    console.print(f"[green]Gallery regenerated in[/green] {out_dir}")


if __name__ == "__main__":
    app()
