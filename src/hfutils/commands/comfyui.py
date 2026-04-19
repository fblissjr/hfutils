"""ComfyUI pack command: convert any local model layout into ComfyUI folders.

Handles three source shapes:
1. Diffusers pipeline (has model_index.json) -> auto-pack component subdirs
2. Component directory (has *.safetensors.index.json or single *.safetensors)
3. Single *.safetensors file

Writes merged/copied files into the standard ComfyUI subfolders:
  diffusion_models/, vae/, text_encoders/, checkpoints/, loras/.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from hfutils.commands.merge import consolidate_component, copy_with_progress
from hfutils.inspect.summary import format_summary_lines, summarize_component

console = Console()

comfyui_app = typer.Typer(
    name="comfyui",
    help="Convert local model files/directories into ComfyUI layout.",
    no_args_is_help=True,
)


# Diffusers component -> (ComfyUI folder, filename suffix).
# Suffix prevents collisions when components share --name.
DIFFUSERS_COMPONENTS: dict[str, tuple[str, str]] = {
    "transformer":    ("diffusion_models", ""),
    "vae":            ("vae",              "_vae"),
    "text_encoder":   ("text_encoders",    "_te"),
    "text_encoder_2": ("text_encoders",    "_te2"),
    "text_encoder_3": ("text_encoders",    "_te3"),
}

# User-supplied --as target -> ComfyUI subfolder
TARGET_FOLDERS: dict[str, str] = {
    "diffusion_model": "diffusion_models",
    "checkpoint":      "checkpoints",
    "vae":             "vae",
    "text_encoder":    "text_encoders",
    "clip":            "text_encoders",
    "lora":            "loras",
}


OpKind = Literal["copy", "consolidate"]


@dataclass
class PackOp:
    label: str             # "transformer", "vae", or "single" (for non-pipeline sources)
    source: Path
    dest: Path
    kind: OpKind           # "copy" = single file, "consolidate" = directory


@dataclass
class PackPlan:
    ops: list[PackOp] = field(default_factory=list)


def _is_diffusers_pipeline(src: Path) -> bool:
    return src.is_dir() and (src / "model_index.json").is_file()


def _component_has_safetensors(subdir: Path) -> bool:
    if not subdir.is_dir():
        return False
    return any(subdir.glob("*.safetensors"))


def _discover_pipeline_components(src: Path) -> list[str]:
    """Return component names present as subfolders with safetensors."""
    components = []
    for name in DIFFUSERS_COMPONENTS:
        if _component_has_safetensors(src / name):
            components.append(name)
    return components


def _dest_for_component(component: str, comfyui_root: Path, name: str) -> Path:
    folder, suffix = DIFFUSERS_COMPONENTS[component]
    return comfyui_root / folder / f"{name}{suffix}.safetensors"


def _dest_for_target(target: str, comfyui_root: Path, name: str) -> Path:
    if target not in TARGET_FOLDERS:
        valid = ", ".join(sorted(TARGET_FOLDERS))
        raise ValueError(f"Unknown --as target '{target}'. Valid: {valid}")
    return comfyui_root / TARGET_FOLDERS[target] / f"{name}.safetensors"


def plan_pack(
    source: Path,
    comfyui_root: Path,
    name: str,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    target: str | None = None,
) -> PackPlan:
    """Build a PackPlan describing what files to produce.

    No filesystem writes happen here. `target` is required for single-file
    and component-dir sources; ignored for diffusers pipelines.
    """
    plan = PackPlan()

    if _is_diffusers_pipeline(source):
        components = _discover_pipeline_components(source)
        if only:
            components = [c for c in components if c in only]
        if skip:
            components = [c for c in components if c not in skip]

        for component in components:
            plan.ops.append(PackOp(
                label=component,
                source=source / component,
                dest=_dest_for_component(component, comfyui_root, name),
                kind="consolidate",
            ))
        return plan

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    if target is None:
        valid = ", ".join(sorted(TARGET_FOLDERS))
        hint = "single file" if source.is_file() else "component directory (no model_index.json)"
        raise ValueError(f"Source is a {hint}; specify destination with --as ({valid})")

    plan.ops.append(PackOp(
        label="single",
        source=source,
        dest=_dest_for_target(target, comfyui_root, name),
        kind="copy" if source.is_file() else "consolidate",
    ))
    return plan


def _print_op_header(op: PackOp) -> None:
    console.print()
    console.print(f"[cyan bold]{op.label}[/cyan bold] -> {op.dest}")
    for line in format_summary_lines(summarize_component(op.source)):
        console.print(line)


def _execute_op(op: PackOp) -> None:
    op.dest.parent.mkdir(parents=True, exist_ok=True)
    if op.kind == "copy":
        copy_with_progress(op.source, op.dest)
    else:
        consolidate_component(op.source, op.dest)


def _print_plan(plan: PackPlan, dry_run: bool) -> None:
    tag = "[yellow]DRY RUN[/yellow] " if dry_run else ""
    console.print(f"{tag}Plan: {len(plan.ops)} operation(s)")
    for op in plan.ops:
        console.print(f"  {op.label:>14s}  {op.kind:>12s}  -> {op.dest}")


@comfyui_app.command()
def pack(
    source: Path = typer.Argument(..., help="Diffusers pipeline dir, component dir, or single .safetensors file"),
    comfyui_root: Path = typer.Argument(..., help="ComfyUI models root (contains diffusion_models/, vae/, ...)"),
    name: str | None = typer.Option(None, "--name", help="Output base name (default: source basename)"),
    only: str | None = typer.Option(None, "--only", help="Comma-separated components to include (pipeline only)"),
    skip: str | None = typer.Option(None, "--skip", help="Comma-separated components to exclude (pipeline only)"),
    target: str | None = typer.Option(None, "--as", help=f"Destination type for non-pipeline sources: {', '.join(sorted(TARGET_FOLDERS))}"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done, do not write files"),
) -> None:
    """Pack a local model into ComfyUI layout.

    Examples:
      hfutils comfyui pack <pipeline_dir> <comfyui_models>
      hfutils comfyui pack <pipeline_dir> <comfyui_models> --only transformer
      hfutils comfyui pack <model.safetensors> <comfyui_models> --as diffusion_model --name <base_name>
    """
    if not source.exists():
        console.print(f"[red]Error:[/red] source not found: {source}")
        raise typer.Exit(1)

    default_name = source.stem if source.is_file() else source.name
    resolved_name = name or default_name
    only_list = [x.strip() for x in only.split(",")] if only else None
    skip_list = [x.strip() for x in skip.split(",")] if skip else None

    try:
        plan = plan_pack(
            source=source,
            comfyui_root=comfyui_root,
            name=resolved_name,
            only=only_list,
            skip=skip_list,
            target=target,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not plan.ops:
        console.print("[yellow]Nothing to pack[/yellow] (no matching components found)")
        raise typer.Exit(1)

    _print_plan(plan, dry_run)
    for op in plan.ops:
        _print_op_header(op)
        if not dry_run:
            _execute_op(op)

    if dry_run:
        console.print("\n[yellow]Dry run complete[/yellow] -- no files written.")
        return
    console.print(f"\n[green]Done.[/green] Wrote {len(plan.ops)} file(s) under {comfyui_root}")
