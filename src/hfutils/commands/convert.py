"""Convert local models between layouts.

Two sub-sub-commands for now:
- `comfyui`: pack into ComfyUI folders (diffusion_models, vae, text_encoders, ...).
- `single`:  merge shards (or copy a single file) into one .safetensors output.
"""

from pathlib import Path

import typer
from rich.console import Console

from hfutils.formats.safetensors import stream_merge
from hfutils.inspect.summary import format_summary_lines, summarize_component
from hfutils.io.progress import copy_with_progress, make_progress
from hfutils.layouts.comfyui import TARGET_FOLDERS, PackOp, plan_pack
from hfutils.sources.detect import Source, SourceKind, detect_source

convert_app = typer.Typer(
    name="convert",
    help="Convert local models between layouts (ComfyUI, single-file, ...).",
    no_args_is_help=True,
)

console = Console()


def _stream_merge_with_progress(shards: list[Path], dst: Path) -> None:
    with make_progress(console) as progress:
        task = progress.add_task(f"merge -> {dst.name}", total=None)
        stream_merge(
            shards, dst,
            on_total=lambda total: progress.update(task, total=total),
            on_progress=lambda n: progress.update(task, advance=n),
        )


def _execute_op(op: PackOp) -> None:
    op.dest.parent.mkdir(parents=True, exist_ok=True)
    if op.kind == "copy":
        copy_with_progress(op.shards[0], op.dest, console)
    else:
        _stream_merge_with_progress(op.shards, op.dest)


def _print_op_preview(op: PackOp) -> None:
    console.print()
    console.print(f"[cyan bold]{op.label}[/cyan bold] -> {op.dest}")
    preview_path = op.source if op.source is not None else op.shards[0]
    for line in format_summary_lines(summarize_component(preview_path)):
        console.print(line)


def _print_plan(ops: list[PackOp], dry_run: bool) -> None:
    tag = "[yellow]DRY RUN[/yellow] " if dry_run else ""
    console.print(f"{tag}Plan: {len(ops)} operation(s)")
    for op in ops:
        console.print(f"  {op.label:>14s}  {op.kind:>6s}  -> {op.dest}")


def _default_name(source: Source) -> str:
    return source.path.stem if source.kind == SourceKind.SAFETENSORS_FILE else source.path.name


@convert_app.command("comfyui")
def comfyui_cmd(
    source: Path = typer.Argument(..., help="Diffusers pipeline dir, component dir, or .safetensors file"),
    comfyui_root: Path = typer.Argument(..., help="ComfyUI models root (contains diffusion_models/, vae/, ...)"),
    name: str | None = typer.Option(None, "--name", help="Output base name (default: source basename)"),
    only: str | None = typer.Option(None, "--only", help="Comma-separated components to include (pipelines only)"),
    skip: str | None = typer.Option(None, "--skip", help="Comma-separated components to exclude (pipelines only)"),
    target: str | None = typer.Option(None, "--as", help=f"Destination type for non-pipeline sources: {', '.join(sorted(TARGET_FOLDERS))}"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview plan and metadata, write nothing"),
) -> None:
    """Pack a local model into ComfyUI folder layout."""
    src = detect_source(source)
    if src.kind == SourceKind.UNKNOWN:
        console.print(f"[red]Error:[/red] unrecognized source: {source}")
        raise typer.Exit(1)

    resolved_name = name or _default_name(src)
    only_list = [x.strip() for x in only.split(",")] if only else None
    skip_list = [x.strip() for x in skip.split(",")] if skip else None

    try:
        ops = plan_pack(
            src, comfyui_root, resolved_name,
            only=only_list, skip=skip_list, target=target,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not ops:
        console.print("[yellow]Nothing to pack[/yellow] (no matching components found)")
        raise typer.Exit(1)

    _print_plan(ops, dry_run)
    for op in ops:
        _print_op_preview(op)
        if not dry_run:
            _execute_op(op)

    if dry_run:
        console.print("\n[yellow]Dry run complete[/yellow] -- no files written.")
        return
    console.print(f"\n[green]Done.[/green] Wrote {len(ops)} file(s) under {comfyui_root}")


@convert_app.command("single")
def single_cmd(
    source: Path = typer.Argument(..., help="Sharded component directory or single .safetensors file"),
    output: Path = typer.Argument(..., help="Output .safetensors file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview metadata, write nothing"),
) -> None:
    """Merge shards (or copy a single file) into one .safetensors output."""
    src = detect_source(source)
    if src.kind == SourceKind.UNKNOWN:
        console.print(f"[red]Error:[/red] unrecognized source: {source}")
        raise typer.Exit(1)
    if src.kind not in (SourceKind.COMPONENT_DIR, SourceKind.SAFETENSORS_FILE):
        console.print(
            f"[red]Error:[/red] `convert single` expects a component directory or "
            f"a single .safetensors file; got {src.kind.value}. "
            f"For a diffusers pipeline, use `convert comfyui` with --only."
        )
        raise typer.Exit(1)

    console.print(f"[cyan bold]{source.name}[/cyan bold] -> {output}")
    for line in format_summary_lines(summarize_component(source)):
        console.print(line)

    if dry_run:
        console.print("\n[yellow]Dry run complete[/yellow] -- no files written.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    if len(src.shards) == 1:
        copy_with_progress(src.shards[0], output, console)
    else:
        _stream_merge_with_progress(src.shards, output)
    console.print("[green]Done.[/green]")
