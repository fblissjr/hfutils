"""Convert local models between layouts.

Two sub-sub-commands for now:
- `comfyui`: pack into ComfyUI folders (diffusion_models, vae, text_encoders, ...).
- `single`:  merge shards (or copy a single file) into one .safetensors output.
"""

from pathlib import Path

import typer
from rich.console import Console

from hfutils.errors import InsufficientSpaceError, PlanError
from hfutils.events import RichObserver
from hfutils.formats.safetensors import (
    Manifest,
    manifest_from_shards,
    stream_merge,
    verify_output,
)
from hfutils.runner import PlanRunner
from hfutils.inspect.summary import format_summary_lines, summarize_component
from hfutils.io.fs import check_free_space
from hfutils.io.progress import COPY_CHUNK, make_progress
from hfutils.layouts.comfyui import ConvertTarget, PackOp, plan_comfyui, plan_single  # PackOp for typing
from hfutils.layouts.plan import PackPlan
from hfutils.sources.detect import detect_source
from hfutils.sources.types import (
    ComponentSource,
    PipelineSource,
    SafetensorsFileSource,
    Source,
    UnknownSource,
)

convert_app = typer.Typer(
    name="convert",
    help="Convert local models between layouts (ComfyUI, single-file, ...).",
    no_args_is_help=True,
)

console = Console()


def _warn(msg: str) -> None:
    console.print(f"[yellow]warn:[/yellow] {msg}")




def _op_total_bytes(op: PackOp) -> int:
    """Total size of the op's source shards via stat().

    Preflight uses this directly: a stat-based sum over-estimates merge output
    size by each shard's header overhead (tens of KB), which is a safe bias
    for an 'is there enough space' check. Progress bars use the same number,
    then `stream_merge.on_total` refines merge-op totals once headers parse.
    """
    return sum(s.stat().st_size for s in op.shards)


def _preflight_space(ops: list[PackOp]) -> None:
    """Refuse to start if any destination filesystem lacks space."""
    from collections import defaultdict
    by_root: dict[Path, int] = defaultdict(int)
    for op in ops:
        by_root[op.dest.parent] += _op_total_bytes(op)
    for root, required in by_root.items():
        try:
            check_free_space(root, required)
        except InsufficientSpaceError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)


def _run_plan(plan: "PackPlan") -> dict[Path, Manifest]:
    """Execute a plan under a RichObserver-driven Progress."""
    return PlanRunner(RichObserver(console)).run(plan)


def _verify_written(dest: Path, manifest: Manifest) -> bool:
    """Confirm the output at `dest` matches `manifest`. Print on outcome."""
    ok, error = verify_output(dest, manifest)
    if not ok:
        console.print(f"[red]Error:[/red] verify {dest.name}: {error}")
        return False
    console.print(f"[green]Verified[/green] {dest.name}: {len(manifest)} tensors match.")
    return True


def _print_op_preview(op: PackOp) -> None:
    console.print()
    console.print(f"[cyan bold]{op.label}[/cyan bold] -> {op.dest}")
    for line in format_summary_lines(summarize_component(op.source)):
        console.print(line)


def _print_plan(ops: list[PackOp], dry_run: bool) -> None:
    tag = "[yellow]DRY RUN[/yellow] " if dry_run else ""
    console.print(f"{tag}Plan: {len(ops)} operation(s)")
    for op in ops:
        console.print(f"  {op.label:>14s}  {op.kind:>6s}  -> {op.dest}")


def _default_name(source: Source) -> str:
    return source.path.stem if isinstance(source, SafetensorsFileSource) else source.path.name


@convert_app.command("comfyui")
def comfyui_cmd(
    source: Path = typer.Argument(..., help="Diffusers pipeline dir, component dir, or .safetensors file"),
    comfyui_root: Path = typer.Argument(..., help="ComfyUI models root (contains diffusion_models/, vae/, ...)"),
    name: str | None = typer.Option(None, "--name", help="Output base name (default: source basename)"),
    only: list[str] = typer.Option([], "--only", help="Pipeline component to include (repeatable)"),
    skip: list[str] = typer.Option([], "--skip", help="Pipeline component to exclude (repeatable)"),
    target: ConvertTarget | None = typer.Option(
        None, "--as",
        help="Destination type for non-pipeline sources.",
        case_sensitive=False,
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview plan and metadata, write nothing"),
    verify: bool = typer.Option(False, "--verify", help="Re-read each output and confirm tensor names/dtypes/shapes match the plan"),
) -> None:
    """Pack a local model into ComfyUI folder layout."""
    src = detect_source(source)
    if isinstance(src, UnknownSource):
        console.print(f"[red]Error:[/red] unrecognized source: {source}")
        raise typer.Exit(1)

    resolved_name = name or _default_name(src)
    target_value = target.value if target is not None else None

    try:
        plan = plan_comfyui(
            src, comfyui_root, resolved_name,
            only=only or None, skip=skip or None, target=target_value,
        )
    except PlanError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not plan:
        console.print("[yellow]Nothing to pack[/yellow] (no matching components found)")
        raise typer.Exit(1)

    for problem in plan.validate():
        _warn(problem)

    _print_plan(plan.ops, dry_run)
    for op in plan.ops:
        _print_op_preview(op)

    if dry_run:
        console.print("\n[yellow]Dry run complete[/yellow] -- no files written.")
        return

    _preflight_space(plan.ops)
    manifests = _run_plan(plan)

    if verify:
        if not all(_verify_written(op.dest, manifests[op.dest]) for op in plan.ops):
            raise typer.Exit(2)

    console.print(f"\n[green]Done.[/green] Wrote {len(plan)} file(s) under {comfyui_root}")


@convert_app.command("single")
def single_cmd(
    source: Path = typer.Argument(..., help="Sharded component dir, single .safetensors file, or diffusers pipeline (use --component)"),
    output: Path = typer.Argument(..., help="Output .safetensors file"),
    component: str | None = typer.Option(
        None, "--component",
        help="When source is a diffusers pipeline: pick one component (transformer, vae, text_encoder, ...)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview plan and metadata, write nothing"),
    verify: bool = typer.Option(False, "--verify", help="Re-read the output and confirm tensor names/dtypes/shapes match the plan"),
) -> None:
    """Merge shards (or copy a single file) into one .safetensors output."""
    src = detect_source(source)
    if isinstance(src, UnknownSource):
        console.print(f"[red]Error:[/red] unrecognized source: {source}")
        raise typer.Exit(1)

    if isinstance(src, PipelineSource):
        if component is None:
            available = ", ".join(src.components) or "(none detected)"
            console.print(
                f"[red]Error:[/red] source is a diffusers pipeline; "
                f"pass --component to pick one. Available: {available}"
            )
            raise typer.Exit(1)
        if component not in src.components:
            available = ", ".join(src.components) or "(none detected)"
            console.print(
                f"[red]Error:[/red] component '{component}' not found in pipeline. "
                f"Available: {available}"
            )
            raise typer.Exit(1)
        src = detect_source(source / component)
        if not isinstance(src, (ComponentSource, SafetensorsFileSource)):
            console.print(f"[red]Error:[/red] component '{component}' has no safetensors to merge.")
            raise typer.Exit(1)
    elif not isinstance(src, (ComponentSource, SafetensorsFileSource)):
        console.print(
            f"[red]Error:[/red] `convert single` expects a component directory, a "
            f"single .safetensors file, or a diffusers pipeline (with --component); "
            f"got {type(src).__name__}."
        )
        raise typer.Exit(1)
    elif component is not None:
        console.print("[red]Error:[/red] --component only applies when source is a diffusers pipeline.")
        raise typer.Exit(1)

    plan = plan_single(src, output)
    op = plan.ops[0]
    _print_plan([op], dry_run)
    _print_op_preview(op)

    if dry_run:
        console.print("\n[yellow]Dry run complete[/yellow] -- no files written.")
        return

    _preflight_space([op])
    manifests = _run_plan(plan)

    if verify and not _verify_written(op.dest, manifests[op.dest]):
        raise typer.Exit(2)

    console.print(f"\n[green]Done.[/green] Wrote {output}")
