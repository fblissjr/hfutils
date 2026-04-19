"""`hfutils convert` -- unified convert command.

One top-level command; `--to` picks the target layout (`comfyui` or `single`).
Per-layout options:
- `--to comfyui`:  `--root` required, plus `--name`, `--only`, `--skip`, `--as`
- `--to single`:   `--out`  required, plus `--component` (for pipeline sources)

Shared options: `--dry-run`, `--verify`.
"""

from collections import defaultdict
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from hfutils.cli import app
from hfutils.errors import InsufficientSpaceError, PlanError
from hfutils.events import RichObserver
from hfutils.formats.safetensors import Manifest, verify_output
from hfutils.inspect.summary import format_summary_lines, summarize_component
from hfutils.io.fs import check_free_space
from hfutils.layouts.comfyui import ConvertTarget, PackOp, plan_comfyui, plan_single
from hfutils.layouts.plan import PackPlan
from hfutils.runner import PlanRunner
from hfutils.sources.detect import detect_source
from hfutils.sources.types import (
    ComponentSource,
    PipelineSource,
    SafetensorsFileSource,
    Source,
    UnknownSource,
)

console = Console()


class ConvertLayout(str, Enum):
    """Values for --to: which target layout to pack into."""
    COMFYUI = "comfyui"
    SINGLE = "single"


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _warn(msg: str) -> None:
    console.print(f"[yellow]warn:[/yellow] {msg}")


def _op_total_bytes(op: PackOp) -> int:
    """Stat-based byte total per op. Safe bias for preflight."""
    return sum(s.stat().st_size for s in op.shards)


def _preflight_space(ops: list[PackOp]) -> None:
    """Refuse to start if any destination filesystem lacks space."""
    by_root: dict[Path, int] = defaultdict(int)
    for op in ops:
        by_root[op.dest.parent] += _op_total_bytes(op)
    for root, required in by_root.items():
        try:
            check_free_space(root, required)
        except InsufficientSpaceError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)


def _print_plan(plan: PackPlan, dry_run: bool) -> None:
    tag = "[yellow]DRY RUN[/yellow] " if dry_run else ""
    console.print(f"{tag}Plan: {len(plan)} operation(s)")
    for op in plan.ops:
        console.print(f"  {op.label:>14s}  {op.kind:>6s}  -> {op.dest}")


def _print_op_preview(op: PackOp) -> None:
    console.print()
    console.print(f"[cyan bold]{op.label}[/cyan bold] -> {op.dest}")
    for line in format_summary_lines(summarize_component(op.source)):
        console.print(line)


def _verify_written(dest: Path, manifest: Manifest) -> bool:
    ok, error = verify_output(dest, manifest)
    if not ok:
        console.print(f"[red]Error:[/red] verify {dest.name}: {error}")
        return False
    console.print(f"[green]Verified[/green] {dest.name}: {len(manifest)} tensors match.")
    return True


def _run_plan(plan: PackPlan) -> dict[Path, Manifest]:
    return PlanRunner(RichObserver(console)).run(plan)


def _default_name(source: Source) -> str:
    return source.path.stem if isinstance(source, SafetensorsFileSource) else source.path.name


def _require_option(value, flag: str, layout: str) -> None:
    if value is None:
        console.print(f"[red]Error:[/red] `--to {layout}` requires {flag}.")
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# Layout-specific plan builders
# --------------------------------------------------------------------------


def _plan_comfyui_from_cli(
    src: Source,
    *,
    root: Path | None,
    name: str | None,
    only: list[str],
    skip: list[str],
    target: ConvertTarget | None,
) -> PackPlan:
    _require_option(root, "--root", "comfyui")
    resolved_name = name or _default_name(src)
    target_value = target.value if target is not None else None
    try:
        return plan_comfyui(
            src, root, resolved_name,  # type: ignore[arg-type]  # root is narrowed above
            only=only or None, skip=skip or None, target=target_value,
        )
    except PlanError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _plan_single_from_cli(
    src: Source, source_path: Path,
    *,
    out: Path | None,
    component: str | None,
) -> PackPlan:
    _require_option(out, "--out", "single")

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
        src = detect_source(source_path / component)
        if not isinstance(src, (ComponentSource, SafetensorsFileSource)):
            console.print(f"[red]Error:[/red] component '{component}' has no safetensors to merge.")
            raise typer.Exit(1)
    elif not isinstance(src, (ComponentSource, SafetensorsFileSource)):
        console.print(
            f"[red]Error:[/red] `--to single` expects a component directory, a "
            f"single .safetensors file, or a diffusers pipeline (with --component); "
            f"got {type(src).__name__}."
        )
        raise typer.Exit(1)
    elif component is not None:
        console.print("[red]Error:[/red] --component only applies when source is a diffusers pipeline.")
        raise typer.Exit(1)

    return plan_single(src, out)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Unified convert command
# --------------------------------------------------------------------------


@app.command()
def convert(
    source: Path = typer.Argument(..., help="Diffusers pipeline, component dir, or .safetensors file"),
    to: ConvertLayout = typer.Option(
        ..., "--to",
        help="Destination layout.",
        case_sensitive=False,
    ),
    # --to comfyui options
    root: Path | None = typer.Option(None, "--root", help="ComfyUI models root (required for --to comfyui)"),
    name: str | None = typer.Option(None, "--name", help="Output base name (default: source basename)"),
    only: list[str] = typer.Option([], "--only", help="Pipeline component to include (repeatable)"),
    skip: list[str] = typer.Option([], "--skip", help="Pipeline component to exclude (repeatable)"),
    target: ConvertTarget | None = typer.Option(
        None, "--as",
        help="Destination type for non-pipeline comfyui packs.",
        case_sensitive=False,
    ),
    # --to single options
    out: Path | None = typer.Option(None, "--out", help="Output .safetensors file (required for --to single)"),
    component: str | None = typer.Option(
        None, "--component",
        help="When source is a diffusers pipeline: pick one component for --to single.",
    ),
    # Shared
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview plan and metadata, write nothing"),
    verify: bool = typer.Option(False, "--verify", help="Re-read each output and confirm it matches the plan"),
) -> None:
    """Convert a local model. `--to` picks the target layout."""
    src = detect_source(source)
    if isinstance(src, UnknownSource):
        console.print(f"[red]Error:[/red] unrecognized source: {source}")
        raise typer.Exit(1)

    if to == ConvertLayout.COMFYUI:
        plan = _plan_comfyui_from_cli(
            src, root=root, name=name, only=only, skip=skip, target=target,
        )
        done_message = f"Wrote {len(plan)} file(s) under {root}"
    else:  # ConvertLayout.SINGLE
        plan = _plan_single_from_cli(src, source, out=out, component=component)
        done_message = f"Wrote {out}"

    if not plan:
        console.print("[yellow]Nothing to pack[/yellow] (no matching components found)")
        raise typer.Exit(1)

    for problem in plan.validate():
        _warn(problem)

    _print_plan(plan, dry_run)
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

    console.print(f"\n[green]Done.[/green] {done_message}")
