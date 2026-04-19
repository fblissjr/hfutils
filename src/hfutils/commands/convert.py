"""Convert local models between layouts.

Two sub-sub-commands for now:
- `comfyui`: pack into ComfyUI folders (diffusion_models, vae, text_encoders, ...).
- `single`:  merge shards (or copy a single file) into one .safetensors output.
"""

from pathlib import Path

import typer
from rich.console import Console

from hfutils.formats.safetensors import read_raw_header, stream_merge
from hfutils.inspect.summary import format_summary_lines, summarize_component
from hfutils.io.fs import InsufficientSpaceError, check_free_space
from hfutils.io.progress import copy_with_progress, make_progress
from hfutils.layouts.comfyui import ConvertTarget, PackOp, plan_pack
from hfutils.sources.detect import Source, SourceKind, detect_source

convert_app = typer.Typer(
    name="convert",
    help="Convert local models between layouts (ComfyUI, single-file, ...).",
    no_args_is_help=True,
)

console = Console()


def _warn(msg: str) -> None:
    console.print(f"[yellow]warn:[/yellow] {msg}")


def _stream_merge_with_progress(shards: list[Path], dst: Path) -> None:
    with make_progress(console) as progress:
        task = progress.add_task(f"merge -> {dst.name}", total=None)
        stream_merge(
            shards, dst,
            on_total=lambda total: progress.update(task, total=total),
            on_progress=lambda n: progress.update(task, advance=n),
            on_warning=_warn,
        )


def _op_output_bytes(op: PackOp) -> int:
    """Sum of tensor-data bytes the op will write. Header reads only."""
    if op.kind == "copy":
        return op.shards[0].stat().st_size
    total = 0
    for shard in op.shards:
        h = read_raw_header(shard)
        for t in h.tensors:
            total += t.data_offset_end - t.data_offset_start
    return total


def _preflight_space(ops: list[PackOp]) -> None:
    """Refuse to start if any destination filesystem lacks space."""
    by_root: dict[Path, int] = {}
    for op in ops:
        root = op.dest.parent
        by_root[root] = by_root.get(root, 0) + _op_output_bytes(op)
    for root, required in by_root.items():
        try:
            check_free_space(root, required)
        except InsufficientSpaceError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)


def _execute_op(op: PackOp) -> None:
    op.dest.parent.mkdir(parents=True, exist_ok=True)
    if op.kind == "copy":
        copy_with_progress(op.shards[0], op.dest, console)
    else:
        _stream_merge_with_progress(op.shards, op.dest)


def _verify_output(op: PackOp) -> bool:
    """Re-read the output header and confirm it matches what the plan said.

    Returns True on success, False on mismatch (and prints the diagnostic).
    """
    expected_tensors: dict[str, tuple[str, list[int]]] = {}
    for shard in op.shards:
        h = read_raw_header(shard)
        for t in h.tensors:
            expected_tensors[t.name] = (t.dtype, t.shape)

    try:
        out = read_raw_header(op.dest)
    except (ValueError, OSError) as e:
        console.print(f"[red]Verify failed for {op.dest.name}:[/red] unreadable output ({e})")
        return False
    out_tensors = {t.name: (t.dtype, t.shape) for t in out.tensors}

    if set(out_tensors) != set(expected_tensors):
        missing = set(expected_tensors) - set(out_tensors)
        extra = set(out_tensors) - set(expected_tensors)
        msg = []
        if missing:
            msg.append(f"missing {len(missing)} tensor(s): {sorted(missing)[:3]}...")
        if extra:
            msg.append(f"unexpected {len(extra)} tensor(s): {sorted(extra)[:3]}...")
        console.print(f"[red]Verify failed for {op.dest.name}:[/red] " + "; ".join(msg))
        return False

    for name, (dtype, shape) in expected_tensors.items():
        if out_tensors[name] != (dtype, shape):
            console.print(
                f"[red]Verify failed for {op.dest.name}:[/red] "
                f"{name} dtype/shape mismatch "
                f"(expected {dtype} {shape}, got {out_tensors[name][0]} {out_tensors[name][1]})"
            )
            return False

    console.print(f"[green]Verified[/green] {op.dest.name}: {len(expected_tensors)} tensors match.")
    return True


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
    if src.kind == SourceKind.UNKNOWN:
        console.print(f"[red]Error:[/red] unrecognized source: {source}")
        raise typer.Exit(1)

    resolved_name = name or _default_name(src)
    target_value = target.value if target is not None else None

    try:
        ops = plan_pack(
            src, comfyui_root, resolved_name,
            only=only or None, skip=skip or None, target=target_value,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not ops:
        console.print("[yellow]Nothing to pack[/yellow] (no matching components found)")
        raise typer.Exit(1)

    _print_plan(ops, dry_run)
    if not dry_run:
        _preflight_space(ops)
    for op in ops:
        _print_op_preview(op)
        if not dry_run:
            _execute_op(op)

    if dry_run:
        console.print("\n[yellow]Dry run complete[/yellow] -- no files written.")
        return

    if verify:
        if not all(_verify_output(op) for op in ops):
            raise typer.Exit(2)

    console.print(f"\n[green]Done.[/green] Wrote {len(ops)} file(s) under {comfyui_root}")


def _single_as_packop(src: Source, output: Path) -> PackOp:
    """Represent `convert single` as a one-op plan so display stays uniform."""
    return PackOp(
        label="single",
        source=src.path,
        dest=output,
        shards=list(src.shards),
    )


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
    if src.kind == SourceKind.UNKNOWN:
        console.print(f"[red]Error:[/red] unrecognized source: {source}")
        raise typer.Exit(1)

    if src.kind == SourceKind.DIFFUSERS_PIPELINE:
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
        if src.kind not in (SourceKind.COMPONENT_DIR, SourceKind.SAFETENSORS_FILE):
            console.print(f"[red]Error:[/red] component '{component}' has no safetensors to merge.")
            raise typer.Exit(1)
    elif src.kind not in (SourceKind.COMPONENT_DIR, SourceKind.SAFETENSORS_FILE):
        console.print(
            f"[red]Error:[/red] `convert single` expects a component directory, a "
            f"single .safetensors file, or a diffusers pipeline (with --component); "
            f"got {src.kind.value}."
        )
        raise typer.Exit(1)
    elif component is not None:
        console.print("[red]Error:[/red] --component only applies when source is a diffusers pipeline.")
        raise typer.Exit(1)

    op = _single_as_packop(src, output)
    _print_plan([op], dry_run)
    _print_op_preview(op)

    if dry_run:
        console.print("\n[yellow]Dry run complete[/yellow] -- no files written.")
        return

    _preflight_space([op])
    _execute_op(op)

    if verify and not _verify_output(op):
        raise typer.Exit(2)

    console.print(f"\n[green]Done.[/green] Wrote {output}")
