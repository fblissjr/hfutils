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
from hfutils.cli_support import cli_guard, fail, verify_and_report
from hfutils.errors import PlanError
from hfutils.events import RichObserver
from hfutils.formats.safetensors import Manifest
from hfutils.inspect.summary import format_summary_lines, summarize_component
from hfutils.io.fs import check_free_space, resolve_existing_ancestor
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


def _preflight_space(ops: list[PackOp]) -> None:
    """Refuse to start if any destination filesystem lacks space.

    Groups by resolved filesystem device, not by dest.parent -- multiple
    destination folders (e.g. diffusion_models/, vae/, text_encoders/) often
    share one physical disk, so their requirements are summed and checked
    against one shared budget rather than each independently against the
    same undiminished free-space total.

    Raises InsufficientSpaceError straight through to cli_guard -- no local
    try/except needed."""
    by_device: dict[int, int] = defaultdict(int)
    device_probe: dict[int, Path] = {}
    device_dests: dict[int, list[Path]] = defaultdict(list)
    for op in ops:
        probe = resolve_existing_ancestor(op.dest.parent)
        if probe is None:
            continue  # pathological; let the OS error naturally at write time
        dev = probe.stat().st_dev
        by_device[dev] += op.total_bytes
        device_probe[dev] = probe
        device_dests[dev].append(op.dest.parent)
    for dev, required in by_device.items():
        label = ", ".join(str(p) for p in sorted(set(device_dests[dev]), key=str))
        check_free_space(device_probe[dev], required, label=label)


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


def _run_plan(plan: PackPlan, *, force: bool = False) -> dict[Path, Manifest]:
    return PlanRunner(RichObserver(console)).run(plan, force=force)


def _default_name(source: Source) -> str:
    return source.path.stem if isinstance(source, SafetensorsFileSource) else source.path.name


def _require_option(value, flag: str, layout: str) -> None:
    if value is None:
        fail(f"`--to {layout}` requires {flag}.")


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
    try:
        return plan_comfyui(
            src, root, resolved_name,  # type: ignore[arg-type]  # root is narrowed above
            only=only or None, skip=skip or None, target=target,
        )
    except PlanError as e:
        fail(str(e))


def _plan_single_from_cli(
    src: Source, source_path: Path,
    *,
    out: Path | None,
    name: str | None,
    component: list[str],
    all_components: bool,
    only: list[str],
    skip: list[str],
) -> PackPlan:
    _require_option(out, "--out", "single")
    assert out is not None

    if isinstance(src, PipelineSource):
        wants_all = all_components or ("all" in component)
        selected = [c for c in component if c != "all"]

        for c in selected:
            if c not in src.components:
                available = ", ".join(src.components) or "(none detected)"
                fail(
                    f"component '{c}' not found in pipeline. "
                    f"Available: {available}"
                )

        if not wants_all and not selected:
            available = ", ".join(src.components) or "(none detected)"
            fail(
                f"source is a diffusers pipeline; "
                f"pass --component to pick one (repeatable), or --all (with --out <dir>) "
                f"to convert all components. Available: {available}"
            )

        if not wants_all and len(selected) == 1 and (out.suffix == ".safetensors" and not out.is_dir()):
            comp = selected[0]
            sub_src = detect_source(source_path / comp)
            if not isinstance(sub_src, (ComponentSource, SafetensorsFileSource)):
                fail(f"component '{comp}' has no safetensors to merge.")
            return plan_single(sub_src, out)

        if out.suffix == ".safetensors" and not out.is_dir():
            fail(
                "converting multiple/all pipeline components with `--to single` requires "
                "`--out` to be an output directory, not a .safetensors file."
            )

        effective_only = selected or (only or None)
        resolved_name = name or _default_name(src)
        return plan_single(
            src, out,
            name=resolved_name,
            only=effective_only,
            skip=skip or None,
        )

    if not isinstance(src, (ComponentSource, SafetensorsFileSource)):
        fail(
            f"`--to single` expects a component directory, a "
            f"single .safetensors file, or a diffusers pipeline (with --component or --all); "
            f"got {type(src).__name__}."
        )
    if component or all_components:
        fail("--component / --all only apply when source is a diffusers pipeline.")

    return plan_single(src, out)


# --------------------------------------------------------------------------
# Unified convert command
# --------------------------------------------------------------------------


@app.command()
@cli_guard
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
    out: Path | None = typer.Option(None, "--out", help="Output .safetensors file or directory (required for --to single)"),
    component: list[str] = typer.Option(
        [], "--component",
        help="When source is a diffusers pipeline: pick component(s) for --to single (repeatable, or use --all).",
    ),
    all_components: bool = typer.Option(
        False, "--all",
        help="Convert all pipeline components (with --to single, writes into --out directory).",
    ),
    # Shared
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview plan and metadata, write nothing"),
    verify: bool = typer.Option(False, "--verify", help="Re-read each output and confirm it matches the plan"),
    force: bool = typer.Option(
        False, "--force",
        help="Write even if a source is incomplete/corrupt (still warns; refused by default).",
    ),
) -> None:
    """Convert a local model. `--to` picks the target layout."""
    src = detect_source(source)
    if isinstance(src, UnknownSource):
        fail(f"unrecognized source: {source}")

    if to == ConvertLayout.COMFYUI:
        plan = _plan_comfyui_from_cli(
            src, root=root, name=name, only=only, skip=skip, target=target,
        )
        done_message = f"Wrote {len(plan)} file(s) under {root}"
    else:  # ConvertLayout.SINGLE
        plan = _plan_single_from_cli(
            src, source,
            out=out, name=name, component=component,
            all_components=all_components, only=only, skip=skip,
        )
        done_message = (
            f"Wrote {len(plan)} file(s) under {out}"
            if len(plan) > 1 or (out is not None and out.is_dir())
            else f"Wrote {out}"
        )


    if not plan:
        fail("Nothing to pack (no matching components found)")

    for problem in plan.validate():
        _warn(problem)
    if plan.missing:
        _warn(f"missing entirely (no weight files found): {', '.join(plan.missing)}")
    for op in plan.broken_ops():
        _warn(f"{op.label}: {op.problem}")

    _print_plan(plan, dry_run)
    for op in plan.ops:
        _print_op_preview(op)

    if dry_run:
        console.print("\n[yellow]Dry run complete[/yellow] -- no files written.")
        return

    _preflight_space(plan.ops)
    manifests = _run_plan(plan, force=force)

    if verify:
        broken_by_dest = {op.dest: op for op in plan.broken_ops()}
        results = []
        for op in plan.ops:
            ok = verify_and_report(op.dest, manifests[op.dest])
            broken_source = broken_by_dest.get(op.dest)
            if ok and broken_source is not None:
                console.print(
                    f"[yellow]note:[/yellow] {op.dest.name} was written with --force despite "
                    f"{broken_source.problem} -- tensor-level verify only confirms the output "
                    f"matches what was actually written, not that the source was complete."
                )
            results.append(ok)
        if not all(results):
            raise typer.Exit(2)

    console.print(f"\n[green]Done.[/green] {done_message}")
