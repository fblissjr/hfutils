"""`hfutils lora` -- LoRA key-scheme conversions.

`lora convert <src>` renames a LoRA's tensor keys so a target loader accepts
it (e.g. a fal.ai Krea-2 LoRA -> ComfyUI's default loader). `lora schemes`
lists the known conversions.

Distinct from `hfutils convert`, which changes on-disk *layout* (routing/merging
whole files into ComfyUI folders). This changes a file's *internal* tensor
names; the bytes are copied verbatim.
"""

from pathlib import Path

import typer
from rich.console import Console

from hfutils.cli_support import cli_guard, fail, verify_and_report
from hfutils.events import RichMergeObserver
from hfutils.formats.lora import (
    LORA_SCHEMES,
    LoraScheme,
    convert_lora_keys,
    detect_lora_scheme,
    fuse_lora,
)
from hfutils.inspect.safetensors import RawHeader, read_raw_header
from hfutils.io.fs import check_free_space
from hfutils.io.progress import make_progress

console = Console()

lora_app = typer.Typer(
    name="lora",
    help="LoRA key-scheme conversions (rename keys for a target loader).",
    no_args_is_help=True,
)


def _scheme_list() -> str:
    return ", ".join(LORA_SCHEMES) or "(none)"


def _resolve_scheme(header: RawHeader, scheme: str) -> LoraScheme:
    """Resolve a scheme name (or 'auto' to detect) to a LoraScheme, or exit."""
    if scheme == "auto":
        resolved = detect_lora_scheme(header)
        if resolved is None:
            fail(
                f"could not detect a known LoRA scheme. Known: {_scheme_list()}. "
                f"Pass --scheme <name> to force one."
            )
        return resolved
    resolved = LORA_SCHEMES.get(scheme)
    if resolved is None:
        fail(f"unknown scheme '{scheme}'. Known: {_scheme_list()}.")
    return resolved


@lora_app.command("schemes")
@cli_guard
def schemes() -> None:
    """List the available LoRA conversion schemes."""
    console.print("[bold]Available LoRA conversion schemes:[/bold]")
    for scheme in LORA_SCHEMES.values():
        console.print(f"  [cyan]{scheme.name}[/cyan]  {scheme.description}")


@lora_app.command("convert")
@cli_guard
def convert(
    source: Path = typer.Argument(..., help="LoRA .safetensors file to convert"),
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="Output path (default: <source>_comfy.safetensors next to the input).",
    ),
    scheme: str = typer.Option(
        "auto", "--scheme",
        help=f"Conversion scheme, or 'auto' to detect. Known: {', '.join(LORA_SCHEMES)}.",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Convert even if the file doesn't match the named scheme's source format.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report the plan, write nothing."),
    verify: bool = typer.Option(False, "--verify", help="Re-read the output and confirm it matches."),
) -> None:
    """Rename a LoRA's keys so a target loader accepts it."""
    if not source.is_file():
        fail(f"input not found: {source}")

    try:
        header = read_raw_header(source)
    except (ValueError, OSError) as e:
        fail(f"could not read safetensors header: {e}")

    resolved = _resolve_scheme(header, scheme)
    if scheme != "auto" and not resolved.matches(header) and not force:
        fail(
            f"{source.name} doesn't look like a '{resolved.name}' source LoRA. "
            f"Re-run with --force to convert anyway."
        )

    dest = out or source.with_name(source.stem + "_comfy.safetensors")

    console.print(f"Scheme: [cyan]{resolved.name}[/cyan] ({resolved.description})")
    console.print(f"  in:  {source}")
    console.print(f"  out: {dest}")
    console.print(f"  tensors: {len(header.tensors)}")

    if dry_run:
        console.print("\n[yellow]Dry run[/yellow] -- no file written.")
        return

    with make_progress(console) as progress:
        observer = RichMergeObserver(progress, f"convert -> {dest.name}", console)
        manifest = convert_lora_keys(source, dest, resolved, observer=observer)

    if verify:
        if not verify_and_report(dest, manifest):
            raise typer.Exit(2)

    console.print(
        f"\n[green]Done.[/green] Wrote {len(manifest)} tensors to {dest}. "
        f"Drop it into ComfyUI/models/loras/ and load with the default LoRA loader."
    )


@lora_app.command("fuse")
@cli_guard
def fuse(
    lora: Path = typer.Argument(..., help="LoRA .safetensors (source or converted naming)"),
    base: Path = typer.Argument(..., help="Full-precision (bf16) base checkpoint .safetensors"),
    out: Path = typer.Option(..., "--out", "-o", help="Output fused checkpoint path."),
    scheme: str = typer.Option(
        "auto", "--scheme",
        help=f"Conversion scheme, or 'auto' to detect. Known: {', '.join(LORA_SCHEMES)}.",
    ),
    strength: float = typer.Option(1.0, "--strength", help="LoRA strength multiplier."),
    alpha: float | None = typer.Option(
        None, "--alpha",
        help="Override LoRA alpha (default: from metadata, then per-module rank).",
    ),
) -> None:
    """Bake a LoRA into the base checkpoint (then quantize for zero runtime cost)."""
    for label, path in (("lora", lora), ("base", base)):
        if not path.is_file():
            fail(f"{label} not found: {path}")

    try:
        header = read_raw_header(lora)
    except (ValueError, OSError) as e:
        fail(f"could not read LoRA header: {e}")

    resolved = _resolve_scheme(header, scheme)
    if resolved.to_base_key is None:
        fail(f"scheme '{resolved.name}' does not support fuse.")

    console.print(f"Scheme: [cyan]{resolved.name}[/cyan]  strength={strength}")
    console.print(f"  lora: {lora}")
    console.print(f"  base: {base}")
    console.print(f"  out:  {out}")

    check_free_space(out.parent, base.stat().st_size)

    with make_progress(console) as progress:
        observer = RichMergeObserver(progress, f"fuse -> {out.name}", console)
        result = fuse_lora(lora, base, out, resolved, strength=strength, alpha=alpha, observer=observer)

    if result.unmatched:
        console.print(f"[yellow]{len(result.unmatched)} LoRA module(s) had no base match[/yellow] (skipped).")
    console.print(
        f"\n[green]Done.[/green] Fused {result.fused}/{result.lora_modules} LoRA modules "
        f"into {out}. Quantize it for zero LoRA runtime cost."
    )
