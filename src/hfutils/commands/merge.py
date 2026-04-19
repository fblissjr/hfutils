"""Merge sharded safetensors files into a single file."""

from pathlib import Path

import orjson
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from hfutils.cli import app

console = Console()

_COPY_CHUNK = 4 * 1024 * 1024  # 4 MiB


def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def copy_with_progress(src: Path, dst: Path) -> None:
    """Byte-accurate progress-tracked file copy. Public: used by `comfyui pack`."""
    total = src.stat().st_size
    with _progress() as progress:
        task = progress.add_task(f"copy {src.name}", total=total)
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            while True:
                chunk = fi.read(_COPY_CHUNK)
                if not chunk:
                    break
                fo.write(chunk)
                progress.update(task, advance=len(chunk))


def _find_index_file(input_dir: Path) -> Path:
    """Locate a `*.safetensors.index.json` file in input_dir.

    Diffusers repos use `diffusion_pytorch_model.safetensors.index.json`;
    transformers repos use `model.safetensors.index.json`. Accept any.
    """
    candidates = sorted(input_dir.glob("*.safetensors.index.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No *.safetensors.index.json file found in {input_dir}"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"Multiple index files found in {input_dir}: {names}. "
            "Run from a directory with a single sharded model."
        )
    return candidates[0]


def _require_safetensors_torch():
    try:
        from safetensors.torch import load_file, save_file
        return load_file, save_file
    except ImportError:
        console.print(
            "[red]Error:[/red] safetensors[torch] is required. "
            "Install with: uv add 'safetensors[torch]'"
        )
        raise typer.Exit(1)


def merge_safetensors(input_dir: Path, output_path: Path) -> None:
    """Merge sharded safetensors into a single file.

    Auto-discovers the `*.safetensors.index.json` file, loads each shard,
    and writes all tensors to a single output file.

    Requires safetensors[torch] to be installed.
    """
    load_file, save_file = _require_safetensors_torch()

    index_path = _find_index_file(input_dir)

    index = orjson.loads(index_path.read_bytes())
    weight_map = index["weight_map"]

    shard_files = list(dict.fromkeys(weight_map.values()))
    shard_paths = [input_dir / s for s in shard_files]
    shard_sizes = [p.stat().st_size for p in shard_paths]

    all_tensors: dict = {}
    with _progress() as progress:
        task = progress.add_task("loading shards", total=sum(shard_sizes))
        for path, size in zip(shard_paths, shard_sizes):
            progress.update(task, description=f"loading {path.name}")
            all_tensors.update(load_file(str(path)))
            progress.update(task, advance=size)

    console.print(f"Writing {len(all_tensors)} tensors to {output_path}...")
    save_file(all_tensors, str(output_path))
    console.print("Done.")


def consolidate_component(input_dir: Path, output_path: Path) -> None:
    """Produce a single safetensors file from a component directory.

    Handles two shapes:
    - Sharded: `*.safetensors.index.json` + multiple shards -> merge
    - Single file: exactly one `*.safetensors` (and no index) -> copy

    Raises FileNotFoundError if the directory has no safetensors, or more
    than one safetensors without a matching index.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    index_candidates = list(input_dir.glob("*.safetensors.index.json"))
    if index_candidates:
        merge_safetensors(input_dir, output_path)
        return

    safetensors_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files in {input_dir}")
    if len(safetensors_files) > 1:
        names = ", ".join(p.name for p in safetensors_files)
        raise FileNotFoundError(
            f"Multiple safetensors in {input_dir} without an index file: "
            f"{names}. Cannot determine how to combine them."
        )

    copy_with_progress(safetensors_files[0], output_path)
    console.print("Done.")


@app.command()
def merge(
    input_dir: Path = typer.Argument(..., help="Directory containing sharded safetensors and index JSON"),
    output: Path = typer.Argument(..., help="Output path for merged .safetensors file"),
) -> None:
    """Merge sharded safetensors into a single file."""
    if not input_dir.is_dir():
        console.print(f"[red]Error:[/red] {input_dir} is not a directory")
        raise typer.Exit(1)

    merge_safetensors(input_dir, output)
