"""Merge sharded safetensors files into a single file."""

from pathlib import Path

import orjson
import typer
from rich.console import Console

from hfutils.cli import app

console = Console()


def merge_safetensors(input_dir: Path, output_path: Path) -> None:
    """Merge sharded safetensors into a single file.

    Reads model.safetensors.index.json, loads each shard, and writes all
    tensors to a single output file.

    Requires safetensors[torch] to be installed.
    """
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        console.print(
            "[red]Error:[/red] safetensors[torch] is required for merge. "
            "Install with: uv add 'safetensors[torch]'"
        )
        raise typer.Exit(1)

    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index = orjson.loads(index_path.read_bytes())
    weight_map = index["weight_map"]

    shard_files = list(dict.fromkeys(weight_map.values()))

    all_tensors = {}
    for shard_name in shard_files:
        shard_path = input_dir / shard_name
        console.print(f"Loading {shard_name}...")
        tensors = load_file(str(shard_path))
        all_tensors.update(tensors)

    console.print(f"Writing {len(all_tensors)} tensors to {output_path}...")
    save_file(all_tensors, str(output_path))
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
