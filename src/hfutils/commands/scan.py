"""Scan local directories for model files."""

from dataclasses import dataclass
from pathlib import Path

import orjson
import typer
from rich.console import Console
from rich.table import Table

from hfutils.cli import app
from hfutils.inspect.common import format_size

console = Console()

MODEL_EXTENSIONS = {".safetensors", ".gguf", ".bin", ".pt", ".pth"}


@dataclass
class ModelEntry:
    name: str
    path: Path
    format: str
    total_size: int
    file_count: int
    sharded: bool
    incomplete: bool
    has_config: bool


def _detect_hf_cache_name(dir_path: Path) -> str | None:
    """If dir is inside an HF cache structure, return the repo name."""
    # Pattern: models--org--name/snapshots/hash/
    for parent in [dir_path] + list(dir_path.parents):
        if parent.name.startswith("models--"):
            parts = parent.name.removeprefix("models--").split("--", 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"
            return parts[0]
    return None


def _scan_model_dir(dir_path: Path, name: str) -> ModelEntry | None:
    """Analyze a single directory for model files."""
    model_files = []
    for f in dir_path.iterdir():
        if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS:
            model_files.append(f)

    if not model_files:
        return None

    # Determine primary format
    extensions = {f.suffix.lower() for f in model_files}
    if ".safetensors" in extensions:
        fmt = "safetensors"
    elif ".gguf" in extensions:
        fmt = "gguf"
    elif ".bin" in extensions or ".pt" in extensions or ".pth" in extensions:
        fmt = "pytorch"
    else:
        fmt = "unknown"

    # Check for sharding
    index_path = dir_path / "model.safetensors.index.json"
    sharded = index_path.exists()
    incomplete = False
    file_count = len(model_files)

    if sharded:
        index = orjson.loads(index_path.read_bytes())
        expected_shards = set(index.get("weight_map", {}).values())
        existing_files = {f.name for f in model_files}
        missing = expected_shards - existing_files
        incomplete = len(missing) > 0
        file_count = len([f for f in model_files if f.suffix.lower() == ".safetensors"])

    total_size = sum(f.stat().st_size for f in model_files)
    has_config = (dir_path / "config.json").exists()

    return ModelEntry(
        name=name,
        path=dir_path,
        format=fmt,
        total_size=total_size,
        file_count=file_count,
        sharded=sharded,
        incomplete=incomplete,
        has_config=has_config,
    )


def scan_directory(root: Path) -> list[ModelEntry]:
    """Scan a directory tree for model files. Returns list of discovered models."""
    root = Path(root)
    results = []

    # Check for HF cache layout: models--*/snapshots/*/
    hf_cache_dirs = list(root.glob("models--*/snapshots/*/"))
    if hf_cache_dirs:
        for snapshot_dir in hf_cache_dirs:
            cache_name = _detect_hf_cache_name(snapshot_dir)
            name = cache_name or snapshot_dir.parent.parent.name
            entry = _scan_model_dir(snapshot_dir, name)
            if entry:
                results.append(entry)

    # Check direct subdirectories (flat layout)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        # Skip HF cache dirs (already handled above)
        if child.name.startswith("models--"):
            continue
        entry = _scan_model_dir(child, child.name)
        if entry:
            results.append(entry)

    return results


@app.command()
def scan(
    directory: Path = typer.Argument(".", help="Directory to scan for model files"),
) -> None:
    """Audit local model storage: formats, sizes, completeness."""
    if not directory.is_dir():
        console.print(f"[red]Error:[/red] {directory} is not a directory")
        raise typer.Exit(1)

    results = scan_directory(directory)

    if not results:
        console.print("No models found.")
        return

    table = Table(title=f"Models in {directory}")
    table.add_column("Name", style="cyan")
    table.add_column("Format")
    table.add_column("Size", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Sharded")
    table.add_column("Status")

    total_size = 0
    for entry in results:
        total_size += entry.total_size
        status = ""
        if entry.incomplete:
            status = "[red]INCOMPLETE[/red]"
        elif not entry.has_config:
            status = "[yellow]no config[/yellow]"
        else:
            status = "[green]ok[/green]"

        table.add_row(
            entry.name,
            entry.format,
            format_size(entry.total_size),
            str(entry.file_count),
            "yes" if entry.sharded else "no",
            status,
        )

    console.print(table)
    console.print(f"\n  Total: {len(results)} models, {format_size(total_size)}")
