"""Inspect model files, directories, pipelines, or trees."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hfutils.cli import app
from hfutils.inspect.architecture import detect_architecture
from hfutils.inspect.common import SafetensorsHeader, format_params, format_size
from hfutils.inspect.directory import inspect_directory
from hfutils.inspect.gguf import read_gguf_header
from hfutils.inspect.safetensors import read_header
from hfutils.sources.detect import Source, SourceKind, detect_source

console = Console()


def _format_arch(arch) -> str | None:
    parts = []
    if arch.family != "Unknown":
        parts.append(arch.family)
    if arch.adapter_type:
        parts.append(arch.adapter_type)
    return " / ".join(parts) if parts else None


def _display_safetensors(header: SafetensorsHeader, path: Path, detail: bool, arch=None) -> None:
    if arch is None:
        arch = detect_architecture(
            [t.name for t in header.tensors],
            metadata=header.metadata or None,
        )

    console.print(f"\n[bold]{path.name}[/bold]")

    arch_str = _format_arch(arch)
    if arch_str:
        console.print(f"  Architecture: {arch_str}")

    console.print(f"  Tensors: {len(header.tensors)}")
    console.print(f"  Parameters: {format_params(header.total_params)}")
    console.print(f"  Size: {format_size(header.total_size_bytes)}")

    breakdown = header.dtype_breakdown()
    if len(breakdown) > 1:
        console.print("\n  [bold]Dtype breakdown:[/bold]")
        for b in breakdown:
            console.print(
                f"    {b.dtype:<8} {format_params(b.param_count):>10} params  "
                f"{b.tensor_count:>4} tensors  {format_size(b.size_bytes):>10}"
            )
    elif breakdown:
        console.print(f"  Dtype: {breakdown[0].dtype}")

    if header.metadata:
        console.print("\n  [bold]Metadata:[/bold]")
        for k, v in header.metadata.items():
            display_v = v if len(v) <= 80 else v[:77] + "..."
            console.print(f"    {k}: {display_v}")

    if arch.training_metadata:
        console.print("\n  [bold]Training info:[/bold]")
        for k, v in arch.training_metadata.items():
            console.print(f"    {k}: {v}")

    if detail:
        table = Table(title="Tensors", show_lines=False)
        table.add_column("Name", style="cyan")
        table.add_column("Shape", justify="right")
        table.add_column("Dtype")
        table.add_column("Params", justify="right")
        table.add_column("Size", justify="right")
        for t in header.tensors:
            shape_str = "x".join(str(d) for d in t.shape)
            table.add_row(
                t.name, shape_str, t.dtype,
                format_params(t.param_count), format_size(t.size_bytes),
            )
        console.print()
        console.print(table)


def _display_gguf(info, path: Path) -> None:
    console.print(f"\n[bold]{path.name}[/bold]")
    console.print(f"  Architecture: {info.architecture}")
    console.print(f"  Tensors: {info.tensor_count}")
    if info.context_length is not None:
        console.print(f"  Context length: {info.context_length:,}")
    if info.embedding_length is not None:
        console.print(f"  Embedding dim: {info.embedding_length:,}")
    if info.block_count is not None:
        console.print(f"  Layers: {info.block_count}")
    if info.vocab_size is not None:
        console.print(f"  Vocab size: {info.vocab_size:,}")
    if info.quantization is not None:
        console.print(f"  File type: {info.quantization}")


def _display_directory(dir_info, detail: bool) -> None:
    console.print(f"\n[bold]{dir_info.path.name}/[/bold]")

    if dir_info.config:
        c = dir_info.config
        if "model_type" in c:
            console.print(f"  Model type: {c['model_type']}")
        if "architectures" in c:
            console.print(f"  Architecture: {', '.join(c['architectures'])}")
        for key, label in (
            ("hidden_size", "Hidden size"),
            ("num_hidden_layers", "Layers"),
            ("num_attention_heads", "Attention heads"),
            ("vocab_size", "Vocab size"),
            ("max_position_embeddings", "Max positions"),
        ):
            if key in c:
                console.print(f"  {label}: {c[key]:,}")

    if not dir_info.model_files:
        if not dir_info.config:
            console.print("  No model files or config found.")
        return

    console.print(f"  Files: {len(dir_info.model_files)}")
    console.print(f"  Total size: {format_size(dir_info.total_file_size)}")
    if dir_info.sharded:
        console.print(f"  Sharded: {dir_info.shard_count} shards")

    if dir_info.safetensors_headers:
        total_params = sum(h.total_params for h in dir_info.safetensors_headers)
        total_bytes = sum(h.total_size_bytes for h in dir_info.safetensors_headers)
        total_tensors = sum(len(h.tensors) for h in dir_info.safetensors_headers)
        if total_tensors:
            console.print(f"  Tensors: {total_tensors}")
            console.print(f"  Parameters: {format_params(total_params)}")
            console.print(f"  Tensor data: {format_size(total_bytes)}")
            names = [t.name for h in dir_info.safetensors_headers for t in h.tensors]
            arch = detect_architecture(names)
            arch_str = _format_arch(arch)
            if arch_str:
                console.print(f"  Detected: {arch_str}")
            if detail:
                all_tensors = [t for h in dir_info.safetensors_headers for t in h.tensors]
                combined = SafetensorsHeader(tensors=all_tensors)
                _display_safetensors(combined, dir_info.path, detail=True, arch=arch)

    if dir_info.gguf_info:
        _display_gguf(dir_info.gguf_info, dir_info.model_files[0])


def _display_pipeline(source: Source, detail: bool) -> None:
    console.print(f"\n[bold]{source.path.name}/[/bold] (diffusers pipeline)")
    if source.pipeline_meta and "_class_name" in source.pipeline_meta:
        console.print(f"  Pipeline: {source.pipeline_meta['_class_name']}")
    console.print(f"  Components: {', '.join(source.components)}")
    for component in source.components:
        subdir = source.path / component
        if not subdir.is_dir():
            continue
        sub_info = inspect_directory(subdir)
        _display_directory(sub_info, detail)


def _display_source(source: Source, detail: bool) -> None:
    if source.kind == SourceKind.SAFETENSORS_FILE:
        _display_safetensors(read_header(source.path), source.path, detail)
    elif source.kind == SourceKind.GGUF_FILE:
        _display_gguf(read_gguf_header(source.path), source.path)
    elif source.kind == SourceKind.COMPONENT_DIR:
        _display_directory(inspect_directory(source.path), detail)
    elif source.kind == SourceKind.DIFFUSERS_PIPELINE:
        _display_pipeline(source, detail)
    else:
        console.print(f"[red]Error:[/red] could not classify {source.path}")
        raise typer.Exit(1)


def _walk_for_models(root: Path) -> list[tuple[str, Source]]:
    """Walk a tree for recognized model directories.

    Handles HuggingFace cache layout (`models--org--name/snapshots/hash/`)
    by yielding the snapshot dir as a model with the org/name as label.
    """
    found: list[tuple[str, Source]] = []

    for snap in sorted(root.glob("models--*/snapshots/*")):
        if not snap.is_dir():
            continue
        src = detect_source(snap)
        if src.kind == SourceKind.UNKNOWN:
            continue
        repo_dir = snap.parent.parent.name  # models--org--name
        parts = repo_dir.removeprefix("models--").split("--", 1)
        name = "/".join(parts) if len(parts) == 2 else parts[0]
        found.append((name, src))

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("models--"):
            continue
        src = detect_source(child)
        if src.kind == SourceKind.UNKNOWN:
            continue
        found.append((child.name, src))

    return found


def _summarize_source_for_table(src: Source) -> tuple[int, int, str]:
    """Return (total_bytes, file_count, kind_label) for the scan table."""
    if src.kind == SourceKind.DIFFUSERS_PIPELINE:
        total = 0
        files = 0
        for component in src.components:
            for f in (src.path / component).glob("*.safetensors"):
                total += f.stat().st_size
                files += 1
        return total, files, "pipeline"

    if src.kind == SourceKind.COMPONENT_DIR:
        total = sum(f.stat().st_size for f in src.shards)
        label = "sharded" if src.sharded else "component"
        return total, len(src.shards), label

    if src.kind == SourceKind.SAFETENSORS_FILE:
        return src.path.stat().st_size, 1, "safetensors"

    if src.kind == SourceKind.GGUF_FILE:
        return src.path.stat().st_size, 1, "gguf"

    return 0, 0, "unknown"


def _display_tree(root: Path, entries: list[tuple[str, Source]]) -> None:
    table = Table(title=f"Models in {root}")
    table.add_column("Name", style="cyan")
    table.add_column("Kind")
    table.add_column("Size", justify="right")
    table.add_column("Files", justify="right")

    total_size = 0
    for name, src in entries:
        size, count, kind_label = _summarize_source_for_table(src)
        total_size += size
        table.add_row(name, kind_label, format_size(size), str(count))

    console.print(table)
    console.print(f"\n  Total: {len(entries)} models, {format_size(total_size)}")


@app.command()
def inspect(
    path: Path = typer.Argument(..., help="File, component dir, diffusers pipeline, or directory tree"),
    detail: bool = typer.Option(False, "--detail", "-d", help="Show architecture detection and full tensor list"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Walk a directory tree (replaces the old `scan` command)"),
) -> None:
    """Inspect model file headers and directory layouts."""
    if not path.exists():
        console.print(f"[red]Error:[/red] {path} not found")
        raise typer.Exit(1)

    if recursive:
        if not path.is_dir():
            console.print("[red]Error:[/red] --recursive requires a directory")
            raise typer.Exit(1)
        entries = _walk_for_models(path)
        if not entries:
            console.print("No models found.")
            return
        _display_tree(path, entries)
        return

    _display_source(detect_source(path), detail)
