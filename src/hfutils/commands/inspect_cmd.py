"""Inspect safetensors and GGUF model files."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hfutils.cli import app
from hfutils.inspect.common import SafetensorsHeader, format_size, format_params

console = Console()


def _display_safetensors(header: SafetensorsHeader, path: Path, detail: bool) -> None:
    """Display safetensors inspection results."""
    from hfutils.inspect.architecture import detect_architecture

    arch = detect_architecture(
        [t.name for t in header.tensors],
        metadata=header.metadata if header.metadata else None,
    )

    console.print(f"\n[bold]{path.name}[/bold]")

    arch_parts = []
    if arch.family != "Unknown":
        arch_parts.append(arch.family)
    if arch.adapter_type:
        arch_parts.append(arch.adapter_type)
    if arch_parts:
        console.print(f"  Architecture: {' / '.join(arch_parts)}")

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
    """Display GGUF inspection results."""
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
    """Display directory-level inspection results."""
    from hfutils.inspect.architecture import detect_architecture

    console.print(f"\n[bold]{dir_info.path.name}/[/bold]")

    # Config info
    if dir_info.config:
        c = dir_info.config
        if "model_type" in c:
            console.print(f"  Model type: {c['model_type']}")
        if "architectures" in c:
            console.print(f"  Architecture: {', '.join(c['architectures'])}")
        config_fields = [
            ("hidden_size", "Hidden size"),
            ("num_hidden_layers", "Layers"),
            ("num_attention_heads", "Attention heads"),
            ("vocab_size", "Vocab size"),
            ("max_position_embeddings", "Max positions"),
        ]
        for key, label in config_fields:
            if key in c:
                console.print(f"  {label}: {c[key]:,}")

    # Model files summary
    if not dir_info.model_files:
        if not dir_info.config:
            console.print("  No model files or config found.")
        return

    console.print(f"  Files: {len(dir_info.model_files)}")
    console.print(f"  Total size: {format_size(dir_info.total_file_size)}")

    if dir_info.sharded:
        console.print(f"  Sharded: {dir_info.shard_count} shards")

    # Aggregate safetensors info
    if dir_info.safetensors_headers:
        all_tensors = []
        for h in dir_info.safetensors_headers:
            all_tensors.extend(h.tensors)

        if all_tensors:
            total_params = sum(t.param_count for t in all_tensors)
            total_bytes = sum(t.size_bytes for t in all_tensors)
            console.print(f"  Tensors: {len(all_tensors)}")
            console.print(f"  Parameters: {format_params(total_params)}")
            console.print(f"  Tensor data: {format_size(total_bytes)}")

            # Architecture detection from tensor names
            arch = detect_architecture([t.name for t in all_tensors])
            arch_parts = []
            if arch.family != "Unknown":
                arch_parts.append(arch.family)
            if arch.adapter_type:
                arch_parts.append(arch.adapter_type)
            if arch_parts:
                console.print(f"  Detected: {' / '.join(arch_parts)}")

            if detail:
                from hfutils.inspect.common import SafetensorsHeader
                combined = SafetensorsHeader(tensors=all_tensors)
                _display_safetensors(combined, dir_info.path, detail=True)

    # GGUF info
    if dir_info.gguf_info:
        _display_gguf(dir_info.gguf_info, dir_info.model_files[0])


@app.command()
def inspect(
    path: Path = typer.Argument(..., help="Path to a .safetensors, .gguf file, or model directory"),
    detail: bool = typer.Option(False, "--detail", "-d", help="Show architecture detection and full tensor list"),
) -> None:
    """Inspect model file headers: tensors, params, VRAM estimate."""
    if not path.exists():
        console.print(f"[red]Error:[/red] {path} not found")
        raise typer.Exit(1)

    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        from hfutils.inspect.safetensors import read_header
        header = read_header(path)
        _display_safetensors(header, path, detail)
    elif suffix == ".gguf":
        from hfutils.inspect.gguf import read_gguf_header
        info = read_gguf_header(path)
        _display_gguf(info, path)
    elif path.is_dir():
        from hfutils.inspect.directory import inspect_directory
        dir_info = inspect_directory(path)
        _display_directory(dir_info, detail)
    else:
        console.print(f"[red]Error:[/red] Unsupported file type: {suffix}")
        raise typer.Exit(1)
