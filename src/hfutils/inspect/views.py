"""Rich display helpers for `hfutils inspect`.

These render `Source` objects (or their enriched parts) to the console.
Kept separate from `commands/inspect.py` so the command module stays thin
(argument parsing + dispatch).
"""

from pathlib import Path

from rich.console import Console
from rich.table import Table

from hfutils.inspect.architecture import detect_architecture
from hfutils.inspect.common import SafetensorsHeader, format_params, format_size
from hfutils.sources.detect import Source, SourceKind


def _format_arch(arch) -> str | None:
    parts = []
    if arch.family != "Unknown":
        parts.append(arch.family)
    if arch.adapter_type:
        parts.append(arch.adapter_type)
    return " / ".join(parts) if parts else None


def display_safetensors(header: SafetensorsHeader, path: Path, detail: bool, console: Console, arch=None) -> None:
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


def display_gguf(info, path: Path, console: Console) -> None:
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


def display_directory(src: Source, detail: bool, console: Console) -> None:
    src.enrich()
    console.print(f"\n[bold]{src.path.name}/[/bold]")

    if src.config:
        c = src.config
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

    if not src.shards and not src.gguf_info:
        if not src.config:
            console.print("  No model files or config found.")
        return

    console.print(f"  Files: {len(src.shards) + (1 if src.gguf_info else 0)}")
    console.print(f"  Total size: {format_size(src.total_file_size)}")
    if src.sharded:
        console.print(f"  Sharded: {src.shard_count} shards")

    if src.safetensors_headers:
        total_params = sum(h.total_params for h in src.safetensors_headers)
        total_bytes = sum(h.total_size_bytes for h in src.safetensors_headers)
        total_tensors = sum(len(h.tensors) for h in src.safetensors_headers)
        if total_tensors:
            console.print(f"  Tensors: {total_tensors}")
            console.print(f"  Parameters: {format_params(total_params)}")
            console.print(f"  Tensor data: {format_size(total_bytes)}")
            names = [t.name for h in src.safetensors_headers for t in h.tensors]
            arch = detect_architecture(names)
            arch_str = _format_arch(arch)
            if arch_str:
                console.print(f"  Detected: {arch_str}")
            if detail:
                all_tensors = [t for h in src.safetensors_headers for t in h.tensors]
                combined = SafetensorsHeader(tensors=all_tensors)
                display_safetensors(combined, src.path, detail=True, console=console, arch=arch)

    if src.gguf_info:
        display_gguf(src.gguf_info, src.path, console)


def display_pipeline(source: Source, detail: bool, console: Console) -> None:
    from hfutils.sources.detect import detect_source
    console.print(f"\n[bold]{source.path.name}/[/bold] (diffusers pipeline)")
    if source.pipeline_meta and "_class_name" in source.pipeline_meta:
        console.print(f"  Pipeline: {source.pipeline_meta['_class_name']}")
    console.print(f"  Components: {', '.join(source.components)}")
    for component in source.components:
        subdir = source.path / component
        if not subdir.is_dir():
            continue
        display_directory(detect_source(subdir), detail, console)


def display_source(source: Source, detail: bool, console: Console) -> None:
    if source.kind == SourceKind.SAFETENSORS_FILE:
        source.enrich()
        display_safetensors(source.safetensors_headers[0], source.path, detail, console)
    elif source.kind == SourceKind.GGUF_FILE:
        source.enrich()
        display_gguf(source.gguf_info, source.path, console)
    elif source.kind in (SourceKind.COMPONENT_DIR, SourceKind.PYTORCH_DIR):
        display_directory(source, detail, console)
    elif source.kind == SourceKind.DIFFUSERS_PIPELINE:
        display_pipeline(source, detail, console)
    else:
        import typer
        console.print(f"[red]Error:[/red] could not classify {source.path}")
        raise typer.Exit(1)


def status_label(src: Source) -> str:
    if src.incomplete:
        return "[red]INCOMPLETE[/red]"
    if not src.has_config:
        return "[yellow]no config[/yellow]"
    return "[green]ok[/green]"


def summarize_source_for_table(src: Source) -> tuple[int, int]:
    """Return (total_bytes, file_count) for the recursive-inspect table."""
    if src.kind == SourceKind.DIFFUSERS_PIPELINE:
        total = 0
        files = 0
        for component in src.components:
            for f in (src.path / component).iterdir():
                if f.is_file() and f.suffix == ".safetensors":
                    total += f.stat().st_size
                    files += 1
        return total, files

    if src.kind in (SourceKind.COMPONENT_DIR, SourceKind.PYTORCH_DIR):
        src.enrich()
        return src.total_file_size, len(src.shards) or 0

    if src.kind in (SourceKind.SAFETENSORS_FILE, SourceKind.GGUF_FILE):
        return src.path.stat().st_size, 1

    return 0, 0


def display_tree(root: Path, entries: list[tuple[str, Source]], console: Console) -> None:
    table = Table(title=f"Models in {root}")
    table.add_column("Name", style="cyan")
    table.add_column("Kind")
    table.add_column("Size", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Status")

    total_size = 0
    for name, src in entries:
        size, count = summarize_source_for_table(src)
        total_size += size
        table.add_row(name, src.display_kind(), format_size(size), str(count), status_label(src))

    console.print(table)
    console.print(f"\n  Total: {len(entries)} models, {format_size(total_size)}")
