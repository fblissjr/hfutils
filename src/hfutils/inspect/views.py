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
from hfutils.sources.detect import detect_source, enrich
from hfutils.sources.types import (
    ComponentSource,
    GgufFileSource,
    PipelineSource,
    PytorchDirSource,
    SafetensorsFileSource,
    Source,
    UnknownSource,
    display_kind,
)


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

    if arch.likely_triggers:
        console.print("\n  [bold]Likely triggers[/bold] [dim](heuristic, top tags from training captions):[/dim]")
        console.print(f"    {', '.join(arch.likely_triggers)}")

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
    if info.rope_freq_base is not None:
        console.print(f"  RoPE freq base: {info.rope_freq_base:g}")
    if info.rope_freq_scale is not None:
        console.print(f"  RoPE freq scale: {info.rope_freq_scale:g}")
    if info.rope_scaling_type is not None:
        console.print(f"  RoPE scaling: {info.rope_scaling_type}")
    if info.bos_token_id is not None:
        console.print(f"  BOS token id: {info.bos_token_id}")
    if info.eos_token_id is not None:
        console.print(f"  EOS token id: {info.eos_token_id}")
    if info.chat_template is not None:
        template = info.chat_template.replace("\n", " ")
        preview = template if len(template) <= 80 else template[:77] + "..."
        console.print(f"  Chat template: {preview}")


def _display_config_fields(config: dict, console: Console) -> None:
    if "model_type" in config:
        console.print(f"  Model type: {config['model_type']}")
    if "architectures" in config:
        console.print(f"  Architecture: {', '.join(config['architectures'])}")
    for key, label in (
        ("hidden_size", "Hidden size"),
        ("num_hidden_layers", "Layers"),
        ("num_attention_heads", "Attention heads"),
        ("vocab_size", "Vocab size"),
        ("max_position_embeddings", "Max positions"),
    ):
        if key in config:
            console.print(f"  {label}: {config[key]:,}")


def display_directory(src: ComponentSource | PytorchDirSource, detail: bool, console: Console) -> None:
    view = enrich(src)
    console.print(f"\n[bold]{src.path.name}/[/bold]")

    if view.config:
        _display_config_fields(view.config, console)

    shards = src.shards if isinstance(src, ComponentSource) else []
    if not shards and not view.gguf_info:
        if not view.config:
            console.print("  No model files or config found.")
        return

    console.print(f"  Files: {len(shards) + (1 if view.gguf_info else 0)}")
    console.print(f"  Total size: {format_size(view.total_file_size)}")
    if isinstance(src, ComponentSource) and src.sharded:
        console.print(f"  Sharded: {len(shards)} shards")

    if view.safetensors_headers:
        combined = SafetensorsHeader.combine(view.safetensors_headers)
        if combined.tensors:
            console.print(f"  Tensors: {len(combined.tensors)}")
            console.print(f"  Parameters: {format_params(combined.total_params)}")
            console.print(f"  Tensor data: {format_size(combined.total_size_bytes)}")
            arch = detect_architecture([t.name for t in combined.tensors])
            arch_str = _format_arch(arch)
            if arch_str:
                console.print(f"  Detected: {arch_str}")
            if detail:
                display_safetensors(combined, src.path, detail=True, console=console, arch=arch)


def display_pipeline(source: PipelineSource, detail: bool, console: Console) -> None:
    console.print(f"\n[bold]{source.path.name}/[/bold] (diffusers pipeline)")
    if source.pipeline_meta and "_class_name" in source.pipeline_meta:
        console.print(f"  Pipeline: {source.pipeline_meta['_class_name']}")
    console.print(f"  Components: {', '.join(source.components)}")
    for component in source.components:
        subdir = source.path / component
        if not subdir.is_dir():
            continue
        sub = detect_source(subdir)
        if isinstance(sub, (ComponentSource, PytorchDirSource)):
            display_directory(sub, detail, console)


def display_source(source: Source, detail: bool, console: Console) -> None:
    match source:
        case SafetensorsFileSource(path=p):
            view = enrich(source)
            display_safetensors(view.safetensors_headers[0], p, detail, console)
        case GgufFileSource(path=p):
            view = enrich(source)
            assert view.gguf_info is not None
            display_gguf(view.gguf_info, p, console)
        case ComponentSource() | PytorchDirSource():
            display_directory(source, detail, console)
        case PipelineSource():
            display_pipeline(source, detail, console)
        case UnknownSource(path=p):
            import typer
            console.print(f"[red]Error:[/red] could not classify {p}")
            raise typer.Exit(1)


def status_label(src: Source) -> str:
    match src:
        case ComponentSource(integrity_error=e) if e is not None:
            return "[red]CORRUPT[/red]"
        case ComponentSource(incomplete=True):
            return "[red]INCOMPLETE[/red]"
        case ComponentSource(has_config=False) | PytorchDirSource(has_config=False):
            return "[yellow]no config[/yellow]"
        case _:
            return "[green]ok[/green]"


def summarize_source_for_table(src: Source) -> tuple[int, int]:
    """Return (total_bytes, file_count) for the recursive-inspect table.

    Uses stat() throughout; never calls enrich() since the table only wants
    sizes, not headers. Keeps the recursive walk cheap."""
    match src:
        case PipelineSource(components=components):
            total = 0
            files = 0
            for component in components:
                for f in (src.path / component).iterdir():
                    if f.is_file() and f.suffix == ".safetensors":
                        total += f.stat().st_size
                        files += 1
            return total, files
        case ComponentSource(shards=shards):
            return sum(f.stat().st_size for f in shards), len(shards)
        case PytorchDirSource(files=files):
            return sum(f.stat().st_size for f in files), len(files)
        case SafetensorsFileSource(path=p) | GgufFileSource(path=p):
            return p.stat().st_size, 1
        case _:
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
        table.add_row(name, display_kind(src), format_size(size), str(count), status_label(src))

    console.print(table)
    console.print(f"\n  Total: {len(entries)} models, {format_size(total_size)}")
