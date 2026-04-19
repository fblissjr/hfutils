"""Compact per-component metadata summary used during conversion.

Reads only safetensors headers and config.json -- never loads tensor data.
"""

from dataclasses import dataclass
from pathlib import Path

from hfutils.inspect.architecture import architecture_name_from_config, detect_architecture
from hfutils.inspect.common import (
    DtypeBreakdown,
    QUANT_DTYPE_LABELS,
    SafetensorsHeader,
    format_params,
    format_size,
    read_json_if_exists,
)
from hfutils.inspect.safetensors import read_header


@dataclass
class ComponentSummary:
    source: Path
    file_count: int = 0
    sharded: bool = False
    tensor_count: int = 0
    total_params: int = 0
    total_bytes: int = 0
    dominant_dtype: str | None = None
    quantization: str | None = None
    architecture: str | None = None


def _gather_safetensors(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".safetensors":
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.safetensors"))
    return []


def _merge_breakdowns(headers: list[SafetensorsHeader]) -> list[DtypeBreakdown]:
    merged: dict[str, DtypeBreakdown] = {}
    for h in headers:
        for b in h.dtype_breakdown():
            agg = merged.setdefault(b.dtype, DtypeBreakdown(b.dtype, 0, 0, 0))
            agg.param_count += b.param_count
            agg.tensor_count += b.tensor_count
            agg.size_bytes += b.size_bytes
    return sorted(merged.values(), key=lambda b: b.size_bytes, reverse=True)


def _detect_architecture_name(source: Path, headers: list[SafetensorsHeader]) -> str | None:
    config_dir = source if source.is_dir() else source.parent
    config = read_json_if_exists(config_dir / "config.json")
    if (name := architecture_name_from_config(config)):
        return name

    names = [t.name for h in headers for t in h.tensors]
    info = detect_architecture(names)
    return info.family if info.family != "Unknown" else None


def summarize_component(source: Path) -> ComponentSummary:
    """Build a ComponentSummary from a file or directory of safetensors."""
    files = _gather_safetensors(source)
    if not files:
        return ComponentSummary(source=source)

    headers = [read_header(f) for f in files]
    sharded = source.is_dir() and any(source.glob("*.safetensors.index.json"))

    breakdowns = _merge_breakdowns(headers)
    dominant = breakdowns[0].dtype if breakdowns else None
    quant = next((QUANT_DTYPE_LABELS[b.dtype] for b in breakdowns if b.dtype in QUANT_DTYPE_LABELS), None)

    return ComponentSummary(
        source=source,
        file_count=len(files),
        sharded=sharded,
        tensor_count=sum(len(h.tensors) for h in headers),
        total_params=sum(h.total_params for h in headers),
        total_bytes=sum(f.stat().st_size for f in files),
        dominant_dtype=dominant,
        quantization=quant,
        architecture=_detect_architecture_name(source, headers),
    )


def format_summary_lines(summary: ComponentSummary) -> list[str]:
    if summary.file_count == 0:
        return [f"  (no safetensors found at {summary.source})"]

    fmt = f"sharded ({summary.file_count} files)" if summary.sharded else (
        f"{summary.file_count} files" if summary.file_count > 1 else "single file"
    )
    lines = [f"  format:       {fmt}"]
    if summary.architecture:
        lines.append(f"  architecture: {summary.architecture}")
    lines.append(f"  tensors:      {summary.tensor_count}")
    if summary.total_params:
        lines.append(f"  params:       {format_params(summary.total_params)}")
    lines.append(f"  size:         {format_size(summary.total_bytes)}")
    if summary.dominant_dtype:
        lines.append(f"  dtype:        {summary.dominant_dtype}")
    if summary.quantization:
        lines.append(f"  quant:        {summary.quantization}")
    return lines
