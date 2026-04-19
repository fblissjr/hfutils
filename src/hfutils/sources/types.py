"""Source variants.

The `Source` union replaces the v0.6 god-dataclass. Every consumer does

    match source:
        case PipelineSource(components=c) if c: ...
        case ComponentSource(shards=shards): ...
        case SafetensorsFileSource(path=p): ...
        case GgufFileSource(path=p): ...
        case PytorchDirSource(files=fs): ...
        case UnknownSource(): ...

which lets the type checker reject `components` on a gguf file, etc.

`EnrichedView` is the extra data that a single-target inspect needs: config
(JSON), safetensors headers, gguf info, total file size. It's a separate
value object so the core source types stay cheap to construct during a
recursive walk; callers pay for `enrich(source)` only when they want the view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from hfutils.inspect.common import SafetensorsHeader
from hfutils.inspect.gguf import GGUFInfo


IntegrityKind = Literal["truncated", "unreadable_header"]


@dataclass(frozen=True)
class IntegrityError:
    """Structured error for one unhealthy shard. Surfaced on ComponentSource."""

    kind: IntegrityKind
    file: Path
    detail: str

    def __str__(self) -> str:
        return f"{self.file.name}: {self.kind} ({self.detail})"


@dataclass(frozen=True)
class PipelineSource:
    """A diffusers pipeline directory (has `model_index.json`)."""

    path: Path
    components: list[str]
    pipeline_meta: dict | None = None


@dataclass(frozen=True)
class ComponentSource:
    """A directory of safetensors shards (sharded or single-file)."""

    path: Path
    shards: list[Path]
    sharded: bool = False
    has_config: bool = False
    incomplete: bool = False
    integrity_error: IntegrityError | None = None


@dataclass(frozen=True)
class SafetensorsFileSource:
    """A single .safetensors file."""

    path: Path


@dataclass(frozen=True)
class GgufFileSource:
    """A single .gguf file."""

    path: Path


@dataclass(frozen=True)
class PytorchDirSource:
    """A directory of legacy pytorch weights (.bin / .pt / .pth)."""

    path: Path
    files: list[Path]
    has_config: bool = False


@dataclass(frozen=True)
class UnknownSource:
    """Unclassifiable path (doesn't exist, or doesn't match any known shape)."""

    path: Path


Source: TypeAlias = (
    PipelineSource
    | ComponentSource
    | SafetensorsFileSource
    | GgufFileSource
    | PytorchDirSource
    | UnknownSource
)


@dataclass
class EnrichedView:
    """Extra data produced by `enrich(source)`. Optional per kind:

    - Pipelines: pipeline_meta is already on the Source; view mostly empty.
    - Components / safetensors files: safetensors_headers + total_file_size + config.
    - GGUF: gguf_info + total_file_size.
    - Pytorch: total_file_size + config.
    - Unknown: empty.
    """

    config: dict | None = None
    total_file_size: int = 0
    safetensors_headers: list[SafetensorsHeader] = field(default_factory=list)
    gguf_info: GGUFInfo | None = None


def display_kind(source: Source) -> str:
    """Short human label for recursive-inspect rows."""
    match source:
        case ComponentSource(sharded=True):
            return "sharded"
        case ComponentSource():
            return "component"
        case PipelineSource():
            return "pipeline"
        case SafetensorsFileSource():
            return "safetensors"
        case GgufFileSource():
            return "gguf"
        case PytorchDirSource():
            return "pytorch"
        case UnknownSource():
            return "unknown"


def source_path(source: Source) -> Path:
    """Every variant has `.path`; this helper avoids `match src: case X: return src.path` boilerplate."""
    return source.path
