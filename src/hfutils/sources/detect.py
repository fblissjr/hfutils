"""Single classification point for CLI inputs.

`detect_source(path)` returns a `Source` describing what was found --
pipeline, component dir, safetensors file, gguf file, pytorch dir, or
unknown -- along with completeness and config-presence flags that the
recursive inspect table surfaces.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import orjson

from hfutils.inspect.common import SafetensorsHeader, read_json_if_exists
from hfutils.inspect.gguf import GGUFInfo, read_gguf_header
from hfutils.inspect.safetensors import read_header, read_raw_header

# Diffusers subfolders we recognize. Weights-bearing ones (transformer, vae,
# text_encoder[_N]) are the `convert comfyui` candidates; scheduler/tokenizer
# are listed so detection works on full pipelines even though they carry no
# weights we ship.
DIFFUSERS_COMPONENT_NAMES: tuple[str, ...] = (
    "transformer",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "scheduler",
    "tokenizer",
)

# File suffixes that count as weights when walking a tree. Legacy `.bin/.pt/.pth`
# pytorch weights are reported but never converted.
WEIGHT_EXTENSIONS: set[str] = {".safetensors", ".gguf", ".bin", ".pt", ".pth"}


class SourceKind(str, Enum):
    DIFFUSERS_PIPELINE = "diffusers_pipeline"
    COMPONENT_DIR = "component_dir"
    SAFETENSORS_FILE = "safetensors_file"
    GGUF_FILE = "gguf_file"
    PYTORCH_DIR = "pytorch_dir"
    UNKNOWN = "unknown"


IntegrityKind = Literal["truncated", "unreadable_header"]


@dataclass
class IntegrityError:
    """Structured error for one unhealthy shard. Surfaced in Source.integrity_error
    so callers can branch on `kind` rather than substring-matching `detail`."""
    kind: IntegrityKind
    file: Path
    detail: str

    def __str__(self) -> str:
        return f"{self.file.name}: {self.kind} ({self.detail})"


@dataclass
class Source:
    path: Path
    kind: SourceKind
    # DIFFUSERS_PIPELINE: subfolder names that contain weights.
    components: list[str] = field(default_factory=list)
    # COMPONENT_DIR or SAFETENSORS_FILE: safetensors file paths.
    shards: list[Path] = field(default_factory=list)
    # True iff a *.safetensors.index.json sits alongside the shards.
    sharded: bool = False
    # Index file lists shards that aren't all on disk. COMPONENT_DIR only.
    incomplete: bool = False
    # Populated when a shard's header is unreadable or its declared tensor-data
    # size doesn't match the file's physical size. Truncation, corruption, etc.
    integrity_error: IntegrityError | None = None
    # config.json present next to the weights (or model_index.json for pipelines).
    has_config: bool = False
    # DIFFUSERS_PIPELINE: parsed model_index.json.
    pipeline_meta: dict | None = None

    # Populated by .enrich() on demand -- reading headers is cheap per-file but
    # wasteful in batch walks, so detect_source leaves these empty by default.
    config: dict | None = None
    total_file_size: int = 0
    safetensors_headers: list[SafetensorsHeader] = field(default_factory=list)
    gguf_info: GGUFInfo | None = None
    _enriched: bool = False

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    def display_kind(self) -> str:
        """Short human label for table rows."""
        if self.kind == SourceKind.COMPONENT_DIR:
            return "sharded" if self.sharded else "component"
        if self.kind == SourceKind.DIFFUSERS_PIPELINE:
            return "pipeline"
        if self.kind == SourceKind.SAFETENSORS_FILE:
            return "safetensors"
        if self.kind == SourceKind.GGUF_FILE:
            return "gguf"
        if self.kind == SourceKind.PYTORCH_DIR:
            return "pytorch"
        return "unknown"

    def enrich(self) -> "Source":
        """Read headers, config, and sizes. Idempotent. Returns self.

        Not thread-safe: callers must not invoke enrich() on the same Source
        from multiple threads. The recursive walker deliberately skips
        enrichment and only classifies; display code on the main thread calls
        enrich() serially afterwards.
        """
        if self._enriched:
            return self

        if self.kind == SourceKind.SAFETENSORS_FILE:
            _populate_safetensors(self, [self.path], config_dir=self.path.parent)
        elif self.kind == SourceKind.GGUF_FILE:
            self.gguf_info = read_gguf_header(self.path)
            self.total_file_size = self.path.stat().st_size
        elif self.kind == SourceKind.COMPONENT_DIR:
            _populate_safetensors(self, self.shards, config_dir=self.path)
        elif self.kind == SourceKind.PYTORCH_DIR:
            self.total_file_size = sum(
                p.stat().st_size for p in self.path.iterdir()
                if p.is_file() and p.suffix in {".bin", ".pt", ".pth"}
            )
            self.config = read_json_if_exists(self.path / "config.json")

        self._enriched = True
        return self


def _populate_safetensors(src: "Source", paths: list[Path], *, config_dir: Path) -> None:
    """Populate `src` with safetensors headers, total size, and config.json.
    Shared by SAFETENSORS_FILE and COMPONENT_DIR enrichment paths."""
    src.safetensors_headers = [read_header(p) for p in paths]
    src.total_file_size = sum(p.stat().st_size for p in paths)
    src.config = read_json_if_exists(config_dir / "config.json")


def _dir_has_weight_file(subdir: Path) -> bool:
    if not subdir.is_dir():
        return False
    for child in subdir.iterdir():
        if child.is_file() and child.suffix in WEIGHT_EXTENSIONS:
            return True
    return False


def _check_incomplete(index_path: Path, present_names: set[str]) -> bool:
    try:
        index = orjson.loads(index_path.read_bytes())
    except (orjson.JSONDecodeError, OSError):
        return False
    expected = set(index.get("weight_map", {}).values())
    return bool(expected - present_names)


def _check_shard_integrity(shards: list[Path]) -> IntegrityError | None:
    """Read each shard's header; confirm declared tensor-data size matches
    file size. Returns a structured `IntegrityError` if anything is off, or
    None if all shards check out."""
    for shard in shards:
        try:
            raw = read_raw_header(shard)
        except (ValueError, orjson.JSONDecodeError, OSError) as e:
            return IntegrityError(kind="unreadable_header", file=shard, detail=str(e))

        declared = raw.tensors[-1].data_offset_end if raw.tensors else 0
        actual = shard.stat().st_size - raw.data_region_start
        if actual < declared:
            return IntegrityError(
                kind="truncated", file=shard,
                detail=f"header declares {declared} bytes of tensor data, file has {actual}",
            )
    return None


def _detect_component_or_pytorch_dir(path: Path) -> Source | None:
    shards = sorted(path.glob("*.safetensors"))
    has_config = (path / "config.json").is_file()

    if shards:
        index_candidates = sorted(path.glob("*.safetensors.index.json"))
        sharded = bool(index_candidates)
        incomplete = (
            _check_incomplete(index_candidates[0], {f.name for f in shards})
            if sharded else False
        )
        integrity_error = _check_shard_integrity(shards)
        return Source(
            path=path,
            kind=SourceKind.COMPONENT_DIR,
            shards=shards,
            sharded=sharded,
            incomplete=incomplete or integrity_error is not None,
            integrity_error=integrity_error,
            has_config=has_config,
        )

    legacy_pytorch = any(
        f.is_file() and f.suffix in {".bin", ".pt", ".pth"}
        for f in path.iterdir()
    )
    if legacy_pytorch:
        return Source(path=path, kind=SourceKind.PYTORCH_DIR, has_config=has_config)

    return None


def detect_source(path: Path) -> Source:
    if not path.exists():
        return Source(path=path, kind=SourceKind.UNKNOWN)

    if path.is_file():
        if path.suffix == ".safetensors":
            return Source(path=path, kind=SourceKind.SAFETENSORS_FILE, shards=[path])
        if path.suffix == ".gguf":
            return Source(path=path, kind=SourceKind.GGUF_FILE)
        return Source(path=path, kind=SourceKind.UNKNOWN)

    if (path / "model_index.json").is_file():
        meta = read_json_if_exists(path / "model_index.json")
        components = [
            name for name in DIFFUSERS_COMPONENT_NAMES
            if _dir_has_weight_file(path / name)
        ]
        return Source(
            path=path,
            kind=SourceKind.DIFFUSERS_PIPELINE,
            components=components,
            pipeline_meta=meta,
            has_config=True,
        )

    component = _detect_component_or_pytorch_dir(path)
    if component is not None:
        return component

    return Source(path=path, kind=SourceKind.UNKNOWN)
