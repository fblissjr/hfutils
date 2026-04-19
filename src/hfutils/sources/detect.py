"""Single classification point for CLI inputs.

`detect_source(path)` returns one of six `Source` variants (see sources/types.py).
`enrich(source)` optionally reads headers/config for a display-ready view.
"""

from pathlib import Path

import orjson

from hfutils.inspect.common import read_json_if_exists
from hfutils.inspect.gguf import read_gguf_header
from hfutils.inspect.safetensors import read_header, read_raw_header
from hfutils.sources.types import (
    ComponentSource,
    EnrichedView,
    GgufFileSource,
    IntegrityError,
    PipelineSource,
    PytorchDirSource,
    SafetensorsFileSource,
    Source,
    UnknownSource,
)

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
        return ComponentSource(
            path=path,
            shards=shards,
            sharded=sharded,
            incomplete=incomplete or integrity_error is not None,
            integrity_error=integrity_error,
            has_config=has_config,
        )

    pytorch_files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix in {".bin", ".pt", ".pth"}
    )
    if pytorch_files:
        return PytorchDirSource(path=path, files=pytorch_files, has_config=has_config)

    return None


def detect_source(path: Path) -> Source:
    if not path.exists():
        return UnknownSource(path=path)

    if path.is_file():
        if path.suffix == ".safetensors":
            return SafetensorsFileSource(path=path)
        if path.suffix == ".gguf":
            return GgufFileSource(path=path)
        return UnknownSource(path=path)

    if (path / "model_index.json").is_file():
        meta = read_json_if_exists(path / "model_index.json")
        components = [
            name for name in DIFFUSERS_COMPONENT_NAMES
            if _dir_has_weight_file(path / name)
        ]
        return PipelineSource(path=path, components=components, pipeline_meta=meta)

    component = _detect_component_or_pytorch_dir(path)
    if component is not None:
        return component

    return UnknownSource(path=path)


def enrich(source: Source) -> EnrichedView:
    """Read headers, config.json, file sizes. Returns a fresh EnrichedView.

    Callers decide when to pay this cost; the recursive walker deliberately
    skips it. This function is stateless (no mutation of `source`), so it's
    safe to call from multiple threads on the same source."""
    match source:
        case SafetensorsFileSource(path=p):
            return EnrichedView(
                config=read_json_if_exists(p.parent / "config.json"),
                total_file_size=p.stat().st_size,
                safetensors_headers=[read_header(p)],
            )
        case GgufFileSource(path=p):
            return EnrichedView(
                total_file_size=p.stat().st_size,
                gguf_info=read_gguf_header(p),
            )
        case ComponentSource(path=p, shards=shards):
            return EnrichedView(
                config=read_json_if_exists(p / "config.json"),
                total_file_size=sum(s.stat().st_size for s in shards),
                safetensors_headers=[read_header(s) for s in shards],
            )
        case PytorchDirSource(path=p, files=files):
            return EnrichedView(
                config=read_json_if_exists(p / "config.json"),
                total_file_size=sum(f.stat().st_size for f in files),
            )
        case PipelineSource():
            # Pipelines don't enrich at the top level; per-component inspection
            # calls enrich() on each subdir's detect_source result.
            return EnrichedView()
        case UnknownSource():
            return EnrichedView()
