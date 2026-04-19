"""Classify a local path into a handled source shape.

Consolidates the four detection paths the codebase used to carry: `merge`
globbed for an index, `comfyui pack` sniffed for model_index.json, `inspect`
looked at extensions, `scan` walked. All those live here now.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import orjson

# Diffusers subfolder names we treat as components. ComfyUI's `unet` folder
# name is gone in modern releases; its diffusers equivalent is `transformer`.
DIFFUSERS_COMPONENT_NAMES: tuple[str, ...] = (
    "transformer",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "scheduler",  # not always a weights folder, but present
    "tokenizer",  # ditto
)


class SourceKind(str, Enum):
    DIFFUSERS_PIPELINE = "diffusers_pipeline"
    COMPONENT_DIR = "component_dir"
    SAFETENSORS_FILE = "safetensors_file"
    GGUF_FILE = "gguf_file"
    UNKNOWN = "unknown"


@dataclass
class Source:
    path: Path
    kind: SourceKind
    # DIFFUSERS_PIPELINE: subfolder names that contain safetensors weights.
    components: list[str] = field(default_factory=list)
    # COMPONENT_DIR: shard paths (1 = single-file, N = sharded).
    # SAFETENSORS_FILE: [self].
    shards: list[Path] = field(default_factory=list)
    # True iff a *.safetensors.index.json exists alongside the shards.
    sharded: bool = False
    # DIFFUSERS_PIPELINE: parsed model_index.json.
    pipeline_meta: dict | None = None


def _component_has_weights(subdir: Path) -> bool:
    return subdir.is_dir() and any(subdir.glob("*.safetensors"))


def _detect_component_dir(path: Path) -> Source | None:
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        return None
    sharded = any(path.glob("*.safetensors.index.json"))
    return Source(
        path=path,
        kind=SourceKind.COMPONENT_DIR,
        shards=shards,
        sharded=sharded,
    )


def detect_source(path: Path) -> Source:
    if not path.exists():
        return Source(path=path, kind=SourceKind.UNKNOWN)

    if path.is_file():
        if path.suffix == ".safetensors":
            return Source(
                path=path,
                kind=SourceKind.SAFETENSORS_FILE,
                shards=[path],
            )
        if path.suffix == ".gguf":
            return Source(path=path, kind=SourceKind.GGUF_FILE)
        return Source(path=path, kind=SourceKind.UNKNOWN)

    # Directory cases.
    model_index = path / "model_index.json"
    if model_index.is_file():
        try:
            meta = orjson.loads(model_index.read_bytes())
        except orjson.JSONDecodeError:
            meta = None
        components = [
            name for name in DIFFUSERS_COMPONENT_NAMES
            if _component_has_weights(path / name)
        ]
        return Source(
            path=path,
            kind=SourceKind.DIFFUSERS_PIPELINE,
            components=components,
            pipeline_meta=meta,
        )

    component = _detect_component_dir(path)
    if component is not None:
        return component

    return Source(path=path, kind=SourceKind.UNKNOWN)
