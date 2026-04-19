"""Directory-level model inspection.

Combines config.json reading with safetensors/GGUF header inspection
for a unified model summary.
"""

from dataclasses import dataclass, field
from pathlib import Path

import orjson

from hfutils.inspect.common import SafetensorsHeader, read_json_if_exists
from hfutils.inspect.gguf import GGUFInfo, read_gguf_header
from hfutils.inspect.safetensors import read_header

MODEL_EXTENSIONS = {".safetensors", ".gguf"}


@dataclass
class DirectoryInfo:
    path: Path
    config: dict | None = None
    model_files: list[Path] = field(default_factory=list)
    safetensors_headers: list[SafetensorsHeader] = field(default_factory=list)
    gguf_info: GGUFInfo | None = None
    sharded: bool = False
    shard_count: int = 0
    total_file_size: int = 0


def inspect_directory(path: Path) -> DirectoryInfo:
    """Inspect a model directory: config.json + model file headers."""
    path = Path(path)
    info = DirectoryInfo(path=path)

    info.config = read_json_if_exists(path / "config.json")

    # Find model files
    for f in sorted(path.iterdir()):
        if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS:
            info.model_files.append(f)
            info.total_file_size += f.stat().st_size

    # Check for sharding (diffusers uses diffusion_pytorch_model.*, transformers uses model.*)
    index_candidates = sorted(path.glob("*.safetensors.index.json"))
    if index_candidates:
        index = orjson.loads(index_candidates[0].read_bytes())
        shard_files = set(index.get("weight_map", {}).values())
        info.sharded = True
        info.shard_count = len(shard_files)

    # Read safetensors headers
    for f in info.model_files:
        if f.suffix.lower() == ".safetensors":
            info.safetensors_headers.append(read_header(f))
        elif f.suffix.lower() == ".gguf" and info.gguf_info is None:
            info.gguf_info = read_gguf_header(f)

    return info
