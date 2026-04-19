"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import orjson
import torch
from safetensors.torch import save_file


def make_gguf_file(path: Path, arch: str = "llama") -> None:
    """Create a minimal GGUF file using the gguf library's writer."""
    from gguf import GGUFWriter

    writer = GGUFWriter(path, arch)
    writer.add_context_length(4096)
    writer.add_embedding_length(2048)
    t = np.zeros((2, 2), dtype=np.float32)
    writer.add_tensor("test", t)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def make_sharded_component(
    dir_path: Path,
    *,
    shard_prefix: str = "shard",
    index_name: str = "model.safetensors.index.json",
) -> dict[str, torch.Tensor]:
    """Create a two-shard safetensors component with an index.

    Returns the combined {name: tensor} dict so callers that need round-trip
    assertions can compare. Two shards is the sweet spot -- big enough to
    exercise the merge path, small enough to be cheap.
    """
    t1 = {"a.weight": torch.randn(4, 4), "a.bias": torch.randn(4)}
    t2 = {"b.weight": torch.randn(4, 4), "b.bias": torch.randn(4)}
    s1 = f"{shard_prefix}-00001-of-00002.safetensors"
    s2 = f"{shard_prefix}-00002-of-00002.safetensors"
    save_file(t1, dir_path / s1)
    save_file(t2, dir_path / s2)
    (dir_path / index_name).write_bytes(orjson.dumps({
        "metadata": {},
        "weight_map": {**{k: s1 for k in t1}, **{k: s2 for k in t2}},
    }))
    return {**t1, **t2}


def make_diffusers_pipeline(tmp: Path) -> None:
    """Create a diffusers pipeline dir with transformer (sharded), vae, text_encoder.

    Standard shape used by many convert/inspect tests. Components:
    - transformer/: sharded via diffusion_pytorch_model
    - vae/: single-file
    - text_encoder/: single-file
    """
    (tmp / "model_index.json").write_bytes(orjson.dumps({
        "_class_name": "P",
        "transformer": ["diffusers", "T"],
        "vae": ["diffusers", "V"],
        "text_encoder": ["transformers", "E"],
    }))
    (tmp / "transformer").mkdir()
    (tmp / "transformer/config.json").write_bytes(orjson.dumps({"_class_name": "T"}))
    make_sharded_component(
        tmp / "transformer",
        shard_prefix="diffusion_pytorch_model",
        index_name="diffusion_pytorch_model.safetensors.index.json",
    )
    (tmp / "vae").mkdir()
    save_file({"enc.w": torch.randn(3, 3)}, tmp / "vae/diffusion_pytorch_model.safetensors")
    (tmp / "text_encoder").mkdir()
    save_file({"emb.w": torch.randn(3, 3)}, tmp / "text_encoder/model.safetensors")
