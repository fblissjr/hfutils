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


# Krea-2 fal LoRA (PEFT naming) synthetic fixture ---------------------------
# Mirrors the key structure fal.ai's Krea-2 trainer emits, so tests exercise
# every rename branch without shipping a real 200 MB LoRA.

_KREA2_ATTN = ["wq", "wk", "wv", "wo", "gate"]
_KREA2_MLP = ["up", "down", "gate"]


def _krea2_block_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    for a in _KREA2_ATTN:
        keys += [f"{prefix}.attn.{a}.lora_A.weight", f"{prefix}.attn.{a}.lora_B.weight"]
    for m in _KREA2_MLP:
        keys += [f"{prefix}.mlp.{m}.lora_A.weight", f"{prefix}.mlp.{m}.lora_B.weight"]
    return keys


def fal_krea2_lora_keys(n_blocks: int = 2) -> list[str]:
    """The full set of PEFT-style keys a fal Krea-2 LoRA contains."""
    p = "base_model.model"
    keys: list[str] = []
    for i in range(n_blocks):
        keys += _krea2_block_keys(f"{p}.blocks.{i}")
    for single in ["first", "last.linear", "tmlp.0", "tmlp.2", "tproj.1", "txtmlp.1", "txtmlp.3"]:
        keys += [f"{p}.{single}.lora_A.weight", f"{p}.{single}.lora_B.weight"]
    keys += [f"{p}.txtfusion.projector.lora_A.weight", f"{p}.txtfusion.projector.lora_B.weight"]
    for i in (0, 1):
        keys += _krea2_block_keys(f"{p}.txtfusion.layerwise_blocks.{i}")
        keys += _krea2_block_keys(f"{p}.txtfusion.refiner_blocks.{i}")
    return keys


def make_fal_krea2_lora(
    path: Path, *, n_blocks: int = 2, metadata: dict[str, str] | None = None,
) -> list[str]:
    """Write a synthetic fal Krea-2 LoRA to `path`; return its source keys."""
    keys = fal_krea2_lora_keys(n_blocks)
    tensors = {k: torch.randn(2, 2) for k in keys}
    save_file(tensors, path, metadata=metadata or {"base_model": "fal/krea-2", "lora_rank": "32"})
    return keys


def fal_module_to_base_key(module_with_suffix: str) -> str:
    """`base_model.model.X.lora_A.weight` -> native base weight `X.weight`."""
    inner = module_with_suffix[len("base_model.model."):]
    inner = inner.rsplit(".lora_", 1)[0]
    return inner + ".weight"


def make_krea2_base(path: Path, *, n_blocks: int = 2, dtype=torch.bfloat16) -> dict[str, "torch.Tensor"]:
    """Write a synthetic Krea-2 base checkpoint (native names) that the LoRA from
    `make_fal_krea2_lora` fuses into. Includes a couple of non-LoRA tensors
    (norms) to verify they pass through unchanged. Returns the tensor dict."""
    base_keys = {fal_module_to_base_key(k) for k in fal_krea2_lora_keys(n_blocks)}
    tensors = {k: torch.randn(2, 2).to(dtype) for k in base_keys}
    # Non-LoRA weights the fuse must copy verbatim.
    for i in range(n_blocks):
        tensors[f"blocks.{i}.prenorm.scale"] = torch.randn(2).to(dtype)
    save_file(tensors, path)
    return tensors


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
