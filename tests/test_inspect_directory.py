"""Tests for directory-level model inspection."""

import struct
from pathlib import Path

import numpy as np
import orjson

from hfutils.inspect.directory import inspect_directory, DirectoryInfo


def _make_safetensors(path: Path, tensors: dict | None = None) -> None:
    """Create a minimal safetensors file."""
    if tensors is None:
        tensors = {"weight": {"dtype": "BF16", "shape": [1024, 1024], "data_offsets": [0, 2097152]}}
    hb = orjson.dumps(tensors)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)


def _make_gguf(path: Path, arch: str = "llama") -> None:
    """Create a minimal GGUF file using the writer."""
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


class TestInspectDirectory:
    def test_safetensors_with_config(self, tmp_path):
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
        (tmp_path / "config.json").write_bytes(orjson.dumps(config))
        _make_safetensors(tmp_path / "model.safetensors")

        info = inspect_directory(tmp_path)
        assert isinstance(info, DirectoryInfo)
        assert info.config is not None
        assert info.config["model_type"] == "llama"
        assert len(info.model_files) == 1
        assert info.model_files[0].name == "model.safetensors"

    def test_sharded_safetensors(self, tmp_path):
        _make_safetensors(tmp_path / "model-00001-of-00002.safetensors")
        _make_safetensors(tmp_path / "model-00002-of-00002.safetensors")
        index = {"metadata": {}, "weight_map": {
            "w1": "model-00001-of-00002.safetensors",
            "w2": "model-00002-of-00002.safetensors",
        }}
        (tmp_path / "model.safetensors.index.json").write_bytes(orjson.dumps(index))

        info = inspect_directory(tmp_path)
        assert info.sharded is True
        assert info.shard_count == 2

    def test_gguf_directory(self, tmp_path):
        _make_gguf(tmp_path / "model.gguf")

        info = inspect_directory(tmp_path)
        assert len(info.model_files) == 1
        assert info.gguf_info is not None
        assert info.gguf_info.architecture == "llama"

    def test_no_model_files(self, tmp_path):
        (tmp_path / "readme.md").write_text("hello")

        info = inspect_directory(tmp_path)
        assert len(info.model_files) == 0

    def test_config_without_model_files(self, tmp_path):
        config = {"model_type": "bert"}
        (tmp_path / "config.json").write_bytes(orjson.dumps(config))

        info = inspect_directory(tmp_path)
        assert info.config is not None
        assert info.config["model_type"] == "bert"
        assert len(info.model_files) == 0

    def test_total_size(self, tmp_path):
        _make_safetensors(tmp_path / "model.safetensors")
        info = inspect_directory(tmp_path)
        assert info.total_file_size > 0
