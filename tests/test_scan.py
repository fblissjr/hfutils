"""Tests for scan command -- auditing local model directories."""

import struct
from pathlib import Path

import orjson

from hfutils.commands.scan import scan_directory, ModelEntry


def _make_safetensors_file(path: Path, size: int = 100) -> None:
    """Create a minimal safetensors file of approximately the given size."""
    header = {"w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
    hb = orjson.dumps(header)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * max(2, size - 8 - len(hb)))


def _make_gguf_file(path: Path) -> None:
    """Create a minimal GGUF file."""
    import numpy as np
    from gguf import GGUFWriter
    writer = GGUFWriter(path, "llama")
    writer.add_context_length(4096)
    t = np.zeros((2, 2), dtype=np.float32)
    writer.add_tensor("test", t)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestScanDirectory:
    def test_finds_single_safetensors_model(self, tmp_path):
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        _make_safetensors_file(model_dir / "model.safetensors")
        (model_dir / "config.json").write_bytes(orjson.dumps({"model_type": "llama"}))

        results = scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0].name == "my-model"
        assert results[0].format == "safetensors"
        assert results[0].sharded is False
        assert results[0].has_config is True

    def test_finds_sharded_model(self, tmp_path):
        model_dir = tmp_path / "big-model"
        model_dir.mkdir()
        _make_safetensors_file(model_dir / "model-00001-of-00002.safetensors")
        _make_safetensors_file(model_dir / "model-00002-of-00002.safetensors")
        index = {"metadata": {}, "weight_map": {
            "w1": "model-00001-of-00002.safetensors",
            "w2": "model-00002-of-00002.safetensors",
        }}
        (model_dir / "model.safetensors.index.json").write_bytes(orjson.dumps(index))

        results = scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0].sharded is True
        assert results[0].file_count == 2

    def test_finds_gguf_model(self, tmp_path):
        model_dir = tmp_path / "gguf-model"
        model_dir.mkdir()
        _make_gguf_file(model_dir / "model.gguf")

        results = scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0].format == "gguf"

    def test_detects_incomplete_sharded(self, tmp_path):
        model_dir = tmp_path / "broken"
        model_dir.mkdir()
        _make_safetensors_file(model_dir / "model-00001-of-00002.safetensors")
        # Missing shard 2
        index = {"metadata": {}, "weight_map": {
            "w1": "model-00001-of-00002.safetensors",
            "w2": "model-00002-of-00002.safetensors",
        }}
        (model_dir / "model.safetensors.index.json").write_bytes(orjson.dumps(index))

        results = scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0].incomplete is True

    def test_finds_multiple_models(self, tmp_path):
        for name in ["model-a", "model-b"]:
            d = tmp_path / name
            d.mkdir()
            _make_safetensors_file(d / "model.safetensors")

        results = scan_directory(tmp_path)
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"model-a", "model-b"}

    def test_hf_cache_layout(self, tmp_path):
        """Detect models in HF cache structure: models--org--name/snapshots/hash/"""
        cache_dir = tmp_path / "models--org--cool-model" / "snapshots" / "abc123"
        cache_dir.mkdir(parents=True)
        _make_safetensors_file(cache_dir / "model.safetensors")

        results = scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0].name == "org/cool-model"

    def test_empty_directory(self, tmp_path):
        results = scan_directory(tmp_path)
        assert len(results) == 0

    def test_ignores_non_model_dirs(self, tmp_path):
        d = tmp_path / "not-a-model"
        d.mkdir()
        (d / "readme.txt").write_text("hello")

        results = scan_directory(tmp_path)
        assert len(results) == 0
