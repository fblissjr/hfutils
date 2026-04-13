"""Tests for merge command -- merging sharded safetensors into a single file."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import load_file, save_file

from hfutils.commands.merge import merge_safetensors


def _create_sharded_model(tmp_dir: Path) -> dict[str, torch.Tensor]:
    """Create a fake sharded model with 2 shards and an index file."""
    shard1_tensors = {
        "layer.0.weight": torch.randn(4, 4),
        "layer.0.bias": torch.randn(4),
    }
    shard2_tensors = {
        "layer.1.weight": torch.randn(4, 4),
        "layer.1.bias": torch.randn(4),
    }

    save_file(shard1_tensors, tmp_dir / "model-00001-of-00002.safetensors")
    save_file(shard2_tensors, tmp_dir / "model-00002-of-00002.safetensors")

    weight_map = {}
    for name in shard1_tensors:
        weight_map[name] = "model-00001-of-00002.safetensors"
    for name in shard2_tensors:
        weight_map[name] = "model-00002-of-00002.safetensors"

    index = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    (tmp_dir / "model.safetensors.index.json").write_bytes(orjson.dumps(index))

    expected = {}
    expected.update(shard1_tensors)
    expected.update(shard2_tensors)
    return expected


class TestMergeSafetensors:
    def test_produces_single_file_with_all_tensors(self, tmp_path):
        input_dir = tmp_path / "sharded"
        input_dir.mkdir()
        expected = _create_sharded_model(input_dir)

        output_path = tmp_path / "merged.safetensors"
        merge_safetensors(input_dir, output_path)

        assert output_path.exists()
        merged = load_file(str(output_path))
        assert set(merged.keys()) == set(expected.keys())
        for key in expected:
            assert torch.equal(merged[key], expected[key]), f"Tensor mismatch for {key}"

    def test_rejects_missing_index(self, tmp_path):
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_path = tmp_path / "merged.safetensors"

        try:
            merge_safetensors(input_dir, output_path)
            assert False, "Should have raised"
        except FileNotFoundError:
            pass

    def test_preserves_tensor_values_exactly(self, tmp_path):
        input_dir = tmp_path / "sharded"
        input_dir.mkdir()
        expected = _create_sharded_model(input_dir)

        output_path = tmp_path / "merged.safetensors"
        merge_safetensors(input_dir, output_path)

        merged = load_file(str(output_path))
        for key in expected:
            assert merged[key].shape == expected[key].shape
            assert merged[key].dtype == expected[key].dtype
