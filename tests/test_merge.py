"""Tests for merge command -- merging sharded safetensors into a single file."""

from pathlib import Path

import orjson
import pytest
import torch
from safetensors.torch import load_file, save_file

from hfutils.commands.merge import consolidate_component, merge_safetensors


def _create_sharded_model(
    tmp_dir: Path,
    shard_prefix: str = "model",
    index_name: str = "model.safetensors.index.json",
) -> dict[str, torch.Tensor]:
    """Create a fake sharded model with 2 shards and an index file."""
    shard1_tensors = {
        "layer.0.weight": torch.randn(4, 4),
        "layer.0.bias": torch.randn(4),
    }
    shard2_tensors = {
        "layer.1.weight": torch.randn(4, 4),
        "layer.1.bias": torch.randn(4),
    }

    shard1_name = f"{shard_prefix}-00001-of-00002.safetensors"
    shard2_name = f"{shard_prefix}-00002-of-00002.safetensors"
    save_file(shard1_tensors, tmp_dir / shard1_name)
    save_file(shard2_tensors, tmp_dir / shard2_name)

    weight_map = {}
    for name in shard1_tensors:
        weight_map[name] = shard1_name
    for name in shard2_tensors:
        weight_map[name] = shard2_name

    index = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    (tmp_dir / index_name).write_bytes(orjson.dumps(index))

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

        with pytest.raises(FileNotFoundError):
            merge_safetensors(input_dir, output_path)

    def test_discovers_diffusers_style_index(self, tmp_path):
        input_dir = tmp_path / "transformer"
        input_dir.mkdir()
        expected = _create_sharded_model(
            input_dir,
            shard_prefix="diffusion_pytorch_model",
            index_name="diffusion_pytorch_model.safetensors.index.json",
        )

        output_path = tmp_path / "merged.safetensors"
        merge_safetensors(input_dir, output_path)

        assert output_path.exists()
        merged = load_file(str(output_path))
        assert set(merged.keys()) == set(expected.keys())
        for key in expected:
            assert torch.equal(merged[key], expected[key]), f"Tensor mismatch for {key}"

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


class TestConsolidateComponent:
    def test_merges_sharded_directory(self, tmp_path):
        input_dir = tmp_path / "transformer"
        input_dir.mkdir()
        expected = _create_sharded_model(
            input_dir,
            shard_prefix="diffusion_pytorch_model",
            index_name="diffusion_pytorch_model.safetensors.index.json",
        )

        output_path = tmp_path / "out.safetensors"
        consolidate_component(input_dir, output_path)

        merged = load_file(str(output_path))
        assert set(merged.keys()) == set(expected.keys())

    def test_copies_single_file_directory(self, tmp_path):
        input_dir = tmp_path / "vae"
        input_dir.mkdir()
        tensors = {"encoder.weight": torch.randn(4, 4)}
        save_file(tensors, input_dir / "diffusion_pytorch_model.safetensors")

        output_path = tmp_path / "out" / "vae.safetensors"
        consolidate_component(input_dir, output_path)

        assert output_path.exists()
        loaded = load_file(str(output_path))
        assert torch.equal(loaded["encoder.weight"], tensors["encoder.weight"])

    def test_errors_on_empty_directory(self, tmp_path):
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            consolidate_component(input_dir, tmp_path / "out.safetensors")

    def test_errors_on_multiple_unindexed_files(self, tmp_path):
        input_dir = tmp_path / "ambiguous"
        input_dir.mkdir()
        save_file({"a.weight": torch.randn(2, 2)}, input_dir / "a.safetensors")
        save_file({"b.weight": torch.randn(2, 2)}, input_dir / "b.safetensors")
        with pytest.raises(FileNotFoundError):
            consolidate_component(input_dir, tmp_path / "out.safetensors")
