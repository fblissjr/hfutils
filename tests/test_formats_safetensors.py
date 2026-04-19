"""Streaming safetensors merge: correctness + memory behavior."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from hfutils.formats.safetensors import stream_merge
from hfutils.inspect.safetensors import read_raw_header


def _write_shards(tmp: Path, shards: list[dict[str, torch.Tensor]]) -> list[Path]:
    paths = []
    for i, tensors in enumerate(shards, 1):
        p = tmp / f"shard-{i:05d}.safetensors"
        save_file(tensors, p)
        paths.append(p)
    return paths


class TestStreamMerge:
    def test_merges_two_shards_preserves_values(self, tmp_path):
        t1 = {"a.weight": torch.randn(4, 4), "a.bias": torch.randn(4)}
        t2 = {"b.weight": torch.randn(4, 4), "b.bias": torch.randn(4)}
        shards = _write_shards(tmp_path, [t1, t2])

        out = tmp_path / "merged.safetensors"
        stream_merge(shards, out)

        merged = load_file(str(out))
        assert set(merged.keys()) == {"a.weight", "a.bias", "b.weight", "b.bias"}
        for k, v in {**t1, **t2}.items():
            assert torch.equal(merged[k], v), f"mismatch on {k}"

    def test_preserves_dtypes_and_shapes(self, tmp_path):
        t1 = {
            "f16": torch.randn(3, 3, dtype=torch.float16),
            "bf16": torch.randn(5, dtype=torch.bfloat16),
        }
        t2 = {
            "f32": torch.randn(2, 2, dtype=torch.float32),
            "i8": torch.randint(-5, 5, (4,), dtype=torch.int8),
        }
        shards = _write_shards(tmp_path, [t1, t2])

        out = tmp_path / "merged.safetensors"
        stream_merge(shards, out)

        merged = load_file(str(out))
        assert merged["f16"].dtype == torch.float16
        assert merged["bf16"].dtype == torch.bfloat16
        assert merged["f32"].dtype == torch.float32
        assert merged["i8"].dtype == torch.int8

    def test_single_shard_is_just_a_copy(self, tmp_path):
        t = {"w": torch.randn(8, 8)}
        shards = _write_shards(tmp_path, [t])
        out = tmp_path / "merged.safetensors"
        stream_merge(shards, out)

        merged = load_file(str(out))
        assert torch.equal(merged["w"], t["w"])

    def test_rejects_duplicate_keys_across_shards(self, tmp_path):
        t1 = {"w": torch.randn(2, 2)}
        t2 = {"w": torch.randn(2, 2)}
        shards = _write_shards(tmp_path, [t1, t2])
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            stream_merge(shards, tmp_path / "out.safetensors")

    def test_preserves_metadata_when_consistent(self, tmp_path):
        # Write a single-shard safetensors with metadata
        t = {"w": torch.randn(2, 2)}
        save_file(t, tmp_path / "a.safetensors", metadata={"model": "test"})
        out = tmp_path / "out.safetensors"
        stream_merge([tmp_path / "a.safetensors"], out)

        header = read_raw_header(out)
        assert header.metadata.get("model") == "test"

    def test_output_is_valid_safetensors_layout(self, tmp_path):
        t1 = {"a": torch.randn(4, 4)}
        t2 = {"b": torch.randn(4, 4)}
        shards = _write_shards(tmp_path, [t1, t2])
        out = tmp_path / "merged.safetensors"
        stream_merge(shards, out)

        header = read_raw_header(out)
        # Every tensor's data must fit inside the file
        total_data = out.stat().st_size - header.data_region_start
        for entry in header.tensors:
            assert entry.data_offset_start <= entry.data_offset_end <= total_data
        # Offsets must be contiguous and ascending (no gaps, no overlap)
        cursor = 0
        for entry in header.tensors:
            assert entry.data_offset_start == cursor
            cursor = entry.data_offset_end
        assert cursor == total_data

    def test_peak_python_allocations_bounded(self, tmp_path):
        # Stream-merging a 64 MiB shard must not trigger Python allocations
        # anywhere near the tensor size. tracemalloc scopes to Python heap
        # (unlike RSS, which is process-wide and depends on cold state).
        import tracemalloc

        big = torch.zeros(4096, 4096, dtype=torch.float32)  # 64 MiB on disk
        save_file({"w": big}, tmp_path / "a.safetensors")
        del big

        tracemalloc.start()
        stream_merge([tmp_path / "a.safetensors"], tmp_path / "out.safetensors")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 16 * 1024 * 1024, f"peak Python allocation was {peak / 1024 / 1024:.1f} MiB"
