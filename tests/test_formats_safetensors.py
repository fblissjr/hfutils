"""Streaming safetensors merge: correctness + memory behavior."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from hfutils.errors import HfutilsError, StreamMergeError
from hfutils.formats.safetensors import manifest_from_shards, stream_merge
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
        with pytest.raises(StreamMergeError, match="[Dd]uplicate"):
            stream_merge(shards, tmp_path / "out.safetensors")

    def test_duplicate_key_is_catchable_as_hfutils_error(self, tmp_path):
        """Library consumers can catch the base HfutilsError without picking subtypes."""
        t1 = {"w": torch.randn(2, 2)}
        t2 = {"w": torch.randn(2, 2)}
        shards = _write_shards(tmp_path, [t1, t2])
        with pytest.raises(HfutilsError):
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

    def test_malformed_header_raises_streammergeerror_not_valueerror(self, tmp_path):
        """read_raw_header raises a bare ValueError on a truncated/malformed
        header -- stream_merge must translate that into StreamMergeError so
        the CLI's cli_guard (which only catches HfutilsError/OSError) can
        surface it cleanly instead of a raw traceback."""
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"\x00\x00\x00")  # shorter than the 8-byte length prefix
        with pytest.raises(StreamMergeError):
            stream_merge([bad], tmp_path / "out.safetensors")

    def test_cleans_up_partial_output_on_failure(self, tmp_path):
        """A mid-copy failure (e.g. a shard truncated after the header was
        read) must not leave a truncated-but-header-valid file at the
        destination -- it should be removed, not left behind."""
        t1 = {"a": torch.randn(4, 4)}
        t2 = {"b": torch.randn(8, 8)}
        shards = _write_shards(tmp_path, [t1, t2])
        # Truncate the second shard's trailing bytes -- its header still
        # declares the original (larger) tensor data size.
        second = shards[1]
        size = second.stat().st_size
        with open(second, "r+b") as f:
            f.truncate(size - 10)

        out = tmp_path / "merged.safetensors"
        with pytest.raises(StreamMergeError):
            stream_merge(shards, out)

        assert not out.exists()

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

        # Observed peak is ~8 MiB (4 MiB read chunk + 4 MiB write buffer).
        # 10 MiB gives 25% headroom; still way under the 16 MiB loose threshold.
        assert peak < 10 * 1024 * 1024, f"peak Python allocation was {peak / 1024 / 1024:.1f} MiB"


class TestManifestFromShards:
    def test_malformed_header_raises_streammergeerror_not_valueerror(self, tmp_path):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"\x00\x00\x00")
        with pytest.raises(StreamMergeError):
            manifest_from_shards([bad])


class TestStreamMergeKeyPrefixNormalization:
    def test_normalizes_bare_qwen_vlm_text_encoder_prefixes(self, tmp_path):
        """Diffusers pipelines saving Qwen3_5Model / Qwen3VLModel via AutoModel
        emit bare `language_model.*` and `visual.*` keys without the outer
        `model.` prefix that ComfyUI's `detect_te_model` and
        `load_text_encoder_state_dicts` require."""
        t1 = {
            "language_model.embed_tokens.weight": torch.randn(4, 4),
            "language_model.layers.0.linear_attn.A_log": torch.randn(2),
            "language_model.layers.0.input_layernorm.weight": torch.randn(4),
        }
        t2 = {
            "visual.blocks.0.attn.qkv.weight": torch.randn(4, 4),
            "visual.patch_embed.proj.weight": torch.randn(4, 4),
            "visual.merger.linear_fc1.weight": torch.randn(4, 4),
        }
        shards = _write_shards(tmp_path, [t1, t2])
        out = tmp_path / "te.safetensors"

        manifest = stream_merge(shards, out)

        expected_keys = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.patch_embed.proj.weight",
            "model.visual.merger.linear_fc1.weight",
        }
        assert set(manifest.keys()) == expected_keys
        merged = load_file(str(out))
        assert set(merged.keys()) == expected_keys
        assert torch.equal(
            merged["model.language_model.layers.0.linear_attn.A_log"],
            t1["language_model.layers.0.linear_attn.A_log"],
        )
        assert torch.equal(
            merged["model.visual.patch_embed.proj.weight"],
            t2["visual.patch_embed.proj.weight"],
        )

    def test_already_prefixed_qwen_vlm_keys_are_untouched(self, tmp_path):
        t1 = {
            "model.language_model.layers.0.linear_attn.A_log": torch.randn(2),
            "model.visual.blocks.0.attn.qkv.weight": torch.randn(4, 4),
        }
        shards = _write_shards(tmp_path, [t1])
        out = tmp_path / "te.safetensors"

        manifest = stream_merge(shards, out)
        assert set(manifest.keys()) == set(t1.keys())

