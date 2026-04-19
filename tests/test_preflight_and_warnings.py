"""Tests for disk-space preflight and metadata-conflict warnings."""

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from hfutils.cli import app
from hfutils.formats.safetensors import _merge_metadata, stream_merge
from hfutils.inspect.safetensors import read_raw_header
from hfutils.io.fs import InsufficientSpaceError, check_free_space

runner = CliRunner()


def _make_sharded(subdir: Path, metadata_per_shard: list[dict[str, str]]) -> list[Path]:
    paths = []
    for i, meta in enumerate(metadata_per_shard, 1):
        p = subdir / f"shard-{i:05d}.safetensors"
        save_file({f"t{i}": torch.randn(2, 2)}, p, metadata=meta)
        paths.append(p)
    return paths


class TestDiskSpaceCheck:
    def test_passes_with_plenty_of_space(self, tmp_path):
        # tmp_path has at least the tests themselves in free space; 1 KB trivially fits.
        check_free_space(tmp_path / "out.safetensors", 1024)

    def test_rejects_oversize_request(self, tmp_path):
        with pytest.raises(InsufficientSpaceError):
            check_free_space(tmp_path / "out.safetensors", 10**20)  # 100 exabytes

    def test_resolves_nonexistent_parent_to_existing_ancestor(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "out.safetensors"
        check_free_space(deep, 1024)  # does not raise; walks up to tmp_path

    def test_convert_single_refuses_insufficient_space(self, tmp_path):
        src = tmp_path / "in.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)

        def _fake_check(dest, required):
            raise InsufficientSpaceError(f"mocked: cannot fit {required} bytes")

        with patch("hfutils.commands.convert.check_free_space", _fake_check):
            result = runner.invoke(app, [
                "convert", "single", str(src), str(tmp_path / "out.safetensors"),
            ])
        assert result.exit_code == 1
        assert "mocked" in result.output
        assert not (tmp_path / "out.safetensors").exists()


class TestMetadataWarnings:
    def test_merge_metadata_records_conflict(self, tmp_path):
        shards = _make_sharded(tmp_path, [
            {"model_type": "v1", "common": "x"},
            {"model_type": "v2", "common": "x"},
        ])
        headers = [read_raw_header(p) for p in shards]

        merged, warnings = _merge_metadata(headers, shards)
        assert merged["model_type"] == "v2"  # last-write-wins
        assert merged["common"] == "x"
        assert any("model_type" in w for w in warnings)
        assert not any("common" in w for w in warnings)

    def test_merge_metadata_no_warnings_on_identical_values(self, tmp_path):
        shards = _make_sharded(tmp_path, [
            {"model_type": "same", "other": "y"},
            {"model_type": "same", "other": "y"},
        ])
        headers = [read_raw_header(p) for p in shards]
        _, warnings = _merge_metadata(headers, shards)
        assert warnings == []

    def test_stream_merge_surfaces_warning_via_callback(self, tmp_path):
        shards = _make_sharded(tmp_path, [
            {"key": "a"},
            {"key": "b"},
        ])
        seen: list[str] = []
        stream_merge(shards, tmp_path / "out.safetensors", on_warning=seen.append)
        assert any("key" in w for w in seen)

    def test_cli_shows_metadata_conflict_warning(self, tmp_path):
        shards = _make_sharded(tmp_path, [
            {"model_type": "a"},
            {"model_type": "b"},
        ])
        # Wrap shards into a sharded component dir
        (tmp_path / "model.safetensors.index.json").write_bytes(orjson.dumps({
            "metadata": {},
            "weight_map": {
                "t1": "shard-00001.safetensors",
                "t2": "shard-00002.safetensors",
            },
        }))
        for p in shards:
            p.rename(tmp_path / p.name)  # already in tmp_path; noop, but explicit

        result = runner.invoke(app, [
            "convert", "single", str(tmp_path), str(tmp_path / "merged.safetensors"),
        ])
        assert result.exit_code == 0, result.output
        assert "warn:" in result.output
        assert "model_type" in result.output
