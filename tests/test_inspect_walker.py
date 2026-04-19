"""Tests for inspect/walker.py. The `inspect --recursive` CLI is covered
separately in tests/test_inspect_recursive.py -- this module tests the
walker primitive in isolation."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import save_file

from hfutils.inspect.walker import walk_for_models
from hfutils.sources.detect import SourceKind


def _make_flat_model(parent: Path, name: str) -> Path:
    d = parent / name
    d.mkdir()
    save_file({"w": torch.randn(2, 2)}, d / "model.safetensors")
    (d / "config.json").write_bytes(orjson.dumps({"model_type": "test"}))
    return d


def _make_hf_cache(parent: Path, org: str, name: str) -> Path:
    snap = parent / f"models--{org}--{name}" / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    save_file({"w": torch.randn(2, 2)}, snap / "model.safetensors")
    return snap


class TestWalkForModels:
    def test_flat_layout(self, tmp_path):
        _make_flat_model(tmp_path, "alpha")
        _make_flat_model(tmp_path, "beta")

        found = walk_for_models(tmp_path)
        names = [n for n, _ in found]
        assert names == ["alpha", "beta"]
        assert all(src.kind == SourceKind.COMPONENT_DIR for _, src in found)

    def test_hf_cache_layout(self, tmp_path):
        _make_hf_cache(tmp_path, "author", "model-a")

        found = walk_for_models(tmp_path)
        assert len(found) == 1
        name, _ = found[0]
        assert name == "author/model-a"

    def test_mixed_layouts(self, tmp_path):
        _make_flat_model(tmp_path, "flat")
        _make_hf_cache(tmp_path, "org", "cached")

        found = walk_for_models(tmp_path)
        names = {n for n, _ in found}
        assert names == {"flat", "org/cached"}

    def test_empty_tree(self, tmp_path):
        assert walk_for_models(tmp_path) == []

    def test_skips_non_model_children(self, tmp_path):
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "readme.md").write_text("hi")
        _make_flat_model(tmp_path, "model")

        found = walk_for_models(tmp_path)
        assert [n for n, _ in found] == ["model"]

    def test_many_entries_deterministic_order(self, tmp_path):
        # Enough entries to exercise thread-pool completion jitter.
        names = [f"model_{i:03d}" for i in range(20)]
        for n in names:
            _make_flat_model(tmp_path, n)

        found = walk_for_models(tmp_path)
        assert [n for n, _ in found] == sorted(names)
