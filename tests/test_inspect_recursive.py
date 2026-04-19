"""CLI-level tests for `hfutils inspect --recursive` (replaces the old `scan` command)."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from hfutils.cli import app

runner = CliRunner()


def _make_flat_model(parent: Path, name: str) -> Path:
    d = parent / name
    d.mkdir()
    save_file({"w": torch.randn(2, 2)}, d / "model.safetensors")
    (d / "config.json").write_bytes(orjson.dumps({"model_type": "test"}))
    return d


class TestInspectRecursive:
    def test_finds_flat_models(self, tmp_path):
        _make_flat_model(tmp_path, "model_a")
        _make_flat_model(tmp_path, "model_b")

        result = runner.invoke(app, ["inspect", str(tmp_path), "--recursive"])
        assert result.exit_code == 0, result.output
        assert "model_a" in result.output
        assert "model_b" in result.output

    def test_reports_no_models_found(self, tmp_path):
        result = runner.invoke(app, ["inspect", str(tmp_path), "--recursive"])
        assert result.exit_code == 0
        assert "No models found" in result.output

    def test_requires_directory(self, tmp_path):
        f = tmp_path / "one.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        result = runner.invoke(app, ["inspect", str(f), "--recursive"])
        assert result.exit_code != 0

    def test_finds_hf_cache_layout(self, tmp_path):
        snap = tmp_path / "models--author--name" / "snapshots" / "abcdef"
        snap.mkdir(parents=True)
        save_file({"w": torch.randn(2, 2)}, snap / "model.safetensors")

        result = runner.invoke(app, ["inspect", str(tmp_path), "--recursive"])
        assert result.exit_code == 0, result.output
        assert "author/name" in result.output
