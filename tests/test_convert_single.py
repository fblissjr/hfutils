"""CLI-level tests for `hfutils convert single`."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from hfutils.cli import app

runner = CliRunner()


def _make_sharded(subdir: Path) -> dict[str, torch.Tensor]:
    t1 = {"a.weight": torch.randn(4, 4), "a.bias": torch.randn(4)}
    t2 = {"b.weight": torch.randn(4, 4), "b.bias": torch.randn(4)}
    save_file(t1, subdir / "model-00001-of-00002.safetensors")
    save_file(t2, subdir / "model-00002-of-00002.safetensors")
    (subdir / "model.safetensors.index.json").write_bytes(orjson.dumps({
        "metadata": {},
        "weight_map": {
            **{k: "model-00001-of-00002.safetensors" for k in t1},
            **{k: "model-00002-of-00002.safetensors" for k in t2},
        },
    }))
    return {**t1, **t2}


class TestConvertSingle:
    def test_merges_sharded_dir(self, tmp_path):
        src_dir = tmp_path / "sharded"
        src_dir.mkdir()
        expected = _make_sharded(src_dir)
        out = tmp_path / "merged.safetensors"

        result = runner.invoke(app, ["convert", "single", str(src_dir), str(out)])
        assert result.exit_code == 0, result.output

        merged = load_file(str(out))
        assert set(merged.keys()) == set(expected.keys())
        for k, v in expected.items():
            assert torch.equal(merged[k], v)

    def test_copies_single_file(self, tmp_path):
        src = tmp_path / "in.safetensors"
        tensors = {"w": torch.randn(3, 3)}
        save_file(tensors, src)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, ["convert", "single", str(src), str(out)])
        assert result.exit_code == 0, result.output

        merged = load_file(str(out))
        assert torch.equal(merged["w"], tensors["w"])

    def test_pipeline_source_without_component_errors(self, tmp_path):
        (tmp_path / "model_index.json").write_bytes(orjson.dumps({"_class_name": "P"}))
        (tmp_path / "transformer").mkdir()
        save_file({"w": torch.randn(2, 2)}, tmp_path / "transformer/model.safetensors")
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, ["convert", "single", str(tmp_path), str(out)])
        assert result.exit_code != 0
        assert "--component" in result.output

    def test_dry_run_writes_nothing(self, tmp_path):
        src_dir = tmp_path / "sharded"
        src_dir.mkdir()
        _make_sharded(src_dir)
        out = tmp_path / "merged.safetensors"

        result = runner.invoke(app, ["convert", "single", str(src_dir), str(out), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert not out.exists()

    def test_missing_source(self, tmp_path):
        result = runner.invoke(app, [
            "convert", "single", str(tmp_path / "nope"), str(tmp_path / "out.safetensors"),
        ])
        assert result.exit_code != 0


class TestConvertSingleComponent:
    def _make_pipeline(self, tmp: Path) -> None:
        (tmp / "model_index.json").write_bytes(orjson.dumps({
            "_class_name": "P",
            "transformer": ["diffusers", "T"],
            "vae": ["diffusers", "V"],
        }))
        (tmp / "transformer").mkdir()
        _make_sharded(tmp / "transformer")
        (tmp / "vae").mkdir()
        save_file({"w": torch.randn(2, 2)}, tmp / "vae/diffusion_pytorch_model.safetensors")

    def test_picks_component_from_pipeline(self, tmp_path):
        self._make_pipeline(tmp_path)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", "single", str(tmp_path), str(out),
            "--component", "transformer",
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()
        merged = load_file(str(out))
        assert set(merged.keys()) == {"a.weight", "a.bias", "b.weight", "b.bias"}

    def test_pipeline_without_component_errors_and_lists_available(self, tmp_path):
        self._make_pipeline(tmp_path)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", "single", str(tmp_path), str(out),
        ])
        assert result.exit_code != 0
        assert "--component" in result.output
        assert "transformer" in result.output
        assert "vae" in result.output

    def test_unknown_component_errors(self, tmp_path):
        self._make_pipeline(tmp_path)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", "single", str(tmp_path), str(out),
            "--component", "bogus",
        ])
        assert result.exit_code != 0
        assert "bogus" in result.output

    def test_component_on_non_pipeline_errors(self, tmp_path):
        src = tmp_path / "flat.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)

        result = runner.invoke(app, [
            "convert", "single", str(src), str(tmp_path / "out.safetensors"),
            "--component", "transformer",
        ])
        assert result.exit_code != 0
        assert "pipeline" in result.output.lower()
