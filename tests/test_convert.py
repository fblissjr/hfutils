"""CLI-level tests for `hfutils convert <src> --to <layout>`."""

import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from hfutils.cli import app
from tests.conftest import make_diffusers_pipeline as _make_pipeline
from tests.conftest import make_sharded_component as _make_sharded

runner = CliRunner()


class TestConvertToComfyui:
    def test_packs_full_pipeline(self, tmp_path):
        _make_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "comfyui",
            "--root", str(comfy), "--name", "MyModel",
        ])
        assert result.exit_code == 0, result.output

        merged = load_file(str(comfy / "diffusion_models/MyModel.safetensors"))
        assert set(merged.keys()) == {"a.weight", "a.bias", "b.weight", "b.bias"}
        assert (comfy / "vae/MyModel_vae.safetensors").exists()
        assert (comfy / "text_encoders/MyModel_te.safetensors").exists()

    def test_only_filter_repeatable(self, tmp_path):
        _make_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "comfyui",
            "--root", str(comfy), "--name", "X",
            "--only", "transformer", "--only", "vae",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy / "diffusion_models/X.safetensors").exists()
        assert (comfy / "vae/X_vae.safetensors").exists()
        assert not (comfy / "text_encoders").exists()

    def test_single_file_requires_as(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)

        result = runner.invoke(app, [
            "convert", str(f), "--to", "comfyui",
            "--root", str(tmp_path / "c"), "--name", "X",
        ])
        assert result.exit_code != 0
        assert "--as" in result.output

    def test_single_file_with_as(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", str(f), "--to", "comfyui",
            "--root", str(comfy), "--name", "X", "--as", "diffusion_model",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy / "diffusion_models/X.safetensors").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        _make_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "comfyui",
            "--root", str(comfy), "--name", "X", "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output
        assert not comfy.exists()

    def test_default_name_from_source_dir(self, tmp_path):
        pipeline_dir = tmp_path / "MyPipeline"
        pipeline_dir.mkdir()
        _make_pipeline(pipeline_dir)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", str(pipeline_dir), "--to", "comfyui",
            "--root", str(comfy), "--only", "transformer",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy / "diffusion_models/MyPipeline.safetensors").exists()

    def test_missing_root_errors_cleanly(self, tmp_path):
        _make_pipeline(tmp_path)
        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "comfyui", "--name", "X",
        ])
        assert result.exit_code != 0
        assert "--root" in result.output


class TestConvertToSingle:
    def test_merges_sharded_dir(self, tmp_path):
        src_dir = tmp_path / "sharded"
        src_dir.mkdir()
        expected = _make_sharded(src_dir)
        out = tmp_path / "merged.safetensors"

        result = runner.invoke(app, [
            "convert", str(src_dir), "--to", "single", "--out", str(out),
        ])
        assert result.exit_code == 0, result.output

        merged = load_file(str(out))
        assert set(merged.keys()) == set(expected.keys())
        for k, v in expected.items():
            assert torch.equal(merged[k], v)

    def test_copies_single_file(self, tmp_path):
        src = tmp_path / "in.safetensors"
        save_file({"w": torch.randn(3, 3)}, src)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", str(src), "--to", "single", "--out", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert torch.equal(load_file(str(out))["w"], load_file(str(src))["w"])

    def test_pipeline_without_component_errors(self, tmp_path):
        _make_pipeline(tmp_path)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "single", "--out", str(out),
        ])
        assert result.exit_code != 0
        assert "--component" in result.output

    def test_component_flag_merges_pipeline_subdir(self, tmp_path):
        _make_pipeline(tmp_path)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "single", "--out", str(out),
            "--component", "transformer",
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()
        merged = load_file(str(out))
        assert set(merged.keys()) == {"a.weight", "a.bias", "b.weight", "b.bias"}

    def test_unknown_component_errors(self, tmp_path):
        _make_pipeline(tmp_path)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "single", "--out", str(out),
            "--component", "bogus",
        ])
        assert result.exit_code != 0
        assert "bogus" in result.output

    def test_component_on_non_pipeline_errors(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", str(f), "--to", "single", "--out", str(out),
            "--component", "transformer",
        ])
        assert result.exit_code != 0

    def test_missing_out_errors_cleanly(self, tmp_path):
        src_dir = tmp_path / "sharded"
        src_dir.mkdir()
        _make_sharded(src_dir)

        result = runner.invoke(app, [
            "convert", str(src_dir), "--to", "single",
        ])
        assert result.exit_code != 0
        assert "--out" in result.output

    def test_dry_run_writes_nothing(self, tmp_path):
        src_dir = tmp_path / "sharded"
        src_dir.mkdir()
        _make_sharded(src_dir)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", str(src_dir), "--to", "single", "--out", str(out),
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output
        assert not out.exists()


class TestConvertValidation:
    def test_unrecognized_source(self, tmp_path):
        result = runner.invoke(app, [
            "convert", str(tmp_path / "nope"),
            "--to", "single", "--out", str(tmp_path / "out.safetensors"),
        ])
        assert result.exit_code != 0
        assert "unrecognized" in result.output

    def test_missing_to_flag(self, tmp_path):
        """--to is required."""
        result = runner.invoke(app, ["convert", str(tmp_path)])
        assert result.exit_code != 0

    def test_invalid_to_value_rejected_by_enum(self, tmp_path):
        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "bogus",
        ])
        assert result.exit_code != 0
