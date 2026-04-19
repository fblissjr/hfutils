"""Tests for the CLI hygiene pass: --version, repeatable --only/--skip, --as Enum."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from hfutils import __version__
from hfutils.cli import app

runner = CliRunner()


def _make_pipeline(tmp: Path) -> None:
    (tmp / "model_index.json").write_bytes(orjson.dumps({
        "_class_name": "P",
        "transformer": ["diffusers", "T"],
        "vae": ["diffusers", "V"],
        "text_encoder": ["transformers", "E"],
    }))
    for name in ("transformer", "vae", "text_encoder"):
        (tmp / name).mkdir()
        save_file({"w": torch.randn(2, 2)}, tmp / name / "model.safetensors")


class TestVersionFlag:
    def test_version_flag_prints_and_exits(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestRepeatableOnly:
    def test_only_accepts_multiple_flags(self, tmp_path):
        _make_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "comfyui",
            "--root", str(comfy), "--name", "X",
            "--only", "transformer",
            "--only", "vae",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy / "diffusion_models/X.safetensors").exists()
        assert (comfy / "vae/X_vae.safetensors").exists()
        assert not (comfy / "text_encoders").exists()

    def test_skip_accepts_multiple_flags(self, tmp_path):
        _make_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "comfyui",
            "--root", str(comfy), "--name", "X",
            "--skip", "text_encoder",
            "--skip", "vae",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy / "diffusion_models/X.safetensors").exists()
        assert not (comfy / "vae").exists()
        assert not (comfy / "text_encoders").exists()


class TestAsEnum:
    def test_invalid_as_value_rejected_at_cli_layer(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)

        result = runner.invoke(app, [
            "convert", str(f), "--to", "comfyui",
            "--root", str(tmp_path / "c"), "--name", "X", "--as", "bogus",
        ])
        assert result.exit_code != 0
        # Typer validation error lists valid values
        assert "diffusion_model" in result.output or "Invalid value" in result.output

    def test_valid_as_value_accepted(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", str(f), "--to", "comfyui",
            "--root", str(comfy), "--name", "X", "--as", "vae",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy / "vae/X.safetensors").exists()


class TestUnifiedDryRun:
    def test_single_dry_run_shows_plan_header(self, tmp_path):
        src = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)

        result = runner.invoke(app, [
            "convert", str(src), "--to", "single",
            "--out", str(tmp_path / "out.safetensors"),
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output
        assert "Plan: 1 operation" in result.output

    def test_comfyui_dry_run_shows_plan_header(self, tmp_path):
        _make_pipeline(tmp_path)
        result = runner.invoke(app, [
            "convert", str(tmp_path), "--to", "comfyui",
            "--root", str(tmp_path / "c"),
            "--name", "X", "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output
        assert "Plan:" in result.output
