"""CLI-level tests for `hfutils lora convert` and `hfutils lora schemes`."""

import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from hfutils.cli import app
from tests.conftest import make_fal_krea2_lora

runner = CliRunner()


class TestLoraConvert:
    def test_default_output_name_and_keys(self, tmp_path):
        src = tmp_path / "Arco.safetensors"
        src_keys = make_fal_krea2_lora(src)

        result = runner.invoke(app, ["lora", "convert", str(src)])
        assert result.exit_code == 0, result.output

        out = tmp_path / "Arco_comfy.safetensors"
        assert out.exists()
        loaded = load_file(str(out))
        assert all(k.startswith("transformer.") for k in loaded)
        assert len(loaded) == len(src_keys)

    def test_explicit_output_path(self, tmp_path):
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)
        dst = tmp_path / "renamed.safetensors"

        result = runner.invoke(app, ["lora", "convert", str(src), "-o", str(dst)])
        assert result.exit_code == 0, result.output
        assert dst.exists()

    def test_verify_flag(self, tmp_path):
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)

        result = runner.invoke(app, ["lora", "convert", str(src), "--verify"])
        assert result.exit_code == 0, result.output
        assert "Verified" in result.output

    def test_dry_run_writes_nothing(self, tmp_path):
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)

        result = runner.invoke(app, ["lora", "convert", str(src), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert not (tmp_path / "fal_comfy.safetensors").exists()

    def test_auto_detect_fails_on_unrelated_file(self, tmp_path):
        src = tmp_path / "plain.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)

        result = runner.invoke(app, ["lora", "convert", str(src)])
        assert result.exit_code != 0
        assert "could not detect" in result.output

    def test_unknown_scheme_rejected(self, tmp_path):
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)

        result = runner.invoke(app, ["lora", "convert", str(src), "--scheme", "nope"])
        assert result.exit_code != 0
        assert "unknown scheme" in result.output

    def test_scheme_mismatch_requires_force(self, tmp_path):
        src = tmp_path / "plain.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)

        result = runner.invoke(app, ["lora", "convert", str(src), "--scheme", "krea2"])
        assert result.exit_code != 0
        assert "--force" in result.output

    def test_force_converts_mismatched_file(self, tmp_path):
        src = tmp_path / "plain.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)

        result = runner.invoke(app, ["lora", "convert", str(src), "--scheme", "krea2", "--force"])
        assert result.exit_code == 0, result.output

    def test_missing_input(self, tmp_path):
        result = runner.invoke(app, ["lora", "convert", str(tmp_path / "nope.safetensors")])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestLoraSchemes:
    def test_lists_krea2(self):
        result = runner.invoke(app, ["lora", "schemes"])
        assert result.exit_code == 0, result.output
        assert "krea2" in result.output
