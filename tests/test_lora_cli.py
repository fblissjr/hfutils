"""CLI-level tests for `hfutils lora convert` and `hfutils lora schemes`."""

import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from hfutils.cli import app
from tests.conftest import make_fal_krea2_lora, make_krea2_base

runner = CliRunner()


class TestLoraConvert:
    def test_default_output_name_and_keys(self, tmp_path):
        src = tmp_path / "MyStyle.safetensors"
        src_keys = make_fal_krea2_lora(src)

        result = runner.invoke(app, ["lora", "convert", str(src)])
        assert result.exit_code == 0, result.output

        out = tmp_path / "MyStyle_comfy.safetensors"
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

    def test_verify_mismatch_exits_2(self, tmp_path, monkeypatch):
        """--verify's failure exit code (2) is distinct from every other
        error path (1) and must survive the cli_guard refactor untouched."""
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)

        monkeypatch.setattr(
            "hfutils.cli_support.verify_output",
            lambda dest, manifest: (False, "forced mismatch"),
        )

        result = runner.invoke(app, ["lora", "convert", str(src), "--verify"])
        assert result.exit_code == 2, result.output
        assert "forced mismatch" in result.output

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

    def test_malformed_header_reported_cleanly(self, tmp_path):
        """A file that exists but is too short to be a real safetensors
        file must exit cleanly, not crash with a raw traceback -- exercises
        the shared read_raw_header_or_raise path now that this command
        relies on cli_guard instead of a local try/except."""
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"\x00\x00\x00")

        result = runner.invoke(app, ["lora", "convert", str(bad)])

        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "Error" in result.output


class TestLoraSchemes:
    def test_lists_krea2(self):
        result = runner.invoke(app, ["lora", "schemes"])
        assert result.exit_code == 0, result.output
        assert "krea2" in result.output


class TestLoraFuse:
    def test_fuse_writes_checkpoint(self, tmp_path):
        lora = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora)
        base = tmp_path / "base.safetensors"
        make_krea2_base(base)
        out = tmp_path / "fused.safetensors"

        result = runner.invoke(app, ["lora", "fuse", str(lora), str(base), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert "Fused" in result.output

    def test_fuse_missing_base(self, tmp_path):
        lora = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora)
        result = runner.invoke(app, [
            "lora", "fuse", str(lora), str(tmp_path / "nope.safetensors"),
            "-o", str(tmp_path / "out.safetensors"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_fuse_malformed_lora_header_reported_cleanly(self, tmp_path):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"\x00\x00\x00")
        base = tmp_path / "base.safetensors"
        make_krea2_base(base)

        result = runner.invoke(app, [
            "lora", "fuse", str(bad), str(base), "-o", str(tmp_path / "out.safetensors"),
        ])

        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "Error" in result.output

    def test_fuse_refuses_insufficient_space(self, tmp_path, monkeypatch):
        """fuse() writes a full base-checkpoint-sized output with no
        preflight check at all -- must gain the same check_free_space guard
        `convert` already has."""
        from hfutils.errors import InsufficientSpaceError

        lora = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora)
        base = tmp_path / "base.safetensors"
        make_krea2_base(base)
        out = tmp_path / "fused.safetensors"

        def _fake_check(dest, required):
            raise InsufficientSpaceError(f"mocked: cannot fit {required} bytes")

        monkeypatch.setattr("hfutils.commands.lora.check_free_space", _fake_check)

        result = runner.invoke(app, ["lora", "fuse", str(lora), str(base), "-o", str(out)])
        assert result.exit_code == 1
        assert "mocked" in result.output
        assert not out.exists()
