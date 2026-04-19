"""Tests for shard integrity detection, --verify, and new architecture rules."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from hfutils.cli import app
from hfutils.inspect.architecture import detect_architecture
from hfutils.sources.detect import detect_source

runner = CliRunner()


def _make_sharded(dir_path: Path) -> None:
    save_file({"a": torch.randn(4, 4)}, dir_path / "shard-00001.safetensors")
    save_file({"b": torch.randn(4, 4)}, dir_path / "shard-00002.safetensors")
    (dir_path / "model.safetensors.index.json").write_bytes(orjson.dumps({
        "metadata": {},
        "weight_map": {
            "a": "shard-00001.safetensors",
            "b": "shard-00002.safetensors",
        },
    }))


class TestShardIntegrity:
    def test_truncated_shard_flagged(self, tmp_path):
        _make_sharded(tmp_path)
        # Truncate one shard in the middle of its tensor data
        shard = tmp_path / "shard-00001.safetensors"
        size = shard.stat().st_size
        with open(shard, "r+b") as f:
            f.truncate(size - 10)

        src = detect_source(tmp_path)
        assert src.incomplete
        assert src.integrity_error is not None
        assert "truncated" in src.integrity_error

    def test_clean_shards_no_integrity_error(self, tmp_path):
        _make_sharded(tmp_path)
        src = detect_source(tmp_path)
        assert not src.incomplete
        assert src.integrity_error is None


class TestVerifyFlag:
    def test_verify_passes_on_clean_merge(self, tmp_path):
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        _make_sharded(src_dir)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", "single", str(src_dir), str(out), "--verify",
        ])
        assert result.exit_code == 0, result.output
        assert "Verified" in result.output

    def test_verify_fails_on_corrupted_output(self, tmp_path):
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        _make_sharded(src_dir)
        out = tmp_path / "out.safetensors"

        # First merge cleanly, then corrupt the output.
        result = runner.invoke(app, ["convert", "single", str(src_dir), str(out)])
        assert result.exit_code == 0, result.output

        # Corrupt by writing a fake header bad-length -- easier: simulate by
        # rewriting output with a different tensor name, then running with --verify.
        # We replay by truncating the output severely; read_raw_header will err.
        with open(out, "r+b") as f:
            f.truncate(8)  # leave only the 8-byte length

        # Running verify as a standalone op: re-invoke convert single with --verify
        # and the same dest file that already "exists" -- but convert will rewrite.
        # So corrupt AFTER the merge by keeping it corrupt and re-running convert.
        # Instead we invoke convert again asking for the same output; it'll be
        # rewritten cleanly, which defeats the test. The cleaner approach: call
        # _verify_output directly.
        from hfutils.commands.convert import _verify_output
        from hfutils.layouts.comfyui import PackOp

        op = PackOp(
            label="x",
            source=src_dir,
            shards=sorted(src_dir.glob("*.safetensors")),
            dest=out,
        )
        assert _verify_output(op) is False


class TestArchitectureRules:
    def test_autoencoder_kl_detected(self):
        info = detect_architecture([
            "encoder.down_blocks.0.resnets.0.conv1.weight",
            "decoder.up_blocks.0.resnets.0.conv1.weight",
        ])
        assert info.family == "AutoencoderKL"

    def test_z_image_detected(self):
        info = detect_architecture([
            "all_final_layer.0.linear.weight",
            "transformer.blocks.0.adaLN_modulation.weight",
        ])
        assert info.family == "Z-Image"


class TestGGUFExtendedFields:
    def test_gguf_info_has_new_fields(self):
        """The dataclass shape is what users of the library import."""
        from hfutils.inspect.gguf import GGUFInfo

        info = GGUFInfo(architecture="test", tensor_count=0)
        assert info.rope_freq_base is None
        assert info.rope_freq_scale is None
        assert info.rope_scaling_type is None
        assert info.bos_token_id is None
        assert info.eos_token_id is None
        assert info.chat_template is None
