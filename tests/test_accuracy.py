"""Tests for shard integrity detection, --verify, and new architecture rules."""

from typer.testing import CliRunner

from hfutils.cli import app
from hfutils.inspect.architecture import detect_architecture
from hfutils.sources.detect import DetectLevel, detect_source

runner = CliRunner()


from tests.conftest import make_sharded_component as _make_sharded  # noqa: E402


class TestShardIntegrity:
    def test_truncated_shard_flagged(self, tmp_path):
        _make_sharded(tmp_path)
        # Truncate one shard in the middle of its tensor data
        shard = tmp_path / "shard-00001-of-00002.safetensors"
        size = shard.stat().st_size
        with open(shard, "r+b") as f:
            f.truncate(size - 10)

        src = detect_source(tmp_path, DetectLevel.FULL)
        assert src.incomplete
        assert src.integrity_error is not None
        assert src.integrity_error.kind == "truncated"
        assert src.integrity_error.file == shard

    def test_clean_shards_no_integrity_error(self, tmp_path):
        _make_sharded(tmp_path)
        src = detect_source(tmp_path, DetectLevel.FULL)
        assert not src.incomplete
        assert src.integrity_error is None

    def test_basic_level_skips_integrity(self, tmp_path):
        """DetectLevel.BASIC (default) doesn't catch truncation -- that's the point
        (walker doesn't need this expense per dir)."""
        _make_sharded(tmp_path)
        shard = tmp_path / "shard-00001-of-00002.safetensors"
        size = shard.stat().st_size
        with open(shard, "r+b") as f:
            f.truncate(size - 10)

        src = detect_source(tmp_path)  # default BASIC
        assert src.integrity_error is None


class TestVerifyFlag:
    def test_verify_passes_on_clean_merge(self, tmp_path):
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        _make_sharded(src_dir)
        out = tmp_path / "out.safetensors"

        result = runner.invoke(app, [
            "convert", str(src_dir), "--to", "single", "--out", str(out), "--verify",
        ])
        assert result.exit_code == 0, result.output
        assert "Verified" in result.output

    def test_verify_fails_on_corrupted_output(self, tmp_path):
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        _make_sharded(src_dir)
        good = tmp_path / "good.safetensors"

        # Produce a known-good output, then corrupt it, then ask the public
        # verify_output helper whether it still matches the plan's manifest.
        result = runner.invoke(app, [
            "convert", str(src_dir), "--to", "single", "--out", str(good),
        ])
        assert result.exit_code == 0, result.output

        from hfutils.formats.safetensors import manifest_from_shards, verify_output

        manifest = manifest_from_shards(sorted(src_dir.glob("*.safetensors")))
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"\x00" * 8)  # 8-byte length prefix only; header read will fail

        ok, err = verify_output(bad, manifest)
        assert ok is False
        assert err is not None


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
