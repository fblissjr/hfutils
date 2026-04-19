"""CLI-level tests for `hfutils convert comfyui`."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from hfutils.cli import app

runner = CliRunner()


def _make_sharded(subdir: Path, shard_prefix: str, index_name: str) -> None:
    s1 = {"layer.0.weight": torch.randn(4, 4)}
    s2 = {"layer.1.weight": torch.randn(4, 4)}
    save_file(s1, subdir / f"{shard_prefix}-00001-of-00002.safetensors")
    save_file(s2, subdir / f"{shard_prefix}-00002-of-00002.safetensors")
    weight_map = {k: f"{shard_prefix}-00001-of-00002.safetensors" for k in s1}
    weight_map.update({k: f"{shard_prefix}-00002-of-00002.safetensors" for k in s2})
    (subdir / index_name).write_bytes(orjson.dumps({"metadata": {}, "weight_map": weight_map}))


def _make_pipeline(tmp: Path) -> None:
    (tmp / "model_index.json").write_bytes(orjson.dumps({
        "_class_name": "P",
        "transformer": ["diffusers", "T"],
        "vae": ["diffusers", "V"],
        "text_encoder": ["transformers", "E"],
    }))
    (tmp / "transformer").mkdir()
    (tmp / "transformer/config.json").write_bytes(orjson.dumps({"_class_name": "T"}))
    _make_sharded(
        tmp / "transformer",
        shard_prefix="diffusion_pytorch_model",
        index_name="diffusion_pytorch_model.safetensors.index.json",
    )
    (tmp / "vae").mkdir()
    save_file({"enc.w": torch.randn(3, 3)}, tmp / "vae/diffusion_pytorch_model.safetensors")
    (tmp / "text_encoder").mkdir()
    save_file({"emb.w": torch.randn(3, 3)}, tmp / "text_encoder/model.safetensors")


class TestConvertComfyui:
    def test_packs_full_pipeline(self, tmp_path):
        _make_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", "comfyui", str(tmp_path), str(comfy),
            "--name", "MyModel",
        ])
        assert result.exit_code == 0, result.output

        merged = load_file(str(comfy / "diffusion_models/MyModel.safetensors"))
        assert set(merged.keys()) == {"layer.0.weight", "layer.1.weight"}
        assert (comfy / "vae/MyModel_vae.safetensors").exists()
        assert (comfy / "text_encoders/MyModel_te.safetensors").exists()

    def test_only_filter(self, tmp_path):
        _make_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", "comfyui", str(tmp_path), str(comfy),
            "--name", "X", "--only", "transformer",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy / "diffusion_models/X.safetensors").exists()
        assert not (comfy / "vae").exists()

    def test_single_file_requires_as(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)

        result = runner.invoke(app, [
            "convert", "comfyui", str(f), str(tmp_path / "c"),
            "--name", "X",
        ])
        assert result.exit_code != 0

    def test_single_file_with_as(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", "comfyui", str(f), str(comfy),
            "--name", "X", "--as", "diffusion_model",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy / "diffusion_models/X.safetensors").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        _make_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", "comfyui", str(tmp_path), str(comfy),
            "--name", "X", "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert not comfy.exists()

    def test_default_name_from_source_dir(self, tmp_path):
        pipeline_dir = tmp_path / "MyPipeline"
        pipeline_dir.mkdir()
        _make_pipeline(pipeline_dir)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", "comfyui", str(pipeline_dir), str(comfy),
            "--only", "transformer",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy / "diffusion_models/MyPipeline.safetensors").exists()
