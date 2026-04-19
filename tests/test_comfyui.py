"""Tests for comfyui pack -- convert any local model layout into ComfyUI folders."""

from pathlib import Path

import orjson
import pytest
import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from hfutils.cli import app
from hfutils.commands.comfyui import plan_pack


runner = CliRunner()


def _make_sharded(subdir: Path, shard_prefix: str, index_name: str) -> dict[str, torch.Tensor]:
    s1 = {"layer.0.weight": torch.randn(4, 4)}
    s2 = {"layer.1.weight": torch.randn(4, 4)}
    save_file(s1, subdir / f"{shard_prefix}-00001-of-00002.safetensors")
    save_file(s2, subdir / f"{shard_prefix}-00002-of-00002.safetensors")
    weight_map = {k: f"{shard_prefix}-00001-of-00002.safetensors" for k in s1}
    weight_map.update({k: f"{shard_prefix}-00002-of-00002.safetensors" for k in s2})
    (subdir / index_name).write_bytes(orjson.dumps({"metadata": {}, "weight_map": weight_map}))
    return {**s1, **s2}


def _make_single(subdir: Path, filename: str) -> dict[str, torch.Tensor]:
    tensors = {"conv.weight": torch.randn(3, 3)}
    save_file(tensors, subdir / filename)
    return tensors


def _make_diffusers_pipeline(tmp: Path) -> dict[str, dict[str, torch.Tensor]]:
    (tmp / "model_index.json").write_bytes(orjson.dumps({
        "_class_name": "ZImagePipeline",
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "text_encoder": ["transformers", "Qwen3Model"],
    }))

    transformer = tmp / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_bytes(orjson.dumps({"_class_name": "ZImageTransformer2DModel"}))
    t_tensors = _make_sharded(
        transformer,
        shard_prefix="diffusion_pytorch_model",
        index_name="diffusion_pytorch_model.safetensors.index.json",
    )

    vae = tmp / "vae"
    vae.mkdir()
    v_tensors = _make_single(vae, "diffusion_pytorch_model.safetensors")

    te = tmp / "text_encoder"
    te.mkdir()
    te_tensors = _make_single(te, "model.safetensors")

    return {"transformer": t_tensors, "vae": v_tensors, "text_encoder": te_tensors}


class TestPlanPack:
    def test_diffusers_pipeline_plans_all_components(self, tmp_path):
        _make_diffusers_pipeline(tmp_path)
        comfy_root = tmp_path / "comfy"

        plan = plan_pack(source=tmp_path, comfyui_root=comfy_root, name="X")

        dests = {(op.label, op.dest.relative_to(comfy_root).as_posix()) for op in plan.ops}
        assert dests == {
            ("transformer", "diffusion_models/X.safetensors"),
            ("vae", "vae/X_vae.safetensors"),
            ("text_encoder", "text_encoders/X_te.safetensors"),
        }

    def test_only_filter_restricts_components(self, tmp_path):
        _make_diffusers_pipeline(tmp_path)
        plan = plan_pack(
            source=tmp_path, comfyui_root=tmp_path / "c", name="X",
            only=["transformer"],
        )
        assert [op.label for op in plan.ops] == ["transformer"]

    def test_skip_filter_excludes_components(self, tmp_path):
        _make_diffusers_pipeline(tmp_path)
        plan = plan_pack(
            source=tmp_path, comfyui_root=tmp_path / "c", name="X",
            skip=["text_encoder"],
        )
        assert set(op.label for op in plan.ops) == {"transformer", "vae"}

    def test_single_file_requires_target(self, tmp_path):
        src = tmp_path / "ltx.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)

        with pytest.raises(ValueError, match="--as"):
            plan_pack(source=src, comfyui_root=tmp_path / "c", name="LTX")

    def test_single_file_with_target_plans_copy(self, tmp_path):
        src = tmp_path / "ltx.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)

        plan = plan_pack(
            source=src, comfyui_root=tmp_path / "c", name="LTX",
            target="diffusion_model",
        )
        assert len(plan.ops) == 1
        op = plan.ops[0]
        assert op.dest.relative_to(tmp_path / "c").as_posix() == "diffusion_models/LTX.safetensors"

    def test_component_dir_with_target_plans_merge(self, tmp_path):
        component = tmp_path / "transformer"
        component.mkdir()
        _make_sharded(
            component,
            shard_prefix="diffusion_pytorch_model",
            index_name="diffusion_pytorch_model.safetensors.index.json",
        )

        plan = plan_pack(
            source=component, comfyui_root=tmp_path / "c", name="Z",
            target="diffusion_model",
        )
        assert len(plan.ops) == 1
        assert plan.ops[0].dest.relative_to(tmp_path / "c").as_posix() == "diffusion_models/Z.safetensors"


class TestPackCommand:
    def test_packs_full_pipeline(self, tmp_path):
        expected = _make_diffusers_pipeline(tmp_path)
        comfy_root = tmp_path / "comfy"

        result = runner.invoke(app, [
            "comfyui", "pack", str(tmp_path), str(comfy_root),
            "--name", "ZImage",
        ])
        assert result.exit_code == 0, result.output

        t = load_file(str(comfy_root / "diffusion_models/ZImage.safetensors"))
        assert set(t.keys()) == set(expected["transformer"].keys())
        assert (comfy_root / "vae/ZImage_vae.safetensors").exists()
        assert (comfy_root / "text_encoders/ZImage_te.safetensors").exists()

    def test_only_filter_via_cli(self, tmp_path):
        _make_diffusers_pipeline(tmp_path)
        comfy_root = tmp_path / "comfy"

        result = runner.invoke(app, [
            "comfyui", "pack", str(tmp_path), str(comfy_root),
            "--name", "Z",
            "--only", "transformer",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy_root / "diffusion_models/Z.safetensors").exists()
        assert not (comfy_root / "vae").exists()
        assert not (comfy_root / "text_encoders").exists()

    def test_single_file_copy(self, tmp_path):
        src = tmp_path / "ltx.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)
        comfy_root = tmp_path / "comfy"

        result = runner.invoke(app, [
            "comfyui", "pack", str(src), str(comfy_root),
            "--name", "LTX",
            "--as", "diffusion_model",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy_root / "diffusion_models/LTX.safetensors").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        _make_diffusers_pipeline(tmp_path)
        comfy_root = tmp_path / "comfy"

        result = runner.invoke(app, [
            "comfyui", "pack", str(tmp_path), str(comfy_root),
            "--name", "Z",
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert not comfy_root.exists()

    def test_default_name_from_source_dir(self, tmp_path):
        pipeline = tmp_path / "MyPipeline"
        pipeline.mkdir()
        _make_diffusers_pipeline(pipeline)
        comfy_root = tmp_path / "comfy"

        result = runner.invoke(app, [
            "comfyui", "pack", str(pipeline), str(comfy_root),
            "--only", "transformer",
        ])
        assert result.exit_code == 0, result.output
        assert (comfy_root / "diffusion_models/MyPipeline.safetensors").exists()

    def test_single_file_without_target_errors(self, tmp_path):
        src = tmp_path / "blob.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)

        result = runner.invoke(app, [
            "comfyui", "pack", str(src), str(tmp_path / "c"),
            "--name", "X",
        ])
        assert result.exit_code != 0
