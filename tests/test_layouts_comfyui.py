"""ComfyUI layout planning: Source -> list of PackOps."""

from pathlib import Path

import orjson
import pytest
import torch
from safetensors.torch import save_file

from hfutils.layouts.comfyui import (
    TARGET_FOLDERS,
    PackOp,
    plan_pack,
)
from hfutils.sources.detect import detect_source


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


class TestPlanPack:
    def test_diffusers_pipeline_plans_all_components(self, tmp_path):
        _make_pipeline(tmp_path)
        src = detect_source(tmp_path)
        comfy = tmp_path / "comfy"

        ops = plan_pack(src, comfy, name="X")

        by_label = {op.label: op for op in ops}
        assert by_label["transformer"].dest == comfy / "diffusion_models/X.safetensors"
        assert by_label["vae"].dest == comfy / "vae/X_vae.safetensors"
        assert by_label["text_encoder"].dest == comfy / "text_encoders/X_te.safetensors"

    def test_only_filter(self, tmp_path):
        _make_pipeline(tmp_path)
        src = detect_source(tmp_path)
        ops = plan_pack(src, tmp_path / "c", name="X", only=["transformer"])
        assert [op.label for op in ops] == ["transformer"]

    def test_skip_filter(self, tmp_path):
        _make_pipeline(tmp_path)
        src = detect_source(tmp_path)
        ops = plan_pack(src, tmp_path / "c", name="X", skip=["text_encoder"])
        assert sorted(op.label for op in ops) == ["transformer", "vae"]

    def test_single_file_requires_target(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        src = detect_source(f)

        with pytest.raises(ValueError, match="--as"):
            plan_pack(src, tmp_path / "c", name="X")

    def test_single_file_with_target(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        src = detect_source(f)

        ops = plan_pack(src, tmp_path / "c", name="X", target="diffusion_model")
        assert len(ops) == 1
        assert ops[0].dest == tmp_path / "c" / "diffusion_models/X.safetensors"
        assert ops[0].kind == "copy"

    def test_component_dir_with_target(self, tmp_path):
        # Sharded component dir
        save_file({"a": torch.randn(2, 2)}, tmp_path / "shard-00001-of-00002.safetensors")
        save_file({"b": torch.randn(2, 2)}, tmp_path / "shard-00002-of-00002.safetensors")
        (tmp_path / "model.safetensors.index.json").write_bytes(orjson.dumps({
            "metadata": {},
            "weight_map": {
                "a": "shard-00001-of-00002.safetensors",
                "b": "shard-00002-of-00002.safetensors",
            },
        }))
        src = detect_source(tmp_path)

        ops = plan_pack(src, tmp_path / "c", name="X", target="diffusion_model")
        assert len(ops) == 1
        assert ops[0].kind == "merge"

    def test_unknown_target_rejected(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        src = detect_source(f)

        with pytest.raises(ValueError, match="Unknown"):
            plan_pack(src, tmp_path / "c", name="X", target="bogus")

    def test_packop_kind_derived_from_shards(self):
        op_copy = PackOp(label="x", dest=Path("/b"), shards=[Path("/a.safetensors")])
        op_merge = PackOp(label="x", dest=Path("/b"), shards=[Path("/a.safetensors"), Path("/b.safetensors")])
        assert op_copy.kind == "copy"
        assert op_merge.kind == "merge"

    def test_target_folders_includes_core_destinations(self):
        for key in ("diffusion_model", "vae", "text_encoder", "checkpoint", "lora"):
            assert key in TARGET_FOLDERS
