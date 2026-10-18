"""ComfyUI layout planning: Source -> list of PackOps."""

from pathlib import Path

import orjson
import pytest
import torch
from safetensors.torch import save_file

from hfutils.errors import PlanError
from hfutils.layouts.comfyui import (
    TARGET_FOLDERS,
    ConvertTarget,
    PackOp,
    plan_comfyui,
)
from hfutils.sources.detect import detect_source
from tests.conftest import make_diffusers_pipeline as _make_full_pipeline


def _truncate(path: Path, remove_bytes: int) -> None:
    size = path.stat().st_size
    with open(path, "r+b") as f:
        f.truncate(size - remove_bytes)


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


class TestPlanComfyui:
    def test_diffusers_pipeline_plans_all_components(self, tmp_path):
        _make_pipeline(tmp_path)
        src = detect_source(tmp_path)
        comfy = tmp_path / "comfy"

        plan = plan_comfyui(src, comfy, name="X")

        by_label = {op.label: op for op in plan.ops}
        assert by_label["transformer"].dest == comfy / "diffusion_models/X.safetensors"
        assert by_label["vae"].dest == comfy / "vae/X_vae.safetensors"
        assert by_label["text_encoder"].dest == comfy / "text_encoders/X_te.safetensors"

    def test_only_filter(self, tmp_path):
        _make_pipeline(tmp_path)
        src = detect_source(tmp_path)
        plan = plan_comfyui(src, tmp_path / "c", name="X", only=["transformer"])
        assert [op.label for op in plan.ops] == ["transformer"]

    def test_skip_filter(self, tmp_path):
        _make_pipeline(tmp_path)
        src = detect_source(tmp_path)
        plan = plan_comfyui(src, tmp_path / "c", name="X", skip=["text_encoder"])
        assert sorted(op.label for op in plan.ops) == ["transformer", "vae"]

    def test_single_file_requires_target(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        src = detect_source(f)

        with pytest.raises(PlanError, match="--as"):
            plan_comfyui(src, tmp_path / "c", name="X")

    def test_single_file_with_target(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        src = detect_source(f)

        plan = plan_comfyui(src, tmp_path / "c", name="X", target=ConvertTarget.DIFFUSION_MODEL)
        assert len(plan) == 1
        assert plan.ops[0].dest == tmp_path / "c" / "diffusion_models/X.safetensors"
        assert plan.ops[0].kind == "copy"

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

        plan = plan_comfyui(src, tmp_path / "c", name="X", target=ConvertTarget.DIFFUSION_MODEL)
        assert len(plan) == 1
        assert plan.ops[0].kind == "merge"

    def test_unknown_target_rejected(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        src = detect_source(f)

        with pytest.raises(PlanError, match="Unknown"):
            plan_comfyui(src, tmp_path / "c", name="X", target="bogus")

    def test_packop_kind_derived_from_shards(self):
        op_copy = PackOp(
            label="x", dest=Path("/b"), source=Path("/a"),
            shards=[Path("/a.safetensors")],
        )
        op_merge = PackOp(
            label="x", dest=Path("/b"), source=Path("/a"),
            shards=[Path("/a.safetensors"), Path("/b.safetensors")],
        )
        assert op_copy.kind == "copy"
        assert op_merge.kind == "merge"

    def test_target_folders_includes_core_destinations(self):
        for key in ("diffusion_model", "vae", "text_encoder", "checkpoint", "lora"):
            assert key in TARGET_FOLDERS


class TestPlanComfyuiIncompleteDetection:
    def test_pipeline_component_missing_shard_is_flagged(self, tmp_path):
        """_plan_component must route through detect_source instead of a
        raw glob, or an interrupted-download pipeline (shard missing,
        index.json present) silently merges into a syntactically valid but
        incomplete output with no signal at all."""
        _make_full_pipeline(tmp_path)
        shard = tmp_path / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors"
        assert shard.exists()
        shard.unlink()

        src = detect_source(tmp_path)
        plan = plan_comfyui(src, tmp_path / "comfy", name="X")

        by_label = {op.label: op for op in plan.ops}
        assert by_label["transformer"].problem is not None
        assert "incomplete" in by_label["transformer"].problem.lower()

    def test_complete_pipeline_has_no_problems(self, tmp_path):
        _make_full_pipeline(tmp_path)
        src = detect_source(tmp_path)
        plan = plan_comfyui(src, tmp_path / "comfy", name="X")

        assert all(op.problem is None for op in plan.ops)

    def test_truncated_unsharded_component_is_flagged(self, tmp_path):
        """A single-file (no index.json) component that's truncated must be
        flagged too -- not just multi-shard sets with a missing shard. This
        is a 'copy'-kind op (len(shards)==1), which the merge-kind EOF check
        never protects."""
        _make_full_pipeline(tmp_path)
        vae_file = tmp_path / "vae" / "diffusion_pytorch_model.safetensors"
        _truncate(vae_file, 20)

        src = detect_source(tmp_path)
        plan = plan_comfyui(src, tmp_path / "comfy", name="X")

        by_label = {op.label: op for op in plan.ops}
        assert by_label["vae"].problem is not None
        assert "truncated" in by_label["vae"].problem.lower()

    def test_truncated_single_file_source_with_target_is_flagged(self, tmp_path):
        """SafetensorsFileSource copy-ops (--as target on a bare file) got no
        problem= check at all -- same gap, different Source variant."""
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(8, 8)}, f)
        _truncate(f, 20)
        src = detect_source(f)

        plan = plan_comfyui(src, tmp_path / "c", name="X", target=ConvertTarget.DIFFUSION_MODEL)

        assert plan.ops[0].problem is not None
        assert "truncated" in plan.ops[0].problem.lower()

    def test_clean_single_file_source_has_no_problem(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        src = detect_source(f)

        plan = plan_comfyui(src, tmp_path / "c", name="X", target=ConvertTarget.DIFFUSION_MODEL)

        assert plan.ops[0].problem is None

    def test_truncated_component_dir_with_target_is_flagged(self, tmp_path):
        """plan_comfyui's ComponentSource branch (--as target on a component
        dir) with a single shard must also be checked."""
        f = tmp_path / "shard-00001-of-00001.safetensors"
        save_file({"w": torch.randn(8, 8)}, f)
        _truncate(f, 20)
        src = detect_source(tmp_path)

        plan = plan_comfyui(src, tmp_path / "c", name="X", target=ConvertTarget.DIFFUSION_MODEL)

        assert plan.ops[0].problem is not None
        assert "truncated" in plan.ops[0].problem.lower()


class TestPlanComfyuiMissingComponents:
    def _pipeline_missing_vae(self, tmp: Path) -> None:
        """model_index.json declares vae, but the vae/ directory never
        made it to disk at all -- total component loss (e.g. every shard's
        download failed), not partial."""
        (tmp / "model_index.json").write_bytes(orjson.dumps({
            "_class_name": "P",
            "transformer": ["diffusers", "T"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "E"],
            "scheduler": ["diffusers", "S"],
            "tokenizer": ["transformers", "Tok"],
        }))
        for name in ("transformer", "text_encoder"):
            (tmp / name).mkdir()
            save_file({"w": torch.randn(2, 2)}, tmp / name / "model.safetensors")

    def test_declared_but_entirely_absent_component_is_flagged(self, tmp_path):
        self._pipeline_missing_vae(tmp_path)
        src = detect_source(tmp_path)

        plan = plan_comfyui(src, tmp_path / "comfy", name="X")

        assert plan.missing == ["vae"]
        # transformer/text_encoder still convert normally
        assert sorted(op.label for op in plan.ops) == ["text_encoder", "transformer"]

    def test_scheduler_and_tokenizer_never_count_as_missing(self, tmp_path):
        """scheduler/tokenizer are legitimately weight-less -- must not be
        flagged as missing just because they have no .safetensors files."""
        self._pipeline_missing_vae(tmp_path)
        src = detect_source(tmp_path)

        plan = plan_comfyui(src, tmp_path / "comfy", name="X")

        assert "scheduler" not in plan.missing
        assert "tokenizer" not in plan.missing

    def test_complete_pipeline_has_no_missing(self, tmp_path):
        _make_full_pipeline(tmp_path)
        src = detect_source(tmp_path)

        plan = plan_comfyui(src, tmp_path / "comfy", name="X")

        assert plan.missing == []

    def test_skip_suppresses_missing_for_intentionally_excluded_component(self, tmp_path):
        """--skip vae on a pipeline missing vae shouldn't hard-stop on a
        component the user explicitly opted out of."""
        self._pipeline_missing_vae(tmp_path)
        src = detect_source(tmp_path)

        plan = plan_comfyui(src, tmp_path / "comfy", name="X", skip=["vae"])

        assert plan.missing == []

    def test_only_suppresses_missing_for_components_not_requested(self, tmp_path):
        self._pipeline_missing_vae(tmp_path)
        src = detect_source(tmp_path)

        plan = plan_comfyui(src, tmp_path / "comfy", name="X", only=["transformer"])

        assert plan.missing == []

    def test_unreadable_model_index_fails_open(self, tmp_path):
        """A malformed model_index.json (read_json_if_exists returns None)
        must not crash or spuriously flag every component as missing."""
        (tmp_path / "model_index.json").write_bytes(b"not valid json")
        (tmp_path / "transformer").mkdir()
        save_file({"w": torch.randn(2, 2)}, tmp_path / "transformer" / "model.safetensors")
        src = detect_source(tmp_path)

        plan = plan_comfyui(src, tmp_path / "comfy", name="X")

        assert plan.missing == []


class TestPlanSinglePipeline:
    def test_pipeline_plans_all_components_into_output_dir(self, tmp_path):
        from hfutils.layouts.comfyui import plan_single

        _make_pipeline(tmp_path)
        src = detect_source(tmp_path)
        out_dir = tmp_path / "out"

        plan = plan_single(src, out_dir, name="MyModel")

        by_label = {op.label: op for op in plan.ops}
        assert by_label["transformer"].dest == out_dir / "MyModel.safetensors"
        assert by_label["vae"].dest == out_dir / "MyModel_vae.safetensors"
        assert by_label["text_encoder"].dest == out_dir / "MyModel_te.safetensors"

    def test_pipeline_plan_single_respects_only_and_skip(self, tmp_path):
        from hfutils.layouts.comfyui import plan_single

        _make_pipeline(tmp_path)
        src = detect_source(tmp_path)
        out_dir = tmp_path / "out"

        plan = plan_single(src, out_dir, name="X", only=["transformer", "vae"])
        assert sorted(op.label for op in plan.ops) == ["transformer", "vae"]

        plan_skip = plan_single(src, out_dir, name="X", skip=["vae"])
        assert sorted(op.label for op in plan_skip.ops) == ["text_encoder", "transformer"]

