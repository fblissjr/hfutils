"""PackPlan object: total_bytes, validate, iteration."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import save_file

from hfutils.layouts.comfyui import PackOp, plan_comfyui, plan_single
from hfutils.layouts.plan import PackPlan
from hfutils.sources.detect import detect_source
from tests.conftest import make_sharded_component as _make_sharded


class TestPackPlan:
    def test_total_bytes_sums_shard_sizes(self, tmp_path):
        _make_sharded(tmp_path)
        plan = plan_single(detect_source(tmp_path), tmp_path / "out.safetensors")
        expected = sum(s.stat().st_size for op in plan.ops for s in op.shards)
        assert plan.total_bytes == expected

    def test_plan_is_iterable_and_has_len(self, tmp_path):
        _make_sharded(tmp_path)
        plan = plan_single(detect_source(tmp_path), tmp_path / "out.safetensors")
        assert len(plan) == 1
        assert list(plan) == plan.ops

    def test_empty_plan_is_falsy(self, tmp_path):
        plan = PackPlan(ops=[], source=detect_source(tmp_path))
        assert not plan

    def test_validate_flags_duplicate_dest(self, tmp_path):
        src = detect_source(tmp_path)  # UnknownSource; we won't use it for anything
        dest = tmp_path / "out.safetensors"
        shard = tmp_path / "shard.safetensors"
        save_file({"w": torch.randn(2, 2)}, shard)
        plan = PackPlan(
            ops=[
                PackOp(label="a", source=shard, dest=dest, shards=[shard]),
                PackOp(label="b", source=shard, dest=dest, shards=[shard]),
            ],
            source=src,
        )
        problems = plan.validate()
        assert any("duplicate" in p.lower() for p in problems)

    def test_validate_flags_empty_shards(self, tmp_path):
        src = detect_source(tmp_path)
        plan = PackPlan(
            ops=[PackOp(label="x", source=tmp_path, dest=tmp_path / "out.safetensors", shards=[])],
            source=src,
        )
        problems = plan.validate()
        assert any("no shards" in p for p in problems)


class TestPlanComfyuiReturnsPackPlan:
    def test_pipeline_returns_plan_with_source_and_meta(self, tmp_path):
        (tmp_path / "model_index.json").write_bytes(orjson.dumps({
            "_class_name": "P",
            "transformer": ["diffusers", "T"],
        }))
        (tmp_path / "transformer").mkdir()
        save_file({"w": torch.randn(2, 2)}, tmp_path / "transformer/model.safetensors")

        src = detect_source(tmp_path)
        plan = plan_comfyui(src, tmp_path / "comfy", name="X")
        assert isinstance(plan, PackPlan)
        assert plan.source is src
        assert plan.meta["target"] == "comfyui"
        assert len(plan) == 1
