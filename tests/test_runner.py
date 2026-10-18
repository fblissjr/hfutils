"""PlanRunner: executes a PackPlan, dispatches events to an Observer."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from hfutils.errors import PlanError
from hfutils.events import CollectingObserver
from hfutils.layouts.comfyui import PackOp, plan_single
from hfutils.layouts.plan import PackPlan
from hfutils.runner import PlanRunner
from hfutils.sources.detect import detect_source
from tests.conftest import make_sharded_component as _make_sharded


class TestPlanRunner:
    def test_merges_a_sharded_plan(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _make_sharded(src_dir)
        out = tmp_path / "out.safetensors"

        plan = plan_single(detect_source(src_dir), out)
        runner = PlanRunner()  # default NullObserver
        manifests = runner.run(plan)

        assert out.exists()
        assert out in manifests
        assert set(manifests[out]) == {"a.weight", "a.bias", "b.weight", "b.bias"}
        merged = load_file(str(out))
        assert set(merged.keys()) == {"a.weight", "a.bias", "b.weight", "b.bias"}

    def test_observer_sees_plan_and_op_lifecycle(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _make_sharded(src_dir)
        out = tmp_path / "out.safetensors"

        plan = plan_single(detect_source(src_dir), out)
        obs = CollectingObserver()
        PlanRunner(observer=obs).run(plan)

        assert obs.plans_started == [plan]
        assert len(obs.ops_started) == 1
        assert len(obs.ops_completed) == 1
        assert len(obs.plans_completed) == 1
        # Progress events emitted while copying
        assert obs.progress  # non-empty
        # Progress bytes sum to something close to the total (stream_merge
        # reports exact tensor-data bytes, which is file size minus header).
        total = sum(n for _, n in obs.progress)
        assert total > 0

    def test_copy_op_also_populates_manifest(self, tmp_path):
        # Single-file source -> copy op
        src = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)
        out = tmp_path / "out.safetensors"

        plan = plan_single(detect_source(src), out)
        manifests = PlanRunner().run(plan)
        assert out in manifests
        assert "w" in manifests[out]

    def test_null_observer_is_default(self, tmp_path):
        """A bare PlanRunner() should be silent but functional."""
        src = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)
        out = tmp_path / "out.safetensors"

        plan = plan_single(detect_source(src), out)
        PlanRunner().run(plan)
        assert out.exists()

    def test_on_plan_complete_called_on_exception(self, tmp_path):
        """If an op raises partway through, on_plan_complete must still fire so
        observers holding resources (rich Progress, open files, etc.) can
        tear down cleanly."""
        src = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)
        plan = plan_single(detect_source(src), tmp_path / "out.safetensors")

        class ExplodingObserver(CollectingObserver):
            def on_op_start(self, op, total_bytes):
                super().on_op_start(op, total_bytes)
                raise RuntimeError("boom")

        exploder = ExplodingObserver()
        try:
            PlanRunner(observer=exploder).run(plan)
        except RuntimeError:
            pass

        assert exploder.plans_started == [plan]
        assert len(exploder.plans_completed) == 1

    def test_refuses_broken_plan_before_on_plan_start(self, tmp_path):
        """A plan with an incomplete/corrupt source op must be rejected
        before any writing (or even on_plan_start) happens -- silently
        merging a component missing shards produces a syntactically valid
        but semantically broken output with no exception to catch it later."""
        shard = tmp_path / "in.safetensors"
        save_file({"w": torch.randn(2, 2)}, shard)
        broken_op = PackOp(
            label="transformer", source=tmp_path, dest=tmp_path / "out.safetensors",
            shards=[shard], problem="incomplete: missing shard(s) per index.json",
        )
        plan = PackPlan(ops=[broken_op], source=detect_source(tmp_path))
        obs = CollectingObserver()

        with pytest.raises(PlanError, match="incomplete"):
            PlanRunner(observer=obs).run(plan)

        assert obs.plans_started == []
        assert not (tmp_path / "out.safetensors").exists()

    def test_force_proceeds_with_a_warning(self, tmp_path):
        """force=True is an explicit opt-in to write a known-broken source
        anyway -- it must still warn (not vanish silently), and must
        actually write the file."""
        shard = tmp_path / "in.safetensors"
        save_file({"w": torch.randn(2, 2)}, shard)
        out = tmp_path / "out.safetensors"
        broken_op = PackOp(
            label="transformer", source=tmp_path, dest=out,
            shards=[shard], problem="incomplete: missing shard(s) per index.json",
        )
        plan = PackPlan(ops=[broken_op], source=detect_source(tmp_path))
        obs = CollectingObserver()

        manifests = PlanRunner(observer=obs).run(plan, force=True)

        assert out.exists()
        assert out in manifests
        assert obs.plans_started == [plan]
        assert any(
            op is broken_op and "incomplete" in message
            for op, message in obs.warnings
        )

    def test_refuses_plan_with_missing_components(self, tmp_path):
        """A plan whose PackPlan.missing is non-empty (a declared pipeline
        component with zero weight files on disk) must be refused just like
        broken_ops(), even though there's no PackOp to attach the problem
        to."""
        shard = tmp_path / "in.safetensors"
        save_file({"w": torch.randn(2, 2)}, shard)
        clean_op = PackOp(label="transformer", source=tmp_path, dest=tmp_path / "out.safetensors", shards=[shard])
        plan = PackPlan(ops=[clean_op], source=detect_source(tmp_path), missing=["vae"])
        obs = CollectingObserver()

        with pytest.raises(PlanError, match="vae"):
            PlanRunner(observer=obs).run(plan)

        assert obs.plans_started == []
        assert not (tmp_path / "out.safetensors").exists()

    def test_copy_branch_cleans_up_partial_output_on_failure(self, tmp_path):
        """Unlike the merge branch (stream_merge, self-cleaning via
        unlink_on_error), the copy branch (single-shard ops, copy_chunks)
        used to leave a partial file behind on a mid-copy failure with no
        cleanup at all."""
        src = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, src)
        out = tmp_path / "out.safetensors"
        plan = plan_single(detect_source(src), out)

        class ExplodingProgressObserver(CollectingObserver):
            def on_op_progress(self, op, bytes_copied):
                super().on_op_progress(op, bytes_copied)
                raise RuntimeError("boom mid-copy")

        with pytest.raises(RuntimeError):
            PlanRunner(observer=ExplodingProgressObserver()).run(plan)

        assert not out.exists()

    def test_force_writes_plan_with_missing_components(self, tmp_path):
        shard = tmp_path / "in.safetensors"
        save_file({"w": torch.randn(2, 2)}, shard)
        out = tmp_path / "out.safetensors"
        clean_op = PackOp(label="transformer", source=tmp_path, dest=out, shards=[shard])
        plan = PackPlan(ops=[clean_op], source=detect_source(tmp_path), missing=["vae"])

        manifests = PlanRunner().run(plan, force=True)

        assert out.exists()
        assert out in manifests

    def test_single_shard_bare_qwen_vlm_still_normalizes_keys(self, tmp_path):
        """Even when a bare Qwen VLM text encoder is in a single shard (op.kind
        == 'copy'), PlanRunner must rewrite the header so keys get the outer
        `model.` prefix rather than byte-copying un-prefixed keys."""
        src = tmp_path / "te.safetensors"
        save_file({
            "language_model.layers.0.linear_attn.A_log": torch.randn(2),
            "visual.blocks.0.attn.qkv.weight": torch.randn(2, 2),
        }, src)
        out = tmp_path / "out.safetensors"

        plan = plan_single(detect_source(src), out)
        manifests = PlanRunner().run(plan)

        expected = {
            "model.language_model.layers.0.linear_attn.A_log",
            "model.visual.blocks.0.attn.qkv.weight",
        }
        assert set(manifests[out].keys()) == expected
        assert set(load_file(str(out)).keys()) == expected

