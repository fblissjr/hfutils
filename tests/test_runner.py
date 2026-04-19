"""PlanRunner: executes a PackPlan, dispatches events to an Observer."""

import torch
from safetensors.torch import load_file, save_file

from hfutils.events import CollectingObserver
from hfutils.layouts.comfyui import plan_single
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
