"""Observer protocols: stream_merge talks to a MergeObserver, PlanRunner
talks to an Observer, and the two are bridged by per_op_merge_observer."""

from pathlib import Path

import torch
from safetensors.torch import save_file

from hfutils.events import (
    CollectingMergeObserver,
    CollectingObserver,
    MergeObserver,
    NullMergeObserver,
    NullObserver,
    Observer,
    per_op_merge_observer,
)
from hfutils.formats.safetensors import stream_merge


def _write_shards(tmp: Path, n: int) -> list[Path]:
    paths = []
    for i in range(1, n + 1):
        p = tmp / f"shard-{i:05d}.safetensors"
        save_file({f"t{i}": torch.randn(4, 4)}, p)
        paths.append(p)
    return paths


class TestMergeObserver:
    def test_null_observer_does_nothing(self, tmp_path):
        shards = _write_shards(tmp_path, 2)
        # Must not raise regardless of what stream_merge calls on it.
        stream_merge(shards, tmp_path / "out.safetensors", observer=NullMergeObserver())

    def test_collecting_observer_captures_events(self, tmp_path):
        shards = _write_shards(tmp_path, 3)
        obs = CollectingMergeObserver()
        stream_merge(shards, tmp_path / "out.safetensors", observer=obs)

        assert obs.total is not None
        assert obs.total > 0
        assert obs.advanced == obs.total  # on_progress sums to the declared total
        assert obs.warnings == []  # no metadata conflicts

    def test_protocol_is_structural(self):
        """Any object with the three methods should satisfy MergeObserver."""
        class DuckObserver:
            def on_total(self, total_bytes): pass
            def on_progress(self, bytes_copied): pass
            def on_warning(self, message): pass

        def accepts(observer: MergeObserver) -> None: pass
        accepts(DuckObserver())  # type-checks at runtime via Protocol


class TestPlanObserverShape:
    def test_null_observer_has_all_six_methods(self):
        obs = NullObserver()
        obs.on_plan_start(None)
        obs.on_op_start(None, 0)
        obs.on_op_progress(None, 0)
        obs.on_op_warning(None, "hi")
        obs.on_op_complete(None, {})
        obs.on_plan_complete(None, {})

    def test_collecting_observer_records_all_scopes(self):
        obs = CollectingObserver()
        obs.on_plan_start("plan")
        obs.on_op_start("op", 100)
        obs.on_op_progress("op", 50)
        obs.on_op_warning("op", "hey")
        obs.on_op_complete("op", {"a": ("F32", (2, 2))})
        obs.on_plan_complete("plan", {})

        assert obs.plans_started == ["plan"]
        assert obs.ops_started == [("op", 100)]
        assert obs.progress == [("op", 50)]
        assert obs.warnings == [("op", "hey")]
        assert len(obs.ops_completed) == 1
        assert len(obs.plans_completed) == 1


class TestAdapter:
    def test_per_op_merge_observer_forwards_progress_and_warnings(self):
        """per_op_merge_observer forwards on_progress and on_warning, tagged
        with the op. on_total is intentionally ignored -- the runner fires
        on_op_start itself with its own stat-based total."""
        plan_obs = CollectingObserver()
        op_sentinel = "op-A"
        merge_obs = per_op_merge_observer(plan_obs, op_sentinel)

        merge_obs.on_total(1024)  # no-op at the adapter level
        merge_obs.on_progress(256)
        merge_obs.on_warning("oops")

        assert plan_obs.ops_started == []  # runner's job, not adapter's
        assert plan_obs.progress == [(op_sentinel, 256)]
        assert plan_obs.warnings == [(op_sentinel, "oops")]
