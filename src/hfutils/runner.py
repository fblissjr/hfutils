"""PlanRunner: executes a PackPlan, dispatches events to an Observer.

The CLI wraps this with a RichObserver; library consumers pass NullObserver
(silent) or their own implementation (JSON logs, GUI, etc.).
"""

from pathlib import Path

from hfutils.events import NullObserver, Observer, per_op_merge_observer
from hfutils.formats.safetensors import (
    Manifest,
    manifest_from_shards,
    stream_merge,
)
from hfutils.io.progress import copy_chunks
from hfutils.layouts.plan import PackPlan


class PlanRunner:
    """Execute a PackPlan. Dispatches events to the configured Observer.

    Usage:
        runner = PlanRunner(observer)
        manifests = runner.run(plan)

    The runner doesn't know about rich, CLI, or typer -- it's a library
    primitive. The CLI layer wraps it with a RichObserver.
    """

    def __init__(self, observer: Observer | None = None) -> None:
        self.observer: Observer = observer if observer is not None else NullObserver()

    def run(self, plan: PackPlan) -> dict[Path, Manifest]:
        """Execute every op, return {dest: Manifest}. Raises StreamMergeError
        / OSError / etc. on failure -- caller handles user-facing messaging.

        `on_plan_complete` is invoked in a finally, so observers that hold
        resources (e.g. RichObserver owns a rich Progress context) tear down
        cleanly even if an op raises partway through."""
        self.observer.on_plan_start(plan)
        manifests: dict[Path, Manifest] = {}
        try:
            for op in plan.ops:
                op.dest.parent.mkdir(parents=True, exist_ok=True)
                self.observer.on_op_start(op, op.total_bytes)

                if op.kind == "copy":
                    src = op.shards[0]
                    copy_chunks(
                        src, op.dest,
                        on_chunk=lambda n, op=op: self.observer.on_op_progress(op, n),
                    )
                    manifest = manifest_from_shards([src])
                else:
                    manifest = stream_merge(
                        op.shards, op.dest,
                        observer=per_op_merge_observer(self.observer, op),
                    )

                manifests[op.dest] = manifest
                self.observer.on_op_complete(op, manifest)
        finally:
            self.observer.on_plan_complete(plan, manifests)

        return manifests
