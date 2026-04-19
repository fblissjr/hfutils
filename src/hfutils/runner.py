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
from hfutils.io.progress import COPY_CHUNK
from hfutils.layouts.comfyui import PackOp
from hfutils.layouts.plan import PackPlan


def _op_total_bytes(op: PackOp) -> int:
    """Stat-based byte total per op. Overcounts merge output by each shard's
    header (tens of KB) -- fine for progress sizing and for preflight (safe
    bias toward 'not enough space')."""
    return sum(s.stat().st_size for s in op.shards)


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
        / OSError / etc. on failure -- caller handles user-facing messaging."""
        self.observer.on_plan_start(plan)

        manifests: dict[Path, Manifest] = {}
        for op in plan.ops:
            op.dest.parent.mkdir(parents=True, exist_ok=True)
            total = _op_total_bytes(op)
            self.observer.on_op_start(op, total)

            if op.kind == "copy":
                src = op.shards[0]
                with open(src, "rb") as fi, open(op.dest, "wb") as fo:
                    while True:
                        chunk = fi.read(COPY_CHUNK)
                        if not chunk:
                            break
                        fo.write(chunk)
                        self.observer.on_op_progress(op, len(chunk))
                manifest = manifest_from_shards([src])
            else:
                manifest = stream_merge(
                    op.shards, op.dest,
                    observer=per_op_merge_observer(self.observer, op),
                )

            manifests[op.dest] = manifest
            self.observer.on_op_complete(op, manifest)

        self.observer.on_plan_complete(plan, manifests)
        return manifests
