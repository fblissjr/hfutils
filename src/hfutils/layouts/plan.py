"""The `PackPlan` object: a list of ops plus metadata and validation.

Every `plan_*` function returns a `PackPlan`. The runner consumes it; the
CLI shows it via its observer on `on_plan_start`.
"""

from dataclasses import dataclass, field
from pathlib import Path

from hfutils.layouts.comfyui import PackOp
from hfutils.sources.types import Source


@dataclass
class PackPlan:
    ops: list[PackOp]
    source: Source
    # Free-form metadata the planner can stash (e.g. target=`comfyui`, `single`).
    meta: dict[str, str] = field(default_factory=dict)
    # Pipeline components model_index.json declares but that have zero weight
    # files on disk at all -- no PackOp is ever built for these, so there's
    # nothing to attach a per-op `problem` to. A hard stop just like
    # broken_ops(), enforced alongside it in PlanRunner.run().
    missing: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """Sum of op byte totals. Each op caches its own total, so validate()-only
        callers don't pay for stats they don't need."""
        return sum(op.total_bytes for op in self.ops)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems with this plan.

        Empty list means the plan is ready to run. Callers decide whether
        problems are fatal or warnings."""
        problems: list[str] = []
        seen_dests: set[Path] = set()
        for op in self.ops:
            if not op.shards:
                problems.append(f"{op.label}: no shards to write")
            if op.dest in seen_dests:
                problems.append(f"{op.dest} is a duplicate destination")
            seen_dests.add(op.dest)
        return problems

    def broken_ops(self) -> list[PackOp]:
        """Ops whose source is incomplete/corrupt (op.problem is set).

        Unlike validate()'s advisory problems, these are a hard stop --
        PlanRunner refuses to run a plan containing any."""
        return [op for op in self.ops if op.problem]

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self):
        return iter(self.ops)

    def __bool__(self) -> bool:
        return bool(self.ops)
