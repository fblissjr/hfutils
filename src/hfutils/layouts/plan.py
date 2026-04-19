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
    total_bytes: int = 0
    # Free-form metadata the planner can stash (e.g. target=`comfyui`, `single`).
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_bytes == 0:
            self.total_bytes = sum(
                sum(s.stat().st_size for s in op.shards)
                for op in self.ops
            )

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

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self):
        return iter(self.ops)

    def __bool__(self) -> bool:
        return bool(self.ops)
