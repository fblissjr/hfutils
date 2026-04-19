"""Observer protocols for library-friendly instrumentation.

Two scopes:

- `MergeObserver` -- events from one `stream_merge` call. Used when a consumer
  runs a single merge and wants to plug progress / warnings into their own
  system (logs, a GUI, etc.).
- `Observer` -- events from a whole `PackPlan` execution. Used by
  `PlanRunner`. A `RichObserver` reproduces the current CLI output; library
  consumers can pass `NullObserver`, `CollectingObserver`, or a custom
  implementation.

Both protocols are structural (Protocol) so callers can pass any object that
quacks right, including callback-style adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from hfutils.formats.safetensors import Manifest
    from hfutils.layouts.comfyui import PackOp


# --------------------------------------------------------------------------
# Merge-scope (one stream_merge call)
# --------------------------------------------------------------------------


class MergeObserver(Protocol):
    """Events from a single stream_merge."""

    def on_total(self, total_bytes: int) -> None: ...
    def on_progress(self, bytes_copied: int) -> None: ...
    def on_warning(self, message: str) -> None: ...


class NullMergeObserver:
    def on_total(self, total_bytes: int) -> None: ...
    def on_progress(self, bytes_copied: int) -> None: ...
    def on_warning(self, message: str) -> None: ...


class CollectingMergeObserver:
    """Test helper: remembers events for later assertion."""

    def __init__(self) -> None:
        self.total: int | None = None
        self.advanced: int = 0
        self.warnings: list[str] = []

    def on_total(self, total_bytes: int) -> None:
        self.total = total_bytes

    def on_progress(self, bytes_copied: int) -> None:
        self.advanced += bytes_copied

    def on_warning(self, message: str) -> None:
        self.warnings.append(message)


# --------------------------------------------------------------------------
# Plan-scope (whole PackPlan, via PlanRunner)
# --------------------------------------------------------------------------


class Observer(Protocol):
    """Events from a whole PackPlan execution. `PlanRunner` drives this."""

    def on_plan_start(self, plan) -> None: ...
    def on_op_start(self, op: "PackOp", total_bytes: int) -> None: ...
    def on_op_progress(self, op: "PackOp", bytes_copied: int) -> None: ...
    def on_op_warning(self, op: "PackOp", message: str) -> None: ...
    def on_op_complete(self, op: "PackOp", manifest: "Manifest") -> None: ...
    def on_plan_complete(self, plan, manifests: "dict[Path, Manifest]") -> None: ...


class NullObserver:
    def on_plan_start(self, plan) -> None: ...
    def on_op_start(self, op: "PackOp", total_bytes: int) -> None: ...
    def on_op_progress(self, op: "PackOp", bytes_copied: int) -> None: ...
    def on_op_warning(self, op: "PackOp", message: str) -> None: ...
    def on_op_complete(self, op: "PackOp", manifest: "Manifest") -> None: ...
    def on_plan_complete(self, plan, manifests: "dict[Path, Manifest]") -> None: ...


class CollectingObserver:
    """Test helper for PlanRunner: records every event."""

    def __init__(self) -> None:
        self.plans_started: list = []
        self.ops_started: list[tuple["PackOp", int]] = []
        self.progress: list[tuple["PackOp", int]] = []
        self.warnings: list[tuple["PackOp", str]] = []
        self.ops_completed: list[tuple["PackOp", "Manifest"]] = []
        self.plans_completed: list = []

    def on_plan_start(self, plan) -> None:
        self.plans_started.append(plan)

    def on_op_start(self, op: "PackOp", total_bytes: int) -> None:
        self.ops_started.append((op, total_bytes))

    def on_op_progress(self, op: "PackOp", bytes_copied: int) -> None:
        self.progress.append((op, bytes_copied))

    def on_op_warning(self, op: "PackOp", message: str) -> None:
        self.warnings.append((op, message))

    def on_op_complete(self, op: "PackOp", manifest: "Manifest") -> None:
        self.ops_completed.append((op, manifest))

    def on_plan_complete(self, plan, manifests: "dict[Path, Manifest]") -> None:
        self.plans_completed.append((plan, manifests))


class RichObserver:
    """The CLI's Observer: renders a single overall Progress with a swappable
    per-op task, and routes warnings to the console as `[yellow]warn:[/yellow]`.

    Lives here (not in commands/convert) so library consumers can opt into
    the rich rendering without importing from the CLI module."""

    def __init__(self, console=None) -> None:
        from rich.console import Console

        from hfutils.io.progress import make_progress

        self._console = console if console is not None else Console()
        self._progress = make_progress(self._console)
        self._overall_task: int | None = None
        self._op_tasks: dict[int, int] = {}  # id(op) -> task id

    def on_plan_start(self, plan) -> None:
        self._progress.__enter__()
        self._overall_task = self._progress.add_task("overall", total=plan.total_bytes)

    def on_op_start(self, op: "PackOp", total_bytes: int) -> None:
        tid = self._progress.add_task(
            f"{op.kind:>5s} -> {op.dest.name}", total=total_bytes,
        )
        self._op_tasks[id(op)] = tid

    def on_op_progress(self, op: "PackOp", bytes_copied: int) -> None:
        tid = self._op_tasks[id(op)]
        self._progress.update(tid, advance=bytes_copied)
        if self._overall_task is not None:
            self._progress.update(self._overall_task, advance=bytes_copied)

    def on_op_warning(self, op: "PackOp", message: str) -> None:
        self._console.print(f"[yellow]warn:[/yellow] {message}")

    def on_op_complete(self, op: "PackOp", manifest: "Manifest") -> None:
        tid = self._op_tasks.pop(id(op), None)
        if tid is not None:
            self._progress.update(tid, visible=False)

    def on_plan_complete(self, plan, manifests: "dict[Path, Manifest]") -> None:
        self._progress.__exit__(None, None, None)


def per_op_merge_observer(observer: Observer, op: "PackOp") -> MergeObserver:
    """Adapt a plan-scope Observer into a merge-scope MergeObserver for one op.

    PlanRunner uses this to hand `stream_merge` an observer that forwards
    progress and warnings (tagged with the op) up to the plan-level
    observer. The runner fires `on_op_start` itself with a stat-based total,
    so `on_total` from stream_merge is intentionally ignored here."""

    class _Adapter:
        def on_total(self, total_bytes: int) -> None:
            # Runner already issued on_op_start; ignore the refined merge total.
            return

        def on_progress(self, bytes_copied: int) -> None:
            observer.on_op_progress(op, bytes_copied)

        def on_warning(self, message: str) -> None:
            observer.on_op_warning(op, message)

    return _Adapter()
