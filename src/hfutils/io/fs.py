"""Filesystem-level helpers shared across commands."""

import shutil
from pathlib import Path

from hfutils.errors import InsufficientSpaceError
from hfutils.inspect.common import format_size

__all__ = ["InsufficientSpaceError", "check_free_space", "resolve_existing_ancestor"]


def resolve_existing_ancestor(path: Path) -> Path | None:
    """Walk up from `path` to the nearest existing ancestor (itself if it
    already exists). Returns None if even the filesystem root doesn't exist
    (pathological). Shared by `check_free_space` and callers that need to
    probe a destination's filesystem device before it's been created."""
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            return None
        probe = probe.parent
    return probe


def check_free_space(dest: Path, required: int, *, label: str | None = None) -> None:
    """Raise `InsufficientSpaceError` if `dest`'s filesystem can't hold `required` bytes.

    Resolves to the nearest existing ancestor when `dest` itself doesn't exist
    (common case: the output directory hasn't been created yet). The error
    message names `label` instead of `dest` when given -- useful when a
    caller has grouped several destinations onto one filesystem device and
    `dest` is just one representative probe path, not the full picture.
    """
    probe = resolve_existing_ancestor(dest)
    if probe is None:
        return  # reached root; let the OS error naturally

    free = shutil.disk_usage(probe).free
    if free < required:
        raise InsufficientSpaceError(
            f"Not enough free space at {label if label is not None else dest}: "
            f"need {format_size(required)}, have {format_size(free)}."
        )
