"""Filesystem-level helpers shared across commands."""

import shutil
from pathlib import Path

from hfutils.errors import InsufficientSpaceError
from hfutils.inspect.common import format_size

__all__ = ["InsufficientSpaceError", "check_free_space"]


def check_free_space(dest: Path, required: int) -> None:
    """Raise `InsufficientSpaceError` if `dest`'s filesystem can't hold `required` bytes.

    Resolves to the nearest existing ancestor when `dest` itself doesn't exist
    (common case: the output directory hasn't been created yet).
    """
    probe = dest
    while not probe.exists():
        if probe.parent == probe:
            return  # reached root; let the OS error naturally
        probe = probe.parent

    free = shutil.disk_usage(probe).free
    if free < required:
        raise InsufficientSpaceError(
            f"Not enough free space at {dest}: "
            f"need {format_size(required)}, have {format_size(free)}."
        )
