"""Shared CLI error-handling helpers.

`fail` and `cli_guard` are the single source of truth for the
try/except -> print -> exit(1) pattern that used to be hand-copied across
`commands/*.py`. `cli_guard` catches `HfutilsError` (the project's own
error hierarchy -- meant to be user-facing) and `OSError` (expected
operational failures: disk full, permission denied, network errors).
Anything else is a real bug, not a user-facing condition, and is
deliberately left to propagate as a full traceback rather than being
swallowed into a fake clean error.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NoReturn, TypeVar

import typer
from rich.console import Console

from hfutils.errors import HfutilsError
from hfutils.formats.safetensors import verify_output

if TYPE_CHECKING:
    from hfutils.formats.safetensors import Manifest

console = Console()

F = TypeVar("F", bound=Callable)


def fail(msg: str, *, code: int = 1) -> NoReturn:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code)


def cli_guard(fn: F) -> F:
    """Safety net for `@app.command()` functions: catch `HfutilsError`/`OSError`
    escaping the command body and `fail()` it cleanly. Anything else (a plain
    `ValueError`/`RuntimeError`/etc.) still raises with a full traceback --
    that's a real bug, not a condition this decorator's job to hide."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (HfutilsError, OSError) as e:
            fail(str(e))
    return wrapper  # type: ignore[return-value]


def verify_and_report(dest: Path, manifest: "Manifest") -> bool:
    """Re-read `dest` and confirm it matches `manifest`; print the result.

    Returns the ok/not-ok bool rather than raising, so a caller checking
    multiple ops can collect every result (e.g. via a list comprehension)
    without an early failure short-circuiting the rest."""
    ok, error = verify_output(dest, manifest)
    if not ok:
        console.print(f"[red]Error:[/red] verify {dest.name}: {error}")
        return False
    console.print(f"[green]Verified[/green] {dest.name}: {len(manifest)} tensors match.")
    return True
