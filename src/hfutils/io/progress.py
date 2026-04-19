"""Shared rich progress helpers used by merge, copy, and pack commands."""

from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

_COPY_CHUNK = 4 * 1024 * 1024  # 4 MiB


def make_progress(console: Console) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def copy_with_progress(src: Path, dst: Path, console: Console) -> None:
    """Byte-accurate progress-tracked file copy."""
    total = src.stat().st_size
    with make_progress(console) as progress:
        task = progress.add_task(f"copy {src.name}", total=total)
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            while True:
                chunk = fi.read(_COPY_CHUNK)
                if not chunk:
                    break
                fo.write(chunk)
                progress.update(task, advance=len(chunk))
