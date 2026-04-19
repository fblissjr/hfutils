"""Shared rich progress helpers used by merge, copy, and pack commands."""

from collections.abc import Callable
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

COPY_CHUNK = 4 * 1024 * 1024  # 4 MiB -- shared with formats.safetensors.stream_merge


def make_progress(console: Console) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def copy_chunks(
    src: Path, dst: Path,
    on_chunk: Callable[[int], None] | None = None,
) -> None:
    """Chunked byte-for-byte file copy. Calls `on_chunk(bytes_written)` after
    each write when provided. Single source of truth for the copy loop;
    `copy_with_progress` (rich) and `runner.PlanRunner` (Observer) both use it."""
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            chunk = fi.read(COPY_CHUNK)
            if not chunk:
                break
            fo.write(chunk)
            if on_chunk is not None:
                on_chunk(len(chunk))


def copy_with_progress(src: Path, dst: Path, console: Console) -> None:
    """Byte-accurate progress-tracked file copy."""
    total = src.stat().st_size
    with make_progress(console) as progress:
        task = progress.add_task(f"copy {src.name}", total=total)
        copy_chunks(src, dst, on_chunk=lambda n: progress.update(task, advance=n))
