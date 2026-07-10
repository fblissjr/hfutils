"""Shared rich progress helpers used by merge, copy, and pack commands."""

import contextlib
from collections.abc import Callable, Iterator
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


def stream_region(
    fin, fout, size: int, *,
    src_desc: str,
    on_chunk: Callable[[int], None] | None = None,
    error_cls: type[Exception] = OSError,
) -> None:
    """Copy `size` bytes from `fin`'s current position to `fout` in COPY_CHUNK
    blocks, raising `error_cls` on a short read. `on_chunk(n)` fires per block
    when given. Shared bounded-copy primitive for callers that read from a
    seek offset rather than a whole file (`io/progress.copy_chunks` is
    EOF-only, from offset 0)."""
    remaining = size
    while remaining > 0:
        chunk = fin.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise error_cls(f"Unexpected EOF reading {src_desc} ({remaining} bytes remaining)")
        fout.write(chunk)
        remaining -= len(chunk)
        if on_chunk is not None:
            on_chunk(len(chunk))


@contextlib.contextmanager
def unlink_on_error(path: Path) -> Iterator[None]:
    """Delete `path` if an exception escapes the block; no-op on success.

    Guards against leaving a truncated-but-header-valid file behind when a
    streaming write (stream_merge, convert_lora_keys, fuse_lora) fails
    partway through."""
    try:
        yield
    except BaseException:
        path.unlink(missing_ok=True)
        raise
