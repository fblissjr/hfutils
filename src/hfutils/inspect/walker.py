"""Walk a directory tree for recognized model layouts.

Recognizes both flat layouts (`root/<name>/`) and HuggingFace cache layouts
(`root/models--org--name/snapshots/hash/`). Returns `(display_name, Source)`
tuples ready for `inspect.views.display_tree`.

Per-directory classification (detect_source) is pure I/O with no shared
state, so we parallelize via a thread pool. Output order is stabilized by
sorting on the display name after the fan-in.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hfutils.sources.detect import Source, SourceKind, detect_source

_WALKER_WORKERS = 8


def _hf_cache_display_name(snapshot_dir: Path) -> str:
    """For `models--org--name/snapshots/hash/`, return `org/name`."""
    repo_dir_name = snapshot_dir.parent.parent.name
    parts = repo_dir_name.removeprefix("models--").split("--", 1)
    return "/".join(parts) if len(parts) == 2 else parts[0]


def _candidate_paths(root: Path) -> list[tuple[str, Path]]:
    """Enumerate (display_name, path) tuples we'll hand to detect_source."""
    out: list[tuple[str, Path]] = []
    for snap in root.glob("models--*/snapshots/*"):
        if snap.is_dir():
            out.append((_hf_cache_display_name(snap), snap))
    for child in root.iterdir():
        if child.is_dir() and not child.name.startswith("models--"):
            out.append((child.name, child))
    return out


def walk_for_models(root: Path) -> list[tuple[str, Source]]:
    """Find recognized model directories under `root`.

    Output is sorted by display name; thread completion order does not affect
    the result.
    """
    candidates = _candidate_paths(root)
    if not candidates:
        return []

    with ThreadPoolExecutor(max_workers=_WALKER_WORKERS) as pool:
        sources = list(pool.map(lambda pair: detect_source(pair[1]), candidates))

    found = [
        (name, src) for (name, _), src in zip(candidates, sources)
        if src.kind != SourceKind.UNKNOWN
    ]
    found.sort(key=lambda entry: entry[0])
    return found
