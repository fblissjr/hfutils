"""Walk a directory tree for recognized model layouts.

Recognizes both flat layouts (`root/<name>/`) and HuggingFace cache layouts
(`root/models--org--name/snapshots/hash/`). Returns `(display_name, Source)`
tuples ready for `inspect.views.display_tree`.
"""

from pathlib import Path

from hfutils.sources.detect import Source, SourceKind, detect_source


def _hf_cache_display_name(snapshot_dir: Path) -> str:
    """For `models--org--name/snapshots/hash/`, return `org/name`."""
    repo_dir_name = snapshot_dir.parent.parent.name
    parts = repo_dir_name.removeprefix("models--").split("--", 1)
    return "/".join(parts) if len(parts) == 2 else parts[0]


def walk_for_models(root: Path) -> list[tuple[str, Source]]:
    """Find recognized model directories under `root`. Output order is stable."""
    found: list[tuple[str, Source]] = []

    for snap in sorted(root.glob("models--*/snapshots/*")):
        if not snap.is_dir():
            continue
        src = detect_source(snap)
        if src.kind == SourceKind.UNKNOWN:
            continue
        found.append((_hf_cache_display_name(snap), src))

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("models--"):
            continue
        src = detect_source(child)
        if src.kind == SourceKind.UNKNOWN:
            continue
        found.append((child.name, src))

    return found
