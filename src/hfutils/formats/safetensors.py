"""Streaming safetensors operations that bypass torch entirely.

The key primitive is `stream_merge`: combine several sharded safetensors into
one output file using only stdlib file I/O plus orjson. Peak memory is the
chunk buffer (a few MiB) plus the merged header JSON -- never the tensor data.

Format recap (LE throughout):
    [8-byte u64: header_length]
    [header_length bytes: JSON header]
    [tensor_data_bytes]

The JSON header maps tensor name -> {dtype, shape, data_offsets: [start, end]}
where offsets are relative to the start of the tensor_data region.
"""

import struct
from collections.abc import Callable
from pathlib import Path

import orjson

from hfutils.errors import StreamMergeError
from hfutils.inspect.safetensors import RawHeader, read_raw_header
from hfutils.io.progress import COPY_CHUNK

# os.copy_file_range was benchmarked and rejected (disk-write-bound, not
# user-space-bound). See internal/log/log_2026-04-19.md for numbers.


def _merge_metadata(
    headers: list[RawHeader],
    shard_paths: list[Path],
) -> tuple[dict[str, str], list[str]]:
    """Merge __metadata__ dicts from all shards. Last-write-wins on conflict.

    Returns `(merged, warnings)`. `warnings` lists human-readable messages for
    every key whose value changed between shards. Sharded checkpoints
    typically replicate the same metadata everywhere, so the warnings list is
    usually empty; when it isn't, the caller should surface it so the user
    knows the final file inherited only the last shard's value.
    """
    merged: dict[str, str] = {}
    warnings: list[str] = []
    for path, h in zip(shard_paths, headers):
        for key, value in h.metadata.items():
            if key in merged and merged[key] != value:
                warnings.append(
                    f"metadata key '{key}': {path.name} overrode prior value"
                )
            merged[key] = value
    return merged, warnings


def _build_merged_header_json(
    headers: list[RawHeader],
    shard_paths: list[Path],
) -> tuple[bytes, list[tuple[Path, int, int, int]], list[str]]:
    """Produce the output header JSON and a list of copy plans.

    Returns:
        header_bytes: the serialized JSON header (no length prefix).
        plan: list of (shard_path, data_region_start + src_offset, size, new_offset)
              describing which bytes from which shard to copy where in the output.
    """
    tensors_json: dict = {}
    plan: list[tuple[Path, int, int, int]] = []
    cursor = 0

    for shard_path, header in zip(shard_paths, headers):
        for entry in header.tensors:
            if entry.name in tensors_json:
                raise StreamMergeError(
                    f"Duplicate tensor '{entry.name}' in shards -- "
                    f"previously seen, also present in {shard_path.name}"
                )
            size = entry.data_offset_end - entry.data_offset_start
            tensors_json[entry.name] = {
                "dtype": entry.dtype,
                "shape": entry.shape,
                "data_offsets": [cursor, cursor + size],
            }
            plan.append((
                shard_path,
                header.data_region_start + entry.data_offset_start,
                size,
                cursor,
            ))
            cursor += size

    metadata, warnings = _merge_metadata(headers, shard_paths)
    if metadata:
        tensors_json["__metadata__"] = metadata

    return orjson.dumps(tensors_json), plan, warnings


# Output manifest from stream_merge: tensor name -> (dtype, tuple-of-shape).
# Consumers (e.g. verify_output) compare the output file's header against this
# instead of re-reading every input shard.
Manifest = dict[str, tuple[str, tuple[int, ...]]]


def stream_merge(
    shard_paths: list[Path],
    output_path: Path,
    *,
    on_total: Callable[[int], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> Manifest:
    """Merge sharded safetensors files into one output, without loading tensors.

    Preserves each shard's tensor insertion order; shards are processed in the
    order given. Raises ValueError on duplicate tensor names across shards.

    Returns a `Manifest` describing the tensors written (name -> (dtype,
    shape)). `verify_output` consumes this so the caller doesn't need to
    re-parse shard headers a second time.

    `on_total`, if provided, is called once with the total tensor-data byte
    count before any copying starts, so a caller can size a progress bar
    without paying for a separate header-read pass.

    `on_progress` is called after each chunk write with the number of bytes
    just written.

    `on_warning` is called for each metadata-key conflict across shards. The
    merged file inherits the last shard's value; callers usually want to
    surface these to the user.
    """
    if not shard_paths:
        raise StreamMergeError("stream_merge requires at least one shard")

    headers = [read_raw_header(p) for p in shard_paths]
    header_bytes, plan, warnings = _build_merged_header_json(headers, shard_paths)

    if on_warning is not None:
        for msg in warnings:
            on_warning(msg)

    if on_total is not None:
        on_total(sum(size for _, _, size, _ in plan))

    manifest: Manifest = {
        entry.name: (entry.dtype, tuple(entry.shape))
        for h in headers for entry in h.tensors
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)

        # `plan` is built in increasing dst_offset order, so the file position
        # is always correct after each copy -- no seeks needed on the output.
        for shard_path, src_offset, size, _dst_offset in plan:
            with open(shard_path, "rb") as src:
                src.seek(src_offset)
                remaining = size
                while remaining > 0:
                    chunk = src.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise StreamMergeError(
                            f"Unexpected EOF reading {shard_path} "
                            f"({remaining} bytes remaining)"
                        )
                    out.write(chunk)
                    remaining -= len(chunk)
                    if on_progress is not None:
                        on_progress(len(chunk))

    return manifest


def verify_output(output_path: Path, manifest: Manifest) -> tuple[bool, str | None]:
    """Re-read `output_path`'s header and confirm it matches the given manifest.

    Returns `(ok, error_message)`. `ok=True` means every tensor in the manifest
    is present in the output with the expected dtype and shape. `error_message`
    is human-readable when `ok=False`.
    """
    try:
        out = read_raw_header(output_path)
    except (ValueError, OSError, orjson.JSONDecodeError) as e:
        return False, f"unreadable output ({e})"

    out_tensors: Manifest = {
        t.name: (t.dtype, tuple(t.shape)) for t in out.tensors
    }

    missing = set(manifest) - set(out_tensors)
    extra = set(out_tensors) - set(manifest)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} tensor(s): {sorted(missing)[:3]}...")
        if extra:
            parts.append(f"unexpected {len(extra)} tensor(s): {sorted(extra)[:3]}...")
        return False, "; ".join(parts)

    for name, expected in manifest.items():
        if out_tensors[name] != expected:
            got = out_tensors[name]
            return False, (
                f"{name} dtype/shape mismatch "
                f"(expected {expected[0]} {list(expected[1])}, "
                f"got {got[0]} {list(got[1])})"
            )

    return True, None


def manifest_from_shards(shard_paths: list[Path]) -> Manifest:
    """Build a Manifest from a list of safetensors shards.

    Used by the `copy` code path (no stream_merge pass) so verify has the same
    contract on both kinds of op without re-reading in two places."""
    return {
        entry.name: (entry.dtype, tuple(entry.shape))
        for p in shard_paths
        for entry in read_raw_header(p).tensors
    }
