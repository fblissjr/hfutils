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

from hfutils.inspect.safetensors import RawHeader, read_raw_header

_COPY_CHUNK = 4 * 1024 * 1024  # 4 MiB


def _merge_metadata(headers: list[RawHeader]) -> dict[str, str]:
    """Merge __metadata__ dicts from all shards. Last-write-wins on conflict.

    Sharded checkpoints typically replicate the same metadata across shards,
    so this is rarely ambiguous. We don't error on conflict because tools
    like transformers stamp `total_size` per shard, which is meaningless in
    the merged file anyway.
    """
    merged: dict[str, str] = {}
    for h in headers:
        merged.update(h.metadata)
    return merged


def _build_merged_header_json(
    headers: list[RawHeader],
    shard_paths: list[Path],
) -> tuple[bytes, list[tuple[Path, int, int, int]]]:
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
                raise ValueError(
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

    metadata = _merge_metadata(headers)
    if metadata:
        tensors_json["__metadata__"] = metadata

    return orjson.dumps(tensors_json), plan


def stream_merge(
    shard_paths: list[Path],
    output_path: Path,
    *,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    """Merge sharded safetensors files into one output, without loading tensors.

    Preserves each shard's tensor insertion order; shards are processed in the
    order given. Raises ValueError on duplicate tensor names across shards.

    `on_progress` is called after each chunk write with the number of bytes
    just written. Total bytes = sum of (shard_size - shard_header_overhead).
    """
    if not shard_paths:
        raise ValueError("stream_merge requires at least one shard")

    headers = [read_raw_header(p) for p in shard_paths]
    header_bytes, plan = _build_merged_header_json(headers, shard_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        data_region_start = out.tell()

        for shard_path, src_offset, size, dst_offset in plan:
            out.seek(data_region_start + dst_offset)
            with open(shard_path, "rb") as src:
                src.seek(src_offset)
                remaining = size
                while remaining > 0:
                    chunk = src.read(min(_COPY_CHUNK, remaining))
                    if not chunk:
                        raise IOError(
                            f"Unexpected EOF reading {shard_path} "
                            f"({remaining} bytes remaining)"
                        )
                    out.write(chunk)
                    remaining -= len(chunk)
                    if on_progress is not None:
                        on_progress(len(chunk))


def total_data_bytes(shard_paths: list[Path]) -> int:
    """Sum of tensor-data bytes across shards. Use to pre-size a progress bar."""
    total = 0
    for p in shard_paths:
        h = read_raw_header(p)
        for entry in h.tensors:
            total += entry.data_offset_end - entry.data_offset_start
    return total
