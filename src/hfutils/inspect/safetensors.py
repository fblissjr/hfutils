"""Safetensors header-only inspection.

Reads the binary header from .safetensors files without loading tensor data.
Format: 8-byte LE uint64 (header length) + JSON header + tensor data.
No torch or safetensors library needed.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import orjson

from hfutils.inspect.common import SafetensorsHeader, TensorInfo


@dataclass
class RawTensorEntry:
    """One tensor as it appears on disk, including byte offsets in the data region."""
    name: str
    dtype: str
    shape: list[int]
    data_offset_start: int  # relative to start of data region
    data_offset_end: int    # relative to start of data region


@dataclass
class RawHeader:
    """Low-level header for streaming I/O. Preserves insertion order and byte offsets."""
    header_length: int                  # bytes of the JSON header
    data_region_start: int              # file offset where tensor data begins (== 8 + header_length)
    metadata: dict[str, str]
    tensors: list[RawTensorEntry]


def _parse_header_bytes(header_bytes: bytes) -> tuple[dict[str, str], list[RawTensorEntry]]:
    header_dict = orjson.loads(header_bytes)
    metadata = header_dict.pop("__metadata__", {}) or {}
    tensors = [
        RawTensorEntry(
            name=name,
            dtype=info["dtype"],
            shape=info["shape"],
            data_offset_start=info["data_offsets"][0],
            data_offset_end=info["data_offsets"][1],
        )
        for name, info in header_dict.items()
    ]
    return metadata, tensors


def read_raw_header(path: Path) -> RawHeader:
    """Read the full header including per-tensor data_offsets.

    Used by streaming writers that need to seek into each tensor's data bytes.
    """
    path = Path(path)
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            raise ValueError(f"File too small to be safetensors: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        header_bytes = f.read(header_length)

    metadata, tensors = _parse_header_bytes(header_bytes)
    return RawHeader(
        header_length=header_length,
        data_region_start=8 + header_length,
        metadata=metadata,
        tensors=tensors,
    )


def read_header(path: Path) -> SafetensorsHeader:
    """Read and parse the header from a safetensors file.

    Only reads the header bytes -- tensor data is never loaded.
    """
    raw = read_raw_header(path)
    tensors = [TensorInfo(name=t.name, shape=t.shape, dtype=t.dtype) for t in raw.tensors]
    return SafetensorsHeader(tensors=tensors, metadata=raw.metadata)
