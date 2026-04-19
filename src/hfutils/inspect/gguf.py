"""GGUF header-only inspection.

Reads the KV metadata section from GGUF files using sequential reads.
Does not memmap the file or touch tensor data. No external dependencies
beyond the standard library.

GGUF v3 binary layout:
  4 bytes  magic (0x47475546 LE = "GGUF")
  4 bytes  version (uint32)
  8 bytes  tensor_count (uint64)
  8 bytes  kv_count (uint64)
  [kv_count KV pairs]
  [tensor info blocks]  -- not read
  [tensor data]         -- not read
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_GGUF_MAGIC = 0x46554747  # "GGUF" in LE
_SUPPORTED_VERSIONS = {2, 3}

# GGUF value type IDs
_UINT8 = 0
_INT8 = 1
_UINT16 = 2
_INT16 = 3
_UINT32 = 4
_INT32 = 5
_FLOAT32 = 6
_BOOL = 7
_STRING = 8
_ARRAY = 9
_UINT64 = 10
_INT64 = 11
_FLOAT64 = 12

# struct format for each scalar type
_SCALAR_FORMATS: dict[int, str] = {
    _UINT8: "<B",
    _INT8: "<b",
    _UINT16: "<H",
    _INT16: "<h",
    _UINT32: "<I",
    _INT32: "<i",
    _FLOAT32: "<f",
    _BOOL: "<B",
    _UINT64: "<Q",
    _INT64: "<q",
    _FLOAT64: "<d",
}


@dataclass
class GGUFInfo:
    architecture: str
    tensor_count: int
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    vocab_size: int | None = None
    quantization: str | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None
    rope_scaling_type: str | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    chat_template: str | None = None


def _read_string(f) -> str:
    """Read a GGUF string: uint64 length + bytes."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_value(f, vtype: int):
    """Read a single GGUF value of the given type."""
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, elem_type) for _ in range(count)]
    fmt = _SCALAR_FORMATS.get(vtype)
    if fmt is None:
        msg = f"Unknown GGUF value type: {vtype}"
        raise ValueError(msg)
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def read_gguf_header(path: Path) -> GGUFInfo:
    """Read metadata from a GGUF file header without touching tensor data."""
    path = Path(path)
    with open(path, "rb") as f:
        # Header
        header_bytes = f.read(24)
        if len(header_bytes) < 24:
            msg = f"File too small to be GGUF: {path}"
            raise ValueError(msg)

        magic, version, tensor_count, kv_count = struct.unpack("<IIQ Q", header_bytes)

        if magic != _GGUF_MAGIC:
            msg = f"Invalid GGUF magic: 0x{magic:08X} (expected 0x{_GGUF_MAGIC:08X})"
            raise ValueError(msg)
        if version not in _SUPPORTED_VERSIONS:
            msg = f"Unsupported GGUF version: {version}"
            raise ValueError(msg)

        # Read KV pairs
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = _read_value(f, vtype)
            metadata[key] = value

    arch = str(metadata.get("general.architecture", "unknown"))

    def _int(key: str) -> int | None:
        v = metadata.get(key)
        return int(v) if v is not None else None

    def _float(key: str) -> float | None:
        v = metadata.get(key)
        return float(v) if v is not None else None

    def _str(key: str) -> str | None:
        v = metadata.get(key)
        return str(v) if v is not None else None

    quant = metadata.get("general.file_type")

    return GGUFInfo(
        architecture=arch,
        tensor_count=tensor_count,
        context_length=_int(f"{arch}.context_length"),
        embedding_length=_int(f"{arch}.embedding_length"),
        block_count=_int(f"{arch}.block_count"),
        vocab_size=_int(f"{arch}.vocab_size"),
        quantization=str(quant) if quant is not None else None,
        rope_freq_base=_float(f"{arch}.rope.freq_base"),
        rope_freq_scale=_float(f"{arch}.rope.scaling.factor"),
        rope_scaling_type=_str(f"{arch}.rope.scaling.type"),
        bos_token_id=_int("tokenizer.ggml.bos_token_id"),
        eos_token_id=_int("tokenizer.ggml.eos_token_id"),
        chat_template=_str("tokenizer.chat_template"),
    )
