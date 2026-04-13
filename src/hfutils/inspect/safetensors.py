"""Safetensors header-only inspection.

Reads the binary header from .safetensors files without loading tensor data.
Format: 8-byte LE uint64 (header length) + JSON header + tensor data.
No torch or safetensors library needed.
"""

import struct
from pathlib import Path

import orjson

from hfutils.inspect.common import SafetensorsHeader, TensorInfo


def read_header(path: Path) -> SafetensorsHeader:
    """Read and parse the header from a safetensors file.

    Only reads the header bytes -- tensor data is never loaded.
    """
    path = Path(path)
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            msg = f"File too small to be safetensors: {path}"
            raise ValueError(msg)

        header_length = struct.unpack("<Q", length_bytes)[0]
        header_bytes = f.read(header_length)

    header_dict = orjson.loads(header_bytes)

    metadata = header_dict.pop("__metadata__", {})

    tensors = []
    for name, info in header_dict.items():
        tensors.append(
            TensorInfo(
                name=name,
                shape=info["shape"],
                dtype=info["dtype"],
            )
        )

    return SafetensorsHeader(tensors=tensors, metadata=metadata)
