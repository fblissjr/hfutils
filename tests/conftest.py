"""Shared test fixtures."""

from pathlib import Path

import numpy as np


def make_gguf_file(path: Path, arch: str = "llama") -> None:
    """Create a minimal GGUF file using the gguf library's writer."""
    from gguf import GGUFWriter

    writer = GGUFWriter(path, arch)
    writer.add_context_length(4096)
    writer.add_embedding_length(2048)
    t = np.zeros((2, 2), dtype=np.float32)
    writer.add_tensor("test", t)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
