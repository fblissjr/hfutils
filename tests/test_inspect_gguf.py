"""Tests for GGUF header inspection."""

from pathlib import Path

import numpy as np
import pytest

from hfutils.inspect.gguf import read_gguf_header, GGUFInfo


def _make_gguf_full(tmp_path: Path, arch: str = "llama") -> Path:
    """Create a GGUF file with full metadata for testing GGUF-specific fields."""
    from gguf import GGUFWriter

    path = tmp_path / "model.gguf"
    writer = GGUFWriter(path, arch)
    writer.add_context_length(4096)
    writer.add_embedding_length(2048)
    writer.add_block_count(16)
    writer.add_vocab_size(32000)

    tensor_data = np.zeros((4, 4), dtype=np.float32)
    writer.add_tensor("blk.0.attn_q.weight", tensor_data)
    writer.add_tensor("blk.0.attn_k.weight", tensor_data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return path


class TestReadGGUFHeader:
    def test_basic_metadata(self, tmp_path):
        path = _make_gguf_full(tmp_path)
        info = read_gguf_header(path)

        assert isinstance(info, GGUFInfo)
        assert info.architecture == "llama"
        assert info.context_length == 4096
        assert info.embedding_length == 2048
        assert info.block_count == 16
        assert info.vocab_size == 32000

    def test_tensor_count(self, tmp_path):
        path = _make_gguf_full(tmp_path)
        info = read_gguf_header(path)
        assert info.tensor_count == 2

    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            read_gguf_header(Path("/nonexistent/model.gguf"))

    def test_different_architecture(self, tmp_path):
        path = _make_gguf_full(tmp_path, arch="gemma")
        info = read_gguf_header(path)
        assert info.architecture == "gemma"

    def test_invalid_magic_raises(self, tmp_path):
        path = tmp_path / "bad.gguf"
        path.write_bytes(b"NOT_GGUF_DATA_HERE__pad_to_24b")
        with pytest.raises(ValueError, match="GGUF magic"):
            read_gguf_header(path)

    def test_file_too_small_raises(self, tmp_path):
        path = tmp_path / "tiny.gguf"
        path.write_bytes(b"\x00\x00")
        with pytest.raises(ValueError):
            read_gguf_header(path)
