"""Tests for safetensors header-only inspection."""

import struct
from pathlib import Path

import orjson
import pytest

from hfutils.inspect.safetensors import read_header
from hfutils.inspect.common import SafetensorsHeader


def _make_safetensors(tmp: Path, header_dict: dict) -> Path:
    """Create a minimal safetensors file with the given header JSON."""
    header_bytes = orjson.dumps(header_dict)
    max_offset = 0
    for key, val in header_dict.items():
        if key == "__metadata__":
            continue
        end = val["data_offsets"][1]
        if end > max_offset:
            max_offset = end

    path = tmp / "model.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * max_offset)
    return path


class TestReadHeader:
    def test_single_tensor_fp16(self, tmp_path):
        header_dict = {
            "weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        }
        path = _make_safetensors(tmp_path, header_dict)
        result = read_header(path)

        assert isinstance(result, SafetensorsHeader)
        assert len(result.tensors) == 1
        t = result.tensors[0]
        assert t.name == "weight"
        assert t.shape == [4, 4]
        assert t.dtype == "F16"
        assert t.param_count == 16
        assert t.size_bytes == 32  # 16 params * 2 bytes

    def test_multiple_tensors(self, tmp_path):
        header_dict = {
            "layer.0.weight": {
                "dtype": "F32",
                "shape": [10, 10],
                "data_offsets": [0, 400],
            },
            "layer.0.bias": {
                "dtype": "F32",
                "shape": [10],
                "data_offsets": [400, 440],
            },
        }
        path = _make_safetensors(tmp_path, header_dict)
        result = read_header(path)

        assert len(result.tensors) == 2
        assert result.total_params == 110
        assert result.total_size_bytes == 440

    def test_metadata_extracted(self, tmp_path):
        header_dict = {
            "__metadata__": {"format": "pt", "source": "test"},
            "weight": {
                "dtype": "BF16",
                "shape": [8],
                "data_offsets": [0, 16],
            },
        }
        path = _make_safetensors(tmp_path, header_dict)
        result = read_header(path)

        assert result.metadata == {"format": "pt", "source": "test"}
        assert len(result.tensors) == 1

    def test_dtype_breakdown(self, tmp_path):
        header_dict = {
            "w1": {"dtype": "F16", "shape": [100], "data_offsets": [0, 200]},
            "w2": {"dtype": "F16", "shape": [50], "data_offsets": [200, 300]},
            "b1": {"dtype": "F32", "shape": [10], "data_offsets": [300, 340]},
        }
        path = _make_safetensors(tmp_path, header_dict)
        result = read_header(path)

        breakdown = result.dtype_breakdown()
        assert len(breakdown) == 2
        assert breakdown[0].dtype == "F16"
        assert breakdown[0].param_count == 150
        assert breakdown[0].tensor_count == 2
        assert breakdown[0].size_bytes == 300
        assert breakdown[1].dtype == "F32"
        assert breakdown[1].param_count == 10

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_header(Path("/nonexistent/model.safetensors"))

    def test_empty_model(self, tmp_path):
        """A safetensors file with only metadata and no tensors."""
        header_dict = {
            "__metadata__": {"format": "pt"},
        }
        header_bytes = orjson.dumps(header_dict)
        path = tmp_path / "empty.safetensors"
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)

        result = read_header(path)
        assert len(result.tensors) == 0
        assert result.total_params == 0
