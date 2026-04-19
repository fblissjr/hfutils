"""Tests for component summary (pre-conversion metadata display)."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import save_file

from hfutils.inspect.summary import format_summary_lines, summarize_component


def test_single_file_summary(tmp_path):
    src = tmp_path / "m.safetensors"
    save_file({"a": torch.randn(4, 4), "b": torch.randn(2)}, src)

    s = summarize_component(src)
    assert s.file_count == 1
    assert not s.sharded
    assert s.tensor_count == 2
    assert s.total_params == 18
    assert s.dominant_dtype == "F32"
    assert s.quantization is None


def test_sharded_dir_summary(tmp_path):
    d = tmp_path / "transformer"
    d.mkdir()
    save_file({"x": torch.randn(4, 4)}, d / "model-00001-of-00002.safetensors")
    save_file({"y": torch.randn(4, 4)}, d / "model-00002-of-00002.safetensors")
    (d / "model.safetensors.index.json").write_bytes(
        orjson.dumps({"metadata": {}, "weight_map": {
            "x": "model-00001-of-00002.safetensors",
            "y": "model-00002-of-00002.safetensors",
        }})
    )

    s = summarize_component(d)
    assert s.sharded
    assert s.file_count == 2
    assert s.tensor_count == 2


def test_architecture_from_config(tmp_path):
    d = tmp_path / "transformer"
    d.mkdir()
    (d / "config.json").write_bytes(orjson.dumps({"_class_name": "ZImageTransformer2DModel"}))
    save_file({"x": torch.randn(2, 2)}, d / "model.safetensors")

    s = summarize_component(d)
    assert s.architecture == "ZImageTransformer2DModel"


def test_format_summary_lines_includes_size(tmp_path):
    src = tmp_path / "m.safetensors"
    save_file({"a": torch.randn(4, 4)}, src)
    lines = format_summary_lines(summarize_component(src))
    joined = "\n".join(lines)
    assert "size:" in joined
    assert "tensors:" in joined


def test_quantization_detected_via_dtype_label(tmp_path):
    # Hand-write an F8_E4M3 header because torch versions vary on float8 save support.
    import struct

    header = {
        "w": {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]},
    }
    body = orjson.dumps(header)
    data = b"\x00" * 16
    path = tmp_path / "q.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(data)

    s = summarize_component(path)
    assert s.quantization == "fp8_e4m3"
