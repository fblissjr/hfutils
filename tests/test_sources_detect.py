"""Source detection: classify a Path into one of the handled shapes."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import save_file

from hfutils.sources.detect import SourceKind, detect_source


def _make_sharded(subdir: Path, index_name: str) -> None:
    save_file({"a.weight": torch.randn(4, 4)}, subdir / "shard-00001-of-00002.safetensors")
    save_file({"b.weight": torch.randn(4, 4)}, subdir / "shard-00002-of-00002.safetensors")
    (subdir / index_name).write_bytes(orjson.dumps({
        "metadata": {},
        "weight_map": {
            "a.weight": "shard-00001-of-00002.safetensors",
            "b.weight": "shard-00002-of-00002.safetensors",
        },
    }))


class TestDetectSource:
    def test_diffusers_pipeline(self, tmp_path):
        (tmp_path / "model_index.json").write_bytes(orjson.dumps({
            "_class_name": "TestPipeline",
            "transformer": ["diffusers", "Test"],
            "vae": ["diffusers", "AutoencoderKL"],
        }))
        (tmp_path / "transformer").mkdir()
        save_file({"w": torch.randn(2, 2)}, tmp_path / "transformer/model.safetensors")
        (tmp_path / "vae").mkdir()
        save_file({"w": torch.randn(2, 2)}, tmp_path / "vae/model.safetensors")

        src = detect_source(tmp_path)
        assert src.kind == SourceKind.DIFFUSERS_PIPELINE
        assert set(src.components) == {"transformer", "vae"}
        assert src.pipeline_meta is not None
        assert src.pipeline_meta["_class_name"] == "TestPipeline"

    def test_sharded_component_dir(self, tmp_path):
        _make_sharded(tmp_path, "model.safetensors.index.json")

        src = detect_source(tmp_path)
        assert src.kind == SourceKind.COMPONENT_DIR
        assert src.sharded
        assert len(src.shards) == 2

    def test_single_file_component_dir(self, tmp_path):
        save_file({"w": torch.randn(2, 2)}, tmp_path / "diffusion_pytorch_model.safetensors")

        src = detect_source(tmp_path)
        assert src.kind == SourceKind.COMPONENT_DIR
        assert not src.sharded
        assert len(src.shards) == 1

    def test_diffusers_naming_for_sharded_component(self, tmp_path):
        _make_sharded(tmp_path, "diffusion_pytorch_model.safetensors.index.json")

        src = detect_source(tmp_path)
        assert src.kind == SourceKind.COMPONENT_DIR
        assert src.sharded

    def test_single_safetensors_file(self, tmp_path):
        f = tmp_path / "model.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)

        src = detect_source(f)
        assert src.kind == SourceKind.SAFETENSORS_FILE
        assert src.shards == [f]

    def test_gguf_file(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)

        src = detect_source(f)
        assert src.kind == SourceKind.GGUF_FILE

    def test_missing_path(self, tmp_path):
        src = detect_source(tmp_path / "nope")
        assert src.kind == SourceKind.UNKNOWN

    def test_empty_dir(self, tmp_path):
        src = detect_source(tmp_path)
        assert src.kind == SourceKind.UNKNOWN
