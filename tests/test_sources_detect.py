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

    def test_incomplete_sharded_dir_flagged(self, tmp_path):
        # Index claims two shards, only one is present.
        save_file({"a.weight": torch.randn(2, 2)}, tmp_path / "shard-00001-of-00002.safetensors")
        (tmp_path / "model.safetensors.index.json").write_bytes(orjson.dumps({
            "metadata": {},
            "weight_map": {
                "a.weight": "shard-00001-of-00002.safetensors",
                "b.weight": "shard-00002-of-00002.safetensors",
            },
        }))
        src = detect_source(tmp_path)
        assert src.kind == SourceKind.COMPONENT_DIR
        assert src.sharded
        assert src.incomplete

    def test_has_config_flag(self, tmp_path):
        save_file({"w": torch.randn(2, 2)}, tmp_path / "model.safetensors")
        (tmp_path / "config.json").write_bytes(orjson.dumps({"model_type": "t"}))
        src = detect_source(tmp_path)
        assert src.has_config

    def test_pytorch_dir_detected(self, tmp_path):
        (tmp_path / "pytorch_model.bin").write_bytes(b"\x00" * 16)
        (tmp_path / "config.json").write_bytes(orjson.dumps({"model_type": "t"}))
        src = detect_source(tmp_path)
        assert src.kind == SourceKind.PYTORCH_DIR
        assert src.has_config

    def test_display_kind_labels(self, tmp_path):
        save_file({"w": torch.randn(2, 2)}, tmp_path / "model.safetensors")
        assert detect_source(tmp_path).display_kind() == "component"
        f = tmp_path / "loose.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        assert detect_source(f).display_kind() == "safetensors"


class TestSourceEnrich:
    def test_safetensors_file_enrich_reads_header(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"a.weight": torch.randn(4, 4), "a.bias": torch.randn(4)}, f)

        src = detect_source(f).enrich()
        assert len(src.safetensors_headers) == 1
        assert src.total_file_size > 0
        assert len(src.safetensors_headers[0].tensors) == 2

    def test_safetensors_file_enrich_picks_up_parent_config(self, tmp_path):
        (tmp_path / "config.json").write_bytes(orjson.dumps({"model_type": "llama"}))
        f = tmp_path / "model.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)

        src = detect_source(f).enrich()
        assert src.config is not None
        assert src.config["model_type"] == "llama"

    def test_component_dir_enrich(self, tmp_path):
        (tmp_path / "config.json").write_bytes(orjson.dumps({"architectures": ["LlamaForCausalLM"]}))
        save_file({"w": torch.randn(4, 4)}, tmp_path / "shard-00001-of-00002.safetensors")
        save_file({"b": torch.randn(4)}, tmp_path / "shard-00002-of-00002.safetensors")
        (tmp_path / "model.safetensors.index.json").write_bytes(orjson.dumps({
            "metadata": {},
            "weight_map": {
                "w": "shard-00001-of-00002.safetensors",
                "b": "shard-00002-of-00002.safetensors",
            },
        }))

        src = detect_source(tmp_path).enrich()
        assert src.sharded
        assert src.shard_count == 2
        assert src.config is not None
        assert src.config["architectures"] == ["LlamaForCausalLM"]
        assert len(src.safetensors_headers) == 2
        assert src.total_file_size > 0

    def test_enrich_is_idempotent(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)

        src = detect_source(f)
        src.enrich()
        headers_id = id(src.safetensors_headers)
        src.enrich()
        assert id(src.safetensors_headers) == headers_id

    def test_empty_dir_enrich_is_noop(self, tmp_path):
        src = detect_source(tmp_path)
        src.enrich()
        assert src.total_file_size == 0
        assert src.safetensors_headers == []
