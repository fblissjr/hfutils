"""Source detection: classify a Path into one of the handled variants."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import save_file

from hfutils.sources.detect import detect_source, enrich
from hfutils.sources.types import (
    ComponentSource,
    GgufFileSource,
    PipelineSource,
    PytorchDirSource,
    SafetensorsFileSource,
    UnknownSource,
)
from tests.conftest import make_sharded_component as _make_sharded


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
        assert isinstance(src, PipelineSource)
        assert set(src.components) == {"transformer", "vae"}
        assert src.pipeline_meta is not None
        assert src.pipeline_meta["_class_name"] == "TestPipeline"

    def test_sharded_component_dir(self, tmp_path):
        _make_sharded(tmp_path, index_name="model.safetensors.index.json")

        src = detect_source(tmp_path)
        assert isinstance(src, ComponentSource)
        assert src.sharded
        assert len(src.shards) == 2

    def test_single_file_component_dir(self, tmp_path):
        save_file({"w": torch.randn(2, 2)}, tmp_path / "diffusion_pytorch_model.safetensors")

        src = detect_source(tmp_path)
        assert isinstance(src, ComponentSource)
        assert not src.sharded
        assert len(src.shards) == 1

    def test_diffusers_naming_for_sharded_component(self, tmp_path):
        _make_sharded(tmp_path, index_name="diffusion_pytorch_model.safetensors.index.json")

        src = detect_source(tmp_path)
        assert isinstance(src, ComponentSource)
        assert src.sharded

    def test_single_safetensors_file(self, tmp_path):
        f = tmp_path / "model.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)

        src = detect_source(f)
        assert isinstance(src, SafetensorsFileSource)
        assert src.path == f

    def test_gguf_file(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)

        src = detect_source(f)
        assert isinstance(src, GgufFileSource)

    def test_missing_path(self, tmp_path):
        src = detect_source(tmp_path / "nope")
        assert isinstance(src, UnknownSource)

    def test_empty_dir(self, tmp_path):
        src = detect_source(tmp_path)
        assert isinstance(src, UnknownSource)

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
        assert isinstance(src, ComponentSource)
        assert src.sharded
        assert src.incomplete

    def test_has_config_flag(self, tmp_path):
        save_file({"w": torch.randn(2, 2)}, tmp_path / "model.safetensors")
        (tmp_path / "config.json").write_bytes(orjson.dumps({"model_type": "t"}))
        src = detect_source(tmp_path)
        assert isinstance(src, ComponentSource)
        assert src.has_config

    def test_pytorch_dir_detected(self, tmp_path):
        (tmp_path / "pytorch_model.bin").write_bytes(b"\x00" * 16)
        (tmp_path / "config.json").write_bytes(orjson.dumps({"model_type": "t"}))
        src = detect_source(tmp_path)
        assert isinstance(src, PytorchDirSource)
        assert src.has_config
        assert len(src.files) == 1

    def test_display_kind_labels(self, tmp_path):
        from hfutils.sources.types import display_kind

        save_file({"w": torch.randn(2, 2)}, tmp_path / "model.safetensors")
        assert display_kind(detect_source(tmp_path)) == "component"
        f = tmp_path / "loose.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        assert display_kind(detect_source(f)) == "safetensors"


class TestEnrich:
    def test_safetensors_file_enrich_reads_header(self, tmp_path):
        f = tmp_path / "m.safetensors"
        save_file({"a.weight": torch.randn(4, 4), "a.bias": torch.randn(4)}, f)

        src = detect_source(f)
        view = enrich(src)
        assert len(view.safetensors_headers) == 1
        assert view.total_file_size > 0
        assert len(view.safetensors_headers[0].tensors) == 2

    def test_safetensors_file_enrich_picks_up_parent_config(self, tmp_path):
        (tmp_path / "config.json").write_bytes(orjson.dumps({"model_type": "llama"}))
        f = tmp_path / "model.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)

        view = enrich(detect_source(f))
        assert view.config is not None
        assert view.config["model_type"] == "llama"

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

        src = detect_source(tmp_path)
        assert isinstance(src, ComponentSource)
        view = enrich(src)
        assert view.config is not None
        assert view.config["architectures"] == ["LlamaForCausalLM"]
        assert len(view.safetensors_headers) == 2
        assert view.total_file_size > 0

    def test_enrich_is_a_free_function_not_stateful(self, tmp_path):
        """enrich() is pure wrt source: you can call it twice and compare."""
        f = tmp_path / "m.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        src = detect_source(f)
        v1 = enrich(src)
        v2 = enrich(src)
        assert v1.total_file_size == v2.total_file_size
        assert len(v1.safetensors_headers) == len(v2.safetensors_headers)

    def test_empty_unknown_enrich_is_empty_view(self, tmp_path):
        src = detect_source(tmp_path)  # empty dir -> UnknownSource
        view = enrich(src)
        assert view.total_file_size == 0
        assert view.safetensors_headers == []
