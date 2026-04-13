"""Tests for architecture detection from tensor names."""

import struct
from pathlib import Path

import orjson
import pytest

from hfutils.inspect.architecture import detect_architecture, ArchitectureInfo


def _make_safetensors_header(tmp_path: Path, tensor_names: list[str], metadata: dict | None = None) -> Path:
    """Create a minimal safetensors file with given tensor names (all F16 [1] shape)."""
    header_dict = {}
    if metadata:
        header_dict["__metadata__"] = metadata
    offset = 0
    for name in tensor_names:
        header_dict[name] = {"dtype": "F16", "shape": [1], "data_offsets": [offset, offset + 2]}
        offset += 2
    header_bytes = orjson.dumps(header_dict)
    path = tmp_path / "model.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)
    return path


class TestDetectDiffusionModels:
    def test_flux(self, tmp_path):
        names = [
            "double_blocks.0.img_attn.qkv.weight",
            "double_blocks.0.txt_attn.qkv.weight",
            "single_blocks.0.linear1.weight",
            "single_blocks.5.linear1.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "Flux"
        assert result.adapter_type is None

    def test_sdxl(self, tmp_path):
        names = [
            "conditioner.embedders.0.transformer.text_model.embeddings.weight",
            "conditioner.embedders.1.model.ln_final.weight",
            "model.diffusion_model.input_blocks.0.0.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "SDXL"

    def test_sd3(self, tmp_path):
        names = [
            "text_encoders.clip_g.transformer.text_model.embeddings.weight",
            "text_encoders.clip_l.transformer.text_model.embeddings.weight",
            "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.weight",
            "transformer.joint_blocks.0.weight",
            "vae.decoder.conv_in.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "SD3"

    def test_sd15(self, tmp_path):
        names = [
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.0.weight",
            "model.diffusion_model.output_blocks.0.0.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "Stable Diffusion"


    def test_hunyuan_video(self):
        names = [
            "double_blocks.0.img_attn_qkv.weight",
            "double_blocks.0.txt_attn_qkv.weight",
            "single_blocks.0.linear1.weight",
            "txt_in.individual_token_refiner.blocks.0.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "Hunyuan Video"

    def test_wan(self):
        names = [
            "blocks.0.self_attn.q.weight",
            "blocks.0.self_attn.k.weight",
            "blocks.0.cross_attn.q.weight",
            "blocks.0.ffn.0.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "Wan"

    def test_mochi(self):
        names = [
            "blocks.0.attn.qkv_x.weight",
            "blocks.0.attn.qkv_y.weight",
            "blocks.0.mlp_x.w1.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "Mochi"

    def test_ltx_video(self):
        names = [
            "transformer_blocks.0.attn1.to_q.weight",
            "transformer_blocks.0.attn1.to_k.weight",
            "transformer_blocks.0.attn2.to_q.weight",
            "caption_projection.linear_1.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "LTX Video"


class TestDetectAdapters:
    def test_lora_ab(self):
        names = [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
        ]
        result = detect_architecture(names)
        assert result.adapter_type == "LoRA"

    def test_lora_down_up(self):
        names = [
            "to_k_lora.down.weight",
            "to_k_lora.up.weight",
            "to_q_lora.down.weight",
            "to_q_lora.up.weight",
        ]
        result = detect_architecture(names)
        assert result.adapter_type == "LoRA"

    def test_dora(self):
        names = [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "model.layers.0.self_attn.q_proj.lora_magnitude_vector",
        ]
        result = detect_architecture(names)
        assert result.adapter_type == "DoRA"


class TestDetectLLM:
    def test_llama_style(self):
        names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "lm_head.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "LLM (Llama-style)"

    def test_unknown(self):
        names = ["foo.bar.weight", "baz.qux.weight"]
        result = detect_architecture(names)
        assert result.family == "Unknown"


class TestMetadataParsing:
    def test_metadata_included(self):
        names = ["weight"]
        metadata = {"ss_base_model_version": "sd_v1", "ss_network_module": "networks.lora"}
        result = detect_architecture(names, metadata=metadata)
        assert result.training_metadata is not None
        assert "sd_v1" in result.training_metadata.get("base_model", "")
