"""Tests for architecture detection from tensor names."""

import orjson

from hfutils.inspect.architecture import detect_architecture, extract_likely_triggers


class TestDetectDiffusionModels:
    def test_flux(self):
        names = [
            "double_blocks.0.img_attn.qkv.weight",
            "double_blocks.0.txt_attn.qkv.weight",
            "single_blocks.0.linear1.weight",
            "single_blocks.5.linear1.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "Flux"
        assert result.adapter_type is None

    def test_sdxl(self):
        names = [
            "conditioner.embedders.0.transformer.text_model.embeddings.weight",
            "conditioner.embedders.1.model.ln_final.weight",
            "model.diffusion_model.input_blocks.0.0.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "SDXL"

    def test_sd3(self):
        names = [
            "text_encoders.clip_g.transformer.text_model.embeddings.weight",
            "text_encoders.clip_l.transformer.text_model.embeddings.weight",
            "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.weight",
            "transformer.joint_blocks.0.weight",
            "vae.decoder.conv_in.weight",
        ]
        result = detect_architecture(names)
        assert result.family == "SD3"

    def test_sd15(self):
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


class TestLikelyTriggers:
    def test_returns_top_tokens_aggregated_across_datasets(self):
        # kohya stores ss_tag_frequency as a stringified JSON of
        # {dataset_name: {tag: count}}.
        freq = {
            "img_a": {"mychar": 30, "smiling": 15, "1girl": 12},
            "img_b": {"mychar": 20, "1girl": 18, "outdoors": 5},
        }
        metadata = {"ss_tag_frequency": orjson.dumps(freq).decode()}
        triggers = extract_likely_triggers(metadata, top=3)
        # mychar: 50, 1girl: 30, smiling: 15
        assert triggers == ["mychar", "1girl", "smiling"]

    def test_handles_flat_mapping(self):
        # Some trainers write a flat {tag: count} dict instead of nested.
        freq = {"trigger_word": 42, "background": 10}
        metadata = {"ss_tag_frequency": orjson.dumps(freq).decode()}
        triggers = extract_likely_triggers(metadata)
        assert triggers[0] == "trigger_word"

    def test_returns_none_when_absent(self):
        assert extract_likely_triggers({}) is None
        assert extract_likely_triggers({"ss_network_module": "x"}) is None

    def test_returns_none_on_malformed_json(self):
        assert extract_likely_triggers({"ss_tag_frequency": "not-json"}) is None

    def test_strips_blank_tokens(self):
        freq = {"d": {"good": 5, "": 100, "  ": 50}}
        metadata = {"ss_tag_frequency": orjson.dumps(freq).decode()}
        assert extract_likely_triggers(metadata) == ["good"]

    def test_detect_architecture_exposes_triggers(self):
        freq = {"d": {"my_trigger": 99, "other": 1}}
        metadata = {"ss_tag_frequency": orjson.dumps(freq).decode()}
        result = detect_architecture(["weight"], metadata=metadata)
        assert result.likely_triggers == ["my_trigger", "other"]
