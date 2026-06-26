"""LoRA key-scheme conversion: rename correctness, data integrity, detection."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from hfutils.errors import LoraConvertError
from hfutils.formats.lora import (
    KREA2_FAL,
    LORA_SCHEMES,
    LoraScheme,
    convert_lora_keys,
    detect_lora_scheme,
)
from hfutils.formats.safetensors import verify_output
from hfutils.inspect.safetensors import read_raw_header
from tests.conftest import fal_krea2_lora_keys, make_fal_krea2_lora

# Exact source->target pairs covering every rename rule. An independent spec:
# if the converter drifts, these break. Derived from ComfyUI's krea2_to_diffusers.
KREA2_PAIRS = [
    ("base_model.model.blocks.0.attn.wq.lora_A.weight",
     "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"),
    ("base_model.model.blocks.1.attn.wk.lora_B.weight",
     "transformer.transformer_blocks.1.attn.to_k.lora_B.weight"),
    ("base_model.model.blocks.0.attn.wv.lora_A.weight",
     "transformer.transformer_blocks.0.attn.to_v.lora_A.weight"),
    ("base_model.model.blocks.0.attn.wo.lora_A.weight",
     "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight"),
    ("base_model.model.blocks.0.attn.gate.lora_A.weight",
     "transformer.transformer_blocks.0.attn.to_gate.lora_A.weight"),
    ("base_model.model.blocks.0.mlp.up.lora_A.weight",
     "transformer.transformer_blocks.0.ff.up.lora_A.weight"),
    ("base_model.model.blocks.0.mlp.down.lora_A.weight",
     "transformer.transformer_blocks.0.ff.down.lora_A.weight"),
    ("base_model.model.blocks.0.mlp.gate.lora_A.weight",
     "transformer.transformer_blocks.0.ff.gate.lora_A.weight"),
    ("base_model.model.first.lora_A.weight",
     "transformer.img_in.lora_A.weight"),
    ("base_model.model.last.linear.lora_A.weight",
     "transformer.final_layer.linear.lora_A.weight"),
    ("base_model.model.tmlp.0.lora_A.weight",
     "transformer.time_embed.linear_1.lora_A.weight"),
    ("base_model.model.tmlp.2.lora_A.weight",
     "transformer.time_embed.linear_2.lora_A.weight"),
    ("base_model.model.tproj.1.lora_A.weight",
     "transformer.time_mod_proj.lora_A.weight"),
    ("base_model.model.txtmlp.1.lora_A.weight",
     "transformer.txt_in.linear_1.lora_A.weight"),
    ("base_model.model.txtmlp.3.lora_A.weight",
     "transformer.txt_in.linear_2.lora_A.weight"),
    ("base_model.model.txtfusion.projector.lora_A.weight",
     "transformer.text_fusion.projector.lora_A.weight"),
    ("base_model.model.txtfusion.layerwise_blocks.0.attn.wo.lora_A.weight",
     "transformer.text_fusion.layerwise_blocks.0.attn.to_out.0.lora_A.weight"),
    ("base_model.model.txtfusion.layerwise_blocks.1.mlp.down.lora_B.weight",
     "transformer.text_fusion.layerwise_blocks.1.ff.down.lora_B.weight"),
    ("base_model.model.txtfusion.refiner_blocks.0.attn.gate.lora_A.weight",
     "transformer.text_fusion.refiner_blocks.0.attn.to_gate.lora_A.weight"),
    ("base_model.model.txtfusion.refiner_blocks.1.mlp.up.lora_A.weight",
     "transformer.text_fusion.refiner_blocks.1.ff.up.lora_A.weight"),
]


class TestKrea2Rename:
    @pytest.mark.parametrize("src,dst", KREA2_PAIRS)
    def test_known_pairs(self, src, dst):
        assert KREA2_FAL.rename(src) == dst

    def test_no_forbidden_tokens_remain(self):
        """Every fal key renames into the transformer.* diffusers namespace with
        no source-only tokens left behind."""
        forbidden = ["base_model", "txtfusion", ".mlp.", ".wq.", ".wk.", ".wv.",
                     ".wo.", "tmlp", "tproj", ".attn.gate."]
        for key in fal_krea2_lora_keys(n_blocks=3):
            out = KREA2_FAL.rename(key)
            assert out.startswith("transformer."), out
            for tok in forbidden:
                assert tok not in out, f"{tok!r} survived in {out!r} (from {key!r})"

    def test_rename_is_injective(self):
        keys = fal_krea2_lora_keys(n_blocks=4)
        renamed = [KREA2_FAL.rename(k) for k in keys]
        assert len(set(renamed)) == len(renamed)


class TestDetectScheme:
    def test_detects_krea2_from_fal_lora(self, tmp_path):
        f = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(f)
        scheme = detect_lora_scheme(read_raw_header(f))
        assert scheme is not None and scheme.name == "krea2"

    def test_returns_none_for_unrelated_file(self, tmp_path):
        f = tmp_path / "plain.safetensors"
        save_file({"w": torch.randn(2, 2)}, f)
        assert detect_lora_scheme(read_raw_header(f)) is None

    def test_krea2_registered(self):
        assert "krea2" in LORA_SCHEMES


class TestConvertLoraKeys:
    def test_keys_renamed_and_loadable(self, tmp_path):
        src = tmp_path / "fal.safetensors"
        src_keys = make_fal_krea2_lora(src)
        dst = tmp_path / "out.safetensors"

        manifest = convert_lora_keys(src, dst, KREA2_FAL)

        loaded = load_file(str(dst))
        expected = {KREA2_FAL.rename(k) for k in src_keys}
        assert set(loaded.keys()) == expected
        assert set(manifest.keys()) == expected

    def test_tensor_data_byte_identical(self, tmp_path):
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)
        dst = tmp_path / "out.safetensors"
        convert_lora_keys(src, dst, KREA2_FAL)

        src_t = load_file(str(src))
        dst_t = load_file(str(dst))
        for src_name, tensor in src_t.items():
            renamed = KREA2_FAL.rename(src_name)
            assert torch.equal(dst_t[renamed], tensor), f"data changed for {src_name}"

    def test_dtype_and_shape_preserved(self, tmp_path):
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)
        dst = tmp_path / "out.safetensors"
        convert_lora_keys(src, dst, KREA2_FAL)

        before = {t.name: (t.dtype, tuple(t.shape)) for t in read_raw_header(src).tensors}
        after = {t.name: (t.dtype, tuple(t.shape)) for t in read_raw_header(dst).tensors}
        for name, spec in before.items():
            assert after[KREA2_FAL.rename(name)] == spec

    def test_metadata_preserved(self, tmp_path):
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src, metadata={"base_model": "fal/krea-2", "lora_rank": "32"})
        dst = tmp_path / "out.safetensors"
        convert_lora_keys(src, dst, KREA2_FAL)

        assert read_raw_header(dst).metadata == {"base_model": "fal/krea-2", "lora_rank": "32"}

    def test_manifest_verifies(self, tmp_path):
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)
        dst = tmp_path / "out.safetensors"
        manifest = convert_lora_keys(src, dst, KREA2_FAL)

        ok, problem = verify_output(dst, manifest)
        assert ok, problem

    def test_observer_receives_byte_total_and_progress(self, tmp_path):
        from hfutils.events import CollectingMergeObserver

        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)
        dst = tmp_path / "out.safetensors"
        obs = CollectingMergeObserver()
        convert_lora_keys(src, dst, KREA2_FAL, observer=obs)

        data_total = src.stat().st_size - read_raw_header(src).data_region_start
        assert obs.total == data_total
        assert obs.advanced == data_total

    def test_collision_raises(self, tmp_path):
        # A degenerate scheme that collapses two distinct keys onto one target.
        src = tmp_path / "two.safetensors"
        save_file({"a.weight": torch.randn(2, 2), "b.weight": torch.randn(2, 2)}, src)
        collapse = LoraScheme(
            name="collapse",
            description="test: maps everything to the same key",
            rename=lambda k: "same.weight",
            matches=lambda h: True,
        )
        with pytest.raises(LoraConvertError, match="collision"):
            convert_lora_keys(src, tmp_path / "out.safetensors", collapse)
