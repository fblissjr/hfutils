"""LoRA key-scheme conversion: rename correctness, data integrity, detection."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from hfutils.errors import LoraConvertError
from hfutils.events import CollectingMergeObserver
from hfutils.formats.lora import (
    KREA2_FAL,
    LORA_SCHEMES,
    LoraScheme,
    convert_lora_keys,
    detect_lora_scheme,
    fuse_lora,
)
from hfutils.formats.safetensors import verify_output
from hfutils.inspect.safetensors import read_raw_header
from tests.conftest import (
    fal_krea2_lora_keys,
    make_fal_krea2_lora,
    make_krea2_base,
)

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

    def test_detects_krea2_from_converted_diffusers_lora(self, tmp_path):
        fal = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(fal)
        conv = tmp_path / "conv.safetensors"
        convert_lora_keys(fal, conv, KREA2_FAL)
        scheme = detect_lora_scheme(read_raw_header(conv))
        assert scheme is not None and scheme.name == "krea2"

    def test_base_checkpoint_does_not_match(self, tmp_path):
        # The base has a text-fusion stack too, but no LoRA tensors.
        base = tmp_path / "base.safetensors"
        make_krea2_base(base)
        assert detect_lora_scheme(read_raw_header(base)) is None

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

    def test_malformed_header_raises_loraconverterror_not_valueerror(self, tmp_path):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"\x00\x00\x00")
        with pytest.raises(LoraConvertError):
            convert_lora_keys(bad, tmp_path / "out.safetensors", KREA2_FAL)

    def test_cleans_up_partial_output_on_failure(self, tmp_path, monkeypatch):
        """A mid-copy failure must not leave a truncated-but-header-valid
        file behind at the destination. convert_lora_keys derives its copy
        size from a live stat() of the source, so pre-truncating the source
        can't induce a mismatch -- monkeypatch the shared copy primitive
        instead to simulate a failure partway through."""
        src = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(src)
        dst = tmp_path / "out.safetensors"

        def _boom(*args, **kwargs):
            raise LoraConvertError("simulated failure")

        monkeypatch.setattr("hfutils.formats.lora.stream_region", _boom)

        with pytest.raises(LoraConvertError):
            convert_lora_keys(src, dst, KREA2_FAL)

        assert not dst.exists()

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


class TestKrea2BaseKey:
    def test_fal_and_diffusers_map_to_same_native_weight(self):
        f = KREA2_FAL.to_base_key
        assert f is not None
        assert f("base_model.model.blocks.0.attn.wq") == "blocks.0.attn.wq.weight"
        assert f("transformer.transformer_blocks.0.attn.to_q") == "blocks.0.attn.wq.weight"
        assert f("base_model.model.blocks.3.mlp.down") == "blocks.3.mlp.down.weight"
        assert f("transformer.transformer_blocks.3.ff.down") == "blocks.3.mlp.down.weight"
        assert f("transformer.img_in") == "first.weight"
        assert f("transformer.final_layer.linear") == "last.linear.weight"
        assert f("transformer.time_embed.linear_1") == "tmlp.0.weight"
        assert f("transformer.time_mod_proj") == "tproj.1.weight"
        assert f("transformer.txt_in.linear_2") == "txtmlp.3.weight"
        assert f("transformer.text_fusion.projector") == "txtfusion.projector.weight"
        assert (f("transformer.text_fusion.refiner_blocks.1.attn.to_out.0")
                == "txtfusion.refiner_blocks.1.attn.wo.weight")
        assert f("something.unrelated") is None


class TestFuseLora:
    def test_fuse_adds_low_rank_delta(self, tmp_path):
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)
        out_p = tmp_path / "fused.safetensors"

        res = fuse_lora(lora_p, base_p, out_p, KREA2_FAL, strength=1.0)
        assert res.fused == res.lora_modules and res.unmatched == []

        lora = load_file(str(lora_p))
        base = load_file(str(base_p))
        fused = load_file(str(out_p))
        key = "blocks.0.attn.wq.weight"
        a = lora["base_model.model.blocks.0.attn.wq.lora_A.weight"]
        b = lora["base_model.model.blocks.0.attn.wq.lora_B.weight"]
        expected = (base[key].to(torch.float32) + b.to(torch.float32) @ a.to(torch.float32)).to(base[key].dtype)
        assert torch.equal(fused[key], expected)

    def test_non_lora_tensors_copied_verbatim(self, tmp_path):
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        base_p = tmp_path / "base.safetensors"
        base = make_krea2_base(base_p)
        out_p = tmp_path / "fused.safetensors"
        fuse_lora(lora_p, base_p, out_p, KREA2_FAL)

        fused = load_file(str(out_p))
        assert torch.equal(fused["blocks.0.prenorm.scale"], base["blocks.0.prenorm.scale"])

    def test_names_shapes_dtypes_preserved(self, tmp_path):
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)
        out_p = tmp_path / "fused.safetensors"
        fuse_lora(lora_p, base_p, out_p, KREA2_FAL)

        b = {t.name: (t.dtype, tuple(t.shape)) for t in read_raw_header(base_p).tensors}
        o = {t.name: (t.dtype, tuple(t.shape)) for t in read_raw_header(out_p).tensors}
        assert o == b

    def test_strength_zero_is_identity(self, tmp_path):
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        base_p = tmp_path / "base.safetensors"
        base = make_krea2_base(base_p)
        out_p = tmp_path / "fused.safetensors"
        fuse_lora(lora_p, base_p, out_p, KREA2_FAL, strength=0.0)

        fused = load_file(str(out_p))
        for k, v in base.items():
            assert torch.equal(fused[k], v), f"strength 0 changed {k}"

    def test_alpha_scales_delta(self, tmp_path):
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)  # rank 2, no alpha in metadata
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)
        out_p = tmp_path / "fused.safetensors"
        # alpha=4, rank=2 -> scale = 2.0
        fuse_lora(lora_p, base_p, out_p, KREA2_FAL, alpha=4.0)

        lora = load_file(str(lora_p))
        base = load_file(str(base_p))
        fused = load_file(str(out_p))
        key = "blocks.0.attn.wq.weight"
        a = lora["base_model.model.blocks.0.attn.wq.lora_A.weight"]
        b = lora["base_model.model.blocks.0.attn.wq.lora_B.weight"]
        expected = (base[key].to(torch.float32) + 2.0 * (b.to(torch.float32) @ a.to(torch.float32))).to(base[key].dtype)
        assert torch.equal(fused[key], expected)

    def test_fp8_base_refused(self, tmp_path):
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p, dtype=torch.float8_e4m3fn)
        with pytest.raises(LoraConvertError, match="quantize"):
            fuse_lora(lora_p, base_p, tmp_path / "fused.safetensors", KREA2_FAL)

    def test_truncated_but_header_valid_lora_raises_loraconverterror(self, tmp_path):
        """A file whose header parses fine (our hand-rolled read_raw_header
        only reads the JSON header, not the data region) but whose tensor
        data is truncated fails inside safetensors.torch's own loader
        (st_load), which raises safetensors.SafetensorError -- neither
        HfutilsError nor OSError, so cli_guard can't catch it unless we
        translate it."""
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        size = lora_p.stat().st_size
        with open(lora_p, "r+b") as f:
            f.truncate(size - 20)
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)

        with pytest.raises(LoraConvertError):
            fuse_lora(lora_p, base_p, tmp_path / "fused.safetensors", KREA2_FAL)

    def test_truncated_but_header_valid_base_raises_loraconverterror(self, tmp_path):
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)
        size = base_p.stat().st_size
        with open(base_p, "r+b") as f:
            f.truncate(size - 20)

        with pytest.raises(LoraConvertError):
            fuse_lora(lora_p, base_p, tmp_path / "fused.safetensors", KREA2_FAL)

    def test_corrupt_base_does_not_delete_pre_existing_output(self, tmp_path):
        """safe_open(base_path) fails before out_path is ever opened in
        fuse_lora's combined `with` statement -- a pre-existing, unrelated
        file already sitting at out_path (e.g. a valid fuse from an earlier
        run) must survive this failure untouched, not get deleted by
        unlink_on_error for a run that never wrote a byte to it."""
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)
        size = base_p.stat().st_size
        with open(base_p, "r+b") as f:
            f.truncate(size - 20)
        out_p = tmp_path / "fused.safetensors"
        out_p.write_bytes(b"pre-existing valid fused checkpoint from an earlier run")

        with pytest.raises(LoraConvertError):
            fuse_lora(lora_p, base_p, out_p, KREA2_FAL)

        assert out_p.exists()
        assert out_p.read_bytes() == b"pre-existing valid fused checkpoint from an earlier run"

    def test_malformed_lora_header_raises_loraconverterror(self, tmp_path):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"\x00\x00\x00")
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)
        with pytest.raises(LoraConvertError):
            fuse_lora(bad, base_p, tmp_path / "fused.safetensors", KREA2_FAL)

    def test_malformed_base_header_raises_loraconverterror(self, tmp_path):
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        bad_base = tmp_path / "bad_base.safetensors"
        bad_base.write_bytes(b"\x00\x00\x00")
        with pytest.raises(LoraConvertError):
            fuse_lora(lora_p, bad_base, tmp_path / "fused.safetensors", KREA2_FAL)

    def test_cleans_up_partial_output_on_failure(self, tmp_path, monkeypatch):
        """A mid-copy failure must not leave a truncated-but-header-valid
        file behind. `prenorm.scale` (the only non-LoRA-targeted tensor in
        the fixture) always goes through the verbatim-copy branch, so
        monkeypatching the shared copy primitive triggers deterministically
        regardless of on-disk tensor order."""
        lora_p = tmp_path / "lora.safetensors"
        make_fal_krea2_lora(lora_p)
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)
        out_p = tmp_path / "fused.safetensors"

        def _boom(*args, **kwargs):
            raise LoraConvertError("simulated failure")

        monkeypatch.setattr("hfutils.formats.lora.stream_region", _boom)

        with pytest.raises(LoraConvertError):
            fuse_lora(lora_p, base_p, out_p, KREA2_FAL)

        assert not out_p.exists()

    def test_diffusers_named_lora_also_fuses(self, tmp_path):
        # Convert first, then fuse from the diffusers-named file.
        fal_p = tmp_path / "fal.safetensors"
        make_fal_krea2_lora(fal_p)
        conv_p = tmp_path / "conv.safetensors"
        convert_lora_keys(fal_p, conv_p, KREA2_FAL)
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)
        out_p = tmp_path / "fused.safetensors"

        res = fuse_lora(conv_p, base_p, out_p, KREA2_FAL)
        assert res.fused == res.lora_modules and res.unmatched == []

    def test_single_sided_module_warns_and_is_skipped(self, tmp_path):
        """A module with a base key match but only one of lora_A/lora_B present
        (malformed/truncated LoRA export) must not vanish silently -- it should
        warn and be skipped, not crash and not be counted as fused."""
        lora_p = tmp_path / "lora.safetensors"
        save_file(
            {
                "base_model.model.blocks.0.attn.wq.lora_A.weight": torch.randn(2, 2),
                "base_model.model.blocks.0.attn.wq.lora_B.weight": torch.randn(2, 2),
                # wk has a base match but only the A factor -- no B.
                "base_model.model.blocks.0.attn.wk.lora_A.weight": torch.randn(2, 2),
            },
            lora_p,
        )
        base_p = tmp_path / "base.safetensors"
        base = make_krea2_base(base_p)
        out_p = tmp_path / "fused.safetensors"
        obs = CollectingMergeObserver()

        res = fuse_lora(lora_p, base_p, out_p, KREA2_FAL, observer=obs)

        assert res.fused == 1
        assert any("wk" in w for w in obs.warnings)

        fused = load_file(str(out_p))
        assert torch.equal(fused["blocks.0.attn.wk.weight"], base["blocks.0.attn.wk.weight"])

    def test_mismatched_ab_shapes_raise_loraconverterror(self, tmp_path):
        """A/B factors with incompatible inner dimensions must raise
        LoraConvertError, not a raw torch error from an unchecked matmul."""
        lora_p = tmp_path / "lora.safetensors"
        save_file(
            {
                "base_model.model.blocks.0.attn.wq.lora_A.weight": torch.randn(2, 2),
                "base_model.model.blocks.0.attn.wq.lora_B.weight": torch.randn(2, 5),  # rank mismatch
            },
            lora_p,
        )
        base_p = tmp_path / "base.safetensors"
        make_krea2_base(base_p)
        out_p = tmp_path / "fused.safetensors"

        with pytest.raises(LoraConvertError):
            fuse_lora(lora_p, base_p, out_p, KREA2_FAL)
