"""Diffusers pipelines in the wild often mix .safetensors and legacy .bin
components. convert comfyui should pack the safetensors parts normally and
surface a clear warning about the .bin components rather than crashing or
silently skipping them."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from hfutils.cli import app
from hfutils.sources.detect import SourceKind, detect_source

runner = CliRunner()


def _make_mixed_pipeline(tmp: Path) -> None:
    """transformer = sharded safetensors; vae = pytorch_model.bin (legacy);
    text_encoder = single-file safetensors."""
    (tmp / "model_index.json").write_bytes(orjson.dumps({
        "_class_name": "MixedPipeline",
        "transformer": ["diffusers", "T"],
        "vae": ["diffusers", "V"],
        "text_encoder": ["transformers", "E"],
    }))
    transformer = tmp / "transformer"
    transformer.mkdir()
    save_file({"w": torch.randn(2, 2)}, transformer / "model.safetensors")

    vae = tmp / "vae"
    vae.mkdir()
    (vae / "pytorch_model.bin").write_bytes(b"\x00" * 1024)

    te = tmp / "text_encoder"
    te.mkdir()
    save_file({"emb": torch.randn(2, 2)}, te / "model.safetensors")


class TestMixedPipeline:
    def test_detect_source_counts_bin_component(self, tmp_path):
        _make_mixed_pipeline(tmp_path)
        src = detect_source(tmp_path)
        assert src.kind == SourceKind.DIFFUSERS_PIPELINE
        # vae has a .bin file; detection treats it as a weight-bearing component
        assert "vae" in src.components
        assert "transformer" in src.components
        assert "text_encoder" in src.components

    def test_convert_comfyui_skips_bin_components(self, tmp_path):
        _make_mixed_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", "comfyui", str(tmp_path), str(comfy),
            "--name", "M",
        ])
        # Transformer + text_encoder pack cleanly; vae/pytorch_model.bin is
        # silently skipped because the planner only routes safetensors.
        assert result.exit_code == 0, result.output
        assert (comfy / "diffusion_models/M.safetensors").exists()
        assert (comfy / "text_encoders/M_te.safetensors").exists()
        # The .bin vae has no .safetensors; plan_pack produces no op for it.
        assert not (comfy / "vae" / "M_vae.safetensors").exists()

    def test_convert_comfyui_with_only_vae_fails_cleanly(self, tmp_path):
        _make_mixed_pipeline(tmp_path)
        comfy = tmp_path / "comfy"

        result = runner.invoke(app, [
            "convert", "comfyui", str(tmp_path), str(comfy),
            "--name", "M", "--only", "vae",
        ])
        # The vae component has no safetensors, so the plan is empty.
        assert result.exit_code != 0
        assert "Nothing to pack" in result.output
