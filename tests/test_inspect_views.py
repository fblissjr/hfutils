"""Tests for inspect display helpers (rich-rendered output)."""

from pathlib import Path

from rich.console import Console

from hfutils.inspect.architecture import ArchitectureInfo
from hfutils.inspect.common import SafetensorsHeader, TensorInfo
from hfutils.inspect.views import display_directory, display_safetensors
from hfutils.sources.detect import detect_source


def _capture(fn, *args, **kwargs) -> str:
    console = Console(record=True, width=200)
    fn(*args, **kwargs, console=console)
    return console.export_text()


def _header(metadata: dict[str, str] | None = None) -> SafetensorsHeader:
    return SafetensorsHeader(
        tensors=[TensorInfo(name="weight", shape=[4, 4], dtype="F16")],
        metadata=metadata or {},
    )


class TestDisplaySafetensors:
    def test_likely_triggers_rendered(self):
        arch = ArchitectureInfo(
            family="Flux",
            adapter_type="LoRA",
            likely_triggers=["mychar", "1girl", "smiling"],
        )
        out = _capture(
            display_safetensors,
            _header(),
            Path("model.safetensors"),
            False,
            arch=arch,
        )
        assert "Likely triggers" in out
        assert "mychar" in out
        assert "1girl" in out

    def test_no_trigger_section_when_absent(self):
        arch = ArchitectureInfo(family="Flux", adapter_type="LoRA")
        out = _capture(
            display_safetensors,
            _header(),
            Path("model.safetensors"),
            False,
            arch=arch,
        )
        assert "Likely triggers" not in out


class TestDisplayDirectory:
    def test_pytorch_dir_without_config_shows_files_not_false_negative(self, tmp_path):
        (tmp_path / "pytorch_model.bin").write_bytes(b"\x00" * 128)
        src = detect_source(tmp_path)

        out = _capture(display_directory, src, False)

        assert "No model files or config found" not in out
        assert "Files: 1" in out

    def test_pytorch_dir_with_config_shows_size(self, tmp_path):
        (tmp_path / "pytorch_model.bin").write_bytes(b"\x00" * 256)
        (tmp_path / "config.json").write_bytes(b'{"model_type": "test"}')
        src = detect_source(tmp_path)

        out = _capture(display_directory, src, False)

        assert "No model files or config found" not in out
        assert "Files: 1" in out
        assert "Total size" in out
