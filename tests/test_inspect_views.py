"""Tests for inspect display helpers (rich-rendered output)."""

from pathlib import Path

from rich.console import Console

from hfutils.inspect.architecture import ArchitectureInfo
from hfutils.inspect.common import SafetensorsHeader, TensorInfo
from hfutils.inspect.views import display_safetensors


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
