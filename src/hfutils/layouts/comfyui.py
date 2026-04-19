"""ComfyUI destination layout.

Knows two things:
- Which ComfyUI subfolder each diffusers pipeline component maps to.
- Which ComfyUI subfolder each user-supplied `--as` target maps to.

Given a `Source`, produces a list of `PackOp`s that a runner can execute.
No filesystem writes happen here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from hfutils.sources.detect import Source, SourceKind


# Diffusers component -> (ComfyUI subfolder, filename suffix).
# Suffix prevents collisions when components share --name.
DIFFUSERS_COMPONENTS: dict[str, tuple[str, str]] = {
    "transformer":    ("diffusion_models", ""),
    "vae":            ("vae",              "_vae"),
    "text_encoder":   ("text_encoders",    "_te"),
    "text_encoder_2": ("text_encoders",    "_te2"),
    "text_encoder_3": ("text_encoders",    "_te3"),
}

# User-supplied --as target -> ComfyUI subfolder.
TARGET_FOLDERS: dict[str, str] = {
    "diffusion_model": "diffusion_models",
    "checkpoint":      "checkpoints",
    "vae":             "vae",
    "text_encoder":    "text_encoders",
    "clip":            "text_encoders",
    "lora":            "loras",
}


OpKind = Literal["copy", "merge"]


@dataclass
class PackOp:
    label: str                          # pipeline component name or "single"
    dest: Path                          # absolute output file
    shards: list[Path] = field(default_factory=list)  # one entry => copy; many => merge
    # `source` points at the user-visible source (a file or a directory) for
    # display; operations consume `shards` directly.
    source: Path | None = None

    @property
    def kind(self) -> OpKind:
        return "copy" if len(self.shards) == 1 else "merge"


def _component_dest(component: str, comfyui_root: Path, name: str) -> Path:
    folder, suffix = DIFFUSERS_COMPONENTS[component]
    return comfyui_root / folder / f"{name}{suffix}.safetensors"


def _target_dest(target: str, comfyui_root: Path, name: str) -> Path:
    if target not in TARGET_FOLDERS:
        raise ValueError(
            f"Unknown --as target '{target}'. Valid: {', '.join(sorted(TARGET_FOLDERS))}"
        )
    return comfyui_root / TARGET_FOLDERS[target] / f"{name}.safetensors"


def _plan_component(source: Source, component: str, comfyui_root: Path, name: str) -> PackOp:
    subdir = source.path / component
    shards = sorted(subdir.glob("*.safetensors"))
    return PackOp(
        label=component,
        source=subdir,
        shards=shards,
        dest=_component_dest(component, comfyui_root, name),
    )


def plan_pack(
    source: Source,
    comfyui_root: Path,
    name: str,
    *,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    target: str | None = None,
) -> list[PackOp]:
    """Build a list of operations to pack `source` into `comfyui_root`.

    - DIFFUSERS_PIPELINE: auto-discovers components; `target` is ignored.
      Components known to `DIFFUSERS_COMPONENTS` are packed; unknown components
      in the pipeline (scheduler/tokenizer) are skipped silently.
    - COMPONENT_DIR or SAFETENSORS_FILE: requires `target`.
    """
    if source.kind == SourceKind.DIFFUSERS_PIPELINE:
        ops: list[PackOp] = []
        for component in source.components:
            if component not in DIFFUSERS_COMPONENTS:
                continue
            if only and component not in only:
                continue
            if skip and component in skip:
                continue
            ops.append(_plan_component(source, component, comfyui_root, name))
        return ops

    if source.kind in (SourceKind.COMPONENT_DIR, SourceKind.SAFETENSORS_FILE):
        if target is None:
            valid = ", ".join(sorted(TARGET_FOLDERS))
            hint = "single file" if source.kind == SourceKind.SAFETENSORS_FILE else "component directory"
            raise ValueError(f"Source is a {hint}; specify destination with --as ({valid})")
        return [PackOp(
            label="single",
            source=source.path,
            shards=list(source.shards),
            dest=_target_dest(target, comfyui_root, name),
        )]

    raise ValueError(f"Cannot pack source kind {source.kind.value} into ComfyUI layout")
