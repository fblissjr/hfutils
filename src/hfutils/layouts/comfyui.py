"""ComfyUI destination layout.

Knows two things:
- Which ComfyUI subfolder each diffusers pipeline component maps to.
- Which ComfyUI subfolder each user-supplied `--as` target maps to.

Given a `Source`, produces a list of `PackOp`s that a runner can execute.
No filesystem writes happen here.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Literal

from hfutils.errors import PlanError
from hfutils.sources.types import (
    ComponentSource,
    PipelineSource,
    SafetensorsFileSource,
    Source,
)


# Diffusers component -> (ComfyUI subfolder, filename suffix).
# Suffix prevents collisions when components share --name.
DIFFUSERS_COMPONENTS: dict[str, tuple[str, str]] = {
    "transformer":    ("diffusion_models", ""),
    "vae":            ("vae",              "_vae"),
    "text_encoder":   ("text_encoders",    "_te"),
    "text_encoder_2": ("text_encoders",    "_te2"),
    "text_encoder_3": ("text_encoders",    "_te3"),
}

class ConvertTarget(str, Enum):
    """CLI-facing values for --as. Source of truth for typer validation."""
    DIFFUSION_MODEL = "diffusion_model"
    CHECKPOINT = "checkpoint"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"
    CLIP = "clip"
    LORA = "lora"


# Which ComfyUI subfolder each ConvertTarget maps to.
TARGET_FOLDERS: dict[ConvertTarget, str] = {
    ConvertTarget.DIFFUSION_MODEL: "diffusion_models",
    ConvertTarget.CHECKPOINT:      "checkpoints",
    ConvertTarget.VAE:             "vae",
    ConvertTarget.TEXT_ENCODER:    "text_encoders",
    ConvertTarget.CLIP:            "text_encoders",
    ConvertTarget.LORA:            "loras",
}


OpKind = Literal["copy", "merge"]


@dataclass
class PackOp:
    label: str                          # pipeline component name or "single"
    dest: Path                          # absolute output file
    source: Path                        # user-visible source (dir or file); display only
    shards: list[Path] = field(default_factory=list)  # one entry => copy; many => merge

    @property
    def kind(self) -> OpKind:
        return "copy" if len(self.shards) == 1 else "merge"

    @cached_property
    def total_bytes(self) -> int:
        """Sum of shard file sizes. Stat-based, cached on first access.

        Slightly overcounts merge output by each shard's header overhead
        (tens of KB) -- a safe bias for preflight and fine for progress
        bar sizing. Single source of truth; used by PackPlan.total_bytes,
        preflight, and the progress runner."""
        return sum(s.stat().st_size for s in self.shards)


def _component_dest(component: str, comfyui_root: Path, name: str) -> Path:
    folder, suffix = DIFFUSERS_COMPONENTS[component]
    return comfyui_root / folder / f"{name}{suffix}.safetensors"


def _target_dest(target: ConvertTarget, comfyui_root: Path, name: str) -> Path:
    folder = TARGET_FOLDERS.get(target)
    if folder is None:
        valid = ", ".join(t.value for t in TARGET_FOLDERS)
        raise PlanError(f"Unknown --as target '{target}'. Valid: {valid}")
    return comfyui_root / folder / f"{name}.safetensors"


def _plan_component(
    source: PipelineSource, component: str, comfyui_root: Path, name: str,
) -> PackOp | None:
    """Returns None if the component subdir has no safetensors (e.g. legacy
    .bin-only components in a mixed pipeline). Callers filter None out."""
    subdir = source.path / component
    shards = sorted(subdir.glob("*.safetensors"))
    if not shards:
        return None
    return PackOp(
        label=component,
        source=subdir,
        shards=shards,
        dest=_component_dest(component, comfyui_root, name),
    )


def plan_comfyui(
    source: Source,
    comfyui_root: Path,
    name: str,
    *,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    target: ConvertTarget | None = None,
) -> "PackPlan":
    """Build a PackPlan to pack `source` into `comfyui_root`.

    - PipelineSource: auto-discovers components; `target` is ignored.
      Components known to `DIFFUSERS_COMPONENTS` are packed; unknown components
      (scheduler/tokenizer) are skipped silently.
    - ComponentSource or SafetensorsFileSource: requires `target`.
    """
    from hfutils.layouts.plan import PackPlan

    match source:
        case PipelineSource(components=components):
            ops: list[PackOp] = []
            for component in components:
                if component not in DIFFUSERS_COMPONENTS:
                    continue
                if only and component not in only:
                    continue
                if skip and component in skip:
                    continue
                op = _plan_component(source, component, comfyui_root, name)
                if op is not None:
                    ops.append(op)
            return PackPlan(ops=ops, source=source, meta={"target": "comfyui"})

        case ComponentSource(path=p, shards=shards):
            if target is None:
                valid = ", ".join(t.value for t in TARGET_FOLDERS)
                raise PlanError(
                    f"Source is a component directory; specify destination with --as ({valid})"
                )
            return PackPlan(
                ops=[PackOp(
                    label="single", source=p, shards=list(shards),
                    dest=_target_dest(target, comfyui_root, name),
                )],
                source=source,
                meta={"target": "comfyui"},
            )

        case SafetensorsFileSource(path=p):
            if target is None:
                valid = ", ".join(t.value for t in TARGET_FOLDERS)
                raise PlanError(
                    f"Source is a single file; specify destination with --as ({valid})"
                )
            return PackPlan(
                ops=[PackOp(
                    label="single", source=p, shards=[p],
                    dest=_target_dest(target, comfyui_root, name),
                )],
                source=source,
                meta={"target": "comfyui"},
            )

        case _:
            raise PlanError(
                f"Cannot pack source of type {type(source).__name__} into ComfyUI layout"
            )


# Backward-compat alias for one release. Dropped in 0.8.
plan_pack = plan_comfyui


def plan_single(source: Source, output: Path) -> "PackPlan":
    """Build a PackPlan with one op: dump the source's safetensors into `output`.

    Callers ensure `source` is a ComponentSource or SafetensorsFileSource."""
    from hfutils.layouts.plan import PackPlan

    match source:
        case ComponentSource(path=p, shards=shards):
            op = PackOp(label="single", source=p, shards=list(shards), dest=output)
        case SafetensorsFileSource(path=p):
            op = PackOp(label="single", source=p, shards=[p], dest=output)
        case _:
            raise PlanError(
                f"plan_single expects a ComponentSource or SafetensorsFileSource; "
                f"got {type(source).__name__}"
            )
    return PackPlan(ops=[op], source=source, meta={"target": "single"})
