"""Architecture detection from tensor names and metadata.

Uses a table-driven approach: each architecture is defined by a set of
required prefix patterns. First match wins, so order matters (more
specific patterns before more general ones).
"""

import re
from dataclasses import dataclass


@dataclass
class ArchitectureInfo:
    family: str = "Unknown"
    adapter_type: str | None = None
    training_metadata: dict[str, str] | None = None


# Each rule: (family_name, list_of_required_conditions)
# A condition is a string prefix (matched with startswith) or a compiled regex.
# First match wins, so more specific patterns must come before general ones.
_FAMILY_RULES: list[tuple[str, list]] = [
    # Hunyuan Video: double_blocks + single_blocks + token_refiner (distinguishes from Flux)
    ("Hunyuan Video", [
        "double_blocks.",
        "single_blocks.",
        "txt_in.individual_token_refiner.",
    ]),

    # Flux: double_blocks + single_blocks (after Hunyuan since Hunyuan is more specific)
    ("Flux", [
        "double_blocks.",
        "single_blocks.",
    ]),

    # Z-Image (Alibaba): all_final_layer prefix is distinctive.
    # We don't add Qwen3 here: its tensor layout is indistinguishable from
    # generic Llama-style LLMs without peeking at config.json, which
    # architecture_name_from_config already handles.
    ("Z-Image", [
        "all_final_layer.",
        re.compile(r".*adaLN_modulation"),
    ]),

    # AutoencoderKL (Stable Diffusion family VAE): encoder.down_blocks + decoder.up_blocks
    ("AutoencoderKL", [
        "encoder.down_blocks.",
        "decoder.up_blocks.",
    ]),

    # Mochi: blocks.N.attn.qkv_x (dual-stream attention with x/y split)
    ("Mochi", [
        re.compile(r"blocks\.\d+\.attn\.qkv_x\."),
        re.compile(r"blocks\.\d+\.attn\.qkv_y\."),
    ]),

    # Wan: blocks.N.self_attn + blocks.N.cross_attn
    ("Wan", [
        re.compile(r"blocks\.\d+\.self_attn\."),
        re.compile(r"blocks\.\d+\.cross_attn\."),
    ]),

    # LTX Video: transformer_blocks + caption_projection
    ("LTX Video", [
        "transformer_blocks.",
        "caption_projection.",
    ]),

    # SD3: text_encoders + transformer (but not transformer.text_model which is SD1/SDXL)
    ("SD3", [
        "text_encoders.",
        re.compile(r"transformer\.(?!text_model)"),
    ]),

    # SDXL: conditioner.embedders
    ("SDXL", [
        "conditioner.embedders.",
    ]),

    # Stable Diffusion 1.x/2.x: cond_stage_model + model.diffusion_model
    ("Stable Diffusion", [
        "cond_stage_model.",
        "model.diffusion_model.",
    ]),

    # LLM (Llama-style): model.layers.N.self_attn + model.layers.N.mlp
    ("LLM (Llama-style)", [
        re.compile(r"model\.layers\.\d+\.self_attn\."),
        re.compile(r"model\.layers\.\d+\.mlp\."),
    ]),
]


def _condition_matches(names: list[str], condition) -> bool:
    """Check if a single condition matches against the tensor name list."""
    if isinstance(condition, str):
        return any(n.startswith(condition) for n in names)
    if isinstance(condition, re.Pattern):
        return any(condition.match(n) for n in names)
    return False


def _detect_family(names: list[str]) -> str:
    """Detect model family from tensor naming patterns. First match wins."""
    for family, conditions in _FAMILY_RULES:
        if all(_condition_matches(names, c) for c in conditions):
            return family
    return "Unknown"


def _detect_adapter(names: list[str]) -> str | None:
    """Detect adapter type from tensor naming patterns."""
    has_lora_ab = any("lora_A" in n or "lora_B" in n for n in names)
    has_lora_down_up = any("_lora.down" in n or "_lora.up" in n for n in names)
    has_magnitude = any("magnitude_vector" in n or "dora_scale" in n for n in names)

    if has_magnitude and (has_lora_ab or has_lora_down_up):
        return "DoRA"
    if has_lora_ab or has_lora_down_up:
        return "LoRA"
    return None


_TRAINING_KEYS = {
    "ss_base_model_version": "base_model",
    "ss_network_module": "network_module",
    "ss_network_dim": "network_dim",
    "ss_network_alpha": "network_alpha",
    "ss_lr": "learning_rate",
    "ss_epoch": "epochs",
    "ss_steps": "steps",
    "ss_resolution": "resolution",
}


def _extract_training_metadata(metadata: dict[str, str]) -> dict[str, str] | None:
    """Extract known training-related metadata keys."""
    result = {label: metadata[raw_key] for raw_key, label in _TRAINING_KEYS.items() if raw_key in metadata}
    return result if result else None


def detect_architecture(
    tensor_names: list[str],
    metadata: dict[str, str] | None = None,
) -> ArchitectureInfo:
    """Detect architecture family, adapter type, and training metadata."""
    return ArchitectureInfo(
        family=_detect_family(tensor_names),
        adapter_type=_detect_adapter(tensor_names),
        training_metadata=_extract_training_metadata(metadata) if metadata else None,
    )


def architecture_name_from_config(config: dict | None) -> str | None:
    """Extract a model class name from a diffusers/transformers config dict."""
    if not config:
        return None
    class_name = config.get("_class_name")
    if isinstance(class_name, str):
        return class_name
    archs = config.get("architectures")
    if isinstance(archs, list) and archs:
        return str(archs[0])
    return None
