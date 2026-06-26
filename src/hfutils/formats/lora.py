"""LoRA key-scheme conversion: rename tensor keys so a LoRA trained in one
framework loads in another, without touching tensor bytes.

Some trainers emit LoRA keys in a naming scheme the target loader doesn't
recognize, so every key is silently skipped at load time. The fix is a pure
rename: the tensor *data* is correct, only the keys are wrong.

This module is streaming and torch-free, like `formats.safetensors`. Only the
header changes; the tensor data region is copied verbatim. The on-disk
`data_offsets` are relative to the start of the data region, so renaming keys
(which only changes the header length) leaves every tensor's bytes valid where
they already sit -- we copy the whole data region in one pass.

Schemes are table-driven (`LORA_SCHEMES`); add a new architecture by adding a
`LoraScheme` entry. The first scheme is `krea2`: fal.ai's Krea-2 trainer emits
PEFT-style keys (`base_model.model.blocks.0.attn.wq.lora_A.weight`) that
ComfyUI's default loader rejects; the diffusers-style keys the official Krea-2
LoRAs ship with (`transformer.transformer_blocks.0.attn.to_q.lora_A.weight`)
load fine. The rename map mirrors ComfyUI's own `krea2_to_diffusers`
(comfy/utils.py) inverted, and was verified to reproduce the official
`krea2_darkbrush` LoRA's key set exactly (528/528 keys, all shapes matching).
"""

from __future__ import annotations

import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import orjson

from hfutils.errors import LoraConvertError
from hfutils.events import MergeObserver, NullMergeObserver
from hfutils.formats.safetensors import Manifest
from hfutils.inspect.safetensors import RawHeader, read_raw_header
from hfutils.io.progress import COPY_CHUNK


@dataclass(frozen=True)
class LoraScheme:
    """One source->target key-rename scheme.

    `rename(key) -> key` maps a single tensor name. `matches(header) -> bool`
    decides whether a file looks like this scheme's source format (used for
    auto-detection and to guard explicit conversions).
    """

    name: str
    description: str
    rename: Callable[[str], str]
    matches: Callable[[RawHeader], bool]


# --------------------------------------------------------------------------
# Krea-2: fal.ai trainer (PEFT) -> ComfyUI / diffusers
# --------------------------------------------------------------------------

# Singleton modules: exact prefix (with trailing dot) -> replacement.
_KREA2_SINGLES: dict[str, str] = {
    "transformer.first.": "transformer.img_in.",
    "transformer.last.linear.": "transformer.final_layer.linear.",
    "transformer.tmlp.0.": "transformer.time_embed.linear_1.",
    "transformer.tmlp.2.": "transformer.time_embed.linear_2.",
    "transformer.tproj.1.": "transformer.time_mod_proj.",
    "transformer.txtmlp.1.": "transformer.txt_in.linear_1.",
    "transformer.txtmlp.3.": "transformer.txt_in.linear_2.",
}

# Submodule renames, applied everywhere they occur (only inside block stacks).
_KREA2_SUBMODULES: dict[str, str] = {
    ".attn.wq.": ".attn.to_q.",
    ".attn.wk.": ".attn.to_k.",
    ".attn.wv.": ".attn.to_v.",
    ".attn.wo.": ".attn.to_out.0.",
    ".attn.gate.": ".attn.to_gate.",
    ".mlp.up.": ".ff.up.",
    ".mlp.down.": ".ff.down.",
    ".mlp.gate.": ".ff.gate.",
}


def _krea2_rename(key: str) -> str:
    """Map one fal Krea-2 PEFT key to its ComfyUI/diffusers equivalent."""
    k = re.sub(r"^base_model\.model\.", "transformer.", key)

    for src, dst in _KREA2_SINGLES.items():
        if k.startswith(src):
            k = dst + k[len(src):]
            break

    # Top-level block stack -> transformer_blocks (anchored to the transformer.
    # prefix so it never touches txtfusion's layerwise_blocks / refiner_blocks).
    k = k.replace("transformer.blocks.", "transformer.transformer_blocks.", 1)
    k = k.replace("transformer.txtfusion.", "transformer.text_fusion.", 1)

    for src, dst in _KREA2_SUBMODULES.items():
        k = k.replace(src, dst)
    return k


def _krea2_matches(header: RawHeader) -> bool:
    """A fal Krea-2 LoRA has the PEFT `base_model.model.` prefix and Krea-2's
    distinctive `txtfusion` text-fusion stack."""
    names = [t.name for t in header.tensors]
    has_peft_prefix = any(n.startswith("base_model.model.") for n in names)
    has_txtfusion = any(".txtfusion." in n for n in names)
    return has_peft_prefix and has_txtfusion


KREA2_FAL = LoraScheme(
    name="krea2",
    description="fal.ai Krea-2 trainer (PEFT) -> ComfyUI / diffusers",
    rename=_krea2_rename,
    matches=_krea2_matches,
)


# Registry of known schemes. Add architectures here.
LORA_SCHEMES: dict[str, LoraScheme] = {
    KREA2_FAL.name: KREA2_FAL,
}


def detect_lora_scheme(header: RawHeader) -> LoraScheme | None:
    """Return the first registered scheme whose source format matches, else None."""
    for scheme in LORA_SCHEMES.values():
        if scheme.matches(header):
            return scheme
    return None


def convert_lora_keys(
    src_path: Path,
    dst_path: Path,
    scheme: LoraScheme,
    *,
    observer: MergeObserver | None = None,
) -> Manifest:
    """Rewrite `src_path` to `dst_path`, renaming tensor keys per `scheme`.

    Tensor data, dtype, shape, and `__metadata__` are preserved; only the
    header's tensor names change. The data region is streamed verbatim, so
    peak memory is the copy buffer plus the header JSON -- never tensor data.

    Returns a `Manifest` (renamed name -> (dtype, shape)); pass it to
    `verify_output` to confirm the written file. Raises `LoraConvertError` if
    two source keys collide on the same renamed key, or on unexpected EOF.
    """
    obs: MergeObserver = observer if observer is not None else NullMergeObserver()

    raw = read_raw_header(src_path)

    new_tensors: dict[str, dict] = {}
    manifest: Manifest = {}
    origin: dict[str, str] = {}  # renamed name -> source name (for collision messages)
    for entry in raw.tensors:
        new_name = scheme.rename(entry.name)
        if new_name in origin:
            raise LoraConvertError(
                f"key collision: '{entry.name}' and '{origin[new_name]}' both "
                f"map to '{new_name}' under scheme '{scheme.name}'"
            )
        origin[new_name] = entry.name
        new_tensors[new_name] = {
            "dtype": entry.dtype,
            "shape": entry.shape,
            # offsets preserved -> the copied data region stays valid as-is.
            "data_offsets": [entry.data_offset_start, entry.data_offset_end],
        }
        manifest[new_name] = (entry.dtype, tuple(entry.shape))

    if raw.metadata:
        new_tensors["__metadata__"] = raw.metadata

    header_bytes = orjson.dumps(new_tensors)

    data_total = src_path.stat().st_size - raw.data_region_start
    obs.on_total(data_total)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        fin.seek(raw.data_region_start)
        remaining = data_total
        while remaining > 0:
            chunk = fin.read(min(COPY_CHUNK, remaining))
            if not chunk:
                raise LoraConvertError(
                    f"Unexpected EOF reading {src_path} ({remaining} bytes remaining)"
                )
            fout.write(chunk)
            remaining -= len(chunk)
            obs.on_progress(len(chunk))

    return manifest
