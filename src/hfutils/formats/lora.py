"""LoRA key-scheme conversion: rename tensor keys so a LoRA trained in one
framework loads in another, without touching tensor bytes.

Some trainers emit LoRA keys in a naming scheme the target loader doesn't
recognize, so every key is silently skipped at load time. The fix is a pure
rename: the tensor *data* is correct, only the keys are wrong.

The key-rename path (`convert_lora_keys`) is streaming and torch-free, like
`formats.safetensors`: only the header changes; the tensor data region is copied
verbatim. The on-disk `data_offsets` are relative to the start of the data
region, so renaming keys (which only changes the header length) leaves every
tensor's bytes valid where they already sit -- we copy the data region in one
pass. `fuse_lora` is the one exception -- it does real tensor math, so it is
torch-gated (the `[ml]` extra) with a lazy import.

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
from typing import TYPE_CHECKING

import orjson

from hfutils.errors import LoraConvertError
from hfutils.events import MergeObserver, NullMergeObserver
from hfutils.formats.safetensors import Manifest
from hfutils.inspect.common import QUANT_DTYPE_LABELS
from hfutils.inspect.safetensors import RawHeader, read_raw_header
from hfutils.io.progress import stream_region, unlink_on_error

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class LoraScheme:
    """One source->target key-rename scheme.

    `rename(key) -> key` maps a single tensor name. `matches(header) -> bool`
    decides whether a file looks like this scheme's source format (used for
    auto-detection and to guard explicit conversions). `to_base_key(module) ->
    base tensor name | None` maps a LoRA module (the key with its `.lora_A/B.weight`
    suffix stripped, in either source or converted naming) to the corresponding
    weight in the base checkpoint; required by `fuse_lora`, optional otherwise.
    """

    name: str
    description: str
    rename: Callable[[str], str]
    matches: Callable[[RawHeader], bool]
    to_base_key: Callable[[str], str | None] | None = None


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
    """A Krea-2 LoRA in either naming. The distinctive signal is Krea-2's
    text-fusion stack -- `txtfusion` (fal source) or `text_fusion` (converted /
    official diffusers). Require LoRA tensors too, so the base checkpoint (which
    also has a text-fusion stack) doesn't match."""
    names = [t.name for t in header.tensors]
    is_lora = any(".lora_" in n for n in names)
    has_text_fusion = any(("txtfusion" in n) or ("text_fusion" in n) for n in names)
    return is_lora and has_text_fusion


# Diffusers inner name -> native (ComfyUI checkpoint) inner name, derived by
# inverting the forward tables above (strip the `transformer.` prefix and the
# trailing dot). Deriving them keeps the two directions from drifting apart.
_KREA2_DIFFUSERS_SINGLES_INV: dict[str, str] = {
    dst[len("transformer."):-1]: src[len("transformer."):-1]
    for src, dst in _KREA2_SINGLES.items()
}
_KREA2_SUBMODULES_INV: dict[str, str] = {
    dst[:-1]: src[:-1] for src, dst in _KREA2_SUBMODULES.items()
}


def _krea2_diffusers_inner_to_native(inner: str) -> str:
    if inner in _KREA2_DIFFUSERS_SINGLES_INV:
        return _KREA2_DIFFUSERS_SINGLES_INV[inner]
    k = inner.replace("transformer_blocks.", "blocks.", 1)
    k = k.replace("text_fusion.", "txtfusion.", 1)
    for src, dst in _KREA2_SUBMODULES_INV.items():
        k = k.replace(src, dst)
    return k


def _krea2_base_key(module: str) -> str | None:
    """Map a LoRA module (no `.lora_A/B.weight` suffix) to its base checkpoint
    weight. Accepts both the fal source naming (`base_model.model.*`) and the
    converted diffusers naming (`transformer.*`); the ComfyUI Krea-2 checkpoint
    stores the native names directly."""
    if module.startswith("base_model.model."):
        native = module[len("base_model.model."):]
    elif module.startswith("transformer."):
        native = _krea2_diffusers_inner_to_native(module[len("transformer."):])
    else:
        return None
    return native + ".weight"


KREA2_FAL = LoraScheme(
    name="krea2",
    description="fal.ai Krea-2 trainer (PEFT) -> ComfyUI / diffusers",
    rename=_krea2_rename,
    matches=_krea2_matches,
    to_base_key=_krea2_base_key,
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


def _read_raw_header_or_raise(path: Path) -> RawHeader:
    """`read_raw_header`, translating malformed-input errors into
    `LoraConvertError` so callers only need to handle `HfutilsError`."""
    try:
        return read_raw_header(path)
    except (ValueError, OSError, orjson.JSONDecodeError) as e:
        raise LoraConvertError(f"could not read header of {path}: {e}") from e


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

    raw = _read_raw_header_or_raise(src_path)

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
    with unlink_on_error(dst_path) as mark_started, \
            open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        mark_started()
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        fin.seek(raw.data_region_start)
        stream_region(
            fin, fout, data_total,
            src_desc=str(src_path), on_chunk=obs.on_progress, error_cls=LoraConvertError,
        )

    return manifest


# --------------------------------------------------------------------------
# Fuse: bake a LoRA into the base checkpoint
# --------------------------------------------------------------------------

def _is_quantized_dtype(dtype: str) -> bool:
    """Quantized weights carry a separate per-tensor scale, so the stored value
    is not the real weight; fusing W + delta into them would be wrong. Uses the
    central quant-dtype table (fp8 / int8 / uint8)."""
    return dtype in QUANT_DTYPE_LABELS


@dataclass(frozen=True)
class FuseResult:
    """Summary of a `fuse_lora` run."""
    fused: int          # base weights that received a LoRA delta
    lora_modules: int   # LoRA modules discovered in the file
    unmatched: list[str]  # LoRA modules with no matching base weight


def fuse_lora(
    lora_path: Path,
    base_path: Path,
    out_path: Path,
    scheme: LoraScheme,
    *,
    strength: float = 1.0,
    alpha: float | None = None,
    observer: MergeObserver | None = None,
) -> FuseResult:
    """Bake `lora_path` into `base_path`, writing a fused checkpoint to `out_path`.

    For each base weight `W` with a matching LoRA module, writes
    `W + (alpha/rank) * strength * (B @ A)` in the base weight's own dtype; every
    other tensor is copied through byte-for-byte. The output header is identical
    to the base (same names, shapes, dtypes), so the result loads exactly like
    the base checkpoint -- now with the LoRA baked in (quantize it afterward for
    zero LoRA runtime cost).

    `alpha` defaults to the LoRA's `lora_alpha` metadata, then to the per-module
    rank (scale 1.0). Requires torch (the `[ml]` extra). Streams the base data
    region tensor by tensor, so peak memory is one weight plus its LoRA factors.

    Raises `LoraConvertError` if the scheme has no base-key map, if a matched
    base weight is a quantized dtype (fuse into the full-precision base instead),
    or on shape mismatch.
    """
    try:
        import torch
        from safetensors import SafetensorError, safe_open
        from safetensors.torch import load_file as st_load
    except ImportError as e:  # pragma: no cover - exercised via the [ml] extra
        raise LoraConvertError(
            f"fuse requires torch + safetensors (install 'hfutils[ml]'): {e}"
        ) from e

    obs: MergeObserver = observer if observer is not None else NullMergeObserver()

    if scheme.to_base_key is None:
        raise LoraConvertError(f"scheme '{scheme.name}' does not support fuse")

    lora_raw = _read_raw_header_or_raise(lora_path)
    if alpha is None:
        meta_alpha = lora_raw.metadata.get("lora_alpha")
        if meta_alpha is not None:
            try:
                alpha = float(meta_alpha)
            except (TypeError, ValueError):
                pass

    # Group LoRA tensors into modules keyed by their base weight name.
    try:
        lora_sd = st_load(str(lora_path))
    except SafetensorError as e:
        raise LoraConvertError(f"could not read LoRA checkpoint {lora_path}: {e}") from e
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora_sd.items():
        if name.endswith(".lora_A.weight"):
            module, side = name[: -len(".lora_A.weight")], "A"
        elif name.endswith(".lora_B.weight"):
            module, side = name[: -len(".lora_B.weight")], "B"
        else:
            continue
        base_key = scheme.to_base_key(module)
        if base_key is None:
            continue
        modules.setdefault(base_key, {})[side] = tensor

    base_raw = _read_raw_header_or_raise(base_path)
    base_names = {t.name for t in base_raw.tensors}

    unmatched = sorted(k for k in modules if k not in base_names)
    fuseable = {k: v for k, v in modules.items() if k in base_names and "A" in v and "B" in v}
    partial = sorted(
        k for k in modules if k in base_names and not ("A" in modules[k] and "B" in modules[k])
    )

    obs.on_total(base_path.stat().st_size - base_raw.data_region_start)
    for k in unmatched:
        obs.on_warning(f"LoRA targets '{k}' which is absent from the base; skipped")
    for k in partial:
        obs.on_warning(f"LoRA module for '{k}' is missing its A or B factor; skipped")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with unlink_on_error(out_path) as mark_started, \
                safe_open(base_path, framework="pt") as bf, \
                open(base_path, "rb") as raw_in, open(out_path, "wb") as out:
            mark_started()
            # Output header == base header verbatim (names/shapes/dtypes/offsets unchanged).
            header_bytes = raw_in.read(8 + base_raw.header_length)[8:]
            out.write(struct.pack("<Q", base_raw.header_length))
            out.write(header_bytes)

            fused = 0
            for entry in sorted(base_raw.tensors, key=lambda t: t.data_offset_start):
                size = entry.data_offset_end - entry.data_offset_start
                if entry.name in fuseable:
                    if _is_quantized_dtype(entry.dtype):
                        raise LoraConvertError(
                            f"base weight '{entry.name}' is {entry.dtype}; fuse into the "
                            f"full-precision (bf16) base, then quantize the result"
                        )
                    w = bf.get_tensor(entry.name)
                    a, b = fuseable[entry.name]["A"], fuseable[entry.name]["B"]
                    rank = a.shape[0]
                    if rank == 0 or b.shape[-1] != rank:
                        raise LoraConvertError(
                            f"'{entry.name}': incompatible LoRA factor shapes "
                            f"A={tuple(a.shape)} B={tuple(b.shape)}"
                        )
                    scale = (alpha / rank if alpha is not None else 1.0) * strength
                    delta = torch.matmul(b.to(torch.float32), a.to(torch.float32)) * scale
                    if delta.shape != tuple(w.shape):
                        raise LoraConvertError(
                            f"shape mismatch fusing '{entry.name}': base {tuple(w.shape)} "
                            f"vs delta {tuple(delta.shape)}"
                        )
                    fused_w = (w.to(torch.float32) + delta).to(w.dtype).contiguous()
                    out.write(fused_w.flatten().view(torch.uint8).numpy())
                    fused += 1
                else:
                    raw_in.seek(base_raw.data_region_start + entry.data_offset_start)
                    stream_region(raw_in, out, size, src_desc=str(base_path), error_cls=LoraConvertError)
                obs.on_progress(size)
    except SafetensorError as e:
        raise LoraConvertError(f"could not read base checkpoint {base_path}: {e}") from e

    return FuseResult(fused=fused, lora_modules=len(modules), unmatched=unmatched)
