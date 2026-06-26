Last updated: 2026-06-26

# hfutils

Local model file toolkit for safetensors and GGUF files, with CivitAI downloads, ComfyUI layout conversion, and LoRA key-scheme conversion.

## Install

```
uv add hfutils          # default: no torch dependency
uv add 'hfutils[ml]'    # add torch for tensor-level Python usage
```

## Commands

```
hfutils --version
hfutils inspect <path>                   # file, component dir, or diffusers pipeline
hfutils inspect <path> --recursive       # walk a directory tree for model dirs
hfutils inspect <path> --detail          # include full tensor list

hfutils convert <src> --to <layout>      # one convert command for all targets

hfutils lora convert <lora.safetensors>  # rename LoRA keys for a target loader
hfutils lora fuse <lora> <base> -o <out> # bake a LoRA into a base checkpoint
hfutils lora schemes                     # list known LoRA conversion schemes

hfutils civitai search|info|dl           # CivitAI sub-app (unchanged)
```

## `lora convert` examples

Some trainers emit LoRA keys in a naming scheme the target loader silently
rejects (every key logs "lora key not loaded" and the LoRA does nothing). The
fix is a pure key rename -- tensor data is copied verbatim.

```
# fal.ai Krea-2 LoRA -> ComfyUI default loader (auto-detected scheme)
hfutils lora convert <lora.safetensors>
# -> <lora>_comfy.safetensors next to the input

# Pick the output path and re-read it to confirm the rename
hfutils lora convert <lora.safetensors> -o <out.safetensors> --verify

# Force a named scheme on a file that doesn't auto-detect
hfutils lora convert <lora.safetensors> --scheme krea2 --force

# Preview the plan, write nothing
hfutils lora convert <lora.safetensors> --dry-run
```

Output matches the diffusers-style naming the official Krea-2 LoRAs ship with,
so ComfyUI loads it through its dedicated Krea-2 remap with zero unmatched keys.

## `lora fuse` examples

Bake a LoRA into the full-precision base so you can quantize the result and run
with zero LoRA runtime cost (no per-inference patch). Requires the `[ml]` extra
(torch). Fuse into the **bf16** base, then quantize the output.

```
# Bake at full strength
hfutils lora fuse <lora.safetensors> <base_bf16.safetensors> -o <fused.safetensors>

# Partial strength (LoRAs trained hot often look best fused at 0.6-0.8)
hfutils lora fuse <lora> <base_bf16> -o <fused> --strength 0.7
```

Accepts the LoRA in either source or converted naming. Streams the base
tensor-by-tensor, so a 26 GB checkpoint fuses in seconds with a few MB of RAM.

## `convert` examples

```
# Full diffusers pipeline -> ComfyUI folders (transformer + vae + text_encoder)
hfutils convert <pipeline_dir> --to comfyui --root <comfyui_models>

# Pick specific components (repeatable --only / --skip)
hfutils convert <pipeline_dir> --to comfyui --root <root> --only transformer --only vae

# Single fused file -> diffusion_models/
hfutils convert <model.safetensors> --to comfyui --root <root> \
    --as diffusion_model --name <base_name>

# Preview plan + per-component metadata, write nothing
hfutils convert <pipeline_dir> --to comfyui --root <root> --dry-run

# Re-read each output after merging and confirm tensors match the plan
hfutils convert <pipeline_dir> --to comfyui --root <root> --verify

# Merge a sharded transformer into one file
hfutils convert <sharded_dir> --to single --out <output.safetensors>

# Merge one component of a pipeline without drilling into the subdir
hfutils convert <pipeline_dir> --to single --out <output.safetensors> --component transformer
```

## Source shapes

`hfutils` auto-detects the source layout:

- **Diffusers pipeline** (`PipelineSource`) — directory with `model_index.json`. Components (`transformer`, `vae`, `text_encoder[_N]`) are discovered from the subfolders.
- **Component directory** (`ComponentSource`) — sharded (`*.safetensors.index.json`) or single-file. Requires `--as <target>` with `--to comfyui`.
- **Single `.safetensors` file** (`SafetensorsFileSource`) — e.g. fused quantized checkpoints.
- **GGUF file** (`GgufFileSource`) — recognized for inspection; shows rope parameters, token ids, and chat template.
- **Pytorch directory** (`PytorchDirSource`) — legacy `.bin` / `.pt` / `.pth`. Recognized and reported; not convertible today.

## Memory

`convert --to single` (and merging inside `convert --to comfyui`) stream tensor bytes straight from each shard into the output. Peak Python allocation is under 10 MiB regardless of model size; a 23 GB sharded transformer merges with ~40 MB RSS at ~4 GB/s.

## Library usage

```python
from pathlib import Path

from hfutils import (
    ComponentSource, PipelineSource,
    DetectLevel, detect_source,
    NullObserver, PlanRunner, RichObserver,
    plan_comfyui, plan_single,
    stream_merge, verify_output,
)

# Classify + dispatch
src = detect_source(Path("/path/to/pipeline"))
match src:
    case PipelineSource(components=c):
        ...
    case ComponentSource(shards=s):
        ...

# Plan and run with the CLI's progress rendering
plan = plan_comfyui(src, Path("/comfy/models"), name="MyModel")
PlanRunner(RichObserver()).run(plan)

# Or run silently with a custom Observer
class MyObserver:
    def on_plan_start(self, plan): ...
    def on_op_start(self, op, total): ...
    # ...etc, see hfutils.Observer

PlanRunner(MyObserver()).run(plan)
```

All error types inherit from `hfutils.HfutilsError`, so library consumers can `except HfutilsError` broadly or pick a specific subtype.
