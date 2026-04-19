Last updated: 2026-04-19

# hfutils

Local model file toolkit for safetensors and GGUF files, with CivitAI downloads and ComfyUI layout conversion.

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

hfutils civitai search|info|dl           # CivitAI sub-app (unchanged)
```

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
    detect_source, DetectLevel,
    plan_comfyui, plan_single,
    PlanRunner, RichObserver, NullObserver,
    stream_merge, verify_output,
)

# Classify
src = detect_source(Path("/path/to/pipeline"))
match src:
    case hfutils.PipelineSource(components=c):
        ...
    case hfutils.ComponentSource(shards=s):
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
