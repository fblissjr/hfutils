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

hfutils convert single  <src> <out>      # merge shards (or copy a single file) into one .safetensors
hfutils convert comfyui <src> <root>     # pack into ComfyUI folder layout

hfutils civitai search <query>           # search CivitAI
hfutils civitai info   <id|url>          # model details
hfutils civitai dl     <id|url>          # resumable download
```

## `convert` examples

```
# Full diffusers pipeline -> ComfyUI folders (transformer + vae + text_encoder)
hfutils convert comfyui <pipeline_dir> <comfyui_models>

# Pick specific components (repeatable --only / --skip)
hfutils convert comfyui <pipeline_dir> <comfyui_models> --only transformer --only vae

# Single fused file -> diffusion_models/
hfutils convert comfyui <model.safetensors> <comfyui_models> \
    --as diffusion_model --name <base_name>

# Preview plan + per-component metadata, write nothing
hfutils convert comfyui <pipeline_dir> <comfyui_models> --dry-run

# Re-read each output after merging and confirm tensors match the plan
hfutils convert comfyui <pipeline_dir> <comfyui_models> --verify

# Merge a sharded transformer into one file
hfutils convert single <sharded_dir> <output.safetensors>

# Merge one component of a pipeline without drilling into the subdir
hfutils convert single <pipeline_dir> <output.safetensors> --component transformer
```

## Source shapes

`hfutils` auto-detects the source layout:

- **Diffusers pipeline** — directory with `model_index.json`. Components (`transformer`, `vae`, `text_encoder[_N]`) are discovered from the subfolders; `convert comfyui` routes the weight-bearing ones into their ComfyUI destinations.
- **Component directory** — sharded (`*.safetensors.index.json` + multiple shards) or single-file. Requires `--as <target>` with `convert comfyui`.
- **Single `.safetensors` file** — e.g. fused quantized checkpoints. Requires `--as <target>` with `convert comfyui`.
- **GGUF file** — recognized for inspection; shows rope parameters, token ids, and chat template when present.
- **Pytorch directory** — legacy `.bin` / `.pt` / `.pth` weights. Recognized and reported; not convertible today.

## Memory

`convert single` (and merging inside `convert comfyui`) stream tensor bytes straight from each shard into the output. Peak Python allocation is under 10 MiB regardless of model size; a 23 GB sharded transformer merges with ~40 MB RSS at ~4 GB/s.

## Library usage

```python
from hfutils import detect_source, stream_merge

src = detect_source(Path("/path/to/sharded-dir"))
if src.kind.value == "component_dir":
    stream_merge(src.shards, Path("/tmp/merged.safetensors"))
```

Exported surface: `detect_source`, `Source`, `SourceKind`, `plan_pack`, `PackOp`, `ConvertTarget`, `stream_merge`, `read_raw_header`, `__version__`.
