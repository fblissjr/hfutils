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
hfutils inspect <path>                   # file, component dir, or diffusers pipeline
hfutils inspect <path> --recursive       # walk a directory tree for model dirs
hfutils inspect <path> --detail          # include full tensor list

hfutils convert single  <src> <out>      # merge shards (or copy a single file) into one .safetensors
hfutils convert comfyui <src> <root>     # pack into ComfyUI folder layout (diffusion_models, vae, ...)

hfutils civitai search <query>           # search CivitAI
hfutils civitai info   <id|url>          # model details
hfutils civitai dl     <id|url>          # resumable download
```

## `convert` examples

```
# Full diffusers pipeline -> ComfyUI folders (transformer + vae + text_encoder)
hfutils convert comfyui <pipeline_dir> <comfyui_models>

# Just one component
hfutils convert comfyui <pipeline_dir> <comfyui_models> --only transformer

# Single fused file -> diffusion_models/
hfutils convert comfyui <model.safetensors> <comfyui_models> \
    --as diffusion_model --name <base_name>

# Preview plan + per-component metadata, write nothing
hfutils convert comfyui <pipeline_dir> <comfyui_models> --dry-run

# Merge a sharded transformer into one file
hfutils convert single <sharded_dir> <output.safetensors>
```

## Source shapes

`hfutils` auto-detects the source layout:

- **Diffusers pipeline** — directory with `model_index.json`. Components (`transformer`, `vae`, `text_encoder[_N]`) are discovered from the subfolders and the relevant ones flow into their ComfyUI destinations.
- **Component directory** — sharded (`*.safetensors.index.json` + multiple shards) or single-file (`diffusion_pytorch_model.safetensors` / `model.safetensors`). Requires `--as <target>` when used with `convert comfyui`.
- **Single `.safetensors` file** — e.g. fused quantized checkpoints. Requires `--as <target>` with `convert comfyui`.
- **GGUF file** — recognized for inspection.

## Memory

`convert single` (and merging inside `convert comfyui`) stream tensor bytes straight from each shard into the output. Peak memory is a few MiB regardless of model size — a 20+ GB sharded model merges with <500 MB RSS.
