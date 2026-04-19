Last updated: 2026-04-19

# hfutils

Local model file toolkit for safetensors and GGUF files, with CivitAI downloads and ComfyUI layout conversion.

## Install

```
uv add hfutils
```

## Commands

```
hfutils inspect model.safetensors            # header summary: tensors, params, VRAM, architecture
hfutils inspect --detail model.safetensors   # full tensor list
hfutils inspect model.gguf                   # GGUF metadata: arch, quant, context length
hfutils inspect ./model-dir/                 # config.json + model file headers combined
hfutils merge ./sharded-model/ out.safetensors  # merge sharded safetensors into one file
hfutils scan /path/to/models/                # audit local model dirs: format, size, completeness

hfutils civitai search <query>               # search CivitAI models
hfutils civitai info <id|url>                # model details and versions
hfutils civitai dl <id|url>                  # download with resume

hfutils comfyui pack <src> <comfyui_models>  # convert a local model into ComfyUI folders
```

## `comfyui pack` examples

```
# Full diffusers pipeline -> ComfyUI layout (transformer, vae, text_encoder)
hfutils comfyui pack <pipeline_dir> <comfyui_models>

# Only a single component
hfutils comfyui pack <pipeline_dir> <comfyui_models> --only transformer

# Single fused file -> diffusion_models/
hfutils comfyui pack <model.safetensors> <comfyui_models> \
    --as diffusion_model --name <base_name>

# Preview the plan + per-component metadata (format, architecture, params, dtype, quant)
hfutils comfyui pack <pipeline_dir> <comfyui_models> --dry-run
```

Source shapes handled:
- **Diffusers pipeline** (has `model_index.json`): auto-discovers `transformer`, `vae`, `text_encoder[_N]` subfolders (the DiT lands in ComfyUI's `diffusion_models/`). Pick/skip with `--only`, `--skip`.
- **Component directory** (sharded or single-file, no `model_index.json`): specify destination with `--as {diffusion_model|checkpoint|vae|text_encoder|clip|lora}`.
- **Single `.safetensors` file**: specify destination with `--as`.

`--dry-run` prints the full plan **and** the per-component metadata (architecture, tensor count, params, dominant dtype, quantization, size) so you can preview exactly what would happen without moving any bytes.
