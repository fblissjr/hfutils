# Changelog

## 0.4.0

- `hfutils comfyui pack` -- pack any local model layout into ComfyUI folders
  - Diffusers pipelines (auto-detect components from `model_index.json`)
  - Component directories (sharded or single-file) with `--as`
  - Single `.safetensors` files (LTX-style fused) with `--as`
  - `--only` / `--skip` filters, `--name`, `--dry-run`
  - Per-component metadata display: format, architecture, tensor count, params, size, dtype, quantization
  - Progress bars for shard load + single-file copy
- `hfutils merge` now auto-discovers `*.safetensors.index.json` (diffusers or transformers style)
- `hfutils merge` gains progress bar for shard loading
- Internal: `consolidate_component()` helper that handles both sharded and single-file component dirs

## 0.3.0

- CivitAI integration: `hfutils civitai search`, `civitai info`, `civitai dl`
- Shared download module with rich progress bars and resume support
- safetensors[torch] now a default dependency (no longer optional)
- providers/ directory for external service integrations

## 0.2.0

- Rebuilt as local model file toolkit (dropped HF download/collection features -- use `hf` CLI for those)
- `hfutils inspect` -- header-only safetensors + GGUF inspection with architecture detection
- `hfutils inspect <directory>` -- combined config.json + model file inspection
- `hfutils merge` -- merge sharded safetensors into single file
- `hfutils scan` -- audit local model directories
- Switched CLI framework from argparse to typer
- Architecture detection (table-driven): Flux, Hunyuan Video, Mochi, Wan, LTX Video, SDXL, SD3, Stable Diffusion, Llama-style LLMs, LoRA/DoRA adapters
- Purpose-built binary header readers for safetensors and GGUF (no ML library dependencies at runtime)

## 0.1.0

- Initial release: HF collection downloader with parallel downloads
