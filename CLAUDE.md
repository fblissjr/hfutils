Last updated: 2026-04-19

# hfutils

Local model file toolkit: safetensors/GGUF inspection, layout conversion (ComfyUI + single-file), CivitAI downloads. Pure-Python streaming merge; no torch at runtime by default.

## Commands

```
hfutils inspect <path>                       # file, component dir, pipeline
hfutils inspect <path> --recursive           # walk a tree (replaces the old `scan`)
hfutils inspect <path> --detail              # full tensor list

hfutils convert single  <src> <out>          # sharded dir or single-file -> one safetensors
hfutils convert comfyui <src> <comfyui_root> # -> diffusion_models/, vae/, text_encoders/, ...

hfutils civitai search|info|dl
```

`convert comfyui` flags: `--name`, `--only`, `--skip`, `--as {diffusion_model|checkpoint|vae|text_encoder|clip|lora}`, `--dry-run`.

## Layout

```
src/hfutils/
  cli.py                  -- typer root only
  commands/
    inspect.py            -- inspect (file/dir/pipeline/tree); dispatch via Source
    convert.py            -- convert sub-app: single + comfyui sub-sub-commands
    civitai.py            -- civitai sub-app (search, info, dl)
  formats/
    safetensors.py        -- stream_merge + total_data_bytes + raw header helpers (pure Python, no torch)
  sources/
    detect.py             -- Source + SourceKind + detect_source()
  layouts/
    comfyui.py            -- DIFFUSERS_COMPONENTS + TARGET_FOLDERS + plan_pack()
  inspect/
    common.py             -- TensorInfo, SafetensorsHeader, DTYPE_SIZES, QUANT_DTYPE_LABELS, format_size/params
    safetensors.py        -- read_header + read_raw_header (exposes per-tensor data_offsets)
    gguf.py               -- GGUF header reader
    architecture.py       -- _FAMILY_RULES + architecture_name_from_config
    directory.py          -- inspect_directory() (config.json + safetensors headers)
    summary.py            -- ComponentSummary for pre-conversion display
  io/
    progress.py           -- make_progress + copy_with_progress (shared rich helpers)
  providers/
    civitai.py            -- CivitaiClient API client
    download.py           -- resumable download + rich progress
tests/
  test_formats_safetensors.py   -- stream_merge correctness + memory bound
  test_sources_detect.py        -- source classification
  test_layouts_comfyui.py       -- plan_pack
  test_convert_comfyui.py       -- CLI smoke for `convert comfyui`
  test_convert_single.py        -- CLI smoke for `convert single`
  test_inspect_recursive.py     -- CLI smoke for `inspect --recursive`
  test_architecture.py, test_civitai.py, test_download.py,
  test_inspect_directory.py, test_inspect_gguf.py, test_inspect_safetensors.py,
  test_summary.py               -- 95 tests total, pytest
```

## Dev

- `uv sync && uv run pytest` -- install + test
- `uv run hfutils --help` -- verify CLI
- TDD: failing test first, then implement
- CLI tests use `typer.testing.CliRunner`: `runner.invoke(app, ["convert", "single", str(src), str(out)])`. See `tests/test_convert_*.py`.

## Dependencies

- Runtime: `typer`, `orjson`, `rich`, `safetensors` (no torch).
- Extras `[ml]`: `torch`, `safetensors[torch]` -- for users who want ad-hoc Python imports.
- Dev: `pytest`, `gguf`, `torch`, `safetensors[torch]` (fixtures need torch).

## Conventions

- All JSON via orjson, never stdlib json
- Rich console for all user-facing output
- Memory-bound tests use `tracemalloc` (Python-heap scoped), never `ru_maxrss` (process-wide, flaky).
- **Source abstraction is the spine.** Every command calls `detect_source(path)` first. Never re-invent format sniffing.
- **Plan/Execute split.** Layout planners (`layouts/comfyui.plan_pack`) return a list of ops with zero filesystem writes. CLI execution is a thin runner.
- **Streaming I/O, not materialization.** `formats/safetensors.stream_merge` is the only code that touches tensor bytes during merge. Do not bring torch or `safetensors.torch.load_file` back into the hot path.
- **Progress helpers live in `io/progress.py`.** Do not fork twin copies.
- **Sub-app pattern**: providers/<name>.py (client) + commands/<name>.py (typer sub-app) + register in `cli.py` via `app.add_typer()`.
- **Architecture detection** is table-driven (`_FAMILY_RULES`); add new architectures by adding table entries.
- **Shared formatters** in `inspect/common.py` (`format_size`, `format_params`, `QUANT_DTYPE_LABELS`, `DTYPE_SIZES`); don't duplicate.

## Gotchas

- Pyright "Import X could not be resolved" warnings in tool output are noise -- the language server isn't wired to the uv venv. Ignore unless the import is actually wrong.
- Do NOT add HF download/search/info commands -- `hf` CLI handles those.
- Use `typer` (full package), NOT `typer-slim` -- typer-slim doesn't provide the `typer` import.
- GGUF parser is hand-rolled (struct.unpack) -- the gguf library memmaps entire files including tensor data.
- Safetensors header: first 8 bytes = LE uint64 header length, then JSON, then tensor data. `data_offsets` in the header are relative to the start of the tensor data region (not the file).
- `_FAMILY_RULES` order matters: more specific patterns first (Hunyuan Video before Flux).
- CivitAI needs a realistic User-Agent (DEFAULT_HEADERS in providers/download.py).
- CivitAI API key via `CIVITAI_API_KEY` env var only (no config file).
- Test fixtures use `safetensors.torch.save_file` (dev dep only); runtime merges never touch torch.
- Diffusers uses `diffusion_pytorch_model.safetensors.index.json`; transformers uses `model.safetensors.index.json`. `*.safetensors.index.json` globbing accepts both.
- `stream_merge` rejects duplicate tensor names across shards with a clear error.
