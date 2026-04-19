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
  __init__.py             -- public API: detect_source, stream_merge, plan_pack, Source, PackOp, ConvertTarget, __version__
  cli.py                  -- typer root + --version callback
  commands/
    inspect.py            -- thin wrapper: argument parsing + dispatch to views/walker
    convert.py            -- convert sub-app: single + comfyui sub-sub-commands
    civitai.py            -- civitai sub-app (search, info, dl)
  formats/
    safetensors.py        -- stream_merge + raw header helpers (pure Python, no torch)
  sources/
    detect.py             -- Source (with .enrich()) + SourceKind + detect_source()
  layouts/
    comfyui.py            -- DIFFUSERS_COMPONENTS + TARGET_FOLDERS + ConvertTarget + plan_pack()
  inspect/
    common.py             -- TensorInfo, SafetensorsHeader, DTYPE_SIZES, QUANT_DTYPE_LABELS, format_size/params, read_json_if_exists
    safetensors.py        -- read_header + read_raw_header (per-tensor data_offsets)
    gguf.py               -- GGUF header reader; extended with rope params, token ids, chat template
    architecture.py       -- _FAMILY_RULES + architecture_name_from_config
    summary.py            -- ComponentSummary for pre-conversion display
    views.py              -- display_* helpers (take explicit Console param)
    walker.py             -- walk_for_models (ThreadPoolExecutor, 8 workers)
  io/
    progress.py           -- make_progress + copy_with_progress + COPY_CHUNK (shared rich helpers)
    fs.py                 -- check_free_space preflight
  providers/
    civitai.py            -- CivitaiClient API client
    download.py           -- resumable download + rich progress (uses make_progress)
tests/                    -- 141 tests, pytest
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
- **Source abstraction is the spine.** Every command calls `detect_source(path)` first. Never re-invent format sniffing. Call `.enrich()` when display needs headers/config/sizes; leave it lazy in batch walks.
- **Plan/Execute split.** Layout planners (`layouts/comfyui.plan_pack`) return a list of ops with zero filesystem writes. CLI execution is a thin runner. Ops whose `shards` is empty are filtered at plan time.
- **Streaming I/O, not materialization.** `formats/safetensors.stream_merge` is the only code that touches tensor bytes during merge. Do not bring torch or `safetensors.torch.load_file` back into the hot path. `os.copy_file_range` was benchmarked and found slower on local SSDs (write-bound); don't assume it's a win without measuring.
- **Progress helpers live in `io/progress.py`.** All commands (`convert`, `inspect`, `civitai dl`) use `make_progress(console)`. Do not fork twin copies.
- **One Progress context across multi-op runs.** `convert comfyui` uses `_run_ops` to render a single overall bar + ephemeral per-op bars. Do not open a new Progress per op.
- **Public API** lives in `hfutils/__init__.py`. When adding a new primitive used externally (e.g. a new format helper, a new Source method), re-export it there.
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
