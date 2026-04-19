Last updated: 2026-04-19

# hfutils

Local model file toolkit: safetensors/GGUF inspection, layout conversion (ComfyUI + single-file), CivitAI downloads. Pure-Python streaming merge; no torch at runtime by default.

## Commands

```
hfutils inspect <path>                       # file, component dir, pipeline
hfutils inspect <path> --recursive           # walk a tree (replaces the old `scan`)
hfutils inspect <path> --detail              # full tensor list

hfutils convert <src> --to <layout> [opts]   # unified convert; --to comfyui | single
hfutils civitai search|info|dl
```

`convert --to comfyui` flags: `--root` (required), `--name`, `--only <c>` (repeat), `--skip <c>` (repeat), `--as {diffusion_model|checkpoint|vae|text_encoder|clip|lora}`, `--dry-run`, `--verify`.
`convert --to single` flags: `--out` (required), `--component <c>` (for pipeline sources), `--dry-run`, `--verify`.

## Layout

```
src/hfutils/
  __init__.py             -- public API (all stable symbols re-exported here)
  cli.py                  -- typer root + --version; registers inspect, convert, civitai
  errors.py               -- HfutilsError + subclasses (SourceError, PlanError, StreamMergeError, VerificationError, InsufficientSpaceError)
  events.py               -- Observer + MergeObserver protocols + NullObserver / CollectingObserver / RichObserver
  runner.py               -- PlanRunner (executes a PackPlan, dispatches events to an Observer)
  commands/
    inspect.py            -- thin CLI wrapper; dispatches via DetectLevel.FULL
    convert.py            -- unified `convert <src> --to <layout>` command
    civitai.py            -- civitai sub-app (search, info, dl)
  formats/
    safetensors.py        -- stream_merge + verify_output + raw header helpers (pure Python, no torch)
  sources/
    detect.py             -- detect_source(path, level: DetectLevel) + enrich(source) -> EnrichedView
    types.py              -- Source union + variants (PipelineSource | ComponentSource | SafetensorsFileSource | GgufFileSource | PytorchDirSource | UnknownSource) + IntegrityError + display_kind()
  layouts/
    comfyui.py            -- DIFFUSERS_COMPONENTS + TARGET_FOLDERS (keyed on ConvertTarget) + plan_comfyui + plan_single; PackOp has a cached total_bytes
    plan.py               -- PackPlan (total_bytes is sum of op.total_bytes, lazily)
  inspect/
    common.py             -- TensorInfo, SafetensorsHeader (with .combine()), DTYPE_SIZES, QUANT_DTYPE_LABELS, format_size/params, read_json_if_exists
    safetensors.py        -- read_header + read_raw_header (per-tensor data_offsets)
    gguf.py               -- GGUF header reader; rope params, token ids, chat template
    architecture.py       -- _FAMILY_RULES + architecture_name_from_config
    summary.py            -- ComponentSummary for pre-conversion display
    views.py              -- display_* helpers (pattern-match on Source variants; explicit Console param)
    walker.py             -- walk_for_models (ThreadPoolExecutor, 8 workers)
  io/
    progress.py           -- make_progress + copy_chunks (primitive) + copy_with_progress (rich wrapper) + COPY_CHUNK
    fs.py                 -- check_free_space preflight
  providers/
    civitai.py            -- CivitaiClient API client
    download.py           -- resumable download + rich progress (uses make_progress)
tests/                    -- 171 tests, pytest (includes Hypothesis property tests for stream_merge)
```

## Dev

- `uv sync && uv run pytest` -- install + test
- `uv run hfutils --help` -- verify CLI
- TDD: failing test first, then implement
- CLI tests use `typer.testing.CliRunner`: `runner.invoke(app, ["convert", str(src), "--to", "single", "--out", str(out)])`. See `tests/test_convert.py`.
- Shared test fixtures: `make_sharded_component()` and `make_diffusers_pipeline()` live in `tests/conftest.py`. Don't re-roll local copies.

## Dependencies

- Runtime: `typer`, `orjson`, `rich`, `safetensors` (no torch).
- Extras `[ml]`: `torch`, `safetensors[torch]` -- for users who want ad-hoc Python imports.
- Dev: `pytest`, `gguf`, `torch`, `safetensors[torch]` (fixtures need torch).

## Conventions

- All JSON via orjson, never stdlib json
- Rich console for all user-facing output
- Memory-bound tests use `tracemalloc` (Python-heap scoped), never `ru_maxrss` (process-wide, flaky).
- **Source is a discriminated union.** `Source = PipelineSource | ComponentSource | SafetensorsFileSource | GgufFileSource | PytorchDirSource | UnknownSource`. Dispatch via `match src:` or `isinstance`. Do not invent a kind enum. `detect_source(path, level: DetectLevel)` is the only classification point; `enrich(source)` returns a fresh `EnrichedView` (immutable pattern, safe to call from threads).
- **DetectLevel.BASIC (default) skips integrity.** Only single-target inspect passes `DetectLevel.FULL`. Walker / convert / plan_* all use default. Adding always-on work per-dir to the walker is a regression.
- **Plan/Execute/Observe split.**
  - `plan_comfyui` / `plan_single` return `PackPlan`. No filesystem writes.
  - `PlanRunner(observer).run(plan)` executes. Observer receives lifecycle events.
  - CLI wraps with `RichObserver`. Library consumers attach `NullObserver`, `CollectingObserver`, or custom.
- **Streaming I/O, not materialization.** `formats/safetensors.stream_merge` is the only code that touches tensor bytes during merge. Do not bring torch or `safetensors.torch.load_file` back into the hot path. `os.copy_file_range` was benchmarked and found slower on local SSDs (write-bound); don't assume it's a win without measuring.
- **Error hierarchy.** Every `raise` in core modules uses a `HfutilsError` subclass: `SourceError`, `PlanError`, `StreamMergeError`, `VerificationError`, `InsufficientSpaceError`. File-format parsers (inspect/safetensors, inspect/gguf) keep `ValueError` for malformed-input cases (not hfutils semantics).
- **Progress helpers live in `io/progress.py`.** `RichObserver` uses `make_progress(console)`. The chunked copy loop is in `copy_chunks(src, dst, on_chunk=cb)`; both `copy_with_progress` and `PlanRunner`'s copy branch use it. Do not fork twin copies.
- **`PackOp.total_bytes` is a `cached_property`.** Single source of truth for preflight + progress sizing. Don't stat shards elsewhere; read `op.total_bytes`.
- **Public API lives in `hfutils/__init__.py`.** When adding a new primitive used externally, re-export it there.
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
