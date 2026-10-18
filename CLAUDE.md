Last updated: 2026-10-18

# hfutils

Local model file toolkit: safetensors/GGUF inspection, layout conversion (ComfyUI + single-file), LoRA key-scheme conversion, CivitAI downloads. Pure-Python streaming merge; no torch at runtime by default.

## Commands

```
hfutils inspect <path>                       # file, component dir, pipeline
hfutils inspect <path> --recursive           # walk a tree (replaces the old `scan`)
hfutils inspect <path> --detail              # full tensor list

hfutils convert <src> --to <layout> [opts]   # unified convert; --to comfyui | single
hfutils lora convert <lora> [opts]           # rename LoRA keys for a target loader
hfutils lora fuse <lora> <base> -o <out>     # bake a LoRA into a base checkpoint ([ml] extra)
hfutils lora schemes                         # list LoRA conversion schemes
hfutils civitai search|info|dl
```

`convert --to comfyui` flags: `--root` (required), `--name`, `--only <c>` (repeat), `--skip <c>` (repeat), `--as {diffusion_model|checkpoint|vae|text_encoder|clip|lora}`, `--dry-run`, `--verify`, `--force`.
`convert --to single` flags: `--out` (required; file or directory), `--component <c>` (repeat, or `all`, for pipeline sources), `--all` (convert all pipeline components into `--out <dir>`), `--name`, `--only <c>` (repeat), `--skip <c>` (repeat), `--dry-run`, `--verify`, `--force`.
`convert`'s `--force` writes an incomplete/corrupt source anyway (still warns; see "Incomplete/corrupt-source hard-stop" below) -- distinct from `lora convert --force`, which bypasses a scheme-mismatch check instead.

## Layout

```
src/hfutils/
  __init__.py             -- public API (all stable symbols re-exported here)
  cli.py                  -- typer root + --version; registers inspect, convert, civitai
  cli_support.py          -- fail() / cli_guard() / verify_and_report(): shared CLI error handling, used by every commands/*.py
  errors.py               -- HfutilsError + subclasses (PlanError, StreamMergeError, LoraConvertError, InsufficientSpaceError)
  events.py               -- Observer + MergeObserver protocols + NullObserver / CollectingObserver / RichObserver (plan-scope) / RichMergeObserver (merge-scope)
  runner.py               -- PlanRunner (executes a PackPlan; hard-stops on PackPlan.broken_ops()/.missing unless force=True; dispatches events to an Observer)
  commands/
    inspect.py            -- thin CLI wrapper; dispatches via DetectLevel.FULL
    convert.py            -- unified `convert <src> --to <layout>` command
    lora.py               -- `lora` sub-app: convert (key-scheme rename) + schemes
    civitai.py            -- civitai sub-app (search, info, dl)
  formats/
    safetensors.py        -- stream_merge + detect_key_prefix_map + verify_output + raw header helpers (pure Python, no torch)
    lora.py               -- LoraScheme + LORA_SCHEMES registry + convert_lora_keys (streaming key rename) + fuse_lora (torch, bake into base)
  sources/
    detect.py             -- detect_source(path, level: DetectLevel) + enrich(source) -> EnrichedView + check_shard_integrity(shards) (declared-vs-actual size; also called directly at plan time for single-shard ops, bypassing DetectLevel -- see Conventions)
    types.py              -- Source union + variants (PipelineSource | ComponentSource | SafetensorsFileSource | GgufFileSource | PytorchDirSource | UnknownSource) + IntegrityError + display_kind()
  layouts/
    comfyui.py            -- DIFFUSERS_COMPONENTS + TARGET_FOLDERS (keyed on ConvertTarget) + plan_comfyui + plan_single (supports PipelineSource, ComponentSource, SafetensorsFileSource); PackOp has a cached total_bytes and an optional problem (incomplete/corrupt source)
    plan.py               -- PackPlan (total_bytes is sum of op.total_bytes, lazily; validate() is CLI-only advisory, broken_ops()/missing are hard-stopped by PlanRunner)
  inspect/
    common.py             -- TensorInfo, SafetensorsHeader (with .combine()), DTYPE_SIZES, QUANT_DTYPE_LABELS, format_size/params, read_json_if_exists
    safetensors.py        -- read_header + read_raw_header (per-tensor data_offsets) + read_raw_header_or_raise(path, error_cls=...) (shared by formats/{safetensors,lora}.py)
    gguf.py               -- GGUF header reader; rope params, token ids, chat template
    architecture.py       -- _FAMILY_RULES + architecture_name_from_config
    summary.py            -- ComponentSummary for pre-conversion display
    views.py              -- display_* helpers (pattern-match on Source variants; explicit Console param)
    walker.py             -- walk_for_models (ThreadPoolExecutor, 8 workers)
  io/
    progress.py           -- make_progress + copy_chunks (primitive; on_chunk + on_open callbacks) + copy_with_progress (rich wrapper) + stream_region (bounded seek-copy, error_cls-parametrized) + unlink_on_error (mark_started()-gated cleanup) + COPY_CHUNK
    fs.py                 -- check_free_space preflight
  providers/
    civitai.py            -- CivitaiClient API client
    download.py           -- resumable download + rich progress (uses make_progress)
tests/                    -- 349 tests, pytest (includes Hypothesis property tests for stream_merge)
```

## Dev

- `uv sync && uv run pytest` -- install + test
- `uv run hfutils --help` -- verify CLI
- `uv add --dev <pkg>` for test/dev deps (hypothesis, pytest plugins, etc.). Bare `uv add` goes to runtime.
- TDD: failing test first, then implement
- CLI tests use `typer.testing.CliRunner`: `runner.invoke(app, ["convert", str(src), "--to", "single", "--out", str(out)])`. See `tests/test_convert.py`.
- Shared test fixtures: `make_sharded_component()` and `make_diffusers_pipeline()` live in `tests/conftest.py`. Don't re-roll local copies.
- Property-based tests use Hypothesis: `@given(...) + @settings(max_examples=N, deadline=None)`, `@st.composite` for nested draws. See `tests/test_stream_merge_properties.py`.
- Observer tests: use `CollectingObserver` / `CollectingMergeObserver` from `hfutils.events` to capture lifecycle events for assertions.

## Dependencies

- Runtime: `typer`, `orjson`, `rich`, `safetensors` (no torch).
- Extras `[ml]`: `torch`, `safetensors[torch]` -- for users who want ad-hoc Python imports.
- Dev: `pytest`, `gguf`, `torch`, `safetensors[torch]` (fixtures need torch).

## Conventions

- All JSON via orjson, never stdlib json
- Rich console for all user-facing output
- Memory-bound tests use `tracemalloc` (Python-heap scoped), never `ru_maxrss` (process-wide, flaky).
- **Source is a discriminated union.** `Source = PipelineSource | ComponentSource | SafetensorsFileSource | GgufFileSource | PytorchDirSource | UnknownSource`. Dispatch via `match src:` or `isinstance`. Do not invent a kind enum. `detect_source(path, level: DetectLevel)` is the only classification point; `enrich(source)` returns a fresh `EnrichedView` (immutable pattern, safe to call from threads).
- **DetectLevel.BASIC (default) skips integrity.** Only single-target inspect passes `DetectLevel.FULL`. Walker / convert / plan_* all use default. Adding always-on work per-dir to the walker is a regression. Exception: single-shard `SafetensorsFileSource` planning calls `check_shard_integrity` directly (one header read scoped to the file being planned, not a directory walk), so `plan_comfyui`/`plan_single` catch a truncated single-file source without waiting on `stream_merge`'s EOF check. This is a scoped call at plan time, not a change to what `DetectLevel.BASIC` itself computes.
- **Plan/Execute/Observe split.**
  - `plan_comfyui` / `plan_single` return `PackPlan`. No filesystem writes.
  - `PlanRunner(observer).run(plan)` executes. Observer receives lifecycle events.
  - CLI wraps with `RichObserver`. Library consumers attach `NullObserver`, `CollectingObserver`, or custom.
- **Streaming I/O, not materialization.** `formats/safetensors.stream_merge` is the only code that touches tensor bytes during merge. Do not bring torch or `safetensors.torch.load_file` back into the hot path. `os.copy_file_range` was benchmarked and found slower on local SSDs (write-bound); don't assume it's a win without measuring. `detect_key_prefix_map` automatically rewrites bare `language_model.*`/`visual.*` keys (from `AutoModel`-saved VLM text encoders such as `Qwen3_5Model` in MOSS-Image-Base) to `model.language_model.*`/`model.visual.*` in the safetensors header so ComfyUI's `detect_te_model` and `load_text_encoder_state_dicts` load them without error; `PlanRunner` also routes single-shard (`copy`) ops with a non-empty prefix map through `stream_merge` so the header rewrite happens even for unsharded files.

- **Error hierarchy.** Every `raise` in core modules uses a `HfutilsError` subclass: `PlanError`, `StreamMergeError`, `LoraConvertError`, `InsufficientSpaceError`. File-format parsers (inspect/safetensors, inspect/gguf) keep `ValueError` for malformed-input cases (not hfutils semantics) -- but any call site that reads a header as a *prerequisite* for an hfutils operation (not the operation itself) should go through `inspect/safetensors.read_raw_header_or_raise(path, error_cls=...)` instead of a bare `read_raw_header`, so a truncated/malformed file surfaces as the caller's own `HfutilsError` subclass rather than an uncaught `ValueError`. `commands/*.py` catch the `HfutilsError` hierarchy via `cli_support.cli_guard` (see below), not per-command try/except.
- **CLI error handling is centralized in `cli_support.py`.** `fail(msg)` prints `[red]Error:[/red] ...` and raises `typer.Exit(1)`; `cli_guard` is a decorator applied to every `@app.command()` that catches `(HfutilsError, OSError)` escaping the command and routes it through `fail()` -- deliberately not a blanket `except Exception`, so a bare `ValueError`/`RuntimeError` (a real bug) still surfaces a full traceback instead of being swallowed as if it were user-facing. `verify_and_report(dest, manifest)` is the shared `--verify` helper (print + `fail()` on mismatch) used by every command that offers `--verify`. Don't hand-roll a local `_error()` or inline try/except in a new command -- use these three.
- **Progress helpers live in `io/progress.py`.** `RichObserver` uses `make_progress(console)`. The chunked copy loop is in `copy_chunks(src, dst, on_chunk=cb, on_open=cb)`; both `copy_with_progress` and `PlanRunner`'s copy branch use it. `stream_region(fin, fout, size, *, src_desc, on_chunk=None, error_cls=OSError)` is the shared bounded seek-copy primitive (`formats/safetensors.py` and `formats/lora.py` both import it from here -- don't reintroduce a per-module copy). Do not fork twin copies.
- **`unlink_on_error(path)` deletes a partially-written destination on failure -- but only once writing has actually started.** It yields a `mark_started()` callback; call it *after* the destination file is successfully opened (inside the `with open(dst, "wb") as ...:` block or immediately after the last context manager in the `with` statement succeeds), never before. Calling it too early deletes files that failed for a reason unrelated to this write (e.g. a prerequisite file that doesn't exist) instead of only cleaning up genuine partial writes.
- **Single-file streaming ops** (`convert_lora_keys`, `fuse_lora`) take a `MergeObserver`; for CLI progress build `RichMergeObserver(progress, label)` (merge-scope, in `events.py`) -- don't hand-roll one. `io/progress.copy_chunks` is EOF-only (own handles, copies from offset 0); bounded copies from a seek offset use `io/progress.stream_region`.
- **Incomplete/corrupt-source hard-stop.** `PackOp.problem: str | None` (set by `_source_problem`/`_copy_integrity_problem` in `layouts/comfyui.py`) flags a single op as incomplete or corrupt; `PackPlan.missing: list[str]` flags a pipeline component with zero weight files on disk at all (no `PackOp` exists to attach a problem to). `PlanRunner.run()` checks both before any write and raises `PlanError` unless `force=True` is passed -- this is enforced in the library layer, not the CLI, so any consumer of `PackPlan`/`PlanRunner` gets the protection automatically. `PackPlan.validate()` (duplicate-destination warnings) stays CLI-only advisory, not hard-stopped by `PlanRunner` -- deliberate asymmetry: a duplicate destination is a footgun but still usable, while an incomplete/missing source guarantees a broken output with no later exception path to catch it. Unifying the two is deferred, not an oversight -- a duplicate-destination override may warrant different (or stricter) semantics than `--force`'s "write it anyway."
- **`PackOp.total_bytes` is a `cached_property`.** Single source of truth for preflight + progress sizing. Don't stat shards elsewhere; read `op.total_bytes`.
- **Public API lives in `hfutils/__init__.py`.** When adding a new primitive used externally, re-export it there.
- **Sub-app pattern**: providers/<name>.py (client) + commands/<name>.py (typer sub-app) + register in `cli.py` via `app.add_typer()`.
- **Architecture detection** is table-driven (`_FAMILY_RULES`); add new architectures by adding table entries.
- **LoRA conversion is scheme-driven** (`formats/lora.LORA_SCHEMES`). Each `LoraScheme` carries `rename(key)->key`, a `matches(header)->bool` detector, and an optional `to_base_key(module)->base weight | None` (required for `fuse`). Add a new trainer->loader conversion by adding a registry entry. `convert_lora_keys` streams the data region verbatim (only the header's tensor names change) and returns a `Manifest` for `verify_output`, mirroring `stream_merge`. The `krea2` rename map inverts ComfyUI's `comfy.utils.krea2_to_diffusers` -- keep them in sync if Krea naming changes. `matches` detects a Krea-2 LoRA in either naming (txtfusion/text_fusion + LoRA tensors), so the base checkpoint (text-fusion stack but no LoRA tensors) doesn't match.
- **`fuse_lora` is the one torch-gated path** (`[ml]` extra): it does real tensor math (`W + (alpha/rank)*strength*B@A`), unlike the streaming key-only ops. torch is imported lazily inside the function; a clear `LoraConvertError` points at `hfutils[ml]` if it's missing. It streams the base data region tensor-by-tensor (low memory on a 26 GB base), reuses the base header verbatim (output names/shapes/dtypes == base), and refuses fp8 bases (the stored weight carries a separate scale -- fuse into bf16, then quantize).
- **Shared formatters** in `inspect/common.py` (`format_size`, `format_params`, `QUANT_DTYPE_LABELS`, `DTYPE_SIZES`); don't duplicate.

## Gotchas

- Pyright "Import X could not be resolved" warnings in tool output are noise -- the language server isn't wired to the uv venv. Ignore unless the import is actually wrong.
- `git commit -m "..."` with raw backticks inside the message: bash runs them as command substitution and silently drops the word. Use HEREDOC (`-m "$(cat <<'EOF' ... EOF)"`) for any message containing backticks.
- SSH signing occasionally flakes with "communication with agent failed" even when `ssh-add -l` shows the key. Retry the commit; do not bypass with `--no-gpg-sign`.
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
