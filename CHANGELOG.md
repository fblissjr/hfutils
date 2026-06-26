# Changelog

## 0.8.0

### Breaking

- **`plan_pack` removed** -- the deprecated alias for `plan_comfyui` (kept since 0.7.0) is gone. Call `plan_comfyui`.

### Major

- **LoRA key-scheme conversion** (`hfutils lora`): rename a LoRA's tensor keys so a target loader accepts it, without touching tensor data.
  - `hfutils lora convert <lora.safetensors>` -- auto-detects the source scheme, writes `<lora>_comfy.safetensors`. Flags: `--out/-o`, `--scheme`, `--force`, `--dry-run`, `--verify`.
  - `hfutils lora schemes` -- lists registered schemes.
  - First scheme `krea2`: fal.ai's Krea-2 trainer emits PEFT-style keys (`base_model.model.blocks.0.attn.wq.lora_A.weight`) that ComfyUI's default loader rejects; this rewrites them to the diffusers-style names the official Krea-2 LoRAs use (`transformer.transformer_blocks.0.attn.to_q.lora_A.weight`). The rename map inverts ComfyUI's own `krea2_to_diffusers`; verified to reproduce the official `krea2_darkbrush` key set exactly (528/528 keys, all shapes matching), and confirmed in ComfyUI (264 LoRA modules all match, vs 0 for the raw file).
- **LoRA fuse** (`hfutils lora fuse <lora> <base> -o <out>`): bake a LoRA into the full-precision base checkpoint (`W + (alpha/rank)*strength*(B@A)` per weight), so the result can be quantized and run with zero LoRA runtime cost. Streams the base data region tensor-by-tensor (a 26 GB base fuses in ~35s at low memory); copies non-LoRA tensors verbatim; output header is identical to the base. Accepts the LoRA in either fal or converted/official diffusers naming. Refuses fp8 bases (fuse into bf16, then quantize). Requires the `[ml]` extra (torch). Verified end-to-end: the fused checkpoint renders the same image as base + runtime-LoRA.
- **Public API / library surface**: `convert_lora_keys`, `fuse_lora`, `FuseResult`, `detect_lora_scheme`, `LoraScheme` (now with `to_base_key`), `LORA_SCHEMES`, `KREA2_FAL`, and `LoraConvertError`. `convert_lora_keys` is streaming and torch-free (mirrors `stream_merge`: takes a `MergeObserver`, returns a `Manifest` for `verify_output`); `fuse_lora` is streaming and torch-gated.

### Medium

- **Error hierarchy** gains `LoraConvertError` (key collision / unexpected EOF during a LoRA convert).
- **`RichMergeObserver`** (`hfutils.events`): a merge-scope rich-progress observer (the counterpart to `RichObserver`), so single-file streaming ops get CLI progress without a hand-rolled observer.

## 0.7.0

### Breaking

- **CLI**: `hfutils convert comfyui <src> <root>` and `hfutils convert single <src> <out>` are gone. One top-level command with `--to`:
  - `hfutils convert <pipeline_dir> --to comfyui --root <models>` (was `convert comfyui ...`)
  - `hfutils convert <sharded_dir>  --to single  --out <out.safetensors>` (was `convert single ...`)
- **Source** is now a discriminated union: `PipelineSource | ComponentSource | SafetensorsFileSource | GgufFileSource | PytorchDirSource | UnknownSource`. `SourceKind` enum is gone; use `isinstance` / `match`. `Source.enrich()` method is replaced by free function `enrich(source) -> EnrichedView`.
- **`stream_merge`**: the three callbacks (`on_total`, `on_progress`, `on_warning`) are replaced by a single `observer: MergeObserver` kwarg.
- **`plan_pack` renamed `plan_comfyui`** and returns `PackPlan` (was `list[PackOp]`). Backward-compat alias `plan_pack` re-exported for one release; removed in 0.8.

### Major

- **Error hierarchy** (`hfutils.errors`): `HfutilsError` base plus `SourceError`, `PlanError`, `StreamMergeError`, `VerificationError`, `InsufficientSpaceError`. Library consumers can catch broadly or narrowly.
- **Observer protocol** (`hfutils.events`): `Observer` (plan-scope, 6 events) + `MergeObserver` (merge-scope, 3 events). Built-in: `NullObserver`, `NullMergeObserver`, `CollectingObserver`, `CollectingMergeObserver`, `RichObserver` (the CLI's default).
- **`PlanRunner`** (`hfutils.runner`): executes a `PackPlan`, dispatches events to the configured Observer. CLI wraps it with `RichObserver`; library consumers attach any Observer implementation.
- **`PackPlan` object** (`hfutils.layouts.plan`): replaces the bare `list[PackOp]`. Exposes `.total_bytes`, `.validate()`, supports `len()`, iteration, truthiness.
- **Unified `convert` CLI**: one command, `--to {comfyui,single}` dispatches. `--root` required for comfyui; `--out` required for single. `--component` on `--to single` picks a pipeline subfolder without drilling in.

### Medium

- **`DetectLevel` kwarg** on `detect_source`: `BASIC` (default, no per-shard header reads) vs `FULL` (+ integrity validation). Walker uses default; `inspect <path>` uses `FULL`.
- **Hypothesis property tests** for `stream_merge`: random tensor dicts with mixed dtypes, random 1..5-shard splits, bytewise round-trip assertion. And a metadata-warning property: fires iff a key's value actually differs across shards.
- **Public API** gains: `HfutilsError` hierarchy, `Observer` / `MergeObserver` + concrete implementations, `PlanRunner`, `PackPlan`, `DetectLevel`, `plan_comfyui`, and the Source variants.

## 0.6.0

### CLI

- `--only`/`--skip` on `convert comfyui` are now repeatable flags (`--only transformer --only vae`), not comma-split strings.
- `--as` is a typer `Enum`; valid values appear in `--help` and invalid values are rejected at the CLI layer.
- New `hfutils --version`.
- `convert single <pipeline_dir> <out.safetensors> --component <name>` lets you merge one component of a diffusers pipeline without drilling into the subdir. Errors list available components when `--component` is missing or wrong.
- `convert --verify` re-reads the output header after a merge and confirms tensor names, dtypes, and shapes match the plan. Exit code 2 on mismatch.
- Unified `DRY RUN Plan` header across `convert comfyui` and `convert single`.
- `convert comfyui` renders a single overall progress bar across all ops (no more bar tear-down between components).

### Backend

- `Source` absorbed `inspect_directory`/`DirectoryInfo`. New `Source.enrich()` lazily populates `config`, `total_file_size`, `safetensors_headers`, and `gguf_info`. `inspect/directory.py` deleted.
- `commands/inspect.py` split: display helpers live in `inspect/views.py`, walking + HF cache detection lives in `inspect/walker.py`. Command file is ~40 lines.
- New public API at the top-level `hfutils` package: `from hfutils import detect_source, stream_merge, plan_pack, Source, SourceKind, PackOp, ConvertTarget, read_raw_header, __version__`.
- New `io/fs.py::check_free_space` preflights disk space in `convert` sub-sub-commands; refuses to start on insufficient space.

### Performance

- `inspect --recursive` walks parallelized via `ThreadPoolExecutor(max_workers=8)`. Output order stable via post-fan-in name sort.
- Benchmarked `os.copy_file_range` against Python buffered copy on a real 23 GB merge: 5.83s vs 5.67s (kernel path is 2.8% *slower* — we're disk-write-bound). Did not ship; kept the Python path.

### Accuracy

- `sources/detect._check_shard_integrity` reads each shard's header and confirms declared tensor-data size matches the physical file size. Truncated / corrupt shards now flag `Source.integrity_error` and show `CORRUPT` in `inspect --recursive`.
- `stream_merge` emits a warning for every metadata key whose value differs between shards (previously silent last-write-wins).
- New `_FAMILY_RULES` entries: Z-Image (`all_final_layer.` + `adaLN_modulation`) and `AutoencoderKL` (encoder.down_blocks + decoder.up_blocks). Deliberately did NOT add Qwen3 — its layout is indistinguishable from generic Llama-style LLMs without `config.json`, which `architecture_name_from_config` already handles.
- `GGUFInfo` gains `rope_freq_base`, `rope_freq_scale`, `rope_scaling_type`, `bos_token_id`, `eos_token_id`, `chat_template`. `inspect` surfaces them when present.

### Testing

- Memory test threshold tightened 16 MiB → 10 MiB (observed peak is 8 MiB).
- New integration test: diffusers pipelines with mixed `.safetensors` + `.bin` components. Surfaced a planner bug where zero-safetensors components emitted empty PackOps; fixed in layouts/comfyui.
- 141 tests total (was 101 at 0.5.0; +40).

### Internal

- 12 logical commits (visible via `git log --oneline`). `main` stayed green between each.

## 0.5.0

### Breaking CLI changes (no deprecation aliases)

- `hfutils merge <dir> <out>`          -> `hfutils convert single <src> <out>`
- `hfutils scan <dir>`                 -> `hfutils inspect <dir> --recursive`
- `hfutils comfyui pack <src> <root>`  -> `hfutils convert comfyui <src> <root>`

`convert single` accepts sharded directories AND single `.safetensors` files; it auto-detects and does the right thing.

### Memory

- `convert single` (and `convert comfyui` when merging sharded components) now uses a pure-Python streaming merger that never loads full tensors into RAM. Peak memory drops from O(model size) to ~4 MiB for any merge, regardless of model size.
- Verified: a ~23 GB sharded transformer merges with peak RSS in the low hundreds of MB.

### Runtime dependencies

- `torch` / `safetensors[torch]` are no longer runtime dependencies. Default install is lean (~2 GB smaller).
- Users who want torch can install `hfutils[ml]`.

### Organization

- New `Source` abstraction (`src/hfutils/sources/detect.py`) classifies any local path into one of: diffusers pipeline, component dir, single safetensors file, GGUF file, unknown. All commands consume this.
- New `layouts/comfyui.py` carries the ComfyUI folder tables and the `plan_pack(source, root, ...)` planner. Commands are thin shims.
- New `formats/safetensors.py` holds the streaming merge + raw-header reader.
- New `io/progress.py` consolidates progress-bar helpers; `merge` and `convert` no longer carry twin copies.
- `inspect/directory.py` index-file discovery now accepts any `*.safetensors.index.json` (fixes a diffusers detection bug that also existed in the old `scan`).

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
