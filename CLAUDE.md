# hfutils

Local model file toolkit for safetensors/GGUF files + CivitAI integration.

## Commands

```
hfutils inspect <file|dir>         # safetensors/GGUF headers, architecture detection, config.json
hfutils inspect --detail <path>    # full tensor list
hfutils merge <input_dir> <output> # sharded safetensors -> single file
hfutils scan <directory>           # audit local model storage
hfutils civitai search <query>     # search CivitAI models
hfutils civitai info <id|url>      # model details and versions
hfutils civitai dl <id|url>        # download with resume
```

## Layout

```
src/hfutils/
  cli.py                -- typer app, registers commands + sub-apps
  commands/
    inspect_cmd.py      -- inspect command (safetensors, GGUF, directories)
    merge.py            -- merge sharded safetensors
    scan.py             -- local directory audit
    civitai.py          -- civitai sub-app (search, info, dl)
  inspect/
    common.py           -- shared types (TensorInfo, SafetensorsHeader) + formatters (format_size, format_params)
    safetensors.py      -- binary header reader (struct.unpack, no safetensors lib)
    gguf.py             -- binary header reader (struct.unpack, no gguf lib)
    architecture.py     -- table-driven model family detection (_FAMILY_RULES)
    directory.py        -- directory-level inspection (config.json + model files)
  providers/
    civitai.py          -- CivitaiClient API client + parse_model_ref + primary_file
    download.py         -- shared download with rich progress + resume (urllib)
tests/                  -- 62 tests, pytest
```

## Dev

- `uv sync && uv run pytest` -- install + test
- `uv run hfutils --help` -- verify CLI
- TDD: failing test first, then implement

## Dependencies

Core: typer, orjson, rich, safetensors[torch]. Dev: pytest, gguf.

## Conventions

- All JSON via orjson, never stdlib json
- Rich console for all user-facing output
- Provider pattern: providers/<name>.py (client) + commands/<name>.py (typer sub-app), register in cli.py via `app.add_typer()`
- Provider clients own their auth (via `auth_headers` property); commands pass it through, never reconstruct it
- Architecture detection is table-driven (`_FAMILY_RULES`); add new architectures by adding table entries
- Shared formatters in inspect/common.py (format_size, format_params); don't duplicate

## Gotchas

- Do NOT add HF download/search/info commands -- `hf` CLI handles those
- Use `typer` (full package), NOT `typer-slim` -- typer-slim doesn't provide the `typer` import
- GGUF parser is hand-rolled (struct.unpack) -- the gguf library memmaps entire files including tensor data
- Safetensors header: first 8 bytes = LE uint64 header length, then JSON, then tensor data. Only read the header.
- `_FAMILY_RULES` order matters: more specific patterns first (Hunyuan Video before Flux)
- CivitAI needs a realistic User-Agent (DEFAULT_HEADERS in providers/download.py)
- CivitAI API key via `CIVITAI_API_KEY` env var only (no config file)
- Test GGUF fixtures use gguf library's GGUFWriter (dev dep only); runtime has no gguf dependency
