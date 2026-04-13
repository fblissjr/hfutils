# hfutils

Local model file toolkit for safetensors and GGUF files. Not a downloader -- use `hf download` for that.

## Commands

- `hfutils inspect <file>` -- read safetensors/GGUF headers (no torch needed for inspect)
- `hfutils inspect <directory>` -- combined config.json + model file inspection
- `hfutils inspect --detail <path>` -- full tensor list
- `hfutils merge <input_dir> <output>` -- merge sharded safetensors (requires `safetensors[torch]`)
- `hfutils scan <directory>` -- audit local model directories

## Project layout

```
src/hfutils/
  cli.py              -- typer app entry point
  commands/            -- one file per CLI subcommand
  inspect/             -- header parsing, architecture detection, shared types
    safetensors.py     -- binary header reader (no external deps)
    gguf.py            -- binary header reader (no external deps)
    architecture.py    -- table-driven tensor-name pattern matching
    directory.py       -- directory-level inspection (config.json + files)
    common.py          -- shared types and formatters
tests/                 -- pytest, mirrors src structure
```

## Dev

- `uv sync` to install
- `uv run pytest` to test (43 tests)
- `uv run hfutils --help` to verify CLI
- TDD: write failing test first, then implement

## Dependencies

Core: typer-slim, orjson, rich. No ML libraries needed at runtime.
Merge only: safetensors[torch] (optional, installed via `uv add 'hfutils[merge]'`).
Dev only: pytest, safetensors[torch], gguf (for test file creation).

## Conventions

- All JSON via orjson, never stdlib json
- Rich console for all user-facing output
- Both safetensors and GGUF inspection use purpose-built binary header readers (no external libs)
- Architecture detection is table-driven (`_FAMILY_RULES` in architecture.py) -- add new architectures by adding entries to the table
