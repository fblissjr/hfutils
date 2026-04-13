# hfutils

Local model file toolkit for safetensors and GGUF files.

## Install

```
uv add hfutils
uv add 'hfutils[merge]'  # adds safetensors[torch] for merge command
```

## Commands

```
hfutils inspect model.safetensors          # header summary: tensors, params, VRAM, architecture
hfutils inspect --detail model.safetensors # full tensor list
hfutils inspect model.gguf                 # GGUF metadata: arch, quant, context length
hfutils inspect ./model-dir/               # config.json + model file headers combined
hfutils merge ./sharded-model/ out.safetensors  # merge sharded safetensors into one file
hfutils scan /path/to/models/              # audit local model dirs: format, size, completeness
```
