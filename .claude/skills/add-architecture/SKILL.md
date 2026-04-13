---
name: add-architecture
description: Add a new model architecture to the tensor-name detection table in architecture.py. Guides through discovering tensor patterns, writing a failing test, and adding the rule.
disable-model-invocation: true
---

# Add Architecture

Add a new model architecture to `src/hfutils/inspect/architecture.py`.

## Arguments

The user provides a model name or HF repo ID (e.g., "CogVideoX", "black-forest-labs/FLUX.1-dev").

## Workflow

### 1. Discover tensor name patterns

Get tensor names from a real model. Options:
- If user has a local safetensors file: `uv run hfutils inspect --detail <file>`
- If user provides an HF repo ID: use the Hugging Face MCP tools to look up model files, or search for the model's architecture documentation
- If neither: web search for "<model_name> safetensors tensor names" or check the model's source code for layer naming

Look for 2-3 tensor name prefixes that **uniquely** identify this architecture. Good patterns:
- Prefixes shared by ALL tensors of a specific type (e.g., `double_blocks.`, `transformer_blocks.`)
- Prefixes that distinguish this model from similar ones (e.g., Hunyuan Video has `txt_in.individual_token_refiner.` which Flux lacks)

### 2. Check for conflicts with existing rules

Read `src/hfutils/inspect/architecture.py` and check `_FAMILY_RULES`. Verify:
- The new patterns don't overlap with existing rules
- If they DO overlap, the new rule needs an additional distinguishing pattern and must be placed BEFORE the more general rule it overlaps with

### 3. Write a failing test (RED)

Add a test to `tests/test_architecture.py` in the `TestDetectDiffusionModels` class (or `TestDetectLLM` if it's an LLM):

```python
def test_<model_name>(self):
    names = [
        # 3-4 representative tensor names from the model
    ]
    result = detect_architecture(names)
    assert result.family == "<Model Name>"
```

Run: `uv run pytest tests/test_architecture.py -v -k test_<model_name>`
Confirm it fails with `assert 'Unknown' == '<Model Name>'` (or shows a wrong match).

### 4. Add the rule to `_FAMILY_RULES` (GREEN)

Add an entry to `_FAMILY_RULES` in `src/hfutils/inspect/architecture.py`:

```python
("<Model Name>", [
    "prefix_a.",
    "prefix_b.",
]),
```

Use string prefixes when possible. Use `re.compile(r"pattern")` only when the prefix contains variable parts (like layer numbers: `blocks\.\d+\.attn\.`).

**Placement matters**: more specific rules BEFORE more general ones. If the new architecture shares prefixes with an existing rule, place it above that rule with an additional distinguishing condition.

Run: `uv run pytest tests/test_architecture.py -v`
Confirm ALL architecture tests pass (not just the new one).

### 5. Run full test suite

`uv run pytest tests/ -q` -- all 62+ tests must pass.
