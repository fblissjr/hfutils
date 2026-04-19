"""Property-based tests for stream_merge: random tensor sets, random shard
splits, round-trip equivalence.

Example-based tests in test_formats_safetensors.py cover the obvious shapes;
these tests blast random inputs at stream_merge looking for corner cases
(exotic dtypes, weird shapes, pathological shard distributions).
"""

from pathlib import Path

import torch
from hypothesis import given, settings, strategies as st
from safetensors.torch import load_file, save_file

from hfutils.events import CollectingMergeObserver
from hfutils.formats.safetensors import stream_merge


_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.uint8]


@st.composite
def tensor_dict(draw) -> dict[str, torch.Tensor]:
    """Generate a dict of tensor_name -> tensor with mixed dtypes and shapes."""
    n_tensors = draw(st.integers(min_value=1, max_value=8))
    names = [f"t{i}" for i in range(n_tensors)]
    tensors: dict[str, torch.Tensor] = {}
    for name in names:
        dtype = draw(st.sampled_from(_DTYPES))
        # Shapes biased toward small; the memory matters less than the dtype
        # and layout permutations.
        ndim = draw(st.integers(min_value=1, max_value=3))
        shape = [draw(st.integers(min_value=1, max_value=8)) for _ in range(ndim)]
        if dtype in (torch.int8, torch.uint8):
            tensors[name] = torch.randint(0, 100, shape, dtype=dtype)
        else:
            tensors[name] = torch.randn(*shape, dtype=dtype)
    return tensors


@st.composite
def shard_split(draw, tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Partition a tensor dict into 1..N shards (preserving tensor name order)."""
    names = list(tensors.keys())
    if len(names) == 1:
        return [tensors]
    # Choose how many shards; each shard non-empty.
    max_shards = min(len(names), 5)
    n_shards = draw(st.integers(min_value=1, max_value=max_shards))
    if n_shards == 1:
        return [tensors]
    # Draw split points: n_shards - 1 cut indices, strictly increasing.
    cuts = sorted(draw(st.lists(
        st.integers(min_value=1, max_value=len(names) - 1),
        min_size=n_shards - 1,
        max_size=n_shards - 1,
        unique=True,
    )))
    shards: list[dict[str, torch.Tensor]] = []
    prev = 0
    for c in cuts:
        shards.append({n: tensors[n] for n in names[prev:c]})
        prev = c
    shards.append({n: tensors[n] for n in names[prev:]})
    return shards


def _write_shards(tmp: Path, shards: list[dict[str, torch.Tensor]]) -> list[Path]:
    paths = []
    for i, tensors in enumerate(shards, 1):
        p = tmp / f"shard-{i:05d}.safetensors"
        save_file(tensors, p)
        paths.append(p)
    return paths


class TestStreamMergeRoundTrip:
    @given(data=st.data())
    @settings(max_examples=30, deadline=None)
    def test_multi_shard_round_trip(self, data, tmp_path_factory):
        """Split a random tensor dict across 1..5 shards; assert round-trip."""
        tensors = data.draw(tensor_dict())
        splits = data.draw(shard_split(tensors))
        tmp = tmp_path_factory.mktemp("multi")

        shards = _write_shards(tmp, splits)
        out = tmp / "merged.safetensors"

        stream_merge(shards, out)

        merged = load_file(str(out))
        assert set(merged.keys()) == set(tensors.keys())
        for k, v in tensors.items():
            assert merged[k].dtype == v.dtype
            assert list(merged[k].shape) == list(v.shape)
            assert torch.equal(merged[k].view(torch.uint8), v.view(torch.uint8))


class TestMetadataWarningProperty:
    @given(
        keys=st.lists(st.text(min_size=1, max_size=10, alphabet="abc"), min_size=1, max_size=4, unique=True),
        values_per_shard=st.lists(st.text(min_size=0, max_size=5, alphabet="xyz"), min_size=2, max_size=4),
    )
    @settings(max_examples=40, deadline=None)
    def test_warning_iff_conflict(self, keys, values_per_shard, tmp_path_factory):
        """A metadata-conflict warning fires exactly when at least one key has
        different values across shards."""
        tmp = tmp_path_factory.mktemp("meta")

        # Build shards: same tensor name across all, different metadata per shard.
        # Each shard gets {key: values_per_shard[shard_idx]} for every key.
        shard_paths = []
        for i, value in enumerate(values_per_shard, 1):
            meta = {k: value for k in keys}
            p = tmp / f"s{i}.safetensors"
            save_file({f"t{i}": torch.randn(2, 2)}, p, metadata=meta)
            shard_paths.append(p)

        obs = CollectingMergeObserver()
        stream_merge(shard_paths, tmp / "merged.safetensors", observer=obs)

        # Conflict if any key's value differs between shards (which happens iff
        # values_per_shard has more than one distinct value).
        expect_conflict = len(set(values_per_shard)) > 1
        if expect_conflict:
            assert obs.warnings, (
                f"Expected a warning for conflicting metadata values {values_per_shard} "
                f"on keys {keys}, got no warnings"
            )
        else:
            assert not obs.warnings, (
                f"Unexpected warning for identical metadata {values_per_shard} "
                f"on keys {keys}: {obs.warnings}"
            )
