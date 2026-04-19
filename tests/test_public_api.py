"""Tests that the top-level `hfutils` package exports the documented surface."""

import hfutils


def test_version_string():
    assert isinstance(hfutils.__version__, str)
    assert hfutils.__version__ == "0.6.0"


def test_detect_source_exported():
    assert callable(hfutils.detect_source)


def test_source_types_exported():
    assert hfutils.Source is not None
    assert hfutils.SourceKind is not None
    assert hfutils.SourceKind.DIFFUSERS_PIPELINE.value == "diffusers_pipeline"


def test_stream_merge_exported():
    assert callable(hfutils.stream_merge)


def test_plan_pack_exported():
    assert callable(hfutils.plan_pack)
    assert hfutils.PackOp is not None
    assert hfutils.ConvertTarget.VAE.value == "vae"


def test_read_raw_header_exported():
    assert callable(hfutils.read_raw_header)


def test_verify_output_exported():
    assert callable(hfutils.verify_output)
    assert callable(hfutils.manifest_from_shards)


def test_integrity_error_exported():
    err = hfutils.IntegrityError(kind="truncated", file=__import__("pathlib").Path("x"), detail="test")
    assert err.kind == "truncated"


def test_plan_single_exported():
    assert callable(hfutils.plan_single)
