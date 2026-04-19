"""Error hierarchy: every hfutils-specific exception inherits from HfutilsError
so library consumers can catch broadly or narrowly."""

from hfutils.errors import (
    HfutilsError,
    InsufficientSpaceError,
    PlanError,
    SourceError,
    StreamMergeError,
    VerificationError,
)


def test_all_errors_inherit_from_hfutils_error():
    subclasses = [
        SourceError,
        PlanError,
        StreamMergeError,
        VerificationError,
        InsufficientSpaceError,
    ]
    for cls in subclasses:
        assert issubclass(cls, HfutilsError), f"{cls.__name__} should inherit from HfutilsError"


def test_errors_importable_from_top_level():
    import hfutils

    assert hfutils.HfutilsError is HfutilsError
    assert hfutils.PlanError is PlanError
    assert hfutils.StreamMergeError is StreamMergeError
    assert hfutils.VerificationError is VerificationError
    assert hfutils.InsufficientSpaceError is InsufficientSpaceError
    assert hfutils.SourceError is SourceError


def test_insufficient_space_compat_reimport():
    """Old import path (io.fs) still exposes InsufficientSpaceError for one release."""
    from hfutils.io.fs import InsufficientSpaceError as FromIoFs

    assert FromIoFs is InsufficientSpaceError
