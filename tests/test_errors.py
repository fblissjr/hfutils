"""Error hierarchy: every hfutils-specific exception inherits from HfutilsError
so library consumers can catch broadly or narrowly."""

from hfutils.errors import (
    HfutilsError,
    InsufficientSpaceError,
    PlanError,
    StreamMergeError,
)


def test_all_errors_inherit_from_hfutils_error():
    subclasses = [
        PlanError,
        StreamMergeError,
        InsufficientSpaceError,
    ]
    for cls in subclasses:
        assert issubclass(cls, HfutilsError), f"{cls.__name__} should inherit from HfutilsError"


def test_errors_importable_from_top_level():
    import hfutils

    assert hfutils.HfutilsError is HfutilsError
    assert hfutils.PlanError is PlanError
    assert hfutils.StreamMergeError is StreamMergeError
    assert hfutils.InsufficientSpaceError is InsufficientSpaceError


def test_insufficient_space_compat_reimport():
    """Old import path (io.fs) still exposes InsufficientSpaceError for one release."""
    from hfutils.io.fs import InsufficientSpaceError as FromIoFs

    assert FromIoFs is InsufficientSpaceError
