"""Exception hierarchy for hfutils.

All internal raises go through these so library consumers can `except
HfutilsError` broadly or pick a specific subtype. The CLI's top-level handler
maps each subtype to an exit code."""


class HfutilsError(Exception):
    """Base for every error hfutils raises."""


class SourceError(HfutilsError):
    """Failure while classifying or validating a source path."""


class PlanError(HfutilsError):
    """Failure while building a PackPlan (e.g. missing --as, unknown target)."""


class StreamMergeError(HfutilsError):
    """Failure during stream_merge (unexpected EOF, duplicate tensor, etc.)."""


class VerificationError(HfutilsError):
    """Post-merge --verify found a mismatch between plan and output."""


class InsufficientSpaceError(HfutilsError):
    """Preflight disk-space check rejected the planned write."""
