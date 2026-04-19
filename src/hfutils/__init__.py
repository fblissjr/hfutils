"""hfutils -- local model file toolkit.

Library surface for programmatic use. The CLI is at `hfutils.cli:app`.
"""

from hfutils.errors import (
    HfutilsError,
    InsufficientSpaceError,
    PlanError,
    SourceError,
    StreamMergeError,
    VerificationError,
)
from hfutils.formats.safetensors import (
    Manifest,
    manifest_from_shards,
    read_raw_header,
    stream_merge,
    verify_output,
)
from hfutils.layouts.comfyui import ConvertTarget, PackOp, plan_pack, plan_single
from hfutils.sources.detect import IntegrityError, Source, SourceKind, detect_source

__version__ = "0.6.0"

__all__ = [
    "ConvertTarget",
    "HfutilsError",
    "InsufficientSpaceError",
    "IntegrityError",
    "Manifest",
    "PackOp",
    "PlanError",
    "Source",
    "SourceError",
    "SourceKind",
    "StreamMergeError",
    "VerificationError",
    "__version__",
    "detect_source",
    "manifest_from_shards",
    "plan_pack",
    "plan_single",
    "read_raw_header",
    "stream_merge",
    "verify_output",
]
