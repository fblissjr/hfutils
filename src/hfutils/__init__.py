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
from hfutils.events import (
    CollectingMergeObserver,
    CollectingObserver,
    MergeObserver,
    NullMergeObserver,
    NullObserver,
    Observer,
    per_op_merge_observer,
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
    "CollectingMergeObserver",
    "CollectingObserver",
    "ConvertTarget",
    "HfutilsError",
    "InsufficientSpaceError",
    "IntegrityError",
    "Manifest",
    "MergeObserver",
    "NullMergeObserver",
    "NullObserver",
    "Observer",
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
    "per_op_merge_observer",
    "plan_pack",
    "plan_single",
    "read_raw_header",
    "stream_merge",
    "verify_output",
]
