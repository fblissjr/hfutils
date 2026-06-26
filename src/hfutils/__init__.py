"""hfutils -- local model file toolkit.

Library surface for programmatic use. The CLI is at `hfutils.cli:app`.
"""

from hfutils.errors import (
    HfutilsError,
    InsufficientSpaceError,
    LoraConvertError,
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
    RichMergeObserver,
    RichObserver,
    per_op_merge_observer,
)
from hfutils.runner import PlanRunner
from hfutils.formats.lora import (
    KREA2_FAL,
    LORA_SCHEMES,
    FuseResult,
    LoraScheme,
    convert_lora_keys,
    detect_lora_scheme,
    fuse_lora,
)
from hfutils.formats.safetensors import (
    Manifest,
    manifest_from_shards,
    read_raw_header,
    stream_merge,
    verify_output,
)
from hfutils.layouts.comfyui import (
    ConvertTarget,
    PackOp,
    plan_comfyui,
    plan_pack,  # deprecated alias; dropped in 0.9
    plan_single,
)
from hfutils.layouts.plan import PackPlan
from hfutils.sources.detect import DetectLevel, detect_source, enrich
from hfutils.sources.types import (
    ComponentSource,
    EnrichedView,
    GgufFileSource,
    IntegrityError,
    PipelineSource,
    PytorchDirSource,
    SafetensorsFileSource,
    Source,
    UnknownSource,
)

__version__ = "0.8.0"

__all__ = [
    "CollectingMergeObserver",
    "CollectingObserver",
    "ComponentSource",
    "ConvertTarget",
    "DetectLevel",
    "EnrichedView",
    "FuseResult",
    "GgufFileSource",
    "HfutilsError",
    "InsufficientSpaceError",
    "IntegrityError",
    "KREA2_FAL",
    "LORA_SCHEMES",
    "LoraConvertError",
    "LoraScheme",
    "Manifest",
    "MergeObserver",
    "NullMergeObserver",
    "NullObserver",
    "Observer",
    "PackOp",
    "PackPlan",
    "PipelineSource",
    "PlanError",
    "PlanRunner",
    "RichObserver",
    "PytorchDirSource",
    "RichMergeObserver",
    "SafetensorsFileSource",
    "Source",
    "SourceError",
    "StreamMergeError",
    "UnknownSource",
    "VerificationError",
    "__version__",
    "convert_lora_keys",
    "detect_lora_scheme",
    "detect_source",
    "enrich",
    "fuse_lora",
    "manifest_from_shards",
    "per_op_merge_observer",
    "plan_comfyui",
    "plan_pack",
    "plan_single",
    "read_raw_header",
    "stream_merge",
    "verify_output",
]
