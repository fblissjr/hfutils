"""hfutils -- local model file toolkit.

Library surface for programmatic use. The CLI is at `hfutils.cli:app`.
"""

from hfutils.formats.safetensors import read_raw_header, stream_merge
from hfutils.layouts.comfyui import ConvertTarget, PackOp, plan_pack
from hfutils.sources.detect import Source, SourceKind, detect_source

__version__ = "0.6.0"

__all__ = [
    "ConvertTarget",
    "PackOp",
    "Source",
    "SourceKind",
    "__version__",
    "detect_source",
    "plan_pack",
    "read_raw_header",
    "stream_merge",
]
