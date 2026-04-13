"""Shared types and formatting for model inspection."""

from dataclasses import dataclass, field


def format_size(size_bytes: int, decimals: int = 2) -> str:
    """Format byte count as human-readable string."""
    if size_bytes >= 1 << 30:
        return f"{size_bytes / (1 << 30):.{decimals}f} GB"
    if size_bytes >= 1 << 20:
        return f"{size_bytes / (1 << 20):.{decimals}f} MB"
    if size_bytes >= 1 << 10:
        return f"{size_bytes / (1 << 10):.{decimals}f} KB"
    return f"{size_bytes} B"


def format_params(count: int) -> str:
    """Format parameter count as human-readable string."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


# Bytes per element for each dtype
DTYPE_SIZES: dict[str, float] = {
    "F64": 8,
    "F32": 4,
    "BF16": 2,
    "F16": 2,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 0.125,
}


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str

    @property
    def param_count(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        bpe = DTYPE_SIZES.get(self.dtype, 0)
        return int(self.param_count * bpe)


@dataclass
class DtypeBreakdown:
    dtype: str
    param_count: int
    tensor_count: int
    size_bytes: int


@dataclass
class SafetensorsHeader:
    tensors: list[TensorInfo]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def total_params(self) -> int:
        return sum(t.param_count for t in self.tensors)

    @property
    def total_size_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors)

    def dtype_breakdown(self) -> list[DtypeBreakdown]:
        by_dtype: dict[str, DtypeBreakdown] = {}
        for t in self.tensors:
            if t.dtype not in by_dtype:
                by_dtype[t.dtype] = DtypeBreakdown(t.dtype, 0, 0, 0)
            b = by_dtype[t.dtype]
            b.param_count += t.param_count
            b.tensor_count += 1
            b.size_bytes += t.size_bytes
        return sorted(by_dtype.values(), key=lambda b: b.size_bytes, reverse=True)
