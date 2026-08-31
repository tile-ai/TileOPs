"""Typed contracts shared by the staged Mixture-of-Experts operators."""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import TypeAlias

import torch

__all__ = [
    "ContiguousLayoutSpec",
    "MGroupedLayoutSpec",
    "MaskedLayoutSpec",
    "NoScaleComputeSpec",
    "RoutingEpilogueSpec",
    "routing_epilogue_reference",
]


class _LayoutKind(str, enum.Enum):
    """Concrete layout specializations understood by kernel candidates."""

    TIGHT_PHYSICAL_PSUM = "tight_physical_psum"
    TIGHT_PER_ROW = "tight_per_row"
    MASKED_PREDICATED = "masked_predicated"


@dataclasses.dataclass(frozen=True, init=False)
class ContiguousLayoutSpec:
    """Requested tight contiguous M-grouped layout semantics.

    Use a named constructor so callers select one supported semantic instead of
    assembling alignment, padding, and tile-boundary policies independently.
    Tight materialization always has alignment one, no padding rows, and
    requires a boundary-aware consumer.
    """

    _kind: _LayoutKind = dataclasses.field(repr=False)

    def __init__(self, kind: _LayoutKind) -> None:
        """Initialize one supported contiguous kind; callers should use a named preset."""
        if kind not in (
            _LayoutKind.TIGHT_PHYSICAL_PSUM,
            _LayoutKind.TIGHT_PER_ROW,
        ):
            raise ValueError(f"{kind!r} is not a contiguous layout kind")
        object.__setattr__(self, "_kind", kind)

    @classmethod
    def tight_physical_psum(cls) -> "ContiguousLayoutSpec":
        """Use tight rows described by per-expert physical segment ends."""
        return cls(_LayoutKind.TIGHT_PHYSICAL_PSUM)

    @classmethod
    def tight_per_row(cls) -> "ContiguousLayoutSpec":
        """Use tight rows described by one expert ID per materialized row."""
        return cls(_LayoutKind.TIGHT_PER_ROW)

    @property
    def selection_key(self) -> str:
        """Return the concrete specialization key recorded in CallSpecs."""
        return self._kind.value

    @property
    def max_m(self) -> None:
        """Contiguous layouts have no fixed per-expert capacity."""
        return None

    def __repr__(self) -> str:
        constructor = {
            _LayoutKind.TIGHT_PHYSICAL_PSUM: "tight_physical_psum",
            _LayoutKind.TIGHT_PER_ROW: "tight_per_row",
        }[self._kind]
        return f"ContiguousLayoutSpec.{constructor}()"


@dataclasses.dataclass(frozen=True)
class MaskedLayoutSpec:
    """Requested fixed-capacity masked M-grouped layout semantics."""

    max_m: int

    def __post_init__(self) -> None:
        if self.max_m < 0:
            raise ValueError("max_m must be non-negative")

    @property
    def selection_key(self) -> str:
        """Return the concrete specialization key recorded in CallSpecs."""
        return _LayoutKind.MASKED_PREDICATED.value


MGroupedLayoutSpec: TypeAlias = ContiguousLayoutSpec | MaskedLayoutSpec


@dataclasses.dataclass(frozen=True)
class RoutingEpilogueSpec:
    """Exactly-once local routing epilogue with fixed reduction/cast semantics."""

    routed_scaling_factor: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be finite and positive")

    @property
    def accumulation_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def output_dtype(self) -> torch.dtype:
        return torch.bfloat16


@dataclasses.dataclass(frozen=True)
class NoScaleComputeSpec:
    """SM90 BF16 grouped GEMM with FP32 accumulation and BF16 output."""

    @property
    def accumulation_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def output_dtype(self) -> torch.dtype:
        return torch.bfloat16


def routing_epilogue_reference(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_indices: torch.Tensor,
    epilogue: RoutingEpilogueSpec,
) -> torch.Tensor:
    """Apply inverse permutation and the routing epilogue in declared order."""
    if not expert_output.is_contiguous():
        raise ValueError("expert_output must be contiguous")
    if expert_output.ndim not in (2, 3):
        raise ValueError("expert_output must be contiguous rank 2 or masked rank 3")
    if topk_weights.ndim != 2:
        raise ValueError("topk_weights must have shape [tokens, top_k]")
    if inverse_indices.ndim != 1 or inverse_indices.numel() != topk_weights.numel():
        raise ValueError("inverse_indices must have one entry per routing weight")
    if topk_weights.dtype is not torch.float32:
        raise TypeError("topk_weights must have dtype torch.float32")
    if expert_output.device != inverse_indices.device:
        raise ValueError("expert_output and inverse indices must share a device")
    if expert_output.device != topk_weights.device:
        raise ValueError("expert_output and topk_weights must share a device")

    num_tokens, top_k = topk_weights.shape
    materialized_rows = expert_output.numel() // expert_output.shape[-1]
    flat_indices = inverse_indices.to(torch.int64)
    local = flat_indices >= 0
    flat_output = expert_output.reshape(materialized_rows, expert_output.shape[-1])
    if materialized_rows == 0:
        gathered = expert_output.new_zeros((num_tokens * top_k, expert_output.shape[-1]))
    else:
        gathered = flat_output.index_select(0, flat_indices.clamp_min(0))
    gathered = gathered.reshape(num_tokens, top_k, -1)
    local = local.reshape(num_tokens, top_k, 1)
    weighted = gathered.to(epilogue.accumulation_dtype) * topk_weights.unsqueeze(-1)
    weighted = torch.where(local, weighted, torch.zeros_like(weighted))
    reduced = weighted.sum(dim=1, dtype=epilogue.accumulation_dtype)
    scaled = reduced * epilogue.routed_scaling_factor
    return scaled.to(epilogue.output_dtype)
