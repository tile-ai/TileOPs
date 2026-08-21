"""Call records for staged Mixture-of-Experts implementation selection."""

import dataclasses
from typing import TYPE_CHECKING

import torch

from tileops.kernels.call_spec import CallSpec

if TYPE_CHECKING:
    from tileops.ops.moe.contracts import (
        ComputeFamilyKey,
        GroupedGemmComputeSpec,
        MGroupedLayoutSpec,
        ResolvedMGroupedLayout,
        RoutingEpilogueSpec,
    )

__all__ = ["MGroupedGemmCall", "PostPermuteCall", "PrePermuteCall"]


@dataclasses.dataclass(frozen=True)
class PrePermuteCall(CallSpec):
    """Complete selection facts for one pre-permute invocation."""

    layout: "MGroupedLayoutSpec | None" = None
    device_type: str = ""
    input_dtype: torch.dtype | None = None
    num_tokens: int = 0
    hidden_size: int = 0
    top_k: int = 0
    routing_input_kind: str = "topk_ids"


@dataclasses.dataclass(frozen=True)
class MGroupedGemmCall(CallSpec):
    """Complete selection facts for one typed M-grouped GEMM invocation."""

    compute: "GroupedGemmComputeSpec | None" = None
    compute_family: "ComputeFamilyKey | None" = None
    layout: "ResolvedMGroupedLayout | None" = None
    device_type: str = ""
    input_dtype: torch.dtype | None = None
    weight_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    materialized_rows: int = 0
    num_experts: int = 0
    n: int = 0
    k: int = 0


@dataclasses.dataclass(frozen=True)
class PostPermuteCall(CallSpec):
    """Complete selection facts for one post-permute invocation."""

    layout: "ResolvedMGroupedLayout | None" = None
    epilogue: "RoutingEpilogueSpec | None" = None
    device_type: str = ""
    input_dtype: torch.dtype | None = None
    routing_weight_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    num_tokens: int = 0
    hidden_size: int = 0
    top_k: int = 0
