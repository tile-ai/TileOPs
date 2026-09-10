"""Call records for staged Mixture-of-Experts implementation selection."""

import dataclasses
from typing import TYPE_CHECKING

import torch

from tileops.kernels.call_spec import CallSpec

if TYPE_CHECKING:
    from tileops.ops.moe.contracts import (
        MGroupedLayoutSpec,
        RoutingEpilogueSpec,
    )

__all__ = ["MGroupedGemmCall", "PostPermuteCall", "PrePermuteCall"]


@dataclasses.dataclass(frozen=True)
class PrePermuteCall(CallSpec):
    """Complete selection facts for one pre-permute invocation."""

    layout: "MGroupedLayoutSpec | None" = None
    device_type: str = ""
    input_dtype: torch.dtype | None = None
    num_experts: int = 0
    num_tokens: int = 0
    hidden_size: int = 0
    top_k: int = 0
    routing_input_kind: str = "topk_ids"


@dataclasses.dataclass(frozen=True)
class MGroupedGemmCall(CallSpec):
    """Complete selection facts for one M-grouped GEMM invocation.

    The layout arrives structured — ``kind`` first, then the contiguous
    sub-axes — so a candidate can claim a region such as "every contiguous
    layout with psum metadata" without enumerating keys. ``m`` is the
    materialized row count (``num_groups * max_m`` for masked layouts); it is a
    fact of the call and not of the built kernel, so the op keys its kernel
    cache on this record with ``m`` reset.
    """

    kind: str = ""  # "contiguous" | "masked"
    packing: str | None = None  # "tight" | "aligned"; None for masked
    metadata_kind: str | None = None  # "physical_psum" | "per_row"; None for masked
    alignment: int = 1  # 1 unless packing == "aligned"
    max_m: int | None = None  # masked only
    # Gated activation fused into the epilogue ("silu_and_mul", ...); None for a
    # plain GEMM. With one, ``n`` is B's stacked gate||up width and C has n / 2.
    activation: str | None = None
    ab_dtype: torch.dtype | None = None
    cd_dtype: torch.dtype | None = None
    num_groups: int = 0
    m: int = 0
    n: int = 0
    k: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        # A record naming no layout is the all-defaults one ``CallSpec.__str__``
        # diffs against; every other one is held to what a layout spec can express.
        layout_fields = (self.kind, self.packing, self.metadata_kind, self.alignment, self.max_m)
        if layout_fields == ("", None, None, 1, None):
            return
        if self.kind == "contiguous":
            if self.packing not in ("tight", "aligned"):
                raise ValueError(
                    f"contiguous layouts pack 'tight' or 'aligned', got {self.packing!r}"
                )
            if self.metadata_kind not in ("physical_psum", "per_row"):
                raise ValueError(
                    f"contiguous metadata is 'physical_psum' or 'per_row', got {self.metadata_kind!r}"
                )
            if self.max_m is not None:
                raise ValueError("contiguous layouts have no max_m")
            if self.packing == "tight" and self.alignment != 1:
                raise ValueError("tight packing has alignment 1")
            if self.packing == "aligned" and self.alignment < 2:
                raise ValueError("aligned packing has alignment > 1")
        elif self.kind == "masked":
            if self.packing is not None or self.metadata_kind is not None or self.alignment != 1:
                raise ValueError("masked layouts carry no packing, metadata_kind or alignment")
            if self.max_m is None or self.max_m < 0:
                raise ValueError("masked layouts need a non-negative max_m")
        else:
            raise ValueError(f"kind is 'contiguous' or 'masked', got {self.kind!r}")


@dataclasses.dataclass(frozen=True)
class PostPermuteCall(CallSpec):
    """Complete selection facts for one post-permute invocation."""

    layout_key: str = ""
    max_m: int | None = None
    epilogue: "RoutingEpilogueSpec | None" = None
    device_type: str = ""
    input_dtype: torch.dtype | None = None
    routing_weight_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    num_experts: int = 0
    materialized_rows: int = 0
    num_tokens: int = 0
    hidden_size: int = 0
    top_k: int = 0
