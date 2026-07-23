"""MoE modular interface — ABC definitions and shared data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from tileops.ops.op_base import Op

__all__ = [
    "ExpertBatch",
    "ExpertBatchOutput",
    "FusedMoEExperts",
    "FusedMoEExpertsModular",
    "FusedMoEPrepareAndFinalize",
    "PrepareResult",
    "WeightedReduce",
    "WeightedReduceNoOp",
]


@dataclass(frozen=True)
class ExpertBatch:
    """Canonical communication-to-compute contract for routed expert rows.

    ``hidden`` uses a tight expert-major layout. Expert ``e`` owns rows
    ``expert_offsets[e]:expert_offsets[e + 1]``; adjacent equal offsets encode
    an empty expert. TileOps borrows all input buffers for the duration of the
    call and does not mutate them.
    """

    hidden: Tensor
    expert_offsets: Tensor
    valid_rows: Tensor | None = None
    layout: str = "tight"

    def __post_init__(self) -> None:
        if self.hidden.ndim != 2:
            raise ValueError(
                f"hidden must be rank 2 [capacity, H], got {self.hidden.shape}"
            )
        if self.expert_offsets.ndim != 1:
            raise ValueError(
                "expert_offsets must be rank 1 [E_local + 1], got "
                f"{self.expert_offsets.shape}"
            )
        if self.expert_offsets.dtype != torch.int32:
            raise ValueError(
                "expert_offsets must use torch.int32, got "
                f"{self.expert_offsets.dtype}"
            )
        if self.hidden.device != self.expert_offsets.device:
            raise ValueError("hidden and expert_offsets must be on the same device")
        if self.layout != "tight":
            raise ValueError(
                f"only layout='tight' is supported, got {self.layout!r}"
            )
        if self.valid_rows is not None:
            if self.valid_rows.numel() != 1:
                raise ValueError("valid_rows must be a scalar tensor")
            if self.valid_rows.device != self.hidden.device:
                raise ValueError(
                    "valid_rows and hidden must be on the same device"
                )

    @property
    def capacity(self) -> int:
        return self.hidden.shape[0]


@dataclass(frozen=True)
class ExpertBatchOutput:
    """Expert MLP output; routing weights have not been applied."""

    hidden: Tensor
    valid_rows: Tensor | None = None
    row_order_preserved: bool = True
    routing_weights_applied: bool = False


def _validate_fused_moe_experts_dtypes(
    op_dtype: torch.dtype,
    output: Tensor,
    hidden_states: Tensor,
    w_gate_up: Tensor,
    w_down: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    expert_map: Tensor | None,
    workspace1: Tensor,
    workspace2: Tensor,
) -> None:
    """Shared dtype validator for FusedMoEExperts subclasses.

    Concrete subclasses route through this helper because the manifest-driven
    ``_validate_dtypes`` codegen path does not handle ``Optional[Tensor]``
    inputs (``expert_map`` is None for single-GPU); having one shared body
    avoids drift between the nopad and padded implementations.
    """
    allowed = (torch.float16, torch.bfloat16)
    if op_dtype not in allowed:
        raise ValueError(f"op dtype must be one of {allowed}, got {op_dtype}")
    for name, t in (
        ("output", output),
        ("hidden_states", hidden_states),
        ("w_gate_up", w_gate_up),
        ("w_down", w_down),
    ):
        if t.dtype != op_dtype:
            raise ValueError(
                f"Expected {name}.dtype == op dtype ({op_dtype}), got {t.dtype}"
            )
    if topk_weights.dtype != torch.float32:
        raise ValueError(f"Expected topk_weights.dtype == float32, got {topk_weights.dtype}")
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"Expected topk_ids.dtype == int32, got {topk_ids.dtype}")
    if expert_map is not None and expert_map.dtype != torch.int32:
        raise ValueError(f"Expected expert_map.dtype == int32, got {expert_map.dtype}")
    for name, t in (("workspace1", workspace1), ("workspace2", workspace2)):
        if t.dtype not in allowed:
            raise ValueError(f"Expected {name}.dtype in {allowed}, got {t.dtype}")


@dataclass
class PrepareResult:
    """Return value of FusedMoEPrepareAndFinalize.prepare().

    T = original token count; T' = post-dispatch count (T'==T when no EP).
    """

    hidden_q: Tensor        # [T', H]  quantized or pass-through hidden states
    scale: Tensor | None    # [T', *]  quantization scale; None for bf16/fp16
    topk_weights: Tensor    # [T', K]  float32, may be remapped by EP dispatch
    topk_ids: Tensor        # [T', K]  int32,   may be remapped by EP dispatch


class WeightedReduce(ABC):
    """Apply topk_weights to expert outputs and reduce to [T, H].

    Provided by FusedMoEExpertsModular.make_weighted_reduce() and called inside
    FusedMoEPrepareAndFinalize.finalize().
    """

    @abstractmethod
    def apply(
        self,
        output: Tensor,        # [T, H]  write destination
        expert_out: Tensor,    # output of FusedMoEExperts.forward()
        topk_weights: Tensor,  # [T', K] float32
        topk_ids: Tensor,      # [T', K] int32
    ) -> None: ...


class WeightedReduceNoOp(WeightedReduce):
    """FusedMoEExperts.forward() has already completed weighted reduction; output is [T, H].

    finalize() copies expert_out to output (no-op when they are the same tensor).
    """

    def apply(
        self,
        output: Tensor,
        expert_out: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
    ) -> None:
        if output is not expert_out:
            output.copy_(expert_out, non_blocking=True)


class FusedMoEPrepareAndFinalize(ABC):
    """Abstraction over EP/TP communication and optional quantization.

    Responsibilities:
    - prepare(): optional quantization + EP All-to-All dispatch
    - finalize(): weighted reduction + EP All-to-All gather

    Out of scope: any GEMM computation, physical token permutation.
    """

    @abstractmethod
    def prepare(
        self,
        hidden: Tensor,             # [T, H]
        topk_weights: Tensor,       # [T, K] float32
        topk_ids: Tensor,           # [T, K] int32
        num_experts: int,
        expert_map: Tensor | None,  # [E_global] int32, for EP local filtering
    ) -> PrepareResult:
        """Post-conditions:
            result.hidden_q.shape[1] == H
            no EP: result.hidden_q.shape[0] == T
        """

    @abstractmethod
    def finalize(
        self,
        output: Tensor,                  # [T, H]  write destination (pre-allocated)
        expert_out: Tensor,              # output of FusedMoEExperts.forward()
        topk_weights: Tensor,            # [T', K] float32
        topk_ids: Tensor,                # [T', K] int32
        weight_and_reduce: WeightedReduce,
    ) -> None:
        """Post-condition: output.shape == (T, H)  (T = original token count)."""


class FusedMoEExperts(Op, ABC):
    """Abstraction over the expert GEMM computation.

    Responsibilities:
    - workspace_shapes(): declare scratch memory needs
    - output_shape(): declare the shape forward() writes
    - forward(): full expert computation (permute + GEMM + activation + GEMM)

    Out of scope: routing, EP communication, quantization.
    """

    @abstractmethod
    def workspace_shapes(
        self,
        M: int,           # T' (post-dispatch token count)
        N: int,           # ffn_size
        K: int,           # hidden_size
        topk: int,
        num_experts: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return (workspace1_shape, workspace2_shape) in element count (not bytes).

        workspace1: gate_up GEMM output buffer.
        workspace2: post-activation buffer.
        Implementations with no external workspace return ((0,), (0,)).
        """

    @abstractmethod
    def output_shape(self, T_prime: int, H: int) -> tuple[int, int]:
        """Return the shape of the tensor written by forward().

        Implementations that perform internal unpermute + weighted reduction
        (Nopad, Padded) return (T_prime, H).  No-EP: T_prime == T.
        Implementations that do not reduce return (T_prime * topk, H).
        """

    @abstractmethod
    def forward(
        self,
        output: Tensor,           # pre-allocated, shape == output_shape()
        hidden_states: Tensor,    # [T', H] from PrepareResult.hidden_q
        w_gate_up: Tensor,        # [E, 2F, H]
        w_down: Tensor,           # [E, H, F]
        topk_weights: Tensor,     # [T', K] float32
        topk_ids: Tensor,         # [T', K] int32
        expert_map: Tensor | None,
        workspace1: Tensor,
        workspace2: Tensor,
        num_experts: int,
    ) -> None:
        """Write expert computation result to output in-place."""


class FusedMoEExpertsModular(FusedMoEExperts, ABC):
    """Extends FusedMoEExperts with pluggable weighted reduction.

    Exposes make_weighted_reduce() so FusedMoEPrepareAndFinalize.finalize() can
    reuse the expert's native reduction kernel.
    """

    @abstractmethod
    def make_weighted_reduce(self) -> WeightedReduce:
        """Return the WeightedReduce instance used by this implementation.

        Implementations that perform internal weighted reduction return
        WeightedReduceNoOp(); others return a kernel-specific subclass.
        """
