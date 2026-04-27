"""MoE modular interface — ABC definitions and shared data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor

__all__ = [
    "MoEExperts",
    "MoEExpertsModular",
    "MoEPrepareAndFinalize",
    "PrepareResult",
    "WeightedReduce",
    "WeightedReduceNoOp",
]


@dataclass
class PrepareResult:
    """Return value of MoEPrepareAndFinalize.prepare().

    T = original token count; T' = post-dispatch count (T'==T when no EP).
    """

    hidden_q: Tensor        # [T', H]  quantized or pass-through hidden states
    scale: Tensor | None    # [T', *]  quantization scale; None for bf16/fp16
    topk_weights: Tensor    # [T', K]  float32, may be remapped by EP dispatch
    topk_ids: Tensor        # [T', K]  int32,   may be remapped by EP dispatch


class WeightedReduce(ABC):
    """Apply topk_weights to expert outputs and reduce to [T, H].

    Provided by MoEExpertsModular.make_weighted_reduce() and called inside
    MoEPrepareAndFinalize.finalize().
    """

    @abstractmethod
    def apply(
        self,
        output: Tensor,        # [T, H]  write destination
        expert_out: Tensor,    # output of MoEExperts.apply()
        topk_weights: Tensor,  # [T', K] float32
        topk_ids: Tensor,      # [T', K] int32
    ) -> None: ...


class WeightedReduceNoOp(WeightedReduce):
    """MoEExperts.apply() has already completed weighted reduction; output is [T, H].

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


class MoEPrepareAndFinalize(ABC):
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
        expert_out: Tensor,              # output of MoEExperts.apply()
        topk_weights: Tensor,            # [T', K] float32
        topk_ids: Tensor,                # [T', K] int32
        weight_and_reduce: WeightedReduce,
    ) -> None:
        """Post-condition: output.shape == (T, H)  (T = original token count)."""


class MoEExperts(ABC):
    """Abstraction over the expert GEMM computation.

    Responsibilities:
    - workspace_shapes(): declare scratch memory needs
    - output_shape(): declare the shape apply() writes
    - apply(): full expert computation (permute + GEMM + activation + GEMM)

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
        """Return the shape of the tensor written by apply().

        Implementations that perform internal unpermute + weighted reduction
        (Nopad, Padded) return (T_prime, H).  No-EP: T_prime == T.
        Implementations that do not reduce return (T_prime * topk, H).
        """

    @abstractmethod
    def apply(
        self,
        output: Tensor,           # pre-allocated, shape == output_shape()
        hidden_q: Tensor,         # [T', H] from PrepareResult.hidden_q
        w1: Tensor,               # [E, 2F, H]
        w2: Tensor,               # [E, H, F]
        topk_weights: Tensor,     # [T', K] float32
        topk_ids: Tensor,         # [T', K] int32
        num_experts: int,
        expert_map: Tensor | None,
        workspace1: Tensor,
        workspace2: Tensor,
    ) -> None:
        """Write expert computation result to output in-place."""


class MoEExpertsModular(MoEExperts, ABC):
    """Extends MoEExperts with pluggable weighted reduction.

    Exposes make_weighted_reduce() so MoEPrepareAndFinalize.finalize() can
    reuse the expert's native reduction kernel.
    """

    @abstractmethod
    def make_weighted_reduce(self) -> WeightedReduce:
        """Return the WeightedReduce instance used by this implementation.

        Implementations that perform internal weighted reduction return
        WeightedReduceNoOp(); others return a kernel-specific subclass.
        """
