"""Weighted inverse permute for staged MoE PostPermute.

Inputs:
  expert_output    [materialized_rows, H] bf16/fp16 expert output
  inverse_indices  [T*K]                  int32 flat route → materialized row
  topk_weights     [T, K]                 float32 routing weights

Output:
  output           [T, H]                 bf16/fp16
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.buffer_utils import tensors_overlap
from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_count

__all__ = ["MoeUnpermuteKernel"]


@functools.lru_cache(maxsize=32)
def _make_unpermute_kernel(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    materialized_rows: int,
    dtype: str,
    threads: int,
    scaling: float = 1.0,
):
    """Build one block per output token with an fp32 reduction."""
    numel = num_tokens * top_k

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _unpermute():
        @T.prim_func
        def _unpermute_main(
            expert_output: T.Tensor([materialized_rows, hidden_size], dtype),
            inverse_indices: T.Tensor([numel], "int32"),
            topk_weights: T.Tensor([num_tokens, top_k], "float32"),
            output: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(num_tokens, threads=threads) as (token_idx,):
                acc = T.alloc_fragment([hidden_size], "float32")
                src = T.alloc_fragment([hidden_size], dtype)

                T.fill(acc, 0.0)

                # num_stages=2 overlaps the latency-bound gathers; top_k < 2 leaves
                # the pipeline deeper than the trip count, so it runs serial.
                for k in T.Pipelined(top_k, num_stages=2) if top_k >= 2 else T.serial(top_k):
                    flat_idx = token_idx * T.int32(top_k) + k
                    slot = inverse_indices[flat_idx]
                    weight = topk_weights[token_idx, k]
                    T.copy(expert_output[slot, 0:hidden_size], src)
                    for j in T.Parallel(hidden_size):
                        acc[j] = acc[j] + T.Cast("float32", src[j]) * weight

                out_frag = T.alloc_fragment([hidden_size], dtype)
                if scaling != 1.0:
                    for j in T.Parallel(hidden_size):
                        out_frag[j] = T.Cast(dtype, acc[j] * T.float32(scaling))
                else:
                    for j in T.Parallel(hidden_size):
                        out_frag[j] = T.Cast(dtype, acc[j])
                T.copy(out_frag, output[token_idx, 0:hidden_size])

        return _unpermute_main

    return _unpermute


class MoeUnpermuteKernel(Kernel):
    """Weighted inverse-permute kernel for staged PostPermute.

    Restores token order from staged inverse indices and applies weighted
    top-k reduction.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        hidden_size: Hidden dimension H.
        materialized_rows: Number of materialized expert-output rows.
        scaling: Scalar multiplied into the reduced output before the cast/store
            (folds ``routed_scaling_factor``). Defaults to 1.0 (no scaling).
        dtype: Data type of expert output and final output (bf16 or fp16).
        config: Optional config dict with ``threads``.
        tune: Whether to autotune.
        sm_count: Device SM count used to choose the low-token launch width.

    Example:
        ```python linenums="1"
        kernel = MoeUnpermuteKernel(num_tokens=4, top_k=2, hidden_size=128, materialized_rows=8)
        output = kernel(expert_output, inverse_indices, topk_weights)
        ```
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        materialized_rows: int,
        scaling: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
        tune: bool = False,
        sm_count: Optional[int] = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.materialized_rows = materialized_rows
        self.dtype = dtype
        self.numel = num_tokens * top_k
        self.sm_count = get_sm_count() if sm_count is None else sm_count
        if self.sm_count <= 0:
            raise ValueError("sm_count must be positive")
        self.init_config(config, tune)

        self._unpermute_fn = _make_unpermute_kernel(
            num_tokens,
            top_k,
            hidden_size,
            materialized_rows,
            self.dtype_str,
            self.config["threads"],
            scaling,
        )

    @property
    def default_config(self) -> dict:
        vector = 8
        vector_threads = min(1024, self.hidden_size // vector)
        while vector_threads > 0 and self.hidden_size % vector_threads != 0:
            vector_threads -= 1
        if self.num_tokens <= self.sm_count:
            return {"threads": max(vector_threads, 1)}
        threads = min(256, self.hidden_size // vector)
        if threads > 0:
            threads = 1 << (threads.bit_length() - 1)
        return {"threads": max(threads, 1)}

    def forward(
        self,
        expert_output: torch.Tensor,
        inverse_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run moe_unpermute.

        Args:
            expert_output: [materialized_rows, H] bf16/fp16 expert output.
            inverse_indices: [T*K] int32 flat route → materialized row.
            topk_weights: [T, K] float32 routing weights.
            out: optional [T, H] output buffer to write into and reuse across
                calls. Allocated internally with ``torch.empty`` if omitted.

        Returns:
            output: [T, H] bf16/fp16 (``out`` if provided).
        """
        assert inverse_indices.dtype == torch.int32
        assert topk_weights.dtype == torch.float32
        assert expert_output.is_cuda

        dev = expert_output.device
        if out is None:
            output = torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype, device=dev)
        else:
            if tuple(out.shape) != (self.num_tokens, self.hidden_size):
                raise ValueError(
                    f"out shape must be {(self.num_tokens, self.hidden_size)}, "
                    f"got {tuple(out.shape)}"
                )
            if out.dtype != self.dtype:
                raise ValueError(f"out dtype must be {self.dtype}, got {out.dtype}")
            # A cross-device or non-contiguous ``out`` would scatter the store.
            if out.device != dev:
                raise ValueError(f"out device must be {dev}, got {out.device}")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            # Overlap races with the concurrent read; disjoint slices of one
            # workspace buffer are fine.
            if tensors_overlap(out, expert_output):
                raise ValueError("out must not overlap expert_output in memory")
            output = out

        fn = self._unpermute_fn()
        fn(expert_output, inverse_indices, topk_weights, output)

        return output
