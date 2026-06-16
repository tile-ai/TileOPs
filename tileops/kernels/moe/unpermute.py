"""MoE token unpermute kernel (cutlass path).

Scatters expert outputs back to original token order, applies routing weights,
and reduces K expert contributions per token.

One block per token (T blocks total):
  - For each of K expert slots: load mm2_pad[fwd_idx[i*K+k]] (vectorized 128-bit)
  - Multiply by topk_weights[i, k] (float32)
  - Accumulate into float32 thread-local buffer
  - Cast to output dtype and store to output[i]

Inputs:
  mm2_pad          [padded_batch_sum, H]  bf16/fp16 down-proj output (padded layout)
  fwd_idx          [T*K]                  int32 forward mapping: flat_idx → padded slot
  topk_weights     [T, K]                 float32 routing weights

Output:
  output           [T, H]     bf16/fp16
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.buffer_utils import tensors_overlap
from tileops.kernels.kernel_base import Kernel

__all__ = ["MoeUnpermuteKernel"]


def _make_unpermute_kernel(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    padded_batch_sum: int,
    dtype: str,
    scaling: float = 1.0,
):
    """One block per token; threads cooperate over the H dimension.

    Threads are capped at 256 (see below), so each thread handles ceil(H /
    threads) elements: a clean multiple of VEC=8 (128-bit uint4 load/store)
    when H // threads is, otherwise partially vectorized (e.g. H=7168 -> 256
    threads -> 28 elems/thread = 3.5x VEC). Accumulation is in float32, cast to
    dtype on store.
    """
    # Cap threads at 256 (the sweep optimum at H=7168: 512 is ~3% slower and
    # 1024 spills the fp32 acc[H] accumulator to local memory) but scale with
    # hidden_size // VEC and keep power-of-2 alignment, so small hidden sizes
    # retain 128-bit (VEC=8) vectorized load/store instead of slow scalar 16-bit
    # ops (e.g. H=512 -> 64 threads of 8 elems each, ~14% faster than 256).
    VEC = 8  # 8 x bf16/fp16 = 128 bits
    threads = min(256, hidden_size // VEC)
    if threads > 0:
        threads = 1 << (threads.bit_length() - 1)
    threads = max(threads, 1)

    numel = num_tokens * top_k

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _unpermute():

        @T.prim_func
        def _unpermute_main(
            mm2_pad: T.Tensor([padded_batch_sum, hidden_size], dtype),    # noqa: F821
            fwd_idx: T.Tensor([numel], "int32"),                          # noqa: F821
            topk_weights: T.Tensor([num_tokens, top_k], "float32"),       # noqa: F821
            output: T.Tensor([num_tokens, hidden_size], dtype),            # noqa: F821
        ):
            with T.Kernel(num_tokens, threads=threads) as (token_idx,):
                # float32 accumulator for this token's H slice
                acc = T.alloc_fragment([hidden_size], "float32")
                src = T.alloc_fragment([hidden_size], dtype)

                # zero accumulator
                T.fill(acc, 0.0)

                # accumulate K expert contributions. Software-pipeline the
                # gathers (num_stages=2) so each scattered row load overlaps the
                # previous slot's accumulate — the K loads are latency-bound.
                # Serial fallback when top_k < 2 (pipeline depth > trip count).
                for k in (T.Pipelined(top_k, num_stages=2)
                          if top_k >= 2 else T.serial(top_k)):
                    flat_idx = token_idx * T.int32(top_k) + k
                    raw_slot = fwd_idx[flat_idx]
                    # EP mode: fwd_idx == -1 marks non-local expert → zero contribution.
                    # Use slot 0 as a safe dummy read; zero the weight to suppress output.
                    safe_slot = T.if_then_else(
                        raw_slot >= T.int32(0), raw_slot, T.int32(0)
                    )
                    weight = topk_weights[token_idx, k] * T.Cast(
                        "float32",
                        T.if_then_else(raw_slot >= T.int32(0), T.int32(1), T.int32(0)),
                    )
                    T.copy(mm2_pad[safe_slot, 0:hidden_size], src)
                    for j in T.Parallel(hidden_size):
                        acc[j] = acc[j] + T.Cast("float32", src[j]) * weight

                # cast (and scale) then store
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
    """MoE token unpermute kernel (cutlass path).

    Scatters padded expert outputs back to original token order with
    weighted reduction using the forward index mapping from moe_permute.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        hidden_size: Hidden dimension H.
        padded_batch_sum: Size of the padded mm2_pad buffer (≥ T*K).
        dtype: Data type of mm2_pad and output (bf16 or fp16).
        config: Optional config dict.
        scaling: Scalar multiplied into the reduced output before the cast/store
            (folds ``routed_scaling_factor``). Defaults to 1.0 (no scaling).

    Example:
        >>> kernel = MoeUnpermuteKernel(num_tokens=4, top_k=2, hidden_size=128, padded_batch_sum=512)
        >>> output = kernel(mm2_pad, fwd_idx, topk_weights)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        padded_batch_sum: int,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
        scaling: float = 1.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.padded_batch_sum = padded_batch_sum
        self.dtype = dtype
        self.numel = num_tokens * top_k

        self._unpermute_fn = _make_unpermute_kernel(
            num_tokens, top_k, hidden_size, padded_batch_sum, self.dtype_str, scaling
        )

        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return {}

    def forward(
        self,
        mm2_pad: torch.Tensor,
        fwd_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run moe_unpermute.

        Args:
            mm2_pad: [padded_batch_sum, H] bf16/fp16 down-proj output (padded layout).
            fwd_idx: [T*K] int32 forward mapping: flat_idx → padded slot.
            topk_weights: [T, K] float32 routing weights.
            out: optional [T, H] output buffer to write into and reuse across
                calls. Allocated internally with ``torch.empty`` if omitted.

        Returns:
            output: [T, H] bf16/fp16 (``out`` if provided).
        """
        assert fwd_idx.dtype == torch.int32
        assert topk_weights.dtype == torch.float32
        assert mm2_pad.is_cuda

        dev = mm2_pad.device
        if out is None:
            output = torch.empty(
                (self.num_tokens, self.hidden_size), dtype=self.dtype, device=dev)
        else:
            if tuple(out.shape) != (self.num_tokens, self.hidden_size):
                raise ValueError(
                    f"out shape must be {(self.num_tokens, self.hidden_size)}, "
                    f"got {tuple(out.shape)}")
            if out.dtype != self.dtype:
                raise ValueError(f"out dtype must be {self.dtype}, got {out.dtype}")
            # The kernel writes ``out`` on mm2_pad's device as a row-major compact
            # tensor; a cross-device or non-contiguous ``out`` would scatter the
            # store to the wrong memory. Reject rather than corrupt silently.
            if out.device != dev:
                raise ValueError(f"out device must be {dev}, got {out.device}")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            # The kernel reads ``mm2_pad`` (gathered per token) while writing
            # ``out`` concurrently, so an ``out`` overlapping ``mm2_pad`` in
            # memory races. Disjoint slices of one workspace buffer are allowed.
            if tensors_overlap(out, mm2_pad):
                raise ValueError("out must not overlap mm2_pad in memory")
            output = out

        fn = self._unpermute_fn()
        fn(mm2_pad, fwd_idx, topk_weights, output)

        return output
