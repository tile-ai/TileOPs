"""MoE token unpermute kernel (cutlass path).

Scatters expert outputs back to original token order, applies routing weights,
and reduces K expert contributions per token.

One block per token (T blocks total):
  - For each of K expert slots: load mm2_out[inv_permuted_idx[i*K+k]] (vectorized 128-bit)
  - Multiply by topk_weights[i, k] (float32)
  - Accumulate into float32 thread-local buffer
  - Cast to output dtype and store to output[i]

Inputs:
  mm2_out          [T*K, H]   bf16/fp16 down-proj output
  inv_permuted_idx [T*K]      int32 inverse mapping from moe_permute
  topk_weights     [T, K]     float32 routing weights

Output:
  output           [T, H]     bf16/fp16
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["MoeUnpermuteKernel"]


def _make_unpermute_kernel(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    dtype: str,
):
    """One block per token. Threads cooperate over H dimension.

    Each thread handles VEC=8 elements (128-bit uint4 load/store).
    Accumulation in float32, cast to dtype on store.
    """
    VEC = 8  # 8 x bf16/fp16 = 128 bits
    threads = min(1024, hidden_size // VEC)
    if threads > 0:
        threads = 1 << (threads.bit_length() - 1)
    threads = max(threads, 1)

    numel = num_tokens * top_k

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _unpermute():

        @T.prim_func
        def _unpermute_main(
            mm2_out: T.Tensor([numel, hidden_size], dtype),          # noqa: F821
            inv_permuted_idx: T.Tensor([numel], "int32"),            # noqa: F821
            topk_weights: T.Tensor([num_tokens, top_k], "float32"),  # noqa: F821
            output: T.Tensor([num_tokens, hidden_size], dtype),      # noqa: F821
        ):
            with T.Kernel(num_tokens, threads=threads) as (token_idx,):
                # float32 accumulator for this token's H slice
                acc = T.alloc_fragment([hidden_size], "float32")
                src = T.alloc_fragment([hidden_size], dtype)

                # zero accumulator
                T.fill(acc, 0.0)

                # accumulate K expert contributions
                for k in T.serial(top_k):
                    flat_idx = token_idx * T.int32(top_k) + k
                    perm_row = inv_permuted_idx[flat_idx]
                    weight = topk_weights[token_idx, k]
                    T.copy(mm2_out[perm_row, 0:hidden_size], src)
                    for j in T.Parallel(hidden_size):
                        acc[j] = acc[j] + T.Cast("float32", src[j]) * weight

                # cast and store
                out_frag = T.alloc_fragment([hidden_size], dtype)
                for j in T.Parallel(hidden_size):
                    out_frag[j] = T.Cast(dtype, acc[j])
                T.copy(out_frag, output[token_idx, 0:hidden_size])

        return _unpermute_main

    return _unpermute


class MoeUnpermuteKernel(Kernel):
    """MoE token unpermute kernel (cutlass path).

    Scatters expert outputs back to original token order with weighted reduction.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        hidden_size: Hidden dimension H.
        dtype: Data type of mm2_out and output (bf16 or fp16).
        config: Optional config dict.

    Example:
        >>> kernel = MoeUnpermuteKernel(num_tokens=4, top_k=2, hidden_size=128)
        >>> output = kernel(mm2_out, inv_permuted_idx, topk_weights)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.numel = num_tokens * top_k

        self._unpermute_fn = _make_unpermute_kernel(
            num_tokens, top_k, hidden_size, self.dtype_str
        )

        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return {}

    def forward(
        self,
        mm2_out: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run moe_unpermute.

        Args:
            mm2_out: [T*K, H] bf16/fp16 down-proj output.
            inv_permuted_idx: [T*K] int32 inverse mapping from moe_permute.
            topk_weights: [T, K] float32 routing weights.

        Returns:
            output: [T, H] bf16/fp16
        """
        assert inv_permuted_idx.dtype == torch.int32
        assert topk_weights.dtype == torch.float32
        assert mm2_out.is_cuda

        dev = mm2_out.device
        output = torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype, device=dev)

        fn = self._unpermute_fn()
        fn(mm2_out, inv_permuted_idx, topk_weights, output)

        return output
