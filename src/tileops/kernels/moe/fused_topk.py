"""MoE fused top-k routing kernel: scoring and top-k selection in one pass.

One warp owns one token, and lane l holds experts {l, l+32, l+64, ...} in
registers, so every reduction is an intra-warp ``shfl_xor`` and the kernel
issues no ``__syncthreads()``. Intermediate scores never reach global memory.
Expert slots past E are held at -inf so they lose every argmax.

``renormalize=True`` divides the K selected weights by their own sum inside the
kernel, so the caller needs no second pass over ``topk_weights``. Under softmax
it also makes the row-sum reduction unnecessary, because
``(exp_i/rowsum) / sum_j(exp_j/rowsum)`` equals ``exp_i / sum_j exp_j``.

``with_correction_bias=True`` adds a per-expert bias to the sigmoid scores for
selection only; ``topk_weights`` still carries the original unbiased sigmoid
score. Used by Kimi K2 and DeepSeek-V3-variant models.

Ties in the argmax go to the lower expert index.

Scoring functions: ``softmax`` (Qwen3, Qwen2, Qwen3.5) and ``sigmoid``
(DeepSeek-V3, GLM-4, Kimi K2).

Outputs:
    topk_weights: [T, K] float32 routing weights, renormalized when asked.
    topk_ids: [T, K] int32 expert indices.
"""

import functools
import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import WARP_LANES

__all__ = ["FusedTopKKernel"]

_SCORING_FUNCS = ("softmax", "sigmoid")


@functools.lru_cache(maxsize=64)
def _fused_topk_kernel(
    num_tokens,
    num_experts,
    top_k,
    scoring_func,
    with_correction_bias=False,
    renormalize=False,
):
    """Build a fused TileLang kernel: scoring + top-k, zero __syncthreads().

    Args:
        num_tokens: T — number of tokens.
        num_experts: E — number of experts.
        top_k: K — experts to select per token.
        scoring_func: "softmax" or "sigmoid" (compile-time constant).
        with_correction_bias: If True, accept a per-expert bias tensor and add it
            to sigmoid scores for top-k selection (bias does NOT affect output
            weights, which remain the original sigmoid scores).
        renormalize: If True, divide the K selected weights by their own sum
            inside the kernel (compile-time constant), so the caller needs no
            follow-up reduction over topk_weights.

    Returns:
        JIT factory _func(TOKENS_PER_BLOCK) → callable.
    """

    @tilelang.jit(out_idx=[])
    def _func(TOKENS_PER_BLOCK):
        WARP_SIZE = WARP_LANES
        ELEMS_PER_THREAD = -(-num_experts // WARP_SIZE)  # ceildiv(E, 32)
        LOG_WARP = int(math.log2(WARP_SIZE))  # = 5 for WARP_SIZE=32
        HALF_WARP = WARP_SIZE // 2  # = 16
        num_blocks = -(-num_tokens // TOKENS_PER_BLOCK)  # ceildiv(T, TPB)

        if with_correction_bias:

            @T.prim_func
            def main(
                gating_output: T.Tensor([num_tokens, num_experts], "float32"),
                correction_bias: T.Tensor([num_experts], "float32"),
                topk_weights: T.Tensor([num_tokens, top_k], "float32"),
                topk_ids: T.Tensor([num_tokens, top_k], "int32"),
            ):
                with T.Kernel(num_blocks, threads=TOKENS_PER_BLOCK * WARP_SIZE) as (block_id,):
                    tx = T.get_thread_binding()
                    warp_id = tx // WARP_SIZE
                    lane_id = tx % WARP_SIZE
                    token_id = block_id * TOKENS_PER_BLOCK + warp_id

                    # my_biased only decides selection; my_scores is what gets written.
                    my_scores = T.alloc_local([ELEMS_PER_THREAD], "float32")
                    my_biased = T.alloc_local([ELEMS_PER_THREAD], "float32")

                    if token_id < num_tokens:
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                my_scores[j] = gating_output[token_id, expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")

                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                val = my_scores[j]
                                sig_val = T.float32(1) / (T.float32(1) + T.exp(-val))
                                my_scores[j] = sig_val
                                my_biased[j] = sig_val + correction_bias[expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")
                                my_biased[j] = -T.infinity("float32")

                        l_best_val = T.alloc_var(T.float32)  # biased (for selection)
                        l_best_orig = T.alloc_var(T.float32)  # original sigmoid (for output)
                        l_best_idx = T.alloc_var(T.int32)

                        if renormalize:
                            sel_vals = T.alloc_local([top_k], "float32")
                            sel_sum = T.alloc_var(T.float32)
                            sel_sum = T.float32(0)

                        for k in T.serial(top_k):
                            l_best_val = -T.infinity("float32")
                            l_best_orig = T.float32(0)
                            l_best_idx = T.int32(-1)
                            for j in T.serial(ELEMS_PER_THREAD):
                                if my_biased[j] > l_best_val:
                                    l_best_val = my_biased[j]
                                    l_best_orig = my_scores[j]
                                    l_best_idx = j * T.int32(WARP_SIZE) + lane_id

                            for i in T.serial(LOG_WARP):
                                mask = T.int32(HALF_WARP) >> i
                                other_val = T.shfl_xor(l_best_val, mask)
                                other_orig = T.shfl_xor(l_best_orig, mask)
                                other_idx = T.shfl_xor(l_best_idx, mask)
                                # Must precede l_best_val/l_best_idx: it reads the old pair.
                                l_best_orig = T.if_then_else(
                                    other_val > l_best_val,
                                    other_orig,
                                    T.if_then_else(
                                        other_val == l_best_val,
                                        T.if_then_else(
                                            other_idx < l_best_idx, other_orig, l_best_orig
                                        ),
                                        l_best_orig,
                                    ),
                                )
                                l_best_idx = T.if_then_else(
                                    other_val > l_best_val,
                                    other_idx,
                                    T.if_then_else(
                                        other_val == l_best_val,
                                        T.if_then_else(
                                            other_idx < l_best_idx, other_idx, l_best_idx
                                        ),
                                        l_best_idx,
                                    ),
                                )
                                l_best_val = T.max(l_best_val, other_val)

                            if renormalize:
                                sel_vals[k] = l_best_orig
                                sel_sum = sel_sum + l_best_orig
                                if lane_id == 0:
                                    topk_ids[token_id, k] = l_best_idx
                            else:
                                if lane_id == 0:
                                    topk_weights[token_id, k] = l_best_orig
                                    topk_ids[token_id, k] = l_best_idx

                            for j in T.serial(ELEMS_PER_THREAD):
                                if j * T.int32(WARP_SIZE) + lane_id == l_best_idx:
                                    my_scores[j] = -T.infinity("float32")
                                    my_biased[j] = -T.infinity("float32")

                        if renormalize:
                            inv_sel_sum = T.alloc_var(T.float32)
                            inv_sel_sum = T.float32(1) / sel_sum
                            if lane_id == 0:
                                for k in T.serial(top_k):
                                    topk_weights[token_id, k] = sel_vals[k] * inv_sel_sum

        else:

            @T.prim_func
            def main(
                gating_output: T.Tensor([num_tokens, num_experts], "float32"),
                topk_weights: T.Tensor([num_tokens, top_k], "float32"),
                topk_ids: T.Tensor([num_tokens, top_k], "int32"),
            ):
                with T.Kernel(num_blocks, threads=TOKENS_PER_BLOCK * WARP_SIZE) as (block_id,):
                    tx = T.get_thread_binding()
                    warp_id = tx // WARP_SIZE
                    lane_id = tx % WARP_SIZE
                    token_id = block_id * TOKENS_PER_BLOCK + warp_id
                    my_scores = T.alloc_local([ELEMS_PER_THREAD], "float32")
                    if token_id < num_tokens:
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                my_scores[j] = gating_output[token_id, expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")

                        if not renormalize:
                            inv_row_sum = T.alloc_var(T.float32)
                            inv_row_sum = T.float32(1)  # default (sigmoid / no renorm)

                        if scoring_func == "softmax":
                            l_max = T.alloc_var(T.float32)

                            l_max = -T.infinity("float32")
                            for j in T.serial(ELEMS_PER_THREAD):
                                l_max = T.max(l_max, my_scores[j])
                            for i in T.serial(LOG_WARP):
                                l_max = T.max(l_max, T.shfl_xor(l_max, T.int32(HALF_WARP) >> i))

                            if renormalize:
                                # The row sum cancels against the selected-sum divisor.
                                for j in T.serial(ELEMS_PER_THREAD):
                                    my_scores[j] = T.exp(my_scores[j] - l_max)
                            else:
                                l_sum = T.alloc_var(T.float32)

                                l_sum = T.float32(0)
                                for j in T.serial(ELEMS_PER_THREAD):
                                    val = T.exp(my_scores[j] - l_max)
                                    my_scores[j] = val
                                    l_sum = l_sum + val
                                for i in T.serial(LOG_WARP):
                                    l_sum = l_sum + T.shfl_xor(l_sum, T.int32(HALF_WARP) >> i)
                                inv_row_sum = T.float32(1) / l_sum

                        else:  # sigmoid: element-wise, no row reduction
                            for j in T.serial(ELEMS_PER_THREAD):
                                expert_idx = j * WARP_SIZE + lane_id
                                if expert_idx < num_experts:
                                    val = my_scores[j]
                                    my_scores[j] = T.float32(1) / (T.float32(1) + T.exp(-val))
                                # Padding already -inf; sigmoid(-inf)≈0, keep -inf for argmax.

                        l_best_val = T.alloc_var(T.float32)
                        l_best_idx = T.alloc_var(T.int32)

                        if renormalize:
                            sel_vals = T.alloc_local([top_k], "float32")
                            sel_sum = T.alloc_var(T.float32)
                            sel_sum = T.float32(0)

                        for k in T.serial(top_k):
                            l_best_val = -T.infinity("float32")
                            l_best_idx = T.int32(-1)
                            for j in T.serial(ELEMS_PER_THREAD):
                                if my_scores[j] > l_best_val:
                                    l_best_val = my_scores[j]
                                    l_best_idx = j * T.int32(WARP_SIZE) + lane_id

                            for i in T.serial(LOG_WARP):
                                mask = T.int32(HALF_WARP) >> i
                                other_val = T.shfl_xor(l_best_val, mask)
                                other_idx = T.shfl_xor(l_best_idx, mask)
                                l_best_idx = T.if_then_else(
                                    other_val > l_best_val,
                                    other_idx,
                                    T.if_then_else(
                                        other_val == l_best_val,
                                        T.if_then_else(
                                            other_idx < l_best_idx, other_idx, l_best_idx
                                        ),
                                        l_best_idx,
                                    ),
                                )
                                l_best_val = T.max(l_best_val, other_val)

                            if renormalize:
                                sel_vals[k] = l_best_val
                                sel_sum = sel_sum + l_best_val
                                if lane_id == 0:
                                    topk_ids[token_id, k] = l_best_idx
                            else:
                                if lane_id == 0:
                                    topk_weights[token_id, k] = l_best_val * inv_row_sum
                                    topk_ids[token_id, k] = l_best_idx

                            for j in T.serial(ELEMS_PER_THREAD):
                                if j * T.int32(WARP_SIZE) + lane_id == l_best_idx:
                                    my_scores[j] = -T.infinity("float32")

                        if renormalize:
                            inv_sel_sum = T.alloc_var(T.float32)
                            inv_sel_sum = T.float32(1) / sel_sum
                            if lane_id == 0:
                                for k in T.serial(top_k):
                                    topk_weights[token_id, k] = sel_vals[k] * inv_sel_sum

        return main

    return _func


class FusedTopKKernel(Kernel):
    """MoE top-k routing kernel — fused scoring + top-k, zero __syncthreads().

    Uses a per-warp algorithm: each warp of 32 lanes independently handles one
    token, keeping expert scores in local registers.  All reductions (softmax
    max/sum, argmax) use warp shfl_xor — no shared memory, no __syncthreads__.

    Barrier count: 0 (vs 22 syncs for the old 1-block-per-token design).

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Number of experts E.
        top_k: Number of experts to select per token K.
        scoring_func: "softmax" or "sigmoid".
        renormalize: If True, normalize selected weights to sum to 1. Fused into
            the kernel: the K winners are divided by their own sum before the
            weight writeback, so no extra pass over topk_weights is launched.
        with_correction_bias: If True, accept a per-expert correction_bias tensor in
            forward(). Adds bias to sigmoid scores for expert selection while writing
            unbiased sigmoid scores to topk_weights. Requires scoring_func="sigmoid".
        config: Optional kernel config dict (key: "TOKENS_PER_BLOCK").
        tune: Whether to autotune. ``TOKENS_PER_BLOCK`` is pinned to the
            one-warp-per-token mapping the algorithm relies on, so
            ``autotune_configs`` is undefined and ``tune=True`` degrades to the
            default config with a warning from ``Kernel.init_config``.

    Note:
        ``dtype`` does not apply. Routing arithmetic is fixed at float32 and
        the outputs are float32 weights plus int32 expert ids; ``forward()``
        up-casts ``gating_output`` of any float dtype before the launch, so no
        element type is selected at construction.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        with_correction_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        if scoring_func not in _SCORING_FUNCS:
            raise ValueError(
                f"Unsupported scoring_func '{scoring_func}'. Expected one of {_SCORING_FUNCS}."
            )
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
        if with_correction_bias and scoring_func != "sigmoid":
            raise ValueError(
                "with_correction_bias=True requires scoring_func='sigmoid'. "
                f"Got scoring_func='{scoring_func}'."
            )

        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias

        self._kernel_fn = _fused_topk_kernel(
            num_tokens, num_experts, top_k, scoring_func, with_correction_bias, renormalize
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"TOKENS_PER_BLOCK": 16}

    def forward(
        self,
        gating_output: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run scoring + top-k selection.

        Args:
            gating_output: [T, E] router logits, any float dtype.
            correction_bias: [E] float32 per-expert bias (required when
                with_correction_bias=True, must be None otherwise).

        Returns:
            topk_weights: [T, K] float32 routing weights.
            topk_ids:     [T, K] int32 expert indices.
        """
        assert gating_output.shape == (self.num_tokens, self.num_experts), (
            f"Expected gating_output shape ({self.num_tokens}, {self.num_experts}), "
            f"got {tuple(gating_output.shape)}"
        )
        assert gating_output.is_cuda, "gating_output must be on CUDA"

        if self.with_correction_bias:
            assert correction_bias is not None, (
                "correction_bias must be provided when with_correction_bias=True"
            )
            assert correction_bias.shape == (self.num_experts,), (
                f"Expected correction_bias shape ({self.num_experts},), "
                f"got {tuple(correction_bias.shape)}"
            )
        else:
            assert correction_bias is None, (
                "correction_bias must be None when with_correction_bias=False"
            )

        logits_f32 = gating_output.to(torch.float32)

        dev = logits_f32.device
        topk_weights = torch.empty(self.num_tokens, self.top_k, dtype=torch.float32, device=dev)
        topk_ids = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)

        fn = self._kernel_fn(self.config["TOKENS_PER_BLOCK"])
        if self.with_correction_bias:
            bias_f32 = correction_bias.to(torch.float32)
            fn(logits_f32, bias_f32, topk_weights, topk_ids)
        else:
            fn(logits_f32, topk_weights, topk_ids)

        return topk_weights, topk_ids
