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

``num_expert_group > 1`` is DeepSeek grouped routing. The E experts split into
groups of ``E // num_expert_group`` consecutive experts; a group scores the sum of
its two best selection scores when a bias is passed, else its best score. Each
lane reduces its own slots per group, then the lanes merge: for a power-of-two group
count as a ``shfl_xor`` transpose, each step halving the groups a lane holds, followed
by one ``shfl_sync`` gather per group; otherwise as a butterfly per group. Every lane
then holds all group scores; the ``topk_group`` best groups are kept by a serial
argmax over those replicated scores, and every slot outside them is set to -inf; the
ungrouped kernel's top-k loop then selects from the rest. ``num_expert_group == 1``
builds the ungrouped kernel, with no grouped code.

``routed_scaling_factor`` other than 1 is folded into the weight writeback.

Ties in the argmax go to the lower expert index, and among groups to the lower
group index.

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

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.moe.call_spec import FusedTopKCall, FusedTopKFwdInterface
from tileops.utils import WARP_LANES

__all__ = ["FusedTopKKernel"]

_SCORING_FUNCS = ("softmax", "sigmoid")


# 64: one build per token count, and a process routes several models' shapes, each with
# and without a bias, over the decode-to-prefill token range.
@functools.lru_cache(maxsize=64)
def _fused_topk_kernel(
    num_tokens,
    num_experts,
    top_k,
    scoring_func,
    input_dtype,
    with_correction_bias=False,
    renormalize=False,
    num_expert_group=1,
    topk_group=1,
    routed_scaling_factor=1.0,
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
        num_expert_group: G -- groups of E // G consecutive experts; 1 builds the
            ungrouped kernel.
        topk_group: Groups kept per token when G > 1.
        routed_scaling_factor: Multiplier folded into the weight writeback when not 1.

    Returns:
        JIT factory _func(TOKENS_PER_BLOCK) → callable.
    """

    @tilelang.jit(
        out_idx=[],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def _func(TOKENS_PER_BLOCK):
        WARP_SIZE = WARP_LANES
        ELEMS_PER_THREAD = -(-num_experts // WARP_SIZE)  # ceildiv(E, 32)
        LOG_WARP = int(math.log2(WARP_SIZE))  # = 5 for WARP_SIZE=32
        HALF_WARP = WARP_SIZE // 2  # = 16
        num_blocks = -(-num_tokens // TOKENS_PER_BLOCK)  # ceildiv(T, TPB)
        grouped = num_expert_group > 1
        scaled = routed_scaling_factor != 1.0
        GROUP_SIZE = num_experts // num_expert_group

        def group_of(j, lane_id):
            """The group of slot ``j``'s expert, ``j * 32 + lane_id``."""
            if GROUP_SIZE % WARP_SIZE == 0:
                # The group spans whole warp rows: slot j of every lane is in one group.
                return (j * WARP_SIZE) // GROUP_SIZE
            return (j * WARP_SIZE + lane_id) // GROUP_SIZE

        # A power-of-two group count reduces the per-group partials as a transpose: each
        # xor step halves the groups a lane holds, so the transpose exchanges G - 1 partials
        # in all instead of G at each of the five steps, and a lane ends with one group's
        # partial; the remaining steps and the gather exchange one each.
        LOG_GROUPS = int(math.log2(num_expert_group))
        TRANSPOSE = num_expert_group == 1 << LOG_GROUPS and num_expert_group <= WARP_SIZE

        @T.macro
        def merge_partials(best1, best2, dst, keep1, keep2, other1, other2):
            """Merge the partner's partial into slot ``dst``: a max, or with a bias the two
            best of two sorted pairs."""
            if with_correction_bias:
                best2[dst] = T.max(T.min(keep1, other1), T.max(keep2, other2))
            best1[dst] = T.max(keep1, other1)

        @T.macro
        def transpose_step(best1, best2, lane_id, half, mask):
            """Lanes with the ``mask`` bit keep the upper ``half`` of their groups, the others
            the lower; either way the kept partial lands in the lower slots."""
            for g in T.unroll(half):
                upper = (lane_id & mask) != 0
                keep1 = T.alloc_var(T.float32)
                keep1 = T.if_then_else(upper, best1[g + half], best1[g])
                other1 = T.shfl_xor(T.if_then_else(upper, best1[g], best1[g + half]), mask)
                keep2 = T.alloc_var(T.float32)
                other2 = T.alloc_var(T.float32)
                if with_correction_bias:
                    keep2 = T.if_then_else(upper, best2[g + half], best2[g])
                    other2 = T.shfl_xor(T.if_then_else(upper, best2[g], best2[g + half]), mask)
                merge_partials(best1, best2, g, keep1, keep2, other1, other2)

        @T.macro
        def keep_top_groups(sel, lane_id):
            """Set ``sel`` to -inf outside the ``topk_group`` best groups."""
            # Per group: the best and, with a bias, second-best selection score.
            best1 = T.alloc_local([num_expert_group], "float32")
            best2 = T.alloc_local([num_expert_group], "float32")
            kept = T.alloc_local([num_expert_group], "int32")
            for g in T.unroll(num_expert_group):
                best1[g] = -T.infinity("float32")
                best2[g] = -T.infinity("float32")
                kept[g] = 0
            for j in T.unroll(ELEMS_PER_THREAD):
                for g in T.unroll(num_expert_group):
                    if group_of(j, lane_id) == g:
                        if with_correction_bias:
                            best2[g] = T.max(best2[g], T.min(best1[g], sel[j]))
                        best1[g] = T.max(best1[g], sel[j])
            if TRANSPOSE:
                # Step s halves the groups held, on lane bit 4 - s; after LOG_GROUPS steps
                # lane l holds group l >> (5 - LOG_GROUPS) in slot 0.
                if LOG_GROUPS >= 1:
                    transpose_step(best1, best2, lane_id, num_expert_group // 2, HALF_WARP)
                if LOG_GROUPS >= 2:
                    transpose_step(best1, best2, lane_id, num_expert_group // 4, HALF_WARP // 2)
                if LOG_GROUPS >= 3:
                    transpose_step(best1, best2, lane_id, num_expert_group // 8, HALF_WARP // 4)
                if LOG_GROUPS >= 4:
                    transpose_step(best1, best2, lane_id, num_expert_group // 16, HALF_WARP // 8)
                if LOG_GROUPS >= 5:
                    transpose_step(best1, best2, lane_id, num_expert_group // 32, HALF_WARP // 16)
                for i in T.unroll(LOG_WARP - LOG_GROUPS):
                    other1 = T.alloc_var(T.float32)
                    other2 = T.alloc_var(T.float32)
                    other1 = T.shfl_xor(best1[0], T.int32(HALF_WARP >> LOG_GROUPS) >> i)
                    if with_correction_bias:
                        other2 = T.shfl_xor(best2[0], T.int32(HALF_WARP >> LOG_GROUPS) >> i)
                    merge_partials(best1, best2, 0, best1[0], best2[0], other1, other2)
                if with_correction_bias:
                    best1[0] = best1[0] + best2[0]
                # Gather every group's score into every lane.
                for g in T.unroll(num_expert_group):
                    best2[g] = T.shfl_sync(best1[0], g << (LOG_WARP - LOG_GROUPS))
                for g in T.unroll(num_expert_group):
                    best1[g] = best2[g]
            else:
                for i in T.unroll(LOG_WARP):
                    for g in T.unroll(num_expert_group):
                        other1 = T.shfl_xor(best1[g], T.int32(HALF_WARP) >> i)
                        if with_correction_bias:
                            other2 = T.shfl_xor(best2[g], T.int32(HALF_WARP) >> i)
                            # Merge two sorted pairs; reads best1 before it is updated.
                            best2[g] = T.max(T.min(best1[g], other1), T.max(best2[g], other2))
                        best1[g] = T.max(best1[g], other1)
                if with_correction_bias:
                    for g in T.unroll(num_expert_group):
                        best1[g] = best1[g] + best2[g]
            # Every lane holds every group score: pick the best groups without shuffles.
            pick = T.alloc_var(T.int32)
            pick_val = T.alloc_var(T.float32)
            for _ in T.unroll(topk_group):
                pick = T.int32(-1)
                pick_val = -T.infinity("float32")
                for g in T.unroll(num_expert_group):
                    if kept[g] == 0 and best1[g] > pick_val:
                        pick = g
                        pick_val = best1[g]
                for g in T.unroll(num_expert_group):
                    if g == pick:
                        kept[g] = 1
            for j in T.unroll(ELEMS_PER_THREAD):
                for g in T.unroll(num_expert_group):
                    if group_of(j, lane_id) == g and kept[g] == 0:
                        sel[j] = -T.infinity("float32")

        if with_correction_bias:

            @T.prim_func
            def main(
                gating_output: T.Tensor([num_tokens, num_experts], input_dtype),
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

                        if grouped:
                            keep_top_groups(my_biased, lane_id)

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
                                    if scaled:
                                        topk_weights[token_id, k] = l_best_orig * T.float32(
                                            routed_scaling_factor
                                        )
                                    else:
                                        topk_weights[token_id, k] = l_best_orig
                                    topk_ids[token_id, k] = l_best_idx

                            for j in T.serial(ELEMS_PER_THREAD):
                                if j * T.int32(WARP_SIZE) + lane_id == l_best_idx:
                                    my_scores[j] = -T.infinity("float32")
                                    my_biased[j] = -T.infinity("float32")

                        if renormalize:
                            inv_sel_sum = T.alloc_var(T.float32)
                            if scaled:
                                inv_sel_sum = T.float32(routed_scaling_factor) / sel_sum
                            else:
                                inv_sel_sum = T.float32(1) / sel_sum
                            if lane_id == 0:
                                for k in T.serial(top_k):
                                    topk_weights[token_id, k] = sel_vals[k] * inv_sel_sum

        else:

            @T.prim_func
            def main(
                gating_output: T.Tensor([num_tokens, num_experts], input_dtype),
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
                            if scaled:
                                inv_row_sum = T.float32(routed_scaling_factor)
                            else:
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
                                if scaled:
                                    inv_row_sum = T.float32(routed_scaling_factor) / l_sum
                                else:
                                    inv_row_sum = T.float32(1) / l_sum

                        else:  # sigmoid: element-wise, no row reduction
                            for j in T.serial(ELEMS_PER_THREAD):
                                expert_idx = j * WARP_SIZE + lane_id
                                if expert_idx < num_experts:
                                    val = my_scores[j]
                                    my_scores[j] = T.float32(1) / (T.float32(1) + T.exp(-val))
                                # Padding already -inf; sigmoid(-inf)≈0, keep -inf for argmax.

                        if grouped:
                            # After the row sum: softmax normalizes over every expert.
                            keep_top_groups(my_scores, lane_id)

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
                            if scaled:
                                inv_sel_sum = T.float32(routed_scaling_factor) / sel_sum
                            else:
                                inv_sel_sum = T.float32(1) / sel_sum
                            if lane_id == 0:
                                for k in T.serial(top_k):
                                    topk_weights[token_id, k] = sel_vals[k] * inv_sel_sum

        return main

    return _func


class FusedTopKKernel(Kernel, FusedTopKFwdInterface):
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
        num_expert_group: Groups of E / num_expert_group consecutive experts the
            selection is restricted by; 1 selects over all experts.
        topk_group: Groups kept per token, best group score first.
        routed_scaling_factor: Multiplier on the written weights.
        dtype: Input logits dtype; routing arithmetic remains float32.
        config: Optional kernel config dict (key: "TOKENS_PER_BLOCK").

    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: FusedTopKCall) -> Entry:
        """One build per routing shape, scoring and grouping semantics, bias presence, dtype
        and device."""
        index = call.device.index if call.device is not None else None
        identity = (
            call.num_tokens,
            call.num_experts,
            call.top_k,
            call.scoring_func,
            call.renormalize,
            call.with_correction_bias,
            call.num_expert_group,
            call.topk_group,
            call.routed_scaling_factor,
            call.dtype,
            index,
        )
        return identity, lambda: cls(
            num_tokens=call.num_tokens,
            num_experts=call.num_experts,
            top_k=call.top_k,
            scoring_func=call.scoring_func,
            renormalize=call.renormalize,
            with_correction_bias=call.with_correction_bias,
            num_expert_group=call.num_expert_group,
            topk_group=call.topk_group,
            routed_scaling_factor=call.routed_scaling_factor,
            dtype=call.dtype,
            device_index=index,
        )

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        with_correction_bias: bool = False,
        num_expert_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
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
        if num_expert_group < 1 or num_experts % num_expert_group != 0:
            raise ValueError(
                f"num_expert_group ({num_expert_group}) must be >= 1 and divide "
                f"num_experts ({num_experts})"
            )
        group_size = num_experts // num_expert_group
        if not 1 <= topk_group <= num_expert_group or top_k > topk_group * group_size:
            raise ValueError(
                f"topk_group ({topk_group}) must lie in [1, {num_expert_group}] and keep "
                f"at least top_k ({top_k}) experts"
            )
        if num_expert_group > 1 and with_correction_bias and group_size < 2:
            raise ValueError("a biased group score sums two experts; each group needs two")

        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.dtype = dtype

        self._kernel_fn = _fused_topk_kernel(
            num_tokens,
            num_experts,
            top_k,
            scoring_func,
            self.dtype_str,
            with_correction_bias,
            renormalize,
            num_expert_group,
            topk_group,
            self.routed_scaling_factor,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"TOKENS_PER_BLOCK": min(16, max(1, math.ceil(self.num_tokens / 16)))}

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

        dev = gating_output.device
        topk_weights = torch.empty(self.num_tokens, self.top_k, dtype=torch.float32, device=dev)
        topk_ids = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)

        fn = self._kernel_fn(self.config["TOKENS_PER_BLOCK"])
        if self.with_correction_bias:
            fn(gating_output, correction_bias, topk_weights, topk_ids)
        else:
            fn(gating_output, topk_weights, topk_ids)

        return topk_weights, topk_ids
