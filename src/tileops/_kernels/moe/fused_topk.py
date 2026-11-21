"""Fuse mixture-of-experts scoring and top-k routing in one warp per token.

Each lane keeps a strided expert subset in registers, so selection has no
global intermediate or block synchronization. The kernel supports softmax,
sigmoid and sqrt-softplus scores, an optional selection-only correction bias
on the element-wise scores, grouped routing, renormalization, and output
scaling. Ties select the lower expert or group. With caller-given experts it
selects nothing: each lane scores the given experts it holds.
"""

import math

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.moe.call_spec import FusedTopKCall, FusedTopKFwdInterface
from tileops._utils import WARP_LANES

__all__ = ["FusedTopKKernel"]


@target_cache(maxsize=64)
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
    with_expert_ids=False,
):
    """Build a fused TileLang kernel: scoring + top-k, zero __syncthreads().

    Args:
        num_tokens: T — number of tokens.
        num_experts: E — number of experts.
        top_k: K — experts to select per token.
        scoring_func: "softmax", "sigmoid" or "sqrtsoftplus" (compile-time constant).
        with_correction_bias: If True, accept a per-expert bias tensor and add it
            to sigmoid or sqrt-softplus scores for top-k selection (bias does NOT
            affect output weights, which remain the original scores).
        renormalize: If True, divide the K selected weights by their own sum
            inside the kernel (compile-time constant), so the caller needs no
            follow-up reduction over topk_weights.
        num_expert_group: G -- groups of E // G consecutive experts; 1 builds the
            ungrouped kernel.
        topk_group: Groups kept per token when G > 1.
        routed_scaling_factor: Multiplier folded into the weight writeback when not 1.
        with_expert_ids: If True, take each token's ``top_k`` experts from an int32
            ``[T, K]`` tensor and score only them; the bias and grouping are unused.

    Returns:
        JIT factory _func(TOKENS_PER_BLOCK) → callable.

    Raises:
        ValueError: From ``_func``, when ``TOKENS_PER_BLOCK`` is outside
            ``[1, 1024 // 32]``: a block is one warp per token, at most 1024 threads.
    """

    @tilelang.jit(
        out_idx=[],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def _func(TOKENS_PER_BLOCK):
        WARP_SIZE = WARP_LANES
        max_block_threads = 1024  # CUDA's limit on threads per block.
        if not 1 <= TOKENS_PER_BLOCK <= max_block_threads // WARP_SIZE:
            raise ValueError(
                f"TOKENS_PER_BLOCK ({TOKENS_PER_BLOCK}) must lie in "
                f"[1, {max_block_threads // WARP_SIZE}]: a block holds one warp per token"
            )
        ELEMS_PER_THREAD = -(-num_experts // WARP_SIZE)
        LOG_WARP = int(math.log2(WARP_SIZE))
        HALF_WARP = WARP_SIZE // 2
        num_blocks = -(-num_tokens // TOKENS_PER_BLOCK)
        grouped = num_expert_group > 1
        GROUP_SIZE = num_experts // num_expert_group

        @T.macro
        def score_in_place(buf, j):
            """Replace logit ``buf[j]`` with its element-wise score: sigmoid or sqrt-softplus."""
            if scoring_func == "sigmoid":
                buf[j] = T.float32(1) / (T.float32(1) + T.exp(-buf[j]))
            else:
                # softplus(x) = max(x, 0) + log1p(e), e = exp(-|x|) <= 1, never overflows exp.
                # log1p(e) = 2 atanh(t) = 2t * sum s^i / (2i + 1), t = e / (2 + e), s = t^2 <= 1/9:
                # terms past s^6 fall below float32 precision, and it is cheaper than log1pf.
                # For x <= 0 the score is h * sqrt(2t * series / e), h = exp(x / 2): h stays
                # normal down to x = -174, where e and a score taken as sqrt(e) flush at -87.
                h = T.exp(T.float32(-0.5) * T.abs(buf[j]))
                e = h * h
                inv = T.float32(1) / (T.float32(1) + T.float32(0.5) * e)  # 2t = e * inv
                s = T.float32(0.25) * e * inv * e * inv
                series = T.alloc_var(T.float32)
                series = T.float32(1 / 13)
                series = series * s + T.float32(1 / 11)
                series = series * s + T.float32(1 / 9)
                series = series * s + T.float32(1 / 7)
                series = series * s + T.float32(1 / 5)
                series = series * s + T.float32(1 / 3)
                series = series * s + T.float32(1)
                log1p_over_e = series * inv
                positive = buf[j] > T.float32(0)
                buf[j] = T.sqrt(
                    T.if_then_else(positive, buf[j] + e * log1p_over_e, log1p_over_e)
                ) * T.if_then_else(positive, T.float32(1), h)

        def group_of(j, lane_id):
            """The group of slot ``j``'s expert, ``j * 32 + lane_id``."""
            if GROUP_SIZE % WARP_SIZE == 0:
                # The group spans whole warp rows: slot j of every lane is in one group.
                return (j * WARP_SIZE) // GROUP_SIZE
            group = (j * WARP_SIZE + lane_id) // GROUP_SIZE
            if num_experts % WARP_SIZE != 0:
                # A padded slot holds -inf and loses to every expert; the clamp keeps the shift
                # in range.
                return T.min(group, num_expert_group - 1)
            return group

        LOG_GROUPS = int(math.log2(num_expert_group))
        TRANSPOSE = num_expert_group == 1 << LOG_GROUPS and num_expert_group <= WARP_SIZE

        @T.macro
        def order_key(value):
            """The int32 whose signed order is the float32 order of ``value``: negative floats
            flip their magnitude bits. The map is its own inverse."""
            bits = T.reinterpret(value, "int32")
            return bits ^ ((bits >> 31) & T.int32(0x7FFFFFFF))

        @T.macro
        def merge_partials(best1, best2, dst, keep1, keep2, other1, other2):
            """Merge the partner's biased partial into slot ``dst``: the two best of two
            sorted pairs."""
            best2[dst] = T.max(T.min(keep1, other1), T.max(keep2, other2))
            best1[dst] = T.max(keep1, other1)

        @T.macro
        def transpose_step(best1, best2, lane_id, half, mask):
            """Lanes with the ``mask`` bit keep the upper ``half`` of their groups, the others
            the lower; either way the kept partial lands in the lower slots."""
            for g in T.unroll(half):
                upper = (lane_id & mask) != 0
                keep1 = T.alloc_var(T.int32)
                keep1 = T.if_then_else(upper, best1[g + half], best1[g])
                other1 = T.shfl_xor(T.if_then_else(upper, best1[g], best1[g + half]), mask)
                if with_correction_bias:
                    keep2 = T.alloc_var(T.int32)
                    other2 = T.alloc_var(T.int32)
                    keep2 = T.if_then_else(upper, best2[g + half], best2[g])
                    other2 = T.shfl_xor(T.if_then_else(upper, best2[g], best2[g + half]), mask)
                    merge_partials(best1, best2, g, keep1, keep2, other1, other2)
                else:
                    best1[g] = T.max(keep1, other1)

        @T.macro
        def group_score(best1, best2, g):
            """The order key of group ``g``'s score: its best selection score, plus the second
            best with a bias."""
            if with_correction_bias:
                best1[g] = order_key(
                    T.reinterpret(order_key(best1[g]), "float32")
                    + T.reinterpret(order_key(best2[g]), "float32")
                )

        @T.macro
        def keep_top_groups(sel, lane_id):
            """Set ``sel`` to -inf outside the ``topk_group`` best groups.

            Groups rank on the order keys of their scores, as the selection does: a NaN score
            ranks above every number, as in ``torch.topk``, and ties go to the lower group.
            """
            # Per group: the order keys of the best and, with a bias, second-best scores.
            best1 = T.alloc_local([num_expert_group], "int32")
            best2 = T.alloc_local([num_expert_group], "int32")
            for g in T.unroll(num_expert_group):
                best1[g] = order_key(-T.infinity("float32"))
                best2[g] = order_key(-T.infinity("float32"))
            for j in T.unroll(ELEMS_PER_THREAD):
                key = order_key(sel[j])
                for g in T.unroll(num_expert_group):
                    if group_of(j, lane_id) == g:
                        if with_correction_bias:
                            best2[g] = T.max(best2[g], T.min(best1[g], key))
                        best1[g] = T.max(best1[g], key)
            if TRANSPOSE:
                # After the transpose lane l holds group l >> (5 - LOG_GROUPS) in slot 0.
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
                    other1 = T.alloc_var(T.int32)
                    other1 = T.shfl_xor(best1[0], T.int32(HALF_WARP >> LOG_GROUPS) >> i)
                    if with_correction_bias:
                        other2 = T.alloc_var(T.int32)
                        other2 = T.shfl_xor(best2[0], T.int32(HALF_WARP >> LOG_GROUPS) >> i)
                        merge_partials(best1, best2, 0, best1[0], best2[0], other1, other2)
                    else:
                        best1[0] = T.max(best1[0], other1)
                group_score(best1, best2, 0)
                # A group ranks below every group scoring higher, or equal with a lower index.
                for g in T.unroll(num_expert_group):
                    best2[g] = T.shfl_sync(best1[0], g << (LOG_WARP - LOG_GROUPS))
                own_group = lane_id >> (LOG_WARP - LOG_GROUPS)
                rank = T.alloc_var(T.int32)
                rank = T.int32(0)
                for g in T.unroll(num_expert_group):
                    beats = T.if_then_else(g < own_group, best2[g] >= best1[0], best2[g] > best1[0])
                    rank = rank + T.if_then_else(beats, 1, 0)
                # Group g is kept when bit g << (5 - LOG_GROUPS), its first lane, is set.
                kept_lanes = T.alloc_var(T.uint32)
                kept_lanes = T.cast(T.ballot(rank < topk_group), "uint32")
                for j in T.unroll(ELEMS_PER_THREAD):
                    first_lane = group_of(j, lane_id) << (LOG_WARP - LOG_GROUPS)
                    if ((kept_lanes >> first_lane) & T.uint32(1)) == T.uint32(0):
                        sel[j] = -T.infinity("float32")
            else:
                for i in T.unroll(LOG_WARP):
                    for g in T.unroll(num_expert_group):
                        other1 = T.shfl_xor(best1[g], T.int32(HALF_WARP) >> i)
                        if with_correction_bias:
                            other2 = T.shfl_xor(best2[g], T.int32(HALF_WARP) >> i)
                            # Merge two sorted pairs; reads best1 before it is updated.
                            best2[g] = T.max(T.min(best1[g], other1), T.max(best2[g], other2))
                        best1[g] = T.max(best1[g], other1)
                for g in T.unroll(num_expert_group):
                    group_score(best1, best2, g)
                # No lane owns a group here, so a serial argmax picks the kept groups; the first
                # group not yet kept is taken when no other beats it.
                kept = T.alloc_local([num_expert_group], "int32")
                for g in T.unroll(num_expert_group):
                    kept[g] = 0
                pick = T.alloc_var(T.int32)
                pick_key = T.alloc_var(T.int32)
                for _ in T.unroll(topk_group):
                    pick = T.int32(-1)
                    pick_key = T.min_value("int32")
                    for g in T.unroll(num_expert_group):
                        if kept[g] == 0 and (pick < 0 or best1[g] > pick_key):
                            pick = g
                            pick_key = best1[g]
                    for g in T.unroll(num_expert_group):
                        if g == pick:
                            kept[g] = 1
                for j in T.unroll(ELEMS_PER_THREAD):
                    for g in T.unroll(num_expert_group):
                        if group_of(j, lane_id) == g and kept[g] == 0:
                            sel[j] = -T.infinity("float32")

        # Lane l holds the selected or given experts k = c * 32 + l, c < SLOTS.
        SLOTS = -(-top_k // WARP_SIZE)

        @T.macro
        def select_top_k(sel, lane_id, ids, picked):
            """Select ``top_k`` experts from ``sel`` by order key, best first, ties to the
            lower expert; a NaN ranks first, as in ``torch.topk``. Padded and selected slots
            hold the least int32, below every expert. Each round is two warp reductions: the
            row max, then the lowest expert holding it. Lane ``k % 32`` keeps round ``k``'s
            expert in ``ids`` and its value in ``picked``.
            """
            key = T.alloc_local([ELEMS_PER_THREAD], "int32")
            for j in T.unroll(ELEMS_PER_THREAD):
                key[j] = T.if_then_else(
                    j * WARP_SIZE + lane_id < num_experts,
                    T.max(order_key(sel[j]), T.min_value("int32") + 1),
                    T.min_value("int32"),
                )
            for c in T.unroll(SLOTS):
                for kk in T.serial(T.min(WARP_SIZE, top_k - c * WARP_SIZE)):
                    lane_max = T.alloc_var(T.int32)
                    lane_max = key[0]
                    for j in T.unroll(1, ELEMS_PER_THREAD):
                        lane_max = T.max(lane_max, key[j])
                    best = T.alloc_var(T.int32)
                    best = T.warp_reduce_max(lane_max)
                    first = T.alloc_var(T.int32)
                    first = T.int32(ELEMS_PER_THREAD)
                    for j in T.unroll(ELEMS_PER_THREAD):
                        jj = ELEMS_PER_THREAD - 1 - j
                        first = T.if_then_else(key[jj] == best, jj, first)
                    expert = T.alloc_var(T.int32)
                    expert = T.warp_reduce_min(first * WARP_SIZE + lane_id)
                    cleared = T.alloc_var(T.int32)
                    cleared = T.if_then_else(
                        (expert & (WARP_SIZE - 1)) == lane_id, expert >> LOG_WARP, ELEMS_PER_THREAD
                    )
                    for j in T.unroll(ELEMS_PER_THREAD):
                        key[j] = T.if_then_else(cleared == j, T.min_value("int32"), key[j])
                    if lane_id == kk:
                        ids[c] = expert
                        picked[c] = T.reinterpret(order_key(best), "float32")

        @T.macro
        def write_weights(topk_weights, topk_ids, token_id, lane_id, ids, weights, scale):
            """Renormalize when asked, scale, and store each lane's experts and weights."""
            factor = T.alloc_var(T.float32)
            factor = scale
            if renormalize:
                total = T.alloc_var(T.float32)
                total = T.float32(0)
                for c in T.unroll(SLOTS):
                    if c * WARP_SIZE + lane_id < top_k:
                        total = total + weights[c]
                # Lanes from top_k on hold no weight, so the butterfly stops at the first power
                # of two covering the lanes that do.
                for i in T.unroll(int(math.ceil(math.log2(min(top_k, WARP_SIZE))))):
                    total = total + T.shfl_xor(total, T.int32(1) << i)
                factor = factor / total
            for c in T.unroll(SLOTS):
                k = c * WARP_SIZE + lane_id
                if k < top_k:
                    topk_weights[token_id, k] = weights[c] * factor
                    topk_ids[token_id, k] = ids[c]

        if with_expert_ids:
            # A given-expert slot past top_k holds -inf, which every score maps to 0.

            @T.prim_func
            def main(
                gating_output: T.Tensor([num_tokens, num_experts], input_dtype),
                expert_ids: T.Tensor([num_tokens, top_k], "int32"),
                topk_weights: T.Tensor([num_tokens, top_k], "float32"),
                topk_ids: T.Tensor([num_tokens, top_k], "int32"),
            ):
                with T.Kernel(num_blocks, threads=TOKENS_PER_BLOCK * WARP_SIZE) as (block_id,):
                    tx = T.get_thread_binding()
                    warp_id = tx // WARP_SIZE
                    lane_id = tx % WARP_SIZE
                    token_id = block_id * TOKENS_PER_BLOCK + warp_id
                    my_weights = T.alloc_local([SLOTS], "float32")
                    if token_id < num_tokens:
                        for c in T.unroll(SLOTS):
                            k = c * WARP_SIZE + lane_id
                            if k < top_k:
                                expert = expert_ids[token_id, k]
                                topk_ids[token_id, k] = expert
                                my_weights[c] = gating_output[token_id, expert]
                            else:
                                my_weights[c] = -T.infinity("float32")

                        scale = T.alloc_var(T.float32)
                        scale = T.float32(routed_scaling_factor)
                        if scoring_func == "softmax":
                            l_max = T.alloc_var(T.float32)
                            l_max = -T.infinity("float32")
                            if renormalize:
                                # The weights renormalize, so shifting by the given logits'
                                # max instead of the row's cancels.
                                for c in T.unroll(SLOTS):
                                    l_max = T.max(l_max, my_weights[c])
                            else:
                                # The weights divide by the sum over the whole row.
                                row = T.alloc_local([ELEMS_PER_THREAD], "float32")
                                for j in T.serial(ELEMS_PER_THREAD):
                                    expert_idx = j * WARP_SIZE + lane_id
                                    if expert_idx < num_experts:
                                        row[j] = gating_output[token_id, expert_idx]
                                    else:
                                        row[j] = -T.infinity("float32")
                                    l_max = T.max(l_max, row[j])
                            for i in T.serial(LOG_WARP):
                                l_max = T.max(l_max, T.shfl_xor(l_max, T.int32(HALF_WARP) >> i))
                            for c in T.unroll(SLOTS):
                                my_weights[c] = T.exp(my_weights[c] - l_max)
                            if not renormalize:
                                l_sum = T.alloc_var(T.float32)
                                l_sum = T.float32(0)
                                for j in T.serial(ELEMS_PER_THREAD):
                                    l_sum = l_sum + T.exp(row[j] - l_max)
                                for i in T.serial(LOG_WARP):
                                    l_sum = l_sum + T.shfl_xor(l_sum, T.int32(HALF_WARP) >> i)
                                scale = scale / l_sum
                        else:
                            for c in T.unroll(SLOTS):
                                score_in_place(my_weights, c)

                        if renormalize:
                            sel_sum = T.alloc_var(T.float32)
                            sel_sum = T.float32(0)
                            for c in T.unroll(SLOTS):
                                sel_sum = sel_sum + my_weights[c]
                            for i in T.serial(LOG_WARP):
                                sel_sum = sel_sum + T.shfl_xor(sel_sum, T.int32(HALF_WARP) >> i)
                            scale = scale / sel_sum
                        for c in T.unroll(SLOTS):
                            k = c * WARP_SIZE + lane_id
                            if k < top_k:
                                topk_weights[token_id, k] = my_weights[c] * scale

        elif with_correction_bias:

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

                    # The biased scores only decide the selection; the weights score the selected
                    # logits again, so the unbiased scores are not kept.
                    my_biased = T.alloc_local([ELEMS_PER_THREAD], "float32")
                    ids = T.alloc_local([SLOTS], "int32")
                    weights = T.alloc_local([SLOTS], "float32")

                    if token_id < num_tokens:
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                my_biased[j] = gating_output[token_id, expert_idx]
                            else:
                                my_biased[j] = -T.infinity("float32")

                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                score_in_place(my_biased, j)
                                my_biased[j] = my_biased[j] + correction_bias[expert_idx]

                        if grouped:
                            keep_top_groups(my_biased, lane_id)

                        select_top_k(my_biased, lane_id, ids, weights)
                        for c in T.unroll(SLOTS):
                            if c * WARP_SIZE + lane_id < top_k:
                                weights[c] = gating_output[token_id, ids[c]]
                                score_in_place(weights, c)
                        write_weights(
                            topk_weights,
                            topk_ids,
                            token_id,
                            lane_id,
                            ids,
                            weights,
                            T.float32(routed_scaling_factor),
                        )

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
                    ids = T.alloc_local([SLOTS], "int32")
                    weights = T.alloc_local([SLOTS], "float32")
                    if token_id < num_tokens:
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                my_scores[j] = gating_output[token_id, expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")

                        scale = T.alloc_var(T.float32)
                        scale = T.float32(routed_scaling_factor)
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
                                scale = scale / l_sum
                            # A NaN or +inf logit, or a row of -inf, leaves a NaN here and makes
                            # the whole softmax row NaN: every expert then ties, as in the
                            # reference, and the lowest are selected.
                            nan_row = T.alloc_var(T.int32)
                            nan_row = T.int32(0)
                            for j in T.serial(ELEMS_PER_THREAD):
                                magnitude = T.reinterpret(my_scores[j], "int32") & T.int32(
                                    0x7FFFFFFF
                                )
                                nan_row = nan_row | T.if_then_else(
                                    magnitude > T.int32(0x7F800000), 1, 0
                                )
                            nan_row = T.warp_reduce_max(nan_row)
                            if nan_row != 0:
                                for j in T.serial(ELEMS_PER_THREAD):
                                    my_scores[j] = T.reinterpret(T.int32(0x7FFFFFFF), "float32")

                        else:  # sigmoid or sqrt-softplus: element-wise, no row reduction
                            for j in T.serial(ELEMS_PER_THREAD):
                                expert_idx = j * WARP_SIZE + lane_id
                                if expert_idx < num_experts:
                                    score_in_place(my_scores, j)
                                # Padding stays -inf, below every score, for the selection.

                        if grouped:
                            # After the row sum: softmax normalizes over every expert.
                            keep_top_groups(my_scores, lane_id)

                        # Unbiased, the selected value is the weight.
                        select_top_k(my_scores, lane_id, ids, weights)
                        write_weights(
                            topk_weights, topk_ids, token_id, lane_id, ids, weights, scale
                        )

        return main

    return _func


class FusedTopKKernel(Kernel, FusedTopKFwdInterface):
    """MoE top-k routing kernel: fused scoring and top-k, one warp per token.

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Number of experts E.
        top_k: Number of experts to select per token K.
        scoring_func: "softmax", "sigmoid" or "sqrtsoftplus".
        renormalize: If True, normalize selected weights to sum to 1.
        with_correction_bias: If True, accept a per-expert correction_bias tensor in
            forward(). Adds bias to the scores for expert selection while writing the
            unbiased scores to topk_weights. Requires sigmoid or sqrt-softplus scores.
        num_expert_group: Groups of E / num_expert_group consecutive experts the
            selection is restricted by; 1 selects over all experts.
        topk_group: Groups kept per token, best group score first.
        routed_scaling_factor: Multiplier on the written weights.
        dtype: Input logits dtype; routing arithmetic remains float32.
        config: Optional kernel config dict (key: "TOKENS_PER_BLOCK").
        with_expert_ids: If True, forward() takes each token's experts and selects nothing.

    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: FusedTopKCall) -> str | None:
        if (
            call.with_correction_bias
            and call.scoring_func == "softmax"
            and not call.with_expert_ids
        ):
            return "adds the correction bias to sigmoid or sqrt-softplus scores, not softmax ones"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: FusedTopKCall) -> Entry:
        """One build per routing shape, scoring and grouping semantics, bias presence, dtype
        and device; caller-given experts build without the bias and grouping they skip."""
        index = call.device.index if call.device is not None else None
        selects = not call.with_expert_ids
        with_correction_bias = call.with_correction_bias and selects
        num_expert_group, topk_group = (
            (call.num_expert_group, call.topk_group) if selects else (1, 1)
        )
        identity = (
            call.num_tokens,
            call.num_experts,
            call.top_k,
            call.scoring_func,
            call.renormalize,
            with_correction_bias,
            num_expert_group,
            topk_group,
            call.routed_scaling_factor,
            call.with_expert_ids,
            call.dtype,
            index,
        )
        return identity, lambda: cls(
            num_tokens=call.num_tokens,
            num_experts=call.num_experts,
            top_k=call.top_k,
            scoring_func=call.scoring_func,
            renormalize=call.renormalize,
            with_correction_bias=with_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            routed_scaling_factor=call.routed_scaling_factor,
            with_expert_ids=call.with_expert_ids,
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
        config: dict | None = None,
        device_index: int | None = None,
        with_expert_ids: bool = False,
    ):
        super().__init__(device_index=device_index)
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
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
        self.with_expert_ids = with_expert_ids
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
            with_expert_ids,
        )
        self.sm_count = self.build_target.sm_count
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Tokens per block that leave the fewest warps on an SM's busiest sub-partition.

        The candidates are the powers of two up to ``min(16, ceil(T / 16))``; a tie takes
        the largest.
        """
        bound = min(16, max(1, math.ceil(self.num_tokens / 16)))
        sub_partitions = 4  # Warp schedulers per SM on every architecture in supported_archs.

        def busiest_sub_partition_warps(tpb: int) -> int:
            blocks_per_sm = math.ceil(math.ceil(self.num_tokens / tpb) / self.sm_count)
            return math.ceil(blocks_per_sm * tpb / sub_partitions)

        candidates = [t for t in (1, 2, 4, 8, 16) if t <= bound]
        tpb = min(candidates, key=lambda t: (busiest_sub_partition_warps(t), -t))
        return {"TOKENS_PER_BLOCK": tpb}

    def forward(
        self,
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run scoring + top-k selection, or score the caller-given experts.

        Args:
            gating_output: [T, E] router logits, any float dtype.
            correction_bias: [E] float32 per-expert bias: required when
                with_correction_bias=True, unread when with_expert_ids=True, None otherwise.
            expert_ids: [T, K] int32 experts of each token, passed exactly when
                with_expert_ids=True.

        Returns:
            topk_weights: [T, K] float32 routing weights.
            topk_ids:     [T, K] int32 expert indices.
        """
        assert gating_output.shape == (self.num_tokens, self.num_experts), (
            f"Expected gating_output shape ({self.num_tokens}, {self.num_experts}), "
            f"got {tuple(gating_output.shape)}"
        )
        self._require_devices(gating_output=gating_output)

        if self.with_expert_ids:
            inputs = (gating_output, expert_ids)
        elif self.with_correction_bias:
            assert correction_bias is not None, (
                "correction_bias must be provided when with_correction_bias=True"
            )
            assert correction_bias.shape == (self.num_experts,), (
                f"Expected correction_bias shape ({self.num_experts},), "
                f"got {tuple(correction_bias.shape)}"
            )
            inputs = (gating_output, correction_bias)
        else:
            assert correction_bias is None, (
                "correction_bias must be None when with_correction_bias=False"
            )
            inputs = (gating_output,)

        dev = gating_output.device
        topk_weights = torch.empty(self.num_tokens, self.top_k, dtype=torch.float32, device=dev)
        topk_ids = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)

        self.program(self._compile)(*inputs, topk_weights, topk_ids)
        return topk_weights, topk_ids

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return self._kernel_fn(cfg["TOKENS_PER_BLOCK"])
