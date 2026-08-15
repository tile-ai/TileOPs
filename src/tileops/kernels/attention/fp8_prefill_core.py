"""Shared native-FP8 math tile for GQA prefill topologies.

Dense, packed-varlen, and paged prefill intentionally keep different loaders:
their address calculation, tail handling, and cache mutation are different.
Once one Q/K/V tile is resident, however, native-FP8 attention must use one
semantic core.  This module owns that core so scale, mask, softcap, online
softmax, accumulator rescaling, and PV cannot drift between topologies.
"""

import tilelang.language as T

from tileops.kernels.online_softmax import (
    LOG2E,
    make_apply_softcap,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

from .prefill_mask import make_bottom_right_attention_mask

__all__ = ["make_native_fp8_prefill_tile_update"]


def make_native_fp8_prefill_tile_update(
    *,
    is_causal: bool,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    accum_dtype: str,
    block_m: int,
    block_n: int,
    dim: int,
):
    """Build the shared QK -> online-softmax -> PV update for one FP8 tile.

    ``score_multiplier`` is the complete dequantization and attention scale
    for this request/head, normally ``q_scale * k_scale * sm_scale``.  V's
    scale is deliberately not applied here: callers apply it once during the
    final output normalization, avoiding one multiply per PV input element.

    ``prob_fp8`` is shared memory in the current correctness schedule.  The
    fixed BN224 fast path replaces that bridge with PR #1873's raw-PTX WGMMA
    register contract; generalizing that layout does not change this semantic
    boundary.
    """
    use_softcap = softcap > 0.0
    initialize_mask = make_bottom_right_attention_mask(
        is_causal,
        window_size_left,
        window_size_right,
        accum_dtype,
        block_m,
        block_n,
    )
    transformed_mask = make_bottom_right_attention_mask(
        is_causal,
        window_size_left,
        window_size_right,
        accum_dtype,
        block_m,
        block_n,
        preserve_valid=True,
    )
    apply_softcap = (
        make_apply_softcap(1.0, softcap, accum_dtype, block_m, block_n)
        if use_softcap
        else None
    )
    online_softmax = make_online_softmax_with_mask_guard(
        LOG2E, accum_dtype, block_m, block_n
    )
    rescale = make_rescale(block_m, dim)

    @T.macro
    def update(
        q_shared,
        k_shared,
        v_shared,
        acc_s,
        prob_fp8,
        acc_o,
        scores_max,
        scores_max_prev,
        scores_scale,
        scores_sum,
        logsum,
        k_idx,
        q_block_idx,
        q_len,
        kv_len,
        position_offset,
        score_multiplier,
    ):
        if use_softcap:
            T.clear(acc_s)
        else:
            initialize_mask(
                acc_s,
                k_idx,
                q_block_idx,
                q_len,
                kv_len,
                position_offset,
            )

        T.gemm(
            q_shared,
            k_shared,
            acc_s,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullRow,
        )
        for i, j in T.Parallel(block_m, block_n):
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum_dtype),
                -T.infinity(accum_dtype),
                acc_s[i, j] * score_multiplier,
            )

        if use_softcap:
            apply_softcap(acc_s)
            transformed_mask(
                acc_s,
                k_idx,
                q_block_idx,
                q_len,
                kv_len,
                position_offset,
            )

        online_softmax(
            acc_s,
            scores_max,
            scores_max_prev,
            scores_scale,
            scores_sum,
            logsum,
        )
        T.copy(acc_s, prob_fp8)
        rescale(acc_o, scores_scale)
        T.gemm(
            prob_fp8,
            v_shared,
            acc_o,
            policy=T.GemmWarpPolicy.FullRow,
        )

    return update
