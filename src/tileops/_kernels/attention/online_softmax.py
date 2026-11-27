"""Online softmax macros shared by the attention kernels."""

import tilelang.language as T

from tileops._kernels.constants import LOG2E

__all__ = [
    "finite_max_floor",
    "make_apply_softcap",
    "make_dense_sink_scale",
    "make_dense_split_sink_merge",
    "make_online_softmax",
    "make_online_softmax_with_mask_guard",
    "make_online_softmax_with_score_scale",
    "make_rescale",
    "make_sink_scale",
]


def finite_max_floor(scale: float) -> float:
    """The finite floor of a running row maximum in raw score units.

    Keep ``m * scale`` finite so a masked score ``s = -inf`` has zero weight
    in ``exp2(s * scale - m * scale)``.
    """
    return -1e38 / max(1.0, scale)


def make_sink_scale(scale, accum_dtype, block_rows, rows_per_head=0, head_group=0):
    """Normalize with raw per-head sinks, guarding empty rows and infinite sinks.

    ``rows_per_head`` describes consecutive query rows of each head; ``head_group``
    describes rows interleaving a query-head group. Both zero means one head per tile.
    The sink is combined after the key reduction in the same stable max/exp frame.
    """

    @T.macro
    def sink_scale(scores_max, logsum, sinks, head, row_scale, row_offset=0):
        for i in T.Parallel(block_rows):
            if head_group:
                sink_head = head + (row_offset + i) % head_group
            else:
                sink_head = head + i // rows_per_head if rows_per_head else head
            sink = sinks[sink_head] * LOG2E
            maximum = T.max(scores_max[i] * scale, sink)
            weight = T.exp2(scores_max[i] * scale - maximum)
            row_scale[i] = T.if_then_else(
                (logsum[i] > 0) & (sink < T.infinity(accum_dtype)),
                weight / (logsum[i] * weight + T.exp2(sink - maximum)),
                0,
            )

    return sink_scale


def make_dense_sink_scale(scale, block_rows, head_of=None):
    """Merge a zero-value sink into a completed online softmax state.

    The returned factor multiplies the ordinary attention output. Merging
    requires the fully reduced row sum, so the sink is counted once per row.
    Keep the row sum separate from the maximum so large logits cannot round it away.
    ``scale`` converts the kernel's running maximum to base-2 logits. Row ``i`` takes the
    sink of head ``head_base + i * head_stride``, or of ``head_of(i, head_base, head_stride)``
    when given.
    """

    @T.macro
    def sink_scale(logsum, scores_max, sinks, factors, head_base, head_stride, valid_rows):
        for i in T.Parallel(block_rows):
            if i < valid_rows:
                sink = (
                    sinks[
                        head_base + i * head_stride
                        if head_of is None
                        else head_of(i, head_base, head_stride)
                    ]
                    * LOG2E
                )
                maximum = T.max(scores_max[i] * scale, sink)
                mass = logsum[i] * T.exp2(scores_max[i] * scale - maximum)
                factors[i] = T.if_then_else(
                    (logsum[i] > 0) & (sink < T.infinity("float32")),
                    mass / (mass + T.exp2(sink - maximum)),
                    0.0,
                )
            else:
                factors[i] = 1.0

    return sink_scale


def make_dense_split_sink_merge(accum_dtype):
    """Seed a split softmax denominator with one zero-value sink.

    Partial LSEs and the maximum use base-2 logits; the sink uses natural logits.
    Consumers add their rescaled partial masses and keep the denominator separate
    from the maximum, preserving sink weight when absolute logits are large.
    """

    @T.macro
    def sink_merge(sink, maximum, denominator):
        sink_logit = sink * LOG2E
        maximum[0] = T.max(maximum[0], sink_logit)
        denominator[0] = T.if_then_else(
            sink_logit == T.infinity(accum_dtype),
            1.0,
            T.exp2(sink_logit - maximum[0]),
        )

    return sink_merge


def make_online_softmax(scale, accum_dtype, block_rows, block_cols):
    """Create the online softmax update step as a T.macro.

    The caller casts ``acc_s`` afterwards and rescales ``acc_o``. A kernel where a block may be
    fully masked uses ``make_online_softmax_with_mask_guard`` instead.
    """

    @T.macro
    def online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_rows):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_rows, block_cols):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_rows):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    return online_softmax


def make_online_softmax_with_score_scale(scale, accum_dtype, block_rows, block_cols):
    """Create an online softmax macro that fuses a positive score scale.

    ``scores_max`` stores the max of the actual scaled logits for the most
    recent tile, not the raw QK max. This keeps the rescale factor valid even
    when ``score_scale`` changes between KV tiles.
    """

    @T.macro
    def online_softmax(
        acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum, score_scale
    ):
        score_scale_softmax = score_scale * scale
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_rows):
            scores_max[i] *= score_scale
        for i in T.Parallel(block_rows):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_rows, block_cols):
            acc_s[i, j] = T.exp2(acc_s[i, j] * score_scale_softmax - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_rows):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    return online_softmax


def make_online_softmax_with_mask_guard(scale, accum_dtype, block_rows, block_cols):
    """Create an online softmax T.macro for blocks that may be fully masked.

    The running maximum never decreases, and it is clamped to at least ``-1e38`` so
    ``exp2(-inf - (-inf))`` never produces NaN. The rescale subtracts the two maxima before
    scaling: scaling each side first lets fast math fuse an FMA whose rounding error at
    ``-1e38`` overflows exp2.
    """

    @T.macro
    def online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_rows):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        for i in T.Parallel(block_rows):
            scores_max[i] = T.max(scores_max[i], T.cast(-1e38, accum_dtype))
        for i in T.Parallel(block_rows):
            scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale)
        for i, j in T.Parallel(block_rows, block_cols):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_rows):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    return online_softmax


def make_apply_softcap(score_scale, softcap, accum_dtype, block_rows, block_cols):
    """Create a T.macro that applies score softcap before online softmax.

    The input score fragment contains raw QK scores plus mask entries. Valid
    scores are transformed as ``softcap * tanh((score * score_scale) / softcap)``.
    Masked ``-inf`` entries are preserved so they remain invisible to softmax.
    """

    @T.macro
    def apply_softcap(acc_s):
        for i, j in T.Parallel(block_rows, block_cols):
            capped = T.cast(softcap, accum_dtype) * T.tanh(
                acc_s[i, j] * T.cast(score_scale / softcap, accum_dtype)
            )
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum_dtype),
                -T.infinity(accum_dtype),
                capped,
            )

    return apply_softcap


def make_rescale(block_rows, head_dim):
    """Create a reusable rescale T.macro for the output accumulator.

    Args:
        block_rows: Number of rows in the output accumulator
        head_dim: Dimension of each head

    Returns:
        rescale: A T.macro that rescales acc_o by scores_scale
    """

    @T.macro
    def rescale(acc_o, scores_scale):
        for i, j in T.Parallel(block_rows, head_dim):
            acc_o[i, j] *= scores_scale[i]

    return rescale
