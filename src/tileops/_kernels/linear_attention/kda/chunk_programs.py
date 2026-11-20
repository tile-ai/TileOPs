"""The TileLang programs the chunked Kimi Delta Attention (KDA) forward runs.

``chunk_prepare`` is chunk-local: nothing in it carries state from one chunk to
the next, so one CTA takes one (chunk, value head) and the launch covers the
machine. It produces the WY vectors ``w`` and ``u``, the gated ``q`` and ``k``
the scan multiplies the state by, the intra-chunk attention matrix, and the
chunk's decay. ``chunk_scan`` is what is left: one CTA per (sequence, value
head) walking that sequence's chunks in order. ``chunk_state`` and ``chunk_output``
split that scan: the first walks the chunks carrying only the state, the second
forms every chunk's output in parallel from the states the first stored.

The WY inverse uses stable 16-row diagonal solves and block matrix products.
Given channel gates use reference-centred tiles where `centred_bounds` holds and per-pair
decays past it, except in the referenced form `chunk_prepare_program` describes.
Raw channel gates take per-pair decays; head gates scale the dot product.
"""

import functools
import math
from typing import NamedTuple

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import LOG2E
from tileops._kernels.grouped_tiling import GroupTiling
from tileops._kernels.linear_attention.chunk_plan import fp32_exponent_range
from tileops._kernels.linear_attention.gdn.prefill_common import step_size
from tileops._kernels.linear_attention.kda.gate import (
    DIAGONAL_LANES,
    diagonal_row_gram,
    raw_gate_gram,
    raw_gate_prefix,
)
from tileops._kernels.linear_attention.kda.inverse import inverse_program
from tileops._kernels.state_pool import state_row

__all__ = [
    "chunk_output_program",
    "chunk_prepare_program",
    "chunk_scan_program",
    "chunk_state_program",
]

# Center each channel-gate exponent within 16 rows: its log-gate difference spans at most
# 8 increments rather than a whole chunk, and `centred_bounds` states when that suffices.
# The fused program and the split preparation share it.
REFERENCE_BLOCK = 16
# Added under the square root of each row norm before taking its reciprocal. The
# fused program normalizes the same rows, so it imports this value.
L2_EPS = 1e-6


class CentredBounds(NamedTuple):
    """The terms of the bound on a reference-centred exponent; see `centred_bounds`."""

    top: int
    floor_log2: float
    bits: float
    flush_limit: float
    query_scale_log2: float | None
    normalized_limit: float


@functools.lru_cache(maxsize=16)  # lint: target-cache allow: dtype and scale determine bounds
def centred_bounds(dtype: str, scale: float) -> CentredBounds:
    """Bound the exponents of the reference-centred channel-gate Gram for log decays.

    The centred Gram stores each row ``x`` scaled by ``2^d`` and ``2^-d`` in *dtype*, ``d``
    the channel's log2 prefix distance from its block's reference row, and multiplies the
    copies in FP32. It stays finite and as accurate as rounding while ``|d|`` is within:

    1. ``top - log2|x|``: a row scaled up stays below the dtype's largest power of two.
    2. ``log2|x| - floor_log2 - bits``: a row scaled down loses one unit of its norm at most.
    3. ``flush_limit``: factors exponentiated apart flush below FP32 only within one unit.

    Args:
        dtype: The storage dtype of the centred copies.
        scale: The query scale; zero leaves no query row to scale.

    Returns:
        The terms above and ``normalized_limit``, the bound for normalized rows.
    """
    info, fp32 = torch.finfo(getattr(torch, dtype)), torch.finfo(torch.float32)
    bits = 1 - math.log2(info.eps)
    top = math.floor(math.log2(info.max))
    floor_log2 = math.log2(max(info.smallest_normal * info.eps, fp32.smallest_normal))
    flush_limit = (-math.log2(fp32.smallest_normal) - bits) / 2
    query_scale_log2 = math.log2(abs(scale)) if scale != 0 else None
    norms_log2 = [0.0] + ([query_scale_log2] if query_scale_log2 is not None else [])
    normalized_limit = min(flush_limit, top - max(norms_log2), min(norms_log2) - floor_log2 - bits)
    return CentredBounds(top, floor_log2, bits, flush_limit, query_scale_log2, normalized_limit)


def centred_row_bound(bounds: CentredBounds, prior, sumsq, scale_log2):
    """Tighten *prior* by terms 1 and 2 of `centred_bounds` for one unnormalized row.

    Args:
        sumsq: The row's FP32 sum of squares before scaling; a zero row keeps *prior*.
        scale_log2: The log2 of the row's stored scale, or ``None`` for a zero scale.
    """
    if scale_log2 is None:
        return prior
    norm_log2 = 0.5 * T.log2(sumsq) + scale_log2
    return T.if_then_else(
        sumsq > 0,
        T.min(
            prior,
            T.min(bounds.top - norm_log2, norm_log2 - bounds.floor_log2 - bounds.bits),
        ),
        prior,
    )


@target_cache(maxsize=64)
def chunk_prepare_program(
    heads: int,
    value_heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    scale: float,
    l2norm: bool,
    total_tokens: int,
    num_seqs: int,
    threads: int = 256,
    lean: bool = False,
    *,
    gate_mode: str = "channel",
    beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    gate_in_kernel: bool = False,
    has_dt_bias: bool = False,
    lower_bound: float | None = None,
    referenced: bool = False,
):
    """Build the chunk-local program for one static packed shape.

    *lean* writes the gated query and key, the intra-chunk attention and the decay
    as soon as the Gram matrices exist and holds the causal mask in registers. The
    inverse's two tiles then reuse the shared memory those writes freed.

    Precomputed channel gates form a chunk's Gram matrices on tensor cores from rows
    rescaled against their block's reference row while the chunk stays within
    `centred_bounds`. With *referenced*, in a dtype of FP32's exponent range, a chunk whose
    half range from its first to its last prefix row also stays within them centres every row
    on that midpoint instead, in one product per Gram matrix; a chunk past the bounds forms its
    diagonal blocks from each pair's decay and the blocks below them on tensor cores from rows
    referenced to their block's first row. Otherwise such a chunk forms each entry from its
    pair's decay, as raw gates do.
    """
    BT = chunk_size
    tiling = GroupTiling(num_seqs, chunk_size)
    num_chunks = tiling.tile_upper_bound(total_tokens)
    BC = REFERENCE_BLOCK
    NC = BT // BC
    REF = BC // 2
    H, HV, K, V = heads, value_heads, dim_k, dim_v
    group = HV // H
    accum = "float32"
    gate_shape = [1, total_tokens, HV, K] if gate_mode == "channel" else [1, total_tokens, HV]
    gate_dtype = dtype if gate_mode == "channel" else accum
    centred = gate_mode == "channel" and not gate_in_kernel
    # BF16 has the exponent range to bound every block by flush_limit once both raw
    # norms pass flag_magnitude; other unnormalized calls bound each row.
    fixed_bound = centred and not l2norm and dtype == "bfloat16"
    # Past the bounds, rows referenced to their block's first row are scaled without limit,
    # so only a dtype of FP32's exponent range keeps them where FP32 factors would be.
    assert not referenced or (centred and fp32_exponent_range(dtype)), (
        "the referenced form takes given gates in a dtype of FP32's exponent range"
    )
    single = centred and num_seqs == 1 and total_tokens > 0
    grid_chunks = (total_tokens + BT - 1) // BT if single else num_chunks
    if centred:
        bounds = centred_bounds(dtype, scale)

    @tilelang.jit(
        out_idx=[-6, -5, -4, -3, -2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            "--use_fast_math",
            *(csrc_include("kda_prefill_helper.h") if centred and not single else []),
        ],
    )
    def build():
        prefix = raw_gate_prefix(BT, K, has_dt_bias, lower_bound) if gate_in_kernel else None
        gram = raw_gate_gram(BT, K) if gate_mode == "channel" else None
        invert = inverse_program(
            chunk_size=BT, block_size=BC, threads=threads, dtype=dtype, seeded=referenced
        )
        if referenced:
            # A lane group per two rows of the chunk.
            assert threads == BT // 2 * DIAGONAL_LANES, "the diagonal pass's lane groups"
            # Its staging tile takes the key tile's memory, which the shared-memory plan counts.
            assert BT * K * torch.finfo(getattr(torch, dtype)).bits // 8 >= BT * 2 * BC * 4, (
                "the diagonal staging tile fits in the key tile"
            )
            diagonal_buffers, diagonal_row = diagonal_row_gram(K, dtype, unroll_columns=False)

        @T.macro
        def store_diagonal(column, part, value, staged, base, row):
            staged[base + row, (part % 2) * BC + column] = value

        def reference_row(own, i):
            """The row whose prefix the stored query and key of row *i* still lack: the row
            itself where they stayed ungated (*own*), else its block's reference row."""
            return i if own else (i // BC) * BC + REF

        # Require both norm-dependent terms in centred_row_bound to admit flush_limit.
        # Compare squared norms to avoid log2; tighten each end by one exponent bit
        # to cover norm-reduction rounding. Outliers take the per-pair path.
        @T.macro
        def flag_magnitude(sums, unsafe, scale_log2):
            if scale_log2 is not None:
                low_exp = (
                    math.ceil(
                        2 * (bounds.floor_log2 + bounds.bits + bounds.flush_limit - scale_log2)
                    )
                    + 1
                )
                high_exp = math.floor(2 * (bounds.top - bounds.flush_limit - scale_log2)) - 1
                low = T.infinity(accum) if low_exp >= 128 else T.float32(2.0 ** max(low_exp, -149))
                high = (
                    T.infinity(accum)
                    if high_exp >= 128
                    else T.float32(0.0 if high_exp < -149 else 2.0**high_exp)
                )
                for i in T.Parallel(BT):
                    if sums[i] > 0 and (sums[i] < low or sums[i] >= high):
                        unsafe[0] = 1

        def block_limit(limit, b):
            """The tightest row bound of block *b*, which covers its end rows' exponents."""
            return functools.reduce(T.min, [limit[b * BC + r] for r in range(BC)])

        def gated_query(qa_s, gc_s, own, i, j):
            return T.cast(T.cast(qa_s[i, j], accum) * T.exp2(gc_s[reference_row(own, i), j]), dtype)

        def gated_key(key, gc_s, own, rows, i, j):
            return T.cast(
                T.cast(key[i, j], accum)
                * T.exp2(gc_s[rows - 1, j] - gc_s[reference_row(own, i), j]),
                dtype,
            )

        @T.macro
        def store_gated(qg, kg, qa_s, key, gc_s, own, rows, bos, ihv):
            for i, j in T.Parallel(BT, K):
                if i < rows:
                    qg[0, bos + i, ihv, j] = gated_query(qa_s, gc_s, own, i, j)
                    kg[0, bos + i, ihv, j] = gated_key(key, gc_s, own, rows, i, j)

        @T.macro
        def scale_keys(b_s, ka_s, gc_s, bt_s, own):
            for i, j in T.Parallel(BT, K):
                b_s[i, j] = T.cast(
                    T.cast(ka_s[i, j], accum) * T.exp2(gc_s[reference_row(own, i), j]) * bt_s[i],
                    dtype,
                )

        @T.macro
        def store_outputs(w, qg, kg, acc, qa_s, ka_s, gc_s, rows, bos, ihv):
            for i, j in T.Parallel(BT, K):
                if i < rows:
                    w[0, bos + i, ihv, j] = T.cast(acc[i, j], dtype)
                    if not (lean or centred):
                        qg[0, bos + i, ihv, j] = gated_query(qa_s, gc_s, True, i, j)
                        kg[0, bos + i, ihv, j] = gated_key(ka_s, gc_s, True, rows, i, j)

        @T.macro
        def fill_causal(mask):
            for i, j in T.Parallel(BT, BT):
                mask[i, j] = T.cast(T.if_then_else(i >= j, 1.0, 0.0), dtype)

        @T.macro
        def store_decay(dec, gc_s, rows, ic, ihv):
            last = T.max(rows - 1, 0)
            for j in T.Parallel(K):
                dec[ic, ihv, j] = T.exp2(gc_s[last, j])
            # The exact grid has no CTA for the upper bound's last decay row; it holds identity.
            if single and total_tokens % BT == 0 and ic == 0:
                for j in T.Parallel(K):
                    dec[num_chunks - 1, ihv, j] = 1.0

        @T.macro
        def store_attention_and_decay(aqk, dec, aq, gc_s, rows, bos, ic, ihv):
            for i, j in T.Parallel(BT, BT):
                if i < rows:
                    aqk[0, bos + i, ihv, j] = T.cast(T.if_then_else(i >= j, aq[i, j], 0.0), dtype)
            store_decay(dec, gc_s, rows, ic, ihv)

        @T.macro
        def store_tail_values(u, v, aqk, dec, aq, gc_s, bt_s, b_s, p_s, acc, rows, bos, ic, ihv):
            """Store a partial chunk's attention and decay, then its WY values ``p_s`` times
            the step-scaled values."""
            if not lean:
                store_attention_and_decay(aqk, dec, aq, gc_s, rows, bos, ic, ihv)
            T.sync_threads()
            for i, j in T.Parallel(BT, V):
                b_s[i, j] = T.cast(
                    T.if_then_else(i < rows, T.cast(v[0, bos + i, ihv, j], accum), 0.0) * bt_s[i],
                    dtype,
                )
            T.sync_threads()
            T.clear(acc)
            T.gemm(p_s, b_s, acc)
            for i, j in T.Parallel(BT, V):
                if i < rows:
                    u[0, bos + i, ihv, j] = T.cast(acc[i, j], dtype)

        @T.macro
        def finish(
            kind,
            qg,
            kg,
            qa_s,
            ka_s,
            b_s,
            gc_s,
            bt_s,
            akk,
            aq,
            tkk,
            rows,
            bos,
            ihv,
            kb_s,
            tqk,
            centre_s=None,
            inverse_norms=None,
        ):
            """Form the Gram matrices of a *kind* of chunk, and what depends on its form.

            ``"chunk"`` forms them from tiles centred on the chunk's midpoint, ``"centred"``
            from tiles centred on each block's reference row, ``"referenced"`` below the
            diagonal blocks from first-row-referenced tiles, ``"pairwise"`` each entry from its
            pair's decay. A centred program also stores the gated rows and scales the keys here,
            so the code after the branch is the same for every kind of chunk but the step size
            and the ``2^centre`` a ``"chunk"`` leaves to the tail.
            """
            if kind == "chunk":
                # One reference for the whole chunk: each channel's midpoint between its first
                # and last prefix rows, so both factors stay within half the chunk's decay. The
                # rows arrive unnormalized and each pass writes tiles it does not read, so no
                # barrier separates its iterations. The keys scaled up carry the step size,
                # which makes ``akk`` the inverse's operand and ``b_s`` the key operand of
                # ``w`` up to each channel's ``2^centre``.
                for i, j in T.Parallel(BT, K):
                    key = T.cast(ka_s[i, j], accum) * (inverse_norms[BT + i] if l2norm else 1.0)
                    up = T.exp2(gc_s[i, j] - centre_s[0, j])
                    down = T.exp2(centre_s[0, j] - gc_s[i, j])
                    kb_s[i, j] = T.cast(key * down, dtype)
                    b_s[i, j] = T.cast(key * up * bt_s[i], dtype)
                    if i < rows:
                        kg[0, bos + i, ihv, j] = T.cast(key * down * centre_s[2, j], dtype)
                T.sync_threads()
                for i, j in T.Parallel(BT, K):
                    query = T.cast(qa_s[i, j], accum) * (
                        inverse_norms[i] * scale if l2norm else 1.0
                    )
                    up = T.exp2(gc_s[i, j] - centre_s[0, j])
                    ka_s[i, j] = T.cast(query * up, dtype)
                    if i < rows:
                        qg[0, bos + i, ihv, j] = T.cast(query * up * centre_s[1, j], dtype)
                T.sync_threads()
                T.clear(akk)
                T.clear(aq)
                T.gemm(b_s, kb_s, akk, transpose_B=True)
                T.gemm(ka_s, kb_s, aq, transpose_B=True)
            elif kind == "centred":
                # Three exponentials serve both Gram matrices and all four
                # reference blocks: every later factor is a rescale of these.
                # Entries above the diagonal grow with the decay and may overflow;
                # every reader selects them away.
                for i, j in T.Parallel(BT, K):
                    kb_s[i, j] = T.cast(
                        T.cast(ka_s[i, j], accum)
                        * T.exp2(gc_s[(i // BC) * BC + REF, j] - gc_s[i, j]),
                        dtype,
                    )
                T.sync_threads()
                for i, j in T.Parallel(BT, K):
                    ka_s[i, j] = T.cast(
                        T.cast(ka_s[i, j], accum)
                        * T.exp2(gc_s[i, j] - gc_s[(i // BC) * BC + REF, j]),
                        dtype,
                    )
                    qa_s[i, j] = T.cast(
                        T.cast(qa_s[i, j], accum)
                        * T.exp2(gc_s[i, j] - gc_s[(i // BC) * BC + REF, j]),
                        dtype,
                    )
                T.sync_threads()

                T.clear(akk)
                T.clear(aq)
                for sub in T.serial(NC):
                    for i, j in T.Parallel(BT, K):
                        b_s[i, j] = T.cast(
                            T.if_then_else(
                                i // BC <= sub,
                                T.cast(kb_s[i, j], accum)
                                * T.exp2(gc_s[sub * BC + REF, j] - gc_s[(i // BC) * BC + REF, j]),
                                0.0,
                            ),
                            dtype,
                        )
                    T.sync_threads()
                    T.clear(tkk)
                    T.clear(tqk)
                    T.gemm(ka_s, b_s, tkk, transpose_B=True)
                    T.gemm(qa_s, b_s, tqk, transpose_B=True)
                    for i, j in T.Parallel(BT, BT):
                        akk[i, j] = T.if_then_else(i // BC == sub, tkk[i, j], akk[i, j])
                        aq[i, j] = T.if_then_else(i // BC == sub, tqk[i, j], aq[i, j])
                    T.sync_threads()
                store_gated(qg, kg, qa_s, kb_s, gc_s, False, rows, bos, ihv)
                scale_keys(b_s, ka_s, gc_s, bt_s, False)
            elif kind == "referenced":
                # The diagonal blocks hold their per-pair entries already. The blocks below
                # them take tensor-core products of rows referenced to the first row of the
                # row's block: for a non-increasing prefix both factors stay within one, so
                # an entry loses only what lies below FP32's range.
                store_gated(qg, kg, qa_s, ka_s, gc_s, True, rows, bos, ihv)
                T.sync_threads()
                for i, j in T.Parallel(BT, K):
                    kb_s[i, j] = T.cast(
                        T.cast(ka_s[i, j], accum) * T.exp2(gc_s[i, j] - gc_s[(i // BC) * BC, j]),
                        dtype,
                    )
                    qa_s[i, j] = T.cast(
                        T.cast(qa_s[i, j], accum) * T.exp2(gc_s[i, j] - gc_s[(i // BC) * BC, j]),
                        dtype,
                    )
                for sub in T.serial(1, NC):
                    for i, j in T.Parallel(BT, K):
                        b_s[i, j] = T.cast(
                            T.if_then_else(
                                i < sub * BC,
                                T.cast(ka_s[i, j], accum) * T.exp2(gc_s[sub * BC, j] - gc_s[i, j]),
                                0.0,
                            ),
                            dtype,
                        )
                    T.sync_threads()
                    T.clear(tkk)
                    T.clear(tqk)
                    T.gemm(kb_s, b_s, tkk, transpose_B=True)
                    T.gemm(qa_s, b_s, tqk, transpose_B=True)
                    for i, j in T.Parallel(BT, BT):
                        akk[i, j] = T.if_then_else(
                            (i // BC == sub) & (j < sub * BC), tkk[i, j], akk[i, j]
                        )
                        aq[i, j] = T.if_then_else(
                            (i // BC == sub) & (j < sub * BC), tqk[i, j], aq[i, j]
                        )
                    T.sync_threads()
                scale_keys(b_s, ka_s, gc_s, bt_s, True)
            elif centred:
                # The per-pair form of `raw_gate_gram` and the stores, kept rolled: these
                # chunks are rare, and the code of a branch not taken still fills the
                # instruction cache.
                T.clear(akk)
                T.clear(aq)
                for c in T.serial(K):
                    for i, j in T.Parallel(BT, BT):
                        if i >= j:
                            weighted = T.cast(ka_s[j, c], accum) * T.exp2(gc_s[i, c] - gc_s[j, c])
                            akk[i, j] += T.cast(ka_s[i, c], accum) * weighted
                            aq[i, j] += T.cast(qa_s[i, c], accum) * weighted
                span = threads // K
                for start in T.serial(0, BT, span):
                    for r, j in T.Parallel(span, K):
                        if start + r < rows:
                            qg[0, bos + start + r, ihv, j] = gated_query(
                                qa_s, gc_s, True, start + r, j
                            )
                            kg[0, bos + start + r, ihv, j] = gated_key(
                                ka_s, gc_s, True, rows, start + r, j
                            )
                        b_s[start + r, j] = T.cast(
                            T.cast(ka_s[start + r, j], accum)
                            * T.exp2(gc_s[start + r, j])
                            * bt_s[start + r],
                            dtype,
                        )
            else:
                if gate_mode == "channel":
                    gram(qa_s, ka_s, gc_s, akk, aq)
                else:
                    # A head gate factors outside the dot product. Apply its
                    # causal difference after GEMM, without FP16 inverse scaling.
                    T.clear(aq)
                    T.gemm(qa_s, ka_s, aq, transpose_B=True)
                    # Mask before exponentiation so future keys cannot produce
                    # growing gate ratios.
                    for i, j in T.Parallel(BT, BT):
                        aq[i, j] = T.if_then_else(
                            i >= j, aq[i, j] * T.exp2(gc_s[i, 0] - gc_s[j, 0]), 0.0
                        )
                    T.sync_threads()
                if lean:
                    store_gated(qg, kg, qa_s, ka_s, gc_s, True, rows, bos, ihv)

        @T.macro
        def prepare(q, k, v, g, beta, cu_seqlens, w, u, qg, kg, aqk, dec, inverse, a_log, dt_bias):
            with T.Kernel(grid_chunks, HV, threads=threads) as (ic, ihv):
                if centred:
                    # The per-pair branch would otherwise lift the register count past
                    # what two CTAs per SM leave each thread.
                    T.annotate_min_blocks_per_sm(2)
                ih = ihv // group
                if not single:
                    tile_cum = T.alloc_shared([num_seqs + 1], "int32")
                    lo = T.alloc_local([1], "int32")
                    hi = T.alloc_local([1], "int32")
                    seq = T.alloc_local([1], "int32")
                    first = T.alloc_local([1], "int32")

                qa_s = T.alloc_shared([BT, K], dtype)
                ka_s = T.alloc_shared([BT, K], dtype)
                if centred:
                    kb_s = T.alloc_shared([BT, K], dtype)
                b_s = T.alloc_shared([BT, K], dtype)
                m_s = T.alloc_shared([BT, BT], dtype)
                p_s = T.alloc_shared([BT, BT], dtype)
                gc_s = T.alloc_shared([BT, K], accum)
                bt_s = T.alloc_shared([BT], accum)
                if lean:
                    mask_s = T.alloc_fragment([BT, BT], dtype)

                akk = T.alloc_fragment([BT, BT], accum)
                aq = T.alloc_fragment([BT, BT], accum)
                tkk = T.alloc_fragment([BT, BT], accum)
                if centred:
                    tqk = T.alloc_fragment([BT, BT], accum)
                    if not l2norm and not fixed_bound:
                        query_limit = T.alloc_fragment([BT], accum)
                        limit = T.alloc_shared([BT], accum)
                    per_pair = T.alloc_local([1], "int32")
                    flag = T.alloc_shared([1], "int32")
                acc = T.alloc_fragment([BT, K], accum)
                rowsum = T.alloc_fragment([BT], accum)
                if referenced and l2norm:
                    inverse_norms = T.alloc_shared([2 * BT], accum)
                if referenced:
                    # Per channel: the chunk's centre, its exponential, and the last row's
                    # exponential relative to it.
                    centre_s = T.alloc_shared([3, K], accum)
                    spread = T.alloc_shared([1], "int32")
                    wide = T.alloc_local([1], "int32")
                    if not fixed_bound:
                        loaded = T.alloc_barrier(threads)

                if single:
                    bos = ic * BT
                    rows = T.min(BT, total_tokens - bos)
                else:
                    tiling.cumsum_offsets(cu_seqlens, tile_cum)
                    live = tile_cum[num_seqs]
                    if centred and ic >= live:
                        for j in T.Parallel(K):
                            dec[ic, ihv, j] = 1.0
                        # Uniform across the CTA: no thread may reach the body barriers.
                        T.call_extern("handle", "tileops::kda_empty_chunk_exit")
                    tiling.decode(T.min(ic, live - 1), tile_cum, lo, hi, seq, first)
                    bos = T.cast(cu_seqlens[seq[0]], "int32") + first[0]
                    rows = T.if_then_else(
                        ic < live, T.min(BT, T.cast(cu_seqlens[seq[0] + 1], "int32") - bos), 0
                    )

                if centred:
                    flag[0] = 0
                if referenced:
                    spread[0] = 0
                if fixed_bound:
                    for i, j in T.Parallel(BT, K):
                        raw_q = T.cast(
                            T.if_then_else(i < rows, q[0, bos + i, ih, j], T.cast(0, dtype)), accum
                        )
                        acc[i, j] = raw_q * raw_q
                        qa_s[i, j] = T.cast(raw_q * scale, dtype)
                if referenced and not fixed_bound and rows == BT:
                    # In flight together while the mask and the step sizes are formed.
                    T.tma_copy(q[0, bos : bos + BT, ih, :], qa_s, barrier=loaded)
                    T.tma_copy(k[0, bos : bos + BT, ih, :], ka_s, barrier=loaded)
                    T.tma_copy(g[0, bos : bos + BT, ihv, :], b_s, barrier=loaded)
                    T.mbarrier_arrive(loaded)
                else:
                    for i, j in T.Parallel(BT, K):
                        if not fixed_bound:
                            qa_s[i, j] = T.if_then_else(
                                i < rows, q[0, bos + i, ih, j], T.cast(0, dtype)
                            )
                        ka_s[i, j] = T.if_then_else(
                            i < rows, k[0, bos + i, ih, j], T.cast(0, dtype)
                        )
                        if gate_mode == "channel" and not gate_in_kernel:
                            b_s[i, j] = T.if_then_else(
                                i < rows, g[0, bos + i, ihv, j], T.cast(0, dtype)
                            )
                if fixed_bound:
                    T.reduce_sum(acc, rowsum, dim=1)
                fill_causal(mask_s if lean else m_s)
                for i in T.Parallel(BT):
                    bt_s[i] = T.if_then_else(
                        i < rows,
                        T.cast(
                            step_size(beta[0, bos + i, ihv], beta_sigmoid, allow_neg_eigval), accum
                        ),
                        T.cast(0, accum),
                    )
                if referenced and not fixed_bound and rows == BT:
                    T.mbarrier_wait_parity(loaded, 0)
                T.sync_threads()
                if fixed_bound:
                    flag_magnitude(rowsum, flag, bounds.query_scale_log2)

                # The chunk-local inclusive log2 prefix remains in FP32.
                if gate_mode == "channel":
                    if gate_in_kernel:
                        prefix(g, a_log, dt_bias, gc_s, bos, ihv, rows)
                        T.sync_threads()
                        T.copy(gc_s, acc)
                    else:
                        T.clear(acc)
                        T.gemm(mask_s if lean else m_s, b_s, acc)
                    for i, j in T.Parallel(BT, K):
                        gc_s[i, j] = acc[i, j] * LOG2E
                elif gate_mode == "head":
                    for i, j in T.Parallel(BT, K):
                        gc_s[i, j] = T.if_then_else(
                            rows > 0,
                            g[0, bos + T.min(i, T.max(rows - 1, 0)), ihv] * LOG2E,
                            0.0,
                        )
                else:
                    T.clear(gc_s)

                if not fixed_bound:
                    for i, j in T.Parallel(BT, K):
                        acc[i, j] = T.cast(qa_s[i, j], accum) * T.cast(qa_s[i, j], accum)
                    T.reduce_sum(acc, rowsum, dim=1)
                    if centred and not l2norm:
                        for i in T.Parallel(BT):
                            query_limit[i] = centred_row_bound(
                                bounds, bounds.flush_limit, rowsum[i], bounds.query_scale_log2
                            )
                    if referenced and l2norm:
                        for i in T.Parallel(BT):
                            inverse_norms[i] = T.rsqrt(rowsum[i] + L2_EPS)
                    else:
                        for i, j in T.Parallel(BT, K):
                            qa_s[i, j] = T.cast(
                                T.cast(qa_s[i, j], accum)
                                * (T.rsqrt(rowsum[i] + L2_EPS) if l2norm else 1.0)
                                * scale,
                                dtype,
                            )
                for i, j in T.Parallel(BT, K):
                    acc[i, j] = T.cast(ka_s[i, j], accum) * T.cast(ka_s[i, j], accum)
                T.reduce_sum(acc, rowsum, dim=1)
                if fixed_bound:
                    flag_magnitude(rowsum, flag, 0.0)
                if centred and not l2norm and not fixed_bound:
                    for i in T.Parallel(BT):
                        limit[i] = centred_row_bound(bounds, query_limit[i], rowsum[i], 0.0)
                if referenced and l2norm:
                    # The inverse norms stay in shared memory for the later scalings of each
                    # kind of chunk: a loop reading the reduced fragment takes its replicated
                    # layout, and one thread in eight. The barrier after the bound tests
                    # publishes them.
                    for i in T.Parallel(BT):
                        inverse_norms[BT + i] = T.rsqrt(rowsum[i] + L2_EPS)
                elif l2norm or not centred:
                    for i, j in T.Parallel(BT, K):
                        ka_s[i, j] = T.cast(
                            T.cast(ka_s[i, j], accum)
                            * (T.rsqrt(rowsum[i] + L2_EPS) if l2norm else 1.0),
                            dtype,
                        )
                if centred:
                    # A block's prefix is monotone for one-signed gates, so |gc_i - gc_REF|
                    # peaks at its first and last rows. The barrier below publishes the flag.
                    for b, j in T.Parallel(NC, K):
                        block_bound = (
                            bounds.flush_limit
                            if fixed_bound
                            else bounds.normalized_limit
                            if l2norm
                            else block_limit(limit, b)
                        )
                        if (T.abs(gc_s[b * BC, j] - gc_s[b * BC + REF, j]) > block_bound) | (
                            T.abs(gc_s[b * BC + REF, j] - gc_s[b * BC + BC - 1, j]) > block_bound
                        ):
                            flag[0] = 1
                if referenced:
                    # The same bound for the chunk's midpoint: half its first-to-last range.
                    # Like the block bound, it takes a monotone prefix (gates of one sign),
                    # whose end rows hold its extremes, now over 64 rows instead of 16.
                    for j in T.Parallel(K):
                        first_row = gc_s[0, j]
                        last_row = gc_s[BT - 1, j]
                        centre = (first_row + last_row) * 0.5
                        centre_s[0, j] = centre
                        centre_s[1, j] = T.exp2(centre)
                        centre_s[2, j] = T.exp2(last_row - centre)
                        if T.abs(first_row - last_row) * 0.5 > (
                            bounds.normalized_limit if l2norm else bounds.flush_limit
                        ):
                            spread[0] = 1
                T.sync_threads()

                operands = (qg, kg, qa_s, ka_s, b_s, gc_s, bt_s, akk, aq, tkk, rows, bos, ihv)
                if centred:
                    # One flag per chunk keeps the branch uniform across the CTA. Each kind
                    # of chunk runs its own straight-line copy up to the inverse; the tiles
                    # the inverse reuses must not stay live across the copies.
                    per_pair[0] = flag[0]
                    if referenced:
                        wide[0] = spread[0]
                    if referenced and l2norm and per_pair[0] + wide[0] != 0:
                        # Every kind but the chunk-centred one reads normalized rows.
                        for i, j in T.Parallel(BT, K):
                            qa_s[i, j] = T.cast(
                                T.cast(qa_s[i, j], accum) * inverse_norms[i] * scale, dtype
                            )
                            ka_s[i, j] = T.cast(
                                T.cast(ka_s[i, j], accum) * inverse_norms[BT + i], dtype
                            )
                        T.sync_threads()
                    if referenced and per_pair[0] != 0:
                        # A chunk past the bound takes per-pair FP32 entries in its diagonal
                        # blocks, staged in a tile whose memory the key tiles reuse after it.
                        dg = T.alloc_shared([BT, 2 * BC], accum)
                        regs = diagonal_buffers()
                        # Lane group i of a block owns its rows i and 15 - i, whose column
                        # batches together cost the same for every group.
                        lanes = T.get_thread_binding() // DIAGONAL_LANES
                        base = (lanes // (BC // 2)) * BC
                        for r in T.serial(2):
                            row = T.if_then_else(
                                r == 0, lanes % (BC // 2), BC - 1 - lanes % (BC // 2)
                            )
                            diagonal_row(
                                qa_s, ka_s, gc_s, base, row, T.get_thread_binding() % DIAGONAL_LANES, regs,
                                store_diagonal, dg, base, row,
                            )  # fmt: skip
                        T.sync_threads()
                        for i, j in T.Parallel(BT, BT):
                            same = (i // BC == j // BC) & (j % BC <= i % BC)
                            akk[i, j] = T.if_then_else(same, dg[i, j % BC], 0.0)
                            aq[i, j] = T.if_then_else(same, dg[i, BC + j % BC], 0.0)
                    if per_pair[0] == 0:
                        if referenced:
                            if wide[0] == 0:
                                if l2norm:
                                    finish("chunk", *operands, kb_s, tqk, centre_s, inverse_norms)
                                else:
                                    finish("chunk", *operands, kb_s, tqk, centre_s)
                            else:
                                finish("centred", *operands, kb_s, tqk)
                        else:
                            finish("centred", *operands, kb_s, tqk)
                    else:
                        finish("referenced" if referenced else "pairwise", *operands, kb_s, tqk)
                else:
                    finish("pairwise", *operands, None, None)
                if lean:
                    store_attention_and_decay(aqk, dec, aq, gc_s, rows, bos, ic, ihv)

                if referenced:
                    # A chunk-centred ``akk`` carries the step size already, and its key
                    # operand of ``w`` lacks each channel's ``2^centre``.
                    chunked = per_pair[0] + wide[0] == 0
                    for i, j in T.Parallel(BT, BT):
                        m_s[i, j] = T.cast(
                            T.if_then_else(
                                i > j, -akk[i, j] * T.if_then_else(chunked, 1.0, bt_s[i]), 0.0
                            ),
                            dtype,
                        )
                    T.sync_threads()
                    invert(m_s, p_s, tkk)
                elif gate_mode == "channel":
                    for i, j in T.Parallel(BT, BT):
                        m_s[i, j] = T.cast(T.if_then_else(i > j, -akk[i, j] * bt_s[i], 0.0), dtype)
                    T.sync_threads()
                    invert(m_s, p_s, tkk)
                else:
                    for i, j in T.Parallel(BT, BT):
                        p_s[i, j] = T.if_then_else(
                            i < rows, inverse[0, bos + i, ihv, j], T.cast(0, dtype)
                        )
                    T.sync_threads()

                if referenced:
                    T.sync_threads()
                    T.clear(acc)
                    T.gemm(p_s, b_s, acc)
                    if not lean and rows == BT:
                        # A whole chunk leaves through tiles the Gram matrices no longer read,
                        # in whole-tile stores.
                        for i, j in T.Parallel(BT, K):
                            ka_s[i, j] = T.cast(
                                acc[i, j] * T.if_then_else(chunked, centre_s[1, j], 1.0), dtype
                            )
                        for i, j in T.Parallel(BT, BT):
                            m_s[i, j] = T.cast(T.if_then_else(i >= j, aq[i, j], 0.0), dtype)
                        for i, j in T.Parallel(BT, V):
                            qa_s[i, j] = T.cast(
                                T.cast(v[0, bos + i, ihv, j], accum) * bt_s[i], dtype
                            )
                        T.sync_threads()
                        T.copy(ka_s, w[0, bos : bos + BT, ihv, :])
                        T.copy(m_s, aqk[0, bos : bos + BT, ihv, :])
                        store_decay(dec, gc_s, rows, ic, ihv)
                        T.clear(acc)
                        T.gemm(p_s, qa_s, acc)
                        for i, j in T.Parallel(BT, V):
                            b_s[i, j] = T.cast(acc[i, j], dtype)
                        T.sync_threads()
                        T.copy(b_s, u[0, bos : bos + BT, ihv, :])
                    else:
                        for i, j in T.Parallel(BT, K):
                            if i < rows:
                                w[0, bos + i, ihv, j] = T.cast(
                                    acc[i, j] * T.if_then_else(chunked, centre_s[1, j], 1.0), dtype
                                )
                        store_tail_values(
                            u, v, aqk, dec, aq, gc_s, bt_s, b_s, p_s, acc, rows, bos, ic, ihv
                        )
                else:
                    if not centred:
                        scale_keys(b_s, ka_s, gc_s, bt_s, True)
                    T.sync_threads()
                    T.clear(acc)
                    T.gemm(p_s, b_s, acc)
                    store_outputs(w, qg, kg, acc, qa_s, ka_s, gc_s, rows, bos, ihv)
                    store_tail_values(
                        u, v, aqk, dec, aq, gc_s, bt_s, b_s, p_s, acc, rows, bos, ic, ihv
                    )

        if gate_mode == "channel" and gate_in_kernel:

            @T.prim_func
            def main(
                q: T.Tensor([1, total_tokens, H, K], dtype),
                k: T.Tensor([1, total_tokens, H, K], dtype),
                v: T.Tensor([1, total_tokens, HV, V], dtype),
                g: T.Tensor(gate_shape, gate_dtype),
                beta: T.Tensor([1, total_tokens, HV], dtype),
                cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
                a_log: T.Tensor([HV], accum),
                dt_bias: T.Tensor([HV * K if has_dt_bias else HV], accum),
                w: T.Tensor([1, total_tokens, HV, K], dtype),
                u: T.Tensor([1, total_tokens, HV, V], dtype),
                qg: T.Tensor([1, total_tokens, HV, K], dtype),
                kg: T.Tensor([1, total_tokens, HV, K], dtype),
                aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
                dec: T.Tensor([num_chunks, HV, K], accum),
            ):
                prepare(q, k, v, g, beta, cu_seqlens, w, u, qg, kg, aqk, dec, None, a_log, dt_bias)
        elif gate_mode == "channel":

            @T.prim_func
            def main(
                q: T.Tensor([1, total_tokens, H, K], dtype),
                k: T.Tensor([1, total_tokens, H, K], dtype),
                v: T.Tensor([1, total_tokens, HV, V], dtype),
                g: T.Tensor(gate_shape, gate_dtype),
                beta: T.Tensor([1, total_tokens, HV], dtype),
                cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
                w: T.Tensor([1, total_tokens, HV, K], dtype),
                u: T.Tensor([1, total_tokens, HV, V], dtype),
                qg: T.Tensor([1, total_tokens, HV, K], dtype),
                kg: T.Tensor([1, total_tokens, HV, K], dtype),
                aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
                dec: T.Tensor([num_chunks, HV, K], accum),
            ):
                prepare(q, k, v, g, beta, cu_seqlens, w, u, qg, kg, aqk, dec, None, None, None)
        else:

            @T.prim_func
            def main(
                q: T.Tensor([1, total_tokens, H, K], dtype),
                k: T.Tensor([1, total_tokens, H, K], dtype),
                v: T.Tensor([1, total_tokens, HV, V], dtype),
                g: T.Tensor(gate_shape, gate_dtype),
                beta: T.Tensor([1, total_tokens, HV], dtype),
                cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
                inverse: T.Tensor([1, total_tokens, HV, BT], dtype),
                w: T.Tensor([1, total_tokens, HV, K], dtype),
                u: T.Tensor([1, total_tokens, HV, V], dtype),
                qg: T.Tensor([1, total_tokens, HV, K], dtype),
                kg: T.Tensor([1, total_tokens, HV, K], dtype),
                aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
                dec: T.Tensor([num_chunks, HV, K], accum),
            ):
                prepare(q, k, v, g, beta, cu_seqlens, w, u, qg, kg, aqk, dec, inverse, None, None)

        return main

    return build()


@target_cache(maxsize=64)
def chunk_scan_program(
    value_heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    total_tokens: int,
    num_seqs: int,
    threads: int = 256,
    value_tile: int | None = None,
    *,
    state_v_first: bool = False,
):
    """Build the sequential chunk scan for one static packed shape.

    The state's value columns update independently, so a CTA may take a *value_tile*
    wide slice of them (all of them by default).
    """
    BT = chunk_size
    tiling = GroupTiling(num_seqs, chunk_size)
    num_chunks = tiling.tile_upper_bound(total_tokens)
    HV, K, V = value_heads, dim_k, dim_v
    BV = V if value_tile is None else value_tile
    value_tiles = V // BV
    accum = "float32"
    state_shape = [num_seqs, HV, V, K] if state_v_first else [num_seqs, HV, K, V]

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.prim_func
        def main(
            w: T.Tensor([1, total_tokens, HV, K], dtype),
            u: T.Tensor([1, total_tokens, HV, V], dtype),
            qg: T.Tensor([1, total_tokens, HV, K], dtype),
            kg: T.Tensor([1, total_tokens, HV, K], dtype),
            aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
            dec: T.Tensor([num_chunks, HV, K], accum),
            h0: T.Tensor(state_shape, accum),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            o: T.Tensor([1, total_tokens, HV, V], dtype),
            ht: T.Tensor(state_shape, accum),
        ):
            with T.Kernel(num_seqs, HV, value_tiles, threads=threads) as (iseq, ihv, iv):
                v0 = iv * BV if value_tiles > 1 else 0
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")

                # The scan walks one sequence, so it needs where that sequence's
                # chunks start on the axis the chunk-local half wrote dec along.
                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                bos = T.cast(cu_seqlens[iseq], "int32")
                length = T.cast(cu_seqlens[iseq + 1], "int32") - bos
                chunk0 = tile_cum[iseq]

                state_s = T.alloc_shared([K, BV], dtype)
                w_s = T.alloc_shared([BT, K], dtype)
                qg_s = T.alloc_shared([BT, K], dtype)
                kg_s = T.alloc_shared([BT, K], dtype)
                a_s = T.alloc_shared([BT, BT], dtype)
                vn_s = T.alloc_shared([BT, BV], dtype)
                state_f = T.alloc_fragment([K, BV], accum)
                vn_f = T.alloc_fragment([BT, BV], accum)
                ws_f = T.alloc_fragment([BT, BV], accum)
                o_f = T.alloc_fragment([BT, BV], accum)
                upd = T.alloc_fragment([K, BV], accum)

                for i, j in T.Parallel(K, BV):
                    if state_v_first:
                        state_f[i, j] = h0[iseq, ihv, v0 + j, i]
                    else:
                        state_f[i, j] = h0[iseq, ihv, i, v0 + j]

                for c in T.serial(T.ceildiv(length, BT)):
                    base = bos + c * BT
                    rows = T.min(BT, length - c * BT)
                    for i, j in T.Parallel(K, BV):
                        state_s[i, j] = T.cast(state_f[i, j], dtype)
                    for i, j in T.Parallel(BT, K):
                        w_s[i, j] = T.if_then_else(
                            i < rows, w[0, base + i, ihv, j], T.cast(0, dtype)
                        )
                        qg_s[i, j] = T.if_then_else(
                            i < rows, qg[0, base + i, ihv, j], T.cast(0, dtype)
                        )
                        kg_s[i, j] = T.if_then_else(
                            i < rows, kg[0, base + i, ihv, j], T.cast(0, dtype)
                        )
                    for i, j in T.Parallel(BT, BT):
                        a_s[i, j] = T.if_then_else(
                            i < rows, aqk[0, base + i, ihv, j], T.cast(0, dtype)
                        )
                    for i, j in T.Parallel(BT, BV):
                        vn_f[i, j] = T.if_then_else(
                            i < rows, T.cast(u[0, base + i, ihv, v0 + j], accum), T.cast(0, accum)
                        )
                    T.sync_threads()

                    T.clear(ws_f)
                    T.gemm(w_s, state_s, ws_f)
                    for i, j in T.Parallel(BT, BV):
                        vn_s[i, j] = T.cast(vn_f[i, j] - ws_f[i, j], dtype)
                    T.clear(o_f)
                    T.sync_threads()
                    T.gemm(qg_s, state_s, o_f)
                    T.gemm(a_s, vn_s, o_f)
                    T.clear(upd)
                    T.gemm(kg_s, vn_s, upd, transpose_A=True)
                    for i, j in T.Parallel(K, BV):
                        state_f[i, j] = state_f[i, j] * dec[chunk0 + c, ihv, i] + upd[i, j]
                    for i, j in T.Parallel(BT, BV):
                        if i < rows:
                            o[0, base + i, ihv, v0 + j] = T.cast(o_f[i, j], dtype)
                    T.sync_threads()

                for i, j in T.Parallel(K, BV):
                    if state_v_first:
                        ht[iseq, ihv, v0 + j, i] = state_f[i, j]
                    else:
                        ht[iseq, ihv, i, v0 + j] = state_f[i, j]

        return main

    return build()


@target_cache(maxsize=64)
def chunk_state_program(
    value_heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    total_tokens: int,
    num_seqs: int,
    value_tile: int,
    threads: int = 128,
    arch: int | None = None,
    with_output: bool = False,
    initial_state: bool = True,
    state_v_first: bool = False,
    pooled: bool = False,
):
    """Build the sequential scan over the chunks, for one static packed shape.

    Per chunk a CTA forms the new values ``vn = u - w S`` and updates
    ``S <- dec * S + kg^T vn``. By default it stores ``vn`` and the state the chunk starts
    from, and ``chunk_output_program`` reads both, one CTA a chunk. *with_output* instead
    forms the chunk's output ``o = qg S + A vn`` in the walk from the same BF16 state and
    new-value tiles, with the same products, so it writes the same ``o`` without the
    round trip of ``h`` and ``vn`` through global memory; the program then takes ``qg``
    and ``aqk`` and returns ``(o, ht)``. A CTA takes a *value_tile* wide slice of the
    state's value columns. Without *initial_state* the walk starts from zero and never reads
    ``h0``, so the caller need not fill it.

    *state_v_first* lays the states out value-major, ``[N, HV, V, K]``. *pooled* reads
    sequence ``n``'s state from slot ``slots[n]`` of a state pool and writes its final state
    back there, so the program returns no state.
    """
    if pooled and not initial_state:
        raise ValueError("a state pool holds the initial states")
    BT = chunk_size
    num_chunks = GroupTiling(num_seqs, chunk_size).tile_upper_bound(total_tokens)
    HV, K, V = value_heads, dim_k, dim_v
    BV = value_tile
    prefetch = arch == 90 and BV < V
    value_tiles = V // BV
    accum = "float32"

    # A pooled program updates the pool in place: it returns the outputs and no state.
    outputs = 1 if with_output else 2

    @tilelang.jit(
        out_idx=list(range(-outputs - (0 if pooled else 1), 0)),
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        # Built here so the jit builder closes over scalars only.
        tiling = GroupTiling(num_seqs, chunk_size)

        @T.macro
        def walk(
            w,
            u,
            qg,
            kg,
            aqk,
            dec,
            h,
            vn,
            o,
            tiles,
            frags,
            bos,
            chunk0,
            ihv,
            v0,
            c,
            rows,
            partial: bool,
        ):
            """One chunk: store its starting state and new values (or its output), update the state.

            A partial chunk's rows past the sequence's end belong to the next one: its
            keys, queries and new values are zeroed so those rows reach no output or state.
            """
            w_s, kg_s, u_s, state_s, vn_s, dec_s, qg_s, a_s = tiles
            state_f, ws_f, upd = frags
            base = bos + c * BT
            if partial:
                for i, j in T.Parallel(BT, K):
                    w_s[i, j] = T.if_then_else(i < rows, w[0, base + i, ihv, j], T.cast(0, dtype))
                    kg_s[i, j] = T.if_then_else(i < rows, kg[0, base + i, ihv, j], T.cast(0, dtype))
                if with_output:
                    for i, j in T.Parallel(BT, K):
                        qg_s[i, j] = T.if_then_else(
                            i < rows, qg[0, base + i, ihv, j], T.cast(0, dtype)
                        )
                    for i, j in T.Parallel(BT, BT):
                        a_s[i, j] = T.if_then_else(
                            i < rows, aqk[0, base + i, ihv, j], T.cast(0, dtype)
                        )
            elif prefetch:
                T.copy(w[0, base : base + BT, ihv, :], w_s)
                T.copy(kg[0, base : base + BT, ihv, :], kg_s)
                if with_output:
                    T.copy(qg[0, base : base + BT, ihv, :], qg_s)
                    T.copy(aqk[0, base : base + BT, ihv, :], a_s)
            else:
                # Single-buffered, the loads go in the order the chunk releases their
                # tiles: on SM90 the pipeline's producer issues them in this order and
                # waits for each tile in turn, so a load queued behind the output's tiles
                # would wait for the whole chunk. The prefetched walk above keeps its own
                # order: with two pipeline stages its tiles load a chunk before their use.
                T.copy(w[0, base : base + BT, ihv, :], w_s)
                T.copy(u[0, base : base + BT, ihv, v0 : v0 + BV], u_s)
                T.copy(kg[0, base : base + BT, ihv, :], kg_s)
                T.copy(dec[chunk0 + c, ihv, :], dec_s)
                if with_output:
                    T.copy(qg[0, base : base + BT, ihv, :], qg_s)
                    T.copy(aqk[0, base : base + BT, ihv, :], a_s)
            if partial or prefetch:
                T.copy(u[0, base : base + BT, ihv, v0 : v0 + BV], u_s)
                T.copy(dec[chunk0 + c, ihv, :], dec_s)
            for i, j in T.Parallel(K, BV):
                state_s[i, j] = T.cast(state_f[i, j], dtype)
            if not with_output:
                for i, j in T.Parallel(K, BV):
                    h[chunk0 + c, ihv, i, v0 + j] = state_s[i, j]
            T.clear(ws_f)
            T.gemm(w_s, state_s, ws_f)
            for i, j in T.Parallel(BT, BV):
                vn_s[i, j] = T.if_then_else(
                    i < rows,
                    T.cast(T.cast(u_s[i, j], accum) - ws_f[i, j], dtype),
                    T.cast(0, dtype),
                )
            if not with_output:
                for i, j in T.Parallel(BT, BV):
                    if i < rows:
                        vn[0, base + i, ihv, v0 + j] = vn_s[i, j]
            T.clear(upd)
            T.gemm(kg_s, vn_s, upd, transpose_A=True)
            # The update goes first: the output products below are off the state's chain.
            for i, j in T.Parallel(K, BV):
                state_f[i, j] = state_f[i, j] * dec_s[i] + upd[i, j]
            if with_output:
                # The products and the BF16 staging of chunk_output_program, over this tile.
                T.clear(ws_f)
                T.gemm(qg_s, state_s, ws_f)
                T.gemm(a_s, vn_s, ws_f)
                T.sync_threads()
                T.copy(ws_f, vn_s)
                T.sync_threads()
                for i, j in T.Parallel(BT, BV):
                    if i < rows:
                        o[0, base + i, ihv, v0 + j] = vn_s[i, j]

        @T.macro
        def scan(w, u, qg, kg, aqk, dec, h0, cu_seqlens, h, vn, o, ht, slots):
            with T.Kernel(num_seqs, HV, value_tiles, threads=threads) as (iseq, ihv, iv):
                v0 = iv * BV
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                bos = T.cast(cu_seqlens[iseq], "int32")
                length = T.cast(cu_seqlens[iseq + 1], "int32") - bos
                chunk0 = tile_cum[iseq]

                state_s = T.alloc_shared([K, BV], dtype)
                w_s = T.alloc_shared([BT, K], dtype)
                kg_s = T.alloc_shared([BT, K], dtype)
                vn_s = T.alloc_shared([BT, BV], dtype)
                state_f = T.alloc_fragment([K, BV], accum)
                u_s = T.alloc_shared([BT, BV], dtype)
                dec_s = T.alloc_shared([K], accum)
                qg_s = T.alloc_shared([BT, K], dtype) if with_output else None
                a_s = T.alloc_shared([BT, BT], dtype) if with_output else None
                if prefetch:
                    w_tail = T.alloc_shared([BT, K], dtype)
                    kg_tail = T.alloc_shared([BT, K], dtype)
                    u_tail = T.alloc_shared([BT, BV], dtype)
                    dec_tail = T.alloc_shared([K], accum)
                    qg_tail = T.alloc_shared([BT, K], dtype) if with_output else None
                    a_tail = T.alloc_shared([BT, BT], dtype) if with_output else None
                ws_f = T.alloc_fragment([BT, BV], accum)
                upd = T.alloc_fragment([K, BV], accum)

                if initial_state:
                    for i, j in T.Parallel(K, BV):
                        if state_v_first:
                            state_f[i, j] = h0[state_row(slots, iseq), ihv, v0 + j, i]
                        else:
                            state_f[i, j] = h0[state_row(slots, iseq), ihv, i, v0 + j]
                else:
                    T.clear(state_f)

                tiles = (w_s, kg_s, u_s, state_s, vn_s, dec_s, qg_s, a_s)
                frags = (state_f, ws_f, upd)
                whole = length // BT
                for c in T.Pipelined(whole, num_stages=2 if prefetch else 1):
                    walk(
                        w,
                        u,
                        qg,
                        kg,
                        aqk,
                        dec,
                        h,
                        vn,
                        o,
                        tiles,
                        frags,
                        bos,
                        chunk0,
                        ihv,
                        v0,
                        c,
                        BT,
                        False,
                    )
                if length % BT != 0:
                    tail_tiles = (
                        (w_tail, kg_tail, u_tail, state_s, vn_s, dec_tail, qg_tail, a_tail)
                        if prefetch
                        else tiles
                    )
                    walk(
                        w,
                        u,
                        qg,
                        kg,
                        aqk,
                        dec,
                        h,
                        vn,
                        o,
                        tail_tiles,
                        frags,
                        bos,
                        chunk0,
                        ihv,
                        v0,
                        whole,
                        length % BT,
                        True,
                    )

                for i, j in T.Parallel(K, BV):
                    if state_v_first:
                        ht[state_row(slots, iseq), ihv, v0 + j, i] = state_f[i, j]
                    else:
                        ht[state_row(slots, iseq), ihv, i, v0 + j] = state_f[i, j]

        state_shape = [HV, V, K] if state_v_first else [HV, K, V]
        if pooled:
            if with_output:

                @T.prim_func
                def main(
                    w: T.Tensor([1, total_tokens, HV, K], dtype),
                    u: T.Tensor([1, total_tokens, HV, V], dtype),
                    qg: T.Tensor([1, total_tokens, HV, K], dtype),
                    kg: T.Tensor([1, total_tokens, HV, K], dtype),
                    aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
                    dec: T.Tensor([num_chunks, HV, K], accum),
                    h0: T.Tensor([T.dynamic("pool"), *state_shape], accum),
                    cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
                    slots: T.Tensor([num_seqs], "int32"),
                    o: T.Tensor([1, total_tokens, HV, V], dtype),
                ):
                    scan(w, u, qg, kg, aqk, dec, h0, cu_seqlens, None, None, o, h0, slots)

            else:

                @T.prim_func
                def main(
                    w: T.Tensor([1, total_tokens, HV, K], dtype),
                    u: T.Tensor([1, total_tokens, HV, V], dtype),
                    kg: T.Tensor([1, total_tokens, HV, K], dtype),
                    dec: T.Tensor([num_chunks, HV, K], accum),
                    h0: T.Tensor([T.dynamic("pool"), *state_shape], accum),
                    cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
                    slots: T.Tensor([num_seqs], "int32"),
                    h: T.Tensor([num_chunks, HV, K, V], dtype),
                    vn: T.Tensor([1, total_tokens, HV, V], dtype),
                ):
                    scan(w, u, None, kg, None, dec, h0, cu_seqlens, h, vn, None, h0, slots)

        elif with_output:

            @T.prim_func
            def main(
                w: T.Tensor([1, total_tokens, HV, K], dtype),
                u: T.Tensor([1, total_tokens, HV, V], dtype),
                qg: T.Tensor([1, total_tokens, HV, K], dtype),
                kg: T.Tensor([1, total_tokens, HV, K], dtype),
                aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
                dec: T.Tensor([num_chunks, HV, K], accum),
                h0: T.Tensor([num_seqs, *state_shape], accum),
                cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
                o: T.Tensor([1, total_tokens, HV, V], dtype),
                ht: T.Tensor([num_seqs, *state_shape], accum),
            ):
                scan(w, u, qg, kg, aqk, dec, h0, cu_seqlens, None, None, o, ht, None)

        else:

            @T.prim_func
            def main(
                w: T.Tensor([1, total_tokens, HV, K], dtype),
                u: T.Tensor([1, total_tokens, HV, V], dtype),
                kg: T.Tensor([1, total_tokens, HV, K], dtype),
                dec: T.Tensor([num_chunks, HV, K], accum),
                h0: T.Tensor([num_seqs, *state_shape], accum),
                cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
                h: T.Tensor([num_chunks, HV, K, V], dtype),
                vn: T.Tensor([1, total_tokens, HV, V], dtype),
                ht: T.Tensor([num_seqs, *state_shape], accum),
            ):
                scan(w, u, None, kg, None, dec, h0, cu_seqlens, h, vn, None, ht, None)

        return main

    return build()


@target_cache(maxsize=64)
def chunk_output_program(
    value_heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    total_tokens: int,
    num_seqs: int,
    threads: int = 128,
):
    """Build the chunk-parallel output ``o = qg h + A vn`` for one static packed shape.

    One CTA takes one (chunk, value head): the gated query against the state the chunk
    starts from, plus the intra-chunk attention against the chunk's new values, both
    written by ``chunk_state_program``.
    """
    BT = chunk_size
    num_chunks = GroupTiling(num_seqs, chunk_size).tile_upper_bound(total_tokens)
    HV, K, V = value_heads, dim_k, dim_v
    accum = "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        # Built here so the jit builder closes over scalars only.
        tiling = GroupTiling(num_seqs, chunk_size)

        @T.prim_func
        def main(
            qg: T.Tensor([1, total_tokens, HV, K], dtype),
            aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
            h: T.Tensor([num_chunks, HV, K, V], dtype),
            vn: T.Tensor([1, total_tokens, HV, V], dtype),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            o: T.Tensor([1, total_tokens, HV, V], dtype),
        ):
            with T.Kernel(num_chunks, HV, threads=threads) as (ic, ihv):
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                qg_s = T.alloc_shared([BT, K], dtype)
                h_s = T.alloc_shared([K, V], dtype)
                a_s = T.alloc_shared([BT, BT], dtype)
                vn_s = T.alloc_shared([BT, V], dtype)
                o_f = T.alloc_fragment([BT, V], accum)

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                # A CTA past the last real chunk decodes the last one and holds no row.
                live = tile_cum[num_seqs]
                tiling.decode(T.min(ic, live - 1), tile_cum, lo, hi, seq, first)
                bos = T.cast(cu_seqlens[seq[0]], "int32") + first[0]
                rows = T.if_then_else(
                    ic < live, T.min(BT, T.cast(cu_seqlens[seq[0] + 1], "int32") - bos), 0
                )
                chunk = T.min(ic, live - 1)

                for i, j in T.Parallel(BT, K):
                    qg_s[i, j] = T.if_then_else(i < rows, qg[0, bos + i, ihv, j], T.cast(0, dtype))
                for i, j in T.Parallel(BT, BT):
                    a_s[i, j] = T.if_then_else(i < rows, aqk[0, bos + i, ihv, j], T.cast(0, dtype))
                for i, j in T.Parallel(BT, V):
                    vn_s[i, j] = T.if_then_else(i < rows, vn[0, bos + i, ihv, j], T.cast(0, dtype))
                T.copy(h[chunk, ihv, :, :], h_s)
                T.clear(o_f)
                T.gemm(qg_s, h_s, o_f)
                T.gemm(a_s, vn_s, o_f)
                # The output leaves through the value tile in 16 B stores rather than
                # scattered from the accumulator fragment.
                T.sync_threads()
                T.copy(o_f, vn_s)
                T.sync_threads()
                for i, j in T.Parallel(BT, V):
                    if i < rows:
                        o[0, bos + i, ihv, j] = vn_s[i, j]

        return main

    return build()
