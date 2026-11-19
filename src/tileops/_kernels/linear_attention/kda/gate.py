"""FP32 raw channel gates, chunk-local prefixes and causal Gram matrices."""

import tilelang.language as T


def raw_gate_arguments(gate_in_kernel, a_log, dt_bias):
    """The tensors a raw-gate program takes after its other operands; none for a given gate.

    A program built without ``dt_bias`` never reads that operand, so ``a_log`` fills its slot.
    """
    if not gate_in_kernel:
        return ()
    return a_log, a_log if dt_bias is None else dt_bias


def log_decay(raw, log_rate, bias, lower_bound):
    """The channel's log-space decay, with an optional bounded sigmoid gate."""
    shifted = T.cast(raw, "float32") + bias
    rate = T.exp(log_rate)
    if lower_bound is not None:
        # exp(A_log) can overflow FP32, and inf * 0 is NaN: a zero shifted gate takes the
        # sigmoid's value at zero directly instead of through rate * shifted.
        return T.if_then_else(
            shifted == 0.0, lower_bound * 0.5, lower_bound / (1.0 + T.exp(-rate * shifted))
        )
    # Below -20, softplus(x)/exp(x) differs from 1 by less than exp(-20)/2 < 2e-9.
    # Combining exponents preserves a small tail before a large rate scales it.
    # Torch softplus uses the identity above 20.
    return T.if_then_else(
        shifted < -20.0,
        -T.exp(log_rate + shifted),
        -rate * T.if_then_else(shifted > 20.0, shifted, T.log1p(T.exp(shifted))),
    )


def raw_gate_prefix(chunk_size, dim_k, has_bias, lower_bound):
    """An FP32 inclusive prefix whose padded rows contribute zero log decay."""

    @T.macro
    def prefix(g, a_log, dt_bias, destination, base, head, rows):
        for token, channel in T.Parallel(chunk_size, dim_k):
            if token < rows:
                bias = dt_bias[head * dim_k + channel] if has_bias else T.float32(0.0)
                destination[token, channel] = log_decay(
                    g[0, base + token, head, channel], a_log[head], bias, lower_bound
                )
            else:
                destination[token, channel] = 0.0
        T.sync_threads()
        running = T.alloc_fragment([dim_k], "float32")
        T.clear(running)
        if T.get_thread_binding() < dim_k:
            for token in T.serial(chunk_size):
                for channel in T.Parallel(dim_k):
                    running[channel] += destination[token, channel]
                    destination[token, channel] = running[channel]
        T.sync_threads()

    return prefix


def raw_gate_gram(chunk_size, dim_k, load=lambda buffer, i, j: buffer[i, j]):
    """Causal channel-gated Gram matrices without reciprocal decay factors.

    *load* reads one FP32 prefix element; the compact fused program stores it split in two.
    """

    @T.macro
    def gram(q, k, cumulative, kk, qk):
        totals = T.alloc_local([2], "float32")
        for row, column in T.Parallel(chunk_size, chunk_size):
            totals[0] = 0.0
            totals[1] = 0.0
            if row >= column:
                for channel in T.serial(dim_k):
                    decay = T.exp2(
                        load(cumulative, row, channel) - load(cumulative, column, channel)
                    )
                    weighted_key = T.cast(k[column, channel], "float32") * decay
                    totals[0] += T.cast(k[row, channel], "float32") * weighted_key
                    totals[1] += T.cast(q[row, channel], "float32") * weighted_key
            kk[row, column] = totals[0]
            qk[row, column] = totals[1]

    return gram


# Lanes sharing a row, and columns sharing a batch, in `diagonal_row_gram`.
DIAGONAL_LANES = 8
DIAGONAL_COLUMNS = 4


def scaled_query(query, qscale):
    """An FP32 query value times *qscale*, or unchanged when the staged query carries its scale."""
    return query if qscale is None else query * qscale


def exact_fma(a, b, c):
    """``a * b + c`` rounded once to FP32, keeping subnormal values under fast math."""
    # T.ieee_fmaf emits __fmaf_rn, which -ftz flushes; __fmaf_ieee_rn ignores the flag.
    return T.call_extern(  # lint: extern-target allow: no T.* intrinsic ignores -ftz
        "float32", "__fmaf_ieee_rn", a, b, c
    )


@T.macro
def group_sum8(values, part):
    """Sums of eight per-lane partials over the eight lanes of a lane group; lane part ends with entry part."""
    for m in T.unroll(4):
        values[m] = exact_fma(
            1.0,
            T.if_then_else((part >> 2) & 1 == 1, values[m + 4], values[m]),
            T.shfl_xor(T.if_then_else((part >> 2) & 1 == 1, values[m], values[m + 4]), 4),
        )
    for m in T.unroll(2):
        values[m] = exact_fma(
            1.0,
            T.if_then_else((part >> 1) & 1 == 1, values[m + 2], values[m]),
            T.shfl_xor(T.if_then_else((part >> 1) & 1 == 1, values[m], values[m + 2]), 2),
        )
    values[0] = exact_fma(
        1.0,
        T.if_then_else(part & 1 == 1, values[1], values[0]),
        T.shfl_xor(T.if_then_else(part & 1 == 1, values[0], values[1]), 1),
    )


def diagonal_row_gram(dim_k, dtype, qscale=None, unroll_columns=True):
    """One row of a 16-row diagonal block of both causal Gram matrices, per pair in FP32.

    Every pair keeps its own exp2(g_r - g_s): kv * (key * exp2(gv - gk)) with both products
    rounded once and no flushed subnormals. Eight lanes share a row and hold K / 8 channels
    as 4-channel vectors {32 m + 4 part}, so a lane group's reads of a staged row fill
    distinct banks. A lane sums four channels per segment, the segments pairwise and the
    eight lanes along a butterfly. Column batches past the row hold only zeros and are
    skipped. *qscale* multiplies each query value where it is read. Without
    *unroll_columns* the four columns of a batch run in a loop, which keeps the code short
    in a program that holds this pass in a branch it seldom takes.

    Returns ``(buffers, row_gram)``: ``buffers()`` allocates the lane's registers, and
    ``row_gram(q, k, cumulative, base, row, part, regs, store, *store_args)`` forms row
    ``base + row`` of the staged tiles against their rows from *base*, calling
    ``store(column, part, value, *store_args)`` once per four columns: lane *part* holds
    column ``column`` of kk (even *part*) or of qk (odd *part*).
    """
    GROUP = DIAGONAL_LANES
    VEC = 4
    SEGS = dim_k // (GROUP * VEC)
    CH = SEGS * VEC
    BW = DIAGONAL_COLUMNS
    assert SEGS in (2, 4) and GROUP == 8 and 2 * BW == GROUP, (
        "the widths group_sum8 and the segment sums take"
    )

    def buffers():
        return (
            T.alloc_local([VEC], dtype),
            T.alloc_local([VEC], dtype),
            T.alloc_local([CH], "float32"),
            T.alloc_local([CH], "float32"),
            T.alloc_local([CH], "float32"),
            T.alloc_local([CH], "float32"),
            T.alloc_local([CH], "float32"),
            T.alloc_local([2 * BW], "float32"),
            T.alloc_local([2 * SEGS], "float32"),
        )

    @T.macro
    def row_gram(q, k, cumulative, base, row, part, regs, store, *store_args):
        tq, tk, qv, kv, gv, key, gk, acc, sub = regs
        for m in T.unroll(SEGS):
            c0 = m * GROUP * VEC + part * VEC
            for v in T.vectorized(VEC):
                tq[v] = q[base + row, c0 + v]
            for v in T.vectorized(VEC):
                tk[v] = k[base + row, c0 + v]
            for v in T.vectorized(VEC):
                gv[m * VEC + v] = cumulative[base + row, c0 + v]
            for v in T.unroll(VEC):
                qv[m * VEC + v] = scaled_query(T.cast(tq[v], "float32"), qscale)
                kv[m * VEC + v] = T.cast(tk[v], "float32")
        for b in T.serial(row // BW + 1):
            for e in T.unroll(2 * BW):
                acc[e] = 0.0
            for j in T.unroll(BW) if unroll_columns else T.serial(BW):
                for m in T.unroll(SEGS):
                    c0 = m * GROUP * VEC + part * VEC
                    for v in T.vectorized(VEC):
                        tk[v] = k[base + b * BW + j, c0 + v]
                    for v in T.vectorized(VEC):
                        gk[m * VEC + v] = cumulative[base + b * BW + j, c0 + v]
                    for v in T.unroll(VEC):
                        key[m * VEC + v] = T.cast(tk[v], "float32")
                for m in T.unroll(SEGS):
                    sub[m] = 0.0
                    sub[SEGS + m] = 0.0
                    for v in T.unroll(VEC):
                        # key * exp2(g_r - g_s) rounded once, then the two products;
                        # one chain of four channels per segment.
                        i = m * VEC + v
                        gk[i] = exact_fma(key[i], T.exp2(gv[i] - gk[i]), -0.0)
                        sub[m] = exact_fma(kv[i], gk[i], sub[m])
                        sub[SEGS + m] = exact_fma(qv[i], gk[i], sub[SEGS + m])
                # Pairwise over the segments, as the warp butterfly pairs lanes.
                if SEGS == 4:
                    acc[2 * j] = exact_fma(
                        1.0,
                        exact_fma(1.0, sub[0], sub[1]),
                        exact_fma(1.0, sub[2], sub[3]),
                    )
                    acc[2 * j + 1] = exact_fma(
                        1.0,
                        exact_fma(1.0, sub[SEGS], sub[SEGS + 1]),
                        exact_fma(1.0, sub[SEGS + 2], sub[SEGS + 3]),
                    )
                else:
                    acc[2 * j] = exact_fma(1.0, sub[0], sub[1])
                    acc[2 * j + 1] = exact_fma(1.0, sub[SEGS], sub[SEGS + 1])
                acc[2 * j] = T.if_then_else(b * BW + j <= row, acc[2 * j], 0.0)
                acc[2 * j + 1] = T.if_then_else(b * BW + j <= row, acc[2 * j + 1], 0.0)
            group_sum8(acc, part)
            store(b * BW + part // 2, part, acc[0], *store_args)

    return buffers, row_gram
