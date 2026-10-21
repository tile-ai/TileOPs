"""FP32 raw channel gates, chunk-local prefixes and causal Gram matrices."""

import tilelang.language as T


def log_decay(raw, log_rate, bias, lower_bound):
    """The channel's log-space decay, with an optional bounded sigmoid gate."""
    shifted = T.cast(raw, "float32") + bias
    rate = T.exp(log_rate)
    if lower_bound is not None:
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


def raw_gate_gram(chunk_size, dim_k):
    """Causal channel-gated Gram matrices without reciprocal decay factors."""

    @T.macro
    def gram(q, k, cumulative, kk, qk):
        totals = T.alloc_local([2], "float32")
        for row, column in T.Parallel(chunk_size, chunk_size):
            totals[0] = 0.0
            totals[1] = 0.0
            if row >= column:
                for channel in T.serial(dim_k):
                    decay = T.exp2(cumulative[row, channel] - cumulative[column, channel])
                    weighted_key = T.cast(k[column, channel], "float32") * decay
                    totals[0] += T.cast(k[row, channel], "float32") * weighted_key
                    totals[1] += T.cast(q[row, channel], "float32") * weighted_key
            kk[row, column] = totals[0]
            qk[row, column] = totals[1]

    return gram
