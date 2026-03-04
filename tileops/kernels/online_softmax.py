"""Shared online softmax utilities for attention kernels.

The online softmax algorithm computes softmax incrementally over blocks:
    1. Track running max (scores_max) for numerical stability
    2. Rescale previous accumulator when a new max is found
    3. Compute exp2-scaled scores (using log2(e) trick for ffma fusion)
    4. Update logsum for final normalization
"""

import tilelang.language as T

# log2(e) -- used to convert exp(x) into exp2(x * log2(e))
# which allows the compiler to fuse the multiply into an ffma instruction.
LOG2E = 1.44269504


def make_log2e_scale(dim):
    """Compute softmax scale with log2(e) factor pre-multiplied.

    Returns (1/sqrt(dim)) * log2(e) for use with exp2-based softmax.
    """
    return (1.0 / dim)**0.5 * LOG2E


def make_online_softmax(scale, accum_dtype, block_rows, block_cols):
    """Create a reusable online softmax T.macro.

    The macro computes the online softmax update step:
        1. Save previous scores_max and reset to -inf
        2. Reduce max over current scores
        3. Compute rescaling factor for previous accumulator
        4. Apply exp2 scaling to current scores
        5. Reduce sum and update logsum

    Note: The caller is responsible for casting acc_s afterwards
    (e.g., ``T.copy(acc_s, acc_s_cast)``) and rescaling acc_o.

    Args:
        scale: Pre-computed scale factor (should include LOG2E multiplication)
        accum_dtype: Accumulator data type (e.g., "float")
        block_rows: Number of rows in the score matrix
        block_cols: Number of columns in the score matrix

    Returns:
        online_softmax: A T.macro that performs the online softmax update
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
