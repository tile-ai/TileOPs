"""Shared reduction primitives for reduction kernels.

Provides reusable utility functions, constants, and T.macro factories
used across all reduction sub-category kernels (sum, max, softmax,
variance, prefix-scan, etc.).

This module must land before any sub-category kernel PR so that shared
infrastructure is available from the start.
"""

import tilelang.language as T

__all__ = [
    "align_up",
    "DEFAULT_ALIGNMENT",
    "make_reduce_epilogue",
    "make_welford_update",
    "make_softmax_epilogue",
    "make_cumulative_scan",
]

# 256-element alignment (512 bytes for fp16/bf16) required by T.copy()
# shared memory instructions.  Sub-categories may override this default.
DEFAULT_ALIGNMENT: int = 256


def align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*.

    Args:
        n: Value to align.
        alignment: Alignment boundary (must be positive).

    Returns:
        Smallest multiple of *alignment* that is >= *n*.

    Raises:
        ValueError: If *alignment* is not positive.
    """
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((n + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Supported op_kind values for each macro factory
# ---------------------------------------------------------------------------
_REDUCE_KINDS = {"sum", "max", "min"}
_SOFTMAX_KINDS = {"softmax", "log_softmax"}
_SCAN_KINDS = {"sum", "prod"}


def make_reduce_epilogue(op_kind: str):
    """Create a post-reduce processing T.macro.

    The returned macro applies a final element-wise transformation to the
    reduced result depending on *op_kind*.

    Supported op_kind values: ``"sum"``, ``"max"``, ``"min"``.

    Args:
        op_kind: The reduction operation kind.

    Returns:
        A ``T.macro`` that performs the post-reduce epilogue step.
        The macro signature is ``epilogue(result, output)`` where both
        are 1-D fragments of the same shape.

    Raises:
        ValueError: If *op_kind* is not supported.
    """
    if op_kind not in _REDUCE_KINDS:
        raise ValueError(
            f"Unsupported op_kind '{op_kind}' for reduce epilogue. "
            f"Expected one of {sorted(_REDUCE_KINDS)}."
        )

    # All reduce epilogues currently use a simple copy; sub-category PRs
    # will specialize the bodies (e.g. abs for L1 norm, noop for max/min).
    @T.macro
    def epilogue(result, output):
        T.copy(result, output)

    return epilogue


def make_welford_update(block_m: int, N_padded: int):
    """Create a single-pass Welford mean+variance update T.macro.

    Uses a two-phase approach that is safe under ``T.Parallel``:
      1. **Parallel phase**: compute per-row sum and sum-of-squares via
         ``T.reduce_sum`` (hardware-accelerated, no data races).
      2. **Per-row phase**: derive mean and M2 from the aggregated sums
         using the standard Welford combination formula.

    This avoids the race condition inherent in naively updating shared
    accumulators (mean, m2, count) inside a ``T.Parallel`` loop.

    Args:
        block_m: Number of rows per thread block.
        N_padded: Padded hidden dimension (aligned to DEFAULT_ALIGNMENT).

    Returns:
        A ``T.macro`` with signature
        ``welford_update(x, mean, m2, count)`` where:

        - *x*: input fragment ``(block_m, N_padded)`` in fp32.
        - *mean*: running mean ``(block_m,)`` in fp32 (updated in-place).
        - *m2*: running M2 ``(block_m,)`` in fp32 (updated in-place).
        - *count*: running element count ``(block_m,)`` in fp32 (updated).
    """

    @T.macro
    def welford_update(x, mean, m2, count):
        # Phase 1: parallel reduction -- safe because T.reduce_sum handles
        # the intra-row reduction internally without user-level races.
        row_sum = T.alloc_fragment((block_m,), "float32")
        sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
        row_sq_sum = T.alloc_fragment((block_m,), "float32")

        T.reduce_sum(x, row_sum, dim=1)

        # Phase 2: per-row combination (no j dimension, no race).
        new_count = T.alloc_fragment((block_m,), "float32")
        new_mean = T.alloc_fragment((block_m,), "float32")
        for i in T.Parallel(block_m):
            new_count[i] = count[i] + float(N_padded)
            new_mean[i] = (mean[i] * count[i] + row_sum[i]) / new_count[i]

        # Compute sum of squared deviations from new_mean in parallel.
        for i, j in T.Parallel(block_m, N_padded):
            sq_diff[i, j] = (x[i, j] - new_mean[i]) * (x[i, j] - new_mean[i])
        T.reduce_sum(sq_diff, row_sq_sum, dim=1)

        # Combine with existing M2 using parallel Welford merge formula:
        #   M2_combined = M2_a + M2_b + delta^2 * (n_a * n_b / n_combined)
        # Here M2_b = row_sq_sum and delta = new_mean - mean (old).
        for i in T.Parallel(block_m):
            delta = new_mean[i] - mean[i]
            m2[i] = (
                m2[i] + row_sq_sum[i] + delta * delta * (count[i] * float(N_padded) / new_count[i])
            )
            mean[i] = new_mean[i]
            count[i] = new_count[i]

    return welford_update


def make_softmax_epilogue(op_kind: str):
    """Create a softmax family post-processing T.macro.

    The returned macro applies the final normalization step for softmax
    or log-softmax.

    Supported op_kind values: ``"softmax"``, ``"log_softmax"``.

    Args:
        op_kind: The softmax variant.

    Returns:
        A ``T.macro`` with signature
        ``epilogue(row_exp, row_sum, block_rows, block_cols, output)``
        where:

        - *row_exp*: exponentiated scores ``(block_rows, block_cols)``.
        - *row_sum*: per-row sums ``(block_rows,)`` in fp32.
        - *block_rows*: number of rows (compile-time constant).
        - *block_cols*: number of columns (compile-time constant).
        - *output*: destination fragment ``(block_rows, block_cols)``.

    Raises:
        ValueError: If *op_kind* is not supported.
    """
    if op_kind not in _SOFTMAX_KINDS:
        raise ValueError(
            f"Unsupported op_kind '{op_kind}' for softmax epilogue. "
            f"Expected one of {sorted(_SOFTMAX_KINDS)}."
        )

    if op_kind == "softmax":

        @T.macro
        def epilogue(row_exp, row_sum, block_rows, block_cols, output):
            """Normalize exponentials: output[i,j] = row_exp[i,j] / row_sum[i]."""
            for i, j in T.Parallel(block_rows, block_cols):
                output[i, j] = row_exp[i, j] / row_sum[i]

    else:  # log_softmax

        @T.macro
        def epilogue(row_exp, row_sum, block_rows, block_cols, output):
            """Log-normalize: output[i,j] = log(row_exp[i,j] / row_sum[i])."""
            for i, j in T.Parallel(block_rows, block_cols):
                output[i, j] = T.log(row_exp[i, j] / row_sum[i])

    return epilogue


def make_cumulative_scan(op_kind: str):
    """Create an inclusive prefix scan T.macro.

    The returned macro performs an inclusive scan (prefix sum or prefix
    product) along the last dimension using a sequential loop
    (``T.Serial``) to maintain the correct data dependency chain.

    Supported op_kind values: ``"sum"``, ``"prod"``.

    Args:
        op_kind: The scan operation kind.

    Returns:
        A ``T.macro`` with signature
        ``scan(input_buf, block_rows, block_cols, output_buf)`` where:

        - *input_buf*: source fragment ``(block_rows, block_cols)``.
        - *block_rows*: number of rows (compile-time constant).
        - *block_cols*: number of columns (compile-time constant).
        - *output_buf*: destination fragment ``(block_rows, block_cols)``.

    Raises:
        ValueError: If *op_kind* is not supported.
    """
    if op_kind not in _SCAN_KINDS:
        raise ValueError(
            f"Unsupported op_kind '{op_kind}' for cumulative scan. "
            f"Expected one of {sorted(_SCAN_KINDS)}."
        )

    if op_kind == "sum":

        @T.macro
        def scan(input_buf, block_rows, block_cols, output_buf):
            """Inclusive prefix sum along the last dimension."""
            # First column is copied as-is.
            for i in T.Parallel(block_rows):
                output_buf[i, 0] = input_buf[i, 0]
            # Sequential scan across columns to maintain dependency.
            for j in T.Serial(1, block_cols):
                for i in T.Parallel(block_rows):
                    output_buf[i, j] = output_buf[i, j - 1] + input_buf[i, j]

    else:  # prod

        @T.macro
        def scan(input_buf, block_rows, block_cols, output_buf):
            """Inclusive prefix product along the last dimension."""
            for i in T.Parallel(block_rows):
                output_buf[i, 0] = input_buf[i, 0]
            for j in T.Serial(1, block_cols):
                for i in T.Parallel(block_rows):
                    output_buf[i, j] = output_buf[i, j - 1] * input_buf[i, j]

    return scan
