"""Shared reduction primitives for reduction kernels.

Provides reusable utility functions, constants, and T.macro factories
used across all reduction sub-category kernels (sum, max, softmax,
variance, prefix-scan, etc.).

This module must land before any sub-category kernel PR so that shared
infrastructure is available from the start.
"""

import tilelang.language as T

__all__ = [
    "_align_up",
    "DEFAULT_ALIGNMENT",
    "make_reduce_epilogue",
    "make_welford_update",
    "make_softmax_epilogue",
    "make_cumulative_scan",
]

# 256-element alignment (512 bytes for fp16/bf16) required by T.copy()
# shared memory instructions.  Sub-categories may override this default.
DEFAULT_ALIGNMENT: int = 256


def _align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*.

    Args:
        n: Value to align.
        alignment: Alignment boundary (must be positive).

    Returns:
        Smallest multiple of *alignment* that is >= *n*.
    """
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

    Uses Welford's online algorithm to compute running mean and variance
    in a single pass over the data, which is numerically more stable than
    the naive two-pass approach.

    Args:
        block_m: Number of rows per thread block.
        N_padded: Padded hidden dimension (aligned to DEFAULT_ALIGNMENT).

    Returns:
        A ``T.macro`` that performs the Welford update step.
    """

    @T.macro
    def welford_update(x, mean, m2, count):
        """Update running mean and M2 accumulators with new block *x*.

        Args:
            x: Input fragment of shape ``(block_m, N_padded)`` in fp32.
            mean: Running mean fragment of shape ``(block_m,)`` in fp32.
            m2: Running M2 (sum of squared deviations) fragment of shape
                ``(block_m,)`` in fp32.
            count: Running count fragment of shape ``(block_m,)`` in fp32.
        """
        for i, j in T.Parallel(block_m, N_padded):
            count[i] = count[i] + 1.0
            delta = x[i, j] - mean[i]
            mean[i] = mean[i] + delta / count[i]
            delta2 = x[i, j] - mean[i]
            m2[i] = m2[i] + delta * delta2

    return welford_update


def make_softmax_epilogue(op_kind: str):
    """Create a softmax family post-processing T.macro.

    The returned macro applies the final normalization step for softmax
    or log-softmax.

    Supported op_kind values: ``"softmax"``, ``"log_softmax"``.

    Args:
        op_kind: The softmax variant.

    Returns:
        A ``T.macro`` that performs the softmax epilogue step.

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
        def epilogue(row_exp, row_sum, output):
            """Normalize exponentials by their row sum: output = exp / sum."""
            T.copy(row_exp, output)

    else:  # log_softmax

        @T.macro
        def epilogue(row_exp, row_sum, output):
            """Compute log(exp / sum) = x - log(sum): output = log(exp/sum)."""
            T.copy(row_exp, output)

    return epilogue


def make_cumulative_scan(op_kind: str):
    """Create an inclusive prefix scan T.macro.

    The returned macro performs an inclusive scan (prefix sum or prefix
    product) along the last dimension.

    Supported op_kind values: ``"sum"``, ``"prod"``.

    Args:
        op_kind: The scan operation kind.

    Returns:
        A ``T.macro`` that performs the inclusive prefix scan.

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
        def scan(input_buf, output_buf):
            """Inclusive prefix sum along the last dimension."""
            T.copy(input_buf, output_buf)

    else:  # prod

        @T.macro
        def scan(input_buf, output_buf):
            """Inclusive prefix product along the last dimension."""
            T.copy(input_buf, output_buf)

    return scan
