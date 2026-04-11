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
    "compute_tile_n",
    "device_smem_budget",
    "DEFAULT_ALIGNMENT",
    "MAX_SINGLE_TILE_COLS",
    "SHARED_MEMORY_BUDGET_BYTES",
    "make_reduce_epilogue",
    "make_welford_update",
    "make_softmax_epilogue",
    "make_cumulative_scan",
]

# 256-element alignment (512 bytes for fp16/bf16) required by T.copy()
# shared memory instructions.  Sub-categories may override this default.
DEFAULT_ALIGNMENT: int = 256

# Maximum column count for a single fragment/shared-memory tile.
# TileLang's vectorizer fails when the *column dimension* of a
# fragment or shared buffer reaches 32768 (a LLVM scalable-vector
# boundary).  Empirical testing on H200 (SM90) confirms that
# 32512 columns compile and execute correctly, while 32768 triggers
# the "scalable vector" error.  We use 32512 (= 32768 - 256) as the
# safe upper bound.
MAX_SINGLE_TILE_COLS: int = 32512

# Default shared memory budget per SM (48 KiB) used to compute the maximum
# block_m that fits within a single thread block's shared memory allocation.
SHARED_MEMORY_BUDGET_BYTES: int = 48 * 1024


def device_smem_budget(device_index: int | None = None) -> int:
    """Return the opt-in shared memory budget for a CUDA device.

    If ``device_index`` is ``None``, the current CUDA device is used.

    Modern GPUs (SM80+) support shared memory well beyond the 48 KiB
    default.  TileLang automatically configures
    ``cudaFuncSetAttribute`` when a kernel allocates more than 48 KiB,
    so it is safe to use the full opt-in budget.

    Falls back to ``SHARED_MEMORY_BUDGET_BYTES`` (48 KiB) only if
    CUDA/device properties are unavailable.  Invalid explicit device
    indices are not silently masked -- only the ``None`` (auto-detect)
    case falls back gracefully.
    """
    explicit = device_index is not None
    try:
        import torch
    except Exception:
        if explicit:
            raise
        return SHARED_MEMORY_BUDGET_BYTES

    try:
        if not torch.cuda.is_available():
            if explicit:
                raise RuntimeError(
                    f"CUDA is not available but explicit device_index={device_index} was requested"
                )
            return SHARED_MEMORY_BUDGET_BYTES

        if device_index is None:
            device_index = torch.cuda.current_device()

        props = torch.cuda.get_device_properties(device_index)
        smem_optin = getattr(props, "shared_memory_per_block_optin", 0)
        if smem_optin > 0:
            return smem_optin
        return getattr(props, "shared_memory_per_block", SHARED_MEMORY_BUDGET_BYTES)
    except (RuntimeError, AssertionError):
        if explicit:
            raise
        return SHARED_MEMORY_BUDGET_BYTES


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


def compute_tile_n(
    block_m: int,
    elem_bytes: int,
    N_padded: int,
    alignment: int = DEFAULT_ALIGNMENT,
    budget: int = SHARED_MEMORY_BUDGET_BYTES,
    num_buffers: int = 1,
) -> int:
    """Compute the tile_n (column chunk) for shared memory, preferring divisibility.

    The budget-derived cap (``tile_n_max``) is the largest multiple of
    *alignment* such that
    ``num_buffers * block_m * tile_n_max * elem_bytes <= budget``.
    The return value may be a smaller divisor of *N_padded* when that
    divisor does not increase the number of N-tiles, because an exact
    division eliminates the nearly-empty remainder tile and reduces
    wasted memory traffic.  For example, N_padded=32768 with
    tile_n_max=32512 gives 2 tiles where tile 2 has only 256 valid
    columns (99.2% waste), while divisor=16384 also gives 2 tiles with
    zero waste.

    When every divisor requires strictly more tiles, the full
    budget-derived cap is returned and the tiled kernel handles the
    single remainder tile via masked loads.

    If N_padded already fits (with *num_buffers* copies), returns N_padded
    (no tiling needed).

    Args:
        block_m: Number of rows per thread block.
        elem_bytes: Bytes per element (e.g. 2 for fp16/bf16, 4 for fp32).
        N_padded: Padded hidden dimension (already aligned to *alignment*).
        alignment: Column alignment boundary (default DEFAULT_ALIGNMENT).
        budget: Shared memory budget in bytes (default 48 KiB).
        num_buffers: Number of shared memory buffers of shape
            ``(block_m, tile_n)`` that must fit simultaneously (default 1).
            Softmax/log_softmax tiled kernels use 2 (one per pass) due to
            TileLang allocator aliasing constraints.

    Returns:
        tile_n: column tile size, a multiple of *alignment*, or N_padded
        if it fits entirely.

    Raises:
        ValueError: If even a single alignment-width slice cannot fit.
    """
    per_buffer = block_m * elem_bytes
    if num_buffers * per_buffer * N_padded <= budget:
        return N_padded

    # Largest multiple of alignment that fits in shared memory
    max_cols = budget // (num_buffers * per_buffer)
    tile_n_max = (max_cols // alignment) * alignment
    if tile_n_max == 0:
        raise ValueError(
            f"Cannot fit even {alignment} columns in {budget} bytes "
            f"with block_m={block_m}, elem_bytes={elem_bytes}, "
            f"num_buffers={num_buffers}."
        )

    # Prefer the largest tile_n that evenly divides N_padded, so that
    # num_tiles * tile_n == N_padded and no remainder tile is needed.
    # Search downward from tile_n_max in alignment-sized steps.
    best_dividing = 0
    for candidate in range(tile_n_max, 0, -alignment):
        if N_padded % candidate == 0:
            best_dividing = candidate
            break

    # Accept the divisor when it does not increase the number of
    # N-tiles.  Each tile incurs a global-memory pass in both passes of
    # the 2-pass softmax, so fewer tiles is always cheaper.  When the
    # divisor gives the same tile count as tile_n_max, prefer the
    # divisor because it eliminates the nearly-empty remainder tile
    # (e.g. N_padded=32768, tile_n_max=32512 → 2 tiles with a 256-col
    # remainder vs divisor=16384 → 2 even tiles with zero waste).
    #
    # When the divisor requires strictly more tiles (e.g. smaller
    # divisors of N_padded), stick with tile_n_max and handle the
    # single remainder tile via masked loads.
    if best_dividing > 0:
        div_tiles = N_padded // best_dividing  # exact division
        max_tiles = (N_padded + tile_n_max - 1) // tile_n_max
        if div_tiles <= max_tiles:
            return best_dividing
    return tile_n_max


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
        batch_mean = T.alloc_fragment((block_m,), "float32")
        new_count = T.alloc_fragment((block_m,), "float32")
        new_mean = T.alloc_fragment((block_m,), "float32")
        for i in T.Parallel(block_m):
            batch_mean[i] = row_sum[i] / float(N_padded)
            new_count[i] = count[i] + float(N_padded)
            new_mean[i] = (mean[i] * count[i] + row_sum[i]) / new_count[i]

        # Compute M2_b = sum((x[j] - batch_mean)^2) -- deviations from
        # the *batch's own mean*, not the combined mean.  The parallel
        # Welford merge formula requires this to be correct.
        for i, j in T.Parallel(block_m, N_padded):
            sq_diff[i, j] = (x[i, j] - batch_mean[i]) * (x[i, j] - batch_mean[i])
        T.reduce_sum(sq_diff, row_sq_sum, dim=1)

        # Combine with existing M2 using parallel Welford merge formula:
        #   M2_combined = M2_a + M2_b + delta^2 * (n_a * n_b / n_combined)
        # Here M2_b = row_sq_sum and delta = batch_mean - mean (old).
        for i in T.Parallel(block_m):
            delta = batch_mean[i] - mean[i]
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
