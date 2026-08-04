"""Shared reduction primitives for reduction kernels.

Provides reusable utility functions, constants, and T.macro factories
used across all reduction sub-category kernels (sum, max, softmax,
variance, prefix-scan, etc.).

This module must land before any sub-category kernel PR so that shared
infrastructure is available from the start.
"""

import itertools

import tilelang.language as T
import torch

__all__ = [
    "AUTOTUNE_THREADS",
    "DEFAULT_ALIGNMENT",
    "DEFAULT_THREADS",
    "MAX_SINGLE_TILE_COLS",
    "SHARED_MEMORY_BUDGET_BYTES",
    "VECTOR_ACCESS_BYTES",
    "BlockConfigPlanner",
    "align_up",
    "compute_tile_n",
    "device_smem_budget",
    "make_cumulative_scan",
    "make_reduce_epilogue",
    "make_softmax_epilogue",
    "make_welford_update",
    "reduce_column_alignment",
    "tune_by_forward",
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

# Width of the vectorized ``ld/st`` TileLang plans for a tile buffer on the
# architectures the reduction kernels declare (SM80-SM90): 128 bits.
VECTOR_ACCESS_BYTES: int = 16

# Thread counts offered by the reduction autotune candidate lists.
AUTOTUNE_THREADS: tuple[int, ...] = (128, 256)

# Thread count used when no candidate sweep runs.
DEFAULT_THREADS: int = 128


def reduce_column_alignment(elem_bytes: int, threads: int) -> int:
    """Return the column count one thread-block pass covers.

    Each thread takes one ``VECTOR_ACCESS_BYTES`` chunk per pass, so a pass is
    ``threads`` chunks wide.  ``layout_ok`` is what callers should ask; this is
    the granularity it measures against and generation aligns to.
    """
    return threads * VECTOR_ACCESS_BYTES // elem_bytes


class BlockConfigPlanner:
    """Derives ``block_m`` / ``threads`` / ``tile_n`` for a row-wise reduction.

    Every reduction kernel that maps one row per ``block_m`` slot makes the same
    three decisions -- whether a row fits in shared memory, which config to run
    untuned, and which candidates to offer the autotuner.  Derived here once so
    the kernels cannot answer them differently.

    Args:
        num_buffers: ``(block_m, tile_n)`` shared buffers alive at once.
            Welford's two-pass kernels allocate 2.
    """

    _BLOCK_MS = (1, 2, 4, 8)
    # Extra tile widths offered to the autotuner beyond the default pick.
    _EXTRA_TILE_N = 2

    def __init__(
        self,
        N_padded: int,
        elem_bytes: int,
        smem_budget: int,
        num_buffers: int = 1,
    ):
        self.N_padded = N_padded
        self.elem_bytes = elem_bytes
        self.smem_budget = smem_budget
        self.num_buffers = num_buffers

    @property
    def _row_bytes(self) -> int:
        """Shared memory one untiled row occupies.

        Excludes ``num_buffers``: that counts tiled shared buffers, and the
        untiled kernels keep their second pass in fragments.
        """
        return self.N_padded * self.elem_bytes

    @property
    def needs_tiling(self) -> bool:
        """Whether one padded row exceeds the column cap or the smem budget."""
        return (
            self.N_padded > MAX_SINGLE_TILE_COLS or self._row_bytes > self.smem_budget
        )

    def _column_alignment(self, block_m: int, threads: int) -> int:
        """Column granularity a tile must respect for this pair.

        ``block_m == 1`` needs only the ``T.copy`` alignment: one row cannot
        have a row-to-row thread-map shift, and the coarser granularity would
        only quantise the width away from an exact divisor of N_padded.
        """
        if block_m == 1:
            return DEFAULT_ALIGNMENT
        return reduce_column_alignment(self.elem_bytes, threads)

    def tile_n_for(self, block_m: int, threads: int) -> int:
        """Return the tile width to use untuned (0: this pair needs no tiling).

        Raises:
            ValueError: If no tile of the required column granularity fits in
                shared memory for this pair.
        """
        # Single-tile probe: one buffer, because the single-tile kernels hold
        # the row in fragments and allocate no second shared copy.
        if self.N_padded <= MAX_SINGLE_TILE_COLS:
            single = compute_tile_n(
                block_m, self.elem_bytes, self.N_padded, budget=self.smem_budget,
            )
            if single == self.N_padded:
                return 0

        col_budget = (
            MAX_SINGLE_TILE_COLS * self.num_buffers * block_m * self.elem_bytes
        )
        return compute_tile_n(
            block_m, self.elem_bytes, self.N_padded,
            alignment=self._column_alignment(block_m, threads),
            budget=min(self.smem_budget, col_budget),
            num_buffers=self.num_buffers,
        )

    def tile_n_candidates(self, block_m: int, threads: int) -> list[int]:
        """Return buildable tile widths to time for this pair, widest first.

        The widest is regularly beaten by a narrower tile trading one global
        pass for a better shared-memory stride, so the next tile counts follow
        it.  Empty when the pair can build no tile at all.
        """
        try:
            default = self.tile_n_for(block_m, threads)
        except ValueError:
            return []
        if default == 0:
            return [0]

        align = self._column_alignment(block_m, threads)
        out = [default]
        fewest = self._num_tiles(default)
        for n_tiles in range(fewest, fewest + self._EXTRA_TILE_N + 1):
            if len(out) > self._EXTRA_TILE_N:
                break
            tile_n = align_up((self.N_padded + n_tiles - 1) // n_tiles, align)
            if 0 < tile_n <= default and tile_n not in out:
                out.append(tile_n)
        return out

    def layout_ok(self, block_m: int, cols: int, threads: int) -> bool:
        """Whether a ``(block_m, cols)`` fragment is known to be reducible.

        A conservative envelope, not the exact rule.  It was the exact rule
        while the kernels wrote these fragments from a two-dimensional
        ``T.Parallel(block_m, cols)``: TileLang flattens that loop, the thread
        owning column *j* of row *i* is ``(i * cols + j) / vec % threads``, and
        the map repeats from row to row only when ``cols`` is a whole number of
        passes or divides one evenly.  Measured 44 of 44 over fp16/fp32 x
        {128, 256} threads x cols in [256, 4096].

        The kernels now serialise the row loop, so the map no longer depends on
        the row and nearly every width builds.  One narrow residue survives --
        softmax, log_softmax and logsumexp fail layout inference at
        ``block_m=4, cols=768`` -- and this envelope still excludes it, so it
        stays as the guard.  Everything it admits builds; some of what it
        rejects would now build too.

        ``block_m == 1`` is unconstrained: a single row cannot shift.
        """
        if block_m == 1:
            return True
        one_pass = reduce_column_alignment(self.elem_bytes, threads)
        return cols % one_pass == 0 or one_pass % cols == 0

    def reject_tile_n(self, block_m: int, tile_n: int, threads: int) -> str:
        """Return why a caller-supplied ``tile_n`` is unusable, or ""."""
        if tile_n <= 0 or tile_n % DEFAULT_ALIGNMENT:
            return (
                f"tile_n={tile_n} must be positive and a multiple of "
                f"{DEFAULT_ALIGNMENT} (the T.copy shared-memory alignment)"
            )
        if tile_n > MAX_SINGLE_TILE_COLS:
            return f"tile_n={tile_n} exceeds the {MAX_SINGLE_TILE_COLS} column cap"
        held = self.num_buffers * block_m * tile_n * self.elem_bytes
        if held > self.smem_budget:
            return (
                f"tile_n={tile_n} with block_m={block_m} needs {held} bytes of "
                f"shared memory, over the {self.smem_budget} budget"
            )
        if not self.layout_ok(block_m, tile_n, threads):
            one_pass = reduce_column_alignment(self.elem_bytes, threads)
            return (
                f"tile_n={tile_n} neither divides nor is a multiple of the "
                f"{one_pass}-column thread-block pass (threads={threads}, "
                f"elem_bytes={self.elem_bytes}), so a (block_m={block_m}, "
                f"tile_n) fragment has no reducible layout"
            )
        return ""

    def _untiled_block_ms(self, threads: int, budget: int | None = None) -> list[int]:
        """Row counts an untiled kernel can build within *budget*.

        ``default_config`` passes the conservative ``SHARED_MEMORY_BUDGET_BYTES``
        and the sweep passes the device budget.  Largest-that-fits is a
        capacity rule, not a performance one -- at 2048x4096 fp16 on H200,
        ``block_m=4`` beats ``block_m=8`` on sum, var and l2 alike -- so the
        untuned path stays inside the smaller envelope and the sweep, which
        times every row count, is what reaches the wider one.
        """
        max_block_m = (budget or self.smem_budget) // self._row_bytes
        return [
            bm for bm in self._BLOCK_MS
            if bm <= max_block_m and self.layout_ok(bm, self.N_padded, threads)
        ]

    def _num_tiles(self, tile_n: int) -> int:
        return (self.N_padded + tile_n - 1) // tile_n

    def default_config(self) -> dict:
        """Return the config used when no candidate sweep runs."""
        if not self.needs_tiling:
            block_ms = self._untiled_block_ms(
                DEFAULT_THREADS, budget=SHARED_MEMORY_BUDGET_BYTES,
            )
            return {
                "block_m": block_ms[-1] if block_ms else 1,
                "threads": DEFAULT_THREADS,
            }

        # Tiled: block_m == 1 always needs the fewest N-tiles.  A larger
        # block_m only shrinks the per-row shared-memory budget, so its tile is
        # narrower and its tile count never lower.
        return {
            "block_m": 1,
            "threads": DEFAULT_THREADS,
            "tile_n": self.tile_n_for(1, DEFAULT_THREADS),
        }

    def autotune_configs(self) -> list[dict]:
        """Return every candidate config, all of them buildable.

        ``tile_n`` is a search dimension, not a function of the other two.
        Pairs with no buildable tile are dropped here rather than left to
        abort the sweep at build time.
        """
        if not self.needs_tiling:
            return [
                {"block_m": bm, "threads": t}
                for t in AUTOTUNE_THREADS
                for bm in self._untiled_block_ms(t)
            ]

        return [
            {"block_m": bm, "threads": t, "tile_n": tile_n}
            for bm, t in itertools.product(self._BLOCK_MS, AUTOTUNE_THREADS)
            for tile_n in self.tile_n_candidates(bm, t)
        ]


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


# Supported op_kind values for each macro factory
_REDUCE_KINDS = {"sum", "max", "min"}
_SOFTMAX_KINDS = {"softmax", "log_softmax"}
_SCAN_KINDS = {"sum", "prod"}



def tune_by_forward(kernel, *probe_inputs, warmup: int = 10, rep: int = 10) -> None:
    """Select the fastest candidate config by timing ``kernel.forward``.

    The tiled reduction paths have no single ``self.kernel`` object for
    TileLang's autotuner to decorate — they dispatch through wrapped helper
    functions — so each candidate is timed through ``forward`` instead.
    Leaves ``kernel.config`` set to the winner, or to ``default_config`` when
    the kernel declares no candidates.
    """
    configs = kernel.autotune_configs
    if not configs:
        kernel.config = kernel.default_config
        return

    print(f'Start autotuning {kernel.__class__.__name__} (tiled path)...')
    best_config, best_time = configs[0], float('inf')
    for cfg in configs:
        kernel.config = cfg
        for _ in range(warmup):
            kernel.forward(*probe_inputs)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            kernel.forward(*probe_inputs)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / rep

        if elapsed < best_time:
            best_time, best_config = elapsed, cfg

    kernel.config = best_config
    print(f'Best config: {kernel.config}')

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
        for i in T.serial(block_m):
            for j in T.Parallel(N_padded):
                dev = x[i, j] - batch_mean[i]
                sq_diff[i, j] = dev * dev
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
