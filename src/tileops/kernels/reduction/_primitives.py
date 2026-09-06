"""Shared reduction primitives for reduction kernels.

Launch planning, shared-memory arithmetic and the row layout the reduction
kernels build on, plus the engine that reduces an ``(A, B)`` buffer down its
rows.

A reduction kernel reduces the trailing axis of a 2-D ``(M, N)`` buffer, while an op
declares an arbitrary-rank tensor and the axes to reduce; the permute and flatten between
the two is a kernel's business, so both sides of the op/backend boundary speak the shapes
the manifest declares. ``axes`` below is the sorted tuple of non-negative axis indices the
reduction runs over — which forms an empty ``dim`` takes, and which ranks it may name, are
the op's contract rather than a kernel's.
"""

import functools
import itertools
from dataclasses import dataclass
from math import prod

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import SHARED_BANK_SPAN_BYTES, VECTOR_ACCESS_BYTES
from tileops.kernels.tiling import ALIGNMENT, align_up
from tileops.utils import device_busy_of

__all__ = [
    "AUTOTUNE_THREADS",
    "DEFAULT_ALIGNMENT",
    "DEFAULT_THREADS",
    "FP32_EXACT_INT_LIMIT",
    "FRAGMENT_ELEMS_PER_THREAD",
    "MAX_SINGLE_TILE_COLS",
    "SHARED_BANK_SPAN_BYTES",
    "SHARED_MEMORY_BUDGET_BYTES",
    "VECTOR_ACCESS_BYTES",
    "BlockConfigPlanner",
    "RowTiledAutotuneMixin",
    "align_up",
    "ceildiv_int",
    "compute_tile_n",
    "device_smem_budget",
    "edge_axis_plan",
    "edge_axis_split",
    "identity_for",
    "reduce_down_rows",
    "restore_reduced",
    "restore_same_shape",
    "rows_for_axes",
    "torch_dtype_nbytes",
    "tune_by_forward",
]

# 256-element alignment (512 bytes for fp16/bf16) required by T.copy()
# shared memory instructions.  Sub-categories may override this default.
DEFAULT_ALIGNMENT: int = ALIGNMENT

# Widest single fragment/shared-memory tile the reduction kernels plan; shared memory
# and the register file are checked separately.
MAX_SINGLE_TILE_COLS: int = 32768

# Default shared memory budget per SM (48 KiB) used to compute the maximum
# block_m that fits within a single thread block's shared memory allocation.
SHARED_MEMORY_BUDGET_BYTES: int = 48 * 1024


# Thread counts offered by the reduction autotune candidate lists.
AUTOTUNE_THREADS: tuple[int, ...] = (128, 256, 512)

# Thread count used when no candidate sweep runs.
DEFAULT_THREADS: int = 256

# Tile elements one thread may hold across its live fragments before ptxas spills.
FRAGMENT_ELEMS_PER_THREAD: int = 64


def ceildiv_int(x: int, y: int) -> int:
    """Return ``ceil(x / y)`` for positive integer dimensions."""
    return -(-x // y)


def identity_for(op_kind: str) -> float:
    """The value a masked lane contributes, leaving the reduction unchanged."""
    if op_kind in ("prod", "all"):
        return 1.0
    if op_kind == "amin":
        return float("inf")
    if op_kind == "amax":
        return float("-inf")
    return 0.0


def torch_dtype_nbytes(dtype: torch.dtype | str) -> int:
    """Return element size for a torch dtype object or dtype-name string."""
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.empty(0, dtype=dtype).element_size()


def reduce_column_alignment(elem_bytes: int, threads: int) -> int:
    """Return the column count one thread-block pass covers.

    Each thread takes one ``VECTOR_ACCESS_BYTES`` chunk per pass, so a pass is
    ``threads`` chunks wide.  ``layout_ok`` is what callers should ask; this is
    the granularity it measures against and generation aligns to.
    """
    return threads * VECTOR_ACCESS_BYTES // elem_bytes


class BlockConfigPlanner:
    """Derives ``block_m`` / ``threads`` / ``tile_n`` for a row-wise reduction.

    Answers three questions for a kernel that maps one row per ``block_m`` slot:
    whether a row fits in shared memory, which config to run untuned, and which
    candidates to offer the autotuner.

    ``autotune_configs`` here serves the ``tune_by_forward`` sweeps. A kernel
    baking ``tile_n`` into the PrimFunc sweeps `RowTiledAutotuneMixin`'s list
    instead, which differs at every width.

    Args:
        num_buffers: ``(block_m, tile_n)`` shared buffers alive at once.
            Welford's two-pass kernels allocate 2.
        frag_slots: ``(block_m, tile_n)`` fragments the kernel keeps alive at once.
            Registers, not shared memory: a tile that fits in shared memory can still
            spill.
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
        frag_slots: int = 1,
    ):
        self.N_padded = N_padded
        self.elem_bytes = elem_bytes
        self.smem_budget = smem_budget
        self.num_buffers = num_buffers
        self.frag_slots = frag_slots

    @property
    def _row_bytes(self) -> int:
        """Shared memory one untiled row occupies.

        Excludes ``num_buffers``: that counts tiled shared buffers, and the
        untiled kernels keep their second pass in fragments.
        """
        return self.N_padded * self.elem_bytes

    def frag_elems(self, block_m: int, cols: int, threads: int) -> int:
        """Tile elements one thread holds across every live fragment."""
        return ceildiv_int(block_m * cols, threads) * self.frag_slots

    def frag_fits(self, block_m: int, cols: int, threads: int) -> bool:
        """Whether a ``(block_m, cols)`` tile stays in registers for this pair."""
        return self.frag_elems(block_m, cols, threads) <= FRAGMENT_ELEMS_PER_THREAD

    @property
    def needs_tiling(self) -> bool:
        """Whether one padded row exceeds what a single untiled pass can hold.

        Three capacities, any one of which forces the tiled kernel: the tile column
        cap, shared memory, and the register file. The register question is
        asked of the narrowest untiled configuration, one row over
        ``DEFAULT_THREADS`` threads, since a larger ``block_m`` only adds to it.
        """
        return (
            self.N_padded > MAX_SINGLE_TILE_COLS
            or self._row_bytes > self.smem_budget
            or not self.frag_fits(1, self.N_padded, DEFAULT_THREADS)
        )

    def _column_alignment(self, block_m: int, threads: int) -> int:
        """Column granularity a tile must respect for this pair.

        ``block_m == 1`` has no row-to-row thread-map shift to respect, so the
        ``T.copy`` alignment would be enough for the copy itself -- but the fragment
        still has to divide across the block, which is what ``layout_ok`` measures.
        """
        if block_m == 1:
            return max(DEFAULT_ALIGNMENT, threads)
        return reduce_column_alignment(self.elem_bytes, threads)

    def tile_n_for(self, block_m: int, threads: int) -> int:
        """Return the tile width to use untuned (0: this pair needs no tiling).

        Raises:
            ValueError: If no tile of the required column granularity fits in
                shared memory for this pair.
        """
        # Single-tile probe: one buffer, because the single-tile kernels hold
        # the row in fragments and allocate no second shared copy.
        if self.N_padded <= MAX_SINGLE_TILE_COLS and self.frag_fits(
            block_m, self.N_padded, threads
        ):
            single = compute_tile_n(
                block_m,
                self.elem_bytes,
                self.N_padded,
                budget=self.smem_budget,
            )
            if single == self.N_padded:
                return 0
        return self.tiled_tile_n(block_m, threads)

    def tiled_tile_n(self, block_m: int, threads: int) -> int:
        """Return the widest tile this pair can build, untiled fit or not.

        Raises:
            ValueError: If no tile of the required column granularity fits in
                shared memory for this pair.
        """
        # The register budget bounds the untiled probe above, not the tile width.
        col_budget = MAX_SINGLE_TILE_COLS * self.num_buffers * block_m * self.elem_bytes
        return compute_tile_n(
            block_m,
            self.elem_bytes,
            self.N_padded,
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

        A conservative envelope, not the exact rule.  It is exact for a fragment
        written from a two-dimensional ``T.Parallel(block_m, cols)``: TileLang
        flattens that loop, the thread owning column *j* of row *i* is
        ``(i * cols + j) / vec % threads``, and the map repeats from row to row
        only when ``cols`` is a whole number of passes or divides one evenly.
        Measured 44 of 44 over fp16/fp32 x {128, 256} threads x cols in
        [256, 4096].

        The kernels serialise the row loop, so the map does not depend on the row
        and nearly every width builds.  One narrow residue survives --
        softmax, log_softmax and logsumexp fail layout inference at
        ``block_m=4, cols=768`` -- and this envelope still excludes it, so it
        stays as the guard.  Everything it admits builds; some of what it
        rejects would now build too.

        ``block_m == 1`` has no row-to-row shift, but the columns still have to divide
        across the block: layout inference finds no layout otherwise. Exact over 72 of
        72 combinations of nine widths, four thread counts and fp16/fp32 -- a
        ``(1, cols)`` fragment builds if and only if ``threads`` divides ``cols``.
        """
        if block_m == 1:
            return cols % threads == 0
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
            if block_m == 1:
                return (
                    f"tile_n={tile_n} is not a multiple of threads={threads}, so a "
                    f"(1, tile_n) fragment has no reducible layout"
                )
            one_pass = reduce_column_alignment(self.elem_bytes, threads)
            return (
                f"tile_n={tile_n} neither divides nor is a multiple of the "
                f"{one_pass}-column thread-block pass (threads={threads}, "
                f"elem_bytes={self.elem_bytes}), so a (block_m={block_m}, "
                f"tile_n) fragment has no reducible layout"
            )
        return ""

    def _untiled_block_ms(self, threads: int, budget: int | None = None) -> list[int]:
        """Row counts an untiled kernel can build within *budget*, ascending.

        ``default_config`` passes the conservative ``SHARED_MEMORY_BUDGET_BYTES``
        and the sweep passes the device budget.  Capacity only, not a ranking:
        which of these to run untuned is ``default_config``'s call.
        """
        max_block_m = (budget or self.smem_budget) // self._row_bytes
        return [
            bm
            for bm in self._BLOCK_MS
            if bm <= max_block_m
            and self.layout_ok(bm, self.N_padded, threads)
            and self.frag_fits(bm, self.N_padded, threads)
        ]

    def _num_tiles(self, tile_n: int) -> int:
        return (self.N_padded + tile_n - 1) // tile_n

    def default_config(self) -> dict:
        """Return the config used when no candidate sweep runs."""
        if not self.needs_tiling:
            block_ms = self._untiled_block_ms(
                DEFAULT_THREADS,
                budget=SHARED_MEMORY_BUDGET_BYTES,
            )
            # The fewest rows a block can take, not the most it can hold.
            return {
                "block_m": block_ms[0] if block_ms else 1,
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


def tune_by_forward(
    kernel,
    *probe_inputs,
    warmup: int = 10,
    rep: int = 10,
    forward=None,
) -> None:
    """Select the fastest candidate config by timing one call per candidate.

    The tiled reduction paths have no single ``self.kernel`` object for
    TileLang's autotuner to decorate — they dispatch through wrapped helper
    functions — so each candidate is timed through a call instead.
    Leaves ``kernel.config`` set to the winner, or to ``default_config`` when
    the kernel declares no candidates.

    Args:
        kernel: The kernel whose ``config`` is being chosen.
        probe_inputs: What to call with.
        warmup: Untimed calls per candidate.
        rep: Timed calls per candidate.
        forward: What to call. Defaults to ``kernel.forward``. A kernel that reshapes
            its input inside ``forward`` passes the row-level entry point instead, so the
            probe is a ``(M, N)`` buffer and the timing excludes a config-independent
            permute.
    """
    call = kernel.forward if forward is None else forward
    configs = kernel.autotune_configs
    if not configs:
        kernel.config = kernel.default_config
        return

    print(f"Start autotuning {kernel.__class__.__name__} (tiled path)...")
    best_config, best_time = configs[0], float("inf")
    for cfg in configs:
        kernel.config = cfg
        for _ in range(warmup):
            call(*probe_inputs)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            call(*probe_inputs)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / rep

        if elapsed < best_time:
            best_time, best_config = elapsed, cfg

    kernel.config = best_config
    print(f"Best config: {kernel.config}")


# The row layout a reduction kernel reduces, and the shape its caller declared.


def _kept(ndim: int, axes: "tuple[int, ...]") -> "list[int]":
    """The axes the reduction leaves, in order."""
    return [i for i in range(ndim) if i not in axes]


def rows_for_axes(x: torch.Tensor, axes: "tuple[int, ...]") -> torch.Tensor:
    """Move *axes* to the end and flatten to ``(M, N)``.

    Reducing every axis gives ``M == 1``.
    """
    kept = _kept(x.ndim, axes)
    n = prod(x.shape[a] for a in axes)
    m = prod(x.shape[i] for i in kept)
    return x.permute(kept + list(axes)).contiguous().reshape(m, n)


def restore_reduced(
    y: torch.Tensor,
    in_shape: "tuple[int, ...]",
    axes: "tuple[int, ...]",
    keepdim: bool,
) -> torch.Tensor:
    """Shape an $[M]$ result the way the reduction's caller expects.

    Reducing every axis without *keepdim* gives a 0-D tensor.
    """
    if keepdim:
        return y.reshape([1 if i in axes else d for i, d in enumerate(in_shape)])
    kept = [in_shape[i] for i in _kept(len(in_shape), axes)]
    return y.reshape(kept) if kept else y.reshape(())


def restore_same_shape(
    y: torch.Tensor,
    in_shape: "tuple[int, ...]",
    axes: "tuple[int, ...]",
) -> torch.Tensor:
    """Undo `rows_for_axes` on a result that kept its input's shape.

    For an op that writes one element per input element — softmax, a prefix scan — whose
    row layout is unwound rather than collapsed.
    """
    kept = _kept(len(in_shape), axes)
    perm = kept + list(axes)
    y = y.reshape([in_shape[i] for i in perm])
    inverse = [0] * len(perm)
    for position, axis in enumerate(perm):
        inverse[axis] = position
    # Contiguous, because the op's fake reports contiguous strides and a mismatch there is
    # a silent wrong answer, not a failure. Free when the reduced axis is already last.
    return y.permute(inverse).contiguous()


class RowTiledAutotuneMixin:
    """The tile_n search shared by the kernels that bake tile_n in at build time.

    A kernel holding a ``(block_m, N_padded)`` row block picks tile_n before the
    autotuner runs, because tile_n decides the kernel's shape and so every distinct
    value costs a recompilation.

    A subclass must set, before autotuning: ``_planner`` (a
    `BlockConfigPlanner`), ``_smem_budget``, ``N_padded``, ``_elem_bytes``,
    ``_MAX_TILE_N_CANDIDATES``, ``_tile_n`` and ``_split_target``, and implement
    ``_build_row_kernel`` and ``_row_forward``.
    """

    def _build_row_kernel(self, tile_n: int):
        raise NotImplementedError

    def _row_forward(self, x):
        raise NotImplementedError

    def _sweep_applies(self) -> bool:
        return True

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed).

        Derived at the granularity the *coarsest* candidate thread count
        needs: tile_n is baked into the kernel at build time and then reused
        across every ``threads`` value the autotuner tries, so one tile has to
        satisfy all of them.
        """
        return self._planner.tile_n_for(block_m, max(AUTOTUNE_THREADS))

    def _untiled_tile_alternative(self) -> list[int]:
        """Return the tiles to time beside an untiled row, widest first.

        tile_n is baked in at build time and reused across every ``threads`` the
        sweep tries, and the register budget binds at the fewest of them, where each
        thread holds the most of the row. A row admitted untiled at the most threads
        can still be run at the fewest, over that budget; offering it a tile leaves
        the choice to measurement. Empty when the row fits untiled at every candidate
        thread count, which is where the fragment is cheap enough not to ask.
        """
        if self._planner.frag_fits(1, self.N_padded, min(AUTOTUNE_THREADS)):
            return []
        try:
            widest = self._planner.tiled_tile_n(1, max(AUTOTUNE_THREADS))
        except ValueError:
            return []
        if widest <= 0:
            return []
        half = widest // 2 // DEFAULT_ALIGNMENT * DEFAULT_ALIGNMENT
        return [widest] if half in (0, widest) else [widest, half]

    def _tile_n_candidates(self) -> list[int]:
        """Return candidate tile_n values for autotune exploration.

        Includes the heuristic tile_n (from block_m=1) plus alternative
        tile_n values derived from ``_tile_n_for_block_m(2)`` and
        ``_tile_n_for_block_m(4)``, with a half-step fallback aligned to
        ``DEFAULT_ALIGNMENT`` when block_m exploration yields no
        alternatives.  tile_n=0 means single-tile (no tiling).  All
        candidates are de-duplicated and sorted descending for
        deterministic ordering.

        Each distinct tile_n value requires a full kernel recompilation,
        which is expensive for large-N workloads (compilations can take
        minutes each).  To keep autotuner wall time practical we cap
        the total number of tile_n candidates at ``_MAX_TILE_N_CANDIDATES``
        (currently 3).

        - When the heuristic default tile_n is 0 (single-tile / small N),
          return ``[0]`` -- the autotuner varies only block_m and threads.
        - Otherwise collect distinct tile_n values from block_m=1..4 and
          return up to ``_MAX_TILE_N_CANDIDATES`` candidates (always
          including the heuristic default).
        """
        default_tn = self._tile_n_for_block_m(1)
        if default_tn == 0:
            return [0, *self._untiled_tile_alternative()]

        candidates: set[int] = {default_tn}
        # Explore tile_n values implied by small block_m values.
        # Higher block_m → smaller tile_n (more N-tiles but better row reuse).
        for bm in (2, 4):
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn > 0 and tn != default_tn:
                candidates.add(tn)

        # Also try half of the default tile_n (rounded to alignment) as a
        # search point when block_m exploration didn't yield alternatives.
        if len(candidates) < 2:
            half_tn = (default_tn // 2 // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
            if half_tn > 0 and half_tn != default_tn:
                candidates.add(half_tn)

        # Cap to avoid excessive compilation time.
        sorted_candidates = sorted(candidates, reverse=True)
        return sorted_candidates[: self._MAX_TILE_N_CANDIDATES]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Sweep the candidates, rebuilding per tile_n, then judge the split pair.

        The split pair is timed on device kernel time: paths launching different
        kernel counts cannot be compared on wall time, which carries the
        host-launch gaps.
        """
        from tilelang.autotuner import autotune as tl_autotune

        from ._split_softmax import split_seg_n

        if not self._sweep_applies():
            self.config = self.default_config
            return

        default = self.default_config
        split_eligible = bool(split_seg_n(self.M, self.N, default["block_m"], self._split_target))

        configs = self.autotune_configs
        if not configs:
            self.config = default
            return

        by_tile_n: dict[int, list[dict]] = {}
        for cfg in configs:
            by_tile_n.setdefault(cfg["tile_n"], []).append(
                {"block_m": cfg["block_m"], "threads": cfg["threads"]}
            )

        best_time = float("inf")
        best_config = None
        for tile_n, group_cfgs in by_tile_n.items():
            kernel = self._build_row_kernel(tile_n)
            tunable_params = list(self._autotune_initial_kwargs(kernel, group_cfgs[0]).keys())
            autotune_kwargs: dict = dict(configs=group_cfgs, warmup=warmup, rep=rep)
            if tunable_params:
                autotune_kwargs["do_not_specialize"] = tunable_params
            if self.autotune_supply_prog is not None:
                autotune_kwargs["supply_prog"] = self.autotune_supply_prog
            autotuned = tl_autotune(**autotune_kwargs)(kernel)
            tuned = self._call_autotuned_kernel(autotuned, kernel, group_cfgs[0])
            if tuned.latency < best_time:
                best_time = tuned.latency
                best_config = {**tuned.config, "tile_n": tile_n}

        if best_config is not None:
            self.config = best_config
            if best_config["tile_n"] != self._tile_n:
                self._tile_n = best_config["tile_n"]
                self.kernel = self._build_row_kernel(self._tile_n)

        if split_eligible and best_config is not None:
            device = torch.device(
                "cuda",
                self.device_index if self.device_index is not None else torch.cuda.current_device(),
            )
            probe = torch.randn(self.M, self.N, dtype=self.dtype, device=device)
            swept_config = dict(self.config, split=False)
            self.config = swept_config
            swept_ms = device_busy_of(lambda: self._row_forward(probe), device)
            self.config = default
            split_ms = device_busy_of(lambda: self._row_forward(probe), device)
            if split_ms > swept_ms:
                self.config = swept_config

    @property
    def autotune_configs(self) -> list[dict]:
        """Generate autotune configs including tile_n candidates.

        tile_n is baked into the kernel at build time, so the autotuner
        rebuilds the kernel for each tile_n value.  Configs include
        ``tile_n`` alongside ``block_m`` and ``threads``.
        """
        budget = self._smem_budget
        smem_per_row = self.N_padded * self._elem_bytes
        max_block_m_no_tile = budget // smem_per_row if smem_per_row > 0 else 16
        threads_list = list(AUTOTUNE_THREADS)

        configs = []
        for tile_n in self._tile_n_candidates():
            if tile_n == 0:
                # Single-tile regime: explore multiple block_m values.
                for bm in [1, 2, 4, 8, 16]:
                    # Can this row count build a tile, and is that tile the whole row.
                    try:
                        bm_tile_n = self._tile_n_for_block_m(bm)
                    except ValueError:
                        continue
                    if bm_tile_n != 0:
                        continue
                    if bm > max_block_m_no_tile:
                        continue
                    for t in threads_list:
                        if not self._planner.layout_ok(bm, self.N_padded, t):
                            continue
                        configs.append({"block_m": bm, "threads": t, "tile_n": 0})
            else:
                # Tiled regime: block_m=1 with each tile_n candidate. Each
                # distinct tile_n triggers a kernel recompilation, so only
                # threads vary within a tile_n regime.
                for t in threads_list:
                    configs.append({"block_m": 1, "threads": t, "tile_n": tile_n})

        if not configs:
            configs = [{"block_m": 1, "threads": 256, "tile_n": self._tile_n}]

        return configs


# ---------------------------------------------------------------------------
# Reducing an (A, B) buffer down its rows: the outer pass every edge-axis
# reduction shares. The inner pass is each kernel class's own; this engine
# takes its partials and reduces them down the lead axis.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LeadingAxisReducePolicy:
    """Launch heuristics for reductions down contiguous leading axes.

    The column tile ``threads * cols_per_thread`` is what a block reads from
    one row, and 512 columns is flat across the shapes this serves; how that
    tile is divided is not. Sixty-four lanes reading eight columns each leaves
    a block too narrow to cover the read's latency, and costs several times
    what 256 lanes reading two costs at the same tile, split count and grid.
    """

    cols_per_thread: int = 2

    threads: int = 256

    target_blocks: int = 256


_LEADING_POLICY = _LeadingAxisReducePolicy()


# Largest integer count fp32 carries exactly; a statistic folded through
# fp32 counts or weights is trusted only below it.
FP32_EXACT_INT_LIMIT = 1 << 24


def edge_axis_plan(
    shape: "tuple[int, ...]",
    k: int,
    j: int,
    elem_bytes: int,
    smem_budget: int,
    **planner_kwargs,
):
    """Split *shape* for an edge-axis reduction and plan its rows pass.

    Returns ``(lead, kept, trail, planner, cfg)``: the leading and trailing
    reduced element counts, the kept middle, and the ``BlockConfigPlanner``
    with its default config for rows of ``trail`` elements.
    """
    ndim = len(shape)
    lead = prod(shape[:k])
    kept = prod(shape[k : ndim - j])
    trail = prod(shape[ndim - j :])
    planner = BlockConfigPlanner(
        align_up(trail, DEFAULT_ALIGNMENT),
        elem_bytes,
        smem_budget,
        **planner_kwargs,
    )
    return lead, kept, trail, planner, planner.default_config()


def edge_axis_split(ndim: int, axes: "tuple[int, ...]") -> "tuple[int, int]":
    """Split *axes* into ``(leading, trailing)`` counts when they hug both edges.

    Returns ``(0, 0)`` unless *axes* is a non-empty prefix plus a non-empty
    suffix with at least one kept axis in between — the layout an edge-axis
    reduction handles without permuting the tensor.
    """
    k = 0
    while k < len(axes) and axes[k] == k:
        k += 1
    j = len(axes) - k
    if k == 0 or j == 0:
        return (0, 0)
    if tuple(axes[k:]) != tuple(range(ndim - j, ndim)):
        return (0, 0)
    if k + j >= ndim:
        return (0, 0)
    return (k, j)


def _leading_row_splits(reduced: int, kept: int, threads: int) -> int:
    """How many ways to split the reduced axis so the grid fills the device."""
    block_b = threads * _LEADING_POLICY.cols_per_thread
    column_blocks = ceildiv_int(kept, block_b)
    return max(1, min(reduced, ceildiv_int(_LEADING_POLICY.target_blocks, column_blocks)))


def _make_down_rows_ops(op_kind: str, divisor: float, out_dtype: str, epilogue: str):
    """Create the per-op macros used by the down-rows reduction."""

    @T.macro
    def init(acc):
        if op_kind == "amax":
            T.fill(acc, -T.infinity("float32"))
        elif op_kind == "amin":
            T.fill(acc, T.infinity("float32"))
        else:
            T.fill(acc, 0.0)

    @T.macro
    def combine(acc, slot, value):
        if op_kind == "amax":
            acc[slot] = T.max(acc[slot], value)
        elif op_kind == "amin":
            acc[slot] = T.min(acc[slot], value)
        else:
            acc[slot] = acc[slot] + value

    @T.macro
    def finish(out_local, slot, accumulated):
        if divisor and epilogue == "sqrt":
            out_local[slot] = T.cast(T.sqrt(accumulated / divisor), out_dtype)
        elif divisor:
            out_local[slot] = T.cast(accumulated / divisor, out_dtype)
        elif epilogue == "sqrt":
            out_local[slot] = T.cast(T.sqrt(accumulated), out_dtype)
        else:
            out_local[slot] = T.cast(accumulated, out_dtype)

    return init, combine, finish


@functools.lru_cache(maxsize=32)
def _down_rows_kernel(
    A: int,
    B: int,
    op_kind: str,
    in_dtype: str,
    out_dtype: str,
    threads: int,
    splits: int,
    divisor: float,
    epilogue: str,
):
    """Build a reduce down the leading axis of an ``(A, B)`` buffer.

    One accumulator per output column, walked down the rows the block owns. Adjacent
    threads take adjacent columns, so every row of the walk is one coalesced pass and
    the buffer is read once in the layout it already has.

    The grid is ``(column blocks, splits)``: a leading-axis reduction has only as many
    output columns as the axes it keeps, and one block per column tile would leave
    the device running a handful of them.

    Args:
        A: Elements the reduction consumes per output column.
        B: Output columns.
        op_kind: One of ``sum`` / ``mean`` / ``amax`` / ``amin``.
        in_dtype: TileLang dtype string of the input.
        out_dtype: TileLang dtype string of the output. A split pass writes fp32
            partials whatever it read; the pass that finishes writes the declared dtype.
        threads: Threads per block.
        splits: Row slices, each its own block row. Above 1 the output is one row of
            partials per slice, for a second call with ``splits=1`` to finish.
        divisor: What the accumulator is divided by before the output cast, or 0 for
            none. Mean's divisor is the row count of the whole reduction, which a
            second pass over partials can no longer see.
        epilogue: ``"sqrt"`` applies a square root before the output cast, for a
            caller whose outer pass finishes a sum-of-squares; ``""`` for none.
    """
    block_b = threads * _LEADING_POLICY.cols_per_thread
    rows_per_split = ceildiv_int(A, splits)
    exact = B % block_b == 0
    rows_exact = rows_per_split * splits == A
    init_acc, combine, finish = _make_down_rows_ops(op_kind, divisor, out_dtype, epilogue)

    @tilelang.jit(out_idx=[1])
    def _func():
        @T.macro
        def walk_row(x, acc, row, pid_b):
            for j in T.Parallel(block_b):
                combine(acc, j, T.cast(x[row, pid_b * block_b + j], "float32"))

        @T.prim_func
        def main(
            x: T.Tensor[(A, B), in_dtype],
            out: T.Tensor[(splits * B,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(B, block_b), splits, threads=threads) as (pid_b, pid_a):
                acc = T.alloc_fragment((block_b,), "float32")
                out_local = T.alloc_fragment((block_b,), out_dtype)

                init_acc(acc)

                # A per-lane select for a bound the whole block shares scalarizes
                # the loop and costs it its vector loads.
                for step in T.serial(rows_per_split):
                    row = pid_a * rows_per_split + step
                    if exact and rows_exact:
                        walk_row(x, acc, row, pid_b)
                    elif exact:
                        with T.If(row < A):  # noqa: SIM117
                            with T.Then():
                                walk_row(x, acc, row, pid_b)
                    else:
                        for j in T.Parallel(block_b):
                            col = pid_b * block_b + j
                            val = T.if_then_else(
                                T.And(row < A, col < B),
                                T.cast(x[row, col], "float32"),
                                T.cast(identity_for(op_kind), "float32"),
                            )
                            combine(acc, j, val)

                if exact:
                    for j in T.Parallel(block_b):
                        finish(out_local, j, acc[j])
                    T.copy(out_local, out[pid_a * B + pid_b * block_b])
                else:
                    # Finish only stored lanes: a masked lane holds the identity
                    # (e.g. +inf), which an integer output dtype cannot take.
                    for j in T.Parallel(block_b):
                        # TileLang requires T.If/T.Then as nested context managers.
                        with T.If(pid_b * block_b + j < B):  # noqa: SIM117
                            with T.Then():
                                finish(out_local, j, acc[j])
                                out[pid_a * B + pid_b * block_b + j] = out_local[j]

        return main

    return _func


def reduce_down_rows(
    flat: torch.Tensor,
    op_kind: str,
    in_dtype: str,
    out_dtype: str,
    divisor: float,
    epilogue: str = "",
) -> torch.Tensor:
    """Reduce an ``(A, B)`` buffer down its rows, writing one *out_dtype* row.

    Splitting the reduced axis is what fills the grid, and each slice leaves an
    fp32 partial row; a second call over those rows finishes the op. The partials
    are a few thousand values against the millions the first pass reads, so the
    second call costs about nothing. ``divisor`` and ``epilogue`` apply only at
    the finishing call.
    """
    reduced, kept = flat.shape
    # An input smaller than one grid's worth of work cannot amortize the
    # extra pass a split costs; reduce it in a single call.
    grid_work = (
        _LEADING_POLICY.threads * _LEADING_POLICY.cols_per_thread * _LEADING_POLICY.target_blocks
    )
    if reduced * kept <= grid_work:
        splits = 1
    else:
        splits = _leading_row_splits(reduced, kept, _LEADING_POLICY.threads)
    if splits == 1:
        single = _down_rows_kernel(
            reduced,
            kept,
            op_kind,
            in_dtype,
            out_dtype,
            _LEADING_POLICY.threads,
            1,
            divisor,
            epilogue,
        )
        return single()(flat)
    partials = _down_rows_kernel(
        reduced,
        kept,
        op_kind,
        in_dtype,
        "float32",
        _LEADING_POLICY.threads,
        splits,
        0.0,
        "",
    )()(flat)
    # Partials are summed even for mean, whose divisor is the whole row count.
    finish = _down_rows_kernel(
        splits,
        kept,
        "sum" if op_kind == "mean" else op_kind,
        "float32",
        out_dtype,
        _LEADING_POLICY.threads,
        1,
        divisor,
        epilogue,
    )
    return finish()(partials.reshape(splits, kept))
