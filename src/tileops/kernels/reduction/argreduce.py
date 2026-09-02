"""Streaming argmax and argmin kernels.

Values and indices are reduced together so first-index and NaN semantics are
preserved without materializing an input tile. Launch geometry adapts to the
row length; input layout remains the responsibility of the reduction Op layer.
"""

import functools
from typing import NamedTuple, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    FRAGMENT_ELEMS_PER_THREAD,
    SHARED_MEMORY_BUDGET_BYTES,
    ceildiv_int,
    restore_reduced,
    rows_for_axes,
    torch_dtype_nbytes,
)
from tileops.utils import WARP_LANES

__all__ = ["ArgreduceKernel"]

_ARGREDUCE_KINDS = {"argmax", "argmin"}
_NUM_ACCUMULATORS = 4

# Magnitude bits of a float32 pattern -- everything but the sign; see _ordering_key.
_KEY_MAGNITUDE = 0x7FFFFFFF

# The key every NaN takes: int32's largest, so no number outranks one and two NaNs tie.
_KEY_NAN = 0x7FFFFFFF

# Below every float's key, so it loses to any candidate: -inf keys to -0x7F800000.
_KEY_IDENTITY = -(2**31)
# A row shorter than this is a handful of passes; splitting cannot save more
# than the second pass costs.
_SPLIT_MIN_N = 32768
# Above this many rows the blocks already queue, and splitting only adds the
# final pass.
_ROWS_SATURATED = 512
# A chunk shorter than this cannot amortize its share of the final pass.
_MIN_CHUNK = 512
# Output-parallel gives a thread the whole axis to walk, so it pays only while
# that walk is short; it loses from N=32 up.
_STRIDED_AXIS_MAX_N = 16


def _row_split_candidates(N: int) -> list[int]:
    """Splits worth ranking for a row of *N*, coarsest first."""
    ceiling = max(1, N // _MIN_CHUNK)
    return sorted({c for c in (4, 8, 16, 32, 64) if c <= ceiling} or {1})


def _splits_row(M: int, N: int) -> bool:
    """Whether a row is worth splitting across blocks.

    Splitting trades one block's serial scan for a second pass over the
    partials. It wins while the row is long enough for the scan to dominate
    that pass and the rows alone leave the device underused.

    The surface this approximates is not monotonic — 4 rows of 8192 want the
    split while 16 do not — so the rule is deliberately the conservative one:
    it declines a win on a few mid-sized shapes rather than taking a loss on
    short ones, which is where `dim=None` and small tensors land.
    """
    return N >= _SPLIT_MIN_N and M < _ROWS_SATURATED


def _plan_row_split(M: int, N: int) -> int:
    """Chunks per row when untuned; 1 means the row stays whole.

    Only a default — the split is a tuning parameter, and the best value moves
    with the shape by more than an order of magnitude.
    """
    if not _splits_row(M, N):
        return 1
    return max(1, min(16, N // _MIN_CHUNK))


def _lanes_per_row(n: int) -> int:
    lanes = 1
    while lanes < min(n, WARP_LANES):
        lanes *= 2
    return lanes


def _ordering_key(op_kind: str):
    """Build the macro turning one element into the int32 key the reduction ranks by.

    Ranking bit patterns collapses PyTorch's ordering -- NaN over every number, two NaNs
    equal, ties to the lower index -- into one integer compare. Spelled out per element
    it costs a dozen instructions, and this kernel is short of its bandwidth by about
    the same factor. Callers bind the result to a local before ranking it: a macro
    argument is substituted at every mention, so passing the call directly would
    recompute the key once per comparison the merge makes.

    Negating first turns argmin into argmax and leaves a NaN a NaN, so NaN outranks in
    both. The key is the magnitude bits, negated in two's complement when the sign bit
    is set: that orders the whole line, and it sends both zeros to 0 without a test,
    which is what lets the index break their tie the way PyTorch does. A positive NaN's
    magnitude already outranks +inf's; the negated ones do not, so NaN is answered
    outright.
    """
    negate = op_kind == "argmin"

    @T.macro
    def key_of(value):
        cast = T.cast(value, "float32")
        oriented = -cast if negate else cast
        bits = T.reinterpret(oriented, "int32")
        sign = bits >> 31
        magnitude = T.bitwise_and(bits, T.int32(_KEY_MAGNITUDE))
        ordered = T.bitwise_xor(magnitude, sign) - sign
        return T.if_then_else(oriented != oriented, T.int32(_KEY_NAN), ordered)

    return key_of


class _PairOps(NamedTuple):
    set_identity: object
    init_accumulators: object
    update: object
    advance: object
    merge_accumulators: object
    warp_reduce: object


def _make_pair_ops(op_kind: str, n: int):
    """Create argreduce-local pair operations shared by all launch paths."""

    @T.macro
    def set_identity(keys, indices, slot):
        keys[slot] = T.int32(_KEY_IDENTITY)
        indices[slot] = T.int32(n)

    @T.macro
    def init_accumulators(keys, indices):
        for accumulator in T.serial(_NUM_ACCUMULATORS):
            set_identity(keys, indices, accumulator)

    @T.macro
    def update(keys, indices, slot, candidate_key, candidate_index):
        """Merge a candidate into a slot: higher key wins, an equal key breaks low."""
        if candidate_key > keys[slot] or (
            candidate_key == keys[slot] and candidate_index < indices[slot]
        ):
            keys[slot] = candidate_key
            indices[slot] = T.cast(candidate_index, "int32")

    @T.macro
    def advance(keys, indices, slot, candidate_key, candidate_index):
        """Merge a candidate reached after everything in the slot: one key comparison.

        A streaming loop hands a slot ascending indices, so an equal key cannot win.
        """
        if candidate_key > keys[slot]:
            keys[slot] = candidate_key
            indices[slot] = T.cast(candidate_index, "int32")

    @T.macro
    def merge_accumulators(keys, indices, best_key, best_index):
        best_key[0] = keys[0]
        best_index[0] = indices[0]
        for accumulator in T.serial(1, _NUM_ACCUMULATORS):
            update(best_key, best_index, 0, keys[accumulator], indices[accumulator])

    @T.macro
    def warp_reduce(best_key, best_index, stages, width):
        for stage in T.serial(stages):
            mask = T.int32(width // 2) >> stage
            update(
                best_key,
                best_index,
                0,
                T.shfl_xor(best_key[0], mask, width=width),
                T.shfl_xor(best_index[0], mask, width=width),
            )

    return _PairOps(
        set_identity=set_identity,
        init_accumulators=init_accumulators,
        update=update,
        advance=advance,
        merge_accumulators=merge_accumulators,
        warp_reduce=warp_reduce,
    )


def _make_block_reduce(
    ops: _PairOps,
    num_warps: int,
):
    """Create the register-to-block pair reduction used by CTA kernels."""

    @T.macro
    def block_reduce(
        keys,
        indices,
        best_key,
        best_index,
        warp_keys,
        warp_indices,
        tx,
    ):
        lane = tx % WARP_LANES
        warp = tx // WARP_LANES

        ops.merge_accumulators(keys, indices, best_key, best_index)
        ops.warp_reduce(best_key, best_index, WARP_LANES.bit_length() - 1, WARP_LANES)

        if lane == 0:
            warp_keys[warp] = best_key[0]
            warp_indices[warp] = best_index[0]
        T.sync_threads()

        ops.set_identity(best_key, best_index, 0)
        if lane < num_warps:
            best_key[0] = warp_keys[lane]
            best_index[0] = warp_indices[lane]

        if warp == 0:
            ops.warp_reduce(best_key, best_index, WARP_LANES.bit_length() - 1, WARP_LANES)

    return block_reduce


@functools.lru_cache(maxsize=64)
def _argreduce_warp_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build the subgroup-per-row kernel used for ordinary row lengths."""
    lanes = _lanes_per_row(N)
    items_per_iteration = lanes * _NUM_ACCUMULATORS
    iterations = (N + items_per_iteration - 1) // items_per_iteration
    log_lanes = lanes.bit_length() - 1

    @tilelang.jit(out_idx=[1])
    def _func(block_m: int, threads: int):
        ops = _make_pair_ops(op_kind, N)
        key_of = _ordering_key(op_kind)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid:
                tx = T.get_thread_binding()
                row = pid * block_m + tx // lanes
                lane = tx % lanes

                keys = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                indices = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                candidate = T.alloc_local((1,), "int32")
                best_key = T.alloc_local((1,), "int32")
                best_index = T.alloc_local((1,), "int32")
                ops.init_accumulators(keys, indices)

                for iteration in T.serial(iterations):
                    for accumulator in T.serial(_NUM_ACCUMULATORS):
                        index = iteration * items_per_iteration + accumulator * lanes + lane
                        if row < M and index < N:
                            candidate[0] = key_of(x[row, index])
                            ops.advance(keys, indices, accumulator, candidate[0], index)

                ops.merge_accumulators(keys, indices, best_key, best_index)
                ops.warp_reduce(best_key, best_index, log_lanes, lanes)

                if row < M and lane == 0:
                    out[row] = T.cast(best_index[0], "int64")

        return main

    return _func


@functools.lru_cache(maxsize=64)
def _argreduce_output_kernel(
    M: int,
    N: int,
    inner_stride: int,
    op_kind: str,
    dtype: str,
):
    """Build the output-parallel kernel for a contiguous non-last-axis reduction.

    The reduction axis is strided here and the output axis is contiguous, so a
    thread takes output elements and walks the axis: adjacent threads then
    read adjacent addresses. Transposing into a last-axis layout instead copies
    the whole tensor and hands a row of ``N`` elements to a block built for long
    rows — on the manifest's 3d workload the copy alone is nearly half the time.

    A block covers ``block_m`` consecutive outputs over ``threads`` threads, and a
    thread takes every ``threads``-th of them so that neighbours stay neighbours in
    both the store and the walk. Where the whole span is one contiguous run of each
    axis position, it is staged in shared memory first and the walk reads from
    there, which is what lets the global read widen from one element per thread to a
    vector.
    """
    elem_bytes = torch_dtype_nbytes(dtype)

    @tilelang.jit(out_idx=[1])
    def _func(block_m: int, threads: int):
        ops = _make_pair_ops(op_kind, N)
        key_of = _ordering_key(op_kind)
        per_thread = max(1, block_m // threads)
        span = per_thread * threads
        # Staging needs one contiguous run, no masked tail, and a tile within budget.
        stage = (
            span == block_m
            and M % span == 0
            and inner_stride % span == 0
            and N * span * elem_bytes <= SHARED_MEMORY_BUDGET_BYTES
        )

        @T.prim_func
        def main(
            x: T.Tensor[(M * N,), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, span), threads=threads) as pid:
                tx = T.get_thread_binding()
                tile = T.alloc_shared((N, span if stage else 1), dtype)
                keys = T.alloc_local((per_thread,), "int32")
                indices = T.alloc_local((per_thread,), "int32")
                candidate = T.alloc_local((1,), "int32")

                if stage:
                    outer = (pid * span) // inner_stride
                    inner = (pid * span) % inner_stride
                    for index, column in T.Parallel(N, span):
                        tile[index, column] = x[
                            outer * N * inner_stride + index * inner_stride + inner + column
                        ]
                    T.sync_threads()

                for slot in T.serial(per_thread):
                    ops.set_identity(keys, indices, slot)

                for index in T.serial(N):
                    for slot in T.serial(per_thread):
                        column = slot * threads + tx
                        if stage:
                            candidate[0] = key_of(tile[index, column])
                            ops.advance(keys, indices, slot, candidate[0], index)
                        else:
                            row = pid * span + column
                            if row < M:
                                offset = (
                                    (row // inner_stride) * N * inner_stride
                                    + index * inner_stride
                                    + row % inner_stride
                                )
                                candidate[0] = key_of(x[offset])
                                ops.advance(keys, indices, slot, candidate[0], index)

                for slot in T.serial(per_thread):
                    row = pid * span + slot * threads + tx
                    if row < M:
                        out[row] = T.cast(indices[slot], "int64")

        return main

    return _func


@functools.lru_cache(maxsize=64)
def _argreduce_cta_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build the block-per-row kernel used for long rows.

    A row the block can hold in registers is read once into a fragment, and the column
    index each key belongs to is the loop variable over that fragment, so nothing has to
    carry it. That reads the row a vector at a time and keeps it there, where walking it
    in global reads one element per thread. The wider the row, the more that is worth.

    A row too wide for that is walked in global instead, one element per thread per
    access.
    """

    @tilelang.jit(out_idx=[1])
    def _func(threads: int):
        num_warps = threads // WARP_LANES
        iterations = (N + threads * _NUM_ACCUMULATORS - 1) // (threads * _NUM_ACCUMULATORS)
        held = threads * FRAGMENT_ELEMS_PER_THREAD >= N
        ops = _make_pair_ops(op_kind, N)
        key_of = _ordering_key(op_kind)
        block_reduce = _make_block_reduce(ops, num_warps)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                warp_keys = T.alloc_shared((num_warps,), "int32")
                warp_indices = T.alloc_shared((num_warps,), "int32")
                row_frag = T.alloc_fragment((1, N if held else 1), dtype)
                # Four slots either way, so both paths share the block reduction below.
                keys = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                indices = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                candidate = T.alloc_local((1,), "int32")
                best_key = T.alloc_local((1,), "int32")
                best_index = T.alloc_local((1,), "int32")
                ops.init_accumulators(keys, indices)

                if held:
                    T.copy(x[row : row + 1, :], row_frag)
                    # The fragment order is unstated, so this merge compares indices.
                    for _, column in T.Parallel(1, N):
                        candidate[0] = key_of(row_frag[0, column])
                        ops.update(keys, indices, 0, candidate[0], column)
                else:
                    for iteration in T.serial(iterations):
                        for accumulator in T.serial(_NUM_ACCUMULATORS):
                            index = (
                                iteration * threads * _NUM_ACCUMULATORS + accumulator * threads + tx
                            )
                            if index < N:
                                candidate[0] = key_of(x[row, index])
                                ops.advance(keys, indices, accumulator, candidate[0], index)

                block_reduce(
                    keys,
                    indices,
                    best_key,
                    best_index,
                    warp_keys,
                    warp_indices,
                    tx,
                )
                if tx == 0:
                    out[row] = T.cast(best_index[0], "int64")

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _argreduce_multicta_partial_kernel(
    M: int,
    N: int,
    op_kind: str,
    dtype: str,
):
    """Build the partial stage for rows split across multiple blocks.

    ``ctas_per_row`` is a tuning parameter: the trade between the block's serial
    scan and a wider final pass moves with the shape. The tuner measures this
    stage alone; the final pass grows slowly with the split, so its ranking can
    differ from the pair's at the margin.
    """

    @tilelang.jit(out_idx=[1, 2])
    def _func(threads: int, ctas_per_row: int):
        chunk_size = (N + ctas_per_row - 1) // ctas_per_row
        num_partials = M * ctas_per_row
        num_warps = threads // WARP_LANES
        iterations = (chunk_size + threads * _NUM_ACCUMULATORS - 1) // (threads * _NUM_ACCUMULATORS)
        ops = _make_pair_ops(op_kind, N)
        key_of = _ordering_key(op_kind)
        block_reduce = _make_block_reduce(ops, num_warps)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            partial_keys: T.Tensor[(num_partials,), "int32"],  # noqa: F821
            partial_indices: T.Tensor[(num_partials,), "int32"],  # noqa: F821
        ):
            with T.Kernel(num_partials, threads=threads) as partial:
                tx = T.get_thread_binding()
                row = partial // ctas_per_row
                split = partial % ctas_per_row
                chunk_start = split * chunk_size
                chunk_end = T.min(chunk_start + chunk_size, N)

                warp_keys = T.alloc_shared((num_warps,), "int32")
                warp_indices = T.alloc_shared((num_warps,), "int32")
                keys = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                indices = T.alloc_local((_NUM_ACCUMULATORS,), "int32")
                candidate = T.alloc_local((1,), "int32")
                best_key = T.alloc_local((1,), "int32")
                best_index = T.alloc_local((1,), "int32")
                ops.init_accumulators(keys, indices)

                for iteration in T.serial(iterations):
                    for accumulator in T.serial(_NUM_ACCUMULATORS):
                        index = (
                            chunk_start
                            + iteration * threads * _NUM_ACCUMULATORS
                            + accumulator * threads
                            + tx
                        )
                        if index < chunk_end:
                            candidate[0] = key_of(x[row, index])
                            ops.advance(keys, indices, accumulator, candidate[0], index)

                block_reduce(
                    keys,
                    indices,
                    best_key,
                    best_index,
                    warp_keys,
                    warp_indices,
                    tx,
                )
                if tx == 0:
                    partial_keys[partial] = best_key[0]
                    partial_indices[partial] = best_index[0]

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _argreduce_multicta_final_kernel(
    M: int,
    N: int,
    op_kind: str,
    ctas_per_row: int,
):
    """Build the final reduction over per-row block partials."""
    num_partials = M * ctas_per_row
    rows_per_block = 8
    threads = rows_per_block * WARP_LANES

    # A lane walks its share of the row's partials, so the split is free to be
    # wider than a warp; reading one partial per lane would drop the rest.
    partials_per_lane = ceildiv_int(ctas_per_row, WARP_LANES)

    @tilelang.jit(out_idx=[2])
    def _func():
        ops = _make_pair_ops(op_kind, N)

        @T.prim_func
        def main(
            partial_keys: T.Tensor[(num_partials,), "int32"],  # noqa: F821
            partial_indices: T.Tensor[(num_partials,), "int32"],  # noqa: F821
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, rows_per_block), threads=threads) as pid:
                tx = T.get_thread_binding()
                row = pid * rows_per_block + tx // WARP_LANES
                lane = tx % WARP_LANES
                best_key = T.alloc_local((1,), "int32")
                best_index = T.alloc_local((1,), "int32")
                ops.set_identity(best_key, best_index, 0)

                for step in T.serial(partials_per_lane):
                    partial = step * WARP_LANES + lane
                    if row < M and partial < ctas_per_row:
                        ops.update(
                            best_key,
                            best_index,
                            0,
                            partial_keys[row * ctas_per_row + partial],
                            partial_indices[row * ctas_per_row + partial],
                        )

                ops.warp_reduce(best_key, best_index, WARP_LANES.bit_length() - 1, WARP_LANES)
                if row < M and lane == 0:
                    out[row] = T.cast(best_index[0], "int64")

        return main

    return _func


class ArgreduceKernel(Kernel):
    """Adaptive streaming argmax/argmin over contiguous rows.

    ``forward`` takes the tensor the op declares and reduces *reduce_axes* of it. Which of
    the four layouts runs is decided here, from the row count, the axis length and the
    axis's stride — so an op hands over the declared tensor and asks for nothing else.

    Args:
        M: Rows the reduction leaves.
        N: Length of the axis it reduces.
        op_kind: One of "argmax", "argmin".
        dtype: Input data type.
        reduce_axes: Non-negative axis indices, ascending, that the reduction runs over.
        keepdim: Whether a reduced axis stays as a length-1 axis.
        inner_stride: Elements between two neighbours along the reduced axis. Greater than
            one means the axis is not the contiguous one, which the strided layout reads
            without transposing.
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
        device_index: The device the input lives on. None of the four layouts plans against
            shared memory, so this is here for the architecture check alone.
    """

    supported_archs: list[int] = [80, 86, 89, 90, 100]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        reduce_axes: "tuple[int, ...]",
        keepdim: bool = False,
        inner_stride: int = 1,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        if op_kind not in _ARGREDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_ARGREDUCE_KINDS)}."
            )
        if N <= 0:
            raise ValueError(
                "Reduction dimension is empty (N=0). "
                "argmax/argmin over an empty dimension is undefined."
            )

        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.reduce_axes = tuple(reduce_axes)
        self.keepdim = keepdim
        self.inner_stride = inner_stride

        if inner_stride > 1 and N <= _STRIDED_AXIS_MAX_N:
            # A short strided axis: a thread takes an output element and walks
            # the axis, which reads coalesced and skips the transpose.
            self.strategy = "output"
            self.kernel = _argreduce_output_kernel(M, N, inner_stride, op_kind, self.dtype_str)
        elif _splits_row(M, N):
            self.strategy = "multi_cta"
            self.kernel = _argreduce_multicta_partial_kernel(M, N, op_kind, self.dtype_str)
        elif N >= 4096:
            self.strategy = "cta"
            self.kernel = _argreduce_cta_kernel(M, N, op_kind, self.dtype_str)
        else:
            self.strategy = "warp"
            self.kernel = _argreduce_warp_kernel(M, N, op_kind, self.dtype_str)
        self.init_config(config, tune)

    def _knobs(self, config: dict) -> dict:
        """Keep only the knobs this strategy's kernel actually takes.

        ``block_m`` packs several rows into one block, which only the warp
        layout does — the other two give a row its own block or its own group of
        them. Deriving the keys from the built kernel keeps a config space from
        naming a knob the kernel would reject.
        """
        parameters = self.kernel.signature.parameters
        return {name: value for name, value in config.items() if name in parameters}

    @property
    def default_config(self) -> dict:
        lanes = _lanes_per_row(self.N)
        target_threads = 256 if self.M >= 8 else max(32, self.M * lanes)
        block_m = max(1, target_threads // lanes)
        if self.strategy == "cta":
            # Enough threads that the row fits in fragments.
            wanted = ceildiv_int(self.N, FRAGMENT_ELEMS_PER_THREAD)
            threads = min(1024, max(256, 1 << max(0, wanted - 1).bit_length()))
            block_m = 1
        elif self.strategy == "multi_cta":
            block_m, threads = 1, 256
        elif self.strategy == "output":
            # Four outputs per thread: the span the staged read wants.
            block_m, threads = 512, 128
        else:
            threads = block_m * lanes
        return self._knobs(
            {
                "block_m": block_m,
                "threads": threads,
                "ctas_per_row": _plan_row_split(self.M, self.N),
            }
        )

    @property
    def autotune_configs(self) -> list[dict]:
        if self.strategy == "multi_cta":
            candidates = [
                {"threads": t, "ctas_per_row": c}
                for t in (128, 256, 512)
                for c in _row_split_candidates(self.N)
            ]
        elif self.strategy == "cta":
            candidates = [{"threads": t} for t in (128, 256, 512, 1024)]
        elif self.strategy == "output":
            candidates = [
                {"block_m": threads * per_thread, "threads": threads}
                for threads in (128, 256, 512)
                for per_thread in (1, 2, 4)
            ]
        else:
            lanes = _lanes_per_row(self.N)
            candidates = []
            for target_threads in (64, 128, 256, 512):
                block_m = max(1, target_threads // lanes)
                candidates.append({"block_m": block_m, "threads": block_m * lanes})
        # Tuning may not come back worse than not tuning, so the default is
        # always among the candidates.
        default = self.default_config
        ranked = [self._knobs(candidate) for candidate in candidates]
        if default not in ranked:
            ranked.append(default)
        return ranked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the index of the extremum along *reduce_axes* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device.

        Returns:
            Int64 indices into the reduced axis.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        if self.M == 0:
            empty = torch.empty((0,), dtype=torch.int64, device=x.device)
            return restore_reduced(empty, in_shape, self.reduce_axes, self.keepdim)
        # The strided layout walks the original buffer, which is why it skips the
        # transpose the other three need.
        buffer = x.reshape(-1) if self.strategy == "output" else rows_for_axes(x, self.reduce_axes)
        y = self._argreduce_rows(buffer)
        return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)

    def _argreduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Run the selected layout over an already-laid-out buffer."""
        block_m = self.config.get("block_m", 1)
        threads = self.config["threads"]
        if self.strategy == "output":
            return self.kernel(block_m, threads)(x)
        if self.strategy == "cta":
            return self.kernel(threads)(x)
        if self.strategy == "multi_cta":
            ctas_per_row = self.config.get("ctas_per_row", 1)
            partial_keys, partial_indices = self.kernel(threads, ctas_per_row)(x)
            final = _argreduce_multicta_final_kernel(self.M, self.N, self.op_kind, ctas_per_row)
            return final()(partial_keys, partial_indices)
        return self.kernel(block_m, threads)(x)
