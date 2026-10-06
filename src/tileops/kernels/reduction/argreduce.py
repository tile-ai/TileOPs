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

from tileops.kernels.constants import STATIC_SHARED_BYTES
from tileops.kernels.kernel_base import Kernel, vector_aligned
from tileops.kernels.reduction._primitives import (
    FRAGMENT_ELEMS_PER_THREAD,
    ceildiv_int,
    restore_reduced,
    rows_for_axes,
    torch_dtype_nbytes,
)
from tileops.kernels.reduction.call_spec import ArgreduceCall, ArgreduceFwdInterface
from tileops.utils import WARP_LANES

__all__ = ["ArgreduceKernel", "ArgreduceSplitKernel", "ArgreduceStridedKernel"]

# Independent (key, index) slots a thread folds into, which every program's pair ops,
# launch shapes and merges assume.
_NUM_ACCUMULATORS = 4


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
        # Everything but the sign bit.
        magnitude = T.bitwise_and(bits, T.int32(0x7FFFFFFF))
        ordered = T.bitwise_xor(magnitude, sign) - sign
        # int32's largest, so no number outranks a NaN and two NaNs tie.
        return T.if_then_else(oriented != oriented, T.int32(0x7FFFFFFF), ordered)

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
        # Below every float's key, so it loses to any candidate: -inf keys to -0x7F800000.
        keys[slot] = T.int32(-(2**31))
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
            and N * span * elem_bytes <= STATIC_SHARED_BYTES
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


class _ArgreduceKernelBase(Kernel, ArgreduceFwdInterface):
    """What the three argmax/argmin programs share: the call they are built from and the
    shape of their result.

    Args:
        call: The call this implementation serves.
        config: Optional kernel configuration dict.
    """

    supported_archs: list[int] = [80, 86, 89, 90, 100]

    def __init__(self, call: ArgreduceCall, config: Optional[dict] = None):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        if call.op_kind not in ("argmax", "argmin"):
            raise ValueError(
                f"Unsupported op_kind '{call.op_kind}'. Expected one of ['argmax', 'argmin']."
            )
        if call.n <= 0:
            raise ValueError(
                "Reduction dimension is empty (N=0). "
                "argmax/argmin over an empty dimension is undefined."
            )
        self.M = call.m
        self.N = call.n
        self.op_kind = call.op_kind
        self.dtype = call.dtype
        self.reduce_axes = call.axes
        self.keepdim = call.keepdim
        self.kernel = self._program(call)
        self.init_config(config)

    def _program(self, call: ArgreduceCall) -> object:
        """The jit factory this implementation launches."""
        raise NotImplementedError

    def _knobs(self, config: dict) -> dict:
        """Keep only the knobs this kernel's program actually takes.

        Deriving the keys from the built kernel keeps a config space from naming a knob
        the kernel would reject.
        """
        parameters = self.kernel.signature.parameters
        return {name: value for name, value in config.items() if name in parameters}

    def _candidates(self) -> list[dict]:
        raise NotImplementedError

    @property
    def autotune_configs(self) -> list[dict]:
        # Tuning may not come back worse than not tuning, so the default is
        # always among the candidates.
        default = self.default_config
        ranked = [self._knobs(candidate) for candidate in self._candidates()]
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
        x = vector_aligned(x)
        in_shape = tuple(x.shape)
        if self.M == 0:
            empty = torch.empty((0,), dtype=torch.int64, device=x.device)
            return restore_reduced(empty, in_shape, self.reduce_axes, self.keepdim)
        y = self._argreduce_rows(rows_for_axes(x, self.reduce_axes))
        return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)

    def _argreduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Run the program over an already-laid-out buffer."""
        raise NotImplementedError


class ArgreduceKernel(_ArgreduceKernelBase):
    """Streaming argmax/argmin of contiguous rows, each held by one warp or one block.

    A row of 4096 or more takes a whole block, so it fits in fragments; a shorter one
    takes a warp's lanes, ``block_m`` rows a block. Rows are the reduced axes moved last.
    """

    general = True

    def _program(self, call: ArgreduceCall) -> object:
        self._cta = call.n >= 4096
        if self._cta:
            return _argreduce_cta_kernel(call.m, call.n, call.op_kind, self.dtype_str)
        return _argreduce_warp_kernel(call.m, call.n, call.op_kind, self.dtype_str)

    @property
    def default_config(self) -> dict:
        if self._cta:
            # Enough threads that the row fits in fragments.
            wanted = ceildiv_int(self.N, FRAGMENT_ELEMS_PER_THREAD)
            threads = min(1024, max(256, 1 << max(0, wanted - 1).bit_length()))
            return self._knobs({"block_m": 1, "threads": threads})
        lanes = _lanes_per_row(self.N)
        target_threads = 256 if self.M >= 8 else max(32, self.M * lanes)
        block_m = max(1, target_threads // lanes)
        return self._knobs({"block_m": block_m, "threads": block_m * lanes})

    def _candidates(self) -> list[dict]:
        if self._cta:
            return [{"threads": t} for t in (128, 256, 512, 1024)]
        lanes = _lanes_per_row(self.N)
        candidates = []
        for target_threads in (64, 128, 256, 512):
            block_m = max(1, target_threads // lanes)
            candidates.append({"block_m": block_m, "threads": block_m * lanes})
        return candidates

    def _argreduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        if self._cta:
            return self.kernel(self.config["threads"])(x)
        return self.kernel(self.config.get("block_m", 1), self.config["threads"])(x)


class ArgreduceSplitKernel(_ArgreduceKernelBase):
    """Argmax/argmin of a few long rows, each split across blocks and merged in a second launch.

    Splitting trades one block's serial scan for a second pass over the partials. It wins
    while the row is long enough for the scan to dominate that pass and the rows alone
    leave the device underused. The surface this approximates is not monotonic — 4 rows of
    8192 want the split while 16 do not — so the region is deliberately the conservative
    one: it declines a win on a few mid-sized shapes rather than taking a loss on short
    ones, which is where ``dim=None`` and small tensors land. The blocks per row are tuned.
    """

    @classmethod
    def applies(cls, call: ArgreduceCall) -> bool:
        # A row shorter than 32768 is a handful of passes, so splitting cannot save more
        # than the second pass costs; past 512 rows the blocks already queue, and
        # splitting only adds the final pass.
        return call.n >= 32768 and call.m < 512

    def _program(self, call: ArgreduceCall) -> object:
        return _argreduce_multicta_partial_kernel(call.m, call.n, call.op_kind, self.dtype_str)

    @property
    def default_config(self) -> dict:
        # Only a default: the best split moves with the shape by more than an order of
        # magnitude.
        split = min(16, self._max_split())
        return self._knobs({"block_m": 1, "threads": 256, "ctas_per_row": split})

    def _candidates(self) -> list[dict]:
        ceiling = self._max_split()
        splits = sorted({c for c in (4, 8, 16, 32, 64) if c <= ceiling} or {1})
        return [{"threads": t, "ctas_per_row": c} for t in (128, 256, 512) for c in splits]

    def _max_split(self) -> int:
        """The most blocks a row may split into: a chunk shorter than 512 cannot
        amortize its share of the final pass."""
        return max(1, self.N // 512)

    def _argreduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        ctas_per_row = self.config.get("ctas_per_row", 1)
        partial_keys, partial_indices = self.kernel(self.config["threads"], ctas_per_row)(x)
        final = _argreduce_multicta_final_kernel(self.M, self.N, self.op_kind, ctas_per_row)
        return final()(partial_keys, partial_indices)


class ArgreduceStridedKernel(_ArgreduceKernelBase):
    """Argmax/argmin along a short strided axis, read in place without a transpose.

    A thread takes an output element and walks the axis, which reads the original buffer
    coalesced. That pays only while the walk is short.
    """

    @classmethod
    def applies(cls, call: ArgreduceCall) -> bool:
        # A thread walks the whole axis, which pays only while the walk is short; it loses
        # from 32 positions up.
        return call.inner_stride > 1 and call.n <= 16

    def _program(self, call: ArgreduceCall) -> object:
        return _argreduce_output_kernel(
            call.m, call.n, call.inner_stride, call.op_kind, self.dtype_str
        )

    @property
    def default_config(self) -> dict:
        # Two outputs a thread over 256 threads beat four over 128 while every block is
        # resident at once and the axis has four positions or more, or once each
        # output's walk reads 24 bytes or more.
        block_m, wide = 512, 256
        props = torch.cuda.get_device_properties(self.device_index)
        resident = props.multi_processor_count * (props.max_threads_per_multi_processor // wide)
        one_wave = ceildiv_int(self.M, block_m) <= resident
        long_walk = self.N * self.dtype.itemsize >= 24
        threads = wide if (one_wave and self.N >= 4) or long_walk else 128
        return self._knobs({"block_m": block_m, "threads": threads})

    def _candidates(self) -> list[dict]:
        return [
            {"block_m": threads * per_thread, "threads": threads}
            for threads in (128, 256, 512)
            for per_thread in (1, 2, 4)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Walk the strided axis of *x* in its own layout; see the base ``forward``."""
        self._require_cuda(x=x)
        x = vector_aligned(x)
        y = self.kernel(self.config["block_m"], self.config["threads"])(x.reshape(-1))
        return restore_reduced(y, tuple(x.shape), self.reduce_axes, self.keepdim)
