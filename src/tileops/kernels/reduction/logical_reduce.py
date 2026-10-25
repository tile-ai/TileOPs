"""Logical reduce kernels (any, all, count_nonzero) using TileLang.

Each row is folded into registers as it is read, one vector access of at most 16 bytes
per lane per step. A count sums the nonzero elements; any and all or together a flag that is
set on an element that decides them, a nonzero one for any and a zero one for all.

The input is read at its own bytes. Where the row allows, bool and 8-bit integers go four
to a 32-bit word and float16, bfloat16 and float32 one 16- or 32-bit lane to an element, and
a lane's truth is a bit test; a complex element is its two real parts; every other dtype is
read as declared.
"""

import functools
import math
from math import prod
from typing import Callable, ClassVar, Mapping, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops.kernels.reduction._primitives import (
    FP32_EXACT_INT_LIMIT,
    ceildiv_int,
    down_rows_launcher,
    down_rows_splits,
    edge_axis_split,
    restore_reduced,
    rows_for_axes,
    tune_by_forward,
)
from tileops.kernels.reduction.call_spec import (
    CountNonzeroFwdInterface,
    LogicalReduceCall,
    LogicalReduceFwdInterface,
)
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = [
    "CountNonzeroEdgeTwoPassKernel",
    "LogicalReduceEdgeFusedKernel",
    "LogicalReduceEdgeTwoPassKernel",
    "LogicalReduceKernel",
]

# The scalar dtype the prim_func declares for each input dtype, and how many of those
# scalars make one element. bool is one byte holding 0 or 1, so int8 reinterprets it; a
# complex element is its real and imaginary parts. Every other dtype is declared as is.
_SCALAR_VIEWS = {
    torch.bool: (torch.int8, 1),
    torch.complex64: (torch.float32, 2),
    torch.complex128: (torch.float64, 2),
}


def _lane_masks(lane_bits: int) -> "tuple[int, int, int]":
    """``(ones, low, high)`` of a 32-bit word of *lane_bits*-bit lanes: each lane's lowest bit,
    its bits below the top one, and its top bit."""
    lanes = 32 // lane_bits
    ones = sum(1 << (i * lane_bits) for i in range(lanes))
    high = ones << (lane_bits - 1)
    return ones, high - ones, high


def _fold_vector(unit_bytes: int, row_units: int, components: int, address: int = 0) -> int:
    """Units one vector access of the fold reads, for rows of *row_units* units.

    The widest power of two within one 16-byte access that holds whole elements and
    keeps every access aligned: it divides the row, and the data at *address* starts
    on its boundary. One element is always possible.
    """
    vec = VECTOR_ACCESS_BYTES // unit_bytes
    while vec > components and (row_units % vec or address % (vec * unit_bytes)):
        vec //= 2
    return max(vec, components)


def _fold_threads(row_units: int, vec: int) -> int:
    """The block width the fold runs a row of *row_units* units at."""
    # A lane folds one vector access per step, and a block is sized so each lane folds at
    # least this many of them across a row: fewer leaves lanes that load nothing while the
    # block still pays for its reduction.
    fold_vectors_per_lane = 2
    fold_max_threads = 1024  # CUDA's cap on threads per block
    lanes = max(ceildiv_int(row_units, vec) // fold_vectors_per_lane, 1)
    lanes = 1 << (lanes.bit_length() - 1)
    return max(WARP_LANES, min(lanes, fold_max_threads))


@functools.lru_cache(maxsize=32)
def _logical_fold_kernel(
    lead: int,
    rows: int,
    cols: int,
    op_kind: str,
    unit_dtype: str,
    components: int,
    pack: int,
    vec: int,
    out_dtype: str,
    stream: bool,
    words: bool = False,
    magnitude: bool = False,
):
    """Build an any/all/count_nonzero kernel that folds each row into registers.

    One block per row; row ``r`` is the ``lead`` contiguous runs ``x[l, r, :]`` of
    ``cols`` elements.

    Args:
        lead: Runs each row is made of, walked serially by its block.
        rows: Output rows.
        cols: Elements in each run.
        op_kind: One of "any", "all", "count_nonzero".
        unit_dtype: TileLang dtype string of the units the input is read as.
        components: Units per element: 2 for a complex element's parts, else 1.
        pack: Elements per unit: the lanes of a 32-bit word where *words*, else 1.
        vec: Units per vector access, a multiple of ``components`` dividing the run.
        out_dtype: TileLang dtype string of the output.
        stream: Whether a full-width vector access loads evict-first.
        words: Whether units are 32-bit words of ``pack`` lanes, tested by bit operations.
        magnitude: Whether a lane or element is a float read as bits, whose sign bit does
            not make it nonzero.

    Returns:
        A TileLang JIT-compiled kernel factory accepting (threads).
    """
    run_units = cols * components // pack
    counts = op_kind == "count_nonzero"
    count_dtype = "int32" if lead * cols < 1 << 31 else "int64"
    acc_dtype = count_dtype if counts else "uint32"
    lane_ones, lane_low, lane_high = _lane_masks(32 // pack) if words else (1, 0, 0)
    # What a vector past the end of the run holds: nothing any or a count takes, and
    # no zero for all.
    pad = (lane_ones if words else 1) if op_kind == "all" else 0

    @tilelang.jit(out_idx=[1], compile_flags=csrc_include("streaming_load.h"))
    def _func(threads):
        def _fold_term(op_kind: str, held, e, components: int, pack: int):
            """What element (or word) *e* of a lane's vector adds to its accumulator."""
            if words:
                word = held[e]
                low = T.Cast("uint32", lane_low)
                high = T.Cast("uint32", lane_high)
                if magnitude:
                    # A float lane's sign bit does not make it nonzero.
                    word = word & low
                if op_kind == "count_nonzero":
                    # Nonzero lanes: ``(b & low) + low`` sets a lane's top bit exactly when its
                    # lower bits are not all clear, never carrying into the next lane; or-ing in
                    # ``b`` adds the top bit itself.
                    return T.popcount((((word & low) + low) | word) & high)
                if op_kind == "any":
                    return word
                # Nonzero exactly when some lane is zero: ``(w - ones) & ~w & high``, a zero lane
                # borrows into its own top bit. ``w ^ ~0`` stands for ``~w``, which CUDA's vector
                # types do not define.
                return (
                    (word - T.Cast("uint32", lane_ones))
                    & (word ^ T.Cast("uint32", 0xFFFFFFFF))
                    & high
                )
            zero = T.cast(0, held.dtype)
            if magnitude:
                # A 16-bit float read as int16 is nonzero when its bits past the sign are.
                # CUDA's half and bfloat16 ``!=`` is ordered and would call NaN zero.
                nonzero = (held[e] & T.cast(0x7FFF, held.dtype)) != zero
            else:
                # An element is nonzero when any of its scalars compares unequal to zero in
                # its own dtype, so -0.0 is zero and NaN is not, as in torch.
                nonzero = held[e * components] != zero
                for q in range(1, components):
                    nonzero = T.Or(nonzero, held[e * components + q] != zero)
            if op_kind == "all":
                nonzero = T.Not(nonzero)
            return nonzero

        def _fold_combine(op_kind: str, acc, term):
            """Fold *term* into the accumulator value *acc*: a sum for a count, else an or."""
            term = T.cast(term, acc.dtype)
            return acc + term if op_kind == "count_nonzero" else acc | term

        step = threads * vec
        full_steps = run_units // step
        tail = full_steps * step != run_units
        num_warps = threads // WARP_LANES
        # Up to this many warps one lane folds their values in serially: at (32, 4096)
        # float16 any 1.535 us against 1.600 for the first warp shuffling them; at 32 warps
        # ((32, 32768) float16 count_nonzero) 2.688 against 2.624, so the shuffle stays.
        lane_fold_max_warps = 16

        @T.macro
        def fold_held(held, acc):
            for e in T.unroll(vec // components):
                acc[0] = _fold_combine(
                    op_kind, acc[0], _fold_term(op_kind, held, e, components, pack)
                )

        @T.prim_func
        def main(
            x: T.Tensor[(lead, rows, run_units), unit_dtype],
            out: T.Tensor[(rows,), out_dtype],
        ):
            with T.Kernel(rows, threads=threads) as row:
                tx = T.get_thread_binding()
                held = T.alloc_local((vec,), unit_dtype)
                acc = T.alloc_local((1,), acc_dtype)
                warp_acc = T.alloc_shared((num_warps,), acc_dtype)

                acc[0] = T.cast(0, acc_dtype)
                for lead_idx in T.serial(lead):
                    for k in T.serial(full_steps):
                        if stream:
                            T.call_extern(
                                "handle",
                                "tileops::load16_evict_first",
                                T.address_of(held[0]),
                                T.address_of(x[lead_idx, row, k * step + tx * vec]),
                            )
                        else:
                            for v in T.vectorized(vec):
                                held[v] = x[lead_idx, row, k * step + tx * vec + v]
                        fold_held(held, acc)
                    if tail:
                        # ``vec`` divides the run, so a lane's vector is wholly inside it
                        # or wholly past it; a select, since a guarded vector load and the
                        # vectorized loop above cannot share a kernel.
                        start = full_steps * step + tx * vec
                        for v in T.serial(vec):
                            held[v] = T.if_then_else(
                                start < run_units,
                                x[lead_idx, row, start + v],
                                T.cast(pad, unit_dtype),
                            )
                        fold_held(held, acc)

                for stage in T.serial(WARP_SHUFFLE_STAGES):
                    acc[0] = _fold_combine(
                        op_kind, acc[0], T.shfl_xor(acc[0], T.int32(WARP_LANES // 2) >> stage)
                    )
                if num_warps > 1:
                    if tx % WARP_LANES == 0:
                        warp_acc[tx // WARP_LANES] = acc[0]
                    T.sync_threads()
                    if num_warps <= lane_fold_max_warps:
                        # Lane 0 already holds warp 0's value and folds the others in.
                        if tx == 0:
                            for w in T.unroll(1, num_warps):
                                acc[0] = _fold_combine(op_kind, acc[0], warp_acc[w])
                    else:
                        # The whole first warp shuffles; lanes past the warp count hold the
                        # identity, and xor offsets below the warp count never reach them.
                        if tx < WARP_LANES:
                            acc[0] = T.if_then_else(
                                tx < num_warps, warp_acc[tx % num_warps], T.cast(0, acc_dtype)
                            )
                            for stage in T.serial(num_warps.bit_length() - 1):
                                acc[0] = _fold_combine(
                                    op_kind,
                                    acc[0],
                                    T.shfl_xor(acc[0], T.int32(num_warps // 2) >> stage),
                                )
                if tx == 0:
                    if counts:
                        out[row] = T.cast(acc[0], out_dtype)
                    elif op_kind == "any":
                        out[row] = T.cast(acc[0] != T.cast(0, acc_dtype), out_dtype)
                    else:
                        out[row] = T.cast(acc[0] == T.cast(0, acc_dtype), out_dtype)

        return main

    return _func


def _fold_units(
    dtype: torch.dtype, cols: int, address: int, op_kind: str
) -> "tuple[torch.dtype, int, int, bool, bool]":
    """What the fold reads runs of *cols* elements of *dtype* at *address* as.

    Returns the unit dtype, units per element, elements per unit, whether units are words
    of lanes, and whether a lane is a float tested by its bits past the sign. Bool and 8-bit integers go four to a 32-bit
    word, and float16, bfloat16 and float32 two or one, where a run is whole words and starts
    on one: a lane then tests a word's elements with bit operations rather than an element
    at a time.
    (32, 4096) float16 any took 1.600 us against 1.664 comparing elements (H200).
    """
    scalar_dtype, components = _SCALAR_VIEWS.get(dtype, (dtype, 1))
    # These floats can be read as words: bits past the sign decide whether each lane is
    # zero, including -0.0 and NaN, as torch compares them.
    word_float_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    size = scalar_dtype.itemsize
    # The word tests split a 32-bit word into lanes of one element.
    word_size = torch.uint32.itemsize
    whole = cols * size % word_size == 0 and address % word_size == 0
    # A count of float lanes measured slower as words: (32, 32768) float16 count_nonzero
    # 2.752 us against 2.624 comparing elements.
    word_floats = scalar_dtype in word_float_dtypes and op_kind != "count_nonzero"
    # Only these floats have a sign-bit mask that matches torch's truth;
    # any other float (float8, where 0x80 is -0.0 or NaN by format) compares elements.
    word_bytes = size == 1 and not scalar_dtype.is_floating_point
    if components == 1 and whole and (word_bytes or word_floats):
        return torch.uint32, 1, word_size // size, True, scalar_dtype in word_float_dtypes
    if scalar_dtype in (torch.float16, torch.bfloat16):
        # Element tests of a 16-bit float are bit tests too; see ``_fold_term``.
        return torch.int16, 1, 1, False, True
    return scalar_dtype, components, 1, False, False


def _fold_view(
    x: torch.Tensor, lead: int, rows: int, cols: int, op_kind: str
) -> "tuple[torch.Tensor, tuple]":
    """*x*, viewed as ``(lead, rows, cols)``, as the units the fold reads, and how it reads them.

    Row ``r`` is the elements ``x[:, r, :]``. Returns the units and the facts of the read
    that select the fold's program, for `_fold_program`; where *x* starts decides some.
    """
    # Views only: a conjugated complex tensor is read unconjugated, which negates only
    # imaginary parts and so no element's truth.
    scalars = x
    if x.dtype == torch.bool:
        scalars = x.view(torch.int8)
    elif x.is_complex():
        scalars = torch.view_as_real(x.conj() if x.is_conj() else x).flatten(-2)
    scalars = scalars.reshape(lead, rows, -1)
    unit_dtype, components, pack, words, magnitude = _fold_units(
        x.dtype, cols, scalars.data_ptr(), op_kind
    )
    units = scalars.view(unit_dtype)
    row_units = units.shape[-1]
    vec = _fold_vector(units.element_size(), row_units, components, units.data_ptr())
    read = (
        Kernel.dtype_to_str(unit_dtype),
        units.element_size(),
        row_units,
        components,
        pack,
        vec,
        words,
        magnitude,
    )
    return units, read


def _fold_program(
    lead: int,
    rows: int,
    cols: int,
    op_kind: str,
    out_dtype: str,
    threads: "int | None",
    unit_dtype: str,
    unit_bytes: int,
    row_units: int,
    components: int,
    pack: int,
    vec: int,
    words: bool,
    magnitude: bool,
):
    """The fold reducing each row of the units `_fold_view` returned to one *out_dtype* value.

    The arguments after *threads* are the read `_fold_view` returned. ``threads=None`` sizes
    the block by the run.
    """
    if threads is None:
        threads = _fold_threads(row_units, vec)
    program = _logical_fold_kernel(
        lead,
        rows,
        cols,
        op_kind,
        unit_dtype,
        components,
        pack,
        vec,
        out_dtype,
        # The fold reads every unit of the input exactly once, so none of it is worth
        # keeping cached: a full-width access loads evict-first. The helper moves
        # exactly one 16-byte vector, so a narrower access keeps the plain load.
        vec * unit_bytes == VECTOR_ACCESS_BYTES,
        words,
        magnitude,
    )
    return program(threads)


class LogicalReduceKernel(Kernel, LogicalReduceFwdInterface, CountNonzeroFwdInterface):
    """Any / all / count_nonzero forward kernel, general over which axes reduce.

    Supports SM80+ architectures. ``forward`` moves *reduce_axes* last and folds each
    resulting row in one pass without staging it in shared memory. Output dtype is bool
    for any/all and int64 for count_nonzero.

    Args:
        shape: Shape of the input.
        reduce_axes: Non-negative axis indices, ascending, that the reduction runs over.
        op_kind: One of "any", "all", "count_nonzero".
        dtype: Input data type (float16, bfloat16, float32, bool, int32, int64,
               complex64, or complex128).
        keepdim: Whether a reduced axis stays as a length-1 axis.
        config: Optional kernel configuration dict.
        device_index: CUDA device the input lives on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general: bool = True

    @classmethod
    def entry_for(cls, call: LogicalReduceCall) -> Entry:
        identity = (
            call.shape,
            call.axes,
            call.op_kind,
            call.dtype,
            call.keepdim,
            call.device_index,
        )
        return identity, lambda: cls(
            call.shape,
            call.axes,
            call.op_kind,
            call.dtype,
            keepdim=call.keepdim,
            device_index=call.device_index,
        )

    def __init__(
        self,
        shape: "tuple[int, ...]",
        reduce_axes: "tuple[int, ...]",
        op_kind: str,
        dtype: torch.dtype,
        keepdim: bool = False,
        config: Optional[dict] = None,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        self.shape = tuple(shape)
        self.reduce_axes = tuple(reduce_axes)
        self.op_kind = op_kind
        self.dtype = dtype
        self.keepdim = keepdim
        self.N = prod(self.shape[a] for a in self.reduce_axes)
        self.M = prod(self.shape) // self.N
        self._scalar_dtype, self._components = _SCALAR_VIEWS.get(dtype, (dtype, 1))
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        unit_dtype, components, pack, _, _ = _fold_units(self.dtype, self.N, 0, self.op_kind)
        row_units = self.N * components // pack
        vec = _fold_vector(unit_dtype.itemsize, row_units, components)
        return {"threads": _fold_threads(row_units, vec)}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t} for t in (128, 256, 512, 1024)]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Pick the block width by timing one call per candidate."""
        device = torch.cuda.current_device()
        shape = (self.M, self.N * self._components)
        if self._scalar_dtype.is_floating_point:
            x = torch.randn(shape, dtype=self._scalar_dtype, device=device)
        else:
            x = torch.randint(0, 2, shape, dtype=self._scalar_dtype, device=device)
        if self.dtype.is_complex:
            x = torch.view_as_complex(x.view(self.M, self.N, 2).contiguous())
        elif self.dtype == torch.bool:
            x = x.bool()
        tune_by_forward(self, x, warmup=warmup, rep=rep, forward=self._reduce_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce *reduce_axes* of *x*.

        Args:
            x: The input, contiguous, of the constructed shape.

        Returns:
            The reduced tensor, dtype bool (any/all) or int64 (count_nonzero).
        """
        y = self._reduce_rows(rows_for_axes(x, self.reduce_axes))
        return restore_reduced(y, self.shape, self.reduce_axes, self.keepdim)

    def _reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the trailing axis of an ``(M, N)`` buffer of the declared dtype."""
        units, read = _fold_view(x, 1, self.M, self.N, self.op_kind)
        counted = self.program(self._compile, *read)(units)
        # 0 or 1 in int8 is bool's own representation, so this is a reinterpretation.
        return counted if self.op_kind == "count_nonzero" else counted.view(torch.bool)

    def _compile(self, cfg: dict, *read):
        """The fold at *cfg*'s thread count, for the read `_fold_view` returned."""
        out_dtype = "int64" if self.op_kind == "count_nonzero" else "int8"
        return _fold_program(1, self.M, self.N, self.op_kind, out_dtype, cfg["threads"], *read)


class LogicalReduceEdgeTwoPassKernel(Kernel, LogicalReduceFwdInterface):
    """Logical reduction of a prefix and a suffix of the axes in two passes.

    No permute: the trailing axes fold as contiguous rows into 0/1 int8 (or fp32 count)
    partials, then the leading axes fold down the columns of those partials — or/and are
    max/min over 0 and 1, a count is a sum. Each pass sizes itself from its extents.

    Args:
        shape: Shape of the input.
        reduce_axes: A non-empty prefix and a non-empty suffix of the axes, ascending,
            with at least one kept axis between them.
        op_kind: One of "any", "all", "count_nonzero".
        dtype: Input data type.
        keepdim: Whether a reduced axis stays as a length-1 axis.
        device_index: CUDA device the input lives on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: LogicalReduceCall) -> "str | None":
        if call.edge_kept <= 0:
            return "serves kept axes between a reduced prefix and suffix"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: LogicalReduceCall) -> Entry:
        identity = (
            call.shape,
            call.axes,
            call.op_kind,
            call.dtype,
            call.keepdim,
            call.device_index,
        )
        return identity, lambda: cls(
            call.shape,
            call.axes,
            call.op_kind,
            call.dtype,
            keepdim=call.keepdim,
            device_index=call.device_index,
        )

    def __init__(
        self,
        shape: "tuple[int, ...]",
        reduce_axes: "tuple[int, ...]",
        op_kind: str,
        dtype: torch.dtype,
        keepdim: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        self.shape = tuple(shape)
        self.reduce_axes = tuple(reduce_axes)
        self.op_kind = op_kind
        self.dtype = dtype
        self.keepdim = keepdim
        k, j = edge_axis_split(len(self.shape), self.reduce_axes)
        self.lead = prod(self.shape[:k])
        self.kept = prod(self.shape[k : len(self.shape) - j])
        self.trail = prod(self.shape[len(self.shape) - j :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        units, read = _fold_view(x, 1, self.lead * self.kept, self.trail, self.op_kind)
        y = self.program(self._launcher, *read)(units)
        if self.op_kind != "count_nonzero":
            y = y.view(torch.bool)
        return restore_reduced(y, self.shape, self.reduce_axes, self.keepdim)

    def _launcher(self, cfg: dict, *read) -> Callable:
        """The rows pass into partials, for the read `_fold_view` returned, then the columns
        pass down them."""
        lead, kept = self.lead, self.kept
        if self.op_kind == "count_nonzero":
            # Partial counts are fp32, exact below 2^24, so the columns pass sums them
            # without an int accumulator and writes int64 itself.
            inner = "float32"
            outer = ("sum", "float32", "int64")
        else:
            inner = "int8"
            outer = ("amax" if self.op_kind == "any" else "amin", "int8", "int8")
        rows_pass = CompiledProgram(
            _fold_program(1, lead * kept, self.trail, self.op_kind, inner, None, *read)
        )
        columns_pass = down_rows_launcher(lead, kept, *outer, 0.0, down_rows_splits(lead, kept))

        def launch(units: torch.Tensor) -> torch.Tensor:
            return columns_pass(rows_pass(units).reshape(lead, kept))

        return launch


class CountNonzeroEdgeTwoPassKernel(LogicalReduceEdgeTwoPassKernel, CountNonzeroFwdInterface):
    """The two-pass edge-axis count: partial counts cross between the passes in fp32."""

    @classmethod
    def refusal(cls, call: LogicalReduceCall) -> "str | None":
        kept = call.edge_kept
        if kept <= 0:
            return "serves kept axes between a reduced prefix and suffix"
        # fp32 is exact up to FP32_EXACT_INT_LIMIT.
        if prod(call.shape) // kept > FP32_EXACT_INT_LIMIT:
            return f"counts in float32, exact up to {FP32_EXACT_INT_LIMIT}"
        return super().refusal(call)


class LogicalReduceEdgeFusedKernel(Kernel, LogicalReduceFwdInterface, CountNonzeroFwdInterface):
    """Logical reduction of a prefix and a suffix of the axes in one pass.

    One block reduces one kept column, walking the leading axes serially while folding
    each contiguous trailing run.

    Args:
        shape: Shape of the input.
        reduce_axes: A non-empty prefix and a non-empty suffix of the axes, ascending,
            with at least one kept axis between them.
        op_kind: One of "any", "all", "count_nonzero".
        dtype: Input data type.
        keepdim: Whether a reduced axis stays as a length-1 axis.
        config: Optional ``{"threads": n}``; the default sizes the block by the trailing run.
        device_index: CUDA device the input lives on.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"logical_reduce_edge_two_pass"})

    # The pass runs one block per kept column and has no other parallelism: the fewest
    # kept columns that fill the device, per calibrated board; none elsewhere.
    _FUSED_MIN_KEPT: ClassVar[Mapping[str, int]] = {"h200": 32}

    @classmethod
    def refusal(cls, call: LogicalReduceCall) -> "str | None":
        kept = call.edge_kept
        if kept <= 0:
            return "serves kept axes between a reduced prefix and suffix"
        least = cls._FUSED_MIN_KEPT.get(call.calibration, math.inf)
        if kept < least:
            return f"fuses at least {least} kept positions on this board, got {kept}"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: LogicalReduceCall) -> Entry:
        identity = (
            call.shape,
            call.axes,
            call.op_kind,
            call.dtype,
            call.keepdim,
            call.device_index,
        )
        return identity, lambda: cls(
            call.shape,
            call.axes,
            call.op_kind,
            call.dtype,
            keepdim=call.keepdim,
            device_index=call.device_index,
        )

    def __init__(
        self,
        shape: "tuple[int, ...]",
        reduce_axes: "tuple[int, ...]",
        op_kind: str,
        dtype: torch.dtype,
        keepdim: bool = False,
        config: Optional[dict] = None,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        self.shape = tuple(shape)
        self.reduce_axes = tuple(reduce_axes)
        self.op_kind = op_kind
        self.dtype = dtype
        self.keepdim = keepdim
        k, j = edge_axis_split(len(self.shape), self.reduce_axes)
        self.lead = prod(self.shape[:k])
        self.kept = prod(self.shape[k : len(self.shape) - j])
        self.trail = prod(self.shape[len(self.shape) - j :])
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """No width: ``forward`` sizes it from the trailing run unless the caller states one."""
        return {"threads": None}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        units, read = _fold_view(x, self.lead, self.kept, self.trail, self.op_kind)
        counted = self.program(self._compile, *read)(units)
        y = counted if self.op_kind == "count_nonzero" else counted.view(torch.bool)
        return restore_reduced(y, self.shape, self.reduce_axes, self.keepdim)

    def _compile(self, cfg: dict, *read):
        """The fold at *cfg*'s thread count, for the read `_fold_view` returned."""
        out_dtype = "int64" if self.op_kind == "count_nonzero" else "int8"
        return _fold_program(
            self.lead,
            self.kept,
            self.trail,
            self.op_kind,
            out_dtype,
            cfg.get("threads"),
            *read,
        )
