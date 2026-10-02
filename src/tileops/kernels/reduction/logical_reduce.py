"""Logical reduce kernels (any, all, count_nonzero) using TileLang.

Each row is folded into registers as it is read, one vector access of at most 16 bytes
per lane per step. A count sums the nonzero elements; any and all or together a flag that is
set on an element that decides them, a nonzero one for any and a zero one for all.

The input is read at its own bytes: bool as int8 (four to a 32-bit word where the row
allows), a complex element as its two real parts, every other dtype as declared.
"""

import functools
import math
from math import prod
from typing import ClassVar, Mapping, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.reduction._primitives import (
    FP32_EXACT_INT_LIMIT,
    ceildiv_int,
    down_rows_once,
    down_rows_split,
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


# A lane folds one vector access per step, and a block is sized so each lane folds at
# least this many of them across a row: fewer leaves lanes that load nothing while the
# block still pays for its reduction.
_FOLD_VECTORS_PER_LANE = 2
_FOLD_MIN_THREADS = WARP_LANES
_FOLD_MAX_THREADS = 1024
# Bytes a 32-bit word packs, and the per-byte masks its byte tests use.
_WORD_BYTES = 4
_BYTE_ONES = 0x01010101
_BYTE_LOW = 0x7F7F7F7F
_BYTE_HIGH = 0x80808080

_STREAMING_LOAD_HELPER_PATH = csrc_path("streaming_load.h")


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
    lanes = max(ceildiv_int(row_units, vec) // _FOLD_VECTORS_PER_LANE, 1)
    lanes = 1 << (lanes.bit_length() - 1)
    return max(_FOLD_MIN_THREADS, min(lanes, _FOLD_MAX_THREADS))


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
        pack: Elements per unit: 4 for bytes read as a 32-bit word, else 1.
        vec: Units per vector access, a multiple of ``components`` dividing the run.
        out_dtype: TileLang dtype string of the output.
        stream: Whether a full-width vector access loads evict-first.

    Returns:
        A TileLang JIT-compiled kernel factory accepting (threads).
    """
    run_units = cols * components // pack
    counts = op_kind == "count_nonzero"
    count_dtype = "int32" if lead * cols < 1 << 31 else "int64"
    acc_dtype = count_dtype if counts else "uint32"
    # What a vector past the end of the run holds: nothing any or a count takes, and
    # no zero for all.
    pad = (_BYTE_ONES if pack > 1 else 1) if op_kind == "all" else 0

    @tilelang.jit(out_idx=[1], compile_flags=["-include", _STREAMING_LOAD_HELPER_PATH])
    def _func(threads):
        def _fold_term(op_kind: str, held, e, components: int, pack: int):
            """What element (or word) *e* of a lane's vector adds to its accumulator."""
            if pack > 1:
                word = held[e]
                high = T.Cast("uint32", _BYTE_HIGH)
                if op_kind == "count_nonzero":
                    # Nonzero bytes: ``(b & 0x7F) + 0x7F`` sets bit 7 exactly when the low seven
                    # bits are not all clear, never carrying into the next byte; or-ing in ``b``
                    # adds bit 7 itself.
                    low = T.Cast("uint32", _BYTE_LOW)
                    return T.popcount((((word & low) + low) | word) & high)
                if op_kind == "any":
                    return word
                # Nonzero exactly when some byte is zero: ``(w - 0x01010101) & ~w & 0x80808080``,
                # a zero byte borrows into its own bit 7. ``w ^ ~0`` stands for ``~w``, which
                # CUDA's vector types do not define.
                return (
                    (word - T.Cast("uint32", _BYTE_ONES))
                    & (word ^ T.Cast("uint32", 0xFFFFFFFF))
                    & high
                )
            # An element is nonzero when any of its scalars compares unequal to zero in its own
            # dtype, so -0.0 is zero and NaN is not, as in torch.
            zero = T.cast(0, held.dtype)
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
                                "tl::tileops_load16_evict_first",
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


def _fold_units(dtype: torch.dtype, cols: int, address: int) -> "tuple[torch.dtype, int, int]":
    """What the fold reads runs of *cols* elements of *dtype* at *address* as.

    Returns the unit dtype, units per element and elements per unit. Bytes go four to a
    32-bit word where a run is whole words and starts on one, so a lane counts a word's
    nonzero bytes at once.
    """
    scalar_dtype, components = _SCALAR_VIEWS.get(dtype, (dtype, 1))
    if scalar_dtype.itemsize == 1 and cols % _WORD_BYTES == 0 and address % _WORD_BYTES == 0:
        return torch.uint32, 1, _WORD_BYTES
    return scalar_dtype, components, 1


def _fold_reduce(
    x: torch.Tensor, lead: int, rows: int, cols: int, op_kind: str, out_dtype: str, threads=None
) -> torch.Tensor:
    """Reduce each row of *x*, viewed as ``(lead, rows, cols)``, to one *out_dtype* value.

    Row ``r`` is the elements ``x[:, r, :]``. ``threads=None`` sizes the block by the run.

    Raises:
        ValueError: *threads* is not a power of two from one warp to
            ``_FOLD_MAX_THREADS``, which the block's shuffle reduction needs.
    """
    # Views only: a conjugated complex tensor is read unconjugated, which negates only
    # imaginary parts and so no element's truth.
    scalars = x
    if x.dtype == torch.bool:
        scalars = x.view(torch.int8)
    elif x.is_complex():
        scalars = torch.view_as_real(x.conj() if x.is_conj() else x).flatten(-2)
    scalars = scalars.reshape(lead, rows, -1)
    unit_dtype, components, pack = _fold_units(x.dtype, cols, scalars.data_ptr())
    units = scalars.view(unit_dtype)
    row_units = units.shape[-1]
    vec = _fold_vector(units.element_size(), row_units, components, units.data_ptr())
    if threads is None:
        threads = _fold_threads(row_units, vec)
    if not (_FOLD_MIN_THREADS <= threads <= _FOLD_MAX_THREADS and threads & (threads - 1) == 0):
        raise ValueError(
            f"threads={threads}: the fold needs a power of two from "
            f"{_FOLD_MIN_THREADS} to {_FOLD_MAX_THREADS}"
        )
    program = _logical_fold_kernel(
        lead,
        rows,
        cols,
        op_kind,
        Kernel.dtype_to_str(unit_dtype),
        components,
        pack,
        vec,
        out_dtype,
        # The fold reads every unit of the input exactly once, so none of it is worth
        # keeping cached: a full-width access loads evict-first. The helper moves
        # exactly one 16-byte vector, so a narrower access keeps the plain load.
        vec * units.element_size() == VECTOR_ACCESS_BYTES,
    )
    return program(threads)(units)


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
        tune: Whether to autotune (default False).
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
        tune: bool = False,
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
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        unit_dtype, components, pack = _fold_units(self.dtype, self.N, 0)
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
        counts = self.op_kind == "count_nonzero"
        counted = _fold_reduce(
            x,
            1,
            self.M,
            self.N,
            self.op_kind,
            "int64" if counts else "int8",
            self.config["threads"],
        )
        # 0 or 1 in int8 is bool's own representation, so this is a reinterpretation.
        return counted if counts else counted.view(torch.bool)


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
    def applies(cls, call: LogicalReduceCall) -> bool:
        return call.edge_kept > 0

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
        rows = self.lead * self.kept
        if self.op_kind == "count_nonzero":
            # Partial counts are fp32, exact below 2^24, so the columns pass sums them
            # without an int accumulator and writes int64 itself.
            partials = _fold_reduce(x, 1, rows, self.trail, self.op_kind, "float32")
            outer = ("sum", "float32", "int64")
        else:
            partials = _fold_reduce(x, 1, rows, self.trail, self.op_kind, "int8")
            outer = ("amax" if self.op_kind == "any" else "amin", "int8", "int8")
        partials = partials.reshape(self.lead, self.kept)
        splits = down_rows_splits(self.lead, self.kept)
        if splits == 1:
            y = down_rows_once(partials, *outer, 0.0)
        else:
            y = down_rows_split(partials, *outer, 0.0, splits)
        if self.op_kind != "count_nonzero":
            y = y.view(torch.bool)
        return restore_reduced(y, self.shape, self.reduce_axes, self.keepdim)


class CountNonzeroEdgeTwoPassKernel(LogicalReduceEdgeTwoPassKernel, CountNonzeroFwdInterface):
    """The two-pass edge-axis count: partial counts cross between the passes in fp32."""

    @classmethod
    def applies(cls, call: LogicalReduceCall) -> bool:
        # fp32 is exact up to FP32_EXACT_INT_LIMIT.
        kept = call.edge_kept
        return kept > 0 and prod(call.shape) // kept <= FP32_EXACT_INT_LIMIT


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
    def applies(cls, call: LogicalReduceCall) -> bool:
        kept = call.edge_kept
        return kept > 0 and kept >= cls._FUSED_MIN_KEPT.get(call.calibration, math.inf)

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
        counts = self.op_kind == "count_nonzero"
        counted = _fold_reduce(
            x,
            self.lead,
            self.kept,
            self.trail,
            self.op_kind,
            "int64" if counts else "int8",
            self.config.get("threads"),
        )
        y = counted if counts else counted.view(torch.bool)
        return restore_reduced(y, self.shape, self.reduce_axes, self.keepdim)
