"""Logical reduce kernels (any, all, count_nonzero) using TileLang.

Truthiness is decided as each element is loaded, then reduced:
  - any: reduce_max (1 if any element is non-zero)
  - all: reduce_min (1 if all elements are non-zero)
  - count_nonzero: reduce_sum (count of non-zero elements per row)

Operates on raw 2D (M, N) tensors; the kernel handles 256-element alignment
padding internally via masked loads with the appropriate identity value.

A bool input is read at its own width: the prim_func declares int8, which the tensor
is reinterpreted into, and writes its int8 result back the same way. Output is bool for
any/all, int64 for count_nonzero.
"""

import functools
from math import prod
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    FP32_EXACT_INT_LIMIT,
    BlockConfigPlanner,
    align_up,
    ceildiv_int,
    device_smem_budget,
    edge_axis_plan,
    edge_axis_split,
    identity_for,
    reduce_down_rows,
    restore_reduced,
    rows_for_axes,
    tune_by_forward,
)
from tileops.kernels.reduction.call_spec import (
    LogicalReduceCall,
    logical_edge_fused_region,
    logical_reduce_region,
)
from tileops.utils import WARP_LANES

__all__ = [
    "LogicalReduceEdgeFusedKernel",
    "LogicalReduceKernel",
    "storage_dtype_for",
    "to_logical_storage",
]

_LOGICAL_REDUCE_KINDS = {"any", "all", "count_nonzero"}

# Dtypes the prim_func cannot take, mapped to one it can. bool is one byte holding 0
# or 1, so int8 reinterprets it free.
_FLOAT32_STORAGE_DTYPE = torch.float32
_INT8_STORAGE_DTYPE = torch.int8
_BYTE_REINTERPRETED_DTYPES = frozenset({torch.bool})
_WIDENED_STORAGE_DTYPES = frozenset(
    {
        torch.complex64,
        torch.complex128,
        torch.int32,
        torch.int64,
    }
)
_UNSUPPORTED_STORAGE_DTYPES = _BYTE_REINTERPRETED_DTYPES | _WIDENED_STORAGE_DTYPES


def storage_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """The dtype the prim_func declares for an input of *dtype*."""
    if dtype in _BYTE_REINTERPRETED_DTYPES:
        return _INT8_STORAGE_DTYPE
    if dtype in _WIDENED_STORAGE_DTYPES:
        return _FLOAT32_STORAGE_DTYPE
    return dtype


def to_logical_storage(x: torch.Tensor) -> torch.Tensor:
    """Present *x* to the kernel in a storage dtype the prim_func declares.

    - bool:        reinterpreted as int8, which copies nothing.
    - int32/int64: cast to float32.
    - complex:     nonzero (either real or imaginary part != 0) -> 1.0, else 0.0.
    """
    if x.dtype in _BYTE_REINTERPRETED_DTYPES:
        return x.view(_INT8_STORAGE_DTYPE)
    if x.dtype in (torch.int32, torch.int64):
        return x.to(torch.float32)
    # complex: element is "truthy" if real != 0 OR imag != 0
    return ((x.real != 0) | (x.imag != 0)).to(torch.float32)


# Logical reduce kernel


def _logical_out_dtype(op_kind: str, partial: bool) -> str:
    """The dtype a logical reduce writes: 0/1 stays int8, a count widens.

    A count written for an outer pass stays fp32, exact below 2^24, so the
    down-rows engine can sum it without an int accumulator.
    """
    if op_kind != "count_nonzero":
        return "int8"
    return "float32" if partial else "int64"


# Elements a lane folds in the edge-fused pass. One block holds a `trail`-wide
# fp32 fragment, so this is what fixes its register footprint per lane rather
# than letting it grow with the row. Eight is the flat optimum at every width
# the manifest asks for.
_FUSED_EDGE_ELEMS_PER_LANE = 8
_FUSED_EDGE_MIN_THREADS = 64
_FUSED_EDGE_MAX_THREADS = 1024


def fused_edge_threads(trail: int) -> int:
    """The thread width the edge-fused pass runs a ``trail``-wide row at."""
    lanes = ceildiv_int(trail, _FUSED_EDGE_ELEMS_PER_LANE)
    lanes = 1 << max(lanes - 1, 0).bit_length()
    return max(_FUSED_EDGE_MIN_THREADS, min(lanes, _FUSED_EDGE_MAX_THREADS))


@functools.lru_cache(maxsize=32)
def _logical_reduce_edge_fused(lead: int, kept: int, trail: int, op_kind: str, dtype: str):
    """Build an any/all/count_nonzero kernel reducing a leading and a trailing axis.

    One block per kept column, walking the leading axis in serial and reducing
    each of its contiguous rows. The two-pass form writes ``lead * kept``
    partials and launches again to fold them; at these sizes that launch is most
    of the time.

    Args:
        lead: Extent of the leading axes, already folded into one.
        kept: Extent of the axis that survives.
        trail: Extent of the trailing axes, already folded into one.
        op_kind: One of "any", "all", "count_nonzero".
        dtype: TileLang dtype string of the input.

    Returns:
        A TileLang JIT-compiled kernel factory accepting (threads).
    """
    trail_padded = align_up(trail, DEFAULT_ALIGNMENT)
    needs_pad = trail_padded != trail
    pad_val = identity_for(op_kind)
    out_dtype = _logical_out_dtype(op_kind, partial=False)
    # any is a max and all is a min over 0/1, so each starts from its identity;
    # a count sums from zero. The column-wise fold starts from the same value a
    # padding column carries, since both are that op's identity.
    init_val = {"any": 0.0, "all": 1.0, "count_nonzero": 0.0}[op_kind]

    @tilelang.jit(out_idx=[1])
    def _func(threads):
        @T.prim_func
        def main(
            x: T.Tensor[(lead, kept, trail), dtype],
            out: T.Tensor[(kept,), out_dtype],
        ):
            with T.Kernel(kept, threads=threads) as pid_k:
                vals = T.alloc_fragment((1, trail_padded), "float32")
                acc = T.alloc_fragment((1,), "float32")

                # The leading axis folds into the fragment column by column, so
                # the block reduces once at the end rather than once per leading
                # index: the reduction is a barrier and a tree, and `lead` of
                # them cost more than the column-wise fold they replace.
                for j in T.Parallel(trail_padded):
                    vals[0, j] = init_val
                for lead_idx in T.serial(lead):
                    for j in T.Parallel(trail_padded):
                        if needs_pad:
                            val = T.if_then_else(
                                j < trail,
                                T.cast(x[lead_idx, pid_k, j], "float32"),
                                T.cast(pad_val, "float32"),
                            )
                        else:
                            val = T.cast(x[lead_idx, pid_k, j], "float32")
                        one = T.if_then_else(val != 0.0, 1.0, 0.0)
                        if op_kind == "any":
                            vals[0, j] = T.max(vals[0, j], one)
                        elif op_kind == "all":
                            vals[0, j] = T.min(vals[0, j], one)
                        else:
                            vals[0, j] = vals[0, j] + one
                if op_kind == "any":
                    T.reduce_max(vals, acc, dim=1)
                elif op_kind == "all":
                    T.reduce_min(vals, acc, dim=1)
                else:
                    T.reduce_sum(vals, acc, dim=1)

                if op_kind == "count_nonzero":
                    out[pid_k] = T.cast(acc[0], out_dtype)
                else:
                    out[pid_k] = T.cast(acc[0] > 0.5, out_dtype)

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _logical_reduce_kernel(M: int, N: int, op_kind: str, dtype: str, partial: bool = False):
    """Build a TileLang any/all/count_nonzero kernel.

    Cast input to bool (0.0 or 1.0 in float32), then:
      - any: reduce_max over the row (1.0 if any element is non-zero)
      - all: reduce_min over the row (1.0 if all elements are non-zero)
      - count_nonzero: reduce_sum over the row (count of non-zero elements)

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Original hidden dimension (last dim, before padding).
        op_kind: One of "any", "all", "count_nonzero".
        dtype: TileLang dtype string (e.g. "float16", "bfloat16", "float32").
        partial: Write partials for an outer pass; only the count changes,
            staying fp32 instead of widening to int64.

    Returns:
        A TileLang JIT-compiled kernel factory accepting (block_m, threads).
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    _pad_val = identity_for(op_kind)
    out_dtype = _logical_out_dtype(op_kind, partial)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.macro
        def compute(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                bool_vals = T.alloc_fragment((block_m, N_padded), "float32")
                result = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), out_dtype)

                # Truthiness is decided at the load, so a tile costs one fragment.
                if _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            val = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_pad_val, "float32"),
                            )
                            bool_vals[i, j] = T.if_then_else(val != 0.0, 1.0, 0.0)
                else:
                    # Load via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            bool_vals[i, j] = T.if_then_else(
                                T.cast(shared_buf[i, j], "float32") != 0.0, 1.0, 0.0
                            )

                if op_kind == "any":
                    # any: result is 1 if max(bool_vals) == 1
                    T.reduce_max(bool_vals, result, dim=1)
                elif op_kind == "all":
                    # all: result is 1 if min(bool_vals) == 1
                    T.reduce_min(bool_vals, result, dim=1)
                else:
                    # count_nonzero: sum of bool values per row
                    T.reduce_sum(bool_vals, result, dim=1)

                if op_kind == "count_nonzero":
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i], out_dtype)
                else:
                    # Cast result to int8 (bool representation: 0 or 1)
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i] > 0.5, "int8")

                T.copy(out_local, out[pid_m * block_m])

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            compute(x, out)

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _logical_reduce_kernel_tiled(
    M: int, N: int, op_kind: str, dtype: str, tile_n: int, partial: bool = False
):
    """Build a tiled TileLang any/all/count_nonzero kernel.

    Iterates over the reduction dimension in chunks of ``tile_n`` columns, for the
    rows a single pass cannot hold. ``partial`` as in ``_logical_reduce_kernel``.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    _pad_val = identity_for(op_kind)
    _past_n = num_tiles * tile_n > N
    out_dtype = _logical_out_dtype(op_kind, partial)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        # Only the last tile can run past N, and only the last block past the row tail.
        needs_mask = _past_n or M % block_m != 0

        @T.macro
        def compute(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                bool_vals = T.alloc_fragment((block_m, tile_n), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                tile_acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), out_dtype)

                if op_kind == "all":
                    T.fill(acc, 1.0)
                else:
                    T.fill(acc, 0.0)

                for t in T.Serial(num_tiles):
                    # Truthiness is decided at the load, so a tile costs one fragment.
                    # In-bounds tiles arrive by T.copy.
                    if needs_mask:
                        with T.If(t < num_tiles - 1):
                            with T.Then():
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        bool_vals[i, j] = T.if_then_else(
                                            T.cast(shared_buf[i, j], "float32") != 0.0, 1.0, 0.0
                                        )
                            with T.Else():
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        val = T.if_then_else(
                                            T.And(
                                                pid_m * block_m + i < M,
                                                t * tile_n + j < N,
                                            ),
                                            T.cast(
                                                x[pid_m * block_m + i, t * tile_n + j],
                                                "float32",
                                            ),
                                            T.cast(_pad_val, "float32"),
                                        )
                                        bool_vals[i, j] = T.if_then_else(val != 0.0, 1.0, 0.0)
                    else:
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                bool_vals[i, j] = T.if_then_else(
                                    T.cast(shared_buf[i, j], "float32") != 0.0, 1.0, 0.0
                                )

                    if op_kind == "any":
                        T.reduce_max(bool_vals, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.max(acc[i], tile_acc[i])
                    elif op_kind == "all":
                        T.reduce_min(bool_vals, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.min(acc[i], tile_acc[i])
                    else:
                        T.reduce_sum(bool_vals, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]

                if op_kind == "count_nonzero":
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i], out_dtype)
                else:
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i] > 0.5, "int8")

                T.copy(out_local, out[pid_m * block_m])

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            compute(x, out)

        return main

    return _func


class LogicalReduceKernel(Kernel):
    """Any / all / count_nonzero forward kernel.

    Supports SM80+ architectures. Handles 256-element alignment padding inside
    the kernel. Casts input to bool (0/1) and reduces via max (any), min (all),
    or sum (count_nonzero). Uses an N-tiled fallback for long rows that exceed
    TileLang's single-fragment column limit.

    Output dtype is bool for any/all and int64 for count_nonzero.

    ``forward`` takes the tensor the op declares and reduces *reduce_axes* of it; the
    permute to rows and the shape of the result are this kernel's business.

    TileLang does not support bool, integer, or complex dtypes as a shared-memory storage
    dtype. When *dtype* is one of these the kernel is compiled for float32 and ``forward``
    converts the input, so an op hands over the tensor its manifest declares and this
    restriction stays inside the implementation that has it.

    Args:
        M: Rows the reduction leaves.
        N: Elements each row reduces.
        op_kind: One of "any", "all", "count_nonzero".
        dtype: Input data type (float32, float16, bfloat16, bool, complex64,
               or complex128).
        reduce_axes: Non-negative axis indices, ascending, that the reduction runs over.
        keepdim: Whether a reduced axis stays as a length-1 axis.
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
        device_index: CUDA device the input lives on, for the shared-memory budget.
    """

    #: A word holds this many bool bytes, and a thread folds a whole vector of
    #: them at once.
    _WORD_BYTES = 4
    _VECTOR_BYTES = 16
    #: Widest block the packed fold takes. One block owns one row, so a wider
    #: one only adds lanes with no words left to fold.
    _PACKED_MAX_THREADS = 128

    def _packed_fold_applies(self, x: torch.Tensor) -> bool:
        """Whether the rows of *x* can be folded a word at a time.

        Reading four bytes as one word needs them contiguous, a whole number of
        words to a row, and a storage offset on a word boundary -- ``view``
        refuses an unaligned one. One block a row keeps a block's words adjacent.
        """
        return (
            self.op_kind in ("any", "all")
            and self.dtype_to_str(self._kernel_dtype) == "int8"
            and self.config["block_m"] == 1
            and x.is_contiguous()
            and x.storage_offset() % self._WORD_BYTES == 0
            and x.shape[-1] % self._WORD_BYTES == 0
        )

    def _packed_fold_launch(self) -> tuple[int, int]:
        """Threads, and words each folds, for one row of this kernel's shape.

        The block covers the row in one step where it can.
        """
        vec = self._VECTOR_BYTES // self._WORD_BYTES
        words = self.N // self._WORD_BYTES
        threads = min(
            self._PACKED_MAX_THREADS,
            max(WARP_LANES, align_up(ceildiv_int(words, vec), WARP_LANES)),
        )
        return threads, vec

    @staticmethod
    def _packed_word_term(op_kind: str, w):
        """What one word contributes: nonzero where the row's answer turns.

        ``any`` asks whether a byte is set, which the word itself answers.
        ``all`` asks whether one is clear, and ``(w - 0x01010101) & ~w &
        0x80808080`` is nonzero exactly when some byte of *w* is zero. Both then
        fold with ``or``.

        Set means nonzero, not one: a bool tensor may hold any other byte, and
        both the general path and ``torch.all`` read it as true.
        """
        if op_kind == "any":
            return w
        ones = T.Cast("uint32", 0x01010101)
        high = T.Cast("uint32", 0x80808080)
        # ``w ^ ~0`` rather than ``~w``: CUDA gives the vector types no operator~.
        flip = T.Cast("uint32", 0xFFFFFFFF)
        return (w - ones) & (w ^ flip) & high

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _packed_kernel(m: int, words: int, op_kind: str):
        """Build an any/all kernel that folds four bools per word.

        One block a row, folding into registers, where the general path stages
        the row through shared memory and widens every byte to fp32 first.
        """
        term = LogicalReduceKernel._packed_word_term

        @tilelang.jit(out_idx=[1])
        def _func(threads: int, vec: int):
            step = threads * vec
            steps = words // step
            exact = steps * step == words

            @T.prim_func
            def main(x: T.Tensor((m, words), "uint32"), out: T.Tensor((m,), "int8")):
                with T.Kernel(m, threads=threads) as bm:
                    acc = T.alloc_fragment((threads, vec), "uint32")
                    lane = T.alloc_fragment((threads,), "uint32")
                    red = T.alloc_fragment((1,), "uint32")
                    T.clear(acc)
                    for k in T.serial(steps):
                        for i, j in T.Parallel(threads, vec):
                            acc[i, j] = acc[i, j] | term(
                                op_kind, x[bm, (k * threads + i) * vec + j]
                            )
                    if not exact:
                        for i, j in T.Parallel(threads, vec):
                            idx = steps * step + i * vec + j
                            with T.If(idx < words):  # noqa: SIM117
                                with T.Then():
                                    acc[i, j] = acc[i, j] | term(op_kind, x[bm, idx])
                    T.reduce_max(acc, lane, dim=1)
                    T.reduce_max(lane, red, dim=0)
                    found = red[0] != T.Cast("uint32", 0)
                    out[bm] = T.Cast("int8", T.Not(found) if op_kind == "all" else found)

            return main

        return _func

    supported_archs: list[int] = [80, 86, 89, 90]
    general: bool = True

    @classmethod
    def applies(cls, call: LogicalReduceCall) -> bool:
        return logical_reduce_region(call)

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        reduce_axes: "tuple[int, ...]",
        keepdim: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        if op_kind not in _LOGICAL_REDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_LOGICAL_REDUCE_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.reduce_axes = tuple(reduce_axes)
        self.keepdim = keepdim
        # TileLang stores neither bool nor integer, so each maps to a declared dtype.
        self._kernel_dtype = storage_dtype_for(dtype)
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch.tensor([], dtype=self._kernel_dtype).element_size()
        self._smem_budget = device_smem_budget(device_index)
        self._planner = BlockConfigPlanner(
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
        )
        self._needs_tiling = self._planner.needs_tiling
        self.kernel = None
        if not self._needs_tiling:
            self.kernel = _logical_reduce_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_to_str(self._kernel_dtype),
            )
        self.init_config(config, tune)
        if self._needs_tiling and not tune:
            bm = self.config.get("block_m", 1)
            threads = self.config.get("threads", DEFAULT_THREADS)
            if "tile_n" not in self.config or self.config["tile_n"] == 0:
                self.config["tile_n"] = self._planner.tile_n_for(bm, threads)
            reason = self._planner.reject_tile_n(bm, self.config["tile_n"], threads)
            if reason:
                raise ValueError(reason)

    @property
    def default_config(self) -> dict:
        return self._planner.default_config()

    @property
    def autotune_configs(self) -> list[dict]:
        return self._planner.autotune_configs()

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune logical reduce, benchmarking tiled configs directly."""
        if not self._needs_tiling:
            return super().autotune(warmup=warmup, rep=rep)
        device = torch.cuda.current_device()
        if self._kernel_dtype.is_floating_point:
            x = torch.randn(self.M, self.N, dtype=self._kernel_dtype, device=device)
        else:
            # int8 storage has no normal distribution; a mix of zero and non-zero will do.
            x = torch.randint(0, 2, (self.M, self.N), dtype=self._kernel_dtype, device=device)
        tune_by_forward(self, x, warmup=warmup, rep=rep, forward=self._reduce_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce *reduce_axes* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device. A dtype TileLang
                cannot store is converted here.

        Returns:
            The reduced tensor, dtype bool (any/all) or int64 (count_nonzero).

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_storage(x)
        k, j = edge_axis_split(x.ndim, self.reduce_axes)
        # A count is carried in fp32 across the two passes: exact while the
        # elements each output reduces stay within fp32's integer range.
        reduced_count = x.numel() // prod(x.shape[k : x.ndim - j]) if k else 0
        if k and (self.op_kind != "count_nonzero" or reduced_count <= FP32_EXACT_INT_LIMIT):
            y = self._reduce_edge_axes(x, k, j)
            return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)
        rows = rows_for_axes(x, self.reduce_axes)
        y = self._reduce_rows(rows)
        return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)

    def _reduce_edge_axes(self, x: torch.Tensor, k: int, j: int) -> torch.Tensor:
        """Reduce a prefix and a suffix of the axes without permuting the tensor.

        Two passes in the tensor's own layout: the trailing axes reduce as
        contiguous rows into 0/1 int8 (or fp32 count) partials, then the
        leading axes fold down the columns of those partials — or/and are max/
        min over 0 and 1, a count is a sum.
        """
        lead, kept, trail, planner, cfg = edge_axis_plan(
            tuple(x.shape), k, j, self._elem_bytes, self._smem_budget
        )
        dtype_str = self.dtype_to_str(self._kernel_dtype)
        if planner.needs_tiling:
            stage = _logical_reduce_kernel_tiled(
                lead * kept, trail, self.op_kind, dtype_str, cfg["tile_n"], partial=True
            )
        else:
            stage = _logical_reduce_kernel(
                lead * kept, trail, self.op_kind, dtype_str, partial=True
            )
        partials = stage(cfg["block_m"], cfg["threads"])(x.reshape(lead * kept, trail))
        partials = partials.reshape(lead, kept)
        if self.op_kind == "count_nonzero":
            # The columns pass writes int64 itself; leaving it in fp32 and casting
            # after costs a third kernel launch.
            return reduce_down_rows(partials, "sum", "float32", "int64", 0.0)
        outer_kind = "amax" if self.op_kind == "any" else "amin"
        return reduce_down_rows(partials, outer_kind, "int8", "int8", 0.0).view(torch.bool)

    def _reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the trailing axis of an ``(M, N)`` buffer.

        The prim_func counts in the storage dtype; the declared output dtype is applied
        here.
        """
        dtype_str = self.dtype_to_str(self._kernel_dtype)
        if self._packed_fold_applies(x):
            words = self.N // self._WORD_BYTES
            threads, vec = self._packed_fold_launch()
            program = self._packed_kernel(self.M, words, self.op_kind)
            counted = program(threads, vec)(x.view(torch.uint32).view(self.M, words))
            return counted.view(torch.bool)
        if self._needs_tiling:
            program = _logical_reduce_kernel_tiled(
                self.M, self.N, self.op_kind, dtype_str, self.config["tile_n"]
            )
        else:
            program = _logical_reduce_kernel(self.M, self.N, self.op_kind, dtype_str)
        counted = program(self.config["block_m"], self.config["threads"])(x)
        if self.op_kind == "count_nonzero":
            return counted.to(torch.int64)
        # 0 or 1 in int8 is bool's own representation, so this is a reinterpretation.
        return counted.view(torch.bool)


class LogicalReduceEdgeFusedKernel(Kernel):
    """Fused logical reduction for a prefix and suffix axis set.

    ``applies`` states which devices reach it. One block reduces one kept column,
    walking the leading axis serially while reducing each contiguous trailing
    row. The general logical reducer remains responsible for non-edge layouts
    and for rows that require N-tiling.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call: LogicalReduceCall) -> bool:
        return logical_edge_fused_region(call)

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        reduce_axes: "tuple[int, ...]",
        keepdim: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        if op_kind not in _LOGICAL_REDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_LOGICAL_REDUCE_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.reduce_axes = tuple(reduce_axes)
        self.keepdim = keepdim
        self._kernel_dtype = storage_dtype_for(dtype)
        self._elem_bytes = torch.tensor([], dtype=self._kernel_dtype).element_size()
        self._smem_budget = device_smem_budget(device_index)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """No width: it follows ``trail``, which only the call's shape carries.

        One block holds a ``trail``-wide fp32 fragment, so a fixed width hands
        every lane ``trail / threads`` registers and the occupancy falls as the
        row grows. ``forward`` reads the row and fills the width in; a caller
        that states one keeps it.
        """
        return {"threads": None}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_storage(x)
        k, j = edge_axis_split(x.ndim, self.reduce_axes)
        if not k:
            raise ValueError("LogicalReduceEdgeFusedKernel requires edge reduction axes")
        lead, kept, trail, planner, cfg = edge_axis_plan(
            tuple(x.shape), k, j, self._elem_bytes, self._smem_budget
        )
        if planner.needs_tiling:
            raise ValueError("LogicalReduceEdgeFusedKernel requires an untiled trailing pass")
        threads = self.config.get("threads") or fused_edge_threads(trail)
        dtype_str = self.dtype_to_str(self._kernel_dtype)
        fused = _logical_reduce_edge_fused(lead, kept, trail, self.op_kind, dtype_str)
        counted = fused(threads)(x.reshape(lead, kept, trail))
        y = counted if self.op_kind == "count_nonzero" else counted.view(torch.bool)
        return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)
