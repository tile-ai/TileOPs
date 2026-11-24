"""Concatenation along one axis: every input copied into its columns of the output rows."""

import dataclasses
import math
from abc import abstractmethod
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import MAX_GRID_YZ_BLOCKS, VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops._kernels.streaming_load import load16_evict_last

__all__ = ["ConcatCall", "ConcatFwdInterface", "ConcatKernel"]

# The inputs one launch copies: `_concat_kernel` declares that many tensor slots, x0 to x3,
# and `ConcatKernel.forward` groups the inputs by it.
_SLOTS = 4
# The threads of a CTA, which `_concat_kernel` launches and `ConcatKernel` counts CTAs by.
# Small CTAs spread a decode row's few hundred vectors over the SMs, and 128 threads divide
# every architecture's per-SM thread limit, so whole CTAs fill an SM.
_THREADS = 128


@dataclasses.dataclass(frozen=True)
class ConcatCall(CallSpec):
    """The joined axis, each input's row width and dtype, and the output dtype.

    An input's row width is the product of its extents from ``dim`` on: the contiguous run of
    elements it holds for one index of the axes before ``dim``.
    """

    dim: int = 0
    widths: tuple[int, ...] = ()
    dtypes: tuple[torch.dtype, ...] = ()
    out_dtype: torch.dtype | None = None


class ConcatFwdInterface(KernelInterface):
    """Tensors joined along one axis into a new tensor."""

    request = ConcatCall

    @abstractmethod
    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Join *tensors* along ``call.dim``.

        Args:
            tensors: Contiguous tensors on ``call.device`` of one rank, equal in every axis
                but ``call.dim``. Element ``j`` has dtype ``call.dtypes[j]`` and row width
                ``call.widths[j]``.

        Returns:
            A new contiguous tensor in ``call.out_dtype``: each input converted to it and
            placed after the previous inputs along ``call.dim``.
        """


@target_cache(maxsize=64)
def _concat_kernel(vec, dtypes, out_dtype, per_thread):
    """One launch: up to ``_SLOTS`` inputs copied into the output viewed as ``[rows, total, vec]``.

    Input ``k``, viewed as ``[rows, cols_k, vec]``, lands at the vectors from ``base`` plus the
    widths of the inputs before it. Each thread copies ``per_thread[k]`` of its vectors, one
    CTA width apart. Rows, widths and ``base`` are dynamic.
    """
    n = len(dtypes)
    assert n <= _SLOTS, f"a launch copies at most {_SLOTS} inputs, got {n}"
    rows = T.dynamic("rows", "int64")
    total = T.dynamic("total", "int64")
    cols = [T.dynamic(f"cols{k}", "int64") for k in range(n)]
    # Input k owns the column tiles from first[k] to first[k + 1], so a CTA's input is a
    # compare of its tile index, the same for every thread of the CTA.
    first, offset = [0], [0]
    for k in range(n):
        first.append(first[k] + T.ceildiv(cols[k], _THREADS * per_thread[k]))
        offset.append(offset[k] + cols[k])
    # An input whose vector is a whole 16-byte access reads it with an L2 hint.
    whole = [vec * DataType(d).bits == VECTOR_ACCESS_BYTES * 8 for d in dtypes]
    # TileLang lowers a vector assignment from a local array to 4-byte stores; its st.global
    # intrinsic of the vector's width writes the vector in one instruction.
    store_bits = vec * DataType(out_dtype).bits
    store, word = {
        32: (T.stg32, "uint32"),
        64: (T.stg64, "uint32x2"),
        128: (T.stg128, "uint32x4"),
    }.get(store_bits, (None, None))

    def slot(k):
        return T.Tensor((rows, cols[k], vec), dtypes[k]) if k < n else None

    @T.macro
    def copy(out, x, k, t, r, base):
        if t >= first[k] and t < first[k + 1]:
            held = T.alloc_local((per_thread[k], vec), dtypes[k])
            conv = T.alloc_local((vec,), out_dtype)
            lane = (t - first[k]) * _THREADS * per_thread[k] + T.get_thread_binding()
            # Every read of the thread issues before its first write.
            for u in T.unroll(per_thread[k]):
                c = lane + u * _THREADS
                if c < cols[k]:
                    if whole[k]:
                        # Evict-last: the read lines stay in L2 while the output's lines go first.
                        load16_evict_last(held, (u, 0), x, (r, c, 0))
                    else:
                        for v in T.vectorized(vec):
                            held[u, v] = x[r, c, v]
            for u in T.unroll(per_thread[k]):
                c = lane + u * _THREADS
                if c < cols[k]:
                    for v in T.vectorized(vec):
                        conv[v] = T.cast(held[u, v], out_dtype)
                    if store is None:
                        for v in T.vectorized(vec):
                            out[r, base + offset[k] + c, v] = conv[v]
                    else:
                        value = conv[0] if vec == 1 else conv[0:vec]
                        store(out[r, base + offset[k] + c, 0:vec], T.reinterpret(value, word))

    @T.prim_func
    def concat(
        out: T.Tensor((rows, total, vec), out_dtype),  # type: ignore
        base: T.int64,  # type: ignore
        x0: slot(0),  # type: ignore
        x1: slot(1) = None,  # type: ignore
        x2: slot(2) = None,  # type: ignore
        x3: slot(3) = None,  # type: ignore
    ):
        # Column tiles vary fastest, so consecutive CTAs walk an output row in order.
        with T.Kernel(first[n], rows, threads=_THREADS) as (bx, r):
            t = T.cast(bx, "int64")
            copy(out, x0, 0, t, r, base)
            if n > 1:
                copy(out, x1, 1, t, r, base)
            if n > 2:
                copy(out, x2, 2, t, r, base)
            if n > 3:
                copy(out, x3, 3, t, r, base)

    return tilelang.compile(
        concat,
        out_idx=[],
        compile_flags=csrc_include("streaming_load.h"),
        # Every access is inside its tensor by the `c < cols[k]` guard; the checks TileLang
        # would add for the dynamic `base` repeat it on each store.
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True},
    )


@target_cache(maxsize=64)
def _concat_flat_kernel(vec, dtype, n, per_thread, resident):
    """One launch of ``n`` same-dtype inputs over the flat output, viewed as ``rows * total``
    16-byte vectors cut into runs of ``_THREADS * per_thread``, one a CTA, so a warp's stores
    fill whole lines. A vector's row is its position times ``mul``, high word, shifted by
    ``shr`` (the position itself where ``mul`` is 0); its input is a compare of its column.

    Without *resident*, each input's load of a vector issues into its own registers. With it,
    for one or two inputs, a vector is one load from the address its column compare picks, and
    nvcc holds the program to *resident* CTAs an SM.
    """
    rows = T.dynamic("rows", "int32")
    total = T.dynamic("total", "int32")
    cols = [T.dynamic(f"cols{k}", "int32") for k in range(n)]
    offset = [0]
    for k in range(n):
        offset.append(offset[k] + cols[k])

    def slot(k):
        return T.Tensor((rows, cols[k], vec), dtype) if k < n else None

    def inside(k, col):
        return col < offset[k + 1] if k == 0 else T.And(col >= offset[k], col < offset[k + 1])

    @T.macro
    def load(x, k, held, u, row, col):
        if inside(k, col):
            # Evict-last, as the per-row program reads a whole vector.
            load16_evict_last(held, (u, 0), x, (row, col - offset[k], 0))

    @T.macro
    def pick(k, held, value, u, col):
        if inside(k, col):
            for v in T.vectorized(vec):
                value[v] = held[u, v]

    @T.prim_func
    def concat(
        out: T.Tensor((rows, total, vec), dtype),  # type: ignore
        mul: T.uint32,  # type: ignore
        shr: T.uint32,  # type: ignore
        x0: slot(0),  # type: ignore
        x1: slot(1) = None,  # type: ignore
        x2: slot(2) = None,  # type: ignore
        x3: slot(3) = None,  # type: ignore
    ):
        with T.Kernel(T.ceildiv(rows * total, _THREADS * per_thread), threads=_THREADS) as bx:
            row = T.alloc_local((per_thread,), "int32")
            col = T.alloc_local((per_thread,), "int32")
            for u in T.unroll(per_thread):
                g = (bx * per_thread + u) * _THREADS + T.get_thread_binding()
                q = T.call_pure_extern("uint32", "__umulhi", T.cast(g, "uint32"), mul)
                row[u] = T.if_then_else(mul == 0, g, T.cast(T.shift_right(q, shr), "int32"))
                col[u] = g - row[u] * total
            # Every read of the thread issues before its first write.
            if resident is not None:
                # The address pick below reads input 1 for every column past input 0.
                assert n <= 2, f"the held program copies one or two inputs, got {n}"
                T.annotate_min_blocks_per_sm(resident)
                held = T.alloc_local((per_thread, vec), dtype)
                for u in T.unroll(per_thread):
                    if row[u] < rows:
                        T.call_extern(
                            "handle",
                            "tileops::load16_evict_last",
                            T.address_of(held[u, 0]),
                            T.address_of(x0[row[u], col[u], 0])
                            if n == 1
                            else T.if_then_else(
                                col[u] < offset[1],
                                T.address_of(x0[row[u], col[u], 0]),
                                T.address_of(x1[row[u], col[u] - offset[1], 0]),
                            ),
                        )
                for u in T.unroll(per_thread):
                    if row[u] < rows:
                        T.stg128(
                            out[row[u], col[u], 0:vec], T.reinterpret(held[u, 0:vec], "uint32x4")
                        )
            else:
                # One register array an input: the loads of a vector's inputs, of which one
                # reads, write different registers and issue together.
                held0 = T.alloc_local((per_thread, vec), dtype)
                held1 = T.alloc_local((per_thread, vec), dtype)
                held2 = T.alloc_local((per_thread, vec), dtype)
                held3 = T.alloc_local((per_thread, vec), dtype)
                value = T.alloc_local((vec,), dtype)
                for u in T.unroll(per_thread):
                    if row[u] < rows:
                        load(x0, 0, held0, u, row[u], col[u])
                        if n > 1:
                            load(x1, 1, held1, u, row[u], col[u])
                        if n > 2:
                            load(x2, 2, held2, u, row[u], col[u])
                        if n > 3:
                            load(x3, 3, held3, u, row[u], col[u])
                for u in T.unroll(per_thread):
                    if row[u] < rows:
                        pick(0, held0, value, u, col[u])
                        if n > 1:
                            pick(1, held1, value, u, col[u])
                        if n > 2:
                            pick(2, held2, value, u, col[u])
                        if n > 3:
                            pick(3, held3, value, u, col[u])
                        T.stg128(
                            out[row[u], col[u], 0:vec], T.reinterpret(value[0:vec], "uint32x4")
                        )

    return tilelang.compile(
        concat,
        out_idx=[],
        compile_flags=csrc_include("streaming_load.h"),
        # Every access is inside its tensor by the row and column compares.
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True},
    )


class ConcatKernel(Kernel, ConcatFwdInterface):
    """Each input copied into its columns of the output rows, converted on the way.

    Viewed as ``[rows, total]``, the output takes input ``j`` at the columns after the
    previous inputs'. A vector is the most elements, up to 16 bytes of the widest dtype, that
    divide every input's row width. A CTA copies one column tile of one input row. A program
    is built per vector, input dtypes and vectors per thread, and serves every row count and
    width. Inputs share a launch ``_SLOTS`` at a time. A launch past one resident wave that
    copies every input, unconverted, in 16-byte vectors into output rows that end inside a line
    takes ``_concat_flat_kernel`` instead, while its positions fit 32 bits. Such a launch of at
    most two inputs that twice the flat program's vectors a thread bring within one wave takes
    twice the vectors.
    """

    # Vectors a thread of the flat program copies by default, one CTA width apart, so the
    # second row lookup overlaps the first read.
    _FLAT_PER_THREAD: ClassVar[int] = 2

    @classmethod
    def entry_for(cls, call: ConcatCall) -> Entry:
        widest = max(d.itemsize for d in (*call.dtypes, call.out_dtype))
        vec = math.gcd(*call.widths, VECTOR_ACCESS_BYTES // widest)
        args = (call.dim, vec, call.dtypes, call.out_dtype)
        index = call.device_index
        resident = call.threads_per_sm // _THREADS
        return (*args, index), lambda: cls(
            *args, device_index=index, sm_count=call.sm_count, resident_ctas=resident
        )

    def __init__(
        self,
        dim: int,
        vec: int,
        dtypes: tuple[torch.dtype, ...],
        out_dtype: torch.dtype,
        device_index: int | None = None,
        *,
        sm_count: int,
        resident_ctas: int,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dim, self.vec, self.dtypes, self.dtype = dim, vec, dtypes, out_dtype
        self.sm_count, self.resident_ctas = sm_count, resident_ctas
        self.init_config()

    def _compile(self, cfg: dict, dtypes: tuple[str, ...], per_thread: tuple[int, ...]):
        """The program of a launch whose inputs have *dtypes* and *per_thread* vectors."""
        return _concat_kernel(self.vec, dtypes, self.dtype_str, per_thread)

    def _compile_flat(self, cfg: dict, inputs: int, per_thread: int, resident: int | None):
        """The program of one launch of *inputs* inputs over the flat output."""
        return _concat_flat_kernel(self.vec, self.dtype_str, inputs, per_thread, resident)

    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        shape = list(tensors[0].shape)
        shape[self.dim] = sum(t.shape[self.dim] for t in tensors)
        out = torch.empty(shape, dtype=self.dtype, device=tensors[0].device)
        rows = math.prod(shape[: self.dim])
        widths = [math.prod(t.shape[self.dim :]) // self.vec for t in tensors]
        flat = out.view(rows, sum(widths), self.vec)
        views = [t.view(rows, w, self.vec) for t, w in zip(tensors, widths, strict=True)]
        # An input of width 0 holds no element, so no launch carries it and no grid is empty.
        present = [j for j, w in enumerate(widths) if w]
        groups = [present[first : first + _SLOTS] for first in range(0, len(present), _SLOTS)]
        # One resident wave: the CTAs the device's thread limit keeps on its SMs at once.
        wave = self.sm_count * self.resident_ctas
        # The vectors of each input that hold 16 bytes.
        most = [VECTOR_ACCESS_BYTES // (self.vec * d.itemsize) for d in self.dtypes]
        total = sum(widths)
        # Rows that end inside a 128-byte L2 line make per-row tiles store lines in part; past
        # one wave, CTAs over the flat output store whole lines for a row lookup a vector.
        line_bytes = 128
        if (
            total * self.vec * self.dtype.itemsize % line_bytes
            and len(groups) == 1
            and self.vec * self.dtype.itemsize == VECTOR_ACCESS_BYTES
            and all(d == self.dtype for d in self.dtypes)
            # The positions, rounded up to a whole CTA run, stay below 2**31.
            and rows * total + _THREADS * self._FLAT_PER_THREAD <= 2**31
            and rows * sum(-(-widths[j] // _THREADS) for j in present) > wave
        ):
            # Row = (position * mul) >> (32 + shr) for every position below 2**31.
            shift = (total - 1).bit_length()
            mul = -(-(1 << (31 + shift)) // total) if total > 1 else 0
            per_thread, resident = self._FLAT_PER_THREAD, None
            # A launch past one resident wave runs its last CTAs after the first ones retire.
            # Twice the vectors a thread bring it within one wave when the CTAs stay resident:
            # one load a vector, from the address its column picks, held to the registers every
            # resident CTA leaves a thread. ptxas fits that without spills for one or two
            # inputs; with three it spills (12 bytes at 32 registers on sm_90).
            runs = [-(-rows * total // (_THREADS * p)) for p in (per_thread, 2 * per_thread)]
            if len(present) <= 2 and runs[0] > wave >= runs[1]:
                per_thread, resident = 2 * per_thread, self.resident_ctas
            self.program(self._compile_flat, len(present), per_thread, resident)(
                flat, mul, shift - 1 if total > 1 else 0, *(views[j] for j in present)
            )
            return out
        # A launch covers at most MAX_GRID_YZ_BLOCKS rows, the grid's y extent.
        for first in range(0, rows, MAX_GRID_YZ_BLOCKS):
            last = min(first + MAX_GRID_YZ_BLOCKS, rows)
            for group in groups:
                kinds = tuple(self.dtype_to_str(self.dtypes[j]) for j in group)

                def ctas(per: list[int], group=group, rows: int = last - first) -> int:
                    return sum(
                        rows * -(-widths[j] // (_THREADS * p))
                        for j, p in zip(group, per, strict=True)
                    )

                # Each thread takes more vectors of every input, doubling them together, while
                # the launch has more CTAs than SMs and the doubling removes CTAs. Beyond one
                # resident wave the launch waits on bandwidth, and an input stops at 16 bytes in
                # flight, the widest read a thread issues. Within one wave it waits on CTA starts
                # and one DRAM round trip, and an input stops at eight vectors (eight 16-byte
                # vectors hold 32 registers of data).
                per = [1] * len(group)
                while (now := ctas(per)) > self.sm_count:
                    grown = [
                        min(2 * p, most[j] if now > wave else 8)
                        for j, p in zip(group, per, strict=True)
                    ]
                    if ctas(grown) >= now:
                        break
                    per = grown
                per_thread = tuple(per)
                self.program(self._compile, kinds, per_thread)(
                    flat[first:last],
                    sum(widths[: group[0]]),
                    *(views[j][first:last] for j in group),
                )
        return out
