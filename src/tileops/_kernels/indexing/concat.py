"""Concatenation along one axis: every input copied into its columns of the output rows."""

import dataclasses
import math
from abc import abstractmethod

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
def _concat_kernel(vec, dtypes, out_dtype):
    """One launch: up to ``_SLOTS`` inputs copied into the output viewed as ``[rows, total, vec]``.

    Input ``k``, viewed as ``[rows, cols_k, vec]``, lands at the vectors from ``base`` plus the
    widths of the inputs before it. Rows, widths and ``base`` are dynamic.
    """
    n = len(dtypes)
    assert n <= _SLOTS, f"a launch copies at most {_SLOTS} inputs, got {n}"
    # Small CTAs spread a decode row's few hundred vectors over the SMs, and 16 of them fill
    # an SM's 2048 resident threads.
    threads = 128
    # A thread copies `per_thread` vectors, one CTA width apart, so each warp instruction stays
    # contiguous. An input narrower than the output reads that fraction of each vector's
    # output bytes; with a whole 16-byte output vector, `per_thread` restores 16 bytes of the
    # launch's narrowest input per thread.
    per_thread = DataType(out_dtype).bits // min(DataType(d).bits for d in dtypes)
    span = threads * per_thread
    rows = T.dynamic("rows", "int64")
    total = T.dynamic("total", "int64")
    cols = [T.dynamic(f"cols{k}", "int64") for k in range(n)]
    # Input k owns the column tiles from first[k] to first[k + 1], so a CTA's input is a
    # compare of its tile index, the same for every thread of the CTA.
    first, offset = [0], [0]
    for k in range(n):
        first.append(first[k] + T.ceildiv(cols[k], span))
        offset.append(offset[k] + cols[k])
    # An input whose vector is a whole 16-byte access reads it with an L2 hint.
    whole = [vec * DataType(d).bits == VECTOR_ACCESS_BYTES * 8 for d in dtypes]
    # An output vector that is a whole 16-byte access is written by one 16-byte store.
    whole_out = vec * DataType(out_dtype).bits == VECTOR_ACCESS_BYTES * 8

    def slot(k):
        return T.Tensor((rows, cols[k], vec), dtypes[k]) if k < n else None

    @T.macro
    def copy(out, x, k, t, r, base):
        if t >= first[k] and t < first[k + 1]:
            held = T.alloc_local((per_thread, vec), dtypes[k])
            lane = (t - first[k]) * span + T.get_thread_binding()
            # All reads issue before the first write, so they are in flight together.
            for u in T.unroll(per_thread):
                c = lane + u * threads
                if c < cols[k]:
                    if whole[k]:
                        # Evict-last: the read lines stay in L2 while the output's lines go first.
                        load16_evict_last(held, (u, 0), x, (r, c, 0))
                    else:
                        for v in T.vectorized(vec):
                            held[u, v] = x[r, c, v]
            for u in T.unroll(per_thread):
                c = lane + u * threads
                if c < cols[k]:
                    if whole_out:
                        wide = T.alloc_local((vec,), out_dtype)
                        for v in T.vectorized(vec):
                            wide[v] = T.cast(held[u, v], out_dtype)
                        T.call_extern(
                            "handle",
                            "tileops::store16_global",
                            T.address_of(out[r, base + offset[k] + c, 0]),
                            T.address_of(wide[0]),
                        )
                    else:
                        for v in T.vectorized(vec):
                            out[r, base + offset[k] + c, v] = T.cast(held[u, v], out_dtype)

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
        with T.Kernel(first[n], rows, threads=threads) as (bx, r):
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
        compile_flags=csrc_include("streaming_load.h", "vector_store.h"),
        # Every access is inside its tensor by the `c < cols[k]` guard; the checks TileLang
        # would add for the dynamic `base` repeat it on each store.
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True},
    )


class ConcatKernel(Kernel, ConcatFwdInterface):
    """Each input copied into its columns of the output rows, converted on the way.

    Viewed as ``[rows, total]``, the output takes input ``j`` at the columns after the
    previous inputs'. A vector is the most elements, up to 16 bytes of the widest dtype, that
    divide every input's row width, so a vector never crosses a row or an input. A thread
    moves one vector, or two when the output is twice as wide as the narrowest input of its
    launch. A program is built for the vector and the dtypes of the inputs it copies; one
    program serves every row count and row width. Inputs share a launch ``_SLOTS`` at a time.
    """

    @classmethod
    def entry_for(cls, call: ConcatCall) -> Entry:
        widest = max(d.itemsize for d in (*call.dtypes, call.out_dtype))
        vec = math.gcd(*call.widths, VECTOR_ACCESS_BYTES // widest)
        args = (call.dim, vec, call.dtypes, call.out_dtype)
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        dim: int,
        vec: int,
        dtypes: tuple[torch.dtype, ...],
        out_dtype: torch.dtype,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dim, self.vec, self.dtypes, self.dtype = dim, vec, dtypes, out_dtype
        self.init_config()

    def _compile(self, cfg: dict, dtypes: tuple[str, ...]):
        """The program of a launch whose inputs have *dtypes*."""
        return _concat_kernel(self.vec, dtypes, self.dtype_str)

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
        # A launch covers at most MAX_GRID_YZ_BLOCKS rows, the grid's y extent.
        for first in range(0, rows, MAX_GRID_YZ_BLOCKS):
            last = min(first + MAX_GRID_YZ_BLOCKS, rows)
            for group in groups:
                kinds = tuple(self.dtype_to_str(self.dtypes[j]) for j in group)
                self.program(self._compile, kinds)(
                    flat[first:last],
                    sum(widths[: group[0]]),
                    *(views[j][first:last] for j in group),
                )
        return out
