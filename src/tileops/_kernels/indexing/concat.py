"""Concatenation as one vectorized copy per input, several inputs to a launch."""

import math

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.indexing.call_spec import ConcatCall, ConcatFwdInterface
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = ["ConcatKernel"]

# The input slots of one program: `_concat_main` declares `x0` to `x{_SLOTS - 1}` and copies
# each slot below `n`, and `ConcatKernel` groups the inputs by it.
_SLOTS = 4


@target_cache(maxsize=64)
def _concat_kernel(vec, dtypes, out_dtype):
    """Compile the copy of input ``j``, viewed as ``[rows, cols_j, vec]``, to the vectors from
    ``base`` plus the previous inputs' ``cols`` of the ``[rows, total, vec]`` output."""
    n = len(dtypes)
    # A thread moves one vector, so the block size only spreads the vectors over the SMs:
    # small blocks reach more SMs when a decode row holds a few hundred vectors.
    threads = 128
    rows = T.dynamic("rows", "int64")
    total = T.dynamic("total", "int64")
    cols = [T.dynamic(f"cols{j}", "int64") for j in range(n)]
    vectors = [rows * cols[j] for j in range(n)]
    # Input j takes blocks starts[j] up to starts[j + 1], and output vectors from offsets[j].
    starts, offsets = [0], [0]
    for j in range(n):
        starts.append(starts[j] + T.ceildiv(vectors[j], threads))
        offsets.append(offsets[j] + cols[j])

    def slot(j):
        return T.Tensor((rows, cols[j], vec), dtypes[j]) if j < n else None

    @T.macro
    def _copy(out, x, j, bx, base):
        if bx >= starts[j] and bx < starts[j + 1]:
            # The thread index, not a `T.Parallel` loop, so that each thread moves one
            # vector whatever layout inference would choose.
            k = T.cast(bx - starts[j], "int64") * threads + T.get_thread_binding()
            if k < vectors[j]:
                r = k // cols[j]
                c = k - r * cols[j]
                for v in T.vectorized(vec):
                    out[r, base + offsets[j] + c, v] = T.cast(x[r, c, v], out_dtype)

    @T.prim_func
    def _concat_main(
        out: T.Tensor((rows, total, vec), out_dtype),  # type: ignore
        base: T.int64,  # type: ignore
        x0: slot(0),  # type: ignore
        x1: slot(1) = None,  # type: ignore
        x2: slot(2) = None,  # type: ignore
        x3: slot(3) = None,  # type: ignore
    ):
        # One idle block past the last input's, so that a group of empty inputs still
        # launches a grid; TileLang takes a dynamic extent as positive and folds a max.
        with T.Kernel(starts[n] + 1, threads=threads) as bx:
            _copy(out, x0, 0, bx, base)
            if n > 1:
                _copy(out, x1, 1, bx, base)
            if n > 2:
                _copy(out, x2, 2, bx, base)
            if n > 3:
                _copy(out, x3, 3, bx, base)

    return tilelang.compile(_concat_main, out_idx=[])


class ConcatKernel(Kernel, ConcatFwdInterface):
    """Each input copied to its columns of the output, converted on the way.

    Viewed as ``[rows, total]``, the output takes input ``j`` at the columns after the
    previous inputs'. A thread moves one vector: the most elements, up to 16 bytes loaded
    and stored, that divide every input's width, so a vector never crosses a row and starts
    aligned. Row counts and widths are dynamic: one program serves every shape with the
    same vector, dtypes and input count. Inputs share a launch in groups of ``_SLOTS``.
    """

    @classmethod
    def entry_for(cls, call: ConcatCall) -> Entry:
        widest = max(d.itemsize for d in (*call.dtypes, call.out_dtype))
        args = (
            call.dim,
            math.gcd(*call.widths, VECTOR_ACCESS_BYTES // widest),
            call.dtypes,
            call.out_dtype,
        )
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
        self.groups = [
            range(first, min(first + _SLOTS, len(dtypes)))
            for first in range(0, len(dtypes), _SLOTS)
        ]
        self.init_config()

    def _compile(self, cfg: dict, group: int):
        """The program of input group *group*."""
        return _concat_kernel(
            self.vec,
            tuple(self.dtype_to_str(self.dtypes[j]) for j in self.groups[group]),
            self.dtype_str,
        )

    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        shape = list(tensors[0].shape)
        shape[self.dim] = sum(t.shape[self.dim] for t in tensors)
        out = torch.empty(shape, dtype=self.dtype, device=tensors[0].device)
        rows = math.prod(shape[: self.dim])
        cols = [math.prod(t.shape[self.dim :]) // self.vec for t in tensors]
        flat = out.view(rows, sum(cols), self.vec)
        for g, group in enumerate(self.groups):
            self.program(self._compile, g)(
                flat,
                sum(cols[: group.start]),
                *(tensors[j].view(rows, cols[j], self.vec) for j in group),
            )
        return out
