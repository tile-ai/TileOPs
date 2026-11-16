"""Row gather of an embedding table: ``out[i, :] = weight[input[i], :]``."""

import dataclasses
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface

__all__ = ["EmbeddingCall", "EmbeddingFwdInterface", "EmbeddingKernel"]


@dataclasses.dataclass(frozen=True)
class EmbeddingCall(CallSpec):
    """The row width and dtypes of one lookup; the index count and table rows are dynamic."""

    dim: int = 0
    dtype: torch.dtype | None = None
    index_dtype: torch.dtype | None = None
    renorm: bool = False


class EmbeddingFwdInterface(KernelInterface):
    """Gather the table rows the indices name."""

    request = EmbeddingCall

    @abstractmethod
    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Copy row ``input[...]`` of *weight* for every index.

        Args:
            input: Indices in ``[0, V)``, any shape, in ``call.index_dtype``, contiguous.
            weight: The table ``[V, call.dim]`` in ``call.dtype``, contiguous.

        Returns:
            A new tensor ``[*input.shape, call.dim]`` in ``call.dtype`` on ``call.device``.
        """


@target_cache(maxsize=32)
def _embedding_kernel(dim, dtype, index_dtype):
    @tilelang.jit(out_idx=[2])
    def _func(block_rows, row_slots, vec, threads):
        total = T.dynamic("total", "int64")
        # Row offsets reach past 2**31 elements in a large table.
        rows = T.dynamic("rows", "int64")
        col_tiles = (dim + row_slots * vec - 1) // (row_slots * vec)
        # A grid's y axis holds 65535 blocks; a wider row's tiles take several passes.
        grid_cols = min(col_tiles, 65535)

        @T.prim_func
        def embedding(
            Index: T.Tensor((total,), index_dtype),
            Weight: T.Tensor((rows, dim), dtype),
            Out: T.Tensor((total, dim), dtype),
        ):
            with T.Kernel(T.ceildiv(total, block_rows), grid_cols, threads=threads) as (bx, by):
                tx = T.get_thread_binding()
                row = bx * block_rows + tx // row_slots
                if tx < block_rows * row_slots and row < total:
                    src = T.alloc_var("int64")
                    src = Index[row]
                    for p in T.serial(T.ceildiv(col_tiles, grid_cols)):
                        # A row of 2**31 elements or more passes int32 before the tail guard.
                        tile = T.cast(p * grid_cols + by, "int64")
                        col = (tile * row_slots + tx % row_slots) * vec
                        if col < dim:
                            for i in T.vectorized(vec):
                                Out[row, col + i] = Weight[src, col + i]

        return embedding

    return _func


class EmbeddingKernel(Kernel, EmbeddingFwdInterface):
    """A vector copy a thread; one CTA copies whole rows, or one column tile of a wide row."""

    @classmethod
    def refusal(cls, call: EmbeddingCall) -> str | None:
        if call.renorm:
            return "does not renormalize rows by max_norm"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: EmbeddingCall) -> Entry:
        index = call.device.index if call.device is not None else None
        return (call.dim, call.dtype, call.index_dtype, index), lambda: cls(
            call.dim, call.dtype, call.index_dtype, device_index=index
        )

    def __init__(self, dim, dtype, index_dtype, config=None, device_index=None):
        super().__init__(device_index=device_index)
        self.dim, self.dtype, self.index_dtype = dim, dtype, index_dtype
        self.kernel = _embedding_kernel(dim, self.dtype_str, self.dtype_to_str(index_dtype))
        self.init_config(config)

    @property
    def default_config(self):
        # A thread copies one vector, as wide as 16 bytes and the row width allow. A CTA
        # covers a column tile of a wide row, or as many narrow rows as fill its threads.
        # Four warps split a 5120-wide bf16 row over five CTAs, so a decode batch still
        # spreads over the SMs.
        threads = 128
        vec = VECTOR_ACCESS_BYTES // self.dtype.itemsize
        while self.dim % vec:
            vec //= 2
        slots = max(1, self.dim // vec)
        return {
            "block_rows": max(1, threads // slots),
            "row_slots": min(slots, threads),
            "vec": vec,
            "threads": threads,
        }

    def forward(self, input, weight):
        return self.program(self._compile)(input.view(-1), weight).view(*input.shape, self.dim)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)
