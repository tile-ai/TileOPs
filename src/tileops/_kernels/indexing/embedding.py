"""Row gather of an embedding table by token index."""

import dataclasses
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface

__all__ = ["EmbeddingCall", "EmbeddingFwdInterface", "EmbeddingKernel"]


@dataclasses.dataclass(frozen=True)
class EmbeddingCall(CallSpec):
    """The row width, dtypes and renormalization of one lookup; the token count and table
    rows stay runtime extents."""

    width: int = 0
    index_dtype: torch.dtype | None = None
    dtype: torch.dtype | None = None
    max_norm: float | None = None
    norm_type: float = 2.0


class EmbeddingFwdInterface(KernelInterface):
    """Return ``weight[input]``: a row of ``weight`` per element of ``input``, after
    renormalizing those rows of ``weight`` in place when call.max_norm is set."""

    request = EmbeddingCall

    @abstractmethod
    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Gather on call.device.

        Args:
            input: Contiguous row ids [*S] in call.index_dtype, each in [0, weight.shape[0]).
            weight: Table [V, call.width] in call.dtype, of any strides; written only when
                call.max_norm is set.

        Returns:
            A new [*S, call.width] tensor in call.dtype.
        """


@target_cache(maxsize=32)
def _embedding_kernel(width, index_dtype, dtype):
    @tilelang.jit(out_idx=[])
    def _func(block_rows, block_cols, threads):
        # A table or an id tensor may hold more than 2^31 - 1 rows.
        tokens = T.dynamic("tokens", "int64")
        rows = T.dynamic("rows", "int64")
        col_blocks = T.ceildiv(width, block_cols)

        @T.prim_func
        def embedding(
            Out: T.Tensor((tokens, width), dtype),
            Index: T.Tensor((tokens,), index_dtype),
            Weight: T.Tensor((rows, width), dtype),
        ):
            # The tile index runs over x, then y: CUDA caps y at 65535 blocks, which a wide
            # table or a long id tensor passes, and x at 2^31 - 1. With x at 2^20, the grid
            # holds 2^20 * 65535 tiles, more than any GPU's memory holds ids or rows for.
            tiles = T.ceildiv(tokens, block_rows) * col_blocks
            x_blocks = T.min(tiles, 1 << 20)
            with T.Kernel(x_blocks, T.ceildiv(tiles, x_blocks), threads=threads) as (bx, by):
                tile = by * x_blocks + bx
                first_token = tile // col_blocks * block_rows
                first_col = tile % col_blocks * block_cols
                # Staging the ids in shared memory lets the copy below vectorize: TileLang
                # cannot plan a vector load whose row index is itself a global load.
                ids = T.alloc_shared((block_rows,), "int64")
                for r in T.Parallel(block_rows):
                    if first_token + r < tokens:
                        ids[r] = T.cast(Index[first_token + r], "int64")
                T.sync_threads()
                for r, c in T.Parallel(block_rows, block_cols):
                    if first_token + r < tokens and first_col + c < width:
                        Out[first_token + r, first_col + c] = Weight[ids[r], first_col + c]

        return embedding

    return _func


class EmbeddingKernel(Kernel, EmbeddingFwdInterface):
    """One copy stage, tiled over output rows and columns."""

    @classmethod
    def refusal(cls, call: EmbeddingCall) -> str | None:
        if call.max_norm is not None:
            return "max_norm renormalizes weight in place, which this kernel does not"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: EmbeddingCall) -> Entry:
        args = dict(width=call.width, index_dtype=call.index_dtype, dtype=call.dtype)
        index = call.device.index
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(self, width, index_dtype, dtype, config=None, device_index=None):
        super().__init__(device_index=device_index)
        self.width, self.index_dtype, self.dtype = width, index_dtype, dtype
        self.kernel = _embedding_kernel(width, self.dtype_to_str(index_dtype), self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self):
        # A 512-column slice of a 16-bit row is 64 16-byte vectors: 128 threads copy two rows
        # of the tile at a time, and a wide row spreads over several column CTAs.
        return {"block_rows": 4, "block_cols": 512, "threads": 128}

    @property
    def autotune_configs(self):
        return [
            dict(block_rows=r, block_cols=c, threads=t)
            for r, c, t in [
                (4, 512, 128),
                (1, 1024, 128),
                (2, 1024, 256),
                (8, 256, 128),
                (16, 128, 128),
                (8, 512, 256),
                (32, 64, 128),
            ]
        ]

    @property
    def autotune_supply_prog(self):
        def supply(params):
            device = torch.device("cuda", self.device_index)
            # The token count and table rows are runtime extents of one compiled program,
            # so one size tunes every call. Table and output each hold 2^24 elements, or one
            # row where a row is wider, so tuning memory follows the call's own row width.
            rows = tokens = max(1, (1 << 24) // self.width)
            index = torch.randint(0, rows, (tokens,), device=device).to(self.index_dtype)
            weight = torch.zeros((rows, self.width), dtype=self.dtype, device=device)
            out = torch.empty((tokens, self.width), dtype=self.dtype, device=device)
            return [out, index, weight]

        return supply

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = torch.empty((*input.shape, self.width), dtype=self.dtype, device=weight.device)
        program = self.program(lambda cfg: self.kernel(**cfg))
        program(out.view(-1, self.width), input.view(-1), weight.contiguous())
        return out
