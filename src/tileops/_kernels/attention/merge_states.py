"""Merge two normalized attention states with natural-log partition weights."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.call_spec import (
    MergeAttentionStatesCall,
    MergeAttentionStatesFwdInterface,
)
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel, vector_aligned

__all__ = ["MergeAttentionStatesKernel"]


@functools.lru_cache(maxsize=32)
def _merge_states_kernel(dim, dtype, cols, vector, tail):
    # Bounds are checked only where written here: TileLang's own checks test the dynamic row
    # count ahead of every load, which delays the loads of a single-wave launch.
    @tilelang.jit(
        out_idx=[4, 5],
        compile_flags=["--use_fast_math"],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True},
    )
    def _func(block_rows, threads):
        rows = T.dynamic("rows")
        # Adjacent threads load adjacent vectors of a row; inference from the
        # one-dimensional weights alone would serialize the value columns.
        layout = T.Fragment(
            (block_rows, cols),
            forward_thread_fn=lambda i, j: ((i * cols + j) // vector) % threads,
            forward_index_fn=lambda i, j: (
                (i * cols + j) // (vector * threads) * vector + j % vector
            ),
        )

        def live_row(row):
            # A tail-free program serves a row count the CTA tile divides.
            return row < rows if tail else True

        def live(row, j):
            return T.And(live_row(row), j < dim) if dim < cols else live_row(row)

        @T.prim_func
        def main(
            Va: T.Tensor((rows, dim), dtype),
            Sa: T.Tensor((rows,), "float32"),
            Vb: T.Tensor((rows, dim), dtype),
            Sb: T.Tensor((rows,), "float32"),
            V: T.Tensor((rows, dim), dtype),
            S: T.Tensor((rows,), "float32"),
        ):
            # A tail-free program covers whole tiles only, so it never reaches past the rows.
            tiles = T.ceildiv(rows, block_rows) if tail else rows // block_rows
            with T.Kernel(tiles, threads=threads) as bx:
                values = T.alloc_fragment((block_rows, cols), dtype)
                # Issue independent value loads before the per-row LSE arithmetic.
                va = T.alloc_fragment((block_rows, cols), dtype)
                vb = T.alloc_fragment((block_rows, cols), dtype)
                T.annotate_layout({values: layout, va: layout, vb: layout})
                for i, j in T.Parallel(block_rows, cols):
                    if live(bx * block_rows + i, j):
                        va[i, j] = Va[bx * block_rows + i, j]
                for i, j in T.Parallel(block_rows, cols):
                    if live(bx * block_rows + i, j):
                        vb[i, j] = Vb[bx * block_rows + i, j]
                wa = T.alloc_fragment((block_rows,), "float32")
                wb = T.alloc_fragment((block_rows,), "float32")
                lse = T.alloc_fragment((block_rows,), "float32")
                empty_a = T.alloc_fragment((block_rows,), "bool")
                empty_b = T.alloc_fragment((block_rows,), "bool")
                for i in T.Parallel(block_rows):
                    row = bx * block_rows + i
                    a = T.if_then_else(live_row(row), Sa[row], -T.infinity("float32"))
                    b = T.if_then_else(live_row(row), Sb[row], -T.infinity("float32"))
                    empty_a[i] = a == -T.infinity("float32")
                    empty_b[i] = b == -T.infinity("float32")
                    maximum = T.max(a, b)
                    safe_max = T.if_then_else(maximum == -T.infinity("float32"), 0, maximum)
                    ea = T.exp(a - safe_max)
                    eb = T.exp(b - safe_max)
                    denominator = ea + eb
                    inverse = 1.0 / T.max(denominator, 1.0)
                    wa[i] = ea * inverse
                    wb[i] = eb * inverse
                    lse[i] = T.log(denominator) + safe_max
                for i, j in T.Parallel(block_rows, cols):
                    if live(bx * block_rows + i, j):
                        # Empty states contribute nothing, even if their unused V is NaN.
                        a = T.if_then_else(empty_a[i], 0, T.cast(va[i, j], "float32"))
                        b = T.if_then_else(empty_b[i], 0, T.cast(vb[i, j], "float32"))
                        values[i, j] = a * wa[i] + b * wb[i]
                for i, j in T.Parallel(block_rows, cols):
                    if live(bx * block_rows + i, j):
                        V[bx * block_rows + i, j] = values[i, j]
                # Each row's LSE is replicated over the row's threads; shared memory hands
                # the tile's LSEs to adjacent lanes for one coalesced store.
                lse_tile = T.alloc_shared((block_rows,), "float32")
                T.copy(lse, lse_tile)
                for i in T.Parallel(block_rows):
                    if live_row(bx * block_rows + i):
                        S[bx * block_rows + i] = lse_tile[i]

        return main

    return _func


class MergeAttentionStatesKernel(Kernel, MergeAttentionStatesFwdInterface):
    """One coalesced pass; rows are dynamic, value width and dtype are specialized."""

    @classmethod
    def entry_for(cls, call: MergeAttentionStatesCall) -> Entry:
        index = call.device.index if call.device is not None else None
        return (call.dim, call.dtype, index), lambda: cls(call.dim, call.dtype, device_index=index)

    def __init__(self, dim, dtype, config=None, device_index=None):
        super().__init__(device_index=device_index)
        self.dtype = dtype
        self.dim = dim
        # The fragment layout and launch shape share this geometry.
        lanes = VECTOR_ACCESS_BYTES // dtype.itemsize
        self._vector = lanes if dim and dim % lanes == 0 else 1
        self._cols = tilelang.next_power_of_2(max(1, dim))
        self.init_config(config)

    @property
    def default_config(self):
        # Eight warps process one value vector per lane. Keep at least eight rows for wide
        # heads; narrow heads use more rows.
        threads = 256
        return {"block_rows": max(8, threads * self._vector // self._cols), "threads": threads}

    def forward(self, v_a, s_a, v_b, s_b):
        tokens, heads, dim = v_a.shape
        rows = tokens * heads
        if rows == 0:
            return torch.empty_like(v_a), torch.empty_like(s_a)
        v, s = self.program(self._compile, rows % self.config["block_rows"] != 0)(
            vector_aligned(v_a.contiguous()).view(rows, dim),
            vector_aligned(s_a.contiguous()).view(rows),
            vector_aligned(v_b.contiguous()).view(rows, dim),
            vector_aligned(s_b.contiguous()).view(rows),
        )
        return v.view(tokens, heads, dim), s.view(tokens, heads)

    def _compile(self, cfg: dict, tail: bool):
        """The program ``forward`` launches, compiled from *cfg*; *tail* guards a partial last tile."""
        return _merge_states_kernel(self.dim, self.dtype_str, self._cols, self._vector, tail)(**cfg)
