"""Merge two normalized attention states with natural-log partition weights."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import (
    MergeAttentionStatesCall,
    MergeAttentionStatesFwdInterface,
)
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned

__all__ = ["MergeAttentionStatesKernel"]


@functools.lru_cache(maxsize=32)
def _merge_states_kernel(dim, dtype, cols, vector):
    @tilelang.jit(out_idx=[4, 5], compile_flags=["--use_fast_math"])
    def _func(block_rows, threads):
        rows = T.dynamic("rows")
        # Adjacent threads load adjacent vectors of a row; inference from the
        # one-dimensional weights alone would serialize the value columns.
        layout = T.Fragment(
            (block_rows, cols),
            forward_thread_fn=lambda i, j: ((i * cols + j) // vector) % threads,
            forward_index_fn=lambda i, j: (i * cols + j) // (vector * threads) * vector
            + j % vector,
        )

        @T.prim_func
        def main(
            Va: T.Tensor((rows, dim), dtype),
            Sa: T.Tensor((rows,), "float32"),
            Vb: T.Tensor((rows, dim), dtype),
            Sb: T.Tensor((rows,), "float32"),
            V: T.Tensor((rows, dim), dtype),
            S: T.Tensor((rows,), "float32"),
        ):
            with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as bx:
                values = T.alloc_fragment((block_rows, cols), dtype)
                T.annotate_layout({values: layout})
                # Issue independent value loads before the per-row LSE arithmetic.
                va = T.alloc_fragment((block_rows, cols), dtype)
                vb = T.alloc_fragment((block_rows, cols), dtype)
                T.annotate_layout({va: layout, vb: layout})
                T.copy(Va[bx * block_rows : (bx + 1) * block_rows, :], va)
                T.copy(Vb[bx * block_rows : (bx + 1) * block_rows, :], vb)
                wa = T.alloc_fragment((block_rows,), "float32")
                wb = T.alloc_fragment((block_rows,), "float32")
                lse = T.alloc_fragment((block_rows,), "float32")
                empty_a = T.alloc_fragment((block_rows,), "bool")
                empty_b = T.alloc_fragment((block_rows,), "bool")
                for i in T.Parallel(block_rows):
                    row = bx * block_rows + i
                    a = T.if_then_else(row < rows, Sa[row], -T.infinity("float32"))
                    b = T.if_then_else(row < rows, Sb[row], -T.infinity("float32"))
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
                    if bx * block_rows + i < rows and j < dim:
                        # Empty states contribute nothing, even if their unused V is NaN.
                        a = T.if_then_else(empty_a[i], 0, T.cast(va[i, j], "float32"))
                        b = T.if_then_else(empty_b[i], 0, T.cast(vb[i, j], "float32"))
                        values[i, j] = a * wa[i] + b * wb[i]
                for i, j in T.Parallel(block_rows, cols):
                    if bx * block_rows + i < rows and j < dim:
                        V[bx * block_rows + i, j] = values[i, j]
                T.copy(lse, S[bx * block_rows : (bx + 1) * block_rows])

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
        self.kernel = _merge_states_kernel(dim, self.dtype_str, self._cols, self._vector)
        self.init_config(config)

    @property
    def default_config(self):
        # Four warps process two value vectors per lane to amortize CTA scheduling.
        # Keep at least eight rows for wide heads; narrow heads use more rows.
        threads = 128
        return {"block_rows": max(8, 2 * threads * self._vector // self._cols), "threads": threads}

    def forward(self, v_a, s_a, v_b, s_b):
        tokens, heads, dim = v_a.shape
        rows = tokens * heads
        if rows == 0:
            return torch.empty_like(v_a), torch.empty_like(s_a)
        v, s = self.kernel(**self.config)(
            vector_aligned(v_a.contiguous()).view(rows, dim),
            vector_aligned(s_a.contiguous()).view(rows),
            vector_aligned(v_b.contiguous()).view(rows, dim),
            vector_aligned(s_b.contiguous()).view(rows),
        )
        return v.view(tokens, heads, dim), s.view(tokens, heads)
