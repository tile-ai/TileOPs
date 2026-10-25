import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops._kernels.sequence_modeling.mhc.call_spec import MHCPostCall, MHCPostFwdInterface

__all__ = ["MHCPostKernel"]


@functools.lru_cache(maxsize=32)
def _mhc_post_kernel(batch: int, n_expand: int, c_x: int, x_dtype: str, vec: int):
    dtype = "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _mhc_func(rows: int, lanes: int):
        steps = 4 // vec
        block_c = lanes * 4

        @T.prim_func
        def mhc_post(
            x_layer_out: T.Tensor([batch, c_x], x_dtype),
            h_post: T.Tensor([batch, n_expand], dtype),
            x_res: T.Tensor([batch, n_expand * c_x], x_dtype),
            x_out: T.Tensor([batch, n_expand * c_x], x_dtype),
        ):
            with T.Kernel(
                batch, T.ceildiv(n_expand, rows), T.ceildiv(c_x, block_c), threads=rows * lanes
            ) as (bx, by, bz):
                tx = T.get_thread_binding()
                res = T.alloc_local([vec], x_dtype)
                layer = T.alloc_local([vec], x_dtype)
                out = T.alloc_local([vec], x_dtype)
                i = by * rows + tx // lanes
                for s in T.unroll(steps):
                    j = bz * block_c + (s * lanes + tx % lanes) * vec
                    if i < n_expand and j < c_x:
                        for v in T.vectorized(vec):
                            res[v] = x_res[bx, i * c_x + j + v]
                        for v in T.vectorized(vec):
                            layer[v] = x_layer_out[bx, j + v]
                        h = h_post[bx, i]
                        for v in T.vectorized(vec):
                            out[v] = T.cast(
                                h * T.cast(layer[v], dtype) + T.cast(res[v], dtype), x_dtype
                            )
                        for v in T.vectorized(vec):
                            x_out[bx, i * c_x + j + v] = out[v]

        return mhc_post

    return _mhc_func


class MHCPostKernel(Kernel, MHCPostFwdInterface):
    """One pass in registers, no shared memory: each output element reads its inputs once.

    A CTA takes one warp per expansion row, up to 32 rows, and 128 columns; each lane
    reads its 4 columns as one vector, or as 2 or 4 when 4 does not divide ``c_x``.
    """

    supported_archs: list[int] = [80, 89, 90]

    @classmethod
    def entry_for(cls, call: MHCPostCall) -> Entry:
        return call, lambda: cls(call.batch, call.n_expand, call.c_x, call.dtype)

    def __init__(
        self,
        batch,
        n_expand,
        c_x,
        dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.weights_dtype = torch.float32
        self.vec = next(v for v in (4, 2, 1) if c_x % v == 0)
        self.kernel = _mhc_post_kernel(
            self.batch, self.n_expand, self.c_x, self.dtype_str, self.vec
        )

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"rows": min(self.n_expand, 32), "lanes": 32}

    def forward(self, x_layer_out, h_post, x_res):
        return self.program(self._compile)(
            vector_aligned(x_layer_out), h_post, vector_aligned(x_res)
        )

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(cfg["rows"], cfg["lanes"])
