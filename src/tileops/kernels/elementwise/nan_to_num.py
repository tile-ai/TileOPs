"""The nan_to_num kernel."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.utils import str2dtype

from ._base import (
    ParametricUnaryKernel,
)
from ._dtype import _clamp_to_dtype_range

__all__ = [
    "NanToNumFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_nan_to_num_kernel(
    N, dtype, nan_val, posinf_val, neginf_val, output_dtype=None, is_fp8=False, threads=256, npt=8
):
    """Build nan_to_num kernel: replace NaN, +Inf, -Inf with given values.

    For non-fp8 dtypes, uses register_copy strategy: fragment load -> compute
    -> fragment store for coalesced memory access.
    """
    out_dtype = output_dtype or dtype
    # A clamp replaces the infinity tests only when the replacements are the
    # dtype's own ends; otherwise it would move finite values too.
    _info = torch.finfo(str2dtype[dtype]) if dtype in str2dtype else None
    clamps = bool(_info is not None and posinf_val == _info.max and neginf_val == _info.min)

    if is_fp8:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            val = x[idx]
                            v32 = T.cast(val, "float32")
                            nan_r = T.cast(nan_val, out_dtype)
                            pos_r = T.cast(posinf_val, out_dtype)
                            neg_r = T.cast(neginf_val, out_dtype)
                            pass_through = T.Cast(out_dtype, v32)
                            result = T.if_then_else(
                                T.isnan(v32),
                                nan_r,
                                T.if_then_else(
                                    T.isinf(v32),
                                    T.if_then_else(v32 > T.cast(0, "float32"), pos_r, neg_r),
                                    pass_through,
                                ),
                            )
                            y[idx] = result

            return main
    else:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        k = i * npt_arg + j
                        val = x_reg[k]
                        v32 = T.cast(val, "float32")
                        nan_r = T.cast(nan_val, val.dtype)
                        pos_r = T.cast(posinf_val, val.dtype)
                        neg_r = T.cast(neginf_val, val.dtype)
                        if clamps:
                            y_reg[k] = T.if_then_else(
                                T.isnan(v32),
                                nan_r,
                                T.cast(
                                    T.min(
                                        T.max(v32, T.cast(neginf_val, "float32")),
                                        T.cast(posinf_val, "float32"),
                                    ),
                                    val.dtype,
                                ),
                            )
                        else:
                            y_reg[k] = T.if_then_else(
                                T.isnan(v32),
                                nan_r,
                                T.if_then_else(
                                    T.isinf(v32),
                                    T.if_then_else(v32 > T.cast(0, "float32"), pos_r, neg_r),
                                    val,
                                ),
                            )
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

    return kernel


class NanToNumFwdKernel(ParametricUnaryKernel):
    """NanToNum: replace NaN, +Inf, -Inf with specified values."""

    def __init__(
        self, N_total, dtype, nan_val=0.0, posinf_val=1e4, neginf_val=-1e4, config=None, tune=False
    ):
        self._raw_nan_val = nan_val
        self._raw_posinf_val = posinf_val
        self._raw_neginf_val = neginf_val
        super().__init__(N_total, dtype, config=config, tune=tune)

    def _post_init_params(self):
        self.nan_val = _clamp_to_dtype_range(self._raw_nan_val, self.output_dtype)
        self.posinf_val = _clamp_to_dtype_range(self._raw_posinf_val, self.output_dtype)
        self.neginf_val = _clamp_to_dtype_range(self._raw_neginf_val, self.output_dtype)

    @staticmethod
    def _builder_fn():
        return _make_nan_to_num_kernel

    def _builder_args(self):
        return (self.nan_val, self.posinf_val, self.neginf_val)
