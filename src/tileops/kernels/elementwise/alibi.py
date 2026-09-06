"""The ALiBi bias kernel."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._dtype import _FLOAT_DTYPES

__all__ = [
    "AlibiFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_alibi_kernel(seq_len, num_heads, dtype, threads=256, npt=8):
    """Build ALiBi kernel: bias[h, i, j] = -slope_h * |i - j|.

    Slopes: slope_h = 2^(-8*h/H) for head h in [0, H).
    Output shape: (num_heads, seq_len, seq_len).
    Total elements: num_heads * seq_len * seq_len.
    """
    N_total = num_heads * seq_len * seq_len
    S2 = seq_len * seq_len

    @tilelang.jit(out_idx=[0])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(out: T.Tensor((N_total,), dtype)):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat = (bx * threads_arg + i) * npt_arg + j
                    if flat < N_total:
                        h = flat // S2
                        rem = flat % S2
                        row = rem // seq_len
                        col = rem % seq_len
                        # slope = 2^(-8 * (h+1) / num_heads)
                        exp_val = (
                            T.cast(-8.0, "float32")
                            * T.cast(h + 1, "float32")
                            / T.cast(num_heads, "float32")
                        )
                        slope = T.exp2(exp_val)
                        dist = T.cast(row - col, "float32")
                        # Use abs via if_then_else since T.abs may not handle int
                        abs_dist = T.if_then_else(dist > T.cast(0, "float32"), dist, -dist)
                        out[flat] = T.Cast(dtype, -slope * abs_dist)

        return main

    return kernel


class AlibiFwdKernel(Kernel):
    """ALiBi position encoding: bias[h, i, j] = -slope_h * |i - j|.

    Generates the full (num_heads, seq_len, seq_len) bias tensor.
    Slopes follow the ALiBi paper: slope_h = 2^(-8*(h+1)/H).

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        dtype: Torch dtype.
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, seq_len, num_heads, dtype, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dtype = dtype
        self.output_dtype = dtype
        cfg = self.default_config
        self.kernel = _make_alibi_kernel(
            seq_len,
            num_heads,
            self.dtype_str,
            cfg["threads"],
            cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        return {"threads": 256, "num_per_thread": 4 if self.dtype == torch.float32 else 8}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self):
        return self._compiled_fn()
