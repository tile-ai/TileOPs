"""The sinusoidal positional-encoding kernel."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._base import (
    _FLOAT_DTYPES,
    _get_fp8_output_dtypes,
    _is_fp8,
)

__all__ = [
    "SinusoidalFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_sinusoidal_kernel(seq_len, d_model, dtype, threads=256, npt=8):
    """Build sinusoidal positional encoding kernel.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Output shape: (seq_len, d_model).
    """
    N_total = seq_len * d_model

    @tilelang.jit(out_idx=[0])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(out: T.Tensor((N_total,), dtype)):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat = (bx * threads_arg + i) * npt_arg + j
                    if flat < N_total:
                        pos = flat // d_model
                        dim = flat % d_model
                        # dim_pair = dim // 2 (the "i" in the formula)
                        dim_pair = dim // 2
                        # angle = pos / 10000^(2*dim_pair / d_model)
                        base = T.cast(10000.0, "float32")
                        exp_frac = (
                            T.cast(dim_pair, "float32")
                            * T.cast(2.0, "float32")
                            / T.cast(d_model, "float32")
                        )
                        divisor = T.pow(base, exp_frac)
                        angle = T.cast(pos, "float32") / divisor
                        # Even dim -> sin, odd dim -> cos
                        is_even = dim % 2 == 0
                        result = T.if_then_else(is_even, T.sin(angle), T.cos(angle))
                        out[flat] = T.Cast(dtype, result)

        return main

    return kernel


class SinusoidalFwdKernel(Kernel):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Sequence length.
        d_model: Model dimension (must be even).
        dtype: Torch dtype.
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, seq_len, d_model, dtype, config=None, tune=False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.seq_len = seq_len
        self.d_model = d_model
        self.dtype = dtype
        self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        cfg = self.default_config
        self.kernel = _make_sinusoidal_kernel(
            seq_len,
            d_model,
            self.dtype_to_str(self.output_dtype),
            cfg["threads"],
            cfg["num_per_thread"],
        )
        self.init_config(config, tune)

    @property
    def default_config(self):
        npt = 4 if self.dtype == torch.float32 else (16 if _is_fp8(self.dtype) else 8)
        return {"threads": 256, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self):
        return self._compiled_fn()
