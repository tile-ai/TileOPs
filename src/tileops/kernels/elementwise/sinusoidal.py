"""The sinusoidal positional-encoding kernel."""

import functools

import tilelang
import tilelang.language as T

from tileops.kernels.kernel_base import Kernel

from ._dtype import _FLOAT_DTYPES

__all__ = [
    "SinusoidalFwdKernel",
]


# Positions and dimension pairs one block covers. The divisor depends on the pair
# and not the position, so a block spanning several positions calls ``pow`` once
# per pair instead of once per element. Both are capped to the tensor at build
# time.
_ROWS, _COLS = 64, 32


@functools.lru_cache(maxsize=32)
def _make_sinusoidal_kernel(seq_len, d_model, dtype, threads=256, rows=_ROWS, cols=_COLS):
    """Build sinusoidal positional encoding kernel.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Output shape: (seq_len, d_model).
    """
    half = d_model // 2
    rows, cols = min(rows, seq_len), min(cols, half)
    width = cols * 2

    @tilelang.jit(out_idx=[0])
    def kernel(threads_arg):
        @T.prim_func
        def main(out: T.Tensor((seq_len, d_model), dtype)):
            with T.Kernel(T.ceildiv(half, cols), T.ceildiv(seq_len, rows), threads=threads_arg) as (
                bc,
                br,
            ):
                # Named `divisor`, not `div`: stdlib.h declares a `div` the emitted
                # CUDA would collide with.
                divisor = T.alloc_shared((cols,), "float32")
                angle = T.alloc_fragment((rows, cols), "float32")
                pe = T.alloc_fragment((rows, width), dtype)
                for c in T.Parallel(cols):
                    exponent = (
                        T.cast(bc * cols + c, "float32")
                        * T.cast(2.0, "float32")
                        / T.cast(d_model, "float32")
                    )
                    divisor[c] = T.pow(T.cast(10000.0, "float32"), exponent)
                for r, c in T.Parallel(rows, cols):
                    angle[r, c] = T.cast(br * rows + r, "float32") / divisor[c]
                for r, c in T.Parallel(rows, width):
                    pe[r, c] = T.Cast(
                        dtype,
                        T.if_then_else(
                            c % 2 == 0, T.sin(angle[r, c // 2]), T.cos(angle[r, c // 2])
                        ),
                    )
                T.copy(pe, out[br * rows : (br + 1) * rows, bc * width : (bc + 1) * width])

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
        if d_model % 2:
            raise ValueError(f"{type(self).__name__} needs an even d_model, got {d_model}")
        self.seq_len = seq_len
        self.d_model = d_model
        self.dtype = dtype
        self.output_dtype = dtype
        self.kernel = _make_sinusoidal_kernel(seq_len, d_model, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self):
        return {"threads": 256}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        self._compiled_fn = self.kernel(self.config["threads"])

    def forward(self):
        return self._compiled_fn()
