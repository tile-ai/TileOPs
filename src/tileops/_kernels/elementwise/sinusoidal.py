"""The sinusoidal positional-encoding kernel."""

import tilelang
import tilelang.language as T

from tileops._kernels.build_target import target_cache
from tileops._kernels.elementwise._base import GeneratorKernel
from tileops._kernels.elementwise.call_spec import SinusoidalCall, SinusoidalFwdInterface
from tileops._kernels.kernel_base import Entry

__all__ = [
    "SinusoidalFwdKernel",
]


@target_cache(maxsize=32)
def _make_sinusoidal_kernel(seq_len, d_model, dtype):
    """Build sinusoidal positional encoding kernel.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Output shape: (seq_len, d_model).
    """
    half = d_model // 2
    # Positions and dimension pairs one block covers. The divisor depends on the pair
    # and not the position, so a block spanning several positions calls ``pow`` once
    # per pair instead of once per element.
    rows, cols = min(64, seq_len), min(32, half)
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


class SinusoidalFwdKernel(GeneratorKernel, SinusoidalFwdInterface):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Sequence length.
        d_model: Model dimension (must be even).
        dtype: Torch dtype.
        config: Optional config dict.
    """

    @classmethod
    def entry_for(cls, call: SinusoidalCall) -> Entry:
        index = call.device_index
        return call, lambda: cls(call.seq_len, call.d_model, call.dtype, device_index=index)

    def __init__(self, seq_len, d_model, dtype, config=None, device_index=None):
        super().__init__(device_index=device_index)
        self.seq_len = seq_len
        self.d_model = d_model
        self.dtype = dtype
        self.output_dtype = dtype
        self.kernel = _make_sinusoidal_kernel(seq_len, d_model, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self):
        return {"threads": 256}

    def _compile(self, cfg: dict):
        return self.kernel(cfg["threads"])

    def forward(self):
        """Return the encoding in the shape the interface publishes."""
        return self._generate().reshape(self.seq_len, self.d_model)
