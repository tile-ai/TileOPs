"""The PReLU kernel."""

import functools

import tilelang
import tilelang.language as T

from ._base import MultiInputElementwiseKernel, _flat

__all__ = [
    "PreluFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_prelu_kernel(N, dtype, C, inner_size, threads=256, npt=8):
    """Build PReLU kernel: y = x if x > 0 else weight[channel] * x.

    Weight is per-channel. Channel index follows PyTorch convention:
    for flat index ``idx``, channel = (idx // inner_size) % C, where
    ``inner_size`` is the product of all dimensions after the channel dim.

    Input and output move through register fragments; the weight is gathered
    per element, since its length is C rather than N.
    """

    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            weight: T.Tensor((C,), dtype),
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    idx = bx * block_size + k
                    val = x_reg[k]
                    ch = (idx // inner_size) % C
                    w = weight[ch]
                    zero = T.cast(0, val.dtype)
                    y_reg[k] = T.if_then_else(val > zero, val, w * val)
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class PreluFwdKernel(MultiInputElementwiseKernel):
    """PReLU: y = x if x > 0 else weight[channel] * x."""

    def __init__(self, N_total, C, inner_size, dtype, config=None, tune=False):
        self.C = C
        self.inner_size = inner_size
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_prelu_kernel

    def _builder_args(self):
        return (self.C, self.inner_size)

    def forward(self, x, weight):
        """Run the kernel.

        ``weight`` is per-channel, not per element, so it is passed flat rather
        than broadcast against ``x``.
        """
        self._require_cuda(x=x, weight=weight)
        return self._compiled_fn(_flat(x), _flat(weight)).reshape(x.shape)
