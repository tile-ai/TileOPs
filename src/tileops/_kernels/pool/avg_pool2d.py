import tilelang
import tilelang.language as T

from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.pool.call_spec import AvgPool2dFwdInterface
from tileops._kernels.pool.common import ACCUM_DTYPE, AvgPoolKernelBase, AvgPoolWindow

__all__ = ["AvgPool2dKernel"]


@target_cache(maxsize=32)
def _avg_pool2d_kernel(window: AvgPoolWindow, dtype: str):
    """One output per thread, every tap read where it lies."""
    h_in, w_in = window.size
    kernel_h, kernel_w = window.kernel
    stride_h, stride_w = window.stride
    pad_h, pad_w = window.pad
    out_h, out_w = window.out
    (low_h, low_w), (limit_h, limit_w) = window.overlap
    rows, total = window.rows, window.outputs
    window_inside = window.window_inside
    one_divisor = window.whole_window_divides or window.divisor_override is not None
    divisor = window.divisor

    @tilelang.jit(out_idx=[1], compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"])
    def _avg_pool2d_func(threads: int):
        # A compile-time truth when the launch divides, and the tail test drops out.
        block_full = total % threads == 0

        @T.macro
        def _store(total_val, oh, ow, out, row):
            """Store one output, taking its divisor from wherever this shape has it."""
            if one_divisor:
                out[row, oh, ow] = T.cast(total_val / T.cast(divisor, ACCUM_DTYPE), dtype)
            else:
                top = oh * stride_h - pad_h
                left = ow * stride_w - pad_w
                # An empty window would divide by zero, so the overlap takes a floor.
                extent = T.max(
                    T.max(T.min(top + kernel_h, limit_h) - T.max(top, low_h), 0)
                    * T.max(T.min(left + kernel_w, limit_w) - T.max(left, low_w), 0),
                    1,
                )
                out[row, oh, ow] = T.cast(total_val / T.cast(extent, ACCUM_DTYPE), dtype)

        @T.prim_func
        def _avg_pool2d_main(
            x: T.Tensor((rows, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, threads), threads=threads) as tile:
                for i in T.Parallel(threads):
                    idx = tile * threads + i
                    if block_full or idx < total:
                        plane_row = idx // out_w
                        ow = idx - plane_row * out_w
                        row = plane_row // out_h
                        oh = plane_row - row * out_h
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        total_val = T.alloc_var(T.float32)
                        total_val = T.cast(0.0, ACCUM_DTYPE)
                        for kh in T.serial(kernel_h):
                            for kw in T.serial(kernel_w):
                                ih = top + kh
                                iw = left + kw
                                # Tested where it is read, so the test predicates the
                                # load; hoisting it per axis makes it a branch.
                                if window_inside or (
                                    (ih >= 0) and (ih < h_in) and (iw >= 0) and (iw < w_in)
                                ):
                                    total_val += T.cast(x[row, ih, iw], ACCUM_DTYPE)
                        _store(total_val, oh, ow, out, row)

        return _avg_pool2d_main

    return _avg_pool2d_func


class AvgPool2dKernel(AvgPoolKernelBase, AvgPool2dFwdInterface):
    """Average pooling over an NCHW plane, with every PyTorch flag combination."""

    _build = staticmethod(_avg_pool2d_kernel)

    @property
    def default_config(self) -> dict:
        return {"threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        # One output per thread, so the block size is the whole launch space.
        return [{"threads": threads} for threads in (128, 256, 512)]
