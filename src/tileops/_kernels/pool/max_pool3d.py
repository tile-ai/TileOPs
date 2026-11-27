import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.pool.call_spec import (
    MaxPool3dFwdInterface,
    MaxPool3dIndicesFwdInterface,
    MaxPoolCall,
)
from tileops._kernels.pool.common import (
    ACCUM_DTYPE,
    MaxPoolKernelBase,
    axes_inside,
    max_pool_out,
)

__all__ = ["MaxPool3dKernel", "MaxPool3dWithIndicesKernel"]


@target_cache(maxsize=32)
def _max_pool3d_kernel(call: MaxPoolCall, dtype: str):
    (d_in, h_in, w_in), (kernel_d, kernel_h, kernel_w) = call.size, call.window
    (stride_d, stride_h, stride_w), (pad_d, pad_h, pad_w) = call.stride, call.pad
    dilation_d, dilation_h, dilation_w = call.dilation
    out = max_pool_out(call)
    out_d, out_h, out_w = out
    rows = call.n * call.c_in
    total = rows * out_d * out_h * out_w
    window_inside = all(axes_inside(call, out))

    @tilelang.jit(out_idx=[1], compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"])
    def _max_pool3d_func(block_m: int, threads: int):
        tile_full = total % block_m == 0

        @T.prim_func
        def _max_pool3d_main(
            x: T.Tensor((rows, d_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_d, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, block_m), threads=threads) as tile:
                for i in T.Parallel(block_m):
                    idx = tile * block_m + i
                    if tile_full or idx < total:
                        plane_row = idx // out_w
                        ow = idx - plane_row * out_w
                        depth_row = plane_row // out_h
                        oh = plane_row - depth_row * out_h
                        row = depth_row // out_d
                        od = depth_row - row * out_d
                        front = od * stride_d - pad_d
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        run = T.alloc_var(T.float32)
                        run = -T.infinity(ACCUM_DTYPE)
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    id_ = front + kd * dilation_d
                                    ih = top + kh * dilation_h
                                    iw = left + kw * dilation_w
                                    # Why: over this many taps, skipping the tap
                                    # beats keeping the warp together.
                                    if window_inside or (
                                        (id_ >= 0)
                                        and (id_ < d_in)
                                        and (ih >= 0)
                                        and (ih < h_in)
                                        and (iw >= 0)
                                        and (iw < w_in)
                                    ):
                                        v = T.cast(x[row, id_, ih, iw], ACCUM_DTYPE)
                                        # NaN enters `run` and never leaves,
                                        # since a later value fails `v > NaN`.
                                        run = T.if_then_else(T.isnan(v) or (v > run), v, run)
                        out[row, od, oh, ow] = T.cast(run, dtype)

        return _max_pool3d_main

    return _max_pool3d_func


@target_cache(maxsize=32)
def _max_pool3d_with_indices_kernel(call: MaxPoolCall, dtype: str):
    (d_in, h_in, w_in), (kernel_d, kernel_h, kernel_w) = call.size, call.window
    (stride_d, stride_h, stride_w), (pad_d, pad_h, pad_w) = call.stride, call.pad
    dilation_d, dilation_h, dilation_w = call.dilation
    out = max_pool_out(call)
    out_d, out_h, out_w = out
    rows = call.n * call.c_in
    total = rows * out_d * out_h * out_w
    window_inside = all(axes_inside(call, out))

    # Only the float32 vector path reads through the csrc header, so only it includes it.
    flags = [*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"]
    if dtype == "float32":
        flags += csrc_include("streaming_load.h")

    @tilelang.jit(out_idx=[1, 2], compile_flags=flags)
    def _max_pool3d_with_indices_func(block_m: int, threads: int):
        tile_full = total % block_m == 0
        per_thread = 2  # outputs a thread on the vector path

        def _tap(x, row, id_, ih, iw, evict_last):
            """Return the read of one tap as an expression.

            A plain function, not a macro branch: a local bound in ``_scan`` would change
            the CUDA generated for the per-element path.
            """
            if evict_last:
                return T.call_extern(
                    dtype, "tileops::load_evict_last", T.address_of(x[row, id_, ih, iw])
                )
            return x[row, id_, ih, iw]

        @T.macro
        def _scan(x, idx, evict_last, vals, picks, slot):
            """Fold output *idx*'s window and store its maximum and the position it was at.

            *evict_last* reads each tap through ``tileops::load_evict_last``. With *slot*
            ``None`` the two go to the output's own place in *vals* and *picks*, the op's
            outputs; otherwise to element *slot* of those local arrays.
            """
            plane_row = idx // out_w
            ow = idx - plane_row * out_w
            depth_row = plane_row // out_h
            oh = plane_row - depth_row * out_h
            row = depth_row // out_d
            od = depth_row - row * out_d
            front = od * stride_d - pad_d
            top = oh * stride_h - pad_h
            left = ow * stride_w - pad_w
            run = T.alloc_var(T.float32)
            best = T.alloc_var(T.int32)
            run = -T.infinity(ACCUM_DTYPE)
            # Seeded at the window's first tap inside the input, so a
            # window of nothing but -inf reports that tap.
            best = (
                (front + T.max(0, T.ceildiv(-front, dilation_d)) * dilation_d) * h_in
                + (top + T.max(0, T.ceildiv(-top, dilation_h)) * dilation_h)
            ) * w_in + (left + T.max(0, T.ceildiv(-left, dilation_w)) * dilation_w)
            for kd in T.serial(kernel_d):
                for kh in T.serial(kernel_h):
                    for kw in T.serial(kernel_w):
                        id_ = front + kd * dilation_d
                        ih = top + kh * dilation_h
                        iw = left + kw * dilation_w
                        if window_inside or (
                            (id_ >= 0)
                            and (id_ < d_in)
                            and (ih >= 0)
                            and (ih < h_in)
                            and (iw >= 0)
                            and (iw < w_in)
                        ):
                            v = T.cast(_tap(x, row, id_, ih, iw, evict_last), ACCUM_DTYPE)
                            # Strict > keeps the first maximum; a NaN
                            # takes the position and holds it, so the
                            # last NaN in the window wins.
                            take = T.isnan(v) or (v > run)
                            run = T.if_then_else(take, v, run)
                            best = T.if_then_else(take, (id_ * h_in + ih) * w_in + iw, best)
            if slot is None:
                vals[row, od, oh, ow] = T.cast(run, dtype)
                picks[row, od, oh, ow] = T.cast(best, "int64")
            else:
                vals[slot] = T.cast(run, dtype)
                picks[slot] = T.cast(best, "int64")

        @T.prim_func
        def _max_pool3d_with_indices_main(
            x: T.Tensor((rows, d_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_d, out_h, out_w), dtype),  # type: ignore
            indices: T.Tensor((rows, out_d, out_h, out_w), "int64"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, block_m), threads=threads) as tile:
                # A float32 thread takes two adjacent outputs, reads every tap evict-last
                # in L1 and L2 and stores both outputs and both indices as vectors.
                # float16 and four outputs a thread are slower this way.
                if (
                    tile_full
                    and window_inside
                    and dtype == "float32"
                    and block_m == per_thread * threads
                ):
                    tx = T.get_thread_binding()
                    vals = T.alloc_local([per_thread], dtype)
                    picks = T.alloc_local([per_thread], "int64")
                    first = tile * block_m + tx * per_thread
                    for i in T.unroll(per_thread):
                        _scan(x, first + i, True, vals, picks, i)
                    out_flat = T.reshape(out, (total,))
                    indices_flat = T.reshape(indices, (total,))
                    for i in T.vectorized(per_thread):
                        out_flat[first + i] = vals[i]
                    for i in T.vectorized(per_thread):
                        indices_flat[first + i] = picks[i]
                else:
                    for i in T.Parallel(block_m):
                        idx = tile * block_m + i
                        if tile_full or idx < total:
                            _scan(x, idx, False, out, indices, None)

        return _max_pool3d_with_indices_main

    return _max_pool3d_with_indices_func


class MaxPool3dKernel(MaxPoolKernelBase, MaxPool3dFwdInterface):
    """Max pooling forward kernel (return_indices=False), one output a thread."""

    _build = staticmethod(_max_pool3d_kernel)


class MaxPool3dWithIndicesKernel(MaxPoolKernelBase, MaxPool3dIndicesFwdInterface):
    """Max pooling forward-with-indices kernel."""

    _build = staticmethod(_max_pool3d_with_indices_kernel)

    @property
    def default_config(self) -> dict:
        # A float32 block of two outputs a thread takes the vector path.
        block_m = 256 if self.dtype == torch.float32 else 512
        return {"block_m": block_m, "threads": 128}
