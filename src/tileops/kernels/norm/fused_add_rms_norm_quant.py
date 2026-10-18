"""Fused residual add, RMSNorm and dynamic per-row quantization, as one TileLang kernel.

One CTA owns one row and holds the float32 sum ``x + residual`` in registers, so every input
is read once and every output written once. Two block-wide reductions run over the held row:
the sum of squares gives the inverse RMS, and the normalized row's absolute maximum gives the
scale. Every access is the widest of up to 16 bytes that divides the row.

The arithmetic is vLLM's ``rms_norm_dynamic_per_token_quant``: the sum is reduced unrounded,
the normalized value is rounded to the storage dtype before and after the weight multiply,
and the codes are the normalized values times the reciprocal of the scale.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.norm.call_spec import (
    FusedAddRMSNormPerTokenQuantFwdInterface,
    RMSNormQuantCall,
)
from tileops.kernels.tiling import align_up
from tileops.utils import WARP_LANES

__all__ = ["FusedAddRMSNormPerTokenQuantKernel"]


@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_quant_kernel(M, N, eps, dtype, quant_dtype, scale_ub, vec):
    """Return the factory for one CTA per row, moving ``vec`` elements per access.

    ``vec`` divides ``N``, so a row's accesses stay aligned without padding it; an access past
    the row's end is skipped whole.
    """
    fp8 = quant_dtype == "float8_e4m3fn"
    qmax = 448.0 if fp8 else 127.0
    # vLLM's floor on a row's scale, so an all-zero row divides by a positive number.
    min_scale = 1.0 / (448.0 * 512.0) if fp8 else float(torch.finfo(torch.float32).eps)
    # Without ``scale_ub`` the amax is clamped at the largest float32, which changes nothing.
    bound = float(torch.finfo(torch.float32).max) if scale_ub is None else scale_ub

    @tilelang.jit(out_idx=[3, 4, 5])
    def _func(threads):
        steps = -(-N // (threads * vec))
        held = steps * vec
        warps = threads // WARP_LANES
        # Every access of every thread lies inside the row, so no access needs its guard.
        whole = N % (threads * vec) == 0

        @T.macro
        def block_sum(tx, acc, warp_part, total):
            """Fold one float32 partial per thread into ``total[0]``."""
            for step in T.serial(WARP_LANES.bit_length() - 1):
                acc[0] += T.shfl_xor(acc[0], T.shift_left(1, step))
            if tx % WARP_LANES == 0:
                warp_part[tx // WARP_LANES] = acc[0]
            T.sync_threads()
            if tx == 0:
                acc[0] = T.cast(0, "float32")
                for w in T.serial(warps):
                    acc[0] += warp_part[w]
                total[0] = acc[0]
            T.sync_threads()

        @T.macro
        def block_max(tx, acc, warp_part, total):
            """Fold one non-negative float32 partial per thread into its maximum, ``total[0]``.

            ``max`` returns its other operand when one is NaN, so a NaN element is skipped,
            as vLLM's ``fmaxf`` skips it.
            """
            for step in T.serial(WARP_LANES.bit_length() - 1):
                acc[0] = T.max(acc[0], T.shfl_xor(acc[0], T.shift_left(1, step)))
            if tx % WARP_LANES == 0:
                warp_part[tx // WARP_LANES] = acc[0]
            T.sync_threads()
            if tx == 0:
                acc[0] = T.cast(0, "float32")
                for w in T.serial(warps):
                    acc[0] = T.max(acc[0], warp_part[w])
                total[0] = acc[0]
            T.sync_threads()

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            residual: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N,), dtype],
            q: T.Tensor[(M, N), quant_dtype],
            scale: T.Tensor[(M, 1), T.float32],
            residual_out: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                a = T.alloc_local([vec], dtype)
                b = T.alloc_local([vec], dtype)
                codes = T.alloc_local([vec], quant_dtype)
                s = T.alloc_local([held], "float32")
                acc = T.alloc_local([1], "float32")
                warp_part = T.alloc_shared([warps], "float32")
                sum_sq = T.alloc_shared([1], "float32")
                amax = T.alloc_shared([1], "float32")

                acc[0] = T.cast(0, "float32")
                for step in T.serial(steps):
                    base = (step * threads + tx) * vec
                    if whole or base < N:
                        for i in T.vectorized(vec):
                            a[i] = x[row, base + i]
                        for i in T.vectorized(vec):
                            b[i] = residual[row, base + i]
                        for i in T.serial(vec):
                            # The float32 sum of two 16-bit floats is exact; it is reduced
                            # unrounded.
                            v = T.cast(a[i], "float32") + T.cast(b[i], "float32")
                            s[step * vec + i] = v
                            acc[0] += v * v
                block_sum(tx, acc, warp_part, sum_sq)
                rrms = T.rsqrt(sum_sq[0] / T.float32(N) + eps)

                acc[0] = T.cast(0, "float32")
                for step in T.serial(steps):
                    base = (step * threads + tx) * vec
                    if whole or base < N:
                        for i in T.vectorized(vec):
                            a[i] = weight[base + i]
                        for i in T.serial(vec):
                            v = s[step * vec + i]
                            b[i] = T.cast(v, dtype)
                            # Rounded to the storage dtype before and after the weight multiply.
                            normed = T.cast(T.cast(v * rrms, dtype), "float32")
                            y = T.cast(T.cast(normed * T.cast(a[i], "float32"), dtype), "float32")
                            s[step * vec + i] = y
                            acc[0] = T.max(acc[0], T.abs(y))
                        for i in T.vectorized(vec):
                            residual_out[row, base + i] = b[i]
                block_max(tx, acc, warp_part, amax)

                row_max = T.min(amax[0], T.float32(bound))
                row_scale = T.max(row_max / qmax, T.float32(min_scale))
                inv_scale = T.float32(1.0) / row_scale
                if tx == 0:
                    scale[row, 0] = row_scale

                for step in T.serial(steps):
                    base = (step * threads + tx) * vec
                    if whole or base < N:
                        for i in T.serial(vec):
                            v = s[step * vec + i] * inv_scale
                            if fp8:
                                # min before max, as vLLM saturates: a NaN becomes +448.
                                codes[i] = T.cast(T.max(T.min(v, qmax), -qmax), quant_dtype)
                            else:
                                # nearbyint rounds half to even and the int32 conversion makes
                                # a NaN 0, as vLLM's cvt.rni.sat does; the integer clamp is its
                                # saturation.
                                rounded = T.cast(T.nearbyint(v), "int32")
                                codes[i] = T.cast(
                                    T.max(T.min(rounded, T.int32(127)), T.int32(-128)), "int8"
                                )
                        for i in T.vectorized(vec):
                            q[row, base + i] = codes[i]

        return main

    return _func


class FusedAddRMSNormPerTokenQuantKernel(Kernel, FusedAddRMSNormPerTokenQuantFwdInterface):
    """Fused residual add, RMSNorm and dynamic per-row quantization.

    Supports SM80+ architectures. One CTA reduces, normalizes and quantizes one row, holding
    it in registers, through accesses of up to 16 bytes.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: RMSNormQuantCall) -> Entry:
        identity = (call.n, call.eps, call.dtype, call.quant_dtype, call.scale_ub)
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        scale_ub: Optional[float],
        config: Optional[dict] = None,
    ):
        """Build for a hidden size, the storage and code dtypes and the amax bound.

        The program for a given row count is resolved in ``forward``.

        Args:
            N: Hidden size the rows are normalized over.
            eps: Epsilon for numerical stability.
            dtype: Element type ``x``, ``residual`` and ``weight`` are stored in.
            quant_dtype: ``torch.float8_e4m3fn`` or ``torch.int8``.
            scale_ub: Upper bound on a row's absolute maximum, or ``None``.
            config: Optional ``{"threads": ...}`` override.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.quant_dtype = quant_dtype
        self.scale_ub = scale_ub
        # The widest access of at most 16 bytes (eight 16-bit elements) that divides the row
        # keeps every access aligned to its size.
        self.vec = next(v for v in (VECTOR_ACCESS_BYTES // 2, 4, 2, 1) if N % v == 0)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # A row of up to 4096 columns keeps four 16-byte accesses a thread at 128 threads;
        # a wider row takes up to 512 threads, so the registers holding it stay few. The
        # widest block up to that size that divides the row's accesses leaves no thread idle
        # on the last step; when none divides, the fewest warps that cover the row.
        target = 128 if self.N <= 4096 else 512
        accesses = self.N // self.vec
        threads = target
        while threads >= WARP_LANES and accesses % threads:
            threads //= 2
        if threads < WARP_LANES:
            threads = min(target, align_up(accesses, WARP_LANES))
        return {"threads": threads}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the fused add, RMSNorm and quantization over the trailing ``N`` elements.

        Args:
            x: ``(rows, N)`` on a CUDA device.
            residual: Shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.

        Returns:
            ``[q, scale, residual_out]``.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_devices(x=x, residual=residual, weight=weight)
        x, residual, weight = (vector_aligned(t) for t in (x, residual, weight))
        m = x.shape[0]

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _fused_add_rms_norm_quant_kernel(
            m,
            self.N,
            self.eps,
            self.dtype_str,
            self.dtype_to_str(self.quant_dtype),
            self.scale_ub,
            self.vec,
        )
        return self.kernel(self.config["threads"])(x, residual, weight)
