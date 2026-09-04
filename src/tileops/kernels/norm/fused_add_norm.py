"""Fused Add + Norm forward kernels using TileLang.

FusedAddLayerNorm: y = LayerNorm(x + residual), also outputs (x + residual)
FusedAddRMSNorm:   y = RMSNorm(x + residual),   also outputs (x + residual)

Fusing the residual add into the normalization kernel eliminates one global
memory round-trip compared to separate add + norm.  Both kernels return dual
outputs ``(y, x + residual)`` so downstream residual connections can reuse the
pre-norm sum without recomputation.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.tiling import ALIGNMENT, align_up
from tileops.utils import get_sm_count

from ._config import select_row_config, select_row_configs

__all__ = ["FusedAddLayerNormKernel", "FusedAddRMSNormKernel"]


# Fused Add + LayerNorm kernel


@functools.lru_cache(maxsize=32)
def _fused_add_layer_norm_kernel(M, N, eps, dtype):
    N_padded = align_up(N, ALIGNMENT)
    pad_count = N_padded - N

    @tilelang.jit(out_idx=[4, 5])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            residual: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            bias: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
            residual_out: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_x = T.alloc_shared((block_m, N_padded), dtype)
                shared_r = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                r_local = T.alloc_fragment((block_m, N_padded), dtype)
                add_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                # Load x and residual via shared memory
                T.copy(x[pid_m * block_m, 0], shared_x)
                T.copy(shared_x, x_local)
                T.copy(residual[pid_m * block_m, 0], shared_r)
                T.copy(shared_r, r_local)

                # Fused add: compute (x + residual) in fp32
                for i, j in T.Parallel(block_m, N_padded):
                    add_f32[i, j] = T.cast(x_local[i, j], "float32") + T.cast(
                        r_local[i, j], "float32"
                    )

                # Store pre-norm sum back in x_local (native dtype) for output
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = add_f32[i, j]

                # --- Mean reduction ---
                T.reduce_sum(add_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(N)

                # --- Centered variance reduction ---
                for i, j in T.Parallel(block_m, N_padded):
                    add_f32[i, j] = (add_f32[i, j] - mean_val[i]) * (add_f32[i, j] - mean_val[i])

                T.reduce_sum(add_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(
                        (acc[i] - float(pad_count) * mean_val[i] * mean_val[i]) / float(N) + eps
                    )

                # --- Output y: (add - mean) * rstd * weight + bias ---
                # Re-cast from x_local (which holds the pre-norm sum in native dtype)
                for i, j in T.Parallel(block_m, N_padded):
                    r_local[i, j] = (T.cast(x_local[i, j], "float32") - mean_val[i]) * rstd[
                        i
                    ] * T.cast(weight[j], "float32") + T.cast(bias[j], "float32")

                T.copy(r_local, shared_x)
                T.copy(shared_x, y[pid_m * block_m, 0])

                # Write residual_out = x + residual
                T.copy(x_local, shared_r)
                T.copy(shared_r, residual_out[pid_m * block_m, 0])

        return main

    return _func


class FusedAddLayerNormKernel(Kernel):
    """Fused Add + LayerNorm forward kernel.

    Computes ``y = LayerNorm(x + residual)`` and returns both ``y`` and
    ``x + residual``.  The residual add is fused into the first load pass
    to save one global memory round-trip.

    Supports SM80+ architectures.  Uses 256-element alignment for shared
    memory copies.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is resolved in ``forward``, memoized by
        ``_fused_add_layer_norm_kernel``.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.N_padded = align_up(N, ALIGNMENT)
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.N_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.N_padded, self.dtype, num_buffers=2)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run fused add + LayerNorm over the trailing ``N`` elements.

        Flattening to 2-D rows happens here, as does the alignment padding the prim_func
        requires.

        Args:
            x: Input whose trailing axis is ``N``, on a CUDA device.
            residual: Residual shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.
            bias: Affine shift holding ``N`` elements, on the same device.

        Returns:
            ``[y, residual_out]``, both shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_cuda(x=x, residual=residual, weight=weight, bias=bias)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        residual = residual.reshape(-1, self.N)
        weight = weight.reshape(self.N)
        bias = bias.reshape(self.N)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _fused_add_layer_norm_kernel(rows.shape[0], self.N, self.eps, self.dtype_str)
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        pad = self.N_padded - self.N
        if pad:
            rows = F.pad(rows, (0, pad))
            residual = F.pad(residual, (0, pad))
            weight = F.pad(weight, (0, pad))
            bias = F.pad(bias, (0, pad))
        outputs = self.kernel(self.config["block_m"], self.config["threads"])(
            rows, residual, weight, bias
        )
        if pad:
            outputs = [out[:, : self.N] for out in outputs]
        return [out.reshape(original_shape) for out in outputs]


# Fused Add + RMSNorm kernel


@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_kernel(M, N, eps, dtype):
    N_padded = align_up(N, ALIGNMENT)

    @tilelang.jit(out_idx=[3, 4])
    def _func(block_m, threads):
        # Forming the sum in shared rather than a second row-wide fragment keeps
        # bfloat16 off the register limit, but only pays when the registers it
        # frees buy resident blocks: a single-row call measured 0.94x.
        sum_in_shared = -(-M // block_m) >= get_sm_count()

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            residual: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
            residual_out: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_x = T.alloc_shared((block_m, N_padded), dtype)
                shared_r = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                xsq_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                sumsq = T.alloc_fragment((block_m,), "float32")
                rrms = T.alloc_fragment((block_m,), "float32")

                if sum_in_shared:
                    T.copy(x[pid_m * block_m, 0], shared_x)
                    T.copy(residual[pid_m * block_m, 0], shared_r)
                    for i, j in T.Parallel(block_m, N_padded):
                        shared_x[i, j] = T.cast(
                            T.cast(shared_x[i, j], "float32") + T.cast(shared_r[i, j], "float32"),
                            dtype,
                        )
                    T.sync_threads()
                    T.copy(shared_x, residual_out[pid_m * block_m, 0])
                    T.copy(shared_x, x_local)

                    for i, j in T.Parallel(block_m, N_padded):
                        xsq_f32[i, j] = T.cast(x_local[i, j], "float32") * T.cast(
                            x_local[i, j], "float32"
                        )

                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = (
                            T.cast(x_local[i, j], "float32")
                            * rrms[i]
                            * T.cast(weight[j], "float32")
                        )

                    T.copy(x_local, shared_r)
                    T.copy(shared_r, y[pid_m * block_m, 0])
                else:
                    r_local = T.alloc_fragment((block_m, N_padded), dtype)

                    T.copy(x[pid_m * block_m, 0], shared_x)
                    T.copy(shared_x, x_local)
                    T.copy(residual[pid_m * block_m, 0], shared_r)
                    T.copy(shared_r, r_local)

                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = T.cast(x_local[i, j], "float32") + T.cast(
                            r_local[i, j], "float32"
                        )

                    for i, j in T.Parallel(block_m, N_padded):
                        xsq_f32[i, j] = T.cast(x_local[i, j], "float32") * T.cast(
                            x_local[i, j], "float32"
                        )

                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                    for i, j in T.Parallel(block_m, N_padded):
                        r_local[i, j] = (
                            T.cast(x_local[i, j], "float32")
                            * rrms[i]
                            * T.cast(weight[j], "float32")
                        )

                    T.copy(r_local, shared_x)
                    T.copy(shared_x, y[pid_m * block_m, 0])

                    T.copy(x_local, shared_r)
                    T.copy(shared_r, residual_out[pid_m * block_m, 0])

        return main

    return _func


# Traffic a row must carry before splitting it across blocks pays the two
# launches that costs. Measured end to end; at half of it the split is 0.94x.
_SPLIT_MIN_ROW_TRAFFIC = 65536

# Blocks a split row is cut into, and the width each is launched with.
_SPLIT_BLOCKS = 16

_SPLIT_THREADS = 128
_SPLIT_PER_THREAD = 8

_WARP_LANES = 32


@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_split_kernel(N, eps, dtype):
    """Return the two factories that normalize one row across many blocks.

    A decode call is one row, so the row-per-block program launches a single
    block and the device idles behind it. These cut the row into
    :data:`_SPLIT_BLOCKS` pieces: the first sums the squares of its piece and
    writes the sum out, which is an output anyway; the second merges the
    partial sums, few enough that every block redoing it beats a launch that
    does it once, and scales its piece.

    The merge is a different order of fp32 additions from the single-block
    reduction, so results agree to tolerance rather than bit for bit.
    """
    accum = "float32"

    @tilelang.jit
    def _stats(splits, threads, per):
        chunk = N // splits
        n_warps = max(threads // _WARP_LANES, 1)

        @T.prim_func
        def main(
            x: T.Tensor([N], dtype),
            residual: T.Tensor([N], dtype),
            residual_out: T.Tensor([N], dtype),
            partial: T.Tensor([splits], accum),
        ):
            with T.Kernel(splits, threads=threads) as bx:
                tx = T.get_thread_binding()
                acc = T.alloc_local([1], accum)
                a = T.alloc_local([per], dtype)
                b = T.alloc_local([per], dtype)
                total = T.alloc_local([per], dtype)
                acc[0] = T.cast(0, accum)
                for step in T.serial(chunk // (threads * per)):
                    base = bx * chunk + (step * threads + tx) * per
                    for i in T.vectorized(per):
                        a[i] = x[base + i]
                    for i in T.vectorized(per):
                        b[i] = residual[base + i]
                    for i in T.serial(per):
                        total[i] = T.cast(T.cast(a[i], accum) + T.cast(b[i], accum), dtype)
                        v = T.cast(total[i], accum)
                        acc[0] += v * v
                    for i in T.vectorized(per):
                        residual_out[base + i] = total[i]
                for step in T.serial(5):
                    acc[0] += T.shfl_xor(acc[0], T.shift_left(1, step))
                warp_totals = T.alloc_shared([n_warps], accum)
                if tx % _WARP_LANES == 0:
                    warp_totals[tx // _WARP_LANES] = acc[0]
                T.sync_threads()
                if tx == 0:
                    acc[0] = T.cast(0, accum)
                    for w in T.serial(n_warps):
                        acc[0] += warp_totals[w]
                    partial[bx] = acc[0]

        return main

    @tilelang.jit(out_idx=[3])
    def _apply(splits, threads, per):
        chunk = N // splits

        @T.prim_func
        def main(
            summed: T.Tensor([N], dtype),
            partial: T.Tensor([splits], accum),
            weight: T.Tensor([N], dtype),
            y: T.Tensor([N], dtype),
        ):
            with T.Kernel(splits, threads=threads) as bx:
                tx = T.get_thread_binding()
                total = T.alloc_local([1], accum)
                total[0] = T.cast(0, accum)
                for s in T.serial(splits):
                    total[0] += partial[s]
                rrms = T.rsqrt(total[0] / T.cast(N, accum) + T.cast(eps, accum))
                v = T.alloc_local([per], dtype)
                o = T.alloc_local([per], dtype)
                for step in T.serial(chunk // (threads * per)):
                    base = bx * chunk + (step * threads + tx) * per
                    for i in T.vectorized(per):
                        v[i] = summed[base + i]
                    for i in T.serial(per):
                        o[i] = T.cast(
                            T.cast(v[i], accum) * rrms * T.cast(weight[base + i], accum),
                            dtype,
                        )
                    for i in T.vectorized(per):
                        y[base + i] = o[i]

        return main

    return _stats, _apply


class FusedAddRMSNormKernel(Kernel):
    """Fused Add + RMSNorm forward kernel.

    Computes ``y = RMSNorm(x + residual)`` and returns both ``y`` and
    ``x + residual``.  The residual add is fused into the first load pass
    to save one global memory round-trip.

    Supports SM80+ architectures.  Uses 256-element alignment for shared
    memory copies.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is resolved in ``forward``, memoized by
        ``_fused_add_rms_norm_kernel``.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.N_padded = align_up(N, ALIGNMENT)
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    # Passes a row makes over memory: x and residual in, sum and result out.
    _ROW_PASSES = 4

    def _splits_for(self, rows: int) -> int:
        """Blocks to cut a row into, or 0 to keep one block to a row.

        The programs below are written for a single row, so this serves the
        decode shape and nothing else. A call with more rows than that has a
        grid to fill already.
        """
        if rows != 1:
            return 0
        if self.N * self._ROW_PASSES < _SPLIT_MIN_ROW_TRAFFIC:
            return 0
        if self.N % (_SPLIT_BLOCKS * _SPLIT_THREADS * _SPLIT_PER_THREAD):
            return 0
        return _SPLIT_BLOCKS

    @property
    def default_config(self) -> dict:
        return select_row_config(self.N_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.N_padded, self.dtype, num_buffers=2)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run fused add + RMSNorm over the trailing ``N`` elements.

        Flattening to 2-D rows happens here, as does the alignment padding the prim_func
        requires.

        Args:
            x: Input whose trailing axis is ``N``, on a CUDA device.
            residual: Residual shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.

        Returns:
            ``[y, residual_out]``, both shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_cuda(x=x, residual=residual, weight=weight)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        residual = residual.reshape(-1, self.N)
        weight = weight.reshape(self.N)

        splits = self._splits_for(rows.shape[0])
        if splits:
            stats, apply_ = _fused_add_rms_norm_split_kernel(self.N, self.eps, self.dtype_str)
            flat_x = rows.reshape(-1)
            residual_out = torch.empty_like(flat_x)
            partial = torch.empty(splits, device=flat_x.device, dtype=torch.float32)
            stats(splits, _SPLIT_THREADS, _SPLIT_PER_THREAD)(
                flat_x, residual.reshape(-1), residual_out, partial
            )
            y = apply_(splits, _SPLIT_THREADS, _SPLIT_PER_THREAD)(residual_out, partial, weight)
            return [y.reshape(original_shape), residual_out.reshape(original_shape)]

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _fused_add_rms_norm_kernel(rows.shape[0], self.N, self.eps, self.dtype_str)
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        pad = self.N_padded - self.N
        if pad:
            rows = F.pad(rows, (0, pad))
            residual = F.pad(residual, (0, pad))
            weight = F.pad(weight, (0, pad))
        outputs = self.kernel(self.config["block_m"], self.config["threads"])(
            rows, residual, weight
        )
        if pad:
            outputs = [out[:, : self.N] for out in outputs]
        return [out.reshape(original_shape) for out in outputs]
