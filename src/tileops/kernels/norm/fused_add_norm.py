"""Fused Add + Norm forward kernels using TileLang.

FusedAddLayerNorm: y = LayerNorm(x + residual), also outputs (x + residual)
FusedAddRMSNorm:   y = RMSNorm(x + residual),   also outputs (x + residual)

Fusing the residual add into the normalization kernel eliminates one global
memory round-trip compared to separate add + norm.  Both kernels return dual
outputs ``(y, x + residual)`` so downstream residual connections can reuse the
pre-norm sum without recomputation.

FusedAddLayerNorm holds the row in a register fragment from the load through the store.
FusedAddRMSNorm reads it through 16-byte accesses and writes nothing until the reduction
has settled, so the two reads and the two writes reach memory as two runs rather than
interleaved; the chunk a CTA writes waits in shared memory in between.

Rows are padded to 256 elements, so a 16-byte access always divides the row across a warp.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.tiling import ALIGNMENT, align_up
from tileops.utils import WARP_LANES

from ._config import select_row_config, select_row_configs

__all__ = ["FusedAddLayerNormKernel", "FusedAddRMSNormKernel"]


# Fused Add + LayerNorm kernel


@functools.lru_cache(maxsize=32)
def _fused_add_layer_norm_kernel(M, N, eps, dtype):
    N_padded = align_up(N, ALIGNMENT)
    pad_count = N_padded - N

    @tilelang.jit(out_idx=[4, 5])
    def _func(block_m, threads):
        # A tail row block runs past the end unless every index is guarded.
        row_guard = M % block_m != 0

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
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                r_local = T.alloc_fragment((block_m, N_padded), dtype)
                add_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                # Both operands are read once and reduced along the row the thread
                # already owns, so neither has to pass through shared memory.
                if row_guard:
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = T.if_then_else(
                            pid_m * block_m + i < M,
                            x[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                    for i, j in T.Parallel(block_m, N_padded):
                        r_local[i, j] = T.if_then_else(
                            pid_m * block_m + i < M,
                            residual[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                else:
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = x[pid_m * block_m + i, j]
                    for i, j in T.Parallel(block_m, N_padded):
                        r_local[i, j] = residual[pid_m * block_m + i, j]

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
                    if (not row_guard) or pid_m * block_m + i < M:
                        y[pid_m * block_m + i, j] = T.cast(
                            (T.cast(x_local[i, j], "float32") - mean_val[i])
                            * rstd[i]
                            * T.cast(weight[j], "float32")
                            + T.cast(bias[j], "float32"),
                            dtype,
                        )

                # Write residual_out = x + residual
                for i, j in T.Parallel(block_m, N_padded):
                    if (not row_guard) or pid_m * block_m + i < M:
                        residual_out[pid_m * block_m + i, j] = x_local[i, j]

        return main

    return _func


class FusedAddLayerNormKernel(Kernel):
    """Fused Add + LayerNorm forward kernel.

    Computes ``y = LayerNorm(x + residual)`` and returns both ``y`` and
    ``x + residual``.  The residual add is fused into the first load pass
    to save one global memory round-trip.

    Supports SM80+ architectures.  Uses 256-element alignment for the shared
    buffer the register-relief path fills; every other path holds the row in a
    register fragment from the load through the store.
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
        return select_row_config()

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

# This kernel serves 16-bit dtypes only, so one 16-byte access moves eight elements.
_VEC = VECTOR_ACCESS_BYTES // 2


@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_kernel(M, N, eps, dtype, splits):
    """Return the factory for one CTA per ``(row, chunk)`` pair.

    A CTA reduces a whole row and writes one chunk of it. Nothing is written until the
    reduction has settled, so the two reads and the two writes reach memory as two runs
    rather than interleaved.

    With ``splits > 1`` the CTAs sharing a row each reduce the whole row, paying reads
    that L2 serves to put a single row on more than one SM in one launch.
    """
    N_padded = align_up(N, ALIGNMENT)
    chunk = N_padded // splits

    @tilelang.jit(out_idx=[3, 4])
    def _func(threads):
        own = chunk // (threads * _VEC)  # accesses this CTA reduces and writes
        rest = (N_padded - chunk) // (threads * _VEC)  # accesses it only reduces
        warps = threads // WARP_LANES
        inv_n = 1.0 / N  # N, not N_padded: a padded column holds zero

        @T.macro
        def block_rsqrt(tx, acc, warp_sums, rrms):
            """Fold one fp32 partial per thread into ``rrms[0]``."""
            for step in T.serial(WARP_LANES.bit_length() - 1):
                acc[0] += T.shfl_xor(acc[0], T.shift_left(1, step))
            if tx % WARP_LANES == 0:
                warp_sums[tx // WARP_LANES] = acc[0]
            T.sync_threads()
            # One thread adds the per-warp totals: a block holds at most 32 warps.
            if tx == 0:
                acc[0] = T.cast(0, "float32")
                for w in T.serial(warps):
                    acc[0] += warp_sums[w]
                rrms[0] = T.rsqrt(acc[0] * inv_n + eps)
            T.sync_threads()

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            residual: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
            residual_out: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(M * splits, threads=threads) as pid:
                row = pid // splits
                mine = (pid % splits) * chunk
                tx = T.get_thread_binding()
                a = T.alloc_local([_VEC], dtype)
                b = T.alloc_local([_VEC], dtype)
                summed = T.alloc_shared([chunk], dtype)
                acc = T.alloc_local([1], "float32")
                warp_sums = T.alloc_shared([warps], "float32")
                rrms = T.alloc_shared([1], "float32")

                acc[0] = T.cast(0, "float32")
                for step in T.serial(own):
                    base = (step * threads + tx) * _VEC
                    for i in T.vectorized(_VEC):
                        a[i] = x[row, mine + base + i]
                    for i in T.vectorized(_VEC):
                        b[i] = residual[row, mine + base + i]
                    for i in T.serial(_VEC):
                        # Adding in the storage dtype is bit-identical to rounding the
                        # f32 sum: f32 holds the exact sum of two 16-bit floats.
                        b[i] = a[i] + b[i]
                        v = T.cast(b[i], "float32")
                        acc[0] += v * v
                    for i in T.vectorized(_VEC):
                        summed[base + i] = b[i]

                # The rest of the row, reduced but not written. Starting past this CTA's
                # own chunk keeps the CTAs sharing a row off the same lines at once.
                for step in T.serial(rest):
                    base = (mine + chunk + (step * threads + tx) * _VEC) % N_padded
                    for i in T.vectorized(_VEC):
                        a[i] = x[row, base + i]
                    for i in T.vectorized(_VEC):
                        b[i] = residual[row, base + i]
                    for i in T.serial(_VEC):
                        v = T.cast(a[i] + b[i], "float32")
                        acc[0] += v * v

                block_rsqrt(tx, acc, warp_sums, rrms)

                scale = rrms[0]
                for step in T.serial(own):
                    base = (step * threads + tx) * _VEC
                    for i in T.vectorized(_VEC):
                        a[i] = summed[base + i]
                    for i in T.vectorized(_VEC):
                        b[i] = weight[mine + base + i]
                    for i in T.serial(_VEC):
                        b[i] = T.cast(
                            T.cast(a[i], "float32") * scale * T.cast(b[i], "float32"), dtype
                        )
                    for i in T.vectorized(_VEC):
                        residual_out[row, mine + base + i] = a[i]
                    for i in T.vectorized(_VEC):
                        y[row, mine + base + i] = b[i]

        return main

    return _func


class FusedAddRMSNormKernel(Kernel):
    """Fused Add + RMSNorm forward kernel.

    Computes ``y = RMSNorm(x + residual)`` and returns both ``y`` and ``x + residual``,
    in the four passes over the row that two inputs and two outputs require.

    Supports SM80+ architectures. One CTA reduces a row and writes one chunk of it,
    through 16-byte accesses, parking that chunk in shared memory while the reduction
    settles.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    # Rows at or below which a row-per-CTA call takes the narrow block: while the shared
    # park is small a narrow block keeps more CTAs resident on an SM, and past this width
    # the park caps residency whatever the block is.
    _NARROW_ROW_COLUMNS = 4096
    _NARROW_THREADS = 128
    _WIDE_THREADS = 512

    # Accesses each CTA makes over the row it reduces, when several CTAs share one row.
    _ROW_ACCESSES = 4

    # Most CTAs a single row is cut across. A power of two, so halving reaches every
    # candidate the row may admit.
    _ROW_SPLITS = 4

    # Threads CUDA allows in one block.
    _MAX_THREADS = 1024

    def _widest_dividing(self, target: int) -> int:
        """The widest block at or below *target* cutting the row into whole accesses.

        Floored at one warp, which a row padded to :data:`ALIGNMENT` always admits.
        """
        threads = self._MAX_THREADS
        while threads > WARP_LANES and (threads > target or (self.N_padded // _VEC) % threads):
            threads //= 2
        return threads

    def _row_threads(self) -> int:
        """Block width for a call holding one row per CTA."""
        narrow = self.N_padded <= self._NARROW_ROW_COLUMNS
        return self._widest_dividing(self._NARROW_THREADS if narrow else self._WIDE_THREADS)

    def _split_row_threads(self) -> int:
        """Block width for a single-row call, where the CTAs share one row.

        Sized by the whole row every CTA reduces rather than the chunk it writes,
        which is why it differs from :meth:`_row_threads`.
        """
        return self._widest_dividing(self.N_padded // (_VEC * self._ROW_ACCESSES))

    def _row_splits(self, threads: int, rows: int) -> int:
        """CTAs to put on one row -- more than one only when the call is a single row.

        A call with rows to spare fills the grid with them. A single row instead leaves
        one CTA holding it and the rest of the device idle.
        """
        if rows != 1:
            return 1
        accesses = self.N_padded // (_VEC * threads)
        splits = self._ROW_SPLITS
        while splits > 1 and accesses % splits:
            splits //= 2
        return splits

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is resolved in ``forward``.

        Args:
            N: Hidden size the rows are normalized over.
            eps: Epsilon for numerical stability.
            dtype: Element type the rows are stored in.
            config: Optional ``{"threads": ...}`` override, for the row-per-CTA case.
            tune: Ignored -- the block width follows from the row width; see
                ``autotune_configs``.

        Raises:
            ValueError: *config* names a width that does not cut the row into whole
                16-byte accesses.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.N_padded = align_up(N, ALIGNMENT)
        self.init_config(config, tune=False)
        threads = self.config["threads"]
        # A partial access truncates the per-CTA loop bounds to zero, which returns an
        # untouched row rather than failing.
        if (self.N_padded // _VEC) % threads:
            raise ValueError(
                f"{type(self).__name__} needs a block width that cuts a row of "
                f"{self.N_padded} columns into whole {VECTOR_ACCESS_BYTES}-byte "
                f"accesses; {threads} leaves a partial one. "
                f"{self._row_threads()} is the width this row takes."
            )

    @property
    def default_config(self) -> dict:
        return {"threads": self._row_threads()}

    @property
    def autotune_configs(self) -> list[dict]:
        """The one width :meth:`_row_threads` gives, so a tuned build equals an untuned one.

        The autotuner times repeats of one candidate back to back, which reads the row
        out of L2 and so ranks a kernel this one never runs as.
        """
        return [self.default_config]

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
        m = rows.shape[0]

        # A single row puts several CTAs on it, which wants a different width from the
        # row-per-CTA case and so does not read the configured one.
        threads = self._split_row_threads() if m == 1 else self.config["threads"]
        splits = self._row_splits(threads, m)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _fused_add_rms_norm_kernel(m, self.N, self.eps, self.dtype_str, splits)

        pad = self.N_padded - self.N
        if pad:
            rows = F.pad(rows, (0, pad))
            residual = F.pad(residual, (0, pad))
            weight = F.pad(weight, (0, pad))
        outputs = self.kernel(threads)(rows, residual, weight)
        if pad:
            outputs = [out[:, : self.N] for out in outputs]
        return [out.reshape(original_shape) for out in outputs]
