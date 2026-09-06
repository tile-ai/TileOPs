"""Adaptive LayerNorm (AdaLN / AdaLN-Zero) kernel using TileLang.

AdaLN:      y = scale * LayerNorm(x) + shift
AdaLN-Zero: y = gate * (scale * LayerNorm(x) + shift)

scale, shift (and optionally gate) are per-token tensors of shape (M, N),
pre-computed by the caller from a conditioning signal.

The `has_gate` parameter controls the variant:
- has_gate=False → AdaLN
- has_gate=True  → AdaLN-Zero

The kernels accept natural ``(M, N)`` tensors. For non-aligned ``N``, boundary
handling stays on device: loads zero-fill the logical reduction tail and stores
write only real output columns. The centered two-pass variance subtracts the
padded-zero contribution explicitly.

AdaLN modulation tensors use predicated ``cp.async`` prefetch for selected
non-aligned shapes. Aligned shapes retain the vectorized ``T.copy`` path.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.tiling import ALIGNMENT, align_up

from ._config import make_row_reduce, select_row_config, select_row_configs

__all__ = ["AdaLayerNormKernel"]


def _should_use_cp_async(
    n: int,
    dtype: torch.dtype,
    has_gate: bool = False,
) -> bool:
    """Select async prefetch when lowering and shared-memory limits allow it."""
    n_padded = align_up(n, ALIGNMENT)
    row_bytes = n * dtype.itemsize
    num_buffers = 4 if has_gate else 3
    shared_bytes = num_buffers * n_padded * dtype.itemsize
    return n_padded != n and row_bytes % 4 == 0 and shared_bytes <= 48 * 1024


@functools.lru_cache(maxsize=32)
def _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False, use_cp_async=False):
    N_padded = align_up(N, ALIGNMENT)
    needs_pad = N_padded != N
    # The async policy guarantees that each source row is a whole number of
    # 4-byte cp.async transactions.
    async_copy_elems = 1 if dtype == "float32" else 2

    @tilelang.jit(out_idx=[4] if not has_gate else [5])
    def _func(block_m, threads):
        @T.macro
        def load_x_padded(x, shared_buf, x_f32, pid_m):
            for i, j in T.Parallel(block_m, N_padded):
                shared_buf[i, j] = T.if_then_else(
                    T.And(pid_m * block_m + i < M, j < N),
                    x[pid_m * block_m + i, j],
                    T.cast(0.0, dtype),
                )
                x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

        @T.macro
        def load_x_aligned(x, shared_buf, x_local, x_f32, pid_m):
            T.copy(x[pid_m * block_m, 0], shared_buf)
            T.copy(shared_buf, x_local)
            for i, j in T.Parallel(block_m, N_padded):
                x_f32[i, j] = T.cast(x_local[i, j], "float32")

        compute_mean_rstd = make_row_reduce(block_m, N, N_padded, eps)

        @T.macro
        def prefetch_modulation(src, dst, pid_m):
            for i, j in T.Parallel(block_m, N_padded // async_copy_elems):
                row = pid_m * block_m + i
                col = j * async_copy_elems
                T.ptx_cp_async(
                    T.tvm_access_ptr(
                        T.type_annotation(dtype),
                        dst.data,
                        i * N_padded + col,
                        async_copy_elems,
                        2,
                    ),
                    T.tvm_access_ptr(
                        T.type_annotation(dtype),
                        src.data,
                        row * N + col,
                        async_copy_elems,
                        1,
                    ),
                    async_copy_elems,
                    predicate=T.And(row < M, col + async_copy_elems <= N),
                )
            T.ptx_commit_group()

        @T.macro
        def load_aligned_modulation(src, shared_buf, local, pid_m):
            T.copy(src[pid_m * block_m, 0], shared_buf)
            T.copy(shared_buf, local)

        @T.macro
        def kernel_body(x, scale, shift, gate, y):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                if needs_pad:
                    if use_cp_async:
                        scale_shared = T.alloc_shared((block_m, N_padded), dtype)
                        shift_shared = T.alloc_shared((block_m, N_padded), dtype)
                        if has_gate:
                            gate_shared = T.alloc_shared((block_m, N_padded), dtype)
                    load_x_padded(x, shared_buf, x_f32, pid_m)
                else:
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    scale_local = T.alloc_fragment((block_m, N_padded), dtype)
                    shift_local = T.alloc_fragment((block_m, N_padded), dtype)
                    if has_gate:
                        gate_local = T.alloc_fragment((block_m, N_padded), dtype)
                    load_x_aligned(x, shared_buf, x_local, x_f32, pid_m)

                if needs_pad and use_cp_async:
                    prefetch_modulation(scale, scale_shared, pid_m)
                    prefetch_modulation(shift, shift_shared, pid_m)
                    if has_gate:
                        prefetch_modulation(gate, gate_shared, pid_m)
                compute_mean_rstd(x_f32, acc, mean_val, rstd)

                if needs_pad:
                    if use_cp_async:
                        T.ptx_wait_group(0)
                        T.sync_threads()
                    for i, j in T.Parallel(block_m, N_padded):
                        if T.And(pid_m * block_m + i < M, j < N):
                            if use_cp_async:
                                value = T.cast(scale_shared[i, j], "float32") * (
                                    T.cast(shared_buf[i, j], "float32") - mean_val[i]
                                ) * rstd[i] + T.cast(shift_shared[i, j], "float32")
                                if has_gate:
                                    value *= T.cast(gate_shared[i, j], "float32")
                            else:
                                value = T.cast(
                                    scale[pid_m * block_m + i, j],
                                    "float32",
                                ) * (T.cast(shared_buf[i, j], "float32") - mean_val[i]) * rstd[
                                    i
                                ] + T.cast(
                                    shift[pid_m * block_m + i, j],
                                    "float32",
                                )
                                if has_gate:
                                    value *= T.cast(
                                        gate[pid_m * block_m + i, j],
                                        "float32",
                                    )
                            y[pid_m * block_m + i, j] = value
                else:
                    load_aligned_modulation(scale, shared_buf, scale_local, pid_m)
                    load_aligned_modulation(shift, shared_buf, shift_local, pid_m)
                    if has_gate:
                        load_aligned_modulation(gate, shared_buf, gate_local, pid_m)
                    for i, j in T.Parallel(block_m, N_padded):
                        value = T.cast(scale_local[i, j], "float32") * (
                            T.cast(x_local[i, j], "float32") - mean_val[i]
                        ) * rstd[i] + T.cast(shift_local[i, j], "float32")
                        if has_gate:
                            value *= T.cast(gate_local[i, j], "float32")
                        x_local[i, j] = value
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

        if not has_gate:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                scale: T.Tensor[(M, N), dtype],
                shift: T.Tensor[(M, N), dtype],
                # _dummy keeps the output tensor at index 4 so that out_idx=[4]
                # is consistent between the non-gated and gated variants.
                _dummy: T.Tensor[(1,), dtype],
                y: T.Tensor[(M, N), dtype],
            ):
                kernel_body(x, scale, shift, _dummy, y)

            return main

        @T.prim_func
        def main_gated(
            x: T.Tensor[(M, N), dtype],
            scale: T.Tensor[(M, N), dtype],
            shift: T.Tensor[(M, N), dtype],
            gate: T.Tensor[(M, N), dtype],
            _dummy: T.Tensor[(1,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            kernel_body(x, scale, shift, gate, y)

        return main_gated

    return _func


class AdaLayerNormKernel(Kernel):
    """Adaptive LayerNorm kernel.

    Supports both AdaLN and AdaLN-Zero variants via the `has_gate` parameter.
    Uses 256-element alignment (512 bytes for fp16/bf16) for shared memory copies.

    Args:
        N: Hidden dimension (last dim).
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        has_gate: If True, uses the AdaLN-Zero variant with gating.
        config: Optional kernel config override.
        tune: If True, autotune the kernel.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        has_gate: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a hidden size, dtype and variant.

        The program for a given row count is resolved in ``forward``, memoized by
        ``_ada_layer_norm_kernel``.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.has_gate = has_gate
        self.N_padded = align_up(N, ALIGNMENT)
        # Shape policy is benchmarked independently from block/thread tuning.
        self.use_cp_async = _should_use_cp_async(N, dtype, has_gate)
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return select_row_config()

    @property
    def autotune_configs(self) -> list[dict]:
        num_buffers = (4 if self.has_gate else 3) if self.use_cp_async else 1
        return select_row_configs(self.N_padded, self.dtype, num_buffers=num_buffers)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize and modulate ``x`` over its trailing ``N`` elements.

        Flattening to 2-D rows happens here.

        Args:
            x: Input whose trailing axis is ``N``, contiguous, on a CUDA device.
            scale: Per-row scale shaped like *x*.
            shift: Per-row shift shaped like *x*.
            gate: Per-row gate shaped like *x*, required when ``has_gate``.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device, or ``gate`` is missing while
                ``has_gate``.
        """
        self._require_cuda(x=x, scale=scale, shift=shift, gate=gate)
        if self.has_gate and gate is None:
            raise ValueError("gate tensor is required when has_gate=True")

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        scale = scale.reshape(-1, self.N)
        shift = shift.reshape(-1, self.N)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _ada_layer_norm_kernel(
            rows.shape[0],
            self.N,
            self.eps,
            self.dtype_str,
            has_gate=self.has_gate,
            use_cp_async=self.use_cp_async,
        )
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        # ``_dummy`` keeps the output at the index ``out_idx`` names; see the prim_func.
        dummy = torch.empty(1, dtype=x.dtype, device=x.device)
        program = self.kernel(self.config["block_m"], self.config["threads"])
        if self.has_gate:
            y = program(rows, scale, shift, gate.reshape(-1, self.N), dummy)
        else:
            y = program(rows, scale, shift, dummy)
        return y.reshape(original_shape)
