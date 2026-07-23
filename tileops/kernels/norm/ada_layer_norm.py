"""Adaptive LayerNorm (AdaLN / AdaLN-Zero) kernel using TileLang.

AdaLN:      y = scale * LayerNorm(x) + shift
AdaLN-Zero: y = gate * (scale * LayerNorm(x) + shift)

scale, shift (and optionally gate) are per-token tensors of shape (M, N),
pre-computed by the caller from a conditioning signal.

The `has_gate` parameter controls the variant:
- has_gate=False → AdaLN
- has_gate=True  → AdaLN-Zero

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared memory
instructions. Padding zeros contribute 0 to sum; the centered two-pass variance
computation subtracts the exact padding bias to keep results numerically stable
even for large-offset inputs.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._config import select_row_config, select_row_configs

__all__ = ["AdaLayerNormKernel"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False, use_cp_async=False):
    N_padded = _align_up(N, ALIGNMENT)
    needs_pad = N_padded != N
    pad_count = N_padded - N  # number of zero-padded elements per row

    @tilelang.jit(out_idx=[4] if not has_gate else [5])
    def _func(block_m, threads):

        if not has_gate:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                scale: T.Tensor[(M, N), dtype],
                shift: T.Tensor[(M, N), dtype],
                # _dummy keeps the output tensor at index 4 so that out_idx=[4]
                # is consistent between the non-gated (4 inputs) and gated (5
                # inputs) variants, matching the tilelang.jit contract.
                _dummy: T.Tensor[(1,), dtype],
                y: T.Tensor[(M, N), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    rstd = T.alloc_fragment((block_m,), "float32")
                    scale_local = T.alloc_fragment((block_m, N_padded), dtype)
                    shift_local = T.alloc_fragment((block_m, N_padded), dtype)
                    if use_cp_async:
                        scale_shared = T.alloc_shared((block_m, N_padded), dtype)
                        shift_shared = T.alloc_shared((block_m, N_padded), dtype)

                    if needs_pad:
                        for i, j in T.Parallel(block_m, N_padded):
                            shared_buf[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                            x_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                        if use_cp_async:
                            # Prefetch modulation tensors while reducing x.
                            # Predication zero-fills the padded tail.
                            T.async_copy(
                                scale[pid_m * block_m, 0:N_padded], scale_shared
                            )
                            T.async_copy(
                                shift[pid_m * block_m, 0:N_padded], shift_shared
                            )
                    else:
                        T.copy(x[pid_m * block_m, 0], shared_buf)
                        T.copy(shared_buf, x_local)
                        for i, j in T.Parallel(block_m, N_padded):
                            x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # --- Mean reduction ---
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(N)

                    # --- Centered variance reduction ---
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(
                            (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                            / float(N)
                            + eps
                        )

                    if needs_pad:
                        if use_cp_async:
                            T.ptx_wait_group(0)
                            T.sync_threads()
                        for i, j in T.Parallel(block_m, N_padded):
                            if T.And(pid_m * block_m + i < M, j < N):
                                if use_cp_async:
                                    y[pid_m * block_m + i, j] = (
                                        T.cast(scale_shared[i, j], "float32")
                                        * (T.cast(shared_buf[i, j], "float32") - mean_val[i])
                                        * rstd[i]
                                        + T.cast(shift_shared[i, j], "float32")
                                    )
                                else:
                                    y[pid_m * block_m + i, j] = (
                                        T.cast(scale[pid_m * block_m + i, j], "float32")
                                        * (T.cast(shared_buf[i, j], "float32") - mean_val[i])
                                        * rstd[i]
                                        + T.cast(shift[pid_m * block_m + i, j], "float32")
                                    )
                    else:
                        T.copy(scale[pid_m * block_m, 0], shared_buf)
                        T.copy(shared_buf, scale_local)
                        T.copy(shift[pid_m * block_m, 0], shared_buf)
                        T.copy(shared_buf, shift_local)
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = (
                                T.cast(scale_local[i, j], "float32")
                                * (T.cast(x_local[i, j], "float32") - mean_val[i])
                                * rstd[i]
                                + T.cast(shift_local[i, j], "float32")
                            )
                        T.copy(x_local, shared_buf)
                        T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

        else:

            @T.prim_func
            def main_gated(
                x: T.Tensor[(M, N), dtype],
                scale: T.Tensor[(M, N), dtype],
                shift: T.Tensor[(M, N), dtype],
                gate: T.Tensor[(M, N), dtype],
                _dummy: T.Tensor[(1,), dtype],
                y: T.Tensor[(M, N), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    rstd = T.alloc_fragment((block_m,), "float32")
                    scale_local = T.alloc_fragment((block_m, N_padded), dtype)
                    shift_local = T.alloc_fragment((block_m, N_padded), dtype)
                    gate_local = T.alloc_fragment((block_m, N_padded), dtype)
                    if use_cp_async:
                        scale_shared = T.alloc_shared((block_m, N_padded), dtype)
                        shift_shared = T.alloc_shared((block_m, N_padded), dtype)
                        gate_shared = T.alloc_shared((block_m, N_padded), dtype)

                    if needs_pad:
                        for i, j in T.Parallel(block_m, N_padded):
                            shared_buf[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                            x_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                        if use_cp_async:
                            T.async_copy(
                                scale[pid_m * block_m, 0:N_padded], scale_shared
                            )
                            T.async_copy(
                                shift[pid_m * block_m, 0:N_padded], shift_shared
                            )
                            T.async_copy(
                                gate[pid_m * block_m, 0:N_padded], gate_shared
                            )
                    else:
                        T.copy(x[pid_m * block_m, 0], shared_buf)
                        T.copy(shared_buf, x_local)
                        for i, j in T.Parallel(block_m, N_padded):
                            x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # --- Mean reduction ---
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(N)

                    # --- Centered variance reduction ---
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(
                            (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                            / float(N)
                            + eps
                        )

                    if needs_pad:
                        if use_cp_async:
                            T.ptx_wait_group(0)
                            T.sync_threads()
                        for i, j in T.Parallel(block_m, N_padded):
                            if T.And(pid_m * block_m + i < M, j < N):
                                if use_cp_async:
                                    y[pid_m * block_m + i, j] = T.cast(
                                        gate_shared[i, j], "float32"
                                    ) * (
                                        T.cast(scale_shared[i, j], "float32")
                                        * (T.cast(shared_buf[i, j], "float32") - mean_val[i])
                                        * rstd[i]
                                        + T.cast(shift_shared[i, j], "float32")
                                    )
                                else:
                                    y[pid_m * block_m + i, j] = T.cast(
                                        gate[pid_m * block_m + i, j], "float32"
                                    ) * (
                                        T.cast(scale[pid_m * block_m + i, j], "float32")
                                        * (T.cast(shared_buf[i, j], "float32") - mean_val[i])
                                        * rstd[i]
                                        + T.cast(shift[pid_m * block_m + i, j], "float32")
                                    )
                    else:
                        T.copy(scale[pid_m * block_m, 0], shared_buf)
                        T.copy(shared_buf, scale_local)
                        T.copy(shift[pid_m * block_m, 0], shared_buf)
                        T.copy(shared_buf, shift_local)
                        T.copy(gate[pid_m * block_m, 0], shared_buf)
                        T.copy(shared_buf, gate_local)
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = T.cast(gate_local[i, j], "float32") * (
                                T.cast(scale_local[i, j], "float32")
                                * (T.cast(x_local[i, j], "float32") - mean_val[i])
                                * rstd[i]
                                + T.cast(shift_local[i, j], "float32")
                            )
                        T.copy(x_local, shared_buf)
                        T.copy(shared_buf, y[pid_m * block_m, 0])

            return main_gated

    return _func


@torch.library.custom_op("top::ada_layer_norm_fwd", mutates_args=())
def _ada_layer_norm_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    use_cp_async: bool,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    dummy = torch.empty(1, dtype=x.dtype, device=x.device)
    return _ada_layer_norm_kernel(
        M,
        N,
        eps,
        dtype_str,
        has_gate=False,
        use_cp_async=use_cp_async,
    )(block_m, threads)(x, scale, shift, dummy)


@_ada_layer_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, threads, use_cp_async, x, scale, shift):
    return torch.empty((M, N), dtype=x.dtype, device=x.device)


@torch.library.custom_op("top::ada_layer_norm_zero_fwd", mutates_args=())
def _ada_layer_norm_zero_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    use_cp_async: bool,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    dummy = torch.empty(1, dtype=x.dtype, device=x.device)
    return _ada_layer_norm_kernel(
        M,
        N,
        eps,
        dtype_str,
        has_gate=True,
        use_cp_async=use_cp_async,
    )(block_m, threads)(x, scale, shift, gate, dummy)


@_ada_layer_norm_zero_wrapped.register_fake
def _(
    M,
    N,
    eps,
    dtype_str,
    block_m,
    threads,
    use_cp_async,
    x,
    scale,
    shift,
    gate,
):
    return torch.empty((M, N), dtype=x.dtype, device=x.device)


class AdaLayerNormKernel(Kernel):
    """Adaptive LayerNorm kernel.

    Supports both AdaLN and AdaLN-Zero variants via the `has_gate` parameter.
    Uses 256-element alignment (512 bytes for fp16/bf16) for shared memory copies.

    Args:
        M: Number of rows (product of all dims except last).
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
        M: int,
        N: int,
        eps: float,
        dtype: torch.dtype,
        has_gate: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.has_gate = has_gate
        self.N_padded = _align_up(N, ALIGNMENT)
        # Async modulation prefetch pays off for medium unaligned rows. Wider
        # rows stay on the direct-load path to avoid shared-memory pressure.
        self.use_cp_async = self.N_padded != N and 512 <= N < 1920
        self.kernel = _ada_layer_norm_kernel(
            self.M,
            self.N,
            self.eps,
            self.dtype_str,
            has_gate=self.has_gate,
            use_cp_async=self.use_cp_async,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.N_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        num_buffers = (4 if self.has_gate else 3) if self.use_cp_async else 1
        return select_row_configs(
            self.N_padded, self.dtype, num_buffers=num_buffers
        )

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.has_gate:
            if gate is None:
                raise ValueError("gate tensor is required when has_gate=True")
            return _ada_layer_norm_zero_wrapped(
                self.M,
                self.N,
                self.eps,
                self.dtype_str,
                self.config["block_m"],
                self.config["threads"],
                self.use_cp_async,
                x,
                scale,
                shift,
                gate,
            )
        else:
            return _ada_layer_norm_wrapped(
                self.M,
                self.N,
                self.eps,
                self.dtype_str,
                self.config["block_m"],
                self.config["threads"],
                self.use_cp_async,
                x,
                scale,
                shift,
            )
