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

import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["AdaLayerNormKernel"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False):
    N_padded = _align_up(N, ALIGNMENT)
    pad_count = N_padded - N  # number of zero-padded elements per row

    @tilelang.jit(out_idx=[4] if not has_gate else [5])
    def _func(block_m, threads):

        if not has_gate:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                scale: T.Tensor[(M, N_padded), dtype],
                shift: T.Tensor[(M, N_padded), dtype],
                _dummy: T.Tensor[(1,), dtype],
                y: T.Tensor[(M, N_padded), dtype],
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

                    # Load input row block via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Cast to fp32 once — reused across all passes
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

                    # Load scale and shift via shared memory
                    T.copy(scale[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, scale_local)
                    T.copy(shift[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, shift_local)

                    # --- Output: y = scale * (x - mean) * rstd + shift ---
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = (
                            T.cast(scale_local[i, j], "float32")
                            * (T.cast(x_local[i, j], "float32") - mean_val[i])
                            * rstd[i]
                            + T.cast(shift_local[i, j], "float32")
                        )

                    # Write output via shared memory
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

        else:

            @T.prim_func
            def main_gated(
                x: T.Tensor[(M, N_padded), dtype],
                scale: T.Tensor[(M, N_padded), dtype],
                shift: T.Tensor[(M, N_padded), dtype],
                gate: T.Tensor[(M, N_padded), dtype],
                _dummy: T.Tensor[(1,), dtype],
                y: T.Tensor[(M, N_padded), dtype],
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

                    # Load input row block via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Cast to fp32 once
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

                    # Load scale, shift, and gate via shared memory
                    T.copy(scale[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, scale_local)
                    T.copy(shift[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, shift_local)
                    T.copy(gate[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, gate_local)

                    # --- Output: y = gate * (scale * (x - mean) * rstd + shift) ---
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = (
                            T.cast(gate_local[i, j], "float32")
                            * (
                                T.cast(scale_local[i, j], "float32")
                                * (T.cast(x_local[i, j], "float32") - mean_val[i])
                                * rstd[i]
                                + T.cast(shift_local[i, j], "float32")
                            )
                        )

                    # Write output via shared memory
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
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    dummy = torch.empty(1, dtype=x.dtype, device=x.device)
    return _ada_layer_norm_kernel(M, N, eps, dtype_str, has_gate=False)(block_m, threads)(
        x, scale, shift, dummy
    )


@_ada_layer_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, threads, x, scale, shift):
    N_padded = _align_up(N, ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


@torch.library.custom_op("top::ada_layer_norm_zero_fwd", mutates_args=())
def _ada_layer_norm_zero_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    dummy = torch.empty(1, dtype=x.dtype, device=x.device)
    return _ada_layer_norm_kernel(M, N, eps, dtype_str, has_gate=True)(block_m, threads)(
        x, scale, shift, gate, dummy
    )


@_ada_layer_norm_zero_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, threads, x, scale, shift, gate):
    N_padded = _align_up(N, ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


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
        self.kernel = _ada_layer_norm_kernel(
            self.M, self.N, self.eps, self.dtype_str, has_gate=self.has_gate
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # Shared memory budget: need buffers for x + scale + shift (+ gate if has_gate)
        # Main constraint: block_m * N_padded * dtype_size for shared_buf < 48KB
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_m = 1
        for bm in [1, 2, 4, 8, 16]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

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
                x,
                scale,
                shift,
            )
