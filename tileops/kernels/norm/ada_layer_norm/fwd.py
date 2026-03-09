"""Adaptive LayerNorm (AdaLN / AdaLN-Zero) kernel using TileLang.

AdaLN:      y = scale * LayerNorm(x) + shift
AdaLN-Zero: y = gate  * (scale * LayerNorm(x) + shift)

where LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps).

The scale, shift, and gate tensors are per-token (shape [M, N]), not per-feature.
Linear projection from a conditioning input to scale/shift/gate is the caller's
responsibility.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared memory
instructions. Padding zeros contribute 0 to sum; the centered two-pass variance
computation subtracts the exact padding bias to keep results numerically stable.
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


def _ada_layer_norm_kernel(M, N, eps, has_gate, dtype):
    """Build the TileLang kernel for AdaLN (and optionally AdaLN-Zero).

    Args:
        M: Number of rows (tokens).
        N: Hidden dimension (original, before padding).
        eps: Epsilon for numerical stability.
        has_gate: If True, an additional gate tensor is applied element-wise.
        dtype: TileLang dtype string (e.g. "float16").

    Returns:
        A tilelang.jit-compiled function factory accepting (block_m, threads).
    """
    N_padded = _align_up(N, ALIGNMENT)
    pad_count = N_padded - N

    if has_gate:

        @tilelang.jit(out_idx=[4])
        def _func(block_m, threads):

            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                scale: T.Tensor[(M, N_padded), dtype],
                shift: T.Tensor[(M, N_padded), dtype],
                gate: T.Tensor[(M, N_padded), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    scale_local = T.alloc_fragment((block_m, N_padded), dtype)
                    shift_local = T.alloc_fragment((block_m, N_padded), dtype)
                    gate_local = T.alloc_fragment((block_m, N_padded), dtype)
                    acc = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    rstd = T.alloc_fragment((block_m,), "float32")

                    # Load input x
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Load scale
                    T.copy(scale[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, scale_local)

                    # Load shift
                    T.copy(shift[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, shift_local)

                    # Load gate
                    T.copy(gate[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, gate_local)

                    # Cast x to fp32 once
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # --- Mean reduction ---
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(N)

                    # --- Centered variance reduction ---
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (
                            x_f32[i, j] - mean_val[i]
                        )

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(
                            (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                            / float(N)
                            + eps
                        )

                    # --- Output: y = gate * (scale * norm(x) + shift) ---
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

                    # Write output
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

        return _func

    else:

        @tilelang.jit(out_idx=[3])
        def _func_no_gate(block_m, threads):

            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                scale: T.Tensor[(M, N_padded), dtype],
                shift: T.Tensor[(M, N_padded), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    scale_local = T.alloc_fragment((block_m, N_padded), dtype)
                    shift_local = T.alloc_fragment((block_m, N_padded), dtype)
                    acc = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    rstd = T.alloc_fragment((block_m,), "float32")

                    # Load input x
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Load scale
                    T.copy(scale[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, scale_local)

                    # Load shift
                    T.copy(shift[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, shift_local)

                    # Cast x to fp32 once
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # --- Mean reduction ---
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(N)

                    # --- Centered variance reduction ---
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (
                            x_f32[i, j] - mean_val[i]
                        )

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(
                            (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                            / float(N)
                            + eps
                        )

                    # --- Output: y = scale * norm(x) + shift ---
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = (
                            T.cast(scale_local[i, j], "float32")
                            * (T.cast(x_local[i, j], "float32") - mean_val[i])
                            * rstd[i]
                            + T.cast(shift_local[i, j], "float32")
                        )

                    # Write output
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

        return _func_no_gate


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
    return _ada_layer_norm_kernel(M, N, eps, False, dtype_str)(block_m, threads)(
        x, scale, shift
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
    return _ada_layer_norm_kernel(M, N, eps, True, dtype_str)(block_m, threads)(
        x, scale, shift, gate
    )


@_ada_layer_norm_zero_wrapped.register_fake
def _fake_zero(M, N, eps, dtype_str, block_m, threads, x, scale, shift, gate):
    N_padded = _align_up(N, ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


class AdaLayerNormKernel(Kernel):
    """Adaptive LayerNorm kernel (AdaLN / AdaLN-Zero).

    Computes y = scale * LayerNorm(x) + shift (AdaLN), or
    y = gate * (scale * LayerNorm(x) + shift) (AdaLN-Zero when has_gate=True).

    Supports SM80+ architectures. Uses 256-element alignment (512 bytes for
    fp16/bf16) for shared memory copies. Single shared buffer reused for
    loading x, scale, shift, gate, and writing output.

    Args:
        M: Number of rows (tokens).
        N: Hidden dimension (last dim).
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        has_gate: If True, expects an additional gate tensor (AdaLN-Zero).
        config: Optional kernel config dict with block_m and threads.
        tune: If True, run autotuning to find best config.
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
            self.M, self.N, self.eps, self.has_gate, self.dtype_str
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """Return default config based on shared memory budget.

        Shared memory budget: 1 buffer * block_m * N_padded * dtype_size < 48KB.
        """
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_m = 1
        for bm in [1, 2, 4, 8, 16]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        """Return candidate configs for autotuning."""
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
        """Run the AdaLN kernel.

        Args:
            x: Input tensor of shape (M, N_padded).
            scale: Per-token scale tensor of shape (M, N_padded).
            shift: Per-token shift tensor of shape (M, N_padded).
            gate: Per-token gate tensor of shape (M, N_padded). Required when
                has_gate=True.

        Returns:
            Output tensor of shape (M, N_padded).
        """
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
