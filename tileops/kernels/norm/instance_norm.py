from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["InstanceNormKernel"]


def _instance_norm_kernel(rows: int, cols: int, channels: int, eps: float, dtype: str,
                          accum_dtype: str = "float") -> Callable:

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _instance_norm_func(threads: int) -> Callable:

        @T.prim_func
        def _instance_norm_main(
            x: T.Tensor((rows, cols), dtype),  # flattened [N*C, S]
            gamma: T.Tensor((channels,), dtype),
            beta: T.Tensor((channels,), dtype),
            y: T.Tensor((rows, cols), dtype),
        ) -> None:
            with T.Kernel(rows, threads=threads) as row_idx:
                tx = T.get_thread_binding(0)

                sum_local = T.alloc_local((1,), accum_dtype)
                sqsum_local = T.alloc_local((1,), accum_dtype)
                sum_all = T.alloc_local((1,), accum_dtype)
                sqsum_all = T.alloc_local((1,), accum_dtype)
                mean = T.alloc_local((1,), accum_dtype)
                var = T.alloc_local((1,), accum_dtype)
                invstd = T.alloc_local((1,), accum_dtype)

                T.clear(sum_local)
                T.clear(sqsum_local)

                for s_blk in T.serial(T.ceildiv(cols, threads)):
                    col = s_blk * threads + tx
                    x_val = T.if_then_else(col < cols, x[row_idx, col].astype(accum_dtype), 0.0)
                    sum_local[0] += x_val
                    sqsum_local[0] += x_val * x_val

                with T.attr(
                    T.comm_reducer(lambda a, b: a + b, [T.Cast(accum_dtype, 0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            sum_local[0],
                            True,
                            sum_all[0],
                            tx,
                            dtype="handle",
                        ))
                with T.attr(
                    T.comm_reducer(lambda a, b: a + b, [T.Cast(accum_dtype, 0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            sqsum_local[0],
                            True,
                            sqsum_all[0],
                            tx,
                            dtype="handle",
                        ))

                mean[0] = sum_all[0] / T.Cast(accum_dtype, cols)
                var[0] = sqsum_all[0] / T.Cast(accum_dtype, cols) - mean[0] * mean[0]
                invstd[0] = 1.0 / T.sqrt(var[0] + T.Cast(accum_dtype, eps))

                c_idx = row_idx % channels
                g = gamma[c_idx].astype(accum_dtype)
                b = beta[c_idx].astype(accum_dtype)
                for s_blk in T.serial(T.ceildiv(cols, threads)):
                    col = s_blk * threads + tx
                    if col < cols:
                        x_val = x[row_idx, col].astype(accum_dtype)
                        normed = (x_val - mean[0]) * invstd[0]
                        y[row_idx, col] = (normed * g + b).astype(dtype)

        return _instance_norm_main

    return _instance_norm_func


class InstanceNormKernel(Kernel):
    """TileLang kernel for instance_norm forward on flattened [N*C, S] tensor."""

    supported_archs: list[int] = [90]

    def __init__(self,
                 num_channels: int,
                 spatial_size: int,
                 eps: float,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.spatial_size = spatial_size
        self.eps = eps
        self.dtype = dtype
        self.kernel = _instance_norm_kernel(
            rows=1,  # placeholder, real compiled kernel is bound in forward for dynamic batch
            cols=spatial_size,
            channels=num_channels,
            eps=eps,
            dtype=self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"threads": 128},
            {"threads": 256},
            {"threads": 512},
        ]

    def _build_for_rows(self, rows: int) -> Callable:
        return _instance_norm_kernel(
            rows=rows,
            cols=self.spatial_size,
            channels=self.num_channels,
            eps=self.eps,
            dtype=self.dtype_str,
        )

    def forward(self, x_2d: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        if x_2d.device.type != "cuda":
            raise ValueError(f"InstanceNormKernel requires CUDA tensor, got {x_2d.device}.")
        if x_2d.dtype != self.dtype:
            raise ValueError(f"input dtype mismatch: expected {self.dtype}, got {x_2d.dtype}.")
        if x_2d.ndim != 2:
            raise ValueError(f"x_2d must be rank-2, got rank-{x_2d.ndim}.")
        if x_2d.shape[1] != self.spatial_size:
            raise ValueError(
                f"spatial size mismatch: expected {self.spatial_size}, got {x_2d.shape[1]}.")
        if gamma.shape != (self.num_channels,) or beta.shape != (self.num_channels,):
            raise ValueError(
                f"gamma/beta shape must be ({self.num_channels},), got {gamma.shape} / {beta.shape}.")
        if gamma.dtype != self.dtype or beta.dtype != self.dtype:
            raise ValueError("gamma/beta must match input dtype.")
        if gamma.device != x_2d.device or beta.device != x_2d.device:
            raise ValueError("gamma/beta must be on same device as input.")

        rows = x_2d.shape[0]
        kernel = self._build_for_rows(rows)
        return kernel(self.config["threads"])(x_2d, gamma, beta)
