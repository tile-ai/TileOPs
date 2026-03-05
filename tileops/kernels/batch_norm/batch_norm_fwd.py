"""Batch Normalization forward kernels (training and inference).

Reference: Ioffe & Szegedy (2015) https://arxiv.org/abs/1502.03167

Two kernels are provided:
  - BatchNormFwdTrainKernel: computes batch stats, normalizes, updates running stats.
  - BatchNormFwdInferKernel: normalizes using pre-computed running stats.

Input layout expected by both kernels: (C, L) where C is the channel count and
L = N * H * W * ... is the product of batch and spatial dimensions.  The op
layer is responsible for reshaping the user-facing tensor to this layout.

Performance notes:
  - Persistent path (block_l >= L): loads all L elements into shared memory once
    and normalizes from there — single global read, eliminates the second pass.
    Active when L <= _PERSISTENT_THRESHOLD (8192).
  - Non-power-of-2 block_l: _find_best_block_l() searches thread counts
    [256, 224, 192, 128, 96, 64, 32] to find the largest valid block_l,
    fixing poor occupancy for L values like 3136 = 2^6 * 7^2.
"""

from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["BatchNormFwdTrainKernel", "BatchNormFwdInferKernel"]

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

# L threshold for the persistent (single global read) training path.
# x_shared uses L * sizeof(dtype) bytes per block:
#   L=8192, fp16 → 16 KB — well within H100 shared memory limits.
_PERSISTENT_THRESHOLD = 8192


def _find_best_threads(L: int) -> int:
    """Largest power-of-2 t in [256, 128, 64, 32] that evenly divides L.

    TileLang's AllReduce template requires a power-of-2 thread count.
    """
    for t in [256, 128, 64, 32]:
        if L % t == 0:
            return t
    return 32  # fallback


def _find_best_block_l(L: int) -> dict:
    """Find best non-persistent block_l config for given L.

    Uses power-of-2 thread counts only (required by TileLang's AllReduce).
    Block_l can be any multiple of `threads` that divides L — including
    non-power-of-2 values such as 448 for L=3136 — giving more tiles per
    channel and better GPU utilization than the strict power-of-2 search.
    block_l is capped at 512 to limit register pressure.
    """
    for threads in [256, 128, 64, 32]:
        for k in range(512 // threads, 0, -1):
            bl = threads * k
            if bl >= L:
                continue
            if L % bl == 0:
                return {"block_l": bl, "num_stages": 2, "threads": threads}
    # Fallback (should rarely be reached).
    for bl in [512, 256, 128, 64, 32, 16]:
        if L % bl == 0:
            return {"block_l": bl, "num_stages": 1, "threads": min(256, bl)}
    raise ValueError(
        f"L={L} is not divisible by any supported block_l. "
        "L must be divisible by at least 16 for the current kernel implementation."
    )


# ---------------------------------------------------------------------------
# Training forward
# ---------------------------------------------------------------------------

def _batch_norm_fwd_train_kernel(
    C: int,
    L: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> Callable:
    """Return the JIT-compiled training-forward kernel factory.

    Kernel computes, per channel:
      1. mean   = sum(x) / L
      2. var    = sum(x^2) / L  -  mean^2
      3. rstd   = 1 / sqrt(var + eps)
      4. y      = weight * (x - mean) * rstd + bias
      5. running_mean/var updated with *momentum*.

    Saved mean and rstd are needed by the backward pass.

    Persistent path (block_l >= L): after pass 1 loads all L elements into
    x_shared, pass 2 normalizes directly from x_shared — no second global read.

    Non-persistent path (block_l < L): two global reads (classic two-pass BN).

    Requirements: L must be divisible by block_l; threads must divide block_l.
    """
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_train_func(block_l: int, num_stages: int, threads: int) -> Callable:

        @T.prim_func
        def _bn_fwd_train(
            x: T.Tensor([C, L], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_out: T.Tensor([C], accum_dtype),
            rstd_out: T.Tensor([C], accum_dtype),
            y: T.Tensor([C, L], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                x_shared = T.alloc_shared([block_l], dtype)

                # Per-element accumulators: each thread owns block_l/threads elements.
                # Accumulated across L/block_l tiles before the cross-thread reduce.
                xsum_frag = T.alloc_fragment([1, block_l], accum_dtype)
                xsq_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(xsum_frag)
                T.clear(xsq_frag)

                # Pass 1 – accumulate sum(x) and sum(x^2) over all tiles.
                for l_tile in T.Pipelined(L // block_l, num_stages=num_stages):
                    T.copy(x[bc, l_tile * block_l:(l_tile + 1) * block_l], x_shared)
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x_shared[j], accum_dtype)
                        xsum_frag[_i, j] += xval
                        xsq_frag[_i, j] += xval * xval

                # Cross-thread reduction along block_l dimension.
                sum_result = T.alloc_fragment([1], accum_dtype)
                sq_result = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(xsum_frag, sum_result, dim=1)
                T.reduce_sum(xsq_frag, sq_result, dim=1)

                # Statistics.
                mean_val = sum_result[0] / T.cast(L, accum_dtype)
                var_val = sq_result[0] / T.cast(L, accum_dtype) - mean_val * mean_val
                rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(
                    var_val + T.cast(eps, accum_dtype))

                # Save for backward.
                mean_out[bc] = mean_val
                rstd_out[bc] = rstd_val

                # Update running statistics.
                mom = T.cast(momentum, accum_dtype)
                running_mean[bc] = (T.cast(1.0, accum_dtype) - mom) * running_mean[bc] + mom * mean_val
                running_var[bc] = (T.cast(1.0, accum_dtype) - mom) * running_var[bc] + mom * var_val

                # Pass 2 – normalize.
                if block_l >= L:
                    # Persistent path: x_shared still holds all L elements from pass 1.
                    # No second global read — read directly from shared memory.
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x_shared[j], accum_dtype)
                        y[bc, j] = T.cast(
                            weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype)
                else:
                    # Non-persistent path: re-read from global memory.
                    for l_tile in T.Pipelined(L // block_l, num_stages=num_stages):
                        T.copy(x[bc, l_tile * block_l:(l_tile + 1) * block_l], x_shared)
                        for _i, j in T.Parallel(1, block_l):
                            xval = T.cast(x_shared[j], accum_dtype)
                            y[bc, l_tile * block_l + j] = T.cast(
                                weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype)

        return _bn_fwd_train

    return _bn_fwd_train_func


class BatchNormFwdTrainKernel(Kernel):
    """Training-mode batch normalization forward kernel.

    Args:
        C: Number of channels.
        L: Total reduction length = N * H * W * ... (must be divisible by block_l).
        dtype: Input/output data type.
        eps: Numerical stability constant.
        momentum: Running-stat update momentum.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        momentum: float = 0.1,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.dtype = dtype
        self.eps = eps
        self.momentum = momentum
        self.kernel = _batch_norm_fwd_train_kernel(C, L, self.dtype_str, eps, momentum)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.L <= _PERSISTENT_THRESHOLD:
            # Persistent path: block_l = L, single global read.
            t = _find_best_threads(self.L)
            return {"block_l": self.L, "num_stages": 1, "threads": t}
        # Non-persistent path: find best block_l with non-power-of-2 thread counts.
        return _find_best_block_l(self.L)

    @property
    def autotune_configs(self) -> list[dict]:
        seen: set = set()
        configs = []

        def _add(cfg: dict) -> None:
            key = (cfg["block_l"], cfg["num_stages"], cfg["threads"])
            if key not in seen:
                seen.add(key)
                configs.append(cfg)

        # Persistent configs (block_l = L); power-of-2 threads only.
        if self.L <= _PERSISTENT_THRESHOLD:
            for t in [256, 128, 64, 32]:
                if self.L % t == 0:
                    _add({"block_l": self.L, "num_stages": 1, "threads": t})

        # Non-persistent configs: power-of-2 threads, block_l can be non-power-of-2.
        for threads in [256, 128, 64, 32]:
            for k in range(512 // threads, 0, -1):
                bl = threads * k
                if bl >= self.L or self.L % bl != 0:
                    continue
                for ns in [2, 3]:
                    _add({"block_l": bl, "num_stages": ns, "threads": threads})

        return configs if configs else [self.default_config]

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ):
        """Run training forward pass.

        Returns:
            y: Normalized output tensor.
            mean_out: Per-channel batch mean (saved for backward).
            rstd_out: Per-channel reciprocal std (saved for backward).
        """
        mean_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        rstd_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        y = self.kernel(
            self.config["block_l"],
            self.config["num_stages"],
            self.config["threads"],
        )(x, weight, bias, running_mean, running_var, mean_out, rstd_out)
        return y, mean_out, rstd_out


# ---------------------------------------------------------------------------
# Inference forward
# ---------------------------------------------------------------------------

def _batch_norm_fwd_infer_kernel(
    C: int,
    L: int,
    dtype: str = "float16",
    eps: float = 1e-5,
) -> Callable:
    """Return the JIT-compiled inference-forward kernel factory.

    Single pass: y = weight * (x - running_mean) / sqrt(running_var + eps) + bias.
    Fused into a pre-computed scale/shift per channel to minimize arithmetic.
    """
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_infer_func(block_l: int, num_stages: int, threads: int) -> Callable:

        @T.prim_func
        def _bn_fwd_infer(
            x: T.Tensor([C, L], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            y: T.Tensor([C, L], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                x_shared = T.alloc_shared([block_l], dtype)

                # Fused scale/shift: avoids recomputing per element.
                scale = weight[bc] / T.sqrt(
                    running_var[bc] + T.cast(eps, accum_dtype))
                shift = bias[bc] - running_mean[bc] * scale

                for l_tile in T.Pipelined(L // block_l, num_stages=num_stages):
                    T.copy(x[bc, l_tile * block_l:(l_tile + 1) * block_l], x_shared)
                    for _i, j in T.Parallel(1, block_l):
                        y[bc, l_tile * block_l + j] = T.cast(
                            T.cast(x_shared[j], accum_dtype) * scale + shift, dtype)

        return _bn_fwd_infer

    return _bn_fwd_infer_func


class BatchNormFwdInferKernel(Kernel):
    """Inference-mode batch normalization forward kernel.

    Args:
        C: Number of channels.
        L: Total reduction length = N * H * W * ... (must be divisible by block_l).
        dtype: Input/output data type.
        eps: Numerical stability constant.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.dtype = dtype
        self.eps = eps
        self.kernel = _batch_norm_fwd_infer_kernel(C, L, self.dtype_str, eps)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return _find_best_block_l(self.L)

    @property
    def autotune_configs(self) -> list[dict]:
        seen: set = set()
        configs = []

        def _add(cfg: dict) -> None:
            key = (cfg["block_l"], cfg["num_stages"], cfg["threads"])
            if key not in seen:
                seen.add(key)
                configs.append(cfg)

        for threads in [256, 128, 64, 32]:
            for k in range(512 // threads, 0, -1):
                bl = threads * k
                if self.L % bl != 0:
                    continue
                for ns in [2, 3]:
                    _add({"block_l": bl, "num_stages": ns, "threads": threads})

        return configs if configs else [self.default_config]

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        return self.kernel(
            self.config["block_l"],
            self.config["num_stages"],
            self.config["threads"],
        )(x, weight, bias, running_mean, running_var)
