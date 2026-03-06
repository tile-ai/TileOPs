"""Batch Normalization backward kernel.

Reference: Ioffe & Szegedy (2015) https://arxiv.org/abs/1502.03167

Given saved mean and rstd from the training forward pass, computes:
  grad_bias[c]   = sum_i( grad_out[c, i] )
  grad_weight[c] = sum_i( grad_out[c, i] * x_hat[c, i] )
  grad_x[c, i]   = weight[c] * rstd[c] / L
                   * ( L * grad_out[c, i]
                       - grad_bias[c]
                       - x_hat[c, i] * grad_weight[c] )

where x_hat[c, i] = (x[c, i] - mean[c]) * rstd[c].

Input layout: (C, L) – same convention as the forward kernels.

Performance notes:
  - Persistent path (block_l >= L): after pass 1 accumulates grad_bias /
    grad_weight while loading grad_out and x into shared memory, pass 2 computes
    grad_x directly from shared memory — eliminates the second global read.
    Active when L <= _PERSISTENT_THRESHOLD (8192).
  - Non-power-of-2 block_l: _find_best_block_l() searches thread counts
    [256, 224, 192, 128, 96, 64, 32] for the best valid block_l.
"""

from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["BatchNormBwdKernel"]

# ---------------------------------------------------------------------------
# Config helpers (mirrors batch_norm_fwd.py)
# ---------------------------------------------------------------------------

_PERSISTENT_THRESHOLD = 8192


def _find_best_threads(L: int) -> int:
    """Largest power-of-2 t in [256, 128, 64, 32] that evenly divides L.

    TileLang's AllReduce template requires a power-of-2 thread count.
    """
    for t in [256, 128, 64, 32]:
        if L % t == 0:
            return t
    return 32


def _find_best_block_l(L: int) -> dict:
    """Find best non-persistent block_l config for given L.

    Uses power-of-2 thread counts only (required by TileLang's AllReduce).
    Block_l can be any multiple of `threads` that divides L, including
    non-power-of-2 values.  block_l is capped at 512.
    """
    for threads in [256, 128, 64, 32]:
        for k in range(512 // threads, 0, -1):
            bl = threads * k
            if bl >= L:
                continue
            if L % bl == 0:
                return {"block_l": bl, "num_stages": 0, "threads": threads}
    for bl in [512, 256, 128, 64, 32, 16]:
        if L % bl == 0:
            return {"block_l": bl, "num_stages": 0, "threads": min(256, bl)}
    raise ValueError(
        f"L={L} is not divisible by any supported block_l. "
        "L must be divisible by at least 16 for the current kernel implementation."
    )


def _batch_norm_bwd_kernel(
    C: int,
    L: int,
    dtype: str = "float16",
) -> Callable:
    """Return the JIT-compiled backward kernel factory.

    Persistent path (block_l >= L): go_shared and x_shared retain all L elements
    after pass 1; pass 2 computes grad_x from shared memory — single global read
    for grad_out and x.

    Non-persistent path (block_l < L): two global reads (classic two-pass BN bwd).

    Requirements: L must be divisible by block_l.
    """
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_bwd_func(block_l: int, num_stages: int, threads: int) -> Callable:

        @T.prim_func
        def _bn_bwd(
            grad_out: T.Tensor([C, L], dtype),
            x: T.Tensor([C, L], dtype),
            weight: T.Tensor([C], accum_dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
            grad_x: T.Tensor([C, L], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                go_shared = T.alloc_shared([block_l], dtype)
                x_shared = T.alloc_shared([block_l], dtype)

                mean_val = mean[bc]
                rstd_val = rstd[bc]
                w_val = weight[bc]

                # Accumulators for sum(grad_out) and sum(grad_out * x_hat).
                do_frag = T.alloc_fragment([1, block_l], accum_dtype)
                do_xhat_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(do_frag)
                T.clear(do_xhat_frag)

                # Pass 1 – accumulate grad_bias and grad_weight contributions.
                if block_l >= L:
                    # Persistent path: T.copy into shared memory, single global read.
                    for l_tile in T.Pipelined(L // block_l, num_stages=num_stages):
                        T.copy(grad_out[bc, l_tile * block_l:(l_tile + 1) * block_l], go_shared)
                        T.copy(x[bc, l_tile * block_l:(l_tile + 1) * block_l], x_shared)
                        for _i, j in T.Parallel(1, block_l):
                            go_val = T.cast(go_shared[j], accum_dtype)
                            x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                            do_frag[_i, j] += go_val
                            do_xhat_frag[_i, j] += go_val * x_hat
                else:
                    # Non-persistent path: direct global memory access avoids async-copy
                    # data race that occurs when T.copy is used inside T.Pipelined.
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            go_val = T.cast(grad_out[bc, l_tile * block_l + j], accum_dtype)
                            x_hat = (T.cast(x[bc, l_tile * block_l + j], accum_dtype) - mean_val) * rstd_val
                            do_frag[_i, j] += go_val
                            do_xhat_frag[_i, j] += go_val * x_hat

                # Cross-thread reduction.
                sum_do = T.alloc_fragment([1], accum_dtype)
                sum_do_xhat = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(do_frag, sum_do, dim=1)
                T.reduce_sum(do_xhat_frag, sum_do_xhat, dim=1)

                # Write grad_bias and grad_weight.
                grad_bias[bc] = sum_do[0]
                grad_weight[bc] = sum_do_xhat[0]

                # Precompute per-channel constant.
                w_rstd_over_L = w_val * rstd_val / T.cast(L, accum_dtype)

                # Pass 2 – compute grad_x.
                if block_l >= L:
                    # Persistent path: go_shared and x_shared hold all L elements.
                    # No second global read needed.
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        gx = w_rstd_over_L * (
                            T.cast(L, accum_dtype) * go_val
                            - sum_do[0]
                            - x_hat * sum_do_xhat[0]
                        )
                        grad_x[bc, j] = T.cast(gx, dtype)
                else:
                    # Non-persistent path: direct global memory access avoids async-copy
                    # data race that occurs when T.copy is used inside T.Pipelined.
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            go_val = T.cast(grad_out[bc, l_tile * block_l + j], accum_dtype)
                            x_hat = (T.cast(x[bc, l_tile * block_l + j], accum_dtype) - mean_val) * rstd_val
                            gx = w_rstd_over_L * (
                                T.cast(L, accum_dtype) * go_val
                                - sum_do[0]
                                - x_hat * sum_do_xhat[0]
                            )
                            grad_x[bc, l_tile * block_l + j] = T.cast(gx, dtype)

        return _bn_bwd

    return _bn_bwd_func


class BatchNormBwdKernel(Kernel):
    """Batch normalization backward kernel.

    Args:
        C: Number of channels.
        L: Total reduction length = N * H * W * ... (must be divisible by block_l).
        dtype: grad_out/x/grad_x data type.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.dtype = dtype
        self.kernel = _batch_norm_bwd_kernel(C, L, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.L <= _PERSISTENT_THRESHOLD:
            # Persistent path: block_l = L, single global read.
            # go_shared and x_shared together use 2 * L * sizeof(dtype) SMEM.
            t = _find_best_threads(self.L)
            return {"block_l": self.L, "num_stages": 1, "threads": t}
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
        # num_stages=0 disables pipelining for correctness in multi-tile loops.
        for threads in [256, 128, 64, 32]:
            for k in range(512 // threads, 0, -1):
                bl = threads * k
                if bl >= self.L or self.L % bl != 0:
                    continue
                _add({"block_l": bl, "num_stages": 0, "threads": threads})

        return configs if configs else [self.default_config]

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ):
        """Run backward pass.

        Returns:
            grad_x: Gradient w.r.t. input.
            grad_weight: Gradient w.r.t. affine scale (gamma).
            grad_bias: Gradient w.r.t. affine shift (beta).
        """
        grad_weight = torch.empty(self.C, device=grad_out.device, dtype=torch.float32)
        grad_bias = torch.empty(self.C, device=grad_out.device, dtype=torch.float32)
        grad_x = self.kernel(
            self.config["block_l"],
            self.config["num_stages"],
            self.config["threads"],
        )(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
        return grad_x, grad_weight, grad_bias
