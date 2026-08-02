"""GroupNorm forward kernel using TileLang.

y = (x - mean) / sqrt(var + eps) * weight + bias

where mean and var are computed over (C/G, *spatial) dimensions for each of
the G groups independently. The input (N, C, *spatial) is reshaped to
(N*G, D) where D = (C/G) * spatial_size, enabling row-wise normalization
identical to LayerNorm.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions. Padding zeros contribute 0 to sum; the centered two-pass
variance computation subtracts the exact padding bias.

Weight and bias are per-channel (C elements). After reshaping, each row of
length D = (C/G) * spatial_size has its own weight/bias slice of length D,
which is tiled from the weight/bias vectors accordingly.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel

from ._config import select_row_config, select_row_configs

__all__ = ["GroupNormKernel", "GroupNormNoAffineKernel"]

ALIGNMENT = 256

# A multiple of every candidate block_m (_config._CANDIDATE_BLOCK_M tops out
# at 8). The row count is padded to this value so the full-tile T.copy never
# crosses the M boundary regardless of the selected block_m.
_M_BLOCK_ALIGN = 16


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def _group_norm_kernel(M, D, eps, dtype):
    """Build a row-wise normalization kernel for shape (M, D_padded).

    This is the core computation shared by GroupNorm and InstanceNorm.
    The caller is responsible for reshaping input/weight/bias into (M, D_padded).

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size (before padding).
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
    """
    D_padded = _align_up(D, ALIGNMENT)
    pad_count = D_padded - D

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, D_padded), dtype],
            weight: T.Tensor[(D_padded,), dtype],
            bias: T.Tensor[(D_padded,), dtype],
            y: T.Tensor[(M, D_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                x_local = T.alloc_fragment((block_m, D_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                # Load input row block via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)
                T.copy(shared_buf, x_local)

                # Cast to fp32 once -- reused across all passes
                for i, j in T.Parallel(block_m, D_padded):
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                # --- Mean reduction ---
                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(D)

                # --- Centered variance reduction ---
                # Rewrite x_f32 in-place with (x - mean)^2.
                # Padded positions (x=0) contribute mean^2; corrected below.
                for i, j in T.Parallel(block_m, D_padded):
                    x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(
                        (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                        / float(D)
                        + eps
                    )

                # --- Output: y = (x - mean) * rstd * weight + bias ---
                for i, j in T.Parallel(block_m, D_padded):
                    x_local[i, j] = (
                        (T.cast(x_local[i, j], "float32") - mean_val[i])
                        * rstd[i]
                        * T.cast(weight[j], "float32")
                        + T.cast(bias[j], "float32")
                    )

                # Write output via shared memory
                T.copy(x_local, shared_buf)
                T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


@torch.library.custom_op("top::group_norm_fwd", mutates_args=())
def _group_norm_wrapped(
    M: int,
    D: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _group_norm_kernel(M, D, eps, dtype_str)(block_m, threads)(x, weight, bias)


@_group_norm_wrapped.register_fake
def _(M, D, eps, dtype_str, block_m, threads, x, weight, bias):
    D_padded = _align_up(D, ALIGNMENT)
    return torch.empty((M, D_padded), dtype=x.dtype, device=x.device)


class GroupNormKernel(Kernel):
    """GroupNorm forward kernel.

    Normalizes each group's (C/G, *spatial) slice independently.
    Input is pre-reshaped to (M, D) where M = N*G, D = (C/G)*spatial_size.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Single shared buffer reused for input load and output store.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        D: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.D = D
        self.eps = eps
        self.dtype = dtype
        self.D_padded = _align_up(D, ALIGNMENT)
        self.kernel = _group_norm_kernel(self.M, self.D, self.eps, self.dtype_str)
        self._identity_affine: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.D_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.D_padded, self.dtype)

    def _identity_affine_for(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached unit-scale / zero-shift buffers of the launch width."""
        buffers = self._identity_affine.get(device)
        if buffers is None:
            buffers = (
                torch.ones(self.D_padded, dtype=self.dtype, device=device),
                torch.zeros(self.D_padded, dtype=self.dtype, device=device),
            )
            self._identity_affine[device] = buffers
        return buffers

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize ``(M, D)`` rows with a row-broadcast affine.

        Args:
            x: Input of shape ``(M, D)``.
            weight: Affine scale of shape ``(D,)``. Omit together with *bias*
                to normalize without an affine.
            bias: Affine shift of shape ``(D,)``. Omit together with *weight*
                to normalize without an affine.

        Returns:
            Tensor of shape ``(M, D)``. The alignment padding the prim_func
            requires is applied and trimmed here.

        Raises:
            ValueError: If exactly one of *weight* / *bias* is omitted.
        """
        if (weight is None) != (bias is None):
            raise ValueError("weight and bias must be supplied together")
        pad = self.D_padded - self.D
        if pad:
            x = F.pad(x, (0, pad))
        if weight is None:
            weight, bias = self._identity_affine_for(x.device)
        elif pad:
            weight = F.pad(weight, (0, pad))
            bias = F.pad(bias, (0, pad))
        y = _group_norm_wrapped(
            self.M,
            self.D,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
            weight,
            bias,
        )
        return y[:, : self.D] if pad else y


@functools.lru_cache(maxsize=32)
def _group_norm_no_affine_kernel(M, D, eps, dtype):
    """Build a row-wise normalization kernel for shape (M, D_padded) without affine.

    Same numerics as :func:`_group_norm_kernel` but omits the trailing
    weight/bias multiply/add — output is ``(x - mean) * rstd``. Used for the
    no-affine variants of GroupNorm and InstanceNorm.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size (before padding).
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
    """
    D_padded = _align_up(D, ALIGNMENT)
    pad_count = D_padded - D

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, D_padded), dtype],
            y: T.Tensor[(M, D_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                x_local = T.alloc_fragment((block_m, D_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                T.copy(x[pid_m * block_m, 0], shared_buf)
                T.copy(shared_buf, x_local)

                for i, j in T.Parallel(block_m, D_padded):
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(D)

                for i, j in T.Parallel(block_m, D_padded):
                    x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(
                        (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                        / float(D)
                        + eps
                    )

                # No-affine output: y = (x - mean) * rstd
                for i, j in T.Parallel(block_m, D_padded):
                    x_local[i, j] = T.cast(
                        (T.cast(x_local[i, j], "float32") - mean_val[i]) * rstd[i],
                        dtype,
                    )

                T.copy(x_local, shared_buf)
                T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


@torch.library.custom_op("top::group_norm_no_affine_fwd", mutates_args=())
def _group_norm_no_affine_wrapped(
    M: int,
    D: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _group_norm_no_affine_kernel(M, D, eps, dtype_str)(block_m, threads)(x)


@_group_norm_no_affine_wrapped.register_fake
def _(M, D, eps, dtype_str, block_m, threads, x):
    D_padded = _align_up(D, ALIGNMENT)
    return torch.empty((M, D_padded), dtype=x.dtype, device=x.device)


class GroupNormNoAffineKernel(Kernel):
    """GroupNorm forward kernel without affine scale/shift.

    Computes ``y = (x - mean) * rstd`` row-wise for shape ``(M, D)`` reshaped
    inputs. Shares the build/launch parameters and shared-memory layout of
    :class:`GroupNormKernel`; only the output stage differs (no weight/bias
    multiply-add). Used by the no-affine variants of GroupNorm and
    InstanceNorm.

    Args:
        M: Number of rows = N * G. Rounded up internally to a whole number
            of ``block_m`` tiles.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        D: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.D = D
        self.eps = eps
        self.dtype = dtype
        self.D_padded = _align_up(D, ALIGNMENT)
        self.M_padded = _align_up(M, _M_BLOCK_ALIGN)
        self.kernel = _group_norm_no_affine_kernel(
            self.M_padded, self.D, self.eps, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.D_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.D_padded, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``(M, D)`` rows without an affine.

        Args:
            x: Input of shape ``(M, D)``.

        Returns:
            Tensor of shape ``(M, D)``. Both the row-count and row-length
            padding the prim_func requires are applied and trimmed here.
        """
        d_pad = self.D_padded - self.D
        m_pad = self.M_padded - self.M
        if d_pad:
            x = F.pad(x, (0, d_pad))
        if m_pad:
            x = F.pad(x, (0, 0, 0, m_pad))
        y = _group_norm_no_affine_wrapped(
            self.M_padded,
            self.D,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )
        if m_pad:
            y = y[: self.M, :]
        if d_pad:
            y = y[:, : self.D]
        return y
