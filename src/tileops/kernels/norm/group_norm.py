"""GroupNorm forward kernel using TileLang.

y = (x - mean) / sqrt(var + eps) * weight[c] + bias[c]

where mean and var are computed over (C/G, *spatial) dimensions for each of
the G groups independently. The input (N, C, *spatial) is reshaped to
(N*G, D) where D = (C/G) * spatial_size, enabling row-wise normalization
identical to LayerNorm.

The affine is per-channel (C elements) while the normalization is per-row, so
the kernel derives the channel from the position inside the row: row m covers
group ``g = m % G`` and column d covers the group-local channel
``d // spatial_size``, hence ``c = g * (C/G) + d // spatial_size``. Applying
the affine here rather than after the kernel saves a full read+write of the
output tensor.

InstanceNorm is the G = C case: channels_per_group is 1, spatial_size is the
whole row, and the derivation collapses to ``c = m % C``.

256-element alignment (512 bytes for fp16/bf16) is required by T.copy() shared
memory instructions. Both kernels here handle a non-aligned D and a tail row
block inside the prim_func, so neither needs a host-side padding copy. Padding
zeros contribute 0 to the mean; the centered two-pass variance computation
subtracts their exact contribution.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._config import (
    NARROW_ROW,
    make_row_reduce,
    row_padding,
    select_row_config_by_width,
    select_row_configs,
    widths_for_row,
)

__all__ = ["GroupNormKernel", "GroupNormNoAffineKernel"]


def _channel_of(row, col, num_groups: int, channels_per_group: int, spatial_size: int):
    """Return the channel owning element $[row \\times col]$ of the (M, D) reshape.

    Row ``m`` of the ``(N*G, (C/G)*spatial_size)`` view holds group
    ``m % G``, and column ``d`` holds that group's local channel
    ``d // spatial_size``.

    Args:
        row: Row index into the (M, D) view.
        col: Column index into the (M, D) view.
        num_groups: Number of groups G.
        channels_per_group: C / G.
        spatial_size: Number of spatial elements per channel.

    Returns:
        Index into the length-C weight / bias vectors.
    """
    return (row % num_groups) * channels_per_group + col // spatial_size


@functools.lru_cache(maxsize=32)
def _group_norm_kernel(M, D, eps, dtype, num_groups, channels_per_group):
    """Build a row-wise normalization kernel with a per-channel affine.

    This is the core computation shared by GroupNorm and InstanceNorm. The
    caller is responsible for reshaping the input into (M, D); weight and
    bias stay in their natural per-channel (C,) layout and are gathered by
    the channel each element belongs to.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
        num_groups: Number of groups G.
        channels_per_group: C / G. Row ``m`` covers channels
            ``(m % G) * channels_per_group`` onwards.
    """
    D_padded = row_padding(D, 4 if dtype == "float32" else 2)
    spatial_size = D // channels_per_group
    C = num_groups * channels_per_group

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):
        # A non-aligned D would read and write columns >= D unless masked.
        masked = D_padded != D
        # One channel owns the whole row exactly when a group holds one channel.
        row_constant_affine = channels_per_group == 1
        # A row whose width the block divides is read from global memory straight
        # into the register fragment. A padded row loses the vectorized copy to a
        # per-element guard, so above NARROW_ROW it stages through shared memory,
        # the only path that allocates it.
        register_direct = not masked or D_padded <= NARROW_ROW
        # A tail row block runs past the end unless every index is guarded.
        guarded = masked or M % block_m != 0
        row_reduce = make_row_reduce(block_m, D, D_padded, eps)

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            weight: T.Tensor[(C,), dtype],
            bias: T.Tensor[(C,), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                if not register_direct:
                    shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                x_local = T.alloc_fragment((block_m, D_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")
                if row_constant_affine:
                    scale = T.alloc_fragment((block_m,), "float32")
                    bias_row = T.alloc_fragment((block_m,), "float32")

                if register_direct and guarded:
                    for i, j in T.Parallel(block_m, D_padded):
                        v = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < D),
                            x[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                        x_local[i, j] = v
                        x_f32[i, j] = T.cast(v, "float32")
                elif register_direct:
                    for i, j in T.Parallel(block_m, D_padded):
                        v = x[pid_m * block_m + i, j]
                        x_local[i, j] = v
                        x_f32[i, j] = T.cast(v, "float32")
                else:
                    # A padded wide row keeps its input in shared memory for the
                    # output pass: the reduction overwrites the fp32 copy with
                    # the centered squares.
                    for i, j in T.Parallel(block_m, D_padded):
                        shared_buf[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < D),
                            x[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                row_reduce(x_f32, acc, mean_val, rstd)

                # --- Output: y = (x - mean) * rstd * weight[c] + bias[c] ---
                if row_constant_affine:
                    # A group of one channel gives the whole row one weight and
                    # one bias, so the channel derivation and the two gathers
                    # leave the element loop. Centering stays in it: folding it
                    # into a shift term subtracts two large products.
                    for i in T.Parallel(block_m):
                        c = (pid_m * block_m + i) % num_groups
                        scale[i] = rstd[i] * T.cast(weight[c], "float32")
                        bias_row[i] = T.cast(bias[c], "float32")
                    for i, j in T.Parallel(block_m, D_padded):
                        if (not guarded) or T.And(pid_m * block_m + i < M, j < D):
                            y[pid_m * block_m + i, j] = T.cast(
                                (
                                    T.cast(
                                        x_local[i, j] if register_direct else shared_buf[i, j],
                                        "float32",
                                    )
                                    - mean_val[i]
                                )
                                * scale[i]
                                + bias_row[i],
                                dtype,
                            )
                elif guarded:
                    for i, j in T.Parallel(block_m, D_padded):
                        if T.And(pid_m * block_m + i < M, j < D):
                            c = _channel_of(
                                pid_m * block_m + i,
                                j,
                                num_groups,
                                channels_per_group,
                                spatial_size,
                            )
                            y[pid_m * block_m + i, j] = (
                                T.cast(
                                    x_local[i, j] if register_direct else shared_buf[i, j],
                                    "float32",
                                )
                                - mean_val[i]
                            ) * rstd[i] * T.cast(weight[c], "float32") + T.cast(bias[c], "float32")
                else:
                    for i, j in T.Parallel(block_m, D_padded):
                        c = _channel_of(
                            pid_m * block_m + i,
                            j,
                            num_groups,
                            channels_per_group,
                            spatial_size,
                        )
                        y[pid_m * block_m + i, j] = (
                            T.cast(x_local[i, j], "float32") - mean_val[i]
                        ) * rstd[i] * T.cast(weight[c], "float32") + T.cast(bias[c], "float32")

        return main

    return _func


class GroupNormKernel(Kernel):
    """GroupNorm forward kernel with a per-channel affine.

    Normalizes each group's (C/G, *spatial) slice independently and applies
    ``weight[c]`` / ``bias[c]`` to every element, with *c* derived from the
    element's position in the row. Input is pre-reshaped to (M, D) where
    M = N*G, D = (C/G)*spatial_size; weight and bias keep their (C,) layout.

    InstanceNorm uses this kernel with ``num_groups=C`` and
    ``channels_per_group=1``.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Single shared buffer reused for input load and output store.

    Args:
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        num_groups: Number of groups G.
        channels_per_group: C / G.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        D: int,
        eps: float,
        dtype: torch.dtype,
        num_groups: int,
        channels_per_group: int,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a row length, dtype and group layout.

        The program for a given row count is resolved in ``forward``, memoized by
        ``_group_norm_kernel``.
        """
        super().__init__()
        self.D = D
        self.eps = eps
        self.dtype = dtype
        self.num_groups = num_groups
        self.channels_per_group = channels_per_group
        self.D_padded = row_padding(D, self.dtype.itemsize)
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return select_row_config_by_width(self.D_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.D_padded, self.dtype, widths=widths_for_row(self.D_padded))

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize ``D``-long rows and apply the per-channel affine.

        Flattening to ``(M, D)`` rows happens here.

        Args:
            x: Input of shape ``(N, C, *spatial)``, contiguous, on a CUDA device.
            weight: Affine scale of shape $[C]$ on the same device.
            bias: Affine shift of shape $[C]$ on the same device.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device, or the affine pair is missing.
        """
        self._require_cuda(x=x, weight=weight, bias=bias)
        if weight is None or bias is None:
            raise ValueError(
                f"{type(self).__name__} applies a per-channel affine; weight and bias are "
                "required. GroupNormNoAffineKernel serves the affine-free call."
            )

        original_shape = x.shape
        rows = x.reshape(-1, self.D)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _group_norm_kernel(
            rows.shape[0],
            self.D,
            self.eps,
            self.dtype_str,
            self.num_groups,
            self.channels_per_group,
        )
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        y = self.kernel(self.config["block_m"], self.config["threads"])(rows, weight, bias)
        return y.reshape(original_shape)


@functools.lru_cache(maxsize=32)
def _group_norm_no_affine_kernel(M, D, eps, dtype):
    """Build a row-wise normalization kernel for shape (M, D) without affine.

    Same numerics and same boundary handling as `_group_norm_kernel`,
    but omits the trailing weight/bias multiply-add — output is
    ``(x - mean) * rstd``. Used for the no-affine variants of GroupNorm and
    InstanceNorm.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
    """
    D_padded = row_padding(D, 4 if dtype == "float32" else 2)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        # A non-aligned D would read and write columns >= D unless masked.
        masked = D_padded != D
        # A row whose width the block divides is read straight into the register
        # fragment; a padded one only while it is narrow.
        register_direct = not masked or D_padded <= NARROW_ROW
        # A tail row block runs past the end unless every index is guarded.
        guarded = masked or M % block_m != 0
        row_reduce = make_row_reduce(block_m, D, D_padded, eps)

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                if not register_direct:
                    shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                x_local = T.alloc_fragment((block_m, D_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                if register_direct and guarded:
                    for i, j in T.Parallel(block_m, D_padded):
                        v = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < D),
                            x[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                        x_local[i, j] = v
                        x_f32[i, j] = T.cast(v, "float32")
                elif register_direct:
                    for i, j in T.Parallel(block_m, D_padded):
                        v = x[pid_m * block_m + i, j]
                        x_local[i, j] = v
                        x_f32[i, j] = T.cast(v, "float32")
                else:
                    # A padded wide row keeps its input in shared memory for the
                    # output pass: the reduction overwrites the fp32 copy with
                    # the centered squares.
                    for i, j in T.Parallel(block_m, D_padded):
                        shared_buf[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < D),
                            x[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                row_reduce(x_f32, acc, mean_val, rstd)

                # No-affine output: y = (x - mean) * rstd.
                if guarded:
                    for i, j in T.Parallel(block_m, D_padded):
                        if T.And(pid_m * block_m + i < M, j < D):
                            y[pid_m * block_m + i, j] = T.cast(
                                (
                                    T.cast(
                                        x_local[i, j] if register_direct else shared_buf[i, j],
                                        "float32",
                                    )
                                    - mean_val[i]
                                )
                                * rstd[i],
                                dtype,
                            )
                else:
                    for i, j in T.Parallel(block_m, D_padded):
                        y[pid_m * block_m + i, j] = T.cast(
                            (T.cast(x_local[i, j], "float32") - mean_val[i]) * rstd[i],
                            dtype,
                        )

        return main

    return _func


class GroupNormNoAffineKernel(Kernel):
    """GroupNorm forward kernel without affine scale/shift.

    Computes ``y = (x - mean) * rstd`` row-wise for shape $[M \\times D]$ reshaped
    inputs. Shares the build/launch parameters and shared-memory layout of
    `GroupNormKernel`; only the output stage differs (no weight/bias
    multiply-add). Used by the no-affine variants of GroupNorm and
    InstanceNorm.

    Args:
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        D: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a row length and dtype.

        The program for a given row count is resolved in ``forward``, memoized by
        ``_group_norm_no_affine_kernel``.
        """
        super().__init__()
        self.D = D
        self.eps = eps
        self.dtype = dtype
        self.D_padded = row_padding(D, self.dtype.itemsize)
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return select_row_config_by_width(self.D_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.D_padded, self.dtype, widths=widths_for_row(self.D_padded))

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize ``D``-long rows without an affine.

        Flattening to ``(M, D)`` rows happens here.

        Args:
            x: Input of shape ``(N, C, *spatial)``, contiguous, on a CUDA device.
            weight: The op's empty affine slot; this kernel has no affine.
            bias: The op's empty affine slot; this kernel has no affine.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: *x* is not on a CUDA device, or an affine tensor was passed.
        """
        self._require_cuda(x=x)
        if weight is not None or bias is not None:
            raise ValueError(
                f"{type(self).__name__} has no affine; GroupNormKernel serves the affine call."
            )

        original_shape = x.shape
        rows = x.reshape(-1, self.D)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _group_norm_no_affine_kernel(rows.shape[0], self.D, self.eps, self.dtype_str)
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        y = self.kernel(self.config["block_m"], self.config["threads"])(rows)
        return y.reshape(original_shape)
