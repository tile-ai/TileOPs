"""InstanceNorm kernels.

InstanceNorm is the special case of GroupNorm with G = C (each channel is
its own group). The math reduces exactly to the GroupNorm row-wise kernel
on a reshape of ``(N, C, *spatial) -> (N*C, spatial_size)``.

`InstanceNormKernel` and `InstanceNormNoAffineKernel` run GroupNorm's TileLang
bodies; these classes exist so that `tileops.ops.norm.instance_norm` and the
manifest can name an InstanceNorm-specific kernel, and so that both take the five
inputs ``InstanceNormFwdOp``'s signature declares.

`InstanceNormFwdTrainKernel` and `InstanceNormFwdInferKernel`, which read the
running statistics, have bodies of their own.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES

from ._config import make_row_reduce, make_shifted_row_reduce, row_padding
from .group_norm import (
    GroupNormKernel,
    GroupNormNoAffineKernel,
    _holds_row_in_registers,
    _RowNormKernel,
)

__all__ = [
    "InstanceNormFwdInferKernel",
    "InstanceNormFwdTrainKernel",
    "InstanceNormKernel",
    "InstanceNormNoAffineKernel",
]


class InstanceNormKernel(GroupNormKernel):
    """InstanceNorm forward kernel with a per-channel affine.

    GroupNorm's kernel with ``num_groups=C`` and ``channels_per_group=1``. The running
    statistics are slots of the op's signature that this kernel does not read: it
    normalizes by the statistics of this call's input.
    """

    def forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(x, weight, bias)


class InstanceNormNoAffineKernel(GroupNormNoAffineKernel):
    """InstanceNorm forward kernel without affine scale/shift.

    GroupNorm's no-affine kernel with ``G = C``. The running statistics and the affine
    pair are slots of the op's signature that this kernel does not read.
    """

    def forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(x, weight, bias)


def _row_per_thread_group(block_m: int, D_padded: int, threads: int, dtype: str):
    """The layout giving row ``i`` of a ``(block_m, D_padded)`` fragment its own threads.

    Row ``i`` lives on threads ``[i * t, (i + 1) * t)`` for ``t = threads // block_m``,
    each holding runs of one vector width, so the row reduction stays inside one
    thread group. Layout inference can otherwise pick a far slower replicated layout.
    """
    row_threads = threads // block_m
    per_thread = D_padded // row_threads
    vector = VECTOR_ACCESS_BYTES // (4 if dtype == "float32" else 2)
    while per_thread % vector:
        vector //= 2
    return T.Fragment(
        [block_m, D_padded],
        forward_thread_fn=lambda i, j: i * row_threads + (j // vector) % row_threads,
        forward_index_fn=lambda i, j: (j // (vector * row_threads)) * vector + j % vector,
    )


@functools.lru_cache(maxsize=32)
def _instance_norm_train_kernel(N, C, D, eps, momentum, dtype, has_weight, has_bias):
    """Build the kernel normalizing by instance statistics and updating the running ones.

    Block ``(c, s)`` normalizes rows ``(n, c)`` with ``n // block_m == s``. With one
    block per channel it writes the running statistics itself; otherwise it writes its
    rows' update sums to ``partial`` for `_instance_norm_stats_kernel`.

    Args:
        N: Batch size.
        C: Number of channels.
        D: Elements per instance, ``prod(spatial)``; above one.
        eps: Epsilon for numerical stability.
        momentum: Weight of an instance's statistics in its update.
        dtype: TileLang dtype string of the input.
        has_weight: Whether ``weight`` is read; otherwise the scale is one.
        has_bias: Whether ``bias`` is read; otherwise the shift is zero.
    """
    accum_dtype = "float32"
    D_padded = row_padding(D, 4 if dtype == "float32" else 2)
    register_direct = _holds_row_in_registers(D, D_padded)
    unbiased = D / (D - 1)

    @tilelang.jit(out_idx=[6])
    def _func(block_m, threads):
        splits = -(-N // block_m)
        # The last block's rows past the last sample read and write nothing.
        guarded = D_padded != D or N % block_m != 0
        row_layout = _row_per_thread_group(block_m, D_padded, threads, dtype)
        if register_direct:
            row_reduce = make_shifted_row_reduce(block_m, D, eps)
        else:
            row_reduce = make_row_reduce(block_m, D, D_padded, eps)

        @T.prim_func
        def main(
            x: T.Tensor[(N, C, D), dtype],
            running_mean: T.Tensor[(C,), accum_dtype],
            running_var: T.Tensor[(C,), accum_dtype],
            weight: T.Tensor[(C,), dtype],
            bias: T.Tensor[(C,), dtype],
            partial: T.Tensor[(splits, 2, C), accum_dtype],
            y: T.Tensor[(N, C, D), dtype],
        ):
            with T.Kernel(C, splits, threads=threads) as (c, s):
                if register_direct:
                    centered_row = T.alloc_fragment((block_m, D_padded), accum_dtype)
                    squares = T.alloc_fragment((block_m, D_padded), accum_dtype)
                    shift = T.alloc_fragment((block_m,), accum_dtype)
                    acc_squares = T.alloc_fragment((block_m,), accum_dtype)
                    T.annotate_layout({centered_row: row_layout, squares: row_layout})
                else:
                    shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, D_padded), accum_dtype)
                    T.annotate_layout({x_f32: row_layout})
                acc = T.alloc_fragment((block_m,), accum_dtype)
                mean_val = T.alloc_fragment((block_m,), accum_dtype)
                rstd = T.alloc_fragment((block_m,), accum_dtype)
                scale = T.alloc_fragment((block_m,), accum_dtype)
                mean_update = T.alloc_fragment((block_m,), accum_dtype)
                var_update = T.alloc_fragment((block_m,), accum_dtype)
                block_sums = T.alloc_shared((2, block_m), accum_dtype)

                # torch reads the running statistics in the input's dtype.
                old_mean = T.cast(T.cast(running_mean[c], dtype), accum_dtype)
                old_var = T.cast(T.cast(running_var[c], dtype), accum_dtype)
                weight_c = T.cast(weight[c], accum_dtype) if has_weight else T.float32(1.0)
                shift_out = T.cast(bias[c], accum_dtype) if has_bias else T.float32(0.0)

                if register_direct:
                    # The shift the one-pass reduction needs; a row past the last
                    # sample reads the last one.
                    for i in T.Parallel(block_m):
                        shift[i] = T.cast(x[T.min(s * block_m + i, N - 1), c, 0], accum_dtype)
                    if guarded:
                        for i, j in T.Parallel(block_m, D_padded):
                            v = T.if_then_else(
                                T.And(s * block_m + i < N, j < D),
                                T.cast(x[s * block_m + i, c, j], accum_dtype) - shift[i],
                                T.cast(0.0, accum_dtype),
                            )
                            centered_row[i, j] = v
                            squares[i, j] = v * v
                    else:
                        for i, j in T.Parallel(block_m, D_padded):
                            v = T.cast(x[s * block_m + i, c, j], accum_dtype) - shift[i]
                            centered_row[i, j] = v
                            squares[i, j] = v * v
                    row_reduce(centered_row, squares, acc, acc_squares, mean_val, rstd)
                else:
                    for i, j in T.Parallel(block_m, D_padded):
                        shared_buf[i, j] = T.if_then_else(
                            T.And(s * block_m + i < N, j < D),
                            x[s * block_m + i, c, j],
                            T.cast(0.0, dtype),
                        )
                        x_f32[i, j] = T.cast(shared_buf[i, j], accum_dtype)
                    row_reduce(x_f32, acc, mean_val, rstd)

                for i in T.Parallel(block_m):
                    scale[i] = rstd[i] * weight_c
                for i, j in T.Parallel(block_m, D_padded):
                    if (not guarded) or T.And(s * block_m + i < N, j < D):
                        y[s * block_m + i, c, j] = T.cast(
                            (
                                T.cast(
                                    centered_row[i, j] if register_direct else shared_buf[i, j],
                                    accum_dtype,
                                )
                                - mean_val[i]
                            )
                            * scale[i]
                            + shift_out,
                            dtype,
                        )

                # Each instance's update of the running statistics, in the input's dtype.
                for i in T.Parallel(block_m):
                    if register_direct:
                        mean_i = shift[i] + mean_val[i]
                        var_i = acc_squares[i] / float(D) - mean_val[i] * mean_val[i]
                    else:
                        mean_i = mean_val[i]
                        var_i = (acc[i] - float(D_padded - D) * mean_val[i] * mean_val[i]) / float(
                            D
                        )
                    valid = s * block_m + i < N
                    mean_update[i] = T.if_then_else(
                        valid,
                        T.cast(
                            T.cast(momentum * mean_i + (1.0 - momentum) * old_mean, dtype),
                            accum_dtype,
                        ),
                        T.float32(0.0),
                    )
                    var_update[i] = T.if_then_else(
                        valid,
                        T.cast(
                            T.cast(momentum * unbiased * var_i + (1.0 - momentum) * old_var, dtype),
                            accum_dtype,
                        ),
                        T.float32(0.0),
                    )

                for i in T.Parallel(block_m):
                    block_sums[0, i] = mean_update[i]
                    block_sums[1, i] = var_update[i]
                T.sync_threads()
                if T.get_thread_binding() == 0:
                    total_mean = T.alloc_var(accum_dtype, init=0.0)
                    total_var = T.alloc_var(accum_dtype, init=0.0)
                    for i in T.serial(block_m):
                        total_mean = total_mean + block_sums[0, i]
                        total_var = total_var + block_sums[1, i]
                    if splits == 1:
                        running_mean[c] = T.cast(T.cast(total_mean / float(N), dtype), accum_dtype)
                        running_var[c] = T.cast(T.cast(total_var / float(N), dtype), accum_dtype)
                    else:
                        partial[s, 0, c] = total_mean
                        partial[s, 1, c] = total_var

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _instance_norm_stats_kernel(N, C, splits, dtype):
    """Build the kernel adding up the blocks' sums into the running statistics.

    Thread ``c`` adds the ``splits`` sums of channel ``c`` in order, so the result
    does not depend on which block finished first.

    Args:
        N: Batch size.
        C: Number of channels.
        splits: Blocks per channel `_instance_norm_train_kernel` ran.
        dtype: TileLang dtype string of the input.
    """
    accum_dtype = "float32"

    @tilelang.jit
    def _func(threads):
        @T.prim_func
        def main(
            partial: T.Tensor[(splits, 2, C), accum_dtype],
            running_mean: T.Tensor[(C,), accum_dtype],
            running_var: T.Tensor[(C,), accum_dtype],
        ):
            with T.Kernel(T.ceildiv(C, threads), threads=threads) as bx:
                c = bx * threads + T.get_thread_binding()
                if c < C:
                    total_mean = T.alloc_var(accum_dtype, init=0.0)
                    total_var = T.alloc_var(accum_dtype, init=0.0)
                    for s in T.serial(splits):
                        total_mean = total_mean + partial[s, 0, c]
                        total_var = total_var + partial[s, 1, c]
                    running_mean[c] = T.cast(T.cast(total_mean / float(N), dtype), accum_dtype)
                    running_var[c] = T.cast(T.cast(total_var / float(N), dtype), accum_dtype)

        return main

    return _func


class _ChannelTableKernel(_RowNormKernel):
    """What the running-statistics kernels share: the ``(N, C, D)`` layout and the affine.

    An absent ``weight`` or ``bias`` is an identity the program applies itself, so the
    launch hands it a placeholder the program never reads.

    Args:
        N: Batch size.
        C: Number of channels.
        D: Elements per instance, ``prod(spatial)``.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        has_weight: Whether ``weight`` is passed.
        has_bias: Whether ``bias`` is passed.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    def __init__(
        self,
        N: int,
        C: int,
        D: int,
        eps: float,
        dtype: torch.dtype,
        has_weight: bool,
        has_bias: bool,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        self.N = N
        self.C = C
        self.has_weight = has_weight
        self.has_bias = has_bias
        super().__init__(D, eps, dtype, config=config, tune=tune)

    def _affine(self, x: torch.Tensor, weight, bias) -> tuple:
        """The affine pair to launch with, a placeholder standing for an absent tensor."""
        if (weight is not None) != self.has_weight or (bias is not None) != self.has_bias:
            raise ValueError(
                f"{type(self).__name__} was built for weight "
                f"{'present' if self.has_weight else 'absent'} and bias "
                f"{'present' if self.has_bias else 'absent'}"
            )
        placeholder = None
        if weight is None or bias is None:
            placeholder = torch.empty(self.C, dtype=x.dtype, device=x.device)
        return (
            placeholder if weight is None else weight,
            placeholder if bias is None else bias,
        )

    def _launch(self, *args):
        """Run the program, tuning it first when a tuning was requested."""
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()
        return self.kernel(**self.config)(*args)


class InstanceNormFwdTrainKernel(_ChannelTableKernel):
    """InstanceNorm forward that also updates the running statistics in place.

    Each block owns one channel and up to ``block_m`` of its samples, normalizes each
    instance by its own statistics, and the channel's ``running_mean[c]`` and
    ``running_var[c]`` move the way ``torch.nn.functional.instance_norm`` moves them: the
    batch mean of each instance's ``momentum`` update, with the unbiased variance, each
    rounded to the input's dtype. A batch that fits one block is one launch; a larger
    one takes a second launch adding the blocks' sums up in a fixed order.

    Args:
        N: Batch size.
        C: Number of channels.
        D: Elements per instance, ``prod(spatial)``; above one.
        eps: Epsilon for numerical stability.
        momentum: Weight of an instance's statistics in its update.
        dtype: Data type (float32, float16, or bfloat16).
        has_weight: Whether ``weight`` is passed.
        has_bias: Whether ``bias`` is passed.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    # Samples one block normalizes together, at most.
    _MAX_BLOCK_M = 8
    # Threads a block of several rows may take. A 1024-thread block holding two fp32
    # fragments of its rows fills a multiprocessor's register file alone.
    _MAX_THREADS = 256
    # Block width of the launch adding up the blocks' sums, one thread per channel.
    _STATS_THREADS = 128

    def __init__(
        self,
        N: int,
        C: int,
        D: int,
        eps: float,
        momentum: float,
        dtype: torch.dtype,
        has_weight: bool,
        has_bias: bool,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        self.momentum = momentum
        super().__init__(N, C, D, eps, dtype, has_weight, has_bias, config=config, tune=tune)

    @property
    def _block_m(self) -> int:
        """Rows per block: the most samples, up to the batch, that fit in one block.

        One row per block where a row alone takes more than ``_MAX_THREADS``.
        """
        row_threads = super().default_config["threads"]
        if not _holds_row_in_registers(self.D, self.D_padded):
            # A row staged through shared memory takes one block of its own: a staged
            # block of several rows compiles to a collapsed layout.
            return 1
        cap = min(self.N, self._MAX_BLOCK_M, self._MAX_THREADS // row_threads)
        block_m = 1
        while block_m * 2 <= cap:
            block_m *= 2
        return block_m

    @property
    def default_config(self) -> dict:
        block_m = self._block_m
        return {"block_m": block_m, "threads": block_m * super().default_config["threads"]}

    @property
    def autotune_configs(self) -> list[dict]:
        # block_m stays fixed: it decides how the batch is cut across blocks, and with
        # it the shape of the buffer the blocks' sums go through.
        block_m = self._block_m
        return [
            {"block_m": block_m, "threads": block_m * t}
            for t in self._row_widths
            if self.D_padded % t == 0 and (block_m == 1 or block_m * t <= self._MAX_THREADS)
        ] or [self.default_config]

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize *x* by its instance statistics and update the running ones in place.

        Args:
            x: Input of shape ``(N, C, *spatial)``, contiguous, on a CUDA device.
            running_mean: ``float32`` running mean of shape $[C]$, contiguous; updated.
            running_var: ``float32`` running variance of shape $[C]$, contiguous; updated.
            weight: Affine scale of shape $[C]$, or ``None``, as built.
            bias: Affine shift of shape $[C]$, or ``None``, as built.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device, or the affine presence differs
                from the one this kernel was built for.
        """
        self._require_cuda(
            x=x, running_mean=running_mean, running_var=running_var, weight=weight, bias=bias
        )
        weight, bias = self._affine(x, weight, bias)
        self.kernel = _instance_norm_train_kernel(
            self.N,
            self.C,
            self.D,
            self.eps,
            self.momentum,
            self.dtype_str,
            self.has_weight,
            self.has_bias,
        )
        splits = -(-self.N // self.config["block_m"])
        partial = torch.empty((splits, 2, self.C), dtype=torch.float32, device=x.device)
        rows = x.reshape(self.N, self.C, self.D)
        y = self._launch(rows, running_mean, running_var, weight, bias, partial)
        if splits > 1:
            stats = _instance_norm_stats_kernel(self.N, self.C, splits, self.dtype_str)
            stats(self._STATS_THREADS)(partial, running_mean, running_var)
        return y.reshape(x.shape)


@functools.lru_cache(maxsize=32)
def _instance_norm_infer_kernel(M, C, D, eps, dtype, has_weight, has_bias):
    """Build the kernel normalizing each ``(n, c)`` row by channel ``c``'s running statistics.

    Row ``m`` of the ``(N*C, D)`` view is channel ``m % C``; torch reads the statistics
    in *dtype*.

    Args:
        M: Number of rows, ``N * C``.
        C: Number of channels.
        D: Elements per row, ``prod(spatial)``.
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string of the input.
        has_weight: Whether ``weight`` is read; otherwise the scale is one.
        has_bias: Whether ``bias`` is read; otherwise the shift is zero.
    """
    accum_dtype = "float32"
    D_padded = row_padding(D, 4 if dtype == "float32" else 2)

    @tilelang.jit(out_idx=[5])
    def _func(block_m, threads):
        guarded = D_padded != D or M % block_m != 0

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            running_mean: T.Tensor[(C,), accum_dtype],
            running_var: T.Tensor[(C,), accum_dtype],
            weight: T.Tensor[(C,), dtype],
            bias: T.Tensor[(C,), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                mean_row = T.alloc_fragment((block_m,), "float32")
                scale = T.alloc_fragment((block_m,), "float32")
                shift = T.alloc_fragment((block_m,), "float32")
                for i in T.Parallel(block_m):
                    c = T.min(pid_m * block_m + i, M - 1) % C
                    mean_row[i] = T.cast(T.cast(running_mean[c], dtype), "float32")
                    scale[i] = T.rsqrt(T.cast(T.cast(running_var[c], dtype), "float32") + eps)
                    if has_weight:
                        scale[i] = scale[i] * T.cast(weight[c], "float32")
                    shift[i] = T.cast(bias[c], "float32") if has_bias else T.float32(0.0)
                for i, j in T.Parallel(block_m, D_padded):
                    if (not guarded) or T.And(pid_m * block_m + i < M, j < D):
                        y[pid_m * block_m + i, j] = T.cast(
                            (T.cast(x[pid_m * block_m + i, j], "float32") - mean_row[i]) * scale[i]
                            + shift[i],
                            dtype,
                        )

        return main

    return _func


class InstanceNormFwdInferKernel(_ChannelTableKernel):
    """InstanceNorm forward by the running statistics, for ``use_input_stats=False``.

    Each ``(n, c)`` row is normalized by channel ``c``'s running mean and variance,
    read in the input's dtype as torch reads them, then takes the affine.

    Args:
        N: Batch size.
        C: Number of channels.
        D: Elements per instance, ``prod(spatial)``.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        has_weight: Whether ``weight`` is passed.
        has_bias: Whether ``bias`` is passed.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize *x* by the running statistics.

        Args:
            x: Input of shape ``(N, C, *spatial)``, contiguous, on a CUDA device.
            running_mean: ``float32`` running mean of shape $[C]$, contiguous.
            running_var: ``float32`` running variance of shape $[C]$, contiguous.
            weight: Affine scale of shape $[C]$, or ``None``, as built.
            bias: Affine shift of shape $[C]$, or ``None``, as built.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device, or the affine presence differs
                from the one this kernel was built for.
        """
        self._require_cuda(
            x=x, running_mean=running_mean, running_var=running_var, weight=weight, bias=bias
        )
        weight, bias = self._affine(x, weight, bias)
        self.kernel = _instance_norm_infer_kernel(
            self.N * self.C,
            self.C,
            self.D,
            self.eps,
            self.dtype_str,
            self.has_weight,
            self.has_bias,
        )
        rows = x.reshape(-1, self.D)
        y = self._launch(rows, running_mean, running_var, weight, bias)
        return y.reshape(x.shape)
