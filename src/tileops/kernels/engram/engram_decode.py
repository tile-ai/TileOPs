"""Engram fused decode kernels — single-token inference.

One decode step runs as two launches:

    k = e @ W_K, v = e @ W_V                             # projections
    new_conv_state[:, :-1] = conv_state[:, 1:]           # cache shift
    ---
    h_norm = RMSNorm(h, w_h)
    k_norm = RMSNorm(k, w_h)
    alpha  = sigmoid(dot(h_norm, k_norm) / sqrt(d))
    v_hat  = alpha * v
    new_conv_state[:, -1] = RMSNorm(v_hat, w_v)
    conv_out = dilated_conv(conv_w, new_conv_state)      # depthwise dilated conv
    y = SiLU(conv_out) + v_hat

The causal convolution uses kernel size w (typically 4) and dilation δ (max N-gram
order). The conv reads w values spaced δ apart from the cache:
    window = [state[-δ*(w-1)], state[-δ*(w-2)], ..., state[-δ], current]
    conv_out = sum(conv_w[p] * window[p])

Inputs:
    e_t:        (B, d_mem)              — gathered N-gram embedding for current token
    h_t:        (B, d)                  — hidden state for current token
    conv_state: (B, max_conv_len, d)    — cached v_hat_norm history (left-padded zeros)
    W_K, W_V:   (d_mem, d)              — projection weights
    rms_w_h:    (d,)                    — RMSNorm weight for h and k
    rms_w_v:    (d,)                    — RMSNorm weight for v_hat
    conv_w:     (w, d)                  — depthwise conv weights (model parameter, w=kernel size)

Outputs:
    y_t:            (B, d)              — output to add as residual to h_t
    new_conv_state: (B, max_conv_len, d)— updated conv state for next step

``forward`` left-pads conv_state to max_conv_len, so the kernels are compiled
once with max_conv_len regardless of the caller's history length.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.tiling import ALIGNMENT, align_up

__all__ = ["EngramDecodeKernel"]


def _projection_block_k(d_mem: int) -> int:
    """Choose a supported MMA K tile; the kernel predicates a partial tail."""
    if d_mem >= 64:
        return 64
    if d_mem >= 32:
        return 32
    return 16


@functools.lru_cache(maxsize=32)
def _engram_project_kernel(batch, d_mem, d_padded, max_conv_len, dtype):
    """Both projections of one decode step, plus the cache shift they do not depend on.

    Args:
        batch: batch size (compile-time).
        d_mem: memory embedding dimension.
        d_padded: model hidden dimension, rounded up to the tiling alignment.
        max_conv_len: max conv cache capacity (compile-time).
        dtype: data type string.

    Returns:
        A builder taking ``(block_m, block_n, block_k, threads, num_stages)``, writing
        ``kv`` as ``[k; v]`` in ``float32`` and every cache slot but the last.
    """
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_m: int, block_n: int, block_k: int, threads: int, num_stages: int):
        @T.macro
        def _project(e_t, weight, out, out_idx: int, row0: int, col0: int):
            """One tile of a projection: ``out[out_idx, row0:, col0:] = e_t @ weight``."""
            e_shared = T.alloc_shared((block_m, block_k), dtype)
            w_shared = T.alloc_shared((block_k, block_n), dtype)
            acc = T.alloc_fragment((block_m, block_n), accum_dtype)
            T.clear(acc)
            if d_mem % block_k == 0:
                for ko in T.Pipelined(d_mem // block_k, num_stages=num_stages):
                    for i, j in T.Parallel(block_m, block_k):
                        e_shared[i, j] = e_t[T.min(row0 + i, batch - 1), ko * block_k + j]
                    T.copy(
                        weight[ko * block_k : (ko + 1) * block_k, col0 : col0 + block_n],
                        w_shared,
                    )
                    T.gemm(e_shared, w_shared, acc)
            else:
                for ko in T.Pipelined(T.ceildiv(d_mem, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_m, block_k):
                        k_idx = ko * block_k + j
                        e_shared[i, j] = T.if_then_else(
                            k_idx < d_mem,
                            e_t[T.min(row0 + i, batch - 1), k_idx],
                            T.cast(0, dtype),
                        )
                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = ko * block_k + i
                        w_shared[i, j] = T.if_then_else(
                            k_idx < d_mem,
                            weight[k_idx, col0 + j],
                            T.cast(0, dtype),
                        )
                    T.gemm(e_shared, w_shared, acc)
            for i, j in T.Parallel(block_m, block_n):
                if row0 + i < batch:
                    out[out_idx, row0 + i, col0 + j] = acc[i, j]

        @T.prim_func
        def main(
            e_t: T.Tensor((batch, d_mem), dtype),
            W_K: T.Tensor((d_mem, d_padded), dtype),
            W_V: T.Tensor((d_mem, d_padded), dtype),
            conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
            kv: T.Tensor((2, batch, d_padded), accum_dtype),
            new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
        ):
            # Keep the two projections on separate grid slices.
            with T.Kernel(
                T.ceildiv(d_padded, block_n), 2, T.ceildiv(batch, block_m), threads=threads
            ) as (col_block, which, row_block):
                row0 = row_block * block_m
                col0 = col_block * block_n
                if which == 0:
                    _project(e_t, W_K, kv, 0, row0, col0)
                    if max_conv_len > 1 and row_block == 0:
                        for bs, j in T.Parallel(batch * (max_conv_len - 1), block_n):
                            b = bs // (max_conv_len - 1)
                            s = bs % (max_conv_len - 1)
                            new_conv_state[b, s, col0 + j] = conv_state[b, s + 1, col0 + j]
                else:
                    _project(e_t, W_V, kv, 1, row0, col0)

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _engram_step_kernel(batch, d, d_padded, max_conv_len, conv_kernel_size, dilation, eps, dtype):
    """Everything in a decode step that reduces over d, one thread block per batch row.

    Args:
        batch: batch size (compile-time).
        d: model hidden dimension, as the norms divide by it.
        d_padded: ``d`` rounded up to the tiling alignment; padding holds zeros.
        max_conv_len: max conv cache capacity (compile-time).
        conv_kernel_size: number of conv taps (w), e.g. 4 (compile-time, model param).
        dilation: dilation factor (δ), e.g. max N-gram order (compile-time, model param).
        eps: RMSNorm epsilon.
        dtype: data type string.

    Returns:
        A builder taking ``(threads,)``, writing ``y_t`` and the last cache slot.
    """
    accum_dtype = "float"
    w = conv_kernel_size

    @tilelang.jit(
        out_idx=[],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads: int):
        @T.prim_func
        def main(
            h_t: T.Tensor((batch, d_padded), dtype),
            kv: T.Tensor((2, batch, d_padded), accum_dtype),
            conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((w, d_padded), dtype),
            y_t: T.Tensor((batch, d_padded), dtype),
            new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
        ):
            with T.Kernel(batch, threads=threads) as (bid,):
                k_local = T.alloc_fragment((d_padded,), accum_dtype)
                v_local = T.alloc_fragment((d_padded,), accum_dtype)
                h_local = T.alloc_fragment((d_padded,), accum_dtype)
                vhat_local = T.alloc_fragment((d_padded,), accum_dtype)
                conv_out = T.alloc_fragment((d_padded,), accum_dtype)

                hsq_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                ksq_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                vsq_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                hk_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                sumsq_h = T.alloc_fragment((1,), accum_dtype)
                sumsq_k = T.alloc_fragment((1,), accum_dtype)
                sumsq_v = T.alloc_fragment((1,), accum_dtype)
                dot_hk = T.alloc_fragment((1,), accum_dtype)

                for j in T.Parallel(d_padded):
                    k_local[j] = kv[0, bid, j]
                    v_local[j] = kv[1, bid, j]
                    h_local[j] = T.cast(h_t[bid, j], accum_dtype)

                for j in T.Parallel(d_padded):
                    hsq_2d[0, j] = h_local[j] * h_local[j]
                T.reduce_sum(hsq_2d, sumsq_h, dim=1)
                rrms_h = T.rsqrt(sumsq_h[0] / float(d) + eps)
                for j in T.Parallel(d_padded):
                    h_local[j] = h_local[j] * rrms_h * T.cast(rms_w_h[j], accum_dtype)

                for j in T.Parallel(d_padded):
                    ksq_2d[0, j] = k_local[j] * k_local[j]
                T.reduce_sum(ksq_2d, sumsq_k, dim=1)
                rrms_k = T.rsqrt(sumsq_k[0] / float(d) + eps)
                for j in T.Parallel(d_padded):
                    k_local[j] = k_local[j] * rrms_k * T.cast(rms_w_h[j], accum_dtype)

                for j in T.Parallel(d_padded):
                    hk_2d[0, j] = h_local[j] * k_local[j]
                T.reduce_sum(hk_2d, dot_hk, dim=1)
                alpha = 1.0 / (1.0 + T.exp(-(dot_hk[0] / T.sqrt(float(d)))))

                for j in T.Parallel(d_padded):
                    vhat_local[j] = alpha * v_local[j]

                for j in T.Parallel(d_padded):
                    vsq_2d[0, j] = vhat_local[j] * vhat_local[j]
                T.reduce_sum(vsq_2d, sumsq_v, dim=1)
                rrms_v = T.rsqrt(sumsq_v[0] / float(d) + eps)
                for j in T.Parallel(d_padded):
                    vhat_local[j] = vhat_local[j] * rrms_v * T.cast(rms_w_v[j], accum_dtype)
                # vhat_local is now v_hat_norm

                for j in T.Parallel(d_padded):
                    new_conv_state[bid, max_conv_len - 1, j] = T.cast(vhat_local[j], dtype)

                # Tap p needs the value from (w-1-p)*δ steps ago. Against the cache as
                # it was on entry, whose last slot is one step ago, that is index
                # max_conv_len - (w-1-p)*δ; tap w-1 is this step, held in vhat_local.
                for j in T.Parallel(d_padded):
                    conv_out[j] = T.cast(conv_w[w - 1, j], accum_dtype) * vhat_local[j]
                for p in T.serial(w - 1):
                    for j in T.Parallel(d_padded):
                        conv_out[j] += T.cast(conv_w[p, j], accum_dtype) * T.cast(
                            conv_state[bid, max_conv_len - (w - 1 - p) * dilation, j],
                            accum_dtype,
                        )

                for j in T.Parallel(d_padded):
                    sig = 1.0 / (1.0 + T.exp(-conv_out[j]))
                    y_t[bid, j] = T.cast(
                        conv_out[j] * sig + alpha * v_local[j],
                        dtype,
                    )

        return main

    return _func


@torch.library.custom_op("tileops::engram_decode", mutates_args=())
def _engram_decode_wrapped(
    batch: int,
    d_mem: int,
    d: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    step_threads: int,
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    conv_state: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> list[torch.Tensor]:
    d_padded = align_up(d, ALIGNMENT)
    kv = torch.empty((2, batch, d_padded), dtype=torch.float32, device=e_t.device)
    y_t = torch.empty((batch, d_padded), dtype=e_t.dtype, device=e_t.device)
    new_conv_state = torch.empty(
        (batch, max_conv_len, d_padded), dtype=e_t.dtype, device=e_t.device
    )
    _engram_project_kernel(batch, d_mem, d_padded, max_conv_len, dtype_str)(
        block_m, block_n, block_k, threads, num_stages
    )(e_t, W_K, W_V, conv_state, kv, new_conv_state)
    _engram_step_kernel(
        batch, d, d_padded, max_conv_len, conv_kernel_size, dilation, eps, dtype_str
    )(step_threads)(h_t, kv, conv_state, rms_w_h, rms_w_v, conv_w, y_t, new_conv_state)
    return [y_t, new_conv_state]


@_engram_decode_wrapped.register_fake
def _(
    batch,
    d_mem,
    d,
    max_conv_len,
    conv_kernel_size,
    dilation,
    eps,
    dtype_str,
    block_m,
    block_n,
    block_k,
    num_stages,
    threads,
    step_threads,
    e_t,
    h_t,
    conv_state,
    W_K,
    W_V,
    rms_w_h,
    rms_w_v,
    conv_w,
):
    d_padded = align_up(d, ALIGNMENT)
    device = e_t.device
    dt = e_t.dtype
    return [
        torch.empty((batch, d_padded), dtype=dt, device=device),
        torch.empty((batch, max_conv_len, d_padded), dtype=dt, device=device),
    ]


class EngramDecodeKernel(Kernel):
    """Engram fused decode kernel — full single-token pipeline.

    Runs the projections and the cache shift split over ``d``, then the RMSNorm
    gating, dilated causal conv and SiLU activation, which reduce over all of
    ``d``, one thread block per batch row.

    Args:
        batch: batch size.
        d_mem: memory embedding dimension.
        d: model hidden dimension.
        max_conv_len: max conv cache capacity (compile-time).
        conv_kernel_size: number of conv taps (w=4 in paper).
        dilation: dilation factor (δ = max N-gram order in paper).
        eps: RMSNorm epsilon.
        dtype: data type.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        d_mem: int,
        d: int,
        max_conv_len: int,
        conv_kernel_size: int,
        dilation: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.d_mem = d_mem
        self.d = d
        self.max_conv_len = max_conv_len
        self.conv_kernel_size = conv_kernel_size
        self.dilation = dilation
        self.eps = eps
        self.dtype = dtype
        self.d_padded = align_up(d, ALIGNMENT)

        min_cache = dilation * (conv_kernel_size - 1)
        if max_conv_len < min_cache:
            raise ValueError(
                f"max_conv_len ({max_conv_len}) must be >= "
                f"dilation * (conv_kernel_size - 1) = {min_cache}"
            )

        self.step_jit = _engram_step_kernel(
            batch,
            d,
            self.d_padded,
            max_conv_len,
            conv_kernel_size,
            dilation,
            eps,
            self.dtype_str,
        )
        self.kernel = _engram_project_kernel(
            batch,
            d_mem,
            self.d_padded,
            max_conv_len,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 16,
            "block_n": min(64, self.d_padded),
            "block_k": _projection_block_k(self.d_mem),
            "num_stages": 4,
            "threads": 128,
            "step_threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_k = _projection_block_k(self.d_mem)
        configs = []
        for block_n in (64, 128):
            if block_n > self.d_padded:
                continue
            threads = (128,) if block_n == 64 else (128, 256)
            for candidate_k in (block_k, block_k // 2):
                if candidate_k < 16:
                    continue
                stages = (2, 3) if block_n == 128 and candidate_k == 64 else (2, 3, 4)
                for num_stages in stages:
                    for num_threads in threads:
                        configs.append(
                            {
                                "block_m": 16,
                                "block_n": block_n,
                                "block_k": candidate_k,
                                "num_stages": num_stages,
                                "threads": num_threads,
                            }
                        )
        return configs

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        """Tune the projection and reduction launches independently."""
        print(f"Start autotuning {self.__class__.__name__} projections...")
        projection = self.tune_jit_kernel(
            self.kernel,
            self.autotune_configs,
            warmup=warmup,
            rep=rep,
        )
        print(f"Start autotuning {self.__class__.__name__} step...")
        step = self.tune_jit_kernel(
            self.step_jit,
            [{"threads": threads} for threads in (128, 256, 512)],
            warmup=warmup,
            rep=rep,
            seed_config={"threads": self.default_config["step_threads"]},
            supply_prog=None,
        )
        config = dict(projection.config)
        config["step_threads"] = step.config["threads"]
        self.config = config
        print(f"Best config: {config}")

    def forward(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        conv_state: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run one fused decode step.

        Args:
            e_t: Gathered N-gram embedding, shape $[B \\times d\\_mem]$.
            h_t: Hidden state for the current token, shape $[B \\times d]$.
            conv_state: Conv history of shape $[B \\times L \\times d]$ with
                ``L <= max_conv_len``; left-padded to the compiled capacity
                here.
            W_K: Key projection weight of shape $[d\\_mem \\times d]$.
            W_V: Value projection weight of shape $[d\\_mem \\times d]$.
            rms_w_h: RMSNorm weight of shape $[d]$ for ``h`` and ``k``.
            rms_w_v: RMSNorm weight of shape $[d]$ for the gated value.
            conv_w: Depthwise conv weights of shape $[w \\times d]$.

        Returns:
            ``[y_t, new_conv_state]`` of shapes $[B \\times d]$ and
            ``(B, max_conv_len, d)``. The alignment padding the prim_func
            requires is applied and trimmed here.
        """
        state_pad = self.max_conv_len - conv_state.shape[1]
        if state_pad > 0:
            conv_state = F.pad(conv_state, (0, 0, state_pad, 0))
        pad = self.d_padded - self.d
        if pad:
            h_t = F.pad(h_t, (0, pad))
            conv_state = F.pad(conv_state, (0, pad))
            W_K = F.pad(W_K, (0, pad))
            W_V = F.pad(W_V, (0, pad))
            rms_w_h = F.pad(rms_w_h, (0, pad))
            rms_w_v = F.pad(rms_w_v, (0, pad))
            conv_w = F.pad(conv_w, (0, pad))
        results = _engram_decode_wrapped(
            self.batch,
            self.d_mem,
            self.d,
            self.max_conv_len,
            self.conv_kernel_size,
            self.dilation,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["step_threads"],
            e_t,
            h_t,
            conv_state,
            W_K,
            W_V,
            rms_w_h,
            rms_w_v,
            conv_w,
        )
        if pad:
            results[0] = results[0][:, : self.d]
            results[1] = results[1][:, :, : self.d]
        return results
