"""Engram fused decode kernel — single-token inference.

In decode mode (seq_len=1), all operations are fused into one kernel per batch:

    k = e @ W_K                                          # GEMV: (d_mem,) -> (d,)
    v = e @ W_V                                          # GEMV: (d_mem,) -> (d,)
    h_norm = RMSNorm(h, w_h)
    k_norm = RMSNorm(k, w_h)
    alpha  = sigmoid(dot(h_norm, k_norm) / sqrt(d))
    v_hat  = alpha * v
    v_hat_norm = RMSNorm(v_hat, w_v)
    conv_state = shift_left(conv_state) | v_hat_norm     # update cache
    conv_out = dilated_conv(conv_w, conv_state, current)  # depthwise dilated conv
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

Grid: (B,) — one thread block per batch element.

The conv_state is padded to max_conv_len by the op before calling this kernel,
so the kernel is compiled once with max_conv_len.
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["EngramDecodeKernel"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _engram_decode_kernel(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, eps, dtype):
    """
    Args:
        batch: batch size (compile-time).
        d_mem: memory embedding dimension.
        d: model hidden dimension.
        max_conv_len: max conv cache capacity (compile-time). Must >= dilation * (conv_kernel_size - 1).
        conv_kernel_size: number of conv taps (w), e.g. 4 (compile-time, model param).
        dilation: dilation factor (δ), e.g. max N-gram order (compile-time, model param).
        eps: RMSNorm epsilon.
        dtype: data type string.
    """
    accum_dtype = "float"
    d_padded = _align_up(d, ALIGNMENT)
    w = conv_kernel_size

    @tilelang.jit(
        out_idx=[8, 9],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads):

        @T.macro
        def _decode_fused(
            e_t: T.Tensor((batch, d_mem), dtype),
            h_t: T.Tensor((batch, d_padded), dtype),
            conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
            W_K: T.Tensor((d_mem, d_padded), dtype),
            W_V: T.Tensor((d_mem, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((w, d_padded), dtype),
            y_t: T.Tensor((batch, d_padded), dtype),
            new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
        ):
            with T.Kernel(batch, threads=threads) as (bid,):
                # --- Registers ---
                e_local = T.alloc_fragment((d_mem,), accum_dtype)
                k_local = T.alloc_fragment((d_padded,), accum_dtype)
                v_local = T.alloc_fragment((d_padded,), accum_dtype)
                h_local = T.alloc_fragment((d_padded,), accum_dtype)
                vhat_local = T.alloc_fragment((d_padded,), accum_dtype)
                conv_out = T.alloc_fragment((d_padded,), accum_dtype)

                # 2D fragments for T.reduce_sum
                hsq_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                ksq_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                vsq_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                hk_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                sumsq_h = T.alloc_fragment((1,), accum_dtype)
                sumsq_k = T.alloc_fragment((1,), accum_dtype)
                sumsq_v = T.alloc_fragment((1,), accum_dtype)
                dot_hk = T.alloc_fragment((1,), accum_dtype)

                # ======== Step 1: Load e_t ========
                for i in T.Parallel(d_mem):
                    e_local[i] = T.cast(e_t[bid, i], accum_dtype)

                # ======== Step 2: GEMV projections ========
                for j in T.Parallel(d_padded):
                    k_local[j] = 0.0
                    v_local[j] = 0.0
                for i in T.serial(d_mem):
                    for j in T.Parallel(d_padded):
                        k_local[j] += e_local[i] * T.cast(W_K[i, j], accum_dtype)
                        v_local[j] += e_local[i] * T.cast(W_V[i, j], accum_dtype)

                # ======== Step 3: Load h_t ========
                for j in T.Parallel(d_padded):
                    h_local[j] = T.cast(h_t[bid, j], accum_dtype)

                # ======== Step 4: RMSNorm(h) ========
                for j in T.Parallel(d_padded):
                    hsq_2d[0, j] = h_local[j] * h_local[j]
                T.reduce_sum(hsq_2d, sumsq_h, dim=1)
                rrms_h = T.rsqrt(sumsq_h[0] / float(d) + eps)
                for j in T.Parallel(d_padded):
                    h_local[j] = h_local[j] * rrms_h * T.cast(rms_w_h[j], accum_dtype)

                # ======== Step 5: RMSNorm(k) ========
                for j in T.Parallel(d_padded):
                    ksq_2d[0, j] = k_local[j] * k_local[j]
                T.reduce_sum(ksq_2d, sumsq_k, dim=1)
                rrms_k = T.rsqrt(sumsq_k[0] / float(d) + eps)
                for j in T.Parallel(d_padded):
                    k_local[j] = k_local[j] * rrms_k * T.cast(rms_w_h[j], accum_dtype)

                # ======== Step 6: Scalar gate ========
                for j in T.Parallel(d_padded):
                    hk_2d[0, j] = h_local[j] * k_local[j]
                T.reduce_sum(hk_2d, dot_hk, dim=1)
                alpha = 1.0 / (1.0 + T.exp(-(dot_hk[0] / T.sqrt(float(d)))))

                # ======== Step 7: v_hat = alpha * v ========
                for j in T.Parallel(d_padded):
                    vhat_local[j] = alpha * v_local[j]

                # ======== Step 8: RMSNorm(v_hat) ========
                for j in T.Parallel(d_padded):
                    vsq_2d[0, j] = vhat_local[j] * vhat_local[j]
                T.reduce_sum(vsq_2d, sumsq_v, dim=1)
                rrms_v = T.rsqrt(sumsq_v[0] / float(d) + eps)
                for j in T.Parallel(d_padded):
                    vhat_local[j] = vhat_local[j] * rrms_v * T.cast(rms_w_v[j], accum_dtype)
                # vhat_local is now v_hat_norm

                # ======== Step 9: Dilated causal conv ========
                # Paper: kernel size w, dilation δ.
                # Window = [state[-(w-1)*δ], state[-(w-2)*δ], ..., state[-δ], current]
                # conv_out = sum_{p=0}^{w-1} conv_w[p] * window[p]
                #
                # State layout (left-padded zeros, valid entries right-aligned):
                #   state[max_conv_len-1] = most recent (1 step ago)
                #   state[max_conv_len-k] = k steps ago
                # Tap p needs the value from (w-1-p)*δ steps ago:
                #   state_idx = max_conv_len - (w-1-p) * δ
                # For p=w-1 (current): we use vhat_local directly.
                # Zeros in padded region automatically contribute nothing.

                for j in T.Parallel(d_padded):
                    conv_out[j] = 0.0

                # History taps (p = 0 to w-2): read from state at dilated positions
                for p in T.serial(w - 1):
                    for j in T.Parallel(d_padded):
                        conv_out[j] += (
                            T.cast(conv_w[p, j], accum_dtype)
                            * T.cast(
                                conv_state[bid, max_conv_len - (w - 1 - p) * dilation, j],
                                accum_dtype,
                            )
                        )
                # Current tap (p = w-1)
                for j in T.Parallel(d_padded):
                    conv_out[j] += T.cast(conv_w[w - 1, j], accum_dtype) * vhat_local[j]

                # ======== Step 10: Update conv_state ========
                # Shift left by 1, append v_hat_norm at end.
                # Zeros shift out from the left at early steps — correct by design.
                for s in T.serial(max_conv_len - 1):
                    for j in T.Parallel(d_padded):
                        new_conv_state[bid, s, j] = conv_state[bid, s + 1, j]
                for j in T.Parallel(d_padded):
                    new_conv_state[bid, max_conv_len - 1, j] = T.cast(vhat_local[j], dtype)

                # ======== Step 11: SiLU + residual ========
                # y = SiLU(conv_out) + v_hat
                # Recompute v_hat = alpha * v (alpha and v still in scope)
                for j in T.Parallel(d_padded):
                    sig = 1.0 / (1.0 + T.exp(-conv_out[j]))
                    y_t[bid, j] = T.cast(
                        conv_out[j] * sig + alpha * v_local[j],
                        dtype,
                    )

        @T.prim_func
        def main(
            e_t: T.Tensor((batch, d_mem), dtype),
            h_t: T.Tensor((batch, d_padded), dtype),
            conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
            W_K: T.Tensor((d_mem, d_padded), dtype),
            W_V: T.Tensor((d_mem, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((w, d_padded), dtype),
            y_t: T.Tensor((batch, d_padded), dtype),
            new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype),
        ):
            _decode_fused(
                e_t, h_t, conv_state,
                W_K, W_V, rms_w_h, rms_w_v, conv_w,
                y_t, new_conv_state,
            )

        return main

    return _func


@torch.library.custom_op("top::engram_decode", mutates_args=())
def _engram_decode_wrapped(
    batch: int,
    d_mem: int,
    d: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    eps: float,
    dtype_str: str,
    threads: int,
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    conv_state: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> list[torch.Tensor]:
    results = _engram_decode_kernel(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, eps, dtype_str,
    )(threads)(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
    return list(results)


@_engram_decode_wrapped.register_fake
def _(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation,
      eps, dtype_str, threads,
      e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w):
    d_padded = _align_up(d, ALIGNMENT)
    device = e_t.device
    dt = e_t.dtype
    return [
        torch.empty((batch, d_padded), dtype=dt, device=device),
        torch.empty((batch, max_conv_len, d_padded), dtype=dt, device=device),
    ]


class EngramDecodeKernel(Kernel):
    """Engram fused decode kernel — full single-token pipeline.

    Fuses GEMV projection, RMSNorm gating, dilated causal conv with state
    update, and SiLU activation into a single kernel launch.

    The conv_state is padded to max_conv_len by the op before calling this kernel.

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
        self.d_padded = _align_up(d, ALIGNMENT)

        min_cache = dilation * (conv_kernel_size - 1)
        if max_conv_len < min_cache:
            raise ValueError(
                f"max_conv_len ({max_conv_len}) must be >= "
                f"dilation * (conv_kernel_size - 1) = {min_cache}"
            )

        self.kernel = _engram_decode_kernel(
            batch, d_mem, d, max_conv_len, conv_kernel_size, dilation,
            eps, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t} for t in [128, 256, 512]]

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
        return _engram_decode_wrapped(
            self.batch, self.d_mem, self.d, self.max_conv_len,
            self.conv_kernel_size, self.dilation, self.eps,
            self.dtype_str, self.config["threads"],
            e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
        )
