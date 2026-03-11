"""Engram GateConv forward kernel (post-projection fusion).

Assumes projection (k = E @ W_K, v = E @ W_V) is done externally via GEMM.
This kernel fuses the remaining memory-bound operations:

    h_norm = RMSNorm(H, w_h)                                   # (M, T, d)
    k_norm = RMSNorm(k, w_h)                                   # (M, T, d)
    alpha  = sigmoid(dot(h_norm, k_norm) / sqrt(d))             # (M, T)
    v_hat  = alpha * v                                          # (M, T, d)
    v_hat_norm = RMSNorm(v_hat, w_v)                            # (M, T, d)
    conv_out = CausalDWConv1D(v_hat_norm, conv_w, kernel=4)     # (M, T, d)
    Y = SiLU(conv_out) + v_hat                                  # (M, T, d)

Saved for backward (strategy B):
    alpha, rrms_h, rrms_k, rrms_v  — all (M, T) scalars per position

Grid: (T, M) — one thread block per (batch, position).
Uses 2D fragments (1, d_padded) to leverage T.reduce_sum for safe reductions.
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["EngramGateConvFwdKernel"]

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype):
    accum_dtype = "float"
    d_padded = _align_up(d, ALIGNMENT)

    @tilelang.jit(
        out_idx=[6, 7, 8, 9, 10, 11],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads):

        @T.macro
        def _gate_fwd(
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((CONV_KERNEL_SIZE, d_padded), dtype),
            Y: T.Tensor((M, seq_len, d_padded), dtype),
            vhat_buf: T.Tensor((M, seq_len, d_padded), dtype),
            alpha_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_h_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_k_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_v_buf: T.Tensor((M, seq_len), accum_dtype),
        ):
            # ======== Pass 1: RMSNorm + gate -> vhat ========
            with T.Kernel(seq_len, M, threads=threads) as (bx, by):
                # 2D fragments for reduce_sum compatibility
                h_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                k_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                v_1d = T.alloc_fragment((d_padded,), accum_dtype)
                hsq = T.alloc_fragment((1, d_padded), accum_dtype)
                ksq = T.alloc_fragment((1, d_padded), accum_dtype)
                hk_prod = T.alloc_fragment((1, d_padded), accum_dtype)
                sumsq_h = T.alloc_fragment((1,), accum_dtype)
                sumsq_k = T.alloc_fragment((1,), accum_dtype)
                dot_hk = T.alloc_fragment((1,), accum_dtype)

                bid = by
                tid = bx

                # Load H and k as 2D (1, d_padded)
                for j in T.Parallel(d_padded):
                    h_2d[0, j] = T.cast(H[bid, tid, j], accum_dtype)
                    k_2d[0, j] = T.cast(k[bid, tid, j], accum_dtype)
                    v_1d[j] = T.cast(v[bid, tid, j], accum_dtype)

                # sumsq_h = sum(h^2)
                for j in T.Parallel(d_padded):
                    hsq[0, j] = h_2d[0, j] * h_2d[0, j]
                T.reduce_sum(hsq, sumsq_h, dim=1)

                # sumsq_k = sum(k^2)
                for j in T.Parallel(d_padded):
                    ksq[0, j] = k_2d[0, j] * k_2d[0, j]
                T.reduce_sum(ksq, sumsq_k, dim=1)

                # rrms values
                rrms_h_val = T.rsqrt(sumsq_h[0] / float(d) + eps)
                rrms_k_val = T.rsqrt(sumsq_k[0] / float(d) + eps)
                rrms_h_buf[bid, tid] = rrms_h_val
                rrms_k_buf[bid, tid] = rrms_k_val

                # h_norm, k_norm and their dot product
                for j in T.Parallel(d_padded):
                    h_2d[0, j] = h_2d[0, j] * rrms_h_val * T.cast(rms_w_h[j], accum_dtype)
                    k_2d[0, j] = k_2d[0, j] * rrms_k_val * T.cast(rms_w_h[j], accum_dtype)
                for j in T.Parallel(d_padded):
                    hk_prod[0, j] = h_2d[0, j] * k_2d[0, j]
                T.reduce_sum(hk_prod, dot_hk, dim=1)

                # alpha = sigmoid(dot / sqrt(d))
                alpha_val = 1.0 / (1.0 + T.exp(-(dot_hk[0] / T.sqrt(float(d)))))
                alpha_buf[bid, tid] = alpha_val

                # vhat = alpha * v, and compute rrms_v while vhat is in registers
                vhsq = T.alloc_fragment((1, d_padded), accum_dtype)
                sumsq_v = T.alloc_fragment((1,), accum_dtype)
                for j in T.Parallel(d_padded):
                    vhat_val = alpha_val * v_1d[j]
                    vhat_buf[bid, tid, j] = T.cast(vhat_val, dtype)
                    vhsq[0, j] = vhat_val * vhat_val
                T.reduce_sum(vhsq, sumsq_v, dim=1)
                rrms_v_buf[bid, tid] = T.rsqrt(sumsq_v[0] / float(d) + eps)

            # ======== Pass 2: conv + SiLU + residual (rrms_v already computed) ========
            with T.Kernel(seq_len, M, threads=threads) as (bx, by):
                conv_out = T.alloc_fragment((d_padded,), accum_dtype)
                vhat_cur = T.alloc_fragment((d_padded,), accum_dtype)

                bid = by
                tid = bx

                # Load vhat at current position (for residual)
                for j in T.Parallel(d_padded):
                    vhat_cur[j] = T.cast(vhat_buf[bid, tid, j], accum_dtype)

                # Causal DWConv1D: conv_out[j] = sum_p conv_w[p,j] * norm(vhat[t-p,j])
                for j in T.Parallel(d_padded):
                    conv_out[j] = 0.0

                # PyTorch conv1d with left-padding: output[t] = sum_p w[p] * input[t - (K-1) + p]
                # So conv_w[p] multiplies vhat_norm at position t - (K-1) + p
                for p in T.serial(CONV_KERNEL_SIZE):
                    src_t = tid - (CONV_KERNEL_SIZE - 1) + p
                    for j in T.Parallel(d_padded):
                        raw_val = T.if_then_else(
                            src_t >= 0,
                            T.cast(vhat_buf[bid, src_t, j], accum_dtype),
                            0.0,
                        )
                        src_rrms = T.if_then_else(
                            src_t >= 0,
                            rrms_v_buf[bid, src_t],
                            0.0,
                        )
                        normed = raw_val * src_rrms * T.cast(rms_w_v[j], accum_dtype)
                        conv_out[j] += T.cast(conv_w[p, j], accum_dtype) * normed

                # SiLU(conv_out) + vhat (residual)
                for j in T.Parallel(d_padded):
                    sig = 1.0 / (1.0 + T.exp(-conv_out[j]))
                    Y[bid, tid, j] = T.cast(conv_out[j] * sig + vhat_cur[j], dtype)

        @T.prim_func
        def main(
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((CONV_KERNEL_SIZE, d_padded), dtype),
            Y: T.Tensor((M, seq_len, d_padded), dtype),
            vhat_buf: T.Tensor((M, seq_len, d_padded), dtype),
            alpha_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_h_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_k_buf: T.Tensor((M, seq_len), accum_dtype),
            rrms_v_buf: T.Tensor((M, seq_len), accum_dtype),
        ):
            _gate_fwd(
                H, k, v, rms_w_h, rms_w_v, conv_w,
                Y, vhat_buf, alpha_buf, rrms_h_buf, rrms_k_buf, rrms_v_buf,
            )

        return main

    return _func


@torch.library.custom_op("top::engram_gate_conv_fwd", mutates_args=())
def _engram_gate_conv_fwd_wrapped(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    dtype_str: str,
    threads: int,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> list[torch.Tensor]:
    results = _engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype_str)(
        threads,
    )(H, k, v, rms_w_h, rms_w_v, conv_w)
    return list(results)


@_engram_gate_conv_fwd_wrapped.register_fake
def _(M, seq_len, d, eps, dtype_str, threads, H, k, v, rms_w_h, rms_w_v, conv_w):
    d_padded = _align_up(d, ALIGNMENT)
    device = H.device
    dt = H.dtype
    return [
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),     # Y
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),     # vhat_buf
        torch.empty((M, seq_len), dtype=torch.float32, device=device),    # alpha
        torch.empty((M, seq_len), dtype=torch.float32, device=device),    # rrms_h
        torch.empty((M, seq_len), dtype=torch.float32, device=device),    # rrms_k
        torch.empty((M, seq_len), dtype=torch.float32, device=device),    # rrms_v
    ]


class EngramGateConvFwdKernel(Kernel):
    """Engram GateConv forward kernel.

    Fuses RMSNorm-based scalar gating and depthwise causal Conv1D
    with SiLU activation. Projection (GEMM) is done externally.

    Inputs:  H (M,T,d), k (M,T,d), v (M,T,d), rms_w_h, rms_w_v, conv_w
    Outputs: Y (M,T,d), vhat (M,T,d), alpha (M,T), rrms_h/k/v (M,T)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        seq_len: int,
        d: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.eps = eps
        self.dtype = dtype
        self.d_padded = _align_up(d, ALIGNMENT)
        self.kernel = _engram_gate_conv_fwd_kernel(
            M, seq_len, d, eps, self.dtype_str,
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
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> list[torch.Tensor]:
        return _engram_gate_conv_fwd_wrapped(
            self.M, self.seq_len, self.d, self.eps,
            self.dtype_str, self.config["threads"],
            H, k, v, rms_w_h, rms_w_v, conv_w,
        )
