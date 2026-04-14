"""Engram GateConv backward kernel.

Given dY (gradient of output), plus saved intermediates from forward
(vhat, alpha, rrms_h, rrms_k, rrms_v) and inputs (H, k, v, weights),
computes:
    dk, dv, dH          — gradients w.r.t. projection outputs and hidden state
    drms_w_h, drms_w_v  — gradients w.r.t. RMSNorm weights
    dconv_w             — gradient w.r.t. conv weights

Three-pass design:
    Pass 1:  SiLU+residual bwd -> dconv_w + store d_silu_out to buffer
    Pass 1b: Conv transpose -> RMSNorm(v_hat) bwd -> d(v_hat)
    Pass 2:  Gate bwd -> RMSNorm(H) bwd, RMSNorm(k) bwd -> dH, dk, dv
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["EngramGateConvBwdKernel"]

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def _engram_gate_conv_bwd_kernel(M, seq_len, d, eps, dtype):
    accum_dtype = "float"
    d_padded = _align_up(d, ALIGNMENT)
    KS = CONV_KERNEL_SIZE

    @tilelang.jit(
        out_idx=[12, 13, 14, 15, 16, 17, 18],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads):

        @T.macro
        def _gate_conv_bwd(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((KS, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            alpha: T.Tensor((M, seq_len), accum_dtype),
            rrms_h: T.Tensor((M, seq_len), accum_dtype),
            rrms_k: T.Tensor((M, seq_len), accum_dtype),
            rrms_v: T.Tensor((M, seq_len), accum_dtype),
            dH: T.Tensor((M, seq_len, d_padded), dtype),
            dk: T.Tensor((M, seq_len, d_padded), dtype),
            dv: T.Tensor((M, seq_len, d_padded), dtype),
            drms_w_h: T.Tensor((d_padded,), accum_dtype),
            drms_w_v: T.Tensor((d_padded,), accum_dtype),
            dconv_w: T.Tensor((KS, d_padded), accum_dtype),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):
            # ======== Zero-init accumulated outputs ========
            with T.Kernel(1, threads=threads) as (bx,):
                for j in T.Parallel(d_padded):
                    drms_w_h[j] = 0.0
                    drms_w_v[j] = 0.0
                for p in T.serial(KS):
                    for j in T.Parallel(d_padded):
                        dconv_w[p, j] = 0.0

            # ======== Pass 1: SiLU bwd + dconv_w ========
            with T.Kernel(seq_len, M, threads=threads) as (bx, by):
                dy_local = T.alloc_fragment((d_padded,), accum_dtype)
                conv_out = T.alloc_fragment((d_padded,), accum_dtype)
                d_silu = T.alloc_fragment((d_padded,), accum_dtype)

                bid = by
                tid = bx

                for j in T.Parallel(d_padded):
                    dy_local[j] = T.cast(dY[bid, tid, j], accum_dtype)

                # Recompute conv_out (same as forward pass 2)
                for j in T.Parallel(d_padded):
                    conv_out[j] = 0.0
                for p in T.serial(KS):
                    src_t = tid - (KS - 1) + p
                    for j in T.Parallel(d_padded):
                        raw_val = T.if_then_else(
                            src_t >= 0,
                            T.cast(vhat[bid, src_t, j], accum_dtype),
                            0.0,
                        )
                        src_rrms = T.if_then_else(
                            src_t >= 0,
                            rrms_v[bid, src_t],
                            0.0,
                        )
                        normed = raw_val * src_rrms * T.cast(rms_w_v[j], accum_dtype)
                        conv_out[j] += T.cast(conv_w[p, j], accum_dtype) * normed

                # ⑨ bwd: Y = SiLU(conv_out) + vhat
                # d_silu = dY * silu'(conv_out)
                for j in T.Parallel(d_padded):
                    sig = 1.0 / (1.0 + T.exp(-conv_out[j]))
                    d_silu[j] = dy_local[j] * (sig + conv_out[j] * sig * (1.0 - sig))

                # dconv_w[p,j] += d_silu[j] * v_hat_norm[src_t, j]
                for p in T.serial(KS):
                    src_t = tid - (KS - 1) + p
                    for j in T.Parallel(d_padded):
                        raw_val = T.if_then_else(
                            src_t >= 0,
                            T.cast(vhat[bid, src_t, j], accum_dtype),
                            0.0,
                        )
                        src_rrms = T.if_then_else(
                            src_t >= 0,
                            rrms_v[bid, src_t],
                            0.0,
                        )
                        normed = raw_val * src_rrms * T.cast(rms_w_v[j], accum_dtype)
                        T.atomic_add(dconv_w[p, j], d_silu[j] * normed)

                # Store d_silu for conv transpose in pass 1b
                for j in T.Parallel(d_padded):
                    dvhat_buf[bid, tid, j] = d_silu[j]

            # ======== Pass 1b: conv transpose + RMSNorm(vhat) bwd ========
            # d(v_hat_norm[t,j]) = sum_p conv_w[p,j] * d_silu[t+(KS-1)-p, j]
            # Then RMSNorm bwd: d(vhat) from d(v_hat_norm) + dY residual
            with T.Kernel(seq_len, M, threads=threads) as (bx, by):
                d_vnorm = T.alloc_fragment((d_padded,), accum_dtype)
                dvhat_local = T.alloc_fragment((d_padded,), accum_dtype)
                vhat_local = T.alloc_fragment((d_padded,), accum_dtype)
                vhat_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                dot_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                dot_sum = T.alloc_fragment((1,), accum_dtype)

                bid = by
                tid = bx

                # Conv transpose
                for j in T.Parallel(d_padded):
                    d_vnorm[j] = 0.0
                for p in T.serial(KS):
                    # Forward: out[t] = sum_p w[p] * input[t-(KS-1)+p]
                    # Transpose: d_input[t] = sum_p w[p] * d_out[t+(KS-1)-p]
                    dst_t = tid + (KS - 1) - p
                    for j in T.Parallel(d_padded):
                        d_si = T.if_then_else(
                            (dst_t >= 0) * (dst_t < seq_len),
                            dvhat_buf[bid, dst_t, j],
                            0.0,
                        )
                        d_vnorm[j] += T.cast(conv_w[p, j], accum_dtype) * d_si

                # Load vhat and rrms_v
                rrms_v_val = rrms_v[bid, tid]
                for j in T.Parallel(d_padded):
                    vhat_local[j] = T.cast(vhat[bid, tid, j], accum_dtype)
                    vhat_2d[0, j] = vhat_local[j]

                # drms_w_v: d(w_v[j]) += d_vnorm[j] * vhat[j] * rrms_v
                for j in T.Parallel(d_padded):
                    T.atomic_add(
                        drms_w_v[j],
                        d_vnorm[j] * vhat_local[j] * rrms_v_val,
                    )

                # ⑦ bwd: RMSNorm(vhat) -> d(vhat)
                # d(vhat) = rrms * w * d_vnorm - rrms^3 * vhat * (1/d) * sum(vhat * w * d_vnorm)
                for j in T.Parallel(d_padded):
                    dot_2d[0, j] = (
                        vhat_local[j]
                        * T.cast(rms_w_v[j], accum_dtype)
                        * d_vnorm[j]
                    )
                T.reduce_sum(dot_2d, dot_sum, dim=1)

                for j in T.Parallel(d_padded):
                    dvhat_local[j] = (
                        rrms_v_val * T.cast(rms_w_v[j], accum_dtype) * d_vnorm[j]
                        - rrms_v_val * rrms_v_val * rrms_v_val
                        * vhat_local[j] * dot_sum[0] / float(d)
                    )

                # Add residual gradient from SiLU bypass (dY)
                for j in T.Parallel(d_padded):
                    dvhat_local[j] += T.cast(dY[bid, tid, j], accum_dtype)

                # Store total d(vhat)
                for j in T.Parallel(d_padded):
                    dvhat_buf[bid, tid, j] = dvhat_local[j]

            # ======== Pass 2: gate backward ========
            with T.Kernel(seq_len, M, threads=threads) as (bx, by):
                dvhat_local = T.alloc_fragment((d_padded,), accum_dtype)
                v_local = T.alloc_fragment((d_padded,), accum_dtype)
                h_local = T.alloc_fragment((d_padded,), accum_dtype)
                k_local = T.alloc_fragment((d_padded,), accum_dtype)
                h_norm_local = T.alloc_fragment((d_padded,), accum_dtype)
                k_norm_local = T.alloc_fragment((d_padded,), accum_dtype)
                dh_local = T.alloc_fragment((d_padded,), accum_dtype)
                dk_local = T.alloc_fragment((d_padded,), accum_dtype)
                dv_local = T.alloc_fragment((d_padded,), accum_dtype)

                # 2D fragments for reduce_sum
                dalpha_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                dalpha_sum = T.alloc_fragment((1,), accum_dtype)
                dot_h_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                dot_h_sum = T.alloc_fragment((1,), accum_dtype)
                dot_k_2d = T.alloc_fragment((1, d_padded), accum_dtype)
                dot_k_sum = T.alloc_fragment((1,), accum_dtype)

                bid = by
                tid = bx

                for j in T.Parallel(d_padded):
                    dvhat_local[j] = dvhat_buf[bid, tid, j]
                    v_local[j] = T.cast(v[bid, tid, j], accum_dtype)
                    h_local[j] = T.cast(H[bid, tid, j], accum_dtype)
                    k_local[j] = T.cast(k[bid, tid, j], accum_dtype)

                alpha_val = alpha[bid, tid]
                rrms_h_val = rrms_h[bid, tid]
                rrms_k_val = rrms_k[bid, tid]

                # ⑥ bwd: vhat = alpha * v
                # dalpha = sum(dvhat * v),  dv = alpha * dvhat
                for j in T.Parallel(d_padded):
                    dalpha_2d[0, j] = dvhat_local[j] * v_local[j]
                    dv_local[j] = alpha_val * dvhat_local[j]
                T.reduce_sum(dalpha_2d, dalpha_sum, dim=1)

                # ⑤ bwd: alpha = sigmoid(dot / sqrt(d))
                # d(dot) = dalpha * alpha * (1 - alpha) / sqrt(d)
                ddot_val = dalpha_sum[0] * alpha_val * (1.0 - alpha_val) / T.sqrt(float(d))

                # Recompute h_norm, k_norm
                for j in T.Parallel(d_padded):
                    h_norm_local[j] = h_local[j] * rrms_h_val * T.cast(rms_w_h[j], accum_dtype)
                    k_norm_local[j] = k_local[j] * rrms_k_val * T.cast(rms_w_h[j], accum_dtype)

                # ③ bwd: RMSNorm(H) -> dH
                # d(h_norm) = ddot * k_norm
                for j in T.Parallel(d_padded):
                    dot_h_2d[0, j] = (
                        h_local[j]
                        * T.cast(rms_w_h[j], accum_dtype)
                        * ddot_val * k_norm_local[j]
                    )
                T.reduce_sum(dot_h_2d, dot_h_sum, dim=1)
                for j in T.Parallel(d_padded):
                    dh_local[j] = (
                        rrms_h_val * T.cast(rms_w_h[j], accum_dtype)
                        * ddot_val * k_norm_local[j]
                        - rrms_h_val * rrms_h_val * rrms_h_val
                        * h_local[j] * dot_h_sum[0] / float(d)
                    )

                # drms_w_h from H
                for j in T.Parallel(d_padded):
                    T.atomic_add(
                        drms_w_h[j],
                        ddot_val * k_norm_local[j] * h_local[j] * rrms_h_val,
                    )

                # ④ bwd: RMSNorm(k) -> dk
                for j in T.Parallel(d_padded):
                    dot_k_2d[0, j] = (
                        k_local[j]
                        * T.cast(rms_w_h[j], accum_dtype)
                        * ddot_val * h_norm_local[j]
                    )
                T.reduce_sum(dot_k_2d, dot_k_sum, dim=1)
                for j in T.Parallel(d_padded):
                    dk_local[j] = (
                        rrms_k_val * T.cast(rms_w_h[j], accum_dtype)
                        * ddot_val * h_norm_local[j]
                        - rrms_k_val * rrms_k_val * rrms_k_val
                        * k_local[j] * dot_k_sum[0] / float(d)
                    )

                # drms_w_h from k
                for j in T.Parallel(d_padded):
                    T.atomic_add(
                        drms_w_h[j],
                        ddot_val * h_norm_local[j] * k_local[j] * rrms_k_val,
                    )

                # Write outputs
                for j in T.Parallel(d_padded):
                    dH[bid, tid, j] = T.cast(dh_local[j], dtype)
                    dk[bid, tid, j] = T.cast(dk_local[j], dtype)
                    dv[bid, tid, j] = T.cast(dv_local[j], dtype)

        @T.prim_func
        def main(
            dY: T.Tensor((M, seq_len, d_padded), dtype),
            H: T.Tensor((M, seq_len, d_padded), dtype),
            k: T.Tensor((M, seq_len, d_padded), dtype),
            v: T.Tensor((M, seq_len, d_padded), dtype),
            rms_w_h: T.Tensor((d_padded,), dtype),
            rms_w_v: T.Tensor((d_padded,), dtype),
            conv_w: T.Tensor((KS, d_padded), dtype),
            vhat: T.Tensor((M, seq_len, d_padded), dtype),
            alpha: T.Tensor((M, seq_len), accum_dtype),
            rrms_h: T.Tensor((M, seq_len), accum_dtype),
            rrms_k: T.Tensor((M, seq_len), accum_dtype),
            rrms_v: T.Tensor((M, seq_len), accum_dtype),
            dH: T.Tensor((M, seq_len, d_padded), dtype),
            dk: T.Tensor((M, seq_len, d_padded), dtype),
            dv: T.Tensor((M, seq_len, d_padded), dtype),
            drms_w_h: T.Tensor((d_padded,), accum_dtype),
            drms_w_v: T.Tensor((d_padded,), accum_dtype),
            dconv_w: T.Tensor((KS, d_padded), accum_dtype),
            dvhat_buf: T.Tensor((M, seq_len, d_padded), accum_dtype),
        ):
            _gate_conv_bwd(
                dY, H, k, v, rms_w_h, rms_w_v, conv_w,
                vhat, alpha, rrms_h, rrms_k, rrms_v,
                dH, dk, dv, drms_w_h, drms_w_v, dconv_w, dvhat_buf,
            )

        return main

    return _func


@torch.library.custom_op("top::engram_gate_conv_bwd", mutates_args=())
def _engram_gate_conv_bwd_wrapped(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    dtype_str: str,
    threads: int,
    dY: torch.Tensor,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
    vhat: torch.Tensor,
    alpha: torch.Tensor,
    rrms_h: torch.Tensor,
    rrms_k: torch.Tensor,
    rrms_v: torch.Tensor,
) -> list[torch.Tensor]:
    results = _engram_gate_conv_bwd_kernel(M, seq_len, d, eps, dtype_str)(
        threads,
    )(dY, H, k, v, rms_w_h, rms_w_v, conv_w,
      vhat, alpha, rrms_h, rrms_k, rrms_v)
    return list(results)


@_engram_gate_conv_bwd_wrapped.register_fake
def _(M, seq_len, d, eps, dtype_str, threads,
      dY, H, k, v, rms_w_h, rms_w_v, conv_w,
      vhat, alpha, rrms_h, rrms_k, rrms_v):
    d_padded = _align_up(d, ALIGNMENT)
    device = dY.device
    dt = dY.dtype
    return [
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),       # dH
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),       # dk
        torch.empty((M, seq_len, d_padded), dtype=dt, device=device),       # dv
        torch.empty((d_padded,), dtype=torch.float32, device=device),       # drms_w_h
        torch.empty((d_padded,), dtype=torch.float32, device=device),       # drms_w_v
        torch.empty((CONV_KERNEL_SIZE, d_padded), dtype=torch.float32, device=device),  # dconv_w
        torch.empty((M, seq_len, d_padded), dtype=torch.float32, device=device),  # dvhat_buf
    ]


class EngramGateConvBwdKernel(Kernel):
    """Engram GateConv backward kernel.

    Three-pass backward:
      Pass 1:  SiLU bwd + dconv_w accumulation
      Pass 1b: Conv transpose + RMSNorm(vhat) bwd -> d(vhat)
      Pass 2:  Gate bwd -> dH, dk, dv + drms_w_h accumulation
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
        self.kernel = _engram_gate_conv_bwd_kernel(
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
        dY: torch.Tensor,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
        vhat: torch.Tensor,
        alpha: torch.Tensor,
        rrms_h: torch.Tensor,
        rrms_k: torch.Tensor,
        rrms_v: torch.Tensor,
    ) -> list[torch.Tensor]:
        return _engram_gate_conv_bwd_wrapped(
            self.M, self.seq_len, self.d, self.eps,
            self.dtype_str, self.config["threads"],
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v,
        )
