"""Fused single-token Engram decode.

The first launch projects K/V and shifts the convolution cache. The second
normalizes, gates, updates the cache, performs the depthwise dilated
convolution, and adds the residual. ``forward`` pads history to the compiled
maximum cache length.
"""

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._kernels.sequence_modeling.engram.call_spec import (
    EngramDecodeCall,
    EngramDecodeFwdInterface,
)
from tileops._kernels.tiling import ALIGNMENT, align_up

__all__ = ["EngramDecodeKernel", "EngramDecodeSplitKernel"]


def _left_pad_history(conv_state: torch.Tensor, max_conv_len: int) -> torch.Tensor:
    """*conv_state* with zero steps prepended up to the compiled cache length."""
    padding = max_conv_len - conv_state.shape[1]
    return F.pad(conv_state, (0, 0, padding, 0)) if padding > 0 else conv_state


@target_cache(maxsize=32)
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


@target_cache(maxsize=32)
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
                # vhat_local holds v_hat_norm.

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


class EngramDecodeKernel(Kernel, EngramDecodeFwdInterface):
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

    @staticmethod
    def _projection_block_k(d_mem: int) -> int:
        """Choose a supported MMA K tile; the kernel predicates a partial tail."""
        if d_mem >= 64:
            return 64
        if d_mem >= 32:
            return 32
        return 16

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: EngramDecodeCall) -> Entry:
        return call, lambda: cls(
            call.batch,
            call.d_mem,
            call.d,
            call.max_conv_len,
            call.conv_kernel_size,
            call.dilation,
            call.eps,
            call.dtype,
        )

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
        config: dict | None = None,
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
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 16,
            "block_n": min(64, self.d_padded),
            "block_k": self._projection_block_k(self.d_mem),
            "num_stages": 4,
            "threads": 128,
            "step_threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_k = self._projection_block_k(self.d_mem)
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
        projection = self.tune_jit_kernel(
            self.kernel,
            self.autotune_configs,
            warmup=warmup,
            rep=rep,
        )
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
        self._publish_tuned(config)

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
    ) -> tuple[torch.Tensor, ...]:
        """Run one fused decode step, applying and trimming the alignment padding the
        prim_func requires."""
        conv_state = _left_pad_history(conv_state, self.max_conv_len)
        pad = self.d_padded - self.d
        if pad:
            h_t = F.pad(h_t, (0, pad))
            conv_state = F.pad(conv_state, (0, pad))
            W_K = F.pad(W_K, (0, pad))
            W_V = F.pad(W_V, (0, pad))
            rms_w_h = F.pad(rms_w_h, (0, pad))
            rms_w_v = F.pad(rms_w_v, (0, pad))
            conv_w = F.pad(conv_w, (0, pad))
        project, step = self.program(self._compile)
        kv = torch.empty((2, self.batch, self.d_padded), dtype=torch.float32, device=e_t.device)
        y_t = torch.empty((self.batch, self.d_padded), dtype=e_t.dtype, device=e_t.device)
        new_conv_state = torch.empty(
            (self.batch, self.max_conv_len, self.d_padded), dtype=e_t.dtype, device=e_t.device
        )
        project(e_t, W_K, W_V, conv_state, kv, new_conv_state)
        step(h_t, kv, conv_state, rms_w_h, rms_w_v, conv_w, y_t, new_conv_state)
        if pad:
            return y_t[:, : self.d], new_conv_state[:, :, : self.d]
        return y_t, new_conv_state

    def _compile(self, cfg: dict) -> tuple[CompiledProgram, CompiledProgram]:
        """The projection and step programs ``forward`` launches, compiled from *cfg*."""
        project = self.kernel(
            cfg["block_m"], cfg["block_n"], cfg["block_k"], cfg["threads"], cfg["num_stages"]
        )
        return CompiledProgram(project), CompiledProgram(self.step_jit(cfg["step_threads"]))


@target_cache(maxsize=32)
@tilelang.jit(out_idx=[3])
def _engram_split_project_kernel():
    """Reduce both 512-wide GEMVs over eight independent 64-row panels."""

    @T.prim_func
    def kernel(
        E: T.Tensor((1, 512), "float16"),
        K: T.Tensor((512, 256), "float16"),
        V: T.Tensor((512, 256), "float16"),
        P: T.Tensor((2, 8, 256), "float32"),
    ):
        with T.Kernel(8, 8, 2, threads=32) as (col, part, kind):
            products = T.alloc_fragment((8, 32), "float32")
            sums = T.alloc_fragment((32,), "float32")
            T.clear(products)
            for group in T.unroll(8):
                if kind == 0:
                    for r, c in T.Parallel(8, 32):
                        products[r, c] += T.cast(
                            E[0, part * 64 + group * 8 + r], "float32"
                        ) * T.cast(K[part * 64 + group * 8 + r, col * 32 + c], "float32")
                else:
                    for r, c in T.Parallel(8, 32):
                        products[r, c] += T.cast(
                            E[0, part * 64 + group * 8 + r], "float32"
                        ) * T.cast(V[part * 64 + group * 8 + r, col * 32 + c], "float32")
            T.reduce_sum(products, sums, dim=0)
            for c in T.Parallel(32):
                P[kind, part, col * 32 + c] = sums[c]

    return kernel


@target_cache(maxsize=32)
@tilelang.jit(out_idx=[6, 7])
def _engram_split_step_kernel(history, taps, dilation, eps):
    """Reduce the partial projections while independent CTAs shift the history."""

    @T.prim_func
    def kernel(
        P: T.Tensor((2, 8, 256), "float32"),
        H: T.Tensor((1, 256), "float16"),
        S: T.Tensor((1, history, 256), "float16"),
        WH: T.Tensor((256,), "float16"),
        WV: T.Tensor((256,), "float16"),
        W: T.Tensor((taps, 256), "float16"),
        Y: T.Tensor((1, 256), "float16"),
        N: T.Tensor((1, history, 256), "float16"),
    ):
        with T.Kernel(history, threads=128) as block:
            k = T.alloc_fragment((256,), "float32")
            v = T.alloc_fragment((256,), "float32")
            hh = T.alloc_fragment((256,), "float32")
            hh_sum = T.alloc_fragment((1,), "float32")
            kk = T.alloc_fragment((256,), "float32")
            kk_sum = T.alloc_fragment((1,), "float32")
            hk = T.alloc_fragment((256,), "float32")
            hk_sum = T.alloc_fragment((1,), "float32")
            vv = T.alloc_fragment((256,), "float32")
            vv_sum = T.alloc_fragment((1,), "float32")
            if block == 0:
                T.clear(k)
                T.clear(v)
                for p in T.unroll(8):
                    for c in T.Parallel(256):
                        k[c] += P[0, p, c]
                        v[c] += P[1, p, c]
                for c in T.Parallel(256):
                    h = T.cast(H[0, c], "float32")
                    w = T.cast(WH[c], "float32")
                    hh[c] = h * h
                    kk[c] = k[c] * k[c]
                    hk[c] = h * k[c] * w * w
                    vv[c] = v[c] * v[c]
                T.reduce_sum(hh, hh_sum, dim=0)
                T.reduce_sum(kk, kk_sum, dim=0)
                T.reduce_sum(hk, hk_sum, dim=0)
                T.reduce_sum(vv, vv_sum, dim=0)
                # The gate is sigmoid(<norm(h), norm(k)> / sqrt(256)); 0.0625 is 1 / sqrt(256).
                alpha = 1.0 / (
                    1.0
                    + T.exp(
                        -hk_sum[0]
                        * 0.0625
                        * T.rsqrt(hh_sum[0] / 256.0 + eps)
                        * T.rsqrt(kk_sum[0] / 256.0 + eps)
                    )
                )
                inv = T.rsqrt(alpha * alpha * vv_sum[0] / 256.0 + eps)
                for c in T.Parallel(256):
                    gated = alpha * v[c]
                    norm = gated * inv * T.cast(WV[c], "float32")
                    conv = T.alloc_var("float32", init=T.cast(W[taps - 1, c], "float32") * norm)
                    for tap in T.serial(taps - 1):
                        prev = history - (taps - 1 - tap) * dilation
                        conv += T.cast(W[tap, c], "float32") * T.cast(S[0, prev, c], "float32")
                    Y[0, c] = conv / (1.0 + T.exp(-conv)) + gated
                    N[0, history - 1, c] = norm
            else:
                for c in T.Parallel(256):
                    N[0, block - 1, c] = S[0, block, c]

    return kernel


class EngramDecodeSplitKernel(Kernel, EngramDecodeFwdInterface):
    """One FP16 token with 512 memory features and 256 hidden features."""

    supported_archs = [90]
    preferred_over = frozenset({"engram_decode"})

    @classmethod
    def refusal(cls, call: EngramDecodeCall) -> str | None:
        if (call.batch, call.d_mem, call.d) != (1, 512, 256) or call.dtype != torch.float16:
            return "requires one FP16 token, memory width 512 and hidden width 256"
        if call.max_conv_len < max(1, call.dilation * (call.conv_kernel_size - 1)):
            return "the convolution must have enough history for every tap"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: EngramDecodeCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: EngramDecodeCall):
        super().__init__(device_index=call.device_index)
        self.call = call

    def _compile(self, cfg):
        call = self.call
        return (
            CompiledProgram(_engram_split_project_kernel()),
            CompiledProgram(
                _engram_split_step_kernel(
                    call.max_conv_len, call.conv_kernel_size, call.dilation, call.eps
                )
            ),
        )

    def forward(self, e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w):
        conv_state = _left_pad_history(conv_state, self.call.max_conv_len)
        project, step = self.program(self._compile)
        return step(project(e_t, W_K, W_V), h_t, conv_state, rms_w_h, rms_w_v, conv_w)
