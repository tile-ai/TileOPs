"""
GLA (Gated Linear Attention) decode (single-step recurrence).

    S_new = diag(exp(gk)) @ S + outer(k, v)
    o     = scale * q^T @ S_new
          = scale * (q * exp(gk))^T @ S + scale * (q . k) * v

where gk is per-key-dimension log-space gate [B, H, DK].

Optimization:
  - T.Pipelined + T.copy: async prefetch state tiles from HBM
  - fp32 scalar accumulation for the recurrent matvec
  - Native dtype: bf16/fp16 halve state bandwidth vs fp32
  - K-tiling: small shared memory footprint → high occupancy
"""
import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["GLADecodeFP32Kernel", "GLADecodeKernel"]

_LOG2E = 1.4426950408889634
_DEFAULT_K_TILE = 16


# =============================================================================
# Low-precision decode kernel — bf16 / fp16, fp32 accumulation
# =============================================================================

@functools.lru_cache(maxsize=32)
def _gla_decode_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int = _DEFAULT_K_TILE,
    dtype: str = "float32",
    scale: float = -1.0,
):
    accum_dtype = "float32"
    if dim_k % k_tile != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by k_tile={k_tile}")

    if scale <= 0:
        scale = dim_k ** -0.5

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _decode_func(num_stages, threads=128):

        @T.prim_func
        def gla_decode(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            gk: T.Tensor([batch, head, dim_k], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                v_shared = T.alloc_shared([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)

                # Preload v into shared (for reuse in output and state update)
                for j in T.Parallel(dim_v):
                    v_shared[j] = T.cast(v[bid, hid, j], accum_dtype)

                # Full-fp32 matvec.  TileLang 0.1.9 cannot reliably lower the
                # old tensor-core fragment copy here for fp16/bf16, and TF32
                # style matvec precision is too loose for recurrent decode.
                # TODO: restore a tensor-core fast path once fragment copies
                # lower reliably without sacrificing recurrent decode numerics.
                T.fill(sq_frag, 0.0)
                for kk in T.Serial(dim_k):
                    gk_val = T.cast(gk[bid, hid, kk], accum_dtype)
                    alpha_i = T.exp2(gk_val * _LOG2E)
                    q_gated = T.cast(q[bid, hid, kk], accum_dtype) * alpha_i
                    for j in T.Parallel(dim_v):
                        sq_frag[j] = (
                            sq_frag[j]
                            + q_gated * T.cast(state[bid, hid, kk, j], accum_dtype)
                        )

                # Keep this scalar reduction separate from the matvec's
                # dim_v-parallel inner loop.
                qk_dot[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    qk_dot[0] += (
                        T.cast(q[bid, hid, kk], accum_dtype)
                        * T.cast(k[bid, hid, kk], accum_dtype)
                    )

                # o = scale * (S @ q_gated) + scale * (q . k) * v
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(
                        scale * sq_frag[j] + scale * qk_dot[0] * v_shared[j],
                        dtype,
                    )

                # === Pass 2: State update with async prefetch ===
                # new_state[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        gk_val = T.cast(gk[bid, hid, kt * k_tile + kk], accum_dtype)
                        alpha_i = T.exp2(gk_val * _LOG2E)
                        new_state[bid, hid, kt * k_tile + kk, j] = T.cast(
                            alpha_i * T.cast(h_tile[kk, j], accum_dtype)
                            + T.cast(k[bid, hid, kt * k_tile + kk], accum_dtype)
                            * v_shared[j],
                            dtype,
                        )

        return gla_decode

    return _decode_func


@torch.library.custom_op("tileops::gla_decode_kernel", mutates_args=())
def _gla_decode_wrapped_kernel(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
    dtype: str,
    scale: float,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_fn = _gla_decode_tl(
        batch, head, dim_k, dim_v, k_tile, dtype, scale,
    )(num_stages, threads)
    return kernel_fn(q, k, v, gk, state)


@_gla_decode_wrapped_kernel.register_fake
def _gla_decode_wrapped_kernel_fake(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
    dtype: str,
    scale: float,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty(batch, head, dim_v, dtype=q.dtype, device=q.device)
    new_state = torch.empty(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    return o, new_state


class GLADecodeKernel(Kernel):
    """GLA single-step decode kernel for low-precision inputs.

    Uses T.Pipelined + T.copy for async state prefetch and full-fp32
    scalar accumulation for the recurrent matvec.  The scalar path avoids
    TileLang 0.1.9 fragment-copy lowering failures on fp16/bf16 and keeps
    decode numerics aligned with the fp32 reference.
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        head: int,
        dim_k: int,
        dim_v: int,
        scale: float = -1.0,
        dtype: str = "float32",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale if scale > 0 else dim_k ** -0.5
        self.dtype = dtype

        if tune:
            self._autotune_with_k_tile()
        else:
            self.init_config(config, tune=False)

        # Cache the JIT-compiled kernel to avoid re-creation overhead
        # on every forward call (_gla_decode_wrapped_kernel is kept
        # for torch.compile compatibility).
        self._kernel_fn = _gla_decode_tl(
            batch, head, dim_k, dim_v,
            self.config["k_tile"], self.dtype_str, self.scale,
        )(self.config["num_stages"], self.config["threads"])

    def _autotune_with_k_tile(self) -> None:
        """Autotune across k_tile, num_stages, and threads."""
        from tilelang.profiler import do_bench

        best_time = float("inf")
        best_config = self.default_config

        B, H, DK, DV = self.batch, self.head, self.dim_k, self.dim_v
        torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                       "bfloat16": torch.bfloat16}[self.dtype_str]
        q = torch.randn(B, H, DK, device="cuda", dtype=torch_dtype)
        k = torch.randn(B, H, DK, device="cuda", dtype=torch_dtype)
        v = torch.randn(B, H, DV, device="cuda", dtype=torch_dtype)
        gk = -torch.rand(B, H, DK, device="cuda", dtype=torch_dtype)
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=torch_dtype)

        print(f"Start autotuning {self.__class__.__name__}...")
        for k_tile in [16, 32, 64]:
            if DK % k_tile != 0:
                continue
            for num_stages in [1, 2, 3]:
                for threads in [128, 256]:
                    try:
                        fn = _gla_decode_tl(
                            B, H, DK, DV, k_tile, self.dtype_str, self.scale,
                        )(num_stages, threads)
                        t = do_bench(lambda _fn=fn: _fn(q, k, v, gk, state),
                                     warmup=10, rep=20)
                        if t < best_time:
                            best_time = t
                            best_config = {"num_stages": num_stages,
                                           "threads": threads, "k_tile": k_tile}
                    except Exception:
                        continue

        self.config = best_config
        print(f"{self.__class__.__name__} initialized with config: {self.config}")

    @property
    def default_config(self) -> dict:
        return {
            "num_stages": 2,
            "threads": 128,
            "k_tile": _DEFAULT_K_TILE,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._kernel_fn(q, k, v, gk, state)


# =============================================================================
# FP32-precision decode kernel (no T.gemm → avoids TF32 mantissa truncation)
# =============================================================================

@functools.lru_cache(maxsize=32)
def _gla_decode_fp32_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int = _DEFAULT_K_TILE,
    scale: float = -1.0,
):
    """FP32 decode kernel using element-wise matvec instead of T.gemm.

    T.gemm on fp32 inputs uses TF32 tensor cores which truncate the mantissa
    to 10 bits (~1e-3 error per op).  For multi-step decode the error
    compounds through the recurrent state.  This kernel avoids T.gemm
    entirely, computing S@q_gated via scalar accumulation in full fp32.
    """
    dtype = "float32"
    accum_dtype = "float32"
    if dim_k % k_tile != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by k_tile={k_tile}")

    if scale <= 0:
        scale = dim_k ** -0.5

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3"],
    )
    def _decode_func(num_stages, threads=128):

        @T.prim_func
        def gla_decode_fp32(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            gk: T.Tensor([batch, head, dim_k], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                # Fragment accumulator for S @ q_gated (full fp32, no TF32)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)

                # Zero-init fragment accumulator
                T.fill(sq_frag, 0.0)

                # === Pass 1: Element-wise matvec (full fp32 precision) ===
                # S @ q_gated where q_gated = q * exp(gk)
                for kk in T.Serial(dim_k):
                    gk_val = gk[bid, hid, kk]
                    alpha_i = T.exp2(gk_val * _LOG2E)
                    q_gated = q[bid, hid, kk] * alpha_i
                    for j in T.Parallel(dim_v):
                        sq_frag[j] = sq_frag[j] + q_gated * state[bid, hid, kk, j]

                # q . k dot product
                qk_dot[0] = 0.0
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q[bid, hid, kk] * k[bid, hid, kk]

                # o = scale * (S @ q_gated) + scale * (q . k) * v
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = (
                        scale * sq_frag[j]
                        + scale * qk_dot[0] * v[bid, hid, j]
                    )

                # === Pass 2: State update with async prefetch ===
                # new_state[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        gk_val = gk[bid, hid, kt * k_tile + kk]
                        alpha_i = T.exp2(gk_val * _LOG2E)
                        new_state[bid, hid, kt * k_tile + kk, j] = (
                            alpha_i * h_tile[kk, j]
                            + k[bid, hid, kt * k_tile + kk] * v[bid, hid, j]
                        )

        return gla_decode_fp32

    return _decode_func


@torch.library.custom_op("tileops::gla_decode_fp32_kernel", mutates_args=())
def _gla_decode_fp32_wrapped_kernel(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
    scale: float,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_fn = _gla_decode_fp32_tl(
        batch, head, dim_k, dim_v, k_tile, scale,
    )(num_stages, threads)
    return kernel_fn(q, k, v, gk, state)


@_gla_decode_fp32_wrapped_kernel.register_fake
def _gla_decode_fp32_wrapped_kernel_fake(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
    scale: float,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty(batch, head, dim_v, dtype=q.dtype, device=q.device)
    new_state = torch.empty(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    return o, new_state


class GLADecodeFP32Kernel(Kernel):
    """FP32-precision GLA decode kernel (no TF32 tensor cores).

    Uses element-wise matvec instead of T.gemm to avoid TF32 mantissa
    truncation that causes ~1e-3 error per step, compounding over multi-step
    decode.  Intended for fp32 dtype only.
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        head: int,
        dim_k: int,
        dim_v: int,
        scale: float = -1.0,
        dtype: str = "float32",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        if dtype != "float32":
            raise ValueError(f"{self.__class__.__name__} only supports float32")
        self.batch = batch
        self.head = head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale if scale > 0 else dim_k ** -0.5

        if tune:
            self._autotune_with_k_tile()
        else:
            self.init_config(config, tune=False)

        # Cache the JIT-compiled kernel
        self._kernel_fn = _gla_decode_fp32_tl(
            batch, head, dim_k, dim_v,
            self.config["k_tile"], self.scale,
        )(self.config["num_stages"], self.config["threads"])

    def _autotune_with_k_tile(self) -> None:
        from tilelang.profiler import do_bench

        best_time = float("inf")
        best_config = self.default_config
        B, H, DK, DV = self.batch, self.head, self.dim_k, self.dim_v

        q = torch.randn(B, H, DK, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, DK, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, DV, device="cuda", dtype=torch.float32)
        gk = -torch.rand(B, H, DK, device="cuda", dtype=torch.float32)
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=torch.float32)

        print(f"Start autotuning {self.__class__.__name__}...")
        for k_tile in [16, 32, 64]:
            if DK % k_tile != 0:
                continue
            for num_stages in [1, 2, 3]:
                for threads in [128, 256]:
                    try:
                        fn = _gla_decode_fp32_tl(
                            B, H, DK, DV, k_tile, self.scale,
                        )(num_stages, threads)
                        t = do_bench(lambda _fn=fn: _fn(q, k, v, gk, state),
                                     warmup=10, rep=20)
                        if t < best_time:
                            best_time = t
                            best_config = {"num_stages": num_stages,
                                           "threads": threads, "k_tile": k_tile}
                    except Exception:
                        continue

        self.config = best_config
        print(f"{self.__class__.__name__} initialized with config: {self.config}")

    @property
    def default_config(self) -> dict:
        return {
            "num_stages": 2,
            "threads": 128,
            "k_tile": _DEFAULT_K_TILE,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._kernel_fn(q, k, v, gk, state)
