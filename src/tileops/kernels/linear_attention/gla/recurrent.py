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

from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    GLADecodeCall,
    GLADecodeFwdInterface,
    head_count_refusal,
)

__all__ = ["GLADecodeFP32Kernel", "GLADecodeKernel"]

_DEFAULT_K_TILE = 16


# Low-precision decode kernel — bf16 / fp16, fp32 accumulation


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
        scale = dim_k**-0.5

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
                h_tile_o = T.alloc_shared([k_tile, dim_v], dtype)
                v_shared = T.alloc_shared([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)
                # Preload q, k, gk into shared memory to avoid global access
                # in pipelined loop (fixes TileLang warp-specialization issue)
                q_shared = T.alloc_shared([dim_k], accum_dtype)
                k_shared = T.alloc_shared([dim_k], accum_dtype)
                gk_shared = T.alloc_shared([dim_k], accum_dtype)

                # Preload v into shared (for reuse in output and state update)
                for j in T.Parallel(dim_v):
                    v_shared[j] = T.cast(v[bid, hid, j], accum_dtype)

                # Preload q, k and gk into shared memory
                for i in T.Parallel(dim_k):
                    q_shared[i] = T.cast(q[bid, hid, i], accum_dtype)
                    k_shared[i] = T.cast(k[bid, hid, i], accum_dtype)
                    gk_shared[i] = T.cast(gk[bid, hid, i], accum_dtype)

                # Full-fp32 matvec.  TileLang 0.1.9 cannot reliably lower the
                # old tensor-core fragment copy here for fp16/bf16, and TF32
                # style matvec precision is too loose for recurrent decode.
                # TODO: restore a tensor-core fast path once fragment copies
                # lower reliably without sacrificing recurrent decode numerics.
                T.fill(sq_frag, 0.0)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile_o)
                    for kk in T.Serial(k_tile):
                        gk_val = gk_shared[kt * k_tile + kk]
                        alpha_i = T.exp2(gk_val * LOG2E)
                        q_gated = q_shared[kt * k_tile + kk] * alpha_i
                        for j in T.Parallel(dim_v):
                            sq_frag[j] = sq_frag[j] + q_gated * T.cast(h_tile_o[kk, j], accum_dtype)

                # Keep this scalar reduction separate from the matvec's
                # dim_v-parallel inner loop.
                qk_dot[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q_shared[kk] * k_shared[kk]

                # o = scale * (S @ q_gated) + scale * (q . k) * v
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(
                        scale * sq_frag[j] + scale * qk_dot[0] * v_shared[j],
                        dtype,
                    )

                # === Pass 2: State update with async prefetch ===
                # new_state[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
                # NOTE: No intermediate variables inside T.Parallel to avoid BindNode issue
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = T.cast(
                            T.exp2(gk_shared[kt * k_tile + kk] * LOG2E)
                            * T.cast(h_tile[kk, j], accum_dtype)
                            + k_shared[kt * k_tile + kk] * v_shared[j],
                            dtype,
                        )

        return gla_decode

    return _decode_func


def _decode_entry(cls: type, call: GLADecodeCall) -> Entry:
    """The entry for a GLA decode kernel: both take the same construction arguments.

    The device index is in the identity because the kernel is compiled for the
    architecture it is built on.
    """
    index = call.device.index if call.device is not None else None
    dtype = Kernel.dtype_to_str(call.dtype)
    identity = (call.batch, call.heads, call.dim_k, call.dim_v, call.scale, dtype, index)
    return identity, lambda: cls(
        call.batch, call.heads, call.dim_k, call.dim_v, scale=call.scale, dtype=dtype
    )


class GLADecodeKernel(Kernel, GLADecodeFwdInterface):
    """GLA single-step decode kernel for low-precision inputs.

    Uses T.Pipelined + T.copy for async state prefetch and full-fp32
    scalar accumulation for the recurrent matvec.  The scalar path avoids
    TileLang 0.1.9 fragment-copy lowering failures on fp16/bf16 and keeps
    decode numerics aligned with the fp32 reference.
    """

    supported_archs: list[int] = [80, 89, 90]
    general = True

    @classmethod
    def applies(cls, call: GLADecodeCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: GLADecodeCall) -> Optional[str]:
        return head_count_refusal(call.heads)

    @classmethod
    def entry_for(cls, call: GLADecodeCall) -> Entry:
        return _decode_entry(cls, call)

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
        self.scale = scale if scale > 0 else dim_k**-0.5
        self.dtype = dtype

        self.init_config(config, tune=False)
        self._build_program()
        if tune:
            self.autotune()

    def _build_program(self) -> None:
        """Compile the decode program the current config states."""
        self._kernel_fn = _gla_decode_tl(
            self.batch,
            self.head,
            self.dim_k,
            self.dim_v,
            self.config["k_tile"],
            self.dtype_str,
            self.scale,
        )(self.config["num_stages"], self.config["threads"])

    def autotune(self, warmup: int = 10, rep: int = 20) -> None:
        """Sweep k_tile, num_stages and threads, then rebuild the program."""
        from tilelang.profiler import do_bench

        best_time = float("inf")
        best_config = self.default_config

        B, H, DK, DV = self.batch, self.head, self.dim_k, self.dim_v
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype_str]
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
                            B,
                            H,
                            DK,
                            DV,
                            k_tile,
                            self.dtype_str,
                            self.scale,
                        )(num_stages, threads)
                        t = do_bench(lambda _fn=fn: _fn(q, k, v, gk, state), warmup=warmup, rep=rep)
                        if t < best_time:
                            best_time = t
                            best_config = {
                                "num_stages": num_stages,
                                "threads": threads,
                                "k_tile": k_tile,
                            }
                    except Exception:
                        continue

        self.config = best_config
        print(f"Best config: {self.config}")
        self._build_program()

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


# FP32-precision decode kernel (no T.gemm → avoids TF32 mantissa truncation)


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
        scale = dim_k**-0.5

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
                # Preload q, k, v, gk into shared memory to avoid global access
                # in pipelined loop (fixes TileLang warp-specialization issue)
                q_shared = T.alloc_shared([dim_k], dtype)
                k_shared = T.alloc_shared([dim_k], dtype)
                v_shared = T.alloc_shared([dim_v], dtype)
                gk_shared = T.alloc_shared([dim_k], dtype)

                # Preload tensors into shared memory
                for i in T.Parallel(dim_k):
                    q_shared[i] = q[bid, hid, i]
                    k_shared[i] = k[bid, hid, i]
                    gk_shared[i] = gk[bid, hid, i]
                for i in T.Parallel(dim_v):
                    v_shared[i] = v[bid, hid, i]

                T.fill(sq_frag, 0.0)

                # === Pass 1: Element-wise matvec (full fp32 precision) ===
                # S @ q_gated where q_gated = q * exp(gk)
                for kk in T.Serial(dim_k):
                    gk_val = gk_shared[kk]
                    alpha_i = T.exp2(gk_val * LOG2E)
                    q_gated = q_shared[kk] * alpha_i
                    for j in T.Parallel(dim_v):
                        sq_frag[j] = sq_frag[j] + q_gated * state[bid, hid, kk, j]

                qk_dot[0] = 0.0
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q_shared[kk] * k_shared[kk]

                # o = scale * (S @ q_gated) + scale * (q . k) * v
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = scale * sq_frag[j] + scale * qk_dot[0] * v_shared[j]

                # === Pass 2: State update with async prefetch ===
                # new_state[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
                # NOTE: No intermediate variables inside T.Parallel to avoid BindNode issue
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = (
                            T.exp2(gk_shared[kt * k_tile + kk] * LOG2E) * h_tile[kk, j]
                            + k_shared[kt * k_tile + kk] * v_shared[j]
                        )

        return gla_decode_fp32

    return _decode_func


class GLADecodeFP32Kernel(Kernel, GLADecodeFwdInterface):
    """FP32-precision GLA decode kernel (no TF32 tensor cores).

    Uses element-wise matvec instead of T.gemm to avoid TF32 mantissa
    truncation that causes ~1e-3 error per step, compounding over multi-step
    decode.  Intended for fp32 dtype only.
    """

    supported_archs: list[int] = [80, 89, 90]

    @classmethod
    def applies(cls, call: GLADecodeCall) -> bool:
        return call.dtype == torch.float32 and head_count_refusal(call.heads) is None

    @classmethod
    def entry_for(cls, call: GLADecodeCall) -> Entry:
        return _decode_entry(cls, call)

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
        self.scale = scale if scale > 0 else dim_k**-0.5

        self.init_config(config, tune=False)
        self._build_program()
        if tune:
            self.autotune()

    def _build_program(self) -> None:
        """Compile the fp32 decode program the current config states."""
        self._kernel_fn = _gla_decode_fp32_tl(
            self.batch,
            self.head,
            self.dim_k,
            self.dim_v,
            self.config["k_tile"],
            self.scale,
        )(self.config["num_stages"], self.config["threads"])

    def autotune(self, warmup: int = 10, rep: int = 20) -> None:
        """Sweep k_tile, num_stages and threads, then rebuild the program."""
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
                            B,
                            H,
                            DK,
                            DV,
                            k_tile,
                            self.scale,
                        )(num_stages, threads)
                        t = do_bench(lambda _fn=fn: _fn(q, k, v, gk, state), warmup=warmup, rep=rep)
                        if t < best_time:
                            best_time = t
                            best_config = {
                                "num_stages": num_stages,
                                "threads": threads,
                                "k_tile": k_tile,
                            }
                    except Exception:
                        continue

        self.config = best_config
        print(f"Best config: {self.config}")
        self._build_program()

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
