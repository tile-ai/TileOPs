"""
Gated DeltaNet decode (single-step recurrence).

    old_val  = S @ k                     # matvec
    v_new    = beta * v - alpha * beta * old_val
    o_inter  = alpha * (S @ q)           # matvec
    o_intra  = (q . k) * v_new           # dot product + scale
    o        = o_inter + o_intra
    S_new    = alpha * S + outer(k, v_new)

where alpha = exp(g).

Optimization:
  - T.Pipelined + T.copy: async prefetch state tiles from HBM
  - T.gemm: fused matvec via [padded_qk; ...] @ S_tile, using tensor cores
  - Native dtype: bf16/fp16 halve state bandwidth vs fp32
  - K-tiling: small shared memory footprint → high occupancy
"""
import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["GatedDeltaNetDecodeFP32Kernel", "GatedDeltaNetDecodeKernel"]

_LOG2E = 1.4426950408889634
_DEFAULT_K_TILE = 16
# T.gemm requires M divisible by 16; we use rows 0 (k) and 1 (q)
_GEMM_M = 16


@functools.lru_cache(maxsize=32)
def _gated_deltanet_decode_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int = _DEFAULT_K_TILE,
    dtype: str = "float32",
):
    accum_dtype = "float32"
    if dim_k % k_tile != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by k_tile={k_tile}")

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _decode_func(num_stages, threads=128):
        @T.macro
        def _decode_body(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            g: T.Tensor([batch, head], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                # State tile for T.Pipelined + T.copy async prefetch
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                # Padded [_GEMM_M, k_tile] for T.gemm (rows 0=k, 1=q, rest=0)
                qk_tile = T.alloc_shared([_GEMM_M, k_tile], dtype)
                # Gemm accumulator (registers): row 0 = S@k, row 1 = S@q
                acc = T.alloc_fragment([_GEMM_M, dim_v], accum_dtype)
                # Shared buffer to extract gemm result from fragment
                acc_shared = T.alloc_shared([2, dim_v], accum_dtype)
                # Shared buffer for intermediate results
                v_new = T.alloc_shared([dim_v], accum_dtype)
                # Local (register) dot product — avoids shared-memory race
                qk_dot = T.alloc_local([1], accum_dtype)

                # Scalars
                g_val = T.cast(g[bid, hid], accum_dtype)
                beta_val = T.cast(beta[bid, hid], accum_dtype)
                alpha = T.exp2(g_val * _LOG2E)
                alpha_beta = alpha * beta_val

                # Zero-init padding rows of qk_tile (rows 2..15)
                for i, j in T.Parallel(_GEMM_M, k_tile):
                    qk_tile[i, j] = T.cast(T.float32(0.0), dtype)

                # === Pass 1: Tiled pipelined gemm for fused matvec ===
                T.clear(acc)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    # Fill rows 0 (k) and 1 (q) for this tile
                    for i in T.Parallel(k_tile):
                        qk_tile[0, i] = k[bid, hid, kt * k_tile + i]
                        qk_tile[1, i] = q[bid, hid, kt * k_tile + i]
                    T.gemm(qk_tile, h_tile, acc, policy=T.GemmWarpPolicy.FullRow)

                # Copy gemm result from fragment to shared (rows 0-1 only)
                T.copy(acc[:2, :], acc_shared)

                # q . k dot product (must be AFTER T.gemm to avoid
                # corrupting fragment state)
                qk_dot[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    qk_dot[0] += (
                        T.cast(q[bid, hid, kk], accum_dtype)
                        * T.cast(k[bid, hid, kk], accum_dtype)
                    )

                # v_new = beta * v - alpha_beta * (S @ k)
                for j in T.Parallel(dim_v):
                    v_new[j] = (
                        beta_val * T.cast(v[bid, hid, j], accum_dtype)
                        - alpha_beta * acc_shared[0, j]
                    )

                # o = alpha * (S @ q) + (q . k) * v_new
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(
                        alpha * acc_shared[1, j] + qk_dot[0] * v_new[j], dtype
                    )

                # === Pass 2: State update with async prefetch ===
                # new_state = alpha * S + outer(k, v_new)
                # Reuses h_tile for pipelined state prefetch.
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = T.cast(
                            alpha * T.cast(h_tile[kk, j], accum_dtype)
                            + T.cast(k[bid, hid, kt * k_tile + kk], accum_dtype)
                            * v_new[j],
                            dtype,
                        )

        @T.prim_func
        def gated_deltanet_decode(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            g: T.Tensor([batch, head], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            _decode_body(q, k, v, g, beta, state, o, new_state)

        return gated_deltanet_decode

    return _decode_func


@torch.library.custom_op("tileops::gated_deltanet_decode_kernel", mutates_args=())
def _gated_deltanet_decode_wrapped_kernel(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
    dtype: str,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_fn = _gated_deltanet_decode_tl(
        batch, head, dim_k, dim_v, k_tile, dtype
    )(num_stages, threads)
    return kernel_fn(q, k, v, g, beta, state)


@_gated_deltanet_decode_wrapped_kernel.register_fake
def _gated_deltanet_decode_wrapped_kernel_fake(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
    dtype: str,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty(batch, head, dim_v, dtype=q.dtype, device=q.device)
    new_state = torch.empty(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    return o, new_state


class GatedDeltaNetDecodeKernel(Kernel):
    """Gated DeltaNet single-step decode kernel.

    Uses T.Pipelined + T.copy for async state prefetch and T.gemm for
    the fused matvec. Supports float32, float16, and bfloat16.
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        head: int,
        dim_k: int,
        dim_v: int,
        dtype: str = "float32",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

        # k_tile is baked into the JIT function; handle autotune manually
        # if tune is requested, otherwise use default/provided k_tile.
        if tune:
            self._autotune_with_k_tile()
        else:
            self.init_config(config, tune=False)

        # Cache the JIT-compiled kernel to avoid re-creation overhead
        # on every forward call (_gated_deltanet_decode_wrapped_kernel
        # is kept for torch.compile compatibility).
        self._kernel_fn = _gated_deltanet_decode_tl(
            batch, head, dim_k, dim_v,
            self.config["k_tile"], self.dtype_str,
        )(self.config["num_stages"], self.config["threads"])

    def _autotune_with_k_tile(self) -> None:
        """Autotune across k_tile, num_stages, and threads."""
        from tilelang.profiler import do_bench

        best_time = float("inf")
        best_config = self.default_config

        # Generate dummy inputs for profiling
        B, H, DK, DV = self.batch, self.head, self.dim_k, self.dim_v
        torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                       "bfloat16": torch.bfloat16}[self.dtype_str]
        q = torch.randn(B, H, DK, device="cuda", dtype=torch_dtype)
        k = torch.randn(B, H, DK, device="cuda", dtype=torch_dtype)
        v = torch.randn(B, H, DV, device="cuda", dtype=torch_dtype)
        g = -torch.rand(B, H, device="cuda", dtype=torch_dtype)
        beta = torch.rand(B, H, device="cuda", dtype=torch_dtype)
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=torch_dtype)

        print(f"Start autotuning {self.__class__.__name__}...")
        for k_tile in [16, 32, 64]:
            if DK % k_tile != 0:
                continue
            for num_stages in [1, 2, 3]:
                for threads in [128, 256]:
                    try:
                        fn = _gated_deltanet_decode_tl(
                            B, H, DK, DV, k_tile, self.dtype_str,
                        )(num_stages, threads)
                        t = do_bench(lambda _fn=fn: _fn(q, k, v, g, beta, state),
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
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._kernel_fn(q, k, v, g, beta, state)


# ---------------------------------------------------------------------------
# FP32-precision decode kernel (no T.gemm → avoids TF32 mantissa truncation)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _gated_deltanet_decode_fp32_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int = _DEFAULT_K_TILE,
):
    """FP32 decode kernel using element-wise matvec instead of T.gemm.

    T.gemm on fp32 inputs uses TF32 tensor cores which truncate the mantissa
    to 10 bits (~1e-3 error per op).  For multi-step decode the error
    compounds through the recurrent state.  This kernel avoids T.gemm
    entirely, computing S@k and S@q via scalar accumulation in full fp32.

    Pass 1 (matvec) uses T.Serial with fragment accumulators and direct
    global memory reads.  The serial k-dimension loop is intentional:
    it ensures scalar multiply-accumulate (no TF32 tensor cores) at the
    cost of lower throughput — acceptable for single-step decode latency.
    Pass 2 (state update) uses T.Pipelined since each tile writes to
    independent global memory.
    """
    dtype = "float32"
    accum_dtype = "float32"
    if dim_k % k_tile != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by k_tile={k_tile}")

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3"],
    )
    def _decode_func(num_stages, threads=128):

        @T.prim_func
        def gated_deltanet_decode_fp32(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            g: T.Tensor([batch, head], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                # Fragment accumulators for S@k and S@q (full fp32, no TF32)
                sk_frag = T.alloc_fragment([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                v_new = T.alloc_shared([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)

                g_val = T.cast(g[bid, hid], accum_dtype)
                beta_val = T.cast(beta[bid, hid], accum_dtype)
                alpha = T.exp2(g_val * _LOG2E)
                alpha_beta = alpha * beta_val

                # Zero-init fragment accumulators
                T.fill(sk_frag, 0.0)
                T.fill(sq_frag, 0.0)

                # === Pass 1: Element-wise matvec (full fp32 precision) ===
                # T.Serial over dim_k is intentional: scalar multiply-accumulate
                # avoids TF32 tensor cores, trading throughput for precision.
                # Reads state from global memory to avoid shared memory races.
                for kk in T.Serial(dim_k):
                    k_val = k[bid, hid, kk]
                    q_val = q[bid, hid, kk]
                    for j in T.Parallel(dim_v):
                        h_val = state[bid, hid, kk, j]
                        sk_frag[j] = sk_frag[j] + k_val * h_val
                        sq_frag[j] = sq_frag[j] + q_val * h_val

                # q . k dot product
                qk_dot[0] = 0.0
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q[bid, hid, kk] * k[bid, hid, kk]

                # v_new = beta * v - alpha_beta * (S @ k)
                for j in T.Parallel(dim_v):
                    v_new[j] = beta_val * v[bid, hid, j] - alpha_beta * sk_frag[j]

                # o = alpha * (S @ q) + (q . k) * v_new
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = alpha * sq_frag[j] + qk_dot[0] * v_new[j]

                # === Pass 2: State update with async prefetch ===
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = (
                            alpha * h_tile[kk, j]
                            + k[bid, hid, kt * k_tile + kk] * v_new[j]
                        )

        return gated_deltanet_decode_fp32

    return _decode_func


class GatedDeltaNetDecodeFP32Kernel(Kernel):
    """FP32-precision Gated DeltaNet decode kernel (no TF32 tensor cores).

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

        if tune:
            self._autotune_with_k_tile()
        else:
            self.init_config(config, tune=False)

        self._kernel_fn = _gated_deltanet_decode_fp32_tl(
            batch, head, dim_k, dim_v,
            self.config["k_tile"],
        )(self.config["num_stages"], self.config["threads"])

    def _autotune_with_k_tile(self) -> None:
        from tilelang.profiler import do_bench

        best_time = float("inf")
        best_config = self.default_config
        B, H, DK, DV = self.batch, self.head, self.dim_k, self.dim_v

        q = torch.randn(B, H, DK, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, DK, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, DV, device="cuda", dtype=torch.float32)
        g = -torch.rand(B, H, device="cuda", dtype=torch.float32)
        beta = torch.rand(B, H, device="cuda", dtype=torch.float32)
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=torch.float32)

        print(f"Start autotuning {self.__class__.__name__}...")
        for k_tile in [16, 32, 64]:
            if DK % k_tile != 0:
                continue
            for num_stages in [1, 2, 3]:
                for threads in [128, 256]:
                    try:
                        fn = _gated_deltanet_decode_fp32_tl(
                            B, H, DK, DV, k_tile,
                        )(num_stages, threads)
                        t = do_bench(lambda _fn=fn: _fn(q, k, v, g, beta, state),
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
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._kernel_fn(q, k, v, g, beta, state)
