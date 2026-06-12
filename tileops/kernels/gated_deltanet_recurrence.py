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
  - fp32 scalar accumulation for the recurrent matvecs
  - Native dtype: bf16/fp16 halve state bandwidth vs fp32
  - K-tiling: small shared memory footprint → high occupancy
"""
import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = [
    "GatedDeltaNetDecodeFP32Kernel",
    "GatedDeltaNetDecodeKernel",
    "GatedDeltaNetDecodeRawCudaFlaStyleKernel",
]

_LOG2E = 1.4426950408889634
_DEFAULT_K_TILE = 16


@functools.lru_cache(maxsize=32)
def _gated_deltanet_decode_raw_cuda_flastyle_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    v_tile: int = 16,
    raw_group_size: int = 2,
    raw_maxrregcount: int = 146,
    dtype: str = "bfloat16",
):
    if dtype != "bfloat16":
        raise ValueError("Raw CUDA Gated DeltaNet decode currently supports bfloat16 only.")
    if dim_k != 128 or dim_v != 128:
        raise ValueError("Raw CUDA Gated DeltaNet decode currently requires DK=DV=128.")
    if dim_v % v_tile != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by v_tile={v_tile}")
    if raw_group_size * v_tile != 32:
        raise ValueError("raw_group_size * v_tile must equal one warp")
    if dim_k % raw_group_size != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by raw_group_size={raw_group_size}")

    total_blocks = batch * head * (dim_v // v_tile)
    k_chunk = dim_k // raw_group_size

    raw_compile_flags = ["-O3", "-DENABLE_BF16", "--use_fast_math"]
    if raw_maxrregcount > 0:
        raw_compile_flags.append(f"--maxrregcount={raw_maxrregcount}")

    @tilelang.jit(
        out_idx=[-2, -1],
        compile_flags=raw_compile_flags,
    )
    def _decode_func(threads=32):
        @T.prim_func
        def gated_deltanet_decode_raw_cuda_flastyle(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            g: T.Tensor([batch, head], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(total_blocks, threads=threads) as (block,):
                tx = T.get_thread_binding()
                vid = block % (dim_v // v_tile)
                nh = block // (dim_v // v_tile)
                bid = nh // head
                hid = nh - bid * head
                v_lane = tx // raw_group_size
                k_rank = tx - v_lane * raw_group_size
                v_idx = vid * v_tile + v_lane
                k_begin = k_rank * k_chunk

                k_shared = T.alloc_shared([dim_k], "float32")
                q_shared = T.alloc_shared([dim_k], "float32")
                h = T.alloc_local([k_chunk], "float32")
                old_partial = T.alloc_var("float32", init=0.0)
                out_partial = T.alloc_var("float32", init=0.0)
                v_new = T.alloc_var("float32", init=0.0)

                for i in T.serial(T.ceildiv(dim_k, 32)):
                    kk = tx + i * 32
                    if kk < dim_k:
                        k_shared[kk] = T.cast(k[bid, hid, kk], "float32")
                        q_shared[kk] = T.cast(q[bid, hid, kk], "float32")
                T.sync_threads()

                alpha = T.exp2(T.cast(g[bid, hid], "float32") * _LOG2E)
                beta_val = T.cast(beta[bid, hid], "float32")

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    h_val = alpha * T.cast(state[bid, hid, kk, v_idx], "float32")
                    h[ii] = h_val
                    old_partial += h_val * k_shared[kk]

                old_val = old_partial + T.shfl_down(old_partial, 1, width=raw_group_size)
                if k_rank == 0:
                    v_new = beta_val * (T.cast(v[bid, hid, v_idx], "float32") - old_val)
                v_new = T.shfl_sync(v_new, v_lane * raw_group_size, width=raw_group_size)

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    h_new = h[ii] + k_shared[kk] * v_new
                    new_state[bid, hid, kk, v_idx] = T.cast(h_new, dtype)
                    out_partial += h_new * q_shared[kk]

                out_val = out_partial + T.shfl_down(out_partial, 1, width=raw_group_size)
                if k_rank == 0:
                    o[bid, hid, v_idx] = T.cast(out_val, dtype)

        return gated_deltanet_decode_raw_cuda_flastyle

    return _decode_func


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
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                sk_frag = T.alloc_fragment([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                v_new = T.alloc_shared([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)

                g_val = T.cast(g[bid, hid], accum_dtype)
                beta_val = T.cast(beta[bid, hid], accum_dtype)
                alpha = T.exp2(g_val * _LOG2E)
                alpha_beta = alpha * beta_val

                # Full-fp32 matvecs.  TileLang 0.1.9 cannot reliably lower
                # the old tensor-core fragment copy here for fp16/bf16.
                # TODO: restore a tensor-core fast path once fragment copies
                # lower reliably without sacrificing recurrent decode numerics.
                T.fill(sk_frag, 0.0)
                T.fill(sq_frag, 0.0)
                for kk in T.Serial(dim_k):
                    k_val = T.cast(k[bid, hid, kk], accum_dtype)
                    q_val = T.cast(q[bid, hid, kk], accum_dtype)
                    for j in T.Parallel(dim_v):
                        h_val = T.cast(state[bid, hid, kk, j], accum_dtype)
                        sk_frag[j] = sk_frag[j] + k_val * h_val
                        sq_frag[j] = sq_frag[j] + q_val * h_val

                # q . k is a scalar reduction, so keep it separate from the
                # matvec loop whose inner work is parallelized over dim_v.
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
                        - alpha_beta * sk_frag[j]
                    )

                # o = alpha * (S @ q) + (q . k) * v_new
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(
                        alpha * sq_frag[j] + qk_dot[0] * v_new[j], dtype
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

    Uses T.Pipelined + T.copy for async state prefetch and full-fp32
    scalar accumulation for the recurrent matvecs.  The scalar path avoids
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


class GatedDeltaNetDecodeRawCudaFlaStyleKernel(Kernel):
    """Hopper bfloat16 decode kernel for the DK=DV=128 Gated DeltaNet case.

    This path maps one warp to one (batch, head, V tile).  Two lanes cooperate
    on one output value when v_tile=16: each lane owns half of the K dimension,
    K/Q are staged in shared memory once per CTA, and the per-lane state slice
    stays live in fp32 registers.  The implementation is intentionally narrow:
    it is used only for bfloat16 DK=DV=128 decode on sm90+ devices.
    """

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        head: int,
        dim_k: int,
        dim_v: int,
        dtype: str = "bfloat16",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        if tune:
            raise NotImplementedError("Raw CUDA Gated DeltaNet decode does not autotune.")
        self.batch = batch
        self.head = head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.init_config(config, tune=False)
        required_threads = self.config["raw_group_size"] * self.config["v_tile"]
        if self.config["threads"] != required_threads:
            raise ValueError(
                f"threads ({self.config['threads']}) must equal raw_group_size * v_tile "
                f"({required_threads}) for the warp-lane mapping used by this kernel."
            )
        self._kernel_fn = _gated_deltanet_decode_raw_cuda_flastyle_tl(
            batch,
            head,
            dim_k,
            dim_v,
            self.config["v_tile"],
            self.config["raw_group_size"],
            self.config["raw_maxrregcount"],
            self.dtype_str,
        )(self.config["threads"])

    @property
    def default_config(self) -> dict:
        return {
            "threads": 32,
            "v_tile": 16,
            "raw_group_size": 2,
            "raw_maxrregcount": 146,
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
