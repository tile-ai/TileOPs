"""MoE-specific 3WG persistent grouped-GEMM kernel with fused activation.

V2 persistent grouped-GEMM kernel with activation fusion in the epilogue.
Based on GroupedGemmPersistent3WGKernel but specialized for MoE gate_up GEMM:
  - Input: A[numel, K] × B[E, 2N, K]^T → gate_up[numel, 2N]
  - Fused epilogue: gate_up → activation(gate[:, :N]) * up[:, N:] → C[numel, N]
  - Supported activations: silu_and_mul, gelu_and_mul

Two compile-time templates routed by block_m:
  * block_m ≤ 64  → Pingpong: 2 math WGs process independent tiles
  * block_m ≥ 128 → Cooperative: 2 math WGs share one tile's M

K-aligned only. Uses static-wave scheduler (no atomic counter).

Performance benefit: eliminates one kernel launch and one global memory round-trip
(gate_up write + activation read) compared to separate GEMM + activation kernels.
"""
import functools
import math
import os

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

_ANCHOR_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../grouped_gemm/_anchor_helper.h")
)

__all__ = ["MoeGroupedGemmPersistent3WGFusedActKernel"]

_DEFAULT_CONFIG = {
    "block_m": 128,        # cooperative template (split-A, shared-B)
    "block_n": 256,
    "block_k": 64,
    "num_stages": 3,
    "threads": 384,
    "group_size_m": 1,
}


class MoeGroupedGemmPersistent3WGFusedActKernel(Kernel):
    """V2 persistent grouped-GEMM kernel with fused activation (K-aligned only).

    Computes A[numel, K] × B[E, 2N, K]^T → C[numel, N] with activation fusion:
      gate_up = A @ B^T  (shape [numel, 2N])
      C = activation(gate_up[:, :N]) * gate_up[:, N:]

    Args:
        numel: T * top_k, tight row count.
        num_experts: Total number of experts E.
        N: Output feature dimension (ffn_size, NOT 2*ffn_size).
        K: Input feature dimension (hidden_size).
        dtype: Activation and weight dtype.
        activation: Activation function name ("silu_and_mul" or "gelu_and_mul").
        sm_count: Number of SMs (auto-detected if None).
        config: Optional config dict with block_m/n/k, num_stages, threads.
        tune: Whether to autotune.

    Example:
        >>> kernel = MoeGroupedGemmPersistent3WGFusedActKernel(
        ...     numel=16384, num_experts=256, N=4096, K=2048,
        ...     dtype=torch.bfloat16, activation="silu_and_mul")
        >>> C = kernel(A, B, true_sizes, true_offsets)  # [numel, N]
    """

    supported_archs: list[int] = [90]

    def __init__(
        self,
        numel: int,
        num_experts: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.bfloat16,
        activation: str = "silu_and_mul",
        sm_count: int = None,
        config=None,
        tune: bool = False,
    ):
        super().__init__()
        if activation not in ("silu_and_mul", "gelu_and_mul"):
            raise ValueError(
                f"activation must be 'silu_and_mul' or 'gelu_and_mul', got {activation!r}"
            )
        self.numel = numel
        self.num_experts = num_experts
        self.N = N
        self.K = K
        self.dtype = dtype
        self.activation = activation
        if sm_count is None:
            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        self.sm_count = sm_count
        self.kernel = lambda: _persistent_grouped_gemm_v2_fused_act_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.activation, self.sm_count, _DEFAULT_CONFIG["block_k"])
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Override base autotune: the JIT factory already accepts config
        params (block_m, block_n, ...) in its signature, so we call it
        directly with each config and benchmark."""
        from tilelang.profiler import do_bench as _do_bench

        print(f"Start autotuning {self.__class__.__name__}...")
        best_ms = float("inf")
        best_cfg = None
        # Build dummy inputs once for benchmarking all configs
        A_dummy = torch.randn(
            self.numel, self.K, dtype=self.dtype, device="cuda") * 0.02
        B_dummy = torch.randn(
            self.num_experts, self.N * 2, self.K, dtype=self.dtype, device="cuda") * 0.02
        # Distribute tokens evenly across experts to avoid negative sizes
        per_expert = self.numel // self.num_experts
        remainder = self.numel % self.num_experts
        sizes = torch.full(
            (self.num_experts,), per_expert, dtype=torch.int32, device="cuda")
        sizes[:remainder] += 1
        offsets = torch.zeros(
            self.num_experts, dtype=torch.int32, device="cuda")
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)

        def _bench_forward():
            return self.forward(A_dummy, B_dummy, sizes, offsets)

        for cfg in self.autotune_configs:
            try:
                self.config = cfg
                _bench_forward()
                ms = _do_bench(_bench_forward, warmup=warmup, rep=rep)
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = cfg
            except Exception:
                continue
        if best_cfg is not None:
            self.config = best_cfg
            print(f"Best config: {best_cfg} ({best_ms:.3f} ms)")
        else:
            self.config = self.default_config
            print("Autotune failed for all configs, using default.")

    @property
    def autotune_configs(self) -> list[dict]:
        SMEM_LIMIT = 228 * 1024
        bytes_per_elem = 2  # bf16/fp16
        configs = []
        # N-dimension tiling is not supported: the epilogue assumes block_n
        # spans the full gate_up width (2N), so block_n is bound to N*2 rather
        # than swept. Any other value would index past C_local_wg{0,1}.
        block_n = self.N * 2
        # ── Pingpong template (bm <= 64): 4 SMEM streams (A_wg0/1, B_wg0/1) ──
        # Note: With fused activation, we also need C_shared for epilogue
        for block_m in (64,):
            for block_k in (64,):
                # Reduce num_stages for pingpong to fit in SMEM
                # With block_n=2N, we need fewer stages
                for num_stages in (2, 3):
                    # 2 × A_smem_wgX (ns, bm, bk) + 2 × B_smem_wgX (ns, bn, bk).
                    smem_main = (
                        2 * num_stages * block_m * block_k * bytes_per_elem
                        + 2 * num_stages * block_n * block_k * bytes_per_elem
                    )
                    # Per-WG C_shared (bm, N) for TMA-store epilogue.
                    smem_c = 2 * block_m * self.N * bytes_per_elem
                    if smem_main + smem_c > SMEM_LIMIT:
                        continue
                    configs.append({
                        "block_m": block_m, "block_n": block_n,
                        "block_k": block_k, "num_stages": num_stages,
                        "threads": 384, "group_size_m": 1,
                    })
        # ── Cooperative template (bm >= 128): split-A, shared B ──
        for block_m in (128,):
            half_m = block_m // 2
            for block_k in (64,):
                for num_stages in (2, 3, 4, 5, 6):
                    # 2 × A_smem_{top,bot} (ns, half_m, bk) + 1 × B_smem.
                    smem_main = (
                        2 * num_stages * half_m * block_k * bytes_per_elem
                        + num_stages * block_n * block_k * bytes_per_elem
                    )
                    # Per-WG C_shared (half_m, bn) for TMA-store epilogue.
                    smem_c = 2 * half_m * block_n * bytes_per_elem
                    if smem_main + smem_c > SMEM_LIMIT:
                        continue
                    configs.append({
                        "block_m": block_m, "block_n": block_n,
                        "block_k": block_k, "num_stages": num_stages,
                        "threads": 384, "group_size_m": 1,
                    })
        return configs

    def forward(self, A, B, true_sizes, true_offsets):
        """Run tile scheduler + GEMM with fused activation epilogue.

        Args:
            A: [numel, K] tight permuted hidden states.
            B: [num_experts, 2N, K] expert weight matrix (gate_up concatenated).
            true_sizes: [E] int32 true token count per expert.
            true_offsets: [E] int32 tight start offset per expert in A.

        Returns:
            C: [numel, N] GEMM output with fused activation.
        """
        # Reuse C buffer to avoid repeated allocations
        if getattr(self, "_C_buffer", None) is None or self._C_buffer.device != A.device:
            self._C_buffer = torch.zeros(self.numel, self.N, dtype=self.dtype, device=A.device)
        C = self._C_buffer

        block_m = self.config["block_m"]
        # N-dimension tiling is not supported: the epilogue assumes block_n
        # spans the full gate_up width (2N). Bind it here rather than trusting
        # self.config["block_n"], so a stale autotune/default config can never
        # drive an out-of-bounds access on C_local_wg{0,1}.
        block_n = self.N * 2
        block_k = self.config["block_k"]
        if self.K % block_k != 0:
            raise ValueError(f"K-aligned only: K={self.K}, block_k={block_k}")

        # TMA limitation: block_n <= 256
        # Since B is [E, 2N, K], we need block_n to cover gate_up width
        # For now, require 2N <= 256 (i.e., N <= 128)
        if self.N * 2 > 256:
            raise NotImplementedError(
                f"Current implementation requires N <= 128 due to TMA boxDim limitation. "
                f"Got N={self.N} (2N={self.N * 2}). "
                f"TODO: Implement N-dimension tiling for larger ffn_size."
            )

        required_rows = self.numel + block_m
        if A.shape[0] < required_rows:
            # Reuse A padding buffer to avoid repeated allocations
            if (getattr(self, "_A_padded", None) is None or
                self._A_padded.shape[0] < required_rows or
                self._A_padded.device != A.device):
                self._A_padded = torch.zeros(required_rows, self.K, dtype=self.dtype, device=A.device)
            self._A_padded[:A.shape[0]].copy_(A)
            A = self._A_padded

        gemm_fn = _persistent_grouped_gemm_v2_fused_act_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.activation, self.sm_count, block_k,
        )(
            block_m, block_n, block_k,
            self.config["num_stages"],
            self.config["threads"],
            self.config.get("group_size_m", 1),
        )
        gemm_fn(A, B, true_sizes, true_offsets, C)
        return C


@functools.lru_cache(maxsize=64)
def _persistent_grouped_gemm_v2_fused_act_kernel(numel, num_experts, N, K, dtype,
                                                  activation, sm_count, block_k):
    """V2 persistent grouped-GEMM JIT factory with fused activation.

    Dispatches between two prim_func templates by block_m:
      * block_m <= 64  → pingpong (2 math WGs work independent tiles)
      * block_m >= 128 → cooperative (2 math WGs split one tile's M via split-A; B is shared)
    """
    if K % block_k != 0:
        raise ValueError(
            f"MoeGroupedGemmPersistent3WGFusedActKernel requires K-alignment "
            f"(K={K} must be divisible by block_k={block_k})."
        )

    # Dynamically set compile flags based on dtype
    compile_flags = ["-O3", "-include", _ANCHOR_HELPER_PATH]
    if dtype == "bfloat16":
        compile_flags.append("-DENABLE_BF16")
    elif dtype == "float16":
        compile_flags.append("-DENABLE_FP16")

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=compile_flags,
    )
    def _func(block_m, block_n, block_k, num_stages, threads, group_size_m):
        if block_m <= 64:
            return _make_pingpong_fused_act_kernel(
                numel, num_experts, N, K, dtype, activation, sm_count,
                block_m, block_n, block_k, num_stages, threads, group_size_m,
            )
        else:
            return _make_cooperative_fused_act_kernel(
                numel, num_experts, N, K, dtype, activation, sm_count,
                block_m, block_n, block_k, num_stages, threads, group_size_m,
            )

    return _func


def _make_pingpong_fused_act_kernel(numel, num_experts, N, K, dtype, activation, sm_count,
                                     block_m, block_n, block_k, num_stages, threads,
                                     group_size_m):
    """Build a @T.prim_func for the pingpong template with fused activation.

    Two math WGs each work an independent tile per wave; each WG holds
    its own SMEM ring (A_smem_wgX, B_smem_wgX of shape (num_stages, ...)).

    Key difference from base 3WG: epilogue computes activation fusion:
      - GEMM produces gate_up [block_m, block_n] in C_local (block_n = 2N)
      - Epilogue splits into gate[:, :N] and up[:, N:], applies activation
      - Output C [numel, N] (half the width of gate_up)
    """
    accum_dtype = "float"
    log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    assert threads == 384, (
        f"V2 pingpong persistent grouped GEMM requires threads=384 "
        f"(1 producer + 2 consumer WGs); got threads={threads}"
    )
    _num_pid_n = math.ceil(N / block_n)  # N-tiles in output (not 2N)
    _max_tiles = numel // block_m + num_experts
    _total_ctas_ub = _max_tiles * _num_pid_n
    _max_waves = (_total_ctas_ub + 2 * sm_count - 1) // (2 * sm_count) + 1
    _k_iters = K // block_k
    A_shape = (numel + block_m, K)
    # B shape: [num_experts, 2N, K] (gate_up concatenated)
    B_shape = (num_experts, N * 2, K)

    @T.prim_func
    def _gemm_main_v2_pingpong_fused_act(
        A: T.Tensor(A_shape, dtype),                        # type: ignore  # noqa: F821
        B: T.Tensor(B_shape, dtype),                        # type: ignore  # noqa: F821
        true_sizes: T.Tensor((num_experts,), "int32"),      # noqa: F821
        true_offsets: T.Tensor((num_experts,), "int32"),    # noqa: F821
        C: T.Tensor((numel, N), dtype),                     # type: ignore  # noqa: F821
    ):
        with T.Kernel(sm_count, threads=threads) as (pid,):
            # ── Per-WG ring-buffered SMEM (num_stages slots) ──
            # B SMEM holds full 2N width (gate_up)
            A_smem_wg0 = T.alloc_shared((num_stages, block_m, block_k), dtype)
            B_smem_wg0 = T.alloc_shared((num_stages, block_n, block_k), dtype)
            A_smem_wg1 = T.alloc_shared((num_stages, block_m, block_k), dtype)
            B_smem_wg1 = T.alloc_shared((num_stages, block_n, block_k), dtype)
            # C_local accumulates gate_up [block_m, block_n] where block_n covers 2N
            C_local_wg0 = T.alloc_fragment((block_m, block_n), accum_dtype)
            C_local_wg1 = T.alloc_fragment((block_m, block_n), accum_dtype)

            # ── Phase-4 epilogue: per-WG cast fragment + TMA-store SMEM ──
            # Output is [block_m, N] after activation fusion (half width)
            C_local_cast_wg0 = T.alloc_fragment((block_m, N), dtype)
            C_local_cast_wg1 = T.alloc_fragment((block_m, N), dtype)
            C_shared_wg0 = T.alloc_shared((block_m, N), dtype)
            C_shared_wg1 = T.alloc_shared((block_m, N), dtype)

            # ── Scheduler SMEM ──
            s_cum = T.alloc_shared((num_experts + 1,), "int32")
            s_total = T.alloc_shared((1,), "int32")
            lo = T.alloc_local((1,), "int32")
            hi = T.alloc_local((1,), "int32")
            ms0 = T.alloc_local((1,), "int32")
            ns0 = T.alloc_local((1,), "int32")
            ex0 = T.alloc_local((1,), "int32")
            v0 = T.alloc_local((1,), "int32")
            ms1 = T.alloc_local((1,), "int32")
            ns1 = T.alloc_local((1,), "int32")
            ex1 = T.alloc_local((1,), "int32")
            v1 = T.alloc_local((1,), "int32")

            T.annotate_layout({
                A_smem_wg0: tilelang.layout.make_swizzled_layout(A_smem_wg0),
                B_smem_wg0: tilelang.layout.make_swizzled_layout(B_smem_wg0),
                A_smem_wg1: tilelang.layout.make_swizzled_layout(A_smem_wg1),
                B_smem_wg1: tilelang.layout.make_swizzled_layout(B_smem_wg1),
                C_shared_wg0: tilelang.layout.make_swizzled_layout(C_shared_wg0),
                C_shared_wg1: tilelang.layout.make_swizzled_layout(C_shared_wg1),
            })

            # ── Per-WG producer/consumer barriers ──
            ab_full_wg0 = T.alloc_barrier([128] * num_stages)
            ab_empty_wg0 = T.alloc_barrier([128] * num_stages)
            ab_full_wg1 = T.alloc_barrier([128] * num_stages)
            ab_empty_wg1 = T.alloc_barrier([128] * num_stages)

            # ── Phase counters ──
            gi_prod_0 = T.alloc_var("int32", init=0)
            gi_prod_1 = T.alloc_var("int32", init=0)
            gi_cons_0 = T.alloc_var("int32", init=0)
            gi_cons_1 = T.alloc_var("int32", init=0)

            tx = T.get_thread_binding()

            # ═════════════════════════════════════════════════════════
            # Producer WG: tx < 128
            # ═════════════════════════════════════════════════════════
            if tx < 128:
                T.dec_max_nreg(24)

                if tx == 0:
                    s_cum[0] = T.int32(0)
                    for e in T.serial(num_experts):
                        s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
                    s_total[0] = s_cum[num_experts] * T.int32(_num_pid_n)
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    base = T.int32(2) * (T.int32(sm_count) * w + pid)

                    # ── Resolve WG0 tile metadata ──
                    flat_id_0 = base
                    if flat_id_0 < total:
                        m_tile_0 = flat_id_0 // T.int32(_num_pid_n)
                        n_tile_0 = flat_id_0 % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_0:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        ex0[0] = lo[0]
                        ms0[0] = (true_offsets[ex0[0]]
                                  + (m_tile_0 - s_cum[ex0[0]]) * T.int32(block_m))
                        # n_start in B is for gate_up (2N width), but tile index is in output space (N)
                        ns0[0] = n_tile_0 * T.int32(block_n)
                        v0[0] = T.int32(1)
                    else:
                        v0[0] = T.int32(0)

                    # ── Resolve WG1 tile metadata ──
                    flat_id_1 = base + T.int32(1)
                    if flat_id_1 < total:
                        m_tile_1 = flat_id_1 // T.int32(_num_pid_n)
                        n_tile_1 = flat_id_1 % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_1:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        ex1[0] = lo[0]
                        ms1[0] = (true_offsets[ex1[0]]
                                  + (m_tile_1 - s_cum[ex1[0]]) * T.int32(block_m))
                        ns1[0] = n_tile_1 * T.int32(block_n)
                        v1[0] = T.int32(1)
                    else:
                        v1[0] = T.int32(0)

                    # ── Interleaved K-loop: WG0 then WG1 each k ──
                    for k in T.Pipelined(_k_iters, num_stages=0):
                        k_start = k * block_k

                        # WG0 stream
                        if v0[0] != 0:
                            slot0 = gi_prod_0 % num_stages
                            T.barrier_wait(
                                ab_empty_wg0[slot0],
                                ((gi_prod_0 // num_stages) & 1) ^ 1)
                            T.tma_copy(
                                A[ms0[0]:ms0[0] + block_m,
                                  k_start:k_start + block_k],
                                A_smem_wg0[slot0, :, :],
                                barrier=ab_full_wg0[slot0],
                            )
                            # Load full block_n width from B (covers gate_up)
                            T.tma_copy(
                                B[ex0[0],
                                  ns0[0]:ns0[0] + block_n,
                                  k_start:k_start + block_k],
                                B_smem_wg0[slot0, :, :],
                                barrier=ab_full_wg0[slot0],
                            )
                            T.barrier_arrive(ab_full_wg0[slot0])
                            gi_prod_0 = gi_prod_0 + 1

                        # WG1 stream
                        if v1[0] != 0:
                            slot1 = gi_prod_1 % num_stages
                            T.barrier_wait(
                                ab_empty_wg1[slot1],
                                ((gi_prod_1 // num_stages) & 1) ^ 1)
                            T.tma_copy(
                                A[ms1[0]:ms1[0] + block_m,
                                  k_start:k_start + block_k],
                                A_smem_wg1[slot1, :, :],
                                barrier=ab_full_wg1[slot1],
                            )
                            T.tma_copy(
                                B[ex1[0],
                                  ns1[0]:ns1[0] + block_n,
                                  k_start:k_start + block_k],
                                B_smem_wg1[slot1, :, :],
                                barrier=ab_full_wg1[slot1],
                            )
                            T.barrier_arrive(ab_full_wg1[slot1])
                            gi_prod_1 = gi_prod_1 + 1

            # ═════════════════════════════════════════════════════════
            # Consumer WG0: 128 ≤ tx < 256
            # ═════════════════════════════════════════════════════════
            elif tx < 256:
                T.inc_max_nreg(240)
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id_0 = T.int32(2) * (T.int32(sm_count) * w + pid)
                    valid_0 = flat_id_0 < total

                    if valid_0:
                        m_tile_0 = flat_id_0 // T.int32(_num_pid_n)
                        n_tile_0 = flat_id_0 % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_0:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        expert_id_0 = lo[0]
                        row_0 = (m_tile_0 - s_cum[expert_id_0]) * T.int32(block_m)
                        m_start_0 = true_offsets[expert_id_0] + row_0
                        n_start_0 = n_tile_0 * T.int32(block_n)  # Output N-space
                        arows_0 = T.min(T.int32(block_m),
                                        true_sizes[expert_id_0] - row_0)
                        acols_0 = T.min(T.int32(block_n),
                                        T.int32(N) - n_start_0)

                        # K-loop: accumulate gate_up [block_m, block_n]
                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_0 % num_stages
                            T.barrier_wait(ab_full_wg0[slot],
                                           (gi_cons_0 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_wg0[slot, :, :],
                                B_smem_wg0[slot, :, :],
                                C_local_wg0,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_local_wg0, num_regs=64)
                            T.barrier_arrive(ab_empty_wg0[slot])
                            gi_cons_0 = gi_cons_0 + 1

                        # ── Fused activation epilogue ──
                        # C_local_wg0 is [block_m, block_n] in fp32
                        # Split into gate[:, :N] and up[:, N:], apply activation
                        # Output C_local_cast_wg0 is [block_m, N]
                        # Split gate and up reads into separate loops
                        for i, j in T.Parallel(block_m, N):
                            gate_val = C_local_wg0[i, j]
                            if activation == "silu_and_mul":
                                g_f32 = T.cast(gate_val, "float32")
                                sigmoid_g = T.sigmoid(g_f32)
                                C_local_cast_wg0[i, j] = T.cast(g_f32 * sigmoid_g, dtype)
                            else:  # gelu_and_mul
                                inv_sqrt2 = T.cast(0.7071067811865476, "float32")
                                half = T.cast(0.5, "float32")
                                one = T.cast(1.0, "float32")
                                g_f32 = T.cast(gate_val, "float32")
                                erf_val = T.erf(g_f32 * inv_sqrt2)
                                gelu_g = g_f32 * half * (one + erf_val)
                                C_local_cast_wg0[i, j] = T.cast(gelu_g, dtype)

                        # Multiply by up values
                        for i, j in T.Parallel(block_m, N):
                            up_val = C_local_wg0[i, j + N]
                            u_f32 = T.cast(up_val, "float32")
                            result_f32 = T.cast(C_local_cast_wg0[i, j], "float32")
                            C_local_cast_wg0[i, j] = T.cast(result_f32 * u_f32, dtype)

                        # Store output [block_m, N]
                        if arows_0 == T.int32(block_m) and acols_0 == T.int32(N):
                            # Fast path: full tile via TMA store
                            T.copy(C_local_cast_wg0, C_shared_wg0)
                            T.copy(C_shared_wg0, C[m_start_0, n_start_0])
                        else:
                            # Slow path: predicated direct STG
                            for i, j in T.Parallel(block_m, N):
                                if i < arows_0 and j < acols_0:
                                    C[m_start_0 + i, n_start_0 + j] = C_local_cast_wg0[i, j]

            # ═════════════════════════════════════════════════════════
            # Consumer WG1: tx ≥ 256
            # ═════════════════════════════════════════════════════════
            else:
                T.inc_max_nreg(240)
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id_1 = T.int32(2) * (T.int32(sm_count) * w + pid) + T.int32(1)
                    valid_1 = flat_id_1 < total

                    if valid_1:
                        m_tile_1 = flat_id_1 // T.int32(_num_pid_n)
                        n_tile_1 = flat_id_1 % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile_1:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        expert_id_1 = lo[0]
                        row_1 = (m_tile_1 - s_cum[expert_id_1]) * T.int32(block_m)
                        m_start_1 = true_offsets[expert_id_1] + row_1
                        n_start_1 = n_tile_1 * T.int32(block_n)
                        arows_1 = T.min(T.int32(block_m),
                                        true_sizes[expert_id_1] - row_1)
                        acols_1 = T.min(T.int32(block_n),
                                        T.int32(N) - n_start_1)

                        # K-loop
                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_1 % num_stages
                            T.barrier_wait(ab_full_wg1[slot],
                                           (gi_cons_1 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_wg1[slot, :, :],
                                B_smem_wg1[slot, :, :],
                                C_local_wg1,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_local_wg1, num_regs=64)
                            T.barrier_arrive(ab_empty_wg1[slot])
                            gi_cons_1 = gi_cons_1 + 1

                        # ── Fused activation epilogue ──
                        # Split gate and up reads into separate loops
                        for i, j in T.Parallel(block_m, N):
                            gate_val = C_local_wg1[i, j]
                            if activation == "silu_and_mul":
                                g_f32 = T.cast(gate_val, "float32")
                                sigmoid_g = T.sigmoid(g_f32)
                                C_local_cast_wg1[i, j] = T.cast(g_f32 * sigmoid_g, dtype)
                            else:  # gelu_and_mul
                                inv_sqrt2 = T.cast(0.7071067811865476, "float32")
                                half = T.cast(0.5, "float32")
                                one = T.cast(1.0, "float32")
                                g_f32 = T.cast(gate_val, "float32")
                                erf_val = T.erf(g_f32 * inv_sqrt2)
                                gelu_g = g_f32 * half * (one + erf_val)
                                C_local_cast_wg1[i, j] = T.cast(gelu_g, dtype)

                        # Multiply by up values
                        for i, j in T.Parallel(block_m, N):
                            up_val = C_local_wg1[i, j + N]
                            u_f32 = T.cast(up_val, "float32")
                            result_f32 = T.cast(C_local_cast_wg1[i, j], "float32")
                            C_local_cast_wg1[i, j] = T.cast(result_f32 * u_f32, dtype)

                        # Store output
                        if arows_1 == T.int32(block_m) and acols_1 == T.int32(N):
                            T.copy(C_local_cast_wg1, C_shared_wg1)
                            T.copy(C_shared_wg1, C[m_start_1, n_start_1])
                        else:
                            for i, j in T.Parallel(block_m, N):
                                if i < arows_1 and j < acols_1:
                                    C[m_start_1 + i, n_start_1 + j] = C_local_cast_wg1[i, j]

    return _gemm_main_v2_pingpong_fused_act


def _make_cooperative_fused_act_kernel(numel, num_experts, N, K, dtype, activation, sm_count,
                                        block_m, block_n, block_k, num_stages, threads,
                                        group_size_m):
    """Build a @T.prim_func for the cooperative template with fused activation.

    Both math WGs split the M dimension of one shared tile in half.
    Producer WG issues 3 TMAs per K-step: A_top, A_bot, and B (full 2N width).
    Math WG0 reads A_smem_top, Math WG1 reads A_smem_bot; both consume the same B.

    Epilogue: each WG computes activation on its half-tile [half_m, block_n] → [half_m, N].
    """
    accum_dtype = "float"
    log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    assert threads == 384, (
        f"V2 cooperative persistent grouped GEMM requires threads=384; got threads={threads}"
    )
    assert block_m % 2 == 0, (
        f"cooperative template requires even block_m; got block_m={block_m}"
    )
    half_m = block_m // 2
    assert half_m >= 64, (
        f"cooperative template requires block_m >= 128 (half_m={half_m} < WGMMA minimum M=64)"
    )

    _num_pid_n = math.ceil(N / block_n)
    _max_tiles = numel // block_m + num_experts
    _total_ctas_ub = _max_tiles * _num_pid_n
    _max_waves = (_total_ctas_ub + sm_count - 1) // sm_count + 1
    _k_iters = K // block_k
    A_shape = (numel + block_m, K)
    B_shape = (num_experts, N * 2, K)

    @T.prim_func
    def _gemm_main_v2_coop_fused_act(
        A: T.Tensor(A_shape, dtype),                        # type: ignore  # noqa: F821
        B: T.Tensor(B_shape, dtype),                        # type: ignore  # noqa: F821
        true_sizes: T.Tensor((num_experts,), "int32"),      # noqa: F821
        true_offsets: T.Tensor((num_experts,), "int32"),    # noqa: F821
        C: T.Tensor((numel, N), dtype),                     # type: ignore  # noqa: F821
    ):
        with T.Kernel(sm_count, threads=threads) as (pid,):
            # ── Split-A SMEM rings + shared B ring ──
            A_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
            A_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
            B_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)
            # Per-WG half-tile fragments [half_m, block_n] for gate_up
            C_local_wg0 = T.alloc_fragment((half_m, block_n), accum_dtype)
            C_local_wg1 = T.alloc_fragment((half_m, block_n), accum_dtype)

            # ── TMA-store epilogue staging (per-WG half-tile, output width N) ──
            C_local_cast_wg0 = T.alloc_fragment((half_m, N), dtype)
            C_local_cast_wg1 = T.alloc_fragment((half_m, N), dtype)
            C_shared_wg0 = T.alloc_shared((half_m, N), dtype)
            C_shared_wg1 = T.alloc_shared((half_m, N), dtype)

            # ── Scheduler SMEM ──
            s_cum = T.alloc_shared((num_experts + 1,), "int32")
            s_total = T.alloc_shared((1,), "int32")
            lo = T.alloc_local((1,), "int32")
            hi = T.alloc_local((1,), "int32")
            ms = T.alloc_local((1,), "int32")
            ns_ = T.alloc_local((1,), "int32")
            ex = T.alloc_local((1,), "int32")
            v = T.alloc_local((1,), "int32")

            T.annotate_layout({
                A_smem_top: tilelang.layout.make_swizzled_layout(A_smem_top),
                A_smem_bot: tilelang.layout.make_swizzled_layout(A_smem_bot),
                B_smem: tilelang.layout.make_swizzled_layout(B_smem),
                C_shared_wg0: tilelang.layout.make_swizzled_layout(C_shared_wg0),
                C_shared_wg1: tilelang.layout.make_swizzled_layout(C_shared_wg1),
            })

            # ── Single shared barrier set ──
            ab_full = T.alloc_barrier([128] * num_stages)
            ab_empty = T.alloc_barrier([256] * num_stages)

            gi_prod = T.alloc_var("int32", init=0)
            gi_cons_0 = T.alloc_var("int32", init=0)
            gi_cons_1 = T.alloc_var("int32", init=0)

            tx = T.get_thread_binding()

            # ═════════════════════════════════════════════════════════
            # Producer WG: tx < 128
            # ═════════════════════════════════════════════════════════
            if tx < 128:
                T.dec_max_nreg(24)

                if tx == 0:
                    s_cum[0] = T.int32(0)
                    for e in T.serial(num_experts):
                        s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
                    s_total[0] = s_cum[num_experts] * T.int32(_num_pid_n)
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id = T.int32(sm_count) * w + pid

                    if flat_id < total:
                        m_tile = flat_id // T.int32(_num_pid_n)
                        n_tile = flat_id % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        ex[0] = lo[0]
                        ms[0] = (true_offsets[ex[0]]
                                 + (m_tile - s_cum[ex[0]]) * T.int32(block_m))
                        ns_[0] = n_tile * T.int32(block_n)
                        v[0] = T.int32(1)
                    else:
                        v[0] = T.int32(0)

                    for k in T.Pipelined(_k_iters, num_stages=0):
                        k_start = k * block_k

                        if v[0] != 0:
                            slot = gi_prod % num_stages
                            T.barrier_wait(
                                ab_empty[slot],
                                ((gi_prod // num_stages) & 1) ^ 1)
                            # Top half of A
                            T.tma_copy(
                                A[ms[0]:ms[0] + half_m,
                                  k_start:k_start + block_k],
                                A_smem_top[slot, :, :],
                                barrier=ab_full[slot],
                            )
                            # Bottom half of A
                            T.tma_copy(
                                A[ms[0] + half_m:ms[0] + block_m,
                                  k_start:k_start + block_k],
                                A_smem_bot[slot, :, :],
                                barrier=ab_full[slot],
                            )
                            # Full B tile (gate_up, width block_n = 2N)
                            T.tma_copy(
                                B[ex[0],
                                  ns_[0]:ns_[0] + block_n,
                                  k_start:k_start + block_k],
                                B_smem[slot, :, :],
                                barrier=ab_full[slot],
                            )
                            T.barrier_arrive(ab_full[slot])
                            gi_prod = gi_prod + 1

            # ═════════════════════════════════════════════════════════
            # Consumer WG0: 128 ≤ tx < 256 — top half
            # ═════════════════════════════════════════════════════════
            elif tx < 256:
                T.inc_max_nreg(240)
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id = T.int32(sm_count) * w + pid

                    if flat_id < total:
                        m_tile = flat_id // T.int32(_num_pid_n)
                        n_tile = flat_id % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        expert_id = lo[0]
                        row = (m_tile - s_cum[expert_id]) * T.int32(block_m)
                        m_start = true_offsets[expert_id] + row
                        n_start = n_tile * T.int32(block_n)
                        # Top-half row count
                        true_arows = true_sizes[expert_id] - row
                        arows0 = T.max(T.int32(0),
                                       T.min(T.int32(half_m), true_arows))
                        acols0 = T.min(T.int32(block_n), T.int32(N) - n_start)

                        # K-loop: accumulate gate_up [half_m, block_n]
                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_0 % num_stages
                            T.barrier_wait(ab_full[slot],
                                           (gi_cons_0 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_top[slot, :, :],
                                B_smem[slot, :, :],
                                C_local_wg0,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_local_wg0, num_regs=64)
                            T.barrier_arrive(ab_empty[slot])
                            gi_cons_0 = gi_cons_0 + 1

                        # ── Fused activation epilogue (top half) ──
                        # Split gate and up reads into separate loops to avoid
                        # conflicting access patterns in layout inference
                        for i, j in T.Parallel(half_m, N):
                            gate_val = C_local_wg0[i, j]
                            if activation == "silu_and_mul":
                                g_f32 = T.cast(gate_val, "float32")
                                sigmoid_g = T.sigmoid(g_f32)
                                C_local_cast_wg0[i, j] = T.cast(g_f32 * sigmoid_g, dtype)
                            else:  # gelu_and_mul
                                inv_sqrt2 = T.cast(0.7071067811865476, "float32")
                                half = T.cast(0.5, "float32")
                                one = T.cast(1.0, "float32")
                                g_f32 = T.cast(gate_val, "float32")
                                erf_val = T.erf(g_f32 * inv_sqrt2)
                                gelu_g = g_f32 * half * (one + erf_val)
                                C_local_cast_wg0[i, j] = T.cast(gelu_g, dtype)

                        # Multiply by up values
                        for i, j in T.Parallel(half_m, N):
                            up_val = C_local_wg0[i, j + N]
                            u_f32 = T.cast(up_val, "float32")
                            result_f32 = T.cast(C_local_cast_wg0[i, j], "float32")
                            C_local_cast_wg0[i, j] = T.cast(result_f32 * u_f32, dtype)

                        # Store output [half_m, N]
                        if arows0 == T.int32(half_m) and acols0 == T.int32(N):
                            T.copy(C_local_cast_wg0, C_shared_wg0)
                            T.copy(C_shared_wg0, C[m_start, n_start])
                        else:
                            for i, j in T.Parallel(half_m, N):
                                if i < arows0 and j < acols0:
                                    C[m_start + i, n_start + j] = C_local_cast_wg0[i, j]

            # ═════════════════════════════════════════════════════════
            # Consumer WG1: tx ≥ 256 — bottom half
            # ═════════════════════════════════════════════════════════
            else:
                T.inc_max_nreg(240)
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    flat_id = T.int32(sm_count) * w + pid

                    if flat_id < total:
                        m_tile = flat_id // T.int32(_num_pid_n)
                        n_tile = flat_id % T.int32(_num_pid_n)

                        lo[0] = T.int32(0)
                        hi[0] = T.int32(num_experts - 1)
                        for _bs in T.serial(log2_up):
                            mid = (lo[0] + hi[0]) >> T.int32(1)
                            if s_cum[mid + 1] <= m_tile:
                                lo[0] = mid + T.int32(1)
                            else:
                                hi[0] = mid
                        expert_id = lo[0]
                        row = (m_tile - s_cum[expert_id]) * T.int32(block_m)
                        m_start = true_offsets[expert_id] + row
                        n_start = n_tile * T.int32(block_n)
                        # Bottom-half row count
                        true_arows = true_sizes[expert_id] - row
                        arows1 = T.max(T.int32(0),
                                       T.min(T.int32(half_m),
                                             true_arows - T.int32(half_m)))
                        acols1 = T.min(T.int32(block_n), T.int32(N) - n_start)

                        # K-loop
                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_1 % num_stages
                            T.barrier_wait(ab_full[slot],
                                           (gi_cons_1 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_bot[slot, :, :],
                                B_smem[slot, :, :],
                                C_local_wg1,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_local_wg1, num_regs=64)
                            T.barrier_arrive(ab_empty[slot])
                            gi_cons_1 = gi_cons_1 + 1

                        # ── Fused activation epilogue (bottom half) ──
                        # Split gate and up reads into separate loops
                        for i, j in T.Parallel(half_m, N):
                            gate_val = C_local_wg1[i, j]
                            if activation == "silu_and_mul":
                                g_f32 = T.cast(gate_val, "float32")
                                sigmoid_g = T.sigmoid(g_f32)
                                C_local_cast_wg1[i, j] = T.cast(g_f32 * sigmoid_g, dtype)
                            else:  # gelu_and_mul
                                inv_sqrt2 = T.cast(0.7071067811865476, "float32")
                                half = T.cast(0.5, "float32")
                                one = T.cast(1.0, "float32")
                                g_f32 = T.cast(gate_val, "float32")
                                erf_val = T.erf(g_f32 * inv_sqrt2)
                                gelu_g = g_f32 * half * (one + erf_val)
                                C_local_cast_wg1[i, j] = T.cast(gelu_g, dtype)

                        # Multiply by up values
                        for i, j in T.Parallel(half_m, N):
                            up_val = C_local_wg1[i, j + N]
                            u_f32 = T.cast(up_val, "float32")
                            result_f32 = T.cast(C_local_cast_wg1[i, j], "float32")
                            C_local_cast_wg1[i, j] = T.cast(result_f32 * u_f32, dtype)

                        # Store output
                        if arows1 == T.int32(half_m) and acols1 == T.int32(N):
                            T.copy(C_local_cast_wg1, C_shared_wg1)
                            T.copy(C_shared_wg1, C[m_start + half_m, n_start])
                        elif arows1 > T.int32(0):
                            for i, j in T.Parallel(half_m, N):
                                if i < arows1 and j < acols1:
                                    C[m_start + half_m + i, n_start + j] = C_local_cast_wg1[i, j]

    return _gemm_main_v2_coop_fused_act
