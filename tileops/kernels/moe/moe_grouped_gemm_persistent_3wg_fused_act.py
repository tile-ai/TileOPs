"""MoE 3WG persistent grouped-GEMM with fused gate/up activation (K-aligned).

Computes  A[numel,K] @ B[E, 2*ffn, K]^T  ->  C[numel, ffn]  with
Per ffn output N-tile [n0, n0+bn):
    gate accumulator = A @ B[e,        n0 : n0+bn, :]^T
    up   accumulator = A @ B[e, ffn +  n0 : n0+bn, :]^T   (shares the same A)
    C[:, n0:n0+bn]   = act(gate) * up        (act in {silu_and_mul, gelu_and_mul})
N-tiling is over ffn; the two accumulators are fused in the epilogue so the
[numel, 2*ffn] gate_up tensor never reaches global memory.
"""
import functools
import math
import os

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel

_ANCHOR_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../grouped_gemm/_anchor_helper.h")
)

__all__ = ["MoeGroupedGemmPersistent3WGFusedActKernel"]

_DEFAULT_CONFIG = {
    "block_m": 128, "block_n": 128, "block_k": 64,
    "num_stages": 3, "threads": 384, "group_size_m": 1,
}


def _act_expr(name):
    """Return a fn(g_fp32) -> activated fp32 TIR expr (compile-time specialized)."""
    if name == "silu_and_mul":
        return lambda g: g / (T.float32(1.0) + T.exp(-g))
    if name == "gelu_and_mul":
        # exact erf GELU: 0.5*g*(1+erf(g/sqrt(2)))
        return lambda g: T.float32(0.5) * g * (T.float32(1.0) + T.erf(g * T.float32(0.7071067811865476)))
    raise ValueError(f"unsupported activation {name!r}")


class MoeGroupedGemmPersistent3WGFusedActKernel(Kernel):
    """3WG persistent grouped-GEMM with fused gate/up activation (K-aligned only)."""

    supported_archs: list[int] = [90]

    def __init__(self, numel, num_experts, N, K, dtype=torch.bfloat16,
                 activation="silu_and_mul", sm_count=None, config=None, tune=False):
        super().__init__()
        if activation not in ("silu_and_mul", "gelu_and_mul"):
            raise ValueError(
                f"activation must be 'silu_and_mul' or 'gelu_and_mul', got {activation!r}")
        self.numel = numel
        self.num_experts = num_experts
        self.N = N            # ffn (output width), NOT 2*ffn
        self.K = K
        self.dtype = dtype
        self.activation = activation
        if sm_count is None:
            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        self.sm_count = sm_count
        self.kernel = lambda: _fused_act_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.activation, self.sm_count, _DEFAULT_CONFIG["block_k"])
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        # Dual-B doubles the B ring and uses two ffn-wide accumulators, so
        # block_n is capped at 128 (2x128 fp32 accum ~= one 256 accum).
        SMEM_LIMIT = 228 * 1024
        bpe = 2
        configs = []
        for block_m in (64,):  # pingpong
            for block_n in (128,):
                for num_stages in (2, 3, 4):
                    bk = 64
                    smem = (2 * num_stages * block_m * bk * bpe          # A wg0/wg1
                            + 2 * 2 * num_stages * block_n * bk * bpe    # B gate+up, wg0/wg1
                            + 2 * block_m * block_n * bpe)               # C_shared wg0/wg1
                    if smem <= SMEM_LIMIT:
                        configs.append({"block_m": block_m, "block_n": block_n,
                                        "block_k": bk, "num_stages": num_stages,
                                        "threads": 384, "group_size_m": 1})
        for block_m in (128,):  # cooperative
            half = block_m // 2
            for block_n in (128,):
                for num_stages in (2, 3, 4):
                    bk = 64
                    smem = (2 * num_stages * half * bk * bpe             # A top/bot
                            + 2 * num_stages * block_n * bk * bpe        # B gate+up (shared)
                            + 2 * half * block_n * bpe)                  # C_shared wg0/wg1
                    if smem <= SMEM_LIMIT:
                        configs.append({"block_m": block_m, "block_n": block_n,
                                        "block_k": bk, "num_stages": num_stages,
                                        "threads": 384, "group_size_m": 1})
        return configs

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        from tilelang.profiler import do_bench as _do_bench

        print(f"Start autotuning {self.__class__.__name__}...")
        best_ms, best_cfg = float("inf"), None
        A = torch.randn(self.numel, self.K, dtype=self.dtype, device="cuda") * 0.02
        B = torch.randn(self.num_experts, 2 * self.N, self.K, dtype=self.dtype, device="cuda") * 0.02
        per = max(1, self.numel // self.num_experts)
        sizes = torch.full((self.num_experts,), per, dtype=torch.int32, device="cuda")
        sizes[-1] = self.numel - per * (self.num_experts - 1)
        offsets = torch.zeros(self.num_experts, dtype=torch.int32, device="cuda")
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        for cfg in self.autotune_configs:
            try:
                self.config = cfg
                self.forward(A, B, sizes, offsets)
                ms = _do_bench(lambda: self.forward(A, B, sizes, offsets), warmup=warmup, rep=rep)
                if ms < best_ms:
                    best_ms, best_cfg = ms, cfg
            except Exception:
                continue
        if best_cfg is not None:
            self.config = best_cfg
            print(f"Best config: {best_cfg} ({best_ms:.3f} ms)")
        else:
            self.config = self.default_config
            print("Autotune failed for all configs, using default.")

    def forward(self, A, B, true_sizes, true_offsets):
        C = torch.zeros(self.numel, self.N, dtype=self.dtype, device=A.device)
        bm, bn, bk = self.config["block_m"], self.config["block_n"], self.config["block_k"]
        if self.K % bk != 0:
            raise ValueError(f"K-aligned only: K={self.K}, block_k={bk}")
        if self.N % bn != 0:
            raise ValueError(f"ffn must be divisible by block_n: N={self.N}, block_n={bn}")
        required_rows = self.numel + bm
        if A.shape[0] < required_rows:
            A = F.pad(A, (0, 0, 0, required_rows - A.shape[0]))
        fn = _fused_act_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.activation, self.sm_count, bk,
        )(bm, bn, bk, self.config["num_stages"], self.config["threads"],
          self.config.get("group_size_m", 1))
        fn(A, B, true_sizes, true_offsets, C)
        return C


def _make_pingpong_fused_act_kernel(numel, num_experts, ffn, K, dtype, activation,
                                    sm_count, block_m, block_n, block_k,
                                    num_stages, threads, group_size_m):
    """Build a @T.prim_func for the pingpong fused-activation template (block_m <= 64).

    Two math WGs each work an independent ffn N-tile per wave; each math WG
    holds its own SMEM rings (A_smem_wgX, B_gate_smem_wgX, B_up_smem_wgX), each
    shaped ``(num_stages, ...)``.  Each consumer issues two WGMMAs per K-step into a
    gate and an up accumulator, then fuses ``act(gate) * up`` in the epilogue so
    the [numel, 2*ffn] gate_up tensor never reaches global memory.
    """
    accum_dtype = "float"
    log2_up = max(1, math.ceil(math.log2(num_experts + 1)))

    assert threads == 384, (
        f"fused-act pingpong persistent grouped GEMM requires threads=384 "
        f"(1 producer + 2 consumer WGs); got threads={threads}"
    )
    # N-tiling is over ffn (the output width), NOT 2*ffn.
    _num_pid_n = math.ceil(ffn / block_n)
    _max_tiles = numel // block_m + num_experts  # over-estimate of M tiles
    _total_ctas_ub = _max_tiles * _num_pid_n
    # Pingpong: each CTA processes 2 tiles per wave (one per math WG).
    _max_waves = (_total_ctas_ub + 2 * sm_count - 1) // (2 * sm_count) + 1
    _k_iters = K // block_k
    A_shape = (numel + block_m, K)

    act = _act_expr(activation)

    @T.prim_func
    def _gemm_main_fused(
        A: T.Tensor(A_shape, dtype),                            # type: ignore  # noqa: F821
        B: T.Tensor((num_experts, 2 * ffn, K), dtype),         # type: ignore  # noqa: F821
        true_sizes: T.Tensor((num_experts,), "int32"),          # noqa: F821
        true_offsets: T.Tensor((num_experts,), "int32"),        # noqa: F821
        C: T.Tensor((numel, ffn), dtype),                       # type: ignore  # noqa: F821
    ):
        with T.Kernel(sm_count, threads=threads) as (pid,):
            # ── Per-WG ring-buffered SMEM (num_stages slots): A + dual B ──
            A_smem_wg0 = T.alloc_shared((num_stages, block_m, block_k), dtype)
            B_gate_smem_wg0 = T.alloc_shared((num_stages, block_n, block_k), dtype)
            B_up_smem_wg0 = T.alloc_shared((num_stages, block_n, block_k), dtype)
            A_smem_wg1 = T.alloc_shared((num_stages, block_m, block_k), dtype)
            B_gate_smem_wg1 = T.alloc_shared((num_stages, block_n, block_k), dtype)
            B_up_smem_wg1 = T.alloc_shared((num_stages, block_n, block_k), dtype)
            # Dual fp32 accumulators per WG (gate + up).
            C_gate_wg0 = T.alloc_fragment((block_m, block_n), accum_dtype)
            C_up_wg0 = T.alloc_fragment((block_m, block_n), accum_dtype)
            C_gate_wg1 = T.alloc_fragment((block_m, block_n), accum_dtype)
            C_up_wg1 = T.alloc_fragment((block_m, block_n), accum_dtype)

            # ── Epilogue: per-WG cast fragment + TMA-store SMEM (ffn-wide) ──
            C_local_cast_wg0 = T.alloc_fragment((block_m, block_n), dtype)
            C_local_cast_wg1 = T.alloc_fragment((block_m, block_n), dtype)
            C_shared_wg0 = T.alloc_shared((block_m, block_n), dtype)
            C_shared_wg1 = T.alloc_shared((block_m, block_n), dtype)

            # ── Scheduler SMEM (no atomic counter / tile pair) ──
            s_cum = T.alloc_shared((num_experts + 1,), "int32")
            s_total = T.alloc_shared((1,), "int32")
            lo = T.alloc_local((1,), "int32")
            hi = T.alloc_local((1,), "int32")
            # Per-tile metadata hoisted to alloc_local so the interleaved
            # K-loop body can read m_start/n_start/expert_id across IfFrame
            # boundaries.
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
                B_gate_smem_wg0: tilelang.layout.make_swizzled_layout(B_gate_smem_wg0),
                B_up_smem_wg0: tilelang.layout.make_swizzled_layout(B_up_smem_wg0),
                A_smem_wg1: tilelang.layout.make_swizzled_layout(A_smem_wg1),
                B_gate_smem_wg1: tilelang.layout.make_swizzled_layout(B_gate_smem_wg1),
                B_up_smem_wg1: tilelang.layout.make_swizzled_layout(B_up_smem_wg1),
                C_shared_wg0: tilelang.layout.make_swizzled_layout(C_shared_wg0),
                C_shared_wg1: tilelang.layout.make_swizzled_layout(C_shared_wg1),
            })

            # ── Per-WG producer/consumer barriers (arrive_count=128) ──
            ab_full_wg0 = T.alloc_barrier([128] * num_stages)
            ab_empty_wg0 = T.alloc_barrier([128] * num_stages)
            ab_full_wg1 = T.alloc_barrier([128] * num_stages)
            ab_empty_wg1 = T.alloc_barrier([128] * num_stages)

            # ── Phase counters: monotonic, never reset ──
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
                # CTA-wide sync: publishes s_cum/s_total to consumers.
                T.sync_threads()

                for w in T.serial(_max_waves):
                    total = s_total[0]
                    base = T.int32(2) * (T.int32(sm_count) * w + pid)

                    # ── Resolve WG0 tile metadata into alloc_local ──
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
                        ns0[0] = n_tile_0 * T.int32(block_n)
                        v0[0] = T.int32(1)
                    else:
                        v0[0] = T.int32(0)

                    # ── Resolve WG1 tile metadata into alloc_local ──
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

                        # WG0 stream: load A once, gate AND up B tiles.
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
                            T.tma_copy(
                                B[ex0[0],
                                  ns0[0]:ns0[0] + block_n,
                                  k_start:k_start + block_k],
                                B_gate_smem_wg0[slot0, :, :],
                                barrier=ab_full_wg0[slot0],
                            )
                            T.tma_copy(
                                B[ex0[0],
                                  ffn + ns0[0]:ffn + ns0[0] + block_n,
                                  k_start:k_start + block_k],
                                B_up_smem_wg0[slot0, :, :],
                                barrier=ab_full_wg0[slot0],
                            )
                            T.barrier_arrive(ab_full_wg0[slot0])
                            gi_prod_0 = gi_prod_0 + 1

                        # WG1 stream: load A once, gate AND up B tiles.
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
                                B_gate_smem_wg1[slot1, :, :],
                                barrier=ab_full_wg1[slot1],
                            )
                            T.tma_copy(
                                B[ex1[0],
                                  ffn + ns1[0]:ffn + ns1[0] + block_n,
                                  k_start:k_start + block_k],
                                B_up_smem_wg1[slot1, :, :],
                                barrier=ab_full_wg1[slot1],
                            )
                            T.barrier_arrive(ab_full_wg1[slot1])
                            gi_prod_1 = gi_prod_1 + 1

            # ═════════════════════════════════════════════════════════
            # Consumer WG0: 128 ≤ tx < 256 — processes flat_id_0 per wave
            # ═════════════════════════════════════════════════════════
            elif tx < 256:
                T.inc_max_nreg(240)
                # CTA-wide sync (pairs with producer's post-init sync).
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
                        n_start_0 = n_tile_0 * T.int32(block_n)
                        arows_0 = T.min(T.int32(block_m),
                                        true_sizes[expert_id_0] - row_0)
                        acols_0 = T.min(T.int32(block_n),
                                        T.int32(ffn) - n_start_0)

                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_0 % num_stages
                            T.barrier_wait(ab_full_wg0[slot],
                                           (gi_cons_0 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_wg0[slot, :, :],
                                B_gate_smem_wg0[slot, :, :],
                                C_gate_wg0,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wgmma_gemm(
                                A_smem_wg0[slot, :, :],
                                B_up_smem_wg0[slot, :, :],
                                C_up_wg0,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_gate_wg0, num_regs=64)
                            T.warpgroup_fence_operand(C_up_wg0, num_regs=64)
                            T.barrier_arrive(ab_empty_wg0[slot])
                            gi_cons_0 = gi_cons_0 + 1

                        # ── Fused epilogue: act(gate) * up → cast → store ──
                        for i, j in T.Parallel(block_m, block_n):
                            C_local_cast_wg0[i, j] = T.cast(
                                act(C_gate_wg0[i, j]) * C_up_wg0[i, j], dtype)
                        if arows_0 == T.int32(block_m):
                            # WG-scoped named barrier BEFORE refilling C_shared:
                            # guarantees the prior wave's TMA store finished
                            # reading C_shared.  MUST be a named barrier
                            # (barrier_id + arrive_count=128), NOT a CTA-wide
                            # T.sync_threads() — this branch is warpgroup-
                            # divergent and a CTA-wide sync would deadlock.
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.copy(C_local_cast_wg0, C_shared_wg0)
                            T.copy(C_shared_wg0, C[m_start_0, n_start_0])
                        else:
                            for i, j in T.Parallel(block_m, block_n):
                                if i < arows_0 and j < acols_0:
                                    C[m_start_0 + i, n_start_0 + j] = C_local_cast_wg0[i, j]

            # ═════════════════════════════════════════════════════════
            # Consumer WG1: tx ≥ 256 — processes flat_id_1 per wave
            # ═════════════════════════════════════════════════════════
            else:
                T.inc_max_nreg(240)
                # CTA-wide sync (pairs with producer's post-init sync).
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
                                        T.int32(ffn) - n_start_1)

                        for k in T.Pipelined(_k_iters, num_stages=0):
                            slot = gi_cons_1 % num_stages
                            T.barrier_wait(ab_full_wg1[slot],
                                           (gi_cons_1 // num_stages) & 1)
                            T.wgmma_gemm(
                                A_smem_wg1[slot, :, :],
                                B_gate_smem_wg1[slot, :, :],
                                C_gate_wg1,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wgmma_gemm(
                                A_smem_wg1[slot, :, :],
                                B_up_smem_wg1[slot, :, :],
                                C_up_wg1,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=(k == 0),
                            )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(C_gate_wg1, num_regs=64)
                            T.warpgroup_fence_operand(C_up_wg1, num_regs=64)
                            T.barrier_arrive(ab_empty_wg1[slot])
                            gi_cons_1 = gi_cons_1 + 1

                        # ── Fused epilogue: act(gate) * up → cast → store ──
                        for i, j in T.Parallel(block_m, block_n):
                            C_local_cast_wg1[i, j] = T.cast(
                                act(C_gate_wg1[i, j]) * C_up_wg1[i, j], dtype)
                        if arows_1 == T.int32(block_m):
                            # WG-scoped named barrier (barrier_id=5) BEFORE the
                            # C_shared refill — see WG0 epilogue rationale.
                            T.sync_threads(barrier_id=5, arrive_count=128)
                            T.copy(C_local_cast_wg1, C_shared_wg1)
                            T.copy(C_shared_wg1, C[m_start_1, n_start_1])
                        else:
                            for i, j in T.Parallel(block_m, block_n):
                                if i < arows_1 and j < acols_1:
                                    C[m_start_1 + i, n_start_1 + j] = C_local_cast_wg1[i, j]

    return _gemm_main_fused


def _make_cooperative_fused_act_kernel(numel, num_experts, ffn, K, dtype, activation,
                                       sm_count, block_m, block_n, block_k,
                                       num_stages, threads, group_size_m):
    raise NotImplementedError("cooperative template — Task 3")


@functools.lru_cache(maxsize=64)
def _fused_act_kernel(numel, num_experts, ffn, K, dtype, activation, sm_count, block_k):
    """Fused gate/up activation grouped-GEMM JIT factory (K-aligned).

    Dispatches by ``block_m``:
      * ``block_m <= 64``  -> pingpong (2 math WGs work independent tiles; per-WG dual-B rings)
      * ``block_m >= 128`` -> cooperative (2 math WGs split one tile's M; shared dual-B rings)
    """
    if K % block_k != 0:
        raise ValueError(f"K-aligned only: K={K}, block_k={block_k}.")

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16", "-include", _ANCHOR_HELPER_PATH],
    )
    def _func(block_m, block_n, block_k, num_stages, threads, group_size_m):
        if block_m <= 64:
            return _make_pingpong_fused_act_kernel(
                numel, num_experts, ffn, K, dtype, activation, sm_count,
                block_m, block_n, block_k, num_stages, threads, group_size_m)
        return _make_cooperative_fused_act_kernel(
            numel, num_experts, ffn, K, dtype, activation, sm_count,
            block_m, block_n, block_k, num_stages, threads, group_size_m)

    return _func
