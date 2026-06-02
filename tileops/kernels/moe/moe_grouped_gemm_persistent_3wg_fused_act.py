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
    raise NotImplementedError("pingpong template — Task 2")


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
