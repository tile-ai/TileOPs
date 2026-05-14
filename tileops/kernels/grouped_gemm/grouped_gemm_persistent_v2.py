"""V2 persistent grouped-GEMM kernel: warp-specialized + static wave + TMA store.

Mirrors the structure of the TileLang SM100 example
``gemm_tcgen5mma_ws_persistent.py``, with all SM100-only primitives
(TMEM, tcgen05_gemm, CLC, 2-CTA MMA) replaced by SM90 equivalents.

Design: see tileops_md/2026-05-15_grouped_gemm_persistent_v2_design.md.

Two compile-time templates routed by ``block_m``:
  * ``block_m ≤ 64``  → Pingpong: 2 math WGs process independent tiles
  * ``block_m ≥ 128`` → Cooperative: 2 math WGs share one tile's M

K-aligned only.  K-unaligned use ``GroupedGemmPersistentKernel``.
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
    os.path.join(os.path.dirname(__file__), "_anchor_helper.h")
)

__all__ = ["GroupedGemmPersistentV2Kernel"]

_DEFAULT_CONFIG = {
    "block_m": 64,         # phase 1: bm=64 (pingpong) only
    "block_n": 256,
    "block_k": 64,
    "num_stages": 2,
    "threads": 384,
    "group_size_m": 1,
}


class GroupedGemmPersistentV2Kernel(Kernel):
    """V2 persistent grouped-GEMM kernel (K-aligned only)."""

    supported_archs: list[int] = [90]

    def __init__(self, numel, num_experts, N, K,
                 dtype=torch.bfloat16, sm_count=None,
                 config=None, tune=False):
        super().__init__()
        self.numel = numel
        self.num_experts = num_experts
        self.N = N
        self.K = K
        self.dtype = dtype
        if sm_count is None:
            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        self.sm_count = sm_count
        self.kernel = lambda: _persistent_grouped_gemm_v2_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.sm_count, self.config["block_k"])
        self.init_config(config, tune)
        self._tile_counter: torch.Tensor | None = None

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        return [dict(_DEFAULT_CONFIG)]  # phase 1: single config

    def forward(self, A, B, true_sizes, true_offsets):
        if self._tile_counter is None or self._tile_counter.device != A.device:
            self._tile_counter = torch.zeros(1, dtype=torch.int32, device=A.device)
        else:
            self._tile_counter.zero_()
        C = torch.zeros(self.numel, self.N, dtype=self.dtype, device=A.device)
        block_m = self.config["block_m"]
        block_n = self.config["block_n"]
        block_k = self.config["block_k"]
        if self.K % block_k != 0:
            raise ValueError(f"K-aligned only: K={self.K}, block_k={block_k}")
        if self.N % block_n != 0:
            raise ValueError(f"N-aligned only: N={self.N}, block_n={block_n}")
        A = F.pad(A, (0, 0, 0, block_m))
        gemm_fn = _persistent_grouped_gemm_v2_kernel(
            self.numel, self.num_experts, self.N, self.K,
            self.dtype_str, self.sm_count, block_k,
        )(
            block_m, block_n, block_k,
            self.config["num_stages"],
            self.config["threads"],
            self.config.get("group_size_m", 1),
        )
        gemm_fn(A, B, true_sizes, true_offsets, C, self._tile_counter)
        return C


@functools.lru_cache(maxsize=64)
def _persistent_grouped_gemm_v2_kernel(numel, num_experts, N, K, dtype,
                                       sm_count, block_k):
    """Build a V2 persistent grouped-GEMM JIT factory."""
    raise NotImplementedError("phase 1: factory body not yet written")
