from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from benchmarks.benchmark_base import BenchmarkBase
from tests.ops.test_gemm import GemmW4A16Test

W4A16_GROUP_SIZE = 128


@dataclass(frozen=True)
class W4A16BenchmarkCase:
    m: int
    n: int
    k: int
    label: str
    scenario: str
    purpose: str
    weight_metadata_mib: float | None = None


FEASIBILITY_CASES = (
    W4A16BenchmarkCase(
        128,
        2112,
        7168,
        "feasibility-prefill-contracting",
        "prefill-contracting-projection",
        "medium-M large-K/small-N case for validating 3WG dequant + WGMMA pipeline",
    ),
    W4A16BenchmarkCase(
        128,
        7168,
        2048,
        "feasibility-prefill-expanding",
        "prefill-expanding-projection",
        "opposite aspect ratio to test Tensor Core throughput and scheduling",
    ),
    W4A16BenchmarkCase(
        1,
        7168,
        2048,
        "feasibility-decode-short-k",
        "decode-short-k",
        "small-K decode case where launch/depack overhead can dominate",
    ),
    W4A16BenchmarkCase(
        1,
        8192,
        8192,
        "feasibility-decode-medium-k",
        "decode-medium-k",
        "medium-K decode case testing weight bandwidth and W4 capacity benefit",
    ),
)


DECODE_AKO_CASES = (
    W4A16BenchmarkCase(
        1,
        8192,
        8192,
        "decode-ako-l2-resident-ish",
        "decode-medium-k",
        "medium K; packed W4 weight + metadata fits within H200 L2, exposing launch/depack/sync overhead",
        34.5,
    ),
    W4A16BenchmarkCase(
        1,
        8192,
        16384,
        "decode-ako-hbm-streaming-threshold",
        "decode-over-l2",
        "just over H200 L2, entering real HBM streaming and validating TMA/buffering behavior",
        69.0,
    ),
    W4A16BenchmarkCase(
        1,
        7168,
        20480,
        "decode-ako-non-power2-low-cta",
        "decode-non-power2-n",
        "non-power-of-two N with only 112 N64 CTAs, exposing occupancy and scheduling limits",
        75.5,
    ),
    W4A16BenchmarkCase(
        1,
        8192,
        81920,
        "decode-ako-long-k-pressure",
        "decode-long-k-stress",
        "very long K far beyond L2, amplifying HBM streaming, depack overlap, activation reuse, and split-K effects",
        345.0,
    ),
)


class StaticGemmW4A16Benchmark(BenchmarkBase[GemmW4A16Test]):
    def __init__(self, test: GemmW4A16Test, memory_bytes: int) -> None:
        super().__init__(test)
        self._memory_bytes = memory_bytes

    def calculate_flops(self) -> float:
        return float(2 * self.workload.m * self.workload.n * self.workload.k)

    def calculate_memory(self) -> float:
        return float(self._memory_bytes)


def w4a16_memory_bytes(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.float16,
    group_size: int = W4A16_GROUP_SIZE,
) -> int:
    groups = k // group_size
    elem_bytes = dtype.itemsize
    return m * k * elem_bytes + n * k // 2 + n * groups * (4 + 1) + m * n * elem_bytes


def dense_a16_memory_bytes(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    elem_bytes = dtype.itemsize
    return (m * k + n * k + m * n) * elem_bytes


def marlin_w4a16_memory_bytes(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.float16,
    group_size: int = W4A16_GROUP_SIZE,
) -> int:
    elem_bytes = dtype.itemsize
    qweight_bytes = (k // 16) * (n * 2) * torch.int32.itemsize
    scale_bytes = (k // group_size) * n * elem_bytes
    zero_bytes = (k // group_size) * (n // 8) * torch.int32.itemsize
    return m * k * elem_bytes + qweight_bytes + scale_bytes + zero_bytes + m * n * elem_bytes


def make_marlin_w4a16_callable(
    m: int,
    n: int,
    k: int,
    use_fp32_reduce: bool,
) -> tuple[Callable[..., torch.Tensor], tuple[Any, ...]]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.scalar_type import scalar_types

    if k % 16 or k % W4A16_GROUP_SIZE or n % 64:
        raise ValueError("Marlin W4A16 benchmark requires K % 128 == 0 and N % 64 == 0")

    device = torch.device("cuda")
    activation = torch.randn((m, k), dtype=torch.float16, device=device)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (k // 16, n * 2),
        dtype=torch.int32,
        device=device,
    )
    scales = torch.rand((k // W4A16_GROUP_SIZE, n), dtype=torch.float16, device=device)
    zeros = torch.randint(
        -(2**31),
        2**31 - 1,
        (k // W4A16_GROUP_SIZE, n // 8),
        dtype=torch.int32,
        device=device,
    )
    workspace = marlin_make_workspace_new(device)

    def marlin(
        a: torch.Tensor,
        packed: torch.Tensor,
        weight_scales: torch.Tensor,
        weight_zeros: torch.Tensor,
        locks: torch.Tensor,
    ) -> torch.Tensor:
        return ops.marlin_gemm(
            a=a,
            c=None,
            b_q_weight=packed,
            b_bias=None,
            b_scales=weight_scales,
            a_scales=None,
            global_scale=None,
            b_zeros=weight_zeros,
            g_idx=None,
            perm=None,
            workspace=locks,
            b_q_type=scalar_types.uint4,
            size_m=m,
            size_n=n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )

    return marlin, (activation, qweight, scales, zeros, workspace)
