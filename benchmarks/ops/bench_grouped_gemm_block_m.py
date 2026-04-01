"""Benchmark: impact of block_m on NT grouped GEMM for MoE workloads.

Varies block_m (16 / 32 / 64 / 128) in the existing grouped_gemm_kernel (NT
layout) to find the sweet spot for small-M-per-expert distributions typical of
Qwen3-MoE with E=256.

FLOPs and memory are computed from the TRUE (unpadded) batch_sum so that
efficiency reflects useful work only; padding overhead shows up as lower TFLOPS.

Usage:
    conda run -n tileops python -m pytest \
        benchmarks/ops/bench_grouped_gemm_block_m.py -vvs
"""

import math
from typing import List, Optional

import pytest
import torch
from tilelang.profiler import do_bench

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as _vllm_fused_experts
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from tileops.kernels.grouped_gemm import grouped_gemm_kernel

_WARMUP = 100
_REP = 200
_BLOCK_M_VARIANTS = [16, 32, 64, 128]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _padded_batch_sum(true_batch_sizes: List[int], block_m: int) -> int:
    return sum(math.ceil(s / block_m) * block_m for s in true_batch_sizes)


def _build_inputs(true_batch_sizes, N, K, dtype, device, block_m):
    """Build padded A, B and metadata for grouped_gemm_kernel (NT)."""
    E = len(true_batch_sizes)
    pad_sum = _padded_batch_sum(true_batch_sizes, block_m)

    A = torch.zeros(pad_sum, K, dtype=dtype, device=device)
    B = torch.randn(E, N, K, dtype=dtype, device=device) * 0.02

    sizes = torch.zeros(E, dtype=torch.int32, device=device)
    offsets = torch.zeros(E, dtype=torch.int32, device=device)
    pad_off = 0
    for i, s in enumerate(true_batch_sizes):
        ps = math.ceil(s / block_m) * block_m
        sizes[i] = ps
        offsets[i] = pad_off
        pad_off += ps

    # Fill valid rows with random data
    true_off = 0
    pad_off = 0
    for _i, s in enumerate(true_batch_sizes):
        ps = math.ceil(s / block_m) * block_m
        A[pad_off:pad_off + s] = torch.randn(s, K, dtype=dtype, device=device)
        true_off += s
        pad_off += ps

    return A, B, sizes, offsets, offsets  # offsets used for both args


def _bench(fn) -> float:
    with torch.no_grad():
        ms = do_bench(fn, warmup=_WARMUP, rep=_REP, backend='cupti')
        if ms <= 0:
            ms = do_bench(fn, warmup=_WARMUP, rep=_REP,
                          backend='event', return_mode='median')
    return ms


# ---------------------------------------------------------------------------
# Cases: (label, true_batch_sizes, N, K)
# ---------------------------------------------------------------------------

_CASES = [
    pytest.param(
        "E128-T512",
        [512 * 8 // 128] * 128, 2048, 2048,
        id="E128-T512"),
    pytest.param(
        "E128-T2048",
        [2048 * 8 // 128] * 128, 2048, 2048,
        id="E128-T2048"),
    pytest.param(
        "E256-T512",
        [512 * 8 // 256] * 256, 2048, 2048,
        id="E256-T512"),
    pytest.param(
        "E256-T2048",
        [2048 * 8 // 256] * 256, 2048, 2048,
        id="E256-T2048"),
    pytest.param(
        "E256-T4096",
        [4096 * 8 // 256] * 256, 2048, 2048,
        id="E256-T4096"),
    # Skewed: simulates non-uniform routing
    pytest.param(
        "E256-T2048-skewed",
        sorted([max(1, int(x)) for x in
                torch.distributions.Exponential(1 / (2048 * 8 / 256)).sample((256,)).tolist()],
               reverse=True)[:256],
        2048, 2048,
        id="E256-T2048-skewed"),
]


@pytest.mark.parametrize("label, true_batch_sizes, N, K", _CASES)
def test_grouped_gemm_block_m(label, true_batch_sizes, N, K):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    E = len(true_batch_sizes)
    true_sum = sum(true_batch_sizes)
    flops = 2.0 * true_sum * N * K  # useful FLOPs only

    results = {}

    for bm in _BLOCK_M_VARIANTS:
        pad_sum = _padded_batch_sum(true_batch_sizes, bm)
        padding_pct = (pad_sum - true_sum) / pad_sum * 100

        A, B, sizes, offsets, pad_offsets = _build_inputs(
            true_batch_sizes, N, K, dtype, device, bm)

        kernel = grouped_gemm_kernel(
            batch_sum=pad_sum,
            batch_count=E,
            N=N,
            K=K,
            dtype=dtype,
            transpose_a=False,
            transpose_b=True,
            config={"block_m": bm, "block_n": 256, "block_k": 64,
                    "num_stages": 2, "threads": 128},
        )

        # Warmup / JIT compile
        kernel(A, B, sizes, offsets, pad_offsets)
        torch.cuda.synchronize()

        ms = _bench(lambda k=kernel, a=A, b=B, s=sizes, o=offsets, po=pad_offsets:
                    k(a, b, s, o, po))

        tflops = flops / ms * 1e-9
        results[bm] = (ms, tflops, padding_pct)


    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {label}  (E={E}, true_sum={true_sum}, N={N}, K={K}, bf16)")
    print(f"{'─'*65}")
    print(f"  {'block_m':>8}  {'pad%':>6}  {'latency':>10}  {'TFLOPS':>8}  {'vs bm=64':>9}")
    base_ms = results[64][0]
    for bm in _BLOCK_M_VARIANTS:
        ms, tflops, pct = results[bm]
        speedup = base_ms / ms
        marker = " <--" if bm == min(results, key=lambda k: results[k][0]) else ""
        print(f"  {bm:>8}  {pct:>5.1f}%  {ms:>9.3f}ms  {tflops:>8.2f}  {speedup:>8.2f}x{marker}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
