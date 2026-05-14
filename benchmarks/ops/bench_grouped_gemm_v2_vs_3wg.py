"""Benchmark GroupedGemmPersistentV2Kernel vs GroupedGemmPersistent3WGKernel.

Phase-2 V2 (pingpong + static-wave scheduler) sanity check: confirm the
new kernel is within +/-5% of the 3-WG kernel at block_m=64 on the
default Qwen3-235B-prefill-style shape, on uniform and skewed routing.
Reports ``torch._grouped_mm`` (CUTLASS-backed) as the non-tileops baseline.

Shape: T=4096, E=128, top_k=8, N=4096, K=2048 (Qwen3-235B prefill, MoE
up-projection w13). Tiles: block_m=64 (pingpong), block_n=256, block_k=64.

Usage:
    TMPDIR=/tmp/tilelang_v2_cache_$USER \\
    conda run -n tileops-tl9 --no-capture-output \\
    python benchmarks/ops/bench_grouped_gemm_v2_vs_3wg.py
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from tilelang.profiler import do_bench  # noqa: E402

from tileops.kernels.grouped_gemm import (  # noqa: E402
    GroupedGemmPersistent3WGKernel,
    GroupedGemmPersistentV2Kernel,
)

_DTYPE = torch.bfloat16
_WARMUP = 25
_REP = 100


def gen_inputs(T, E, K_top, N, K_hidden, distribution):
    torch.manual_seed(42)
    dev = "cuda"
    numel = T * K_top

    if distribution == "uniform":
        per = numel // E
        sizes = torch.full((E,), per, dtype=torch.int32, device=dev)
        sizes[:numel % E] += 1
    elif distribution == "skewed":
        top = max(1, E // 5)
        per_top = (4 * numel) // (5 * top)
        sizes = torch.full((E,), 1, dtype=torch.int32, device=dev)
        sizes[:top] = per_top
        rem = numel - sizes.sum().item()
        sizes[0] += rem
    else:
        raise ValueError(distribution)
    offsets = torch.zeros(E, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)

    A = torch.randn(int(sizes.sum().item()), K_hidden, dtype=_DTYPE, device=dev) * 0.02
    B = torch.randn(E, N, K_hidden, dtype=_DTYPE, device=dev) * 0.02
    return A, B, sizes, offsets, numel


def pytorch_grouped_gemm(A, B_KN, offs_cumsum):
    """torch._grouped_mm baseline (CUTLASS-backed grouped GEMM in PyTorch 2.10+).

    Expects A: [numel, K], B_KN: [E, K, N] (pre-transposed from NT layout
    [E, N, K]), offs: [E] int32 cumulative sizes (no leading 0).
    """
    return torch._grouped_mm(A, B_KN, offs_cumsum)


def tflops(ms, numel, N, K):
    return 2.0 * numel * N * K / ms * 1e-9


def run_case(T, E, K_top, N, K, label):
    numel = T * K_top
    flops = 2.0 * numel * N * K
    sm = torch.cuda.get_device_properties(0).multi_processor_count

    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"  Grouped GEMM: M_total={numel}, N={N}, K={K}, groups={E}, bf16")
    print(f"  (T={T}, top_k={K_top}, E={E}; sm_count={sm})")
    print(f"  FLOPs = {flops / 1e12:.3f} TFLOPs")
    print(f"{'=' * 100}")

    k3 = GroupedGemmPersistent3WGKernel(numel=numel, num_experts=E, N=N, K=K,
                                        dtype=_DTYPE, sm_count=sm)
    kv2 = GroupedGemmPersistentV2Kernel(numel=numel, num_experts=E, N=N, K=K,
                                        dtype=_DTYPE, sm_count=sm)

    header = (f"  {'distribution':<10}  {'3WG ms':>8}  {'V2 ms':>8}  {'torch ms':>9}  "
              f"{'V2/torch':>9}  {'V2/3WG':>8}  {'TFLOPs (3WG/V2/torch)':>24}  {'max_diff':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for dist in ("uniform", "skewed"):
        A, B, sizes, offsets, _ = gen_inputs(T, E, K_top, N, K, dist)

        # Prepare torch._grouped_mm inputs: B in [E, K, N] layout, cumulative offsets.
        B_KN = B.transpose(1, 2).contiguous()
        offs_cumsum = torch.cumsum(sizes, dim=0).to(torch.int32)

        C_3wg = k3(A, B, sizes, offsets)
        C_v2 = kv2(A, B, sizes, offsets)
        torch.cuda.synchronize()
        md = (C_3wg - C_v2).abs().max().item()

        t_3wg = do_bench(lambda: k3(A, B, sizes, offsets),  # noqa: B023
                         warmup=_WARMUP, rep=_REP)
        t_v2 = do_bench(lambda: kv2(A, B, sizes, offsets),  # noqa: B023
                        warmup=_WARMUP, rep=_REP)
        t_pt = do_bench(
            lambda: pytorch_grouped_gemm(A, B_KN, offs_cumsum),  # noqa: B023
            warmup=_WARMUP, rep=_REP)

        tf3 = tflops(t_3wg, numel, N, K)
        tfv2 = tflops(t_v2, numel, N, K)
        tfpt = tflops(t_pt, numel, N, K)
        tf_str = f"{tf3:.1f}/{tfv2:.1f}/{tfpt:.1f}"
        print(f"  {dist:<10}  {t_3wg:>8.3f}  {t_v2:>8.3f}  {t_pt:>9.3f}  "
              f"{t_v2/t_pt:>9.3f}  {t_v2/t_3wg:>8.3f}  {tf_str:>24}  {md:>10.2e}")


def main():
    cases = [
        # T,    E,   top_k, N,     K,    label
        (4096, 128,  8,  4096, 2048, "V2 vs 3WG default  T=4096 E=128 N=4096 K=2048"),
    ]
    for T, E, K_top, N, K, label in cases:
        run_case(T, E, K_top, N, K, label)


if __name__ == "__main__":
    main()
