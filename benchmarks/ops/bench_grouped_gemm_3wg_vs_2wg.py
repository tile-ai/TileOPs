"""Benchmark GroupedGemmPersistent3WGKernel vs torch._grouped_mm (CUTLASS-backed).

torch._grouped_mm (PyTorch 2.10+) dispatches to CUTLASS grouped GEMM kernels
internally, making it the strongest available baseline for Hopper grouped GEMM.Cases:
  Qwen3-235B-decode   E=128 T=512
  Qwen3-235B-prefill  E=128 T=4096
  DeepSeek-V3-decode  E=256 T=512
  DeepSeek-V3-prefill E=256 T=4096
× {uniform, skewed} distribution = 8 cases.

Usage:
    TILELANG_CLEANUP_TEMP_FILES=1 TMPDIR=/tmp/jieneng_tvm_tmp \\
    conda run -n tileops-tl9 --no-capture-output \\
    python benchmarks/ops/bench_grouped_gemm_3wg_vs_2wg.py
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


def pytorch_grouped_gemm(A, B_KN, offs_cumsum, C_buf):
    """torch._grouped_mm baseline (CUTLASS-backed grouped GEMM in PyTorch 2.10+).

    Args:
        A: [numel, K] bf16
        B_KN: [E, K, N] bf16 (pre-transposed from our NT layout [E, N, K])
        offs_cumsum: [E] int32 cumulative offsets (no leading 0)
        C_buf: unused, kept for API compat
    """
    return torch._grouped_mm(A, B_KN, offs_cumsum)


def tflops(ms, numel, N, K):
    return 2.0 * numel * N * K / ms * 1e-9


def run_case(T, E, K_top, N, K, label):
    numel = T * K_top
    flops = 2.0 * numel * N * K
    sm = torch.cuda.get_device_properties(0).multi_processor_count

    print(f"\n{'═' * 88}")
    print(f"  {label}")
    print(f"  Grouped GEMM: M_total={numel}, N={N}, K={K}, groups={E}, bf16")
    print(f"  (from MoE: T={T}, top_k={K_top}, E={E}; sm_count={sm})")
    print(f"  FLOPs = {flops / 1e12:.3f} TFLOPs")
    print(f"{'═' * 88}")

    for dist in ("uniform", "skewed"):
        A, B, sizes, offsets, _ = gen_inputs(T, E, K_top, N, K, dist)
        sz_summary = (
            f"M_per_group: min={sizes.min().item()}, "
            f"max={sizes.max().item()}, "
            f"mean={sizes.float().mean().item():.0f}")
        print(f"\n  distribution={dist:<8}  ({sz_summary})")
        print(f"  {'impl':>18}  {'ms':>8}  {'TFLOPS':>8}  {'vs torch':>10}")
        print(f"  {'-' * 18}  {'-' * 8}  {'-' * 8}  {'-' * 10}")

        k3 = GroupedGemmPersistent3WGKernel(numel=numel, num_experts=E, N=N, K=K,
                                            dtype=_DTYPE, sm_count=sm)

        # Prepare inputs for torch._grouped_mm: B must be [E, K, N] (NN layout)
        # and offs is cumulative sizes without leading 0.
        B_KN = B.transpose(1, 2).contiguous()
        offs_cumsum = torch.cumsum(sizes, dim=0).to(torch.int32)

        C_3wg = k3(A, B, sizes, offsets)
        torch.cuda.synchronize()
        C_pt = pytorch_grouped_gemm(A, B_KN, offs_cumsum, None)
        torch.cuda.synchronize()
        md = (C_pt - C_3wg).abs().max().item()

        t_pt = do_bench(
            lambda: pytorch_grouped_gemm(A, B_KN, offs_cumsum, None),  # noqa: B023
            warmup=_WARMUP, rep=_REP)
        t_3wg = do_bench(lambda: k3(A, B, sizes, offsets), warmup=_WARMUP, rep=_REP)  # noqa: B023

        spd = f"{t_pt / t_3wg:.2f}x"
        print(f"  {'torch._grouped_mm':>18}  {t_pt:>8.3f}  {tflops(t_pt, numel, N, K):>8.2f}  {'1.00x':>10}")
        print(f"  {'3-WG':>18}  {t_3wg:>8.3f}  {tflops(t_3wg, numel, N, K):>8.2f}  {spd:>10}")
        print(f"  max_diff (torch vs 3-WG): {md:.2e}")


def main():
    cases = [
        # T,    E,   top_k, N,     K,    label
        (512,  128,  8,  4096, 7168, "Qwen3-235B-decode   E=128 T=512"),
        (4096, 128,  8,  4096, 7168, "Qwen3-235B-prefill  E=128 T=4096"),
        (512,  256,  8,  4096, 7168, "DeepSeek-V3-decode  E=256 T=512"),
        (4096, 256,  8,  4096, 7168, "DeepSeek-V3-prefill E=256 T=4096"),
    ]
    for T, E, K_top, N, K, label in cases:
        run_case(T, E, K_top, N, K, label)


if __name__ == "__main__":
    main()
