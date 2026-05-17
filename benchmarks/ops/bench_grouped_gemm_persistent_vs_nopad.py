"""Compare GroupedGemmPersistentKernel vs nopad vs padded reference.

Tests the three implementations across:
  - Uniform distribution (each expert gets equal tokens)
  - Skewed distribution (80% tokens to top-20% experts)
  - Both decode (T=512) and prefill (T=4096) shapes
  - Qwen3-235B (E=128) and DeepSeek-V3 (E=256)
"""
import math

import torch
import torch.nn.functional as F

from tileops.kernels.grouped_gemm import GroupedGemmKernel, GroupedGemmPersistentKernel
from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel

_WARMUP = 50
_ITERS = 200
_DTYPE = torch.bfloat16


def bench(fn, warmup=_WARMUP, iters=_ITERS) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def tflops(ms, numel, N, K) -> float:
    return 2.0 * numel * N * K / ms * 1e-9


def gen_inputs(T, E, K_top, N, K_hidden, distribution):
    """Generate (A, B, sizes, offsets, numel) at the given distribution.

    Args:
        distribution: "uniform" — each expert gets numel/E tokens (last gets remainder).
                      "skewed"  — top-20% experts each take 4× the average; remaining
                                  experts share the rest evenly.  Roughly 80/20.
    """
    torch.manual_seed(42)
    dev = "cuda"
    numel = T * K_top

    if distribution == "uniform":
        per = max(1, numel // E)
        sizes = torch.full((E,), per, dtype=torch.int32, device=dev)
        sizes[-1] = numel - per * (E - 1)
    elif distribution == "skewed":
        top = max(1, E // 5)  # top 20% experts
        per_top = (numel * 4 // 5) // top  # they take 80%
        per_rest = max(1, (numel - per_top * top) // (E - top))
        sizes = torch.zeros(E, dtype=torch.int32, device=dev)
        sizes[:top] = per_top
        sizes[top:] = per_rest
        sizes[0] += numel - sizes.sum().item()
    else:
        raise ValueError(distribution)

    offsets = torch.zeros(E, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    A = torch.randn(numel, K_hidden, dtype=_DTYPE, device=dev) * 0.02
    B = torch.randn(E, N, K_hidden, dtype=_DTYPE, device=dev) * 0.02
    return A, B, sizes, offsets, numel


def bench_nopad(numel, E, N, K, A, B, sizes, offsets):
    kernel = MoeGroupedGemmNopadKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=_DTYPE)
    kernel(A, B, sizes, offsets)  # JIT compile + warm
    torch.cuda.synchronize()
    return bench(lambda: kernel(A, B, sizes, offsets))


def bench_persistent(numel, E, N, K, A, B, sizes, offsets):
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=_DTYPE, sm_count=sm)
    kernel(A, B, sizes, offsets)
    torch.cuda.synchronize()
    return bench(lambda: kernel(A, B, sizes, offsets))


def bench_padded(numel, E, N, K, A, B, sizes, offsets, block_m=64):
    """Padded reference: block_m-aligned per-expert slabs."""
    sizes_py = sizes.tolist()
    padded_batch_sum = sum(math.ceil(s / block_m) * block_m for s in sizes_py)
    pad_offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
    cur = 0
    for i, s in enumerate(sizes_py):
        pad_offsets[i] = cur
        cur += math.ceil(s / block_m) * block_m
    A_pad = torch.zeros(padded_batch_sum, K, dtype=_DTYPE, device="cuda")
    for s, so, po in zip(sizes_py, offsets.tolist(), pad_offsets.tolist(), strict=True):
        A_pad[po:po + s] = A[so:so + s]
    kernel = GroupedGemmKernel(
        batch_sum=padded_batch_sum, batch_count=E, N=N, K=K,
        dtype=_DTYPE, transpose_a=False, transpose_b=True)
    kernel(A_pad, B, sizes, pad_offsets, pad_offsets)
    torch.cuda.synchronize()
    return bench(lambda: kernel(A_pad, B, sizes, pad_offsets, pad_offsets))


def run_case(T, E, K_top, N, K, label):
    numel = T * K_top
    flops = 2.0 * numel * N * K

    print(f"\n{'═' * 88}")
    print(f"  {label}")
    print(f"  T={T}, E={E}, top_k={K_top}, numel={numel}, N={N}, K={K}, bf16")
    print(f"  FLOPs = {flops / 1e12:.3f} TFLOPs (pure GEMM, no permute/unpermute)")
    print(f"{'═' * 88}")

    for dist in ("uniform", "skewed"):
        A, B, sizes, offsets, _ = gen_inputs(T, E, K_top, N, K, dist)
        sizes_summary = (
            f"min={sizes.min().item()}, max={sizes.max().item()}, "
            f"mean={sizes.float().mean().item():.0f}")
        print(f"\n  distribution={dist:<8}  ({sizes_summary})")
        print(f"  {'impl':>18}  {'ms':>8}  {'TFLOPS':>8}  {'vs padded':>10}  {'vs nopad':>10}")
        print(f"  {'-' * 18}  {'-' * 8}  {'-' * 8}  {'-' * 10}  {'-' * 10}")

        results = {}
        try:
            results["padded (ref)"] = bench_padded(numel, E, N, K, A, B, sizes, offsets)
        except Exception as ex:
            print(f"  padded FAILED: {ex}")
        try:
            results["nopad"] = bench_nopad(numel, E, N, K, A, B, sizes, offsets)
        except Exception as ex:
            print(f"  nopad FAILED: {ex}")
        try:
            results["persistent"] = bench_persistent(numel, E, N, K, A, B, sizes, offsets)
        except Exception as ex:
            print(f"  persistent FAILED: {ex}")

        ms_padded = results.get("padded (ref)")
        ms_nopad = results.get("nopad")
        for impl, ms in results.items():
            tf = tflops(ms, numel, N, K)
            spd_pad = f"{ms_padded / ms:.2f}x" if ms_padded else "n/a"
            spd_nop = f"{ms_nopad / ms:.2f}x" if ms_nopad else "n/a"
            print(f"  {impl:>18}  {ms:>8.3f}  {tf:>8.2f}  {spd_pad:>10}  {spd_nop:>10}")


if __name__ == "__main__":
    cases = [
        # T,    E,   top_k, N,     K,    label
        (512,  128,  8,  4096, 7168, "Qwen3-235B-decode   E=128 T=512"),
        (4096, 128,  8,  4096, 7168, "Qwen3-235B-prefill  E=128 T=4096"),
        (512,  256,  8,  4096, 7168, "DeepSeek-V3-decode  E=256 T=512"),
        (4096, 256,  8,  4096, 7168, "DeepSeek-V3-prefill E=256 T=4096"),
    ]
    for T, E, K_top, N, K, label in cases:
        run_case(T, E, K_top, N, K, label)

    print(f"\n{'═' * 88}")
    print("  Done.")
