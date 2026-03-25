"""
GEMM / Tensor Core Throughput Benchmark

Measures:
- cuBLAS GEMM TFLOPS (fp16, bf16)
- TileLang GEMM TFLOPS (fp16, bf16)
- Speedup comparison
- pct_of_cublas_peak: relative to best cuBLAS result per dtype
- pct_of_theoretical_tensor_peak: relative to theoretical tensor core TFLOPS
"""

import tilelang
import tilelang.language as T
import torch

from benchmarks.hardware.utils import (
    achieved_pct,
    bench,
    calc_tflops,
    get_theoretical_peaks,
    make_csv,
    make_row,
    print_env_header,
)

WARMUP = 100
REP = 200

GEMM_CASES = [
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    (8192, 28672, 8192),   # Llama FFN-like
    (4096, 16384, 4096),   # Wide FFN
]

DTYPES = [
    ("fp16", torch.float16, "float16"),
    ("bf16", torch.bfloat16, "bfloat16"),
]


def make_tilelang_gemm(M, N, K, dtype_str="float16"):
    """Create a TileLang GEMM kernel."""
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def gemm_kernel(block_m: int, block_n: int, block_k: int,
                    threads: int, num_stages: int,
                    enable_rasteration: bool):

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype_str),
            B: T.Tensor((K, N), dtype_str),
            C: T.Tensor((M, N), dtype_str),
        ):
            with T.Kernel(
                T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads
            ) as (bx, by):
                a_shared = T.alloc_shared((block_m, block_k), dtype_str)
                b_shared = T.alloc_shared((block_k, block_n), dtype_str)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                c_shared = T.alloc_shared((block_m, block_n), dtype_str)

                T.annotate_layout({
                    c_shared: tilelang.layout.make_swizzled_layout(c_shared),
                })
                T.use_swizzle(10, enable=enable_rasteration)
                T.clear(c_local)

                for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                    T.copy(A[by * block_m, k * block_k], a_shared)
                    T.copy(B[k * block_k, bx * block_n], b_shared)
                    T.gemm(a_shared, b_shared, c_local)

                T.copy(c_local, c_shared)
                T.copy(c_shared, C[by * block_m, bx * block_n])

        return main

    return gemm_kernel


def benchmark_gemm_throughput():
    """Run GEMM throughput benchmark."""
    env = print_env_header()
    csv = make_csv("gemm_throughput")

    peaks = get_theoretical_peaks()
    theo_tflops_map = {}
    if peaks:
        theo_tflops_map = {
            "fp16": peaks.get("fp16_tensor_tflops"),
            "bf16": peaks.get("bf16_tensor_tflops"),
        }
        print(f"\nTheoretical tensor core peaks: "
              f"FP16={theo_tflops_map.get('fp16')} TFLOPS, "
              f"BF16={theo_tflops_map.get('bf16')} TFLOPS")

    # First pass: measure cuBLAS for all cases to find per-dtype peak
    print("\n[Phase 1: cuBLAS baseline]")
    cublas_results = {}
    cublas_peak = {}

    for dtype_name, torch_dtype, _tl_dtype in DTYPES:
        cublas_peak[dtype_name] = 0.0
        for M, N, K in GEMM_CASES:
            flops = 2.0 * M * N * K
            shape_str = f"{M}x{N}x{K}"
            A = torch.randn(M, K, device="cuda", dtype=torch_dtype)
            B = torch.randn(K, N, device="cuda", dtype=torch_dtype)

            lat = bench(lambda _a=A, _b=B: torch.mm(_a, _b), warmup=WARMUP, rep=REP)
            tflops = calc_tflops(flops, lat)
            cublas_results[(dtype_name, shape_str)] = (lat, tflops)
            cublas_peak[dtype_name] = max(cublas_peak[dtype_name], tflops)

            del A, B
            torch.cuda.empty_cache()

        print(f"  {dtype_name} cuBLAS peak: {cublas_peak[dtype_name]:.2f} TFLOPS")

    # Second pass: measure TileLang and print full table
    print("\n[GEMM Throughput: cuBLAS vs TileLang]")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'Dtype':>5} "
          f"{'cBLAS TF':>9} {'%Theo':>7} "
          f"{'TL TF':>9} {'%cBLAS':>7} {'%Theo':>7} {'Speed':>7}")
    print("-" * 72)

    for dtype_name, torch_dtype, tl_dtype in DTYPES:
        theo_peak = theo_tflops_map.get(dtype_name)

        for M, N, K in GEMM_CASES:
            flops = 2.0 * M * N * K
            shape_str = f"{M}x{N}x{K}"

            cublas_lat, cublas_tflops = cublas_results[(dtype_name, shape_str)]
            cublas_pct_theo = achieved_pct(cublas_tflops, theo_peak)
            cublas_pct_str = f"{cublas_pct_theo:.1f}%" if cublas_pct_theo else "N/A"

            csv.writerow(make_row(env, benchmark="gemm", backend="cublas",
                                  dtype=dtype_name, shape=shape_str,
                                  size_bytes=(M * K + K * N + M * N) * torch_dtype.itemsize,
                                  warmup=WARMUP, rep=REP,
                                  latency_ms=cublas_lat, latency_us=cublas_lat * 1000,
                                  tflops=cublas_tflops,
                                  achieved_pct_of_peak=cublas_pct_theo,
                                  notes=f"torch.mm, theo_peak={theo_peak}TF"))

            # TileLang
            A = torch.randn(M, K, device="cuda", dtype=torch_dtype)
            B = torch.randn(K, N, device="cuda", dtype=torch_dtype)

            tl_tflops_str = "ERR"
            tl_pct_cublas_str = "N/A"
            tl_pct_theo_str = "N/A"
            speedup_str = "N/A"

            try:
                kernel_fn = make_tilelang_gemm(M, N, K, tl_dtype)
                kernel = kernel_fn(
                    block_m=128, block_n=128, block_k=64,
                    threads=128, num_stages=3, enable_rasteration=True,
                )
                tl_lat = bench(lambda _k=kernel, _a=A, _b=B: _k(_a, _b), warmup=WARMUP, rep=REP)
                tl_tflops = calc_tflops(flops, tl_lat)
                tl_tflops_str = f"{tl_tflops:.1f}"

                pct_cublas = achieved_pct(tl_tflops, cublas_peak[dtype_name])
                pct_theo = achieved_pct(tl_tflops, theo_peak)
                tl_pct_cublas_str = f"{pct_cublas:.1f}%" if pct_cublas else "N/A"
                tl_pct_theo_str = f"{pct_theo:.1f}%" if pct_theo else "N/A"

                speedup = tl_tflops / cublas_tflops if cublas_tflops > 0 else 0
                speedup_str = f"{speedup:.2f}x"

                csv.writerow(make_row(env, benchmark="gemm", backend="tilelang",
                                      dtype=dtype_name, shape=shape_str,
                                      size_bytes=(M * K + K * N + M * N) * torch_dtype.itemsize,
                                      warmup=WARMUP, rep=REP,
                                      latency_ms=tl_lat, latency_us=tl_lat * 1000,
                                      tflops=tl_tflops,
                                      achieved_pct_of_peak=pct_theo,
                                      notes=f"block=128x128x64 stages=3, "
                                            f"%cublas_peak={pct_cublas:.1f}"))
            except Exception as e:
                csv.writerow(make_row(env, benchmark="gemm", backend="tilelang",
                                      dtype=dtype_name, shape=shape_str,
                                      notes=f"ERROR: {str(e)[:60]}"))

            print(f"{M:>6} {N:>6} {K:>6} {dtype_name:>5} "
                  f"{cublas_tflops:>9.1f} {cublas_pct_str:>7} "
                  f"{tl_tflops_str:>9} {tl_pct_cublas_str:>7} {tl_pct_theo_str:>7} "
                  f"{speedup_str:>7}")

            del A, B
            torch.cuda.empty_cache()

        print()

    # Summary
    print("[Summary]")
    for dtype_name in ["fp16", "bf16"]:
        theo = theo_tflops_map.get(dtype_name)
        cpeak = cublas_peak.get(dtype_name, 0)
        print(f"  {dtype_name}: cuBLAS peak = {cpeak:.1f} TFLOPS"
              f" ({achieved_pct(cpeak, theo):.1f}% of theoretical {theo} TFLOPS)"
              if theo else f"  {dtype_name}: cuBLAS peak = {cpeak:.1f} TFLOPS")

    csv.close()
    print(f"\nCSV saved: {csv.path}")


if __name__ == "__main__":
    benchmark_gemm_throughput()
