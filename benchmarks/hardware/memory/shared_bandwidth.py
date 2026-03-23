"""
Shared Memory Bandwidth Benchmark

Measures:
- Shared memory throughput via global->smem->global copy kernels
- Various tile sizes, dtypes, and vector widths
- Bank conflict / occupancy effects visible through bandwidth variation
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tilelang
import tilelang.language as T
import torch
from utils import (
    achieved_pct,
    bench,
    calc_bandwidth_gbs,
    get_measured_peak_bw,
    make_csv,
    make_row,
    print_env_header,
)

WARMUP = 100
REP = 200

# Total data size for throughput measurement
TOTAL_ELEMENTS_FP16 = 16 * 1024 * 1024  # 16M elements = 32 MB
TOTAL_ELEMENTS_FP32 = 8 * 1024 * 1024   # 8M elements = 32 MB

DTYPES = [
    ("fp16", torch.float16, "float16", 2, TOTAL_ELEMENTS_FP16),
    ("fp32", torch.float32, "float", 4, TOTAL_ELEMENTS_FP32),
]

# tile_bytes_per_cta: 4KB, 8KB, 16KB, 32KB, 64KB
TILE_CONFIGS_FP16 = [
    (4096, 2048),
    (8192, 4096),
    (16384, 8192),
    (32768, 16384),
    (65536, 32768),
]

TILE_CONFIGS_FP32 = [
    (4096, 1024),
    (8192, 2048),
    (16384, 4096),
    (32768, 8192),
    (65536, 16384),
]


def make_smem_copy_kernel_2d(N_rows, N_cols, tile_rows, tile_cols, dtype_str, threads=256):
    """
    2D shared memory copy kernel: global -> shared -> global.
    Uses 2D tiles to better reflect real kernel patterns.
    """

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3"])
    def kernel():

        @T.prim_func
        def main(
            src: T.Tensor((N_rows, N_cols), dtype_str),
            dst: T.Tensor((N_rows, N_cols), dtype_str),
        ):
            num_tiles_x = T.ceildiv(N_cols, tile_cols)
            num_tiles_y = T.ceildiv(N_rows, tile_rows)
            with T.Kernel(num_tiles_x, num_tiles_y, threads=threads) as (bx, by):
                smem = T.alloc_shared((tile_rows, tile_cols), dtype_str)
                T.copy(src[by * tile_rows, bx * tile_cols], smem)
                T.copy(smem, dst[by * tile_rows, bx * tile_cols])

        return main

    return kernel


def benchmark_shared_memory():
    """Run shared memory bandwidth benchmark."""
    env = print_env_header()
    csv = make_csv("shared_memory")
    peak_bw = get_measured_peak_bw()

    print("\n[Shared Memory Bandwidth: Global -> SMEM -> Global]")
    print(f"{'Dtype':>5} {'Tile(B)':>8} {'Tile Shape':>14} {'Threads':>8} "
          f"{'BW (GB/s)':>12} {'Achieved%':>10} {'Notes':>20}")
    print("-" * 85)

    for dtype_name, torch_dtype, tl_dtype, itemsize, total_n in DTYPES:
        tile_configs = TILE_CONFIGS_FP16 if itemsize == 2 else TILE_CONFIGS_FP32

        for tile_bytes, tile_elements in tile_configs:
            tile_cols = 128 if itemsize == 2 else 64
            tile_rows = tile_elements // tile_cols

            total_cols = tile_cols * 16
            total_rows = total_n // total_cols

            total_rows = (total_rows // tile_rows) * tile_rows
            total_cols = (total_cols // tile_cols) * tile_cols
            actual_bytes = total_rows * total_cols * itemsize

            for threads in [128, 256]:
                src = torch.randn(total_rows, total_cols, device="cuda", dtype=torch_dtype)
                dst = torch.empty_like(src)

                try:
                    kernel_fn = make_smem_copy_kernel_2d(
                        total_rows, total_cols, tile_rows, tile_cols, tl_dtype, threads)
                    kernel = kernel_fn()

                    lat = bench(lambda _k=kernel, _s=src: _k(_s), warmup=WARMUP, rep=REP)
                    bw = calc_bandwidth_gbs(2 * actual_bytes, lat)
                    pct = achieved_pct(bw, peak_bw)
                    pct_str = f"{pct:.1f}%" if pct else "N/A"

                    num_warps = threads // 32
                    tile_shape = f"{tile_rows}x{tile_cols}"
                    note = f"{num_warps}warps"

                    print(f"{dtype_name:>5} {tile_bytes:>8} {tile_shape:>14} {threads:>8} "
                          f"{bw:>12.2f} {pct_str:>10} {note:>20}")

                    csv.writerow(make_row(env, benchmark="shared_memory", backend="tilelang",
                                          dtype=dtype_name, shape=tile_shape,
                                          size_bytes=actual_bytes,
                                          block_dim=threads, num_warps=num_warps,
                                          warmup=WARMUP, rep=REP,
                                          latency_ms=lat, latency_us=lat * 1000,
                                          bandwidth_gbs=bw, bandwidth_tbs=bw / 1000,
                                          achieved_pct_of_peak=pct,
                                          notes=f"tile={tile_bytes}B {note}"))
                except Exception as e:
                    print(f"{dtype_name:>5} {tile_bytes:>8} {'N/A':>14} {threads:>8} "
                          f"{'ERROR':>12} {'':>10} {str(e)[:20]:>20}")

                del src, dst
                torch.cuda.empty_cache()

        print()

    csv.close()
    print(f"CSV saved: {csv.path}")


if __name__ == "__main__":
    if get_measured_peak_bw() is None:
        from utils import set_measured_peak_bw

        from benchmarks.hardware.memory.hbm_bandwidth import _measure_peak_copy_bw
        peak = _measure_peak_copy_bw()
        set_measured_peak_bw(peak)
    benchmark_shared_memory()
