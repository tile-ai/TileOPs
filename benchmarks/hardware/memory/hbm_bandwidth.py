"""HBM Bandwidth Benchmark.

Measures read, write, and copy bandwidth across sizes and dtypes.
Compares against theoretical peak from GPU profile (single source of truth).
"""

import torch

from benchmarks.hardware.utils import achieved_pct, bench, calc_bandwidth_gbs
from tileops.perf import load_profile

WARMUP = 100
REP = 200

SIZES_BYTES = {
    "1MB": 1 * 1024 * 1024,
    "8MB": 8 * 1024 * 1024,
    "64MB": 64 * 1024 * 1024,
    "256MB": 256 * 1024 * 1024,
    "1GB": 1024 * 1024 * 1024,
    "2GB": 2 * 1024 * 1024 * 1024,
}

DTYPES = [
    ("fp16", torch.float16, 2),
    ("bf16", torch.bfloat16, 2),
    ("fp32", torch.float32, 4),
]


def measure_peak_copy_bw():
    """Measure peak copy bandwidth at 2GB. Returns bandwidth in GB/s."""
    nbytes = 2 * 1024 * 1024 * 1024
    n = nbytes // 4
    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    latency = bench(lambda _d=dst, _s=src: _d.copy_(_s), warmup=WARMUP, rep=REP)
    bw = calc_bandwidth_gbs(2 * nbytes, latency)
    del src, dst
    torch.cuda.empty_cache()
    return bw


def benchmark_bandwidth(profile_name="h200"):
    """Run full HBM bandwidth benchmark. Returns measured peak BW in GB/s."""
    profile = load_profile(profile_name)
    theo_bw = profile["hbm"]["theoretical"] / 1e9  # bytes/s → GB/s

    measured_bw = measure_peak_copy_bw()

    print(f"\nTheoretical peak HBM BW: {theo_bw} GB/s" if theo_bw else "\nTheoretical peak: unknown")
    print(f"Measured peak copy BW (2GB fp32): {measured_bw:.2f} GB/s")
    if theo_bw:
        print(f"Calibration factor: {measured_bw / theo_bw:.4f}")
    print()

    peak_bw = theo_bw or measured_bw

    print("[HBM Bandwidth: Read / Write / Copy]")
    print(f"{'Op':>6} {'Dtype':>5} {'Size':>6} {'BW (GB/s)':>12} {'%Theo':>8} {'%Meas':>8}")
    print("-" * 50)

    for dtype_name, dtype, itemsize in DTYPES:
        for size_label, size_bytes in SIZES_BYTES.items():
            n = size_bytes // itemsize
            actual_bytes = n * itemsize

            src = torch.randn(n, device="cuda", dtype=dtype)
            dst = torch.empty_like(src)

            for op_name, op_fn, bw_bytes in [
                ("read", lambda _s=src: _s.sum(), actual_bytes),
                ("write", lambda _d=dst: _d.zero_(), actual_bytes),
                ("copy", lambda _d=dst, _s=src: _d.copy_(_s), 2 * actual_bytes),
            ]:
                lat = bench(op_fn, warmup=WARMUP, rep=REP)
                bw = calc_bandwidth_gbs(bw_bytes, lat)
                pct_theo = achieved_pct(bw, theo_bw)
                pct_meas = achieved_pct(bw, measured_bw)
                pct_theo_str = f"{pct_theo:.1f}%" if pct_theo else "N/A"
                pct_meas_str = f"{pct_meas:.1f}%" if pct_meas else "N/A"

                print(f"{op_name:>6} {dtype_name:>5} {size_label:>6} "
                      f"{bw:>12.2f} {pct_theo_str:>8} {pct_meas_str:>8}")

            del src, dst
            torch.cuda.empty_cache()

        print()

    print("\n[HBM Strided Read Bandwidth]")
    print(f"{'Dtype':>5} {'Stride(B)':>10} {'BW (GB/s)':>12} {'%Theo':>8}")
    print("-" * 40)

    strides_bytes = [32, 128, 512]
    base_size = 256 * 1024 * 1024

    for dtype_name, dtype, itemsize in DTYPES:
        for stride_bytes in strides_bytes:
            stride_elements = max(stride_bytes // itemsize, 1)
            n_total = base_size // itemsize
            data = torch.randn(n_total, device="cuda", dtype=dtype)
            strided = data[::stride_elements]
            accessed_bytes = strided.numel() * itemsize

            lat = bench(lambda _s=strided: _s.sum(), warmup=WARMUP, rep=REP)
            bw = calc_bandwidth_gbs(accessed_bytes, lat)
            pct = achieved_pct(bw, peak_bw)
            pct_str = f"{pct:.1f}%" if pct else "N/A"
            print(f"{dtype_name:>5} {stride_bytes:>10} {bw:>12.2f} {pct_str:>8}")

            del data, strided
            torch.cuda.empty_cache()

        print()

    return measured_bw


if __name__ == "__main__":
    benchmark_bandwidth()
