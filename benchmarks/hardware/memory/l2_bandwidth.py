"""
L2 Cache Bandwidth Benchmark

Measures L2 cache behavior using repeated access patterns:
- Sequential reuse: each CTA repeatedly reads the same tile from global memory
- Random reuse: gather from random indices within the working set

The key insight: by repeatedly accessing the SAME working set many times,
small working sets will be served from L2 cache (high BW), while large
working sets will spill to HBM (lower BW). The transition reveals L2 capacity.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from utils import (
    achieved_pct,
    bench,
    calc_bandwidth_gbs,
    get_measured_peak_bw,
    get_theoretical_peaks,
    make_csv,
    make_row,
    print_env_header,
)

WARMUP = 100
REP = 200

WORKING_SET_BYTES = {
    "1MB": 1 * 1024 * 1024,
    "4MB": 4 * 1024 * 1024,
    "8MB": 8 * 1024 * 1024,
    "16MB": 16 * 1024 * 1024,
    "32MB": 32 * 1024 * 1024,
    "64MB": 64 * 1024 * 1024,
    "128MB": 128 * 1024 * 1024,
    "256MB": 256 * 1024 * 1024,
}

# Number of repeated reads over the same working set to amplify L2 effect
NUM_REPEATS = 20


def benchmark_l2_cache():
    """Run L2 cache bandwidth benchmark with repeated access patterns."""
    env = print_env_header()
    csv = make_csv("l2_cache")

    peaks = get_theoretical_peaks()
    theo_bw = peaks["hbm_bw_gbs"] if peaks else None
    measured_bw = get_measured_peak_bw()

    print(f"\nL2 cache size: {env.get('l2_cache_mb', '?')} MB")
    if theo_bw:
        print(f"Theoretical peak HBM BW: {theo_bw} GB/s")
    if measured_bw:
        print(f"Measured peak copy BW: {measured_bw:.2f} GB/s")
    print(f"Each working set read {NUM_REPEATS}x to warm L2 and measure effective BW.\n")

    # ---- Sequential Reuse ----
    print("[L2 Cache: Sequential Reuse]")
    print(f"{'Working Set':>12} {'Eff BW (GB/s)':>14} {'%Theo':>8} {'Note':>10}")
    print("-" * 50)

    prev_bw = None
    for label, ws_bytes in WORKING_SET_BYTES.items():
        n = ws_bytes // 4  # float32
        data = torch.randn(n, device="cuda", dtype=torch.float32)

        def repeated_read(_data=data):
            s = torch.tensor(0.0, device="cuda")
            for _ in range(NUM_REPEATS):
                s += _data.sum()
            return s

        lat = bench(repeated_read, warmup=WARMUP, rep=REP)
        total_bytes = ws_bytes * NUM_REPEATS
        bw = calc_bandwidth_gbs(total_bytes, lat)
        pct = achieved_pct(bw, theo_bw)
        pct_str = f"{pct:.1f}%" if pct else "N/A"

        note = ""
        if prev_bw and bw < prev_bw * 0.85:
            note = "<-- drop"
        prev_bw = bw

        print(f"{label:>12} {bw:>14.2f} {pct_str:>8} {note:>10}")
        csv.writerow(make_row(env, benchmark="l2_seq_reuse", backend="pytorch",
                              dtype="fp32", working_set_bytes=ws_bytes,
                              warmup=WARMUP, rep=REP,
                              latency_ms=lat, latency_us=lat * 1000,
                              bandwidth_gbs=bw, bandwidth_tbs=bw / 1000,
                              achieved_pct_of_peak=pct,
                              notes=f"seq reuse x{NUM_REPEATS} {note}".strip()))

        del data
        torch.cuda.empty_cache()

    print()

    # ---- Random Reuse (gather) ----
    print("[L2 Cache: Random Gather Reuse]")
    print(f"{'Working Set':>12} {'Eff BW (GB/s)':>14} {'%Theo':>8} {'Note':>10}")
    print("-" * 50)

    gather_count = 1024 * 1024  # 1M random accesses per iteration

    prev_bw = None
    for label, ws_bytes in WORKING_SET_BYTES.items():
        n = ws_bytes // 4
        data = torch.randn(n, device="cuda", dtype=torch.float32)
        indices = torch.randint(0, n, (gather_count,), device="cuda", dtype=torch.int64)

        def repeated_gather(_data=data, _idx=indices):
            s = torch.tensor(0.0, device="cuda")
            for _ in range(NUM_REPEATS):
                s += _data[_idx].sum()
            return s

        lat = bench(repeated_gather, warmup=WARMUP, rep=REP)
        total_bytes = gather_count * 4 * NUM_REPEATS
        bw = calc_bandwidth_gbs(total_bytes, lat)
        pct = achieved_pct(bw, theo_bw)
        pct_str = f"{pct:.1f}%" if pct else "N/A"

        note = ""
        if prev_bw and bw < prev_bw * 0.85:
            note = "<-- drop"
        prev_bw = bw

        print(f"{label:>12} {bw:>14.2f} {pct_str:>8} {note:>10}")
        csv.writerow(make_row(env, benchmark="l2_random_reuse", backend="pytorch",
                              dtype="fp32", working_set_bytes=ws_bytes,
                              warmup=WARMUP, rep=REP,
                              latency_ms=lat, latency_us=lat * 1000,
                              bandwidth_gbs=bw, bandwidth_tbs=bw / 1000,
                              achieved_pct_of_peak=pct,
                              notes=f"random gather x{NUM_REPEATS}, "
                                    f"{gather_count} accesses {note}".strip()))

        del data, indices
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
    benchmark_l2_cache()
