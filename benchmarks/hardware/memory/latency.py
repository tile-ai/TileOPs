"""
Latency Benchmark

Measures:
- Null kernel launch overhead
- torch.cuda.synchronize() overhead
- Small memcpy latency (4B, 4KB, 1MB)
- Pointer-chasing global-load latency via CUDA kernel (single-thread dependent load chain)
  Uses precompiled pointer_chase binary for strict serialization.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess

import torch
from utils import (
    bench,
    make_csv,
    make_row,
    print_env_header,
)

WARMUP = 200
REP = 500

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POINTER_CHASE_BIN = os.path.join(SCRIPT_DIR, "pointer_chase")


def _ensure_pointer_chase_compiled():
    """Compile pointer_chase.cu if binary doesn't exist."""
    if os.path.isfile(POINTER_CHASE_BIN):
        return True
    cu_file = os.path.join(SCRIPT_DIR, "pointer_chase.cu")
    if not os.path.isfile(cu_file):
        return False
    try:
        subprocess.run(
            ["nvcc", "-O0", "-Wno-deprecated-gpu-targets", "-o", POINTER_CHASE_BIN, cu_file],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return True
    except Exception:
        return False


def benchmark_latency():
    """Run all latency benchmarks."""
    env = print_env_header()
    csv = make_csv("latency")

    # 1. Null kernel launch
    print("\n[Kernel Launch Overhead]")
    print(f"{'Operation':>35} {'Latency (us)':>14}")
    print("-" * 53)

    x = torch.tensor([1.0], device="cuda")

    null_ops = [
        ("scalar add (x + 1)", lambda: x + 1.0),
        ("scalar mul (x * 2)", lambda: x * 2.0),
        ("scalar fma (x * 2 + 1)", lambda: x * 2.0 + 1.0),
        ("empty kernel (torch.zeros(1,cuda))", lambda: torch.zeros(1, device="cuda")),
    ]

    for name, fn in null_ops:
        lat = bench(fn, warmup=WARMUP, rep=REP)
        lat_us = lat * 1000
        print(f"{name:>35} {lat_us:>14.2f}")
        csv.writerow(make_row(env, benchmark="launch_overhead", backend="pytorch",
                              warmup=WARMUP, rep=REP,
                              latency_ms=lat, latency_us=lat_us,
                              notes=name))

    print()

    # 2. Synchronize overhead
    print("[Synchronization Overhead]")
    print(f"{'Operation':>35} {'Latency (us)':>14}")
    print("-" * 53)

    lat = bench(lambda: torch.cuda.synchronize(), warmup=WARMUP, rep=REP)
    lat_us = lat * 1000
    print(f"{'torch.cuda.synchronize()':>35} {lat_us:>14.2f}")
    csv.writerow(make_row(env, benchmark="sync_overhead", backend="pytorch",
                          warmup=WARMUP, rep=REP,
                          latency_ms=lat, latency_us=lat_us,
                          notes="torch.cuda.synchronize()"))

    event = torch.cuda.Event(enable_timing=True)

    def event_sync():
        event.record()
        event.synchronize()

    lat = bench(event_sync, warmup=WARMUP, rep=REP)
    lat_us = lat * 1000
    print(f"{'event record + sync':>35} {lat_us:>14.2f}")
    csv.writerow(make_row(env, benchmark="sync_overhead", backend="pytorch",
                          warmup=WARMUP, rep=REP,
                          latency_ms=lat, latency_us=lat_us,
                          notes="event record + sync"))

    print()

    # 3. Small memcpy latency
    print("[Small Memcpy Latency (H2D)]")
    print(f"{'Size':>35} {'Latency (us)':>14}")
    print("-" * 53)

    memcpy_sizes = [
        ("4 B", 1),
        ("4 KB", 1024),
        ("1 MB", 256 * 1024),
    ]

    for label, n_elements in memcpy_sizes:
        cpu_tensor = torch.randn(n_elements, dtype=torch.float32)
        gpu_tensor = torch.empty(n_elements, device="cuda", dtype=torch.float32)

        lat = bench(lambda _g=gpu_tensor, _c=cpu_tensor: _g.copy_(_c), warmup=WARMUP, rep=REP)
        lat_us = lat * 1000
        print(f"{label:>35} {lat_us:>14.2f}")
        csv.writerow(make_row(env, benchmark="memcpy_h2d", backend="pytorch",
                              size_bytes=n_elements * 4,
                              warmup=WARMUP, rep=REP,
                              latency_ms=lat, latency_us=lat_us,
                              notes=f"H2D {label}"))

        del cpu_tensor, gpu_tensor

    print()

    # 4. Pointer-chasing global load latency
    print("[Global Memory Latency -- Pointer Chase (single-thread, full-cycle)]")

    if not _ensure_pointer_chase_compiled():
        print("  SKIPPED: could not compile pointer_chase.cu (needs nvcc)")
        csv.writerow(make_row(env, benchmark="mem_latency_chase",
                              notes="SKIPPED: compilation failed"))
    else:
        chase_configs = [
            ("32 KB",    32,    0),
            ("128 KB",   128,   0),
            ("512 KB",   512,   0),
            ("1 MB",     1024,  0),
            ("4 MB",     4096,  0),
            ("16 MB",    16384, 4194304),
            ("64 MB",    65536, 8388608),
        ]

        print("  Single-thread CUDA kernel: next = __ldg(&data[next])")
        print("  Full cycle traversal (num_chases = n_elements)")
        print(f"{'Working Set':>35} {'ns/load':>10} {'Level':>8}")
        print("-" * 57)

        for label, ws_kb, max_chases in chase_configs:
            try:
                args = [POINTER_CHASE_BIN, str(ws_kb)]
                if max_chases > 0:
                    args.append(str(max_chases))
                result = subprocess.run(
                    args, capture_output=True, text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    print(f"{label:>35} {'ERROR':>10} {result.stderr[:40]}")
                    continue

                parts = result.stdout.strip().split(",")
                ns_per_load = float(parts[3])

                if ns_per_load < 50:
                    level = "L1"
                elif ns_per_load < 200:
                    level = "L2"
                else:
                    level = "HBM"

                print(f"{label:>35} {ns_per_load:>10.1f} {level:>8}")
                csv.writerow(make_row(env, benchmark="mem_latency_chase",
                                      backend="cuda_kernel",
                                      working_set_bytes=ws_kb * 1024,
                                      latency_us=ns_per_load / 1000,
                                      notes=f"{label} {ns_per_load:.1f}ns/load {level}"))

            except subprocess.TimeoutExpired:
                print(f"{label:>35} {'TIMEOUT':>10}")
            except Exception as e:
                print(f"{label:>35} {'ERROR':>10} {str(e)[:30]}")

        print()
        print("  Interpretation:")
        print("  - L1 (<50 ns): data fits in L1 cache (~192-256 KB per SM)")
        print("  - L2 (50-200 ns): L1 miss, L2 hit")
        print("  - HBM (>200 ns): L2 miss, served from HBM")

    print()
    csv.close()
    print(f"CSV saved: {csv.path}")


if __name__ == "__main__":
    benchmark_latency()
