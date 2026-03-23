#!/usr/bin/env python3
"""Stream / Event / Host-Device Sync Overhead Cookbook

Measures:
1. cudaEventRecord + cudaStreamWaitEvent cost
2. Host-side cudaStreamSynchronize / torch.cuda.synchronize cost
3. Multi-stream dependency overhead vs overlap benefit
4. When sync overhead breaks the pipeline

Method: torch.cuda events for device timing, time.perf_counter for host timing.
All measurements: warmup + N repeats, report median.
"""

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

N_REPEATS = 50
WARMUP = 10


def host_timer(fn, n=N_REPEATS, warmup=WARMUP):
    """Measure host-side wall time of fn() in microseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # us
    return statistics.median(times)


def device_timer(fn, n=N_REPEATS, warmup=WARMUP):
    """Measure device time of fn(stream) using CUDA events, in microseconds."""
    stream = torch.cuda.Stream()
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            fn(stream)
    torch.cuda.synchronize()

    times = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with torch.cuda.stream(stream):
            fn(stream)
        end.record(stream)
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # ms -> us
    return statistics.median(times)


def main():
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name} | SMs: {prop.multi_processor_count}")
    print(f"N_REPEATS = {N_REPEATS}, WARMUP = {WARMUP}\n")

    # Pre-allocate tensors for compute kernels
    SIZE = 4096
    A = torch.randn(SIZE, SIZE, device='cuda', dtype=torch.float16)
    B = torch.randn(SIZE, SIZE, device='cuda', dtype=torch.float16)
    small = torch.randn(64, device='cuda')

    # ==============================================================
    # 1. Host-side sync costs
    # ==============================================================
    print("=== Host-side sync costs (host wall time) ===")
    print("operation,us")

    # torch.cuda.synchronize() on idle GPU
    t = host_timer(lambda: torch.cuda.synchronize())
    print(f"torch.cuda.synchronize (idle),{t:.2f}")

    # torch.cuda.synchronize() after trivial kernel
    def sync_after_trivial():
        small + small  # trivial kernel
        torch.cuda.synchronize()
    t = host_timer(sync_after_trivial)
    print(f"torch.cuda.synchronize (after trivial kernel),{t:.2f}")

    # cudaStreamSynchronize on default stream (idle)
    s = torch.cuda.Stream()
    t = host_timer(lambda: s.synchronize())
    print(f"stream.synchronize (idle),{t:.2f}")

    # cudaStreamSynchronize after trivial kernel on that stream
    def sync_stream_trivial():
        with torch.cuda.stream(s):
            small + small
        s.synchronize()
    t = host_timer(sync_stream_trivial)
    print(f"stream.synchronize (after trivial kernel),{t:.2f}")

    print()

    # ==============================================================
    # 2. Event record + wait cost
    # ==============================================================
    print("=== Event record + stream wait cost ===")
    print("operation,us")

    # Cost of event.record() alone (host side)
    evt = torch.cuda.Event(enable_timing=True)
    t = host_timer(lambda: evt.record())
    print(f"event.record (host cost),{t:.2f}")

    # Cost of event.record() + event.wait() on another stream
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    evt2 = torch.cuda.Event()

    def event_record_wait():
        evt2.record(s1)
        evt2.wait(s2)  # s2 waits for s1's event
    t = host_timer(event_record_wait)
    print(f"event.record + event.wait (host cost),{t:.2f}")

    # Device-side cost: how much latency does event dependency add?
    # Compare: s2 runs kernel directly vs s2 waits for s1's event then runs kernel
    def kernel_direct(stream):
        small + small

    def kernel_after_event_wait(stream):
        e = torch.cuda.Event()
        e.record(torch.cuda.current_stream())
        e.wait(stream)
        small + small

    t_direct = device_timer(kernel_direct)
    t_wait = device_timer(kernel_after_event_wait)
    print(f"trivial kernel direct (device),{t_direct:.2f}")
    print(f"trivial kernel after event.wait (device),{t_wait:.2f}")
    print(f"event dependency overhead (device),{t_wait - t_direct:.2f}")

    print()

    # ==============================================================
    # 3. Multi-stream overlap vs dependency overhead
    # ==============================================================
    print("=== Multi-stream overlap effectiveness ===")
    print("scenario,us,speedup_vs_serial")

    # Heavy kernel: GEMM
    def run_gemm(stream):
        with torch.cuda.stream(stream):
            torch.mm(A, B)

    # Serial: N GEMMs on one stream
    for n_kernels in [2, 4, 8]:
        torch.cuda.Stream()  # ensure stream exists
        def serial(stream, nk=n_kernels):
            for _ in range(nk):
                torch.mm(A, B)
        t_serial = device_timer(serial)

        # Parallel: N GEMMs on N streams
        streams = [torch.cuda.Stream() for _ in range(n_kernels)]
        def parallel(stream, ss=streams, nk=n_kernels):
            for i in range(nk):
                with torch.cuda.stream(ss[i]):
                    torch.mm(A, B)
        t_parallel = device_timer(parallel)

        speedup = t_serial / t_parallel if t_parallel > 0 else 0
        print(f"{n_kernels}x GEMM serial,{t_serial:.1f},1.00x")
        print(f"{n_kernels}x GEMM parallel ({n_kernels} streams),{t_parallel:.1f},{speedup:.2f}x")

    print()

    # Small kernels benefit more from overlap
    print("=== Multi-stream overlap: small kernels ===")
    print("scenario,us,speedup_vs_serial")
    small_a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
    small_b = torch.randn(256, 256, device='cuda', dtype=torch.float16)

    for n_kernels in [2, 4, 8, 16]:
        def serial_small(stream, nk=n_kernels):
            for _ in range(nk):
                torch.mm(small_a, small_b)
        t_serial = device_timer(serial_small)

        streams = [torch.cuda.Stream() for _ in range(n_kernels)]
        def parallel_small(stream, ss=streams, nk=n_kernels):
            for i in range(nk):
                with torch.cuda.stream(ss[i]):
                    torch.mm(small_a, small_b)
        t_parallel = device_timer(parallel_small)

        speedup = t_serial / t_parallel if t_parallel > 0 else 0
        print(f"{n_kernels}x small GEMM serial,{t_serial:.1f},1.00x")
        print(f"{n_kernels}x small GEMM parallel,{t_parallel:.1f},{speedup:.2f}x")

    print()

    # ==============================================================
    # 4. Chain of event dependencies
    # ==============================================================
    print("=== Chain of event dependencies (N stages, each depends on previous) ===")
    print("stages,us,overhead_per_stage_us")

    # Baseline: N trivial kernels on one stream (no event overhead)
    for n_stages in [2, 4, 8, 16, 32]:
        torch.cuda.Stream()  # ensure stream exists
        def baseline_chain(stream, ns=n_stages):
            for _ in range(ns):
                small + small
        t_base = device_timer(baseline_chain)

        # With events: each stage on its own stream, chained by events
        chain_streams = [torch.cuda.Stream() for _ in range(n_stages)]
        def event_chain(stream, ss=chain_streams, ns=n_stages):
            for i in range(ns):
                with torch.cuda.stream(ss[i]):
                    if i > 0:
                        e = torch.cuda.Event()
                        e.record(ss[i-1])
                        e.wait(ss[i])
                    small + small
        t_chain = device_timer(event_chain)

        overhead = (t_chain - t_base) / n_stages if n_stages > 0 else 0
        print(f"{n_stages} stages baseline,{t_base:.2f},0.00")
        print(f"{n_stages} stages event-chained,{t_chain:.2f},{overhead:.2f}")

    print()

    # ==============================================================
    # 5. Sync frequency impact on throughput
    # ==============================================================
    print("=== Sync frequency impact on throughput ===")
    print("pattern,total_us,effective_tflops")

    n_gemms = 20
    flops_per_gemm = 2 * SIZE**3

    # No intermediate sync
    def all_async(stream):
        for _ in range(n_gemms):
            torch.mm(A, B)
    t_async = device_timer(all_async)
    tflops_async = n_gemms * flops_per_gemm / (t_async * 1e-6) / 1e12
    print(f"{n_gemms} GEMMs no sync,{t_async:.1f},{tflops_async:.1f}")

    # Sync every K kernels
    for sync_every in [1, 2, 5, 10]:
        def with_sync(stream, se=sync_every):
            for i in range(n_gemms):
                torch.mm(A, B)
                if (i + 1) % se == 0:
                    torch.cuda.synchronize()
        t_sync = device_timer(with_sync)
        tflops_sync = n_gemms * flops_per_gemm / (t_sync * 1e-6) / 1e12
        pct = tflops_sync / tflops_async * 100 if tflops_async > 0 else 0
        print(f"{n_gemms} GEMMs sync every {sync_every},{t_sync:.1f},{tflops_sync:.1f} ({pct:.0f}%)")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
