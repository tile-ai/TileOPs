"""
Quick benchmark for SSDStatePassingFwdKernel across explicit configs.
Prints median latency (us) for each config on the representative workloads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics

import torch

from tileops.kernels.mamba.ssd_state_passing import SSDStatePassingFwdKernel

WORKLOADS = [
    # label,          B,  C,   H,   D
    ("latency-370m-4k",  1, 16,  32, 128),
    ("serving-370m-4k",  8, 16,  32, 128),
    ("longctx-370m-32k", 4, 128, 32, 128),
    ("latency-1p3b-4k",  1, 16,  64, 128),
    ("serving-1p3b-4k",  8, 16,  64, 128),
]

CONFIGS = [
    {"block_d": 256, "threads": 128},
    {"block_d": 128, "threads":  64},
    {"block_d":  64, "threads":  32},
    {"block_d":  32, "threads":  16},
]

WARMUP = 20
REPS   = 100
dtype  = torch.float16

def bench(kernel, states, da_chunk_cumsum, init):
    for _ in range(WARMUP):
        kernel.forward(states, da_chunk_cumsum, init)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        kernel.forward(states, da_chunk_cumsum, init)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)  # us
    return statistics.median(times)

print(f"{'workload':<22} {'block_d':>8} {'threads':>8} {'us':>10}")
print("-" * 52)

for label, B, C, H, D in WORKLOADS:
    states = torch.randn(B, C, H, D, dtype=dtype, device="cuda")
    da_chunk_cumsum     = torch.randn(B, H, C, dtype=torch.float32, device="cuda")
    init   = torch.randn(B, H, D, dtype=torch.float32, device="cuda")

    for cfg in CONFIGS:
        try:
            k = SSDStatePassingFwdKernel(B, C, H, D, has_initial_states=True,
                                         dtype=dtype, config=cfg)
            t = bench(k, states, da_chunk_cumsum, init)
            print(f"{label:<22} {cfg['block_d']:>8} {cfg['threads']:>8} {t:>10.2f}")
        except Exception as ex:
            print(f"{label:<22} {cfg['block_d']:>8} {cfg['threads']:>8} {'ERR':>10}  # {ex}")
    print()
