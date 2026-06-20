#!/bin/bash
# Compare baseline vs opt kernel across different shapes on GPU1
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "Benchmarking: Baseline vs Opt Kernel"
echo "GPU: $(nvidia-smi -i 1 --query-gpu=name --format=csv,noheader)"
echo "=========================================="
echo ""

# Test different shapes: (batch, chunk_len, n_heads, d_state, n_groups)
SHAPES=(
    "1 256 64 64 8"      # Original test shape
    "1 512 64 64 8"      # Longer chunk
    "2 256 64 64 8"      # Larger batch
    "1 256 32 64 4"      # Fewer heads
    "1 256 128 64 16"    # More heads
    "1 256 64 128 8"     # Larger state
)

RESULTS_FILE="bench_results_shapes_$(date +%Y%m%d_%H%M%S).txt"

echo "Results will be saved to: $RESULTS_FILE"
echo ""

for shape in "${SHAPES[@]}"; do
    read -r batch chunk_len n_heads d_state n_groups <<< "$shape"

    echo "=========================================="
    echo "Shape: B=$batch, L=$chunk_len, H=$n_heads, N=$d_state, G=$n_groups"
    echo "=========================================="

    # Benchmark baseline
    echo "Testing BASELINE..."
    baseline_time=$(python -c "
import torch
import sys
sys.path.insert(0, '.')
from tileops.kernels.mamba.ssd_chunk_scan import SSDChunkScanFwdKernel

B, L, H, N, G = $batch, $chunk_len, $n_heads, $d_state, $n_groups
P = 64  # d_head fixed

kernel = SSDChunkScanFwdKernel(B, 1, L, H, P, N, G, 'float32', tune=False)

# Create inputs
x = torch.randn(B, 1, L, H, P, device='cuda:0', dtype=torch.float32)
cb = torch.randn(B, 1, L, G, N, device='cuda:0', dtype=torch.float32)
dA_cumsum = torch.randn(B, H, 1, L, device='cuda:0', dtype=torch.float32)
C_mat = torch.randn(B, L, G, N, device='cuda:0', dtype=torch.float32)
prev_states = torch.randn(B, 1, H, P, N, device='cuda:0', dtype=torch.float32)
dt = torch.randn(B, H, 1, L, device='cuda:0', dtype=torch.float32)

# Warmup
for _ in range(10):
    out = kernel(x, cb, dA_cumsum, C_mat, prev_states, dt)

torch.cuda.synchronize()

# Benchmark
import time
times = []
for _ in range(100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = kernel(x, cb, dA_cumsum, C_mat, prev_states, dt)
    torch.cuda.synchronize()
    end = time.perf_counter()
    times.append((end - start) * 1000)

import statistics
print(f'{statistics.mean(times):.4f}')
" 2>/dev/null)

    echo "Baseline: ${baseline_time}ms"

    # Benchmark opt
    echo "Testing OPT..."
    opt_time=$(python -c "
import torch
import sys
sys.path.insert(0, '.')
from tileops.kernels.mamba.ssd_chunk_scan_opt import SSDChunkScanFwdOptKernel

B, L, H, N, G = $batch, $chunk_len, $n_heads, $d_state, $n_groups
P = 64

kernel = SSDChunkScanFwdOptKernel(B, 1, L, H, P, N, G, 'float32', tune=False)

x = torch.randn(B, 1, L, H, P, device='cuda:0', dtype=torch.float32)
cb = torch.randn(B, 1, L, G, N, device='cuda:0', dtype=torch.float32)
dA_cumsum = torch.randn(B, H, 1, L, device='cuda:0', dtype=torch.float32)
C_mat = torch.randn(B, L, G, N, device='cuda:0', dtype=torch.float32)
prev_states = torch.randn(B, 1, H, P, N, device='cuda:0', dtype=torch.float32)
dt = torch.randn(B, H, 1, L, device='cuda:0', dtype=torch.float32)

for _ in range(10):
    out = kernel(x, cb, dA_cumsum, C_mat, prev_states, dt)

torch.cuda.synchronize()

import time
times = []
for _ in range(100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = kernel(x, cb, dA_cumsum, C_mat, prev_states, dt)
    torch.cuda.synchronize()
    end = time.perf_counter()
    times.append((end - start) * 1000)

import statistics
print(f'{statistics.mean(times):.4f}')
" 2>/dev/null)

    echo "Opt:      ${opt_time}ms"

    # Calculate speedup
    speedup=$(python -c "print(f'{$baseline_time/$opt_time:.2f}x')")

    echo "Speedup:  $speedup"
    echo ""

    # Save to file
    echo "B=$batch L=$chunk_len H=$n_heads N=$d_state G=$n_groups | Baseline: ${baseline_time}ms | Opt: ${opt_time}ms | Speedup: $speedup" >> "$RESULTS_FILE"
done

echo "=========================================="
echo "Summary saved to: $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
