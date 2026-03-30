"""Autotune MoeGroupedGemmNopadKernel for Kimi K2 configuration.

Kimi K2: E=384, K=8, H=7168, F=2048
Focus on T=4096 (large prefill) where vLLM is 36% faster.

Usage:
    conda run -n tileops python benchmarks/ops/autotune_moe_kimi_k2.py
"""

import torch
from tileops.ops.moe import FusedMoe

def main():
    # Kimi K2 config
    T, E, K, H, F = 4096, 384, 8, 7168, 2048
    dtype = torch.bfloat16
    dev = "cuda"

    print(f"Autotuning Kimi K2: T={T}, E={E}, K={K}, H={H}, F={F}")
    print(f"Target: beat vLLM 21.41ms (134.78 TFLOPs)")
    print("=" * 80)

    # Generate inputs
    torch.manual_seed(42)
    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    correction_bias = torch.randn(E, dtype=torch.float32, device=dev) * 0.1
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

    # Create op and enable autotune on experts layer
    print("\nCreating FusedMoe and enabling autotune on experts...")
    op = FusedMoe(
        num_tokens=T,
        num_experts=E,
        top_k=K,
        hidden_size=H,
        ffn_size=F,
        scoring_func="sigmoid",
        renormalize=True,
        with_correction_bias=True,
        routed_scaling_factor=2.827,
        layout="nopad",
        dtype=dtype,
    )

    # Enable autotune on the grouped GEMM kernels
    op._experts._gemm_gate_up.tune = True
    op._experts._gemm_down.tune = True

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = op(hidden, gating, w_gate_up, w_down, correction_bias)
    torch.cuda.synchronize()

    # Benchmark best config
    print("\nBenchmarking best config...")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    n_iters = 100
    start.record()
    for _ in range(n_iters):
        _ = op(hidden, gating, w_gate_up, w_down, correction_bias)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / n_iters
    flops = T * K * 4 * F * H
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    print(f"\nBest configs:")
    print(f"  gate_up GEMM: {op._experts._gemm_gate_up.kernel.config}")
    print(f"  down GEMM: {op._experts._gemm_down.kernel.config}")
    print(f"Time: {elapsed_ms:.2f} ms")
    print(f"TFLOPS: {tflops:.2f}")
    print(f"vs vLLM 21.41ms (134.78 TFLOPs): {21.41/elapsed_ms:.2f}x")

if __name__ == "__main__":
    main()
