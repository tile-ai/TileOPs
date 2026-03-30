"""Debug Kimi K2 timing difference between autotune and benchmark.

Uses exact same measurement as benchmark (triton do_bench with CUPTI).
"""

import torch
from tilelang.profiler import do_bench
from tileops.ops.moe import FusedMoe

def main():
    T, E, K, H, F = 4096, 384, 8, 7168, 2048
    dtype = torch.bfloat16
    dev = "cuda"

    print(f"Kimi K2: T={T}, E={E}, K={K}, H={H}, F={F}")
    print("=" * 80)

    # Generate inputs
    torch.manual_seed(42)
    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    correction_bias = torch.randn(E, dtype=torch.float32, device=dev) * 0.1
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

    # Create op
    op = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
        routed_scaling_factor=2.827,
        layout="nopad", dtype=dtype,
    )

    # Warmup
    op(hidden, gating, w_gate_up, w_down, correction_bias)
    torch.cuda.synchronize()

    # Benchmark with CUPTI (same as benchmark.py)
    def bench_fn():
        return op(hidden, gating, w_gate_up, w_down, correction_bias)

    with torch.no_grad():
        latency = do_bench(bench_fn, warmup=100, rep=200, backend='cupti')
        if latency <= 0:
            print("CUPTI unavailable, falling back to CUDA events...")
            latency = do_bench(bench_fn, warmup=100, rep=200,
                             backend='event', return_mode='median')

    flops = T * K * 4 * F * H
    tflops = flops / (latency * 1e-3) / 1e12

    print(f"\nMeasurement (same as benchmark):")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  TFLOPS: {tflops:.2f}")
    print(f"  vs vLLM 21.41ms (134.78 TFLOPs): {21.41/latency:.2f}x")

if __name__ == "__main__":
    main()
